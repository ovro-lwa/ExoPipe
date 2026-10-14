"""Serve-time exclusion gate for EVERY tier (G17; FOLD_IN row f1996ee; spec serve.md 5 and 12.8).

Gregg 2026-10-13, ruling 2 'Fine.' (RULINGS-GREGG-20261013-DECISIONS-1-7-AND-TENMIN-TABLES): the serve re-reads
data_decisions.csv and refuses a run holding any row on an excluded cell-band. Production implements it for the snapshot
tier only (snapshot_serve_exclusion_gate.py, f1996ee). Its row keying (cand_id '<lst>_<date>_<band>_...') keys 0 deep and
0 sub-LST rows (their cand_ids start 'VD_' / 'ID_' / 'V1_' ...), so this module adds the tier keyers:

  snapshot   keys_snapshot(row): the in-place gate's own order (snapshot_serve_exclusion_gate.py:27-47): cand_id
             <lst>_<date>_<band>_<S>_<tier>_<ts>; lst + date + band; cell + subband; cell + member_sb; cell + bands (';'-list)
  deep, sublst  keys_tier(row): cand_id, else best_cand_id '<S><x>_<LST>_<date>[Tnn]_<band>_<n>' -> (LST, date[:10], band);
             else source_id as a page id 'Jname_LST_date[Tnn]_band' (the pages_index rows); queue ids are page ids too

Verdicts come from core.exclusions (a FRESH Exclusions(), never the process cache: the list is read now). For the snapshot
tier the in-place gate (production's reader, coadd_full_pipeline.is_excluded) also runs, and the two readers must agree
(a disagreement refuses with both lists). Whether a hit REFUSES is config serve.EXCLUSION_GATE_TIERS; for the keys not in
it the hits are reported (unit notes, dry-run plan). Gregg 2026-10-14 approved enforcing it for the deep and sub-LST pages
and the snapshot forced pages too (R-GREGG-20261014-REFACTOR-DECISIONS-LIST-RECOMMENDATIONS); production ('snapshot' only)
is the parity file's value.

What is scanned. Snapshot (and the forced pages): every CSV of the run, as the in-place gate scans it (snapshot_run_files).
Deep / sub-LST, PER PAGE: exactly what the page's review server reads (scan_page) - its queue's page ids, and for its
--catalog-dir each band's <band>_sources_survivors.csv and <band>_source_members.csv, each survivor's page folder
<run>/<review_dir>: detections.csv and its panel images (classify_transients.load_sources, non-blind). A file the server
does not read is not scanned: on 2026-10-14 the deep multiples view folders held 34 rows on excluded nights, all in
*_pages_index_*.csv files of 2026-10-07 that production leaves in place and never reads (deep chat, confirmed read only);
the served pages held none.
"""
import csv
import glob
import os
import re
import collections
from collections import Counter

from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.serve import scripts

_LST = re.compile(r'^\d\dh$')
_DATE = re.compile(r'^\d{4}-\d{2}-\d{2}')


def keys_snapshot(row):
    """(lst, date, band) keys of a snapshot row, by the first identifying column set present (the in-place gate's
    keys_of, snapshot_serve_exclusion_gate.py:27-47, same order and tests)."""
    cid = (row.get('cand_id') or '').strip()
    if cid:
        p = cid.split('_')
        if len(p) >= 3 and p[0].endswith('h') and len(p[1]) == 10:
            return [(p[0], p[1], p[2])]
    lst, date, band = (row.get('lst') or '').strip(), (row.get('date') or '').strip(), (row.get('band') or '').strip()
    if lst and date and band:
        return [(lst, date, band)]
    cell = (row.get('cell') or '').strip()
    if cell and '_' in cell:
        l_, d_ = cell.split('_', 1)
        for col in ('subband', 'member_sb'):
            v = (row.get(col) or '').strip()
            if v:
                return [(l_, d_, v)]
        v = (row.get('bands') or '').strip()
        if v:
            return [(l_, d_, b.strip()) for b in v.replace(',', ';').split(';') if b.strip()]
    return []


def _tier_key(token_string, first):
    p = token_string.split('_')
    i = first
    if len(p) >= i + 3 and _LST.match(p[i]) and _DATE.match(p[i + 1]):
        return (p[i], p[i + 1][:10], p[i + 2])
    return None


def keys_tier(row):
    """(lst, date, band) keys of a deep / sub-LST view row: cand_id, else best_cand_id, else the page id in source_id."""
    for col in ('cand_id', 'best_cand_id'):
        v = (row.get(col) or '').strip()
        if v:
            k = _tier_key(v, 1)
            if k:
                return [k]
    sid = (row.get('source_id') or '').strip()
    if sid.startswith('J') and sid.count('_') >= 3:
        k = _tier_key(sid, 1)
        if k:
            return [k]
    return []


def page_id_key(pid):
    return _tier_key(pid, 1) if pid.startswith('J') and pid.count('_') >= 3 else None


def fresh_exclusions():
    from exopipe.phase3.core.exclusions import Exclusions
    return Exclusions()


class Scan:
    def __init__(self):
        self.hits = []            # (file, row id, lst, date, band)
        self.n_rows = self.n_keyed = self.n_files = 0
        self.cache = {}
        self.unreadable = []      # files / folders a report-only scan could not read (scan_page strict=False)

    def per_key(self):
        return Counter((h[2], h[3], h[4]) for h in self.hits)

    def summary(self):
        per = self.per_key()
        return (f'{self.n_rows} rows in {self.n_files} files, {self.n_keyed} keyed ({len(self.cache)} distinct cell-bands), '
                f'{len(self.hits)} on excluded cell-bands' + (': ' + '; '.join(f'{k[0]} {k[1]} {k[2]}: {n}' for k, n in
                                                                              sorted(per.items())) if per else '')
                + (f'; {len(self.unreadable)} not readable (report only): {self.unreadable[:3]}' if self.unreadable else ''))


def _check(scan, ex, rel, rid, keys):
    if keys:
        scan.n_keyed += 1
    for k in keys:
        if k not in scan.cache:
            scan.cache[k] = ex.is_excluded(k[0], k[2], k[1])
        if scan.cache[k]:
            scan.hits.append((rel, rid, k[0], k[1], k[2]))


def scan_files(paths, keyer, base, ex=None, scan=None):
    """Key every row of the CSV files ``paths`` with ``keyer``; verdicts from core.exclusions (fresh)."""
    ex = ex or fresh_exclusions()
    scan = scan or Scan()
    for p in paths:
        rel = os.path.relpath(p, base)
        try:
            with open(p, newline='') as fh:
                rd = csv.DictReader(fh)
                if not rd.fieldnames:
                    continue
                scan.n_files += 1
                for row in rd:
                    scan.n_rows += 1
                    _check(scan, ex, rel, row.get('cand_id') or row.get('best_cand_id') or row.get('source_id') or
                           row.get('cell') or '', keyer(row))
        except (OSError, csv.Error, UnicodeDecodeError) as e:
            raise Refusal(f'G17 serve-time exclusion gate: {rel} could not be read ({e}) - refused')
    return scan


def scan_queue(path, base, ex=None, scan=None):
    """Key the page ids of a queue file."""
    ex = ex or fresh_exclusions()
    scan = scan or Scan()
    rel = os.path.relpath(path, base)
    scan.n_files += 1
    for l in open(path):
        if l.startswith('J'):
            scan.n_rows += 1
            pid = l.split()[0]
            k = page_id_key(pid)
            _check(scan, ex, rel, pid, [k] if k else [])
    return scan


def snapshot_run_files(run):
    """Every CSV of a snapshot run that the in-place gate reads (judgements and its own refusal list excepted)."""
    out = []
    for p in sorted(glob.glob(os.path.join(run, '**', '*.csv'), recursive=True)):
        rel = os.path.relpath(p, run)
        if rel.startswith('judgements' + os.sep) or os.path.basename(p) == 'serve_exclusion_refusals.csv':
            continue
        out.append(p)
    return out


def scan_snapshot_run(run, ex=None):
    return scan_files(snapshot_run_files(run), keys_snapshot, run, ex=ex)


# The review server of every deep and sub-LST page (program 'classify': exopipe_ops/classify_transients.py), mirrored here
# and drift-tested against its source (tests/test_serve.py test_exclusion_gate_server_mirrors):
SERVER_BANDS = ('Blue', 'Full', 'Green', 'Red', '82MHz', '78MHz', '73MHz', '69MHz', '64MHz', '59MHz', '55MHz', '50MHz',
                '46MHz', '41MHz', '36MHz', '32MHz', '27MHz', '23MHz', '18MHz')                  # BANDS (:49-52)
SERVER_PANEL_RE = re.compile(r'^rank\d+_(\d{2}h)_(\d{4}-\d{2}-\d{2}(?:_bin\d+)?)_([A-Za-z0-9]+?)'
                             r'(_blind)?\.png$')                                                # PNG_RE (:54-55)


def _server_band(token):
    """The band of a panel name's band token: a deep / sub-LST token is the band; a snapshot token carries its instant after
    the band (classify_transients.py:1532-1536)."""
    if token in SERVER_BANDS:
        return token
    pre = [b for b in SERVER_BANDS if token.startswith(b)]
    return max(pre, key=len) if pre else token


def scan_page(U, cat_dir, queue, ex=None, loc_offsets=None, strict=True):
    """Deep / sub-LST: the rows of ONE page as its review server (classify_transients.load_sources, non-blind) shows them:
      - its queue's page ids;
      - each band's <band>_sources_survivors.csv of --catalog-dir (SERVER_BANDS);
      - each survivor's page folder <U>/<review_dir>: its detections.csv rows and every non-blind panel image (the server
        shows every panel in the folder, keyed here by the (LST, night, band) of its name);
      - the <band>_source_members.csv rows the server turns into shown detections: members of a survivor's identity, for an
        identity whose folders hold no detections.csv (the server skips a member of no survivor, and lists members as
        detections only when no detections.csv was read, :641-660);
      - with ``loc_offsets`` (a page run with --loc-offsets): the rows of that localization_offsets.csv for the cand_ids the
        page shows (they are page text).
    ``strict`` False (report only): a file or folder that vanished or cannot be read (a pass running) is listed in
    scan.unreadable, never raised - a report never stops a serve; True: it refuses (the page cannot be verified)."""
    ex = ex or fresh_exclusions()
    scan = Scan()

    def unreadable(path, e):
        rel = os.path.relpath(path, U)
        if strict:
            raise Refusal(f'G17 serve-time exclusion gate: {rel} could not be read ({e}) - refused')
        scan.unreadable.append(rel)

    def rows_of(path):
        try:
            with open(path, newline='') as fh:
                return list(csv.DictReader(fh))
        except (OSError, csv.Error, UnicodeDecodeError) as e:
            unreadable(path, e)
            return None

    def check_rows(path, rows):
        scan.n_files += 1
        rel = os.path.relpath(path, U)
        for row in rows:
            scan.n_rows += 1
            _check(scan, ex, rel, row.get('cand_id') or row.get('best_cand_id') or row.get('source_id') or '',
                   keys_tier(row))

    ident, folders = {}, collections.defaultdict(list)          # survivor source_id -> identity; identity -> folders
    for band in SERVER_BANDS:
        sp = os.path.join(cat_dir, f'{band}_sources_survivors.csv')
        if not os.path.exists(sp):
            continue
        rows = rows_of(sp)
        if rows is None:
            continue
        check_rows(sp, rows)
        for row in rows:
            sid = row.get('source_id', '')
            ident[sid] = (row.get('identity_id') or '').strip() or sid
            rd = (row.get('review_dir') or '').strip()
            if rd and os.path.isdir(os.path.join(U, rd)):
                folders[ident[sid]].append(os.path.join(U, rd))
    shown, with_csv = set(), set()
    for idt, fl in sorted(folders.items()):
        for f in sorted(set(fl)):
            dp = os.path.join(f, 'detections.csv')
            if os.path.exists(dp):
                rows = rows_of(dp)
                if rows is not None:
                    with_csv.add(idt)
                    shown.update(r.get('cand_id', '') for r in rows)
                    check_rows(dp, rows)
            try:
                names = sorted(os.listdir(f))
            except OSError as e:
                unreadable(f, e)
                continue
            rel = os.path.relpath(f, U)
            for fn in names:
                m = SERVER_PANEL_RE.match(fn)
                if m and not m.group(4):                 # the server shows the non-blind panels only
                    scan.n_rows += 1
                    _check(scan, ex, os.path.join(rel, fn), fn, [(m.group(1), m.group(2)[:10], _server_band(m.group(3)))])
    for band in SERVER_BANDS:
        mp = os.path.join(cat_dir, f'{band}_source_members.csv')
        if not os.path.exists(mp):
            continue
        rows = rows_of(mp)
        if rows is None:
            continue
        mine = [r for r in rows if r.get('source_id', '') in ident and ident[r['source_id']] not in with_csv]
        shown.update(r.get('cand_id', '') for r in mine)
        check_rows(mp, mine)
    if loc_offsets and os.path.exists(loc_offsets):
        rows = rows_of(loc_offsets)
        if rows is not None:
            check_rows(loc_offsets, [r for r in rows if r.get('cand_id', '') in shown])
    if queue and os.path.exists(queue):
        try:
            scan_queue(queue, U, ex=ex, scan=scan)
        except (OSError, UnicodeDecodeError) as e:
            unreadable(queue, e)
    return scan


def page_loc_offsets(pg, U):
    """The localization_offsets.csv a page's server reads with --loc-offsets (classify_transients.py:2469-2470), else None."""
    if not getattr(pg, 'loc_offsets', False):
        return None
    d = pg.at('loc_dir', U) if pg.c.get('loc_dir') else os.path.join(U, 'source_catalogs')
    return os.path.join(d, 'localization_offsets.csv')


def scan_server_page(pg, U, strict=True, ex=None):
    """scan_page for a planned review page (its queue, catalogue view and offsets table under run ``U``)."""
    return scan_page(U, pg.at('catalog', U), pg.at('queue', U), ex=ex, loc_offsets=page_loc_offsets(pg, U), strict=strict)


def scan_server_argv(argv, strict=True, ex=None):
    """scan_page for a RUNNING classify server, from its argv (--run, --queue, --catalog-dir, --loc-dir, --loc-offsets)."""
    from exopipe.phase3.lib.serve import procs
    U = procs.argv_value(argv, '--run')
    ld = procs.argv_value(argv, '--loc-dir')
    lo = os.path.join(ld or os.path.join(U, 'source_catalogs'), 'localization_offsets.csv') if '--loc-offsets' in argv else None
    return scan_page(U, procs.argv_value(argv, '--catalog-dir') or os.path.join(U, 'source_catalogs'),
                     procs.argv_value(argv, '--queue'), ex=ex, loc_offsets=lo, strict=strict)


def run_inplace_gate(run_dir, host=None, unit=None):
    """The production gate on ``run_dir``. Local: core.legacy.run with site PYTHON 'default' (snapshot_serve.sh:47, the
    review host's py38). On ``host`` (the chain host, snapshot_serve.sh:26, its development Python = site 'py36'): over
    ssh - core.legacy.run runs local children only, so this one remote call is a plain ssh (SSH_PREFIX_ENV, BatchMode),
    production only (a parity run gates its own copy). Returns (rc, output, refusal rows)."""
    if host:
        import subprocess
        from exopipe.phase3.config import CFG
        cmd = f'{scripts.interpreter("py36")} {scripts.path("snapshot_gate")} {run_dir}'
        env = dict(os.environ, **CFG.runner.SSH_PREFIX_ENV)
        p = subprocess.run(['ssh', '-n', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', host, cmd], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=CFG.serve.LEGACY_TIMEOUT_S)
        return p.returncode, p.stdout.decode('utf-8', 'replace'), None
    res = scripts.run('snapshot_gate', [run_dir], unit=unit, refusal_codes=(1, 2))
    refusals = []
    rp = os.path.join(run_dir, 'serve_exclusion_refusals.csv')
    if res.rc == 1 and os.path.exists(rp):
        with open(rp, newline='') as f:
            refusals = list(csv.DictReader(f))
    return res.rc, res.tail, refusals


def agree(scan, refusals, rc):
    """(True, detail) when the in-place gate (production reader) and core.exclusions agree on the excluded keys."""
    mine = scan.per_key()
    theirs = Counter((r['lst'], r['date'], r['band']) for r in (refusals or []))
    if rc == 0 and not mine:
        return True, 'both readers: 0 rows on excluded cell-bands'
    if rc == 1 and mine == theirs:
        return True, f'both readers: the same {sum(mine.values())} rows on {len(mine)} excluded cell-bands'
    return False, (f'the two exclusion readers DISAGREE: in-place gate rc {rc}, {dict(theirs)}; core.exclusions '
                   f'{dict(mine)}')


def gate_key(tier, family='review'):
    """The key of serve.EXCLUSION_GATE_TIERS for a page family: the tier for its review pages, '<tier>-<family>' else."""
    return tier if family == 'review' else f'{tier}-{family}'


def refuses(tier, family='review'):
    from exopipe.phase3.config import CFG
    return gate_key(tier, family) in CFG.serve.EXCLUSION_GATE_TIERS
