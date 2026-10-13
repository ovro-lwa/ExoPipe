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
(a disagreement refuses with both lists). Whether a hit REFUSES is config serve.EXCLUSION_GATE_TIERS (snapshot today);
for the other tiers the hits are reported (unit notes, dry-run plan) - enabling deep / sub-LST is Gregg's (spec 12.8).
"""
import csv
import glob
import os
import re
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

    def per_key(self):
        return Counter((h[2], h[3], h[4]) for h in self.hits)

    def summary(self):
        per = self.per_key()
        return (f'{self.n_rows} rows in {self.n_files} files, {self.n_keyed} keyed ({len(self.cache)} distinct cell-bands), '
                f'{len(self.hits)} on excluded cell-bands' + (': ' + '; '.join(f'{k[0]} {k[1]} {k[2]}: {n}' for k, n in
                                                                              sorted(per.items())) if per else ''))


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


def scan_tier_views(U, dirs, queues=(), ex=None):
    """Deep / sub-LST: every CSV of the served view directories plus the served queues' page ids."""
    ex = ex or fresh_exclusions()
    files = sorted(sum((glob.glob(os.path.join(d, '*.csv')) for d in dirs if os.path.isdir(d)), []))
    scan = scan_files(files, keys_tier, U, ex=ex)
    for q in queues:
        if os.path.exists(q):
            scan_queue(q, U, ex=ex, scan=scan)
    return scan


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
