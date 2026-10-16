"""B-02 steps 2 and 4 (plan/wave2/B02_DESIGN.md "Steps 2 and 4"; R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32
B-02 (a), R-GREGG-20261014-REFACTOR-DECISIONS-LIST-RECOMMENDATIONS item 5): with every removal rule run once, in the search
(search_detect.REMOVAL_RULES 'once-in-search'), the deep main chain's publish conformance only checks (catalog_conform.MODE
'check'), and the cross-cell pulsar groups are a stage of their own between the merge and the check. Under 'remove'
(production; the parity file) deep_chain.st_conform runs p3s2_conform.py as today and neither function here runs.

Both read the attempts the merge MERGED - the harvest manifest less the attempts the merge refused as stale (deep_chain
writes that list, MERGED_ATTEMPTS, and refuses 'check' when the merge skipped an unharvested attempt).

pulsars(U, manifest, label): the pulsar stage. The merged catalog and, per merged attempt, the search's table decisions that
name a pulsar (table_pulsar_decisions.csv: kept as the pulsar's parent, or cut although it lies within a beam) and the
gate-ledger rows of the cut ones; the groups are production's (removal_rules.pulsar_consistency): a group consistent with
its pulsar gets its cut detections back - rows built from their ledger rows, as the v110 re-admission builds one, the ID
<S>R_<lst>_<date>_<band>_<pixel> in the catalog's shape (the VM_ rows' fields: downstream reads the LST from field 2) - and an
inconsistent one loses its pulsar-exempted detections. Writes <U>/master_catalog.csv (the merged rows, changed only so, in the
merged columns - deep_chain.ensure_master gives every stage its predecessor's recorded catalog), PULSAR_RECORD (every row it
changed, with its words - the check puts them in the restored rows' conform_note), PULSAR_REMOVED (the rows it removed, with
their reason) and PULSAR_GROUPS (every group: pulsar, members, separation, 99 % radius, consistent).

check(U, manifest, label, merged): the check. It refuses, before reading a row, a run with an attempt not searched under
'once-in-search' at the current RULES_VERSION. Then every removal rule is re-evaluated, from what the run recorded and
today's inputs, in both directions: (a) every catalog row, final detection and searched frame passes every rule; (b) every
detection or frame the search removed by one of them is still removed, and every night it left out as excluded by
data_decisions (its record, refusals/<cell>_datadecisions_excluded.csv) is excluded still or was searched since (an
append) - else a lifted exclusion is a night never searched. A ledger row is evaluated where the search evaluated it: its
pixel's sky position on its frame's grid (5 decimals - the catalog's position), the frame's own time block and BMAJ, the
noise the search recorded for the frame (frame_noise.csv), the rules in the search's order (elevation floor, Jupiter track,
bright-source table, row line). The search's own gates are not re-run (size above 8 sigma among them: the ledger holds its
inputs rounded, and production's deep conformance runs no size step); the axis ratio is checked at the search's limit on
the finals and production's own computation on the catalog rows. Any finding: CHECK_REPORT written, REFUSED
(nothing published). Else production's outputs with nothing removed - master_catalog_asrun.csv (a copy of ``merged``, the
merge's output), master_catalog.csv with the conform columns, master_catalog_removed.csv (the pulsar stage's removals) -
and the frame caches, as p3s2_conform.py writes them; so conformant + removed = as-run + the restored rows, as production's
conformant + removed = as-run + the re-admitted VM_ rows.
Not seen: a lifted exclusion of a night the search never had - one the masters build left out as excluded, so the cell
has no residual of it (2,355 of today's excluded cell-nights; the record covers the 145 with a residual). The master
then lacks a night admissible now, as it lacks a night of new data: a state of the masters' coverage, not of the
search's removals (B02_DESIGN.md "Step 3c, first review").

python -m exopipe.phase3.lib.catalog.conform_check pulsars <U> <merged attempts> <label>
python -m exopipe.phase3.lib.catalog.conform_check check <U> <merged attempts> <label> <merged catalogue> [<pulsar record>
    <pulsar removed>]   (deep_chain passes the pulsar stage's recorded copies)
The chain hands its config overrides to the child in OVERRIDES_ENV (applied as forwarded overrides the step authorized).
"""
import collections
import csv
import json
import math
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from exopipe.phase3.config import CFG
from exopipe.phase3.lib.search import removal_rules as RR

MODES = ('check', 'remove')                       # catalog_conform.MODE
CHECK_REPORT = 'conform_check.csv'                # every finding of a refused check
CHECK_COLS = ['direction', 'rule', 'lst', 'scope', 'stokes', 'date', 'pix_x', 'pix_y', 'ra', 'dec', 'cand_id', 'words']
PULSAR_RECORD = 'pulsar_stage.csv'                # every catalog row the pulsar stage restored or removed
PULSAR_RECORD_COLS = ['action', 'lst', 'scope', 'stokes', 'date', 'pix_x', 'pix_y', 'ra', 'dec', 'cand_id', 'words']
PULSAR_GROUPS = 'pulsar_groups.csv'               # every group the stage judged
PULSAR_GROUP_COLS = ['pulsar', 'n', 'members', 'sep_arcsec', 'r99_arcsec', 'consistent']
PULSAR_REMOVED = 'master_catalog_pulsar_removed.csv'   # the rows the pulsar stage removed, each with its removal_reason
MERGED_ATTEMPTS = 'merged_attempts.csv'           # the attempts the merge merged (deep_chain writes it in the pulsars stage)
RESTORED_TAG = 'R'                                # <S>R_<lst>_<date>_<band>_<n>: a restored row (VM_: a v110 re-admission)
OVERRIDES_ENV = 'EXOPIPE3_CONFORM_CHECK_OVERRIDES'   # the chain's config overrides, for the module child
B02_VETOES = ('elevation', 'jupiter-track', 'table', 'row-line')   # the search's removal rules a ledger row names
FRAME_THREADS = 8                                 # parallel header reads (Lustre-bound)


def conform_mode(value=None):
    """catalog_conform.MODE (``value`` checked); an unknown value REFUSES - a typo never selects either."""
    value = CFG.catalog_conform.MODE if value is None else value
    if value not in MODES:
        raise RuntimeError(f'catalog_conform.MODE = {value!r} is not one of {MODES} - REFUSING')
    return value


def is_restored(cand_id):
    """A row the pulsar stage restored (its ID: <S>R_...)."""
    return len(cand_id) > 3 and cand_id[0] in 'IV' and cand_id[1:3] == f'{RESTORED_TAG}_'


class Refused(SystemExit):
    """A refusal of the stage: exit status 2 with the reason (core.legacy.run maps it to a refused unit)."""
    def __init__(self, msg):
        print(f'REFUSED: {msg}', flush=True)
        super().__init__(2)


def _deps():
    """The modules and the exclusion test the rules read (production's: gate_inputs, source_localization, the bright-source
    table, data_decisions) - one function, so the tests can give synthetic ones."""
    import gate_inputs as G
    import source_localization as LOC
    from exopipe.phase3.lib.masters.coadd import load_exclusions, is_excluded
    from exopipe.phase3.lib.search import parent_curve_table as PCT
    EXCL = load_exclusions()
    return dict(G=G, LOC=LOC, PCT=PCT, excluded=lambda l, b, d: is_excluded(l, b, d, EXCL))


def literal(v):
    """``v`` as Python literal text that ast.literal_eval gives back as the same value and type: repr, recursively, with an
    infinity written 1e999 / -1e999 (repr's 'inf' is no literal). A NaN has no literal: refused."""
    if isinstance(v, float) and not isinstance(v, bool):
        if math.isnan(v):
            raise RuntimeError('a NaN config value cannot be forwarded to the module child - REFUSING')
        return ('1e999' if v > 0 else '-1e999') if math.isinf(v) else repr(v)
    if isinstance(v, tuple):
        return '(' + ', '.join(literal(x) for x in v) + (',)' if len(v) == 1 else ')')
    if isinstance(v, list):
        return '[' + ', '.join(literal(x) for x in v) + ']'
    if isinstance(v, dict):
        return '{' + ', '.join(f'{literal(k)}: {literal(x)}' for k, x in v.items()) + '}'
    return repr(v)


def forward_overrides(cfg):
    """The run's config overrides - each key with its final value as literal text (literal()), which the child parses back as
    a --set value is parsed (tuples, tuple keys, str types and infinities survive) - and its parity-replay flag, as the JSON
    the chain hands the module child in OVERRIDES_ENV, so the child judges with the config the unit records."""
    return json.dumps({'parity': bool(getattr(cfg, 'parity_replay', False)),
                       'overrides': [[o[0], literal(o[2])] for o in cfg.overrides]})


def apply_forwarded(text):
    """In the child: the chain's overrides (forward_overrides), each set as an override the step already authorized."""
    if not text:
        return 0
    d = json.loads(text)
    CFG.authorize(mode='forward')
    if d.get('parity'):
        CFG.parity_replay = True
    for k, v in d['overrides']:
        CFG.set(k, v, source='forwarded by the catalog chain unit')
    return len(d['overrides'])


def _rows(p):
    if not os.path.exists(p):
        return []
    with open(p, newline='') as fh:
        return list(csv.DictReader(fh))


def _headerless(p):
    if not os.path.exists(p):
        return []
    with open(p, newline='') as fh:
        return [r for r in csv.reader(fh) if r]


def _write(path, cols, rows):
    """A CSV written whole and moved into place (a reader never sees half of it)."""
    with open(path + '.part', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    os.replace(path + '.part', path)


def _veto(r):
    """A ledger row's B-02 rule ('elevation', 'jupiter-track', 'table', 'row-line') or ''."""
    v = r.get('veto_reason') or ''
    v = 'jupiter-track' if v.startswith('jupiter-track:') else v
    return v if v in B02_VETOES else ''


# ---------------------------------------------------------------------------------------------------------------------
# the merged attempts and what each recorded
# ---------------------------------------------------------------------------------------------------------------------

def attempts(manifest, label):
    """[{cell, lst, scope, stokes, dir, rd}] of a manifest in the harvest manifest's columns (cell, stokes, node, att, dir, ...;
    the chain passes MERGED_ATTEMPTS); rd = the attempt's run directory <dir>/transients/Run_<label>."""
    out = []
    for r in _rows(manifest):
        lst, scope = r['cell'].split('_', 1)
        out.append(dict(cell=r['cell'], lst=lst, scope=scope, stokes=r['stokes'], dir=r['dir'],
                        rd=os.path.join(r['dir'], 'transients', f'Run_{label}')))
    return out


def rules_findings(atts):
    """The attempts not searched under 'once-in-search' at the current RULES_VERSION (an unreadable record included)."""
    want = RR.rules_record('once-in-search')
    bad = []
    for a in atts:
        try:
            rec = RR.attempt_rules(a['dir'])
        except RuntimeError as e:
            bad.append(f"{a['cell']} {a['stokes']}: {e}")
            continue
        if rec != want:
            bad.append(f"{a['cell']} {a['stokes']}: searched under {rec['rules']!r} (rules version {rec['version']}), not "
                       f"'once-in-search' at {want['version']}")
    return bad


_WCS_CARDS = ('CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2', 'CUNIT1', 'CUNIT2', 'PC1_1', 'PC1_2',
              'PC2_1', 'PC2_2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'LONPOLE', 'LATPOLE', 'RADESYS', 'EQUINOX')


class Frames:
    """The archived search frames of the attempts (<dir>/Run_<label>/{iresid,vresid}/<cell>/<date>_<S>_redo_resid_<band>.fits):
    per frame its time block and BMAJ, and its celestial WCS - shared by every frame whose WCS cards are the same (a cell's
    frames are on its master's grid) - read once each, in parallel (prefetch); or the error that stopped the read."""
    def __init__(self, atts, label):
        self.dir = {(a['cell'], a['stokes']): a['dir'] for a in atts}
        self.label = label
        self._f, self._w = {}, {}

    def path(self, stokes, lst, scope, date):
        d = self.dir.get((f'{lst}_{scope}', stokes))
        sub = 'iresid' if stokes == 'I' else 'vresid'
        return os.path.join(d, f'Run_{self.label}', sub, f'{lst}_{scope}', f'{date}_{stokes}_redo_resid_{scope}.fits') if d else None

    def _read(self, k):
        from astropy.io import fits
        p = self.path(*k)
        try:
            h = fits.getheader(p)
            cards = tuple((c, h.get(c)) for c in _WCS_CARDS)
            return k, dict(path=p, times=(str(h['DATE-OBS']), str(h['DATE-END'])), bmaj=float(h['BMAJ']), cards=cards,
                           _h=h)
        except Exception as e:
            return k, dict(path=p, error=f'{type(e).__name__}: {e}')

    def prefetch(self, keys):
        todo = [k for k in set(keys) if k not in self._f]
        if todo:
            with ThreadPoolExecutor(FRAME_THREADS) as ex:
                for k, rec in ex.map(self._read, todo):
                    self._store(k, rec)

    def _store(self, k, rec):
        h = rec.pop('_h', None)
        if 'error' not in rec:
            try:
                if rec['cards'] not in self._w:
                    from astropy.wcs import WCS
                    w = WCS(h).celestial
                    if w.naxis != 2:        # parses, but no sky axes (no CTYPE cards): no position can be read
                        raise ValueError(f'the celestial WCS has {w.naxis} axes')
                    self._w[rec['cards']] = w
                rec['wcs'] = self._w[rec['cards']]
            except Exception as e:          # a header that reads but whose WCS does not give a sky position: the frame's
                rec = dict(path=rec['path'], error=f'WCS: {type(e).__name__}: {e}')   # error
        self._f[k] = rec

    def get(self, stokes, lst, scope, date):
        k = (stokes, lst, scope, date)
        if k not in self._f:
            self._store(*self._read(k))
        return self._f[k]


class LazyDay:
    """The daytime verdict of a frame (removal_rules.daytime_verdicts), computed when a pulsar group asks for it (production's
    conform computes it for every merged frame; the group rule reads only the frames of the rows near a named pulsar). An
    unreadable frame has no verdict: get() gives the default, as the conform's dict does."""
    def __init__(self, frames):
        self.frames, self.quar, self._v = frames, RR.solar_quarantine(CFG.catalog.SOLAR_AUDIT), {}

    def get(self, key, default=None):
        if key not in self._v:
            fr = self.frames.get(*key)
            self._v[key] = None if 'error' in fr else RR.daytime_verdicts({key: fr['times']}, self.quar).get(key)
        v = self._v[key]
        return default if v is None else v


def match(crows, frows, cap=None):
    """(matched catalogue indices, matched final indices) of one key: catalogue rows [(i, RA, Dec)] and finals [(j, RA, Dec)]
    paired one-to-one by nearest position within ``cap`` deg (RA-wrapped) - the merge gate's rule (removal_rules.backed_rows)."""
    cap = CFG.catalog_merge.BACKING_CAP_DEG if cap is None else cap
    pairs = []
    for i, ra, dec in crows:
        for j, fra, fdec in frows:
            dra = (fra - ra + 180.0) % 360.0 - 180.0
            d = float(np.sqrt(dra ** 2 * np.cos(np.radians(dec)) ** 2 + (fdec - dec) ** 2))
            if d < cap:
                pairs.append((d, i, j))
    pairs.sort()
    uc, uf = set(), set()
    for d, i, j in pairs:
        if i in uc or j in uf:
            continue
        uc.add(i)
        uf.add(j)
    return uc, uf


def rowline(ys, min_n):
    """The indices the search's row-line veto removes from one epoch's detections (pixel rows ``ys``): every detection within
    +/-1 row of a row holding ``min_n`` or more (deep_search.py's loop, both Stokes)."""
    ys = np.asarray(ys, int)
    drop = set()
    if len(ys) >= min_n:
        for y in set(ys.tolist()):
            m = np.where(np.abs(ys - y) <= 1)[0]
            if len(m) >= min_n:
                drop.update(m.tolist())
    return drop


def sky(fr, r):
    """A ledger row's sky position as the search evaluated it: its pixel's, on its frame's grid, to 5 decimals."""
    s = fr['wcs'].pixel_to_world(int(float(r['pix_x'])), int(float(r['pix_y'])))
    return round(float(s.ra.deg), 5), round(float(s.dec.deg), 5)


def elevation(lst, ra, dec, fr, epoch):
    """removal_rules.frame_elevation of a row on its frame (the same function, from the frame's own DATE-OBS / DATE-END)."""
    return RR.deep_elevation(lst, ra, dec, epoch=epoch, frame_time=fr['times'])


# ---------------------------------------------------------------------------------------------------------------------
# the search's rules, as the search applies them (deep_search.py under 'once-in-search')
# ---------------------------------------------------------------------------------------------------------------------

def cell_base(cell, roots_csv=None):
    """The cell's coadd root directory, as the search reads it (roots.csv: cell,,root -> <root>/<cell>)."""
    for l in open(roots_csv or CFG.catalog.DEEP_ROOTS_CSV):
        q = l.strip().split(',')
        if q and q[0] == cell and len(q) > 2:
            return os.path.join(q[2], cell)
    return None


def i_noise_refused(base, scope, x=None, min_n=None):
    """The nights the search's Stokes I noise rule refuses (deep_search's sanity gate): qa3_rms_mjy above x times the upper
    middle of the cell's finite positive weights.csv values (none with fewer than min_n), an admitted night judged by its own
    admission value (lib/search/admitted.values); {night: (value, median)}, or None when the gate does not run."""
    from exopipe.phase3.lib.search import admitted as ADS
    x = CFG.search_detect.SANITY_X if x is None else x
    min_n = CFG.search_detect.SANITY_MIN_NIGHTS if min_n is None else min_n
    wr = {}
    for r in _rows(os.path.join(base, 'coadd_final', 'weights.csv')):
        try:
            wr[r['date']] = float(r['qa3_rms_mjy'])
        except (TypeError, ValueError, KeyError):
            pass
    fin = sorted(v for v in wr.values() if np.isfinite(v) and v > 0)
    if len(fin) < min_n:
        return None
    med = fin[len(fin) // 2]
    adm, _bad = ADS.nights(base, scope, 'deep')
    wr = {**wr, **ADS.values(adm, 'qa3_rms_mjy')}
    return {d: (v, med) for d, v in wr.items() if v > x * med}


def table_decisions(rows, frames, noise, pos, el, G, PCT):
    """{id(row): (cut, note)} of the bright-source table for ledger rows, one call per frame: the search's inputs - TBLNOISE
    from the attempt's frame_noise.csv (``noise`` {(Stokes, lst, scope, date): Jy}), the frame's BMAJ, the pixel's sky
    position (``pos``), its frame-mid elevation (``el``) - against today's tables and parents. Rows whose frame or noise is
    missing are left out (the caller reports them)."""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r['stokes'], r['lst'], r['scope'], r['date'])].append(r)
    out = {}
    for (st, l, b, d), rs in groups.items():
        fr = frames.get(st, l, b, d)
        nz = noise.get((st, l, b, d))
        if 'error' in fr or nz is None:
            continue
        cuts, notes = PCT.removals(st, b, G.parents(l, b), nz, fr['bmaj'], [pos[id(r)][0] for r in rs],
                                   [pos[id(r)][1] for r in rs], elev=[el[id(r)] for r in rs], phase_centre=G.phase_centre(l, b))
        for r, x, n in zip(rs, cuts, notes):
            out[id(r)] = (bool(x), n)
    return out


def _ledger(a):
    """The attempt's gate-ledger rows of its own cell and Stokes, exact duplicate rows collapsed (a cell executed twice)."""
    seen, out = set(), []
    for r in _rows(os.path.join(a['rd'], 'v_deep_ledger.csv')):
        k = tuple(r.values())
        if k in seen or r['stokes'] != a['stokes'] or r['lst'] != a['lst'] or r['scope'] != a['scope']:
            continue
        seen.add(k)
        out.append(r)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the pulsar stage (step 4)
# ---------------------------------------------------------------------------------------------------------------------

def restored_row(cols, led, tpd):
    """A catalog row for a table-cut detection the pulsar stage gives back, in the merged catalogue's columns: those its ledger
    row holds (as the v110 re-admission builds one) and the position the search judged it at (the pixel's); its ID
    <S>R_<lst>_<date>_<band>_<pix_x:04d><pix_y:04d> - field 2 the LST, as every catalog ID (downstream parses it), and the
    pixel as its number, so a detection keeps its ID from pass to pass (one detection per pixel and frame)."""
    r = dict.fromkeys(cols, '')
    n = f"{int(float(tpd['pix_x'])):04d}{int(float(tpd['pix_y'])):04d}"
    r.update(cand_id=f"{tpd['stokes']}{RESTORED_TAG}_{tpd['lst']}_{tpd['date']}_{tpd['scope']}_{n}", lst=tpd['lst'],
             date=tpd['date'], band=tpd['scope'], stokes=tpd['stokes'], cadence='deep', ra=tpd['ra'], dec=tpd['dec'])
    for k in ('peak_jy', 'snr', 'mstruct', 'mstruct_raw', 'sstack', 'jupiter_deg', 'n_v_epoch', 'global_snr', 'size_beams',
              'size_flag', 'nearest_ateam', 'moon_deg', 'axis_norm', 'fp_prediction'):
        if k in r:
            r[k] = led.get(k, '')
    return r


def pulsars(U, manifest, label, catalog_dir=None, out=print):
    """The pulsar stage (module docstring). Returns its counts."""
    D = _deps()
    CD = catalog_dir or U
    atts = attempts(manifest, label)
    if not atts:
        raise Refused(f'pulsar stage: no attempt in {manifest}')
    bad = rules_findings(atts)
    if bad:
        raise Refused(f'pulsar stage: {len(bad)} attempt(s) not searched under the once-in-search rules: {bad[:3]}')
    src = os.path.join(CD, 'master_catalog.csv')
    raw = _rows(src)
    if not raw:
        raise Refused(f'pulsar stage: {src} has no rows')
    cols = list(raw[0].keys())
    frames = Frames(atts, label)
    index = {(r['lst'], r['band'], r['stokes'], r['date'], round(RR.f(r['ra']), 5), round(RR.f(r['dec']), 5)): i
             for i, r in enumerate(raw)}
    rows = list(raw)
    cut, cand, missing = {}, {}, []
    for a in atts:
        led = None
        seen = set()
        for t in _rows(os.path.join(a['rd'], RR.TABLE_PULSAR)):
            k = tuple(t.values())
            if k in seen:                   # an exact duplicate decision (a cell executed twice): once
                continue
            seen.add(k)
            if led is None:
                led = collections.defaultdict(list)
                for r in _ledger(a):
                    led[(r['stokes'], r['date'], int(float(r['pix_x'])), int(float(r['pix_y'])))].append(r)
            px = (t['stokes'], t['date'], int(float(t['pix_x'])), int(float(t['pix_y'])))
            pos = (round(RR.f(t['ra']), 5), round(RR.f(t['dec']), 5))
            if t['cut'] == '0':
                i = index.get((t['lst'], t['scope'], t['stokes'], t['date']) + pos)
                if i is None:               # a kept decision the row line removed after the table - its ledger row says so
                    if not any(x.get('veto_reason') == 'row-line' for x in led.get(px, ())):
                        missing.append(f"kept {t['stokes']} {t['lst']}_{t['scope']} {t['date']} pixel ({t['pix_x']}, "
                                       f"{t['pix_y']}): neither a catalog row nor a 'row-line' ledger row")
                    continue
                cut[i] = (False, t['note'])
            else:
                lr = next((x for x in led.get(px, ()) if x.get('veto_reason') == 'table'), None)
                if lr is None:
                    missing.append(f"cut {t['stokes']} {t['lst']}_{t['scope']} {t['date']} pixel ({t['pix_x']}, {t['pix_y']}): "
                                   f"no 'table' ledger row")
                    continue
                rows.append(restored_row(cols, lr, t))
                i = len(rows) - 1
                cut[i] = (True, t['note'])
                cand[i] = (lr, t)
    if missing:
        raise Refused(f'pulsar stage: {len(missing)} table decision(s) naming a pulsar do not join the catalog and the gate '
                      f'ledger, e.g. {missing[:3]}')
    before = dict(cut)
    members = []
    if not cut:                             # no table decision names a pulsar: no group to judge
        n_keep = n_drop = n_psr = 0
        plog = []
    else:
        n_keep, n_drop, plog, n_psr = RR.pulsar_consistency(rows, cut, LazyDay(frames),
                                                            lambda r: D['excluded'](r['lst'], r['band'], r['date']),
                                                            D['G'], D['LOC'], RR.pulsar_positions(), groups_out=members)
    restored, removed, rec = [], set(), []
    for i, (x, words) in cut.items():
        if i in cand and not x:             # a cut detection of a consistent group: given back
            r = rows[i]
            restored.append(i)
            rec.append(dict(action='restored', lst=r['lst'], scope=r['band'], stokes=r['stokes'], date=r['date'],
                            pix_x=cand[i][1]['pix_x'], pix_y=cand[i][1]['pix_y'], ra=r['ra'], dec=r['dec'], cand_id=r['cand_id'],
                            words=words))
        elif i not in cand and x and not before[i][0]:   # a pulsar-exempted detection of an inconsistent group: removed
            r = rows[i]
            removed.add(i)
            rec.append(dict(action='removed', lst=r['lst'], scope=r['band'], stokes=r['stokes'], date=r['date'], pix_x='',
                            pix_y='', ra=r['ra'], dec=r['dec'], cand_id=r['cand_id'], words=words))
    kept = [r for i, r in enumerate(rows) if i < len(raw) and i not in removed] + [rows[i] for i in sorted(restored)]
    _write(src, cols, kept)
    _write(os.path.join(CD, PULSAR_RECORD), PULSAR_RECORD_COLS, rec)
    _write(os.path.join(CD, PULSAR_REMOVED), cols + ['removal_reason'],
           [dict(rows[i], removal_reason=cut[i][1]) for i in sorted(removed)])
    _write(os.path.join(CD, PULSAR_GROUPS), PULSAR_GROUP_COLS,
           [dict(pulsar=pn, n=n, members='|'.join(rows[i]['cand_id'] for i in mem), sep_arcsec=sep, r99_arcsec=r99,
                 consistent=int(ok)) for (pn, n, sep, r99), (_pn, mem, ok) in zip(plog, members)])
    out(f'pulsar stage: {n_psr} pulsars in table decisions, {len(plog)} groups; restored {len(restored)} table-cut detections '
        f'of pulsar-consistent sources, removed {len(removed)} pulsar-exempted detections of inconsistent sources; '
        f'{len(raw)} merged rows -> {len(kept)}', flush=True)
    return dict(raw=len(raw), kept=len(kept), restored=len(restored), removed=len(removed), groups=len(plog))


# ---------------------------------------------------------------------------------------------------------------------
# the check (step 2)
# ---------------------------------------------------------------------------------------------------------------------

def check(U, manifest, label, merged=None, catalog_dir=None, out=print, pulsar_record=None, pulsar_removed=None):
    """The check (module docstring). Returns its counts; REFUSES (exit 2) on any finding. ``pulsar_record`` /
    ``pulsar_removed``: the pulsar stage's recorded copies (the chain passes them; default <catalog dir>/PULSAR_*)."""
    D = _deps()
    G, PCT, excluded = D['G'], D['PCT'], D['excluded']
    c, sd = CFG.catalog_conform, CFG.search_detect
    epoch = c.DEEP_ELEV_EPOCH
    CD = catalog_dir or U
    atts = attempts(manifest, label)
    if not atts:
        raise Refused(f'conform check: no attempt in {manifest}')
    bad = rules_findings(atts)
    if bad:
        raise Refused(f'conform check: {len(bad)} attempt(s) not searched under the once-in-search rules (re-search those '
                      f'cells, or publish with catalog_conform.MODE=remove under the ruling ID): {bad[:3]}')
    src = os.path.join(CD, 'master_catalog.csv')
    raw = _rows(src)
    if not raw:
        raise Refused(f'conform check: {src} has no rows')
    cols = list(raw[0].keys())
    pulsar_record = pulsar_record or os.path.join(CD, PULSAR_RECORD)
    pulsar_removed = pulsar_removed or os.path.join(CD, PULSAR_REMOVED)
    if not os.path.exists(pulsar_record):
        raise Refused(f'conform check: no {pulsar_record} - the pulsar stage runs before the check')
    prec = _rows(pulsar_record)
    p_restored = {(p['lst'], p['scope'], p['stokes'], p['date'], round(RR.f(p['ra']), 5), round(RR.f(p['dec']), 5)): p['words']
                  for p in prec if p['action'] == 'restored'}
    p_removed = {(p['lst'], p['scope'], p['stokes'], p['date'], round(RR.f(p['ra']), 5), round(RR.f(p['dec']), 5)) for p in prec
                 if p['action'] == 'removed'}
    quar = RR.solar_quarantine(CFG.catalog.SOLAR_AUDIT)
    find = []

    def finding(direction, rule, r, words, pos=None):
        ra, dec = pos if pos else (r.get('ra', ''), r.get('dec', ''))
        find.append(dict(direction=direction, rule=rule, lst=r.get('lst', ''), scope=r.get('scope', r.get('band', '')),
                         stokes=r.get('stokes', ''), date=r.get('date', ''), pix_x=r.get('pix_x', ''), pix_y=r.get('pix_y', ''),
                         ra=ra, dec=dec, cand_id=r.get('cand_id', ''), words=words))

    def record(rule, l, b, st, words, date=''):
        find.append(dict(dict.fromkeys(CHECK_COLS, ''), direction='record', rule=rule, lst=l, scope=b, stokes=st, date=date,
                         words=words))

    finals, notes, n_ledger, bases, inoise, tracks = collections.defaultdict(list), {}, 0, {}, {}, {}
    for a in atts:                          # one attempt at a time: its frames, then its ledger rows
        l, b, st = a['lst'], a['scope'], a['stokes']
        frames = Frames([a], label)
        base = bases[a['cell']] = bases[a['cell']] if a['cell'] in bases else cell_base(a['cell'])
        if base is None:
            record('roots', l, b, st, f'{a["cell"]} is not in {CFG.catalog.DEEP_ROOTS_CSV}: its weights.csv cannot be read')
        if base not in inoise:
            inoise[base] = i_noise_refused(base, b) if base else None
        if (l, b) not in tracks:
            skipped = []
            tracks[(l, b)] = RR.jupiter_track(l, os.path.join(base, 'coadd_final', 'weights.csv') if base else None,
                                              skipped=skipped)
            if skipped:                     # the search refuses a track that leaves a night out (B-30, ruled 'refuse')
                record('jupiter-track', l, b, '', f'the Jupiter track has no position for {len(skipped)} co-added night(s): '
                                                  f'{skipped[:3]}')
        # ---- frame level: daytime, exclusions and the noise rules on every frame the attempt searched
        fn = [r for r in _rows(os.path.join(a['rd'], RR.FRAME_NOISE)) if r['stokes'] == st and r['lst'] == l and r['scope'] == b]
        if not fn:
            record('frame noise', l, b, st, f'{a["rd"]}/{RR.FRAME_NOISE}: no {st} row - the search recorded no frame')
        ref = os.path.join(a['rd'], 'refusals')
        day_ref = {r[2]: r[4] for r in _headerless(os.path.join(ref, f'{l}_{b}_daytime_excluded.csv')) if len(r) > 4 and r[3] == st}
        xref = {r[2] for r in _headerless(os.path.join(ref, f'{l}_{b}_extraction_{st}_excluded.csv')) if len(r) > 2}
        searched = [r for r in fn if r['admitted'] == '1' and r['date'] not in xref]   # extraction refused: not searched
        ledger = [r for r in _ledger(a) if r['final'] == '1' or _veto(r) or r.get('veto_reason') == 'v110:marginal']
        n_ledger += len(ledger)
        frames.prefetch({(st, l, b, r['date']) for r in searched} | {(st, l, b, d) for d in day_ref}
                        | {(st, l, b, r['date']) for r in ledger})
        ft = {}
        for d in {r['date'] for r in searched} | set(day_ref):
            fr = frames.get(st, l, b, d)
            if 'error' in fr:
                record('frame', l, b, st, f'search frame {fr["path"]} unreadable: {fr["error"]}', d)
            else:
                ft[(st, l, b, d)] = fr['times']
        day = RR.daytime_verdicts(ft, quar)
        noise = {}
        for r in fn:
            if r['tblnoise_jy'] != '':
                noise[(st, l, b, r['date'])] = float(r['tblnoise_jy'])
        for r in searched:
            k = (st, l, b, r['date'])
            row = dict(lst=l, scope=b, stokes=st, date=r['date'])
            if day.get(k):
                finding('a', 'daytime', row, f'searched frame: {day[k]}')
            if excluded(l, b, r['date']):
                finding('a', 'excluded', row, RR.EXCLUDED)
            if st == 'V' and r['vsanmed_jy'] and float(r['noise_jy'] or 0) > float(r['vsanity_x']) * float(r['vsanmed_jy']):
                finding('a', 'noise', row, f'searched V frame noise {r["noise_jy"]} > {r["vsanity_x"]} x the V median '
                                           f'{r["vsanmed_jy"]} it was judged with')
            if st == 'I' and inoise.get(base) and r['date'] in inoise[base]:
                v, m = inoise[base][r['date']]
                finding('a', 'noise', row, f'searched I frame: qa3 rms {v:.1f} mJy > {sd.SANITY_X:g} x the cell median {m:.1f}')
        for d, why in sorted(day_ref.items()):
            k = (st, l, b, d)
            if k in ft and not day.get(k):
                finding('b', 'daytime', dict(lst=l, scope=b, stokes=st, date=d), f'refused as {why!r}; today the daytime gate '
                                                                                 f'admits the frame')
        taken = {r['date'] for r in fn}     # nights a search of the attempt built a frame of (an append may take a lifted one up)
        for d in sorted({r[2] for r in _headerless(os.path.join(ref, f'{l}_{b}_datadecisions_excluded.csv')) if len(r) > 2}):
            if not excluded(l, b, d) and d not in taken:   # left out as excluded, admitted now, never searched since
                finding('b', 'excluded', dict(lst=l, scope=b, stokes=st, date=d),
                        'left out by the search as excluded by data_decisions; data_decisions admits it now and no search '
                        'of the attempt took it up - re-search the cell (a night absent from the master record is refused '
                        "by the master membership gate until the cell's masters are rebuilt)")
        for r in _headerless(os.path.join(ref, f'{l}_{b}_sanity_V_excluded.csv')) if st == 'V' else ():
            fnr = next((x for x in fn if x['date'] == r[2]), None)
            if fnr is None or not fnr['vsanmed_jy'] or not (float(fnr['noise_jy'] or 0) > float(fnr['vsanity_x']) * float(fnr['vsanmed_jy'])):
                finding('b', 'noise', dict(lst=l, scope=b, stokes='V', date=r[2]), 'refused by the V noise rule; its recorded '
                                                                                   'noise and median do not refuse it')
        for r in _headerless(os.path.join(ref, f'{l}_{b}_sanity_I_excluded.csv')) if st == 'I' else ():
            if not (inoise.get(base) and r[2] in inoise[base]):
                finding('b', 'noise', dict(lst=l, scope=b, stokes='I', date=r[2]), 'refused by the I noise rule; today\'s '
                                                                                   'qa3 values do not refuse it')
        # ---- candidate level: each ledger row through the search's rules in its order
        today, P, E = {}, {}, {}
        for r in ledger:
            if r.get('veto_reason') == 'v110:marginal':
                finding('b', 'v110', r, 'a v110 marginal detection removed (marginal is kept and labelled, Gregg 2026-09-04)')
                continue
            fr = frames.get(st, l, b, r['date'])
            if 'error' in fr:
                finding('a' if r['final'] == '1' else 'b', 'frame', r, f'search frame unreadable: {fr["error"]}')
                if r['final'] == '1':           # its catalogue row is still backed by it (the ledger's position): one
                    finals[(l, b, st, r['date'])].append((r, (RR.f(r['ra']), RR.f(r['dec']))))   # finding per defect
                continue
            pos = P[id(r)] = sky(fr, r)
            el, words = elevation(l, *pos, fr, epoch)
            E[id(r)] = el
            why = RR.elevation_reason(el, words)
            if why:
                today[id(r)] = ('elevation', why)
                continue
            hit = RR.jupiter_track_hit(*pos, b, tracks[(l, b)])
            if hit:
                today[id(r)] = ('jupiter-track', f'within the track of Jupiter on co-added night {hit}')
                continue
            today[id(r)] = ('', '')
        tab_rows = [r for r in ledger if today.get(id(r)) == ('', '') and _veto(r) not in ('elevation', 'jupiter-track')]
        tab = table_decisions(tab_rows, frames, noise, P, E, G, PCT)
        for r in tab_rows:
            if id(r) not in tab:
                finding('a' if r['final'] == '1' else 'b', 'table', r, 'no frame noise recorded for the frame (frame_noise.csv)')
                today[id(r)] = ('?', '')
            elif tab[id(r)][0]:
                today[id(r)] = ('table', RR.table_reason(tab[id(r)][1]))
        epochs = collections.defaultdict(list)      # the row line, among what passes the table today (the search's set)
        for r in tab_rows:
            if today[id(r)] == ('', ''):
                epochs[r['date']].append(r)
        for rs in epochs.values():
            for i in rowline([int(float(r['pix_y'])) for r in rs], sd.ROWLINE_MIN):
                today[id(rs[i])] = ('row-line', f'row-line: >= {sd.ROWLINE_MIN} detections of this epoch on one pixel row +/-1')
        for r in ledger:
            if id(r) not in today:
                continue
            rule, words = today[id(r)]
            key6 = (l, b, st, r['date'])
            pos = P[id(r)]
            if r['final'] == '1':
                if (key6 + pos) in p_removed:
                    continue                # the pulsar stage removed it (its record)
                finals[key6].append((r, pos))   # its catalogue row is backed by it whatever a rule says today
                if rule == '?':
                    continue                # its table could not be judged: reported above
                if rule:
                    finding('a', rule, r, words, pos)
                    continue
                notes[key6 + pos] = tab.get(id(r), (False, ''))[1]
                an = RR.f(r.get('axis_norm'))   # the search's axis limit (rounded recorded values cannot exceed it when kept)
                if abs(RR.f(r['snr'])) > sd.SIZE_SNR_GATE and np.isfinite(an) and an > sd.AXIS_REMOVE:
                    finding('a', 'axis', r, f'axis ratio {an} x beam > {sd.AXIS_REMOVE} above {sd.SIZE_SNR_GATE:g} sigma', pos)
            elif rule != '?':
                if (key6 + pos) in p_restored:
                    if rule != 'table':
                        finding('b', 'pulsar', r, f'restored by the pulsar stage as a table cut; today: {rule or "kept"}', pos)
                    continue
                if not rule:
                    finding('b', _veto(r), r, f'removed by {r["veto_reason"]!r}; today that rule and the ones before it keep '
                                              f'it (it would reach the search\'s later gates)', pos)

    # ---- catalog level: duplicates, exclusions, backing (one catalog row per final, the pulsar stage's record apart)
    seen, groups = set(), collections.defaultdict(list)
    for i, r in enumerate(raw):
        k6 = (r['lst'], r['band'], r['stokes'], r['date'], round(RR.f(r['ra']), 5), round(RR.f(r['dec']), 5))
        dk = k6[:4] + (round(RR.f(r['ra']), 4), round(RR.f(r['dec']), 4))
        if dk in seen:
            finding('a', 'duplicate', r, RR.DUPLICATE)
        seen.add(dk)
        if excluded(r['lst'], r['band'], r['date']):
            finding('a', 'excluded', r, RR.EXCLUDED)
        if k6 in p_restored:
            continue
        groups[k6[:4]].append((i, RR.f(r['ra']), RR.f(r['dec'])))
    for k in set(groups) | set(finals):
        cr = groups.get(k, [])
        fl = finals.get(k, [])
        uc, uf = match(cr, [(j, p[0], p[1]) for j, (_r, p) in enumerate(fl)])
        for i, _ra, _de in cr:
            if i not in uc:
                finding('a', 'unbacked', raw[i], RR.UNBACKED)
        for j, (r, pos) in enumerate(fl):
            if j not in uf:
                finding('b', 'unpublished', r, 'a final detection without its catalog row', pos)

    vlab = RR.v110_labels(U)
    fkeys = {(r['stokes'], r['lst'], r['band'], r['date']) for r in raw}
    RR.frame_times(U, fkeys)
    BR, BM = RR.frame_beam_ratios(U), RR.frame_bmaj(U)
    kept = []
    for r in raw:
        r = dict(r)
        k6 = (r['lst'], r['band'], r['stokes'], r['date'], round(RR.f(r['ra']), 5), round(RR.f(r['dec']), 5))
        fk = (r['stokes'], r['lst'], r['band'], r['date'])
        v = vlab.get((r['lst'], r['band'], r['date'], round(RR.f(r['ra']), 2), round(RR.f(r['dec']), 2)), '')
        r['fp_prediction'] = r.get('fp_prediction') or ((v if v else 'in-chain pass (verdict file not found)')
                                                        if r['stokes'] == 'V' else '')
        r['conform_note'] = p_restored[k6] if k6 in p_restored else (notes.get(k6, '') or '')
        cols_, why = RR.size_axis(r, BM.get(fk), BR.get(fk))      # production's columns and flags; a removal it would make
        r.update(cols_)                                           # (a row the search left without its own axis measure) is
        if why:                                                   # a finding, not a silent keep
            finding('a', 'size' if why.startswith('gaussfit') else 'axis', r, why)
        kept.append(r)

    if find:
        _write(os.path.join(CD, CHECK_REPORT), CHECK_COLS, find)
        n = collections.Counter((x['direction'], x['rule']) for x in find)
        raise Refused(f'conform check: {len(find)} finding(s) - ' + ', '.join(f'{d}:{r} {v}' for (d, r), v in sorted(n.items()))
                      + f' (direction a: a published row or searched frame a rule removes; b: a removal no rule makes today) '
                      f'- {CHECK_REPORT} written, nothing published; re-search the cells')

    # ---- passed: production's outputs, nothing removed
    removed = _rows(pulsar_removed)
    if len(removed) != len(p_removed):
        raise Refused(f'conform check: {pulsar_removed} holds {len(removed)} rows, the pulsar stage\'s record {len(p_removed)}')
    shutil.copy(merged or src, os.path.join(CD, 'master_catalog_asrun.csv'))
    ocols = cols + ['fp_prediction', 'size_beams', 'size_flag', 'axis_norm', 'axis_flag', 'conform_note']
    _write(src, ocols, kept)
    _write(os.path.join(CD, 'master_catalog_removed.csv'), ocols + ['removal_reason'], removed)
    if os.path.exists(os.path.join(CD, CHECK_REPORT)):
        os.remove(os.path.join(CD, CHECK_REPORT))       # a passing check leaves no report of an earlier refusal
    out(f'conform check: {len(raw)} rows, every removal rule re-evaluated in both directions ({n_ledger} ledger rows, '
        f'{len(atts)} attempts): nothing removed, nothing missed -> {len(kept)} conformant (the pulsar stage restored '
        f'{len(p_restored)}, removed {len(p_removed)})', flush=True)
    return dict(raw=len(raw), kept=len(kept), ledger=n_ledger, attempts=len(atts))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not ((len(argv) == 4 and argv[0] == 'pulsars') or (len(argv) in (5, 7) and argv[0] == 'check')):
        raise SystemExit('usage: conform_check pulsars <run dir> <merged attempts> <label> | '
                         'check <run dir> <merged attempts> <label> <merged catalogue> [<pulsar record> <pulsar removed>]')
    from exopipe.phase3.core import env
    env.legacy()
    n = apply_forwarded(os.environ.get(OVERRIDES_ENV, ''))
    if n:
        print(f'conform_check: {n} config override(s) forwarded by the chain', flush=True)
    if argv[0] == 'pulsars':
        pulsars(argv[1], argv[2], argv[3])
    else:
        check(argv[1], argv[2], argv[3], merged=argv[4], pulsar_record=argv[5] if len(argv) == 7 else None,
              pulsar_removed=argv[6] if len(argv) == 7 else None)


if __name__ == '__main__':
    main()
