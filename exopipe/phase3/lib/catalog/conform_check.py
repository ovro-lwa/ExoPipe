"""B-02 steps 2 and 4 (plan/wave2/B02_DESIGN.md "Steps 2 and 4"; R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32
B-02 (a), R-GREGG-20261014-REFACTOR-DECISIONS-LIST-RECOMMENDATIONS item 5): with every removal rule run once, in the search
(search_detect.REMOVAL_RULES 'once-in-search'), the deep main chain's publish conformance only checks (catalog_conform.MODE
'check'), and the cross-cell pulsar groups are a stage of their own between the merge and the check. Under 'remove'
(production; the parity file) deep_chain.st_conform runs p3s2_conform.py as today and neither function here runs.

pulsars(U, manifest, label): the pulsar stage. The merged catalog and, per published attempt (the harvest manifest), the
search's table decisions that name a pulsar (table_pulsar_decisions.csv: kept as the pulsar's parent, or cut although it lies
within a beam) and the gate-ledger rows of the cut ones; the groups are production's (removal_rules.pulsar_consistency): a
group consistent with its pulsar gets its cut detections back - rows built from their ledger rows, as the v110 re-admission
builds one - and an inconsistent one loses its pulsar-exempted detections. Writes <U>/master_catalog.csv (the merged rows,
changed only so, in the merged columns - deep_chain.ensure_master gives every stage its predecessor's recorded catalog),
PULSAR_RECORD (every row it changed, with its words - the check puts them in the restored rows' conform_note), PULSAR_REMOVED
(the rows it removed, with their reason) and PULSAR_GROUPS (every group).

check(U, manifest, label, merged): the check. It refuses, before reading a row, a run with an attempt not searched under
'once-in-search' at the current RULES_VERSION. Then every removal rule is re-evaluated, from what the run recorded and
today's inputs, in both directions: (a) every catalog row and every final ledger detection passes every rule; (b) every
detection or frame the search removed by one of them is still removed. A ledger row is evaluated where the search evaluated
it: its pixel's sky position on its frame's grid (5 decimals - the catalog's position), the frame's own header (time block,
BMAJ), the noise the search recorded for the frame (frame_noise.csv), the rules in the search's order (elevation floor,
Jupiter track, bright-source table, row line). Any finding: CHECK_REPORT written, REFUSED (nothing published). Else
production's outputs with nothing removed - master_catalog_asrun.csv (a copy of ``merged``, the merge's output),
master_catalog.csv with the conform columns, master_catalog_removed.csv (the pulsar stage's removals) - and the frame
caches, as p3s2_conform.py writes them; so conformant + removed = as-run + the restored PR_ rows, as production's
conformant + removed = as-run + the re-admitted VM_ rows.

python -m exopipe.phase3.lib.catalog.conform_check pulsars <U> <harvest manifest> <label>
python -m exopipe.phase3.lib.catalog.conform_check check <U> <harvest manifest> <label> <merged catalogue>   (deep_chain)
"""
import collections
import csv
import os
import shutil
import sys

import numpy as np

from exopipe.phase3.config import CFG
from exopipe.phase3.lib.search import removal_rules as RR

MODES = ('check', 'remove')                       # catalog_conform.MODE
CHECK_REPORT = 'conform_check.csv'                # every finding of a refused check
CHECK_COLS = ['direction', 'rule', 'lst', 'scope', 'stokes', 'date', 'pix_x', 'pix_y', 'ra', 'dec', 'cand_id', 'words']
PULSAR_RECORD = 'pulsar_stage.csv'                # every catalog row the pulsar stage restored or removed
PULSAR_RECORD_COLS = ['action', 'lst', 'scope', 'stokes', 'date', 'pix_x', 'pix_y', 'ra', 'dec', 'cand_id', 'words']
PULSAR_GROUPS = 'pulsar_groups.csv'               # every group the stage judged
PULSAR_GROUP_COLS = ['pulsar', 'n', 'sep_arcsec', 'r99_arcsec', 'consistent']
PULSAR_REMOVED = 'master_catalog_pulsar_removed.csv'   # the rows the pulsar stage removed, each with its removal_reason
B02_VETOES = ('elevation', 'jupiter-track', 'table', 'row-line')   # the search's removal rules a ledger row names


def conform_mode(value=None):
    """catalog_conform.MODE (``value`` checked); an unknown value REFUSES - a typo never selects either."""
    value = CFG.catalog_conform.MODE if value is None else value
    if value not in MODES:
        raise RuntimeError(f'catalog_conform.MODE = {value!r} is not one of {MODES} - REFUSING')
    return value


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


def _veto(r):
    """A ledger row's B-02 rule ('elevation', 'jupiter-track', 'table', 'row-line') or ''."""
    v = r.get('veto_reason') or ''
    v = 'jupiter-track' if v.startswith('jupiter-track:') else v
    return v if v in B02_VETOES else ''


# ---------------------------------------------------------------------------------------------------------------------
# the published attempts and what each recorded
# ---------------------------------------------------------------------------------------------------------------------

def attempts(manifest, label):
    """[{cell, lst, scope, stokes, dir, rd}] of the harvest manifest (deep_chain st_harvest: cell, stokes, node, att, dir, ...);
    rd = the attempt's run directory <dir>/transients/Run_<label>."""
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


class Frames:
    """The archived search frames of the attempts (removal_rules' layout: <dir>/Run_<label>/{iresid,vresid}/<cell>/...), read
    once each: header, celestial WCS, DATE-OBS, DATE-END, BMAJ; or the error that stopped the read."""
    def __init__(self, atts, label):
        self.dir = {(a['cell'], a['stokes']): a['dir'] for a in atts}
        self.label = label
        self._f = {}

    def path(self, stokes, lst, scope, date):
        d = self.dir.get((f'{lst}_{scope}', stokes))
        sub = 'iresid' if stokes == 'I' else 'vresid'
        return os.path.join(d, f'Run_{self.label}', sub, f'{lst}_{scope}', f'{date}_{stokes}_redo_resid_{scope}.fits') if d else None

    def get(self, stokes, lst, scope, date):
        k = (stokes, lst, scope, date)
        if k not in self._f:
            from astropy.io import fits
            from astropy.wcs import WCS
            p = self.path(*k)
            try:
                h = fits.getheader(p)
                self._f[k] = dict(path=p, header=h, wcs=WCS(h).celestial, times=(str(h['DATE-OBS']), str(h['DATE-END'])),
                                  bmaj=float(h['BMAJ']))
            except Exception as e:
                self._f[k] = dict(path=p, error=f'{type(e).__name__}: {e}')
        return self._f[k]


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


def table_decisions(rows, frames, noise, epoch, G, PCT):
    """{id(row): (cut, note)} of the bright-source table for ledger rows (all of one frame per call is not required): the
    search's inputs - TBLNOISE from the attempt's frame_noise.csv (``noise`` {(Stokes, lst, scope, date): Jy}), the frame's
    BMAJ, the pixel's sky position, its frame-mid elevation - against today's tables and parents. Rows whose frame or noise
    is missing are left out (the caller reports them)."""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r['stokes'], r['lst'], r['scope'], r['date'])].append(r)
    out = {}
    for (st, l, b, d), rs in groups.items():
        fr = frames.get(st, l, b, d)
        nz = noise.get((st, l, b, d))
        if 'error' in fr or nz is None:
            continue
        pos = [sky(fr, r) for r in rs]
        el = [RR.frame_elevation(l, ra, de, fr['header'], epoch=epoch)[0] for ra, de in pos]
        cuts, notes = PCT.removals(st, b, G.parents(l, b), nz, fr['bmaj'], [p[0] for p in pos], [p[1] for p in pos], elev=el,
                                   phase_centre=G.phase_centre(l, b))
        for r, x, n in zip(rs, cuts, notes):
            out[id(r)] = (bool(x), n)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the pulsar stage (step 4)
# ---------------------------------------------------------------------------------------------------------------------

def _ledger_by_pixel(rd):
    """{(Stokes, date, pix_x, pix_y): the 'table' ledger row there} of an attempt's gate ledger."""
    out = {}
    for r in _rows(os.path.join(rd, 'v_deep_ledger.csv')):
        if r.get('veto_reason') != 'table':
            continue
        try:
            out.setdefault((r['stokes'], r['date'], int(float(r['pix_x'])), int(float(r['pix_y']))), r)
        except (TypeError, ValueError):
            pass
    return out


def restored_row(cols, led, tpd, j):
    """A catalog row for a table-cut detection the pulsar stage gives back, in the merged catalogue's columns: those its ledger
    row holds (as the v110 re-admission builds one) and the position the search judged it at (the pixel's)."""
    r = dict.fromkeys(cols, '')
    r.update(cand_id=f"PR_{tpd['stokes']}_{tpd['lst']}_{tpd['date']}_{tpd['scope']}_{j:04d}", lst=tpd['lst'], date=tpd['date'],
             band=tpd['scope'], stokes=tpd['stokes'], cadence='deep', ra=tpd['ra'], dec=tpd['dec'])
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
        for t in _rows(os.path.join(a['rd'], RR.TABLE_PULSAR)):
            pos = (round(RR.f(t['ra']), 5), round(RR.f(t['dec']), 5))
            if t['cut'] == '0':
                i = index.get((t['lst'], t['scope'], t['stokes'], t['date']) + pos)
                if i is None:
                    continue                # a kept decision whose detection a later gate (the row line) removed
                cut[i] = (False, t['note'])
            else:
                led = _ledger_by_pixel(a['rd']) if led is None else led
                lr = led.get((t['stokes'], t['date'], int(t['pix_x']), int(t['pix_y'])))
                if lr is None:
                    missing.append(f"{t['stokes']} {t['lst']}_{t['scope']} {t['date']} pixel ({t['pix_x']}, {t['pix_y']})")
                    continue
                rows.append(restored_row(cols, lr, t, len(rows)))
                i = len(rows) - 1
                cut[i] = (True, t['note'])
                cand[i] = (lr, t)
    if missing:
        raise Refused(f'pulsar stage: {len(missing)} cut table decision(s) naming a pulsar without their ledger row '
                      f"(veto 'table'), e.g. {missing[:3]}")
    before = dict(cut)
    if not cut:                             # no table decision names a pulsar: no group to judge
        n_keep = n_drop = n_psr = 0
        plog = []
    else:
        ft = {}
        for k in {(r['stokes'], r['lst'], r['band'], r['date']) for r in rows}:   # every row a group can draw on
            fr = frames.get(*k)                                                    # (production: all merged rows)
            if 'error' in fr:
                raise Refused(f'pulsar stage: frame {fr["path"]} unreadable ({fr["error"]})')
            ft[k] = fr['times']
        day = RR.daytime_verdicts(ft, RR.solar_quarantine(CFG.catalog.SOLAR_AUDIT))
        n_keep, n_drop, plog, n_psr = RR.pulsar_consistency(rows, cut, day, lambda r: D['excluded'](r['lst'], r['band'], r['date']),
                                                            D['G'], D['LOC'], RR.pulsar_positions())
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
    with open(src + '.part', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(kept)
    os.replace(src + '.part', src)
    with open(os.path.join(CD, PULSAR_RECORD), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=PULSAR_RECORD_COLS)
        w.writeheader()
        w.writerows(rec)
    with open(os.path.join(CD, PULSAR_REMOVED), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols + ['removal_reason'], extrasaction='ignore')
        w.writeheader()
        w.writerows(dict(rows[i], removal_reason=cut[i][1]) for i in sorted(removed))
    with open(os.path.join(CD, PULSAR_GROUPS), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(PULSAR_GROUP_COLS)
        for (pn, n, sep, r99) in plog:
            w.writerow([pn, n, sep, r99, int(sep <= r99)])
    out(f'pulsar stage: {n_psr} pulsars in table decisions, {len(plog)} groups; restored {len(restored)} table-cut detections '
        f'of pulsar-consistent sources, removed {len(removed)} pulsar-exempted detections of inconsistent sources; '
        f'{len(raw)} merged rows -> {len(kept)}', flush=True)
    return dict(raw=len(raw), kept=len(kept), restored=len(restored), removed=len(removed), groups=len(plog))


# ---------------------------------------------------------------------------------------------------------------------
# the check (step 2)
# ---------------------------------------------------------------------------------------------------------------------

def check(U, manifest, label, merged=None, catalog_dir=None, out=print):
    """The check (module docstring). Returns its counts; REFUSES (exit 2) on any finding."""
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
    frames = Frames(atts, label)
    quar = RR.solar_quarantine(CFG.catalog.SOLAR_AUDIT)
    find = []

    def finding(direction, rule, r, words, pos=None):
        ra, dec = pos if pos else (r.get('ra', ''), r.get('dec', ''))
        find.append(dict(direction=direction, rule=rule, lst=r.get('lst', ''), scope=r.get('scope', r.get('band', '')),
                         stokes=r.get('stokes', ''), date=r.get('date', ''), pix_x=r.get('pix_x', ''), pix_y=r.get('pix_y', ''),
                         ra=ra, dec=dec, cand_id=r.get('cand_id', ''), words=words))

    # the pulsar stage's record: rows it restored (not ledger finals) and removed (ledger finals not in the catalog)
    if not os.path.exists(os.path.join(CD, PULSAR_RECORD)):
        raise Refused(f'conform check: no {PULSAR_RECORD} in {CD} - the pulsar stage runs before the check')
    prec = _rows(os.path.join(CD, PULSAR_RECORD))
    p_restored = {(p['lst'], p['scope'], p['stokes'], p['date'], round(RR.f(p['ra']), 5), round(RR.f(p['dec']), 5)): p['words']
                  for p in prec if p['action'] == 'restored'}
    p_removed = {(p['lst'], p['scope'], p['stokes'], p['date'], round(RR.f(p['ra']), 5), round(RR.f(p['dec']), 5)) for p in prec
                 if p['action'] == 'removed'}

    # ---- frame level: daytime, exclusions and the noise rules on every frame each attempt processed
    noise, ledger, tracks, inoise, bases = {}, [], {}, {}, {}
    for a in atts:
        l, b, st = a['lst'], a['scope'], a['stokes']
        base = bases.setdefault(a['cell'], cell_base(a['cell']))
        if base is None:
            find.append(dict(dict.fromkeys(CHECK_COLS, ''), direction='record', rule='roots', lst=l, scope=b, stokes=st,
                             words=f'{a["cell"]} is not in {CFG.catalog.DEEP_ROOTS_CSV}: its weights.csv cannot be read'))
        if base not in inoise:
            inoise[base] = i_noise_refused(base, b) if base else None
        fn = [r for r in _rows(os.path.join(a['rd'], RR.FRAME_NOISE)) if r['stokes'] == st]
        if not fn:
            find.append(dict(dict.fromkeys(CHECK_COLS, ''), direction='record', rule='frame noise', lst=l, scope=b, stokes=st,
                             words=f'{a["rd"]}/{RR.FRAME_NOISE}: no {st} row - the search recorded no frame'))
        ref = os.path.join(a['rd'], 'refusals')
        day_ref = {(r[3], r[2]): r[4] for r in _headerless(os.path.join(ref, f'{l}_{b}_daytime_excluded.csv')) if len(r) > 4
                   and r[3] == st}
        keys = {(st, l, b, r['date']) for r in fn if r['admitted'] == '1'} | {(st, l, b, d) for (_s, d) in day_ref}
        ft = {}
        for k in keys:
            fr = frames.get(*k)
            if 'error' in fr:
                find.append(dict(dict.fromkeys(CHECK_COLS, ''), direction='record', rule='frame', lst=l, scope=b, stokes=st,
                                 date=k[3], words=f'search frame {fr["path"]} unreadable: {fr["error"]}'))
            else:
                ft[k] = fr['times']
        day = RR.daytime_verdicts(ft, quar)
        for r in fn:
            k = (st, l, b, r['date'])
            if r['tblnoise_jy'] != '':
                noise[k] = float(r['tblnoise_jy'])
            if r['admitted'] != '1':
                continue
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
        for (s_, d), why in sorted(day_ref.items()):
            k = (st, l, b, d)
            if k in ft and not day.get(k):
                finding('b', 'daytime', dict(lst=l, scope=b, stokes=st, date=d), f'refused as {why!r}; today the daytime gate '
                                                                                 f'admits the frame')
        for r in _headerless(os.path.join(ref, f'{l}_{b}_sanity_V_excluded.csv')) if st == 'V' else ():
            fnr = next((x for x in fn if x['date'] == r[2]), None)
            if fnr is None or not fnr['vsanmed_jy'] or not (float(fnr['noise_jy'] or 0) > float(fnr['vsanity_x']) * float(fnr['vsanmed_jy'])):
                finding('b', 'noise', dict(lst=l, scope=b, stokes='V', date=r[2]), 'refused by the V noise rule; its recorded '
                                                                                   'noise and median do not refuse it')
        for r in _headerless(os.path.join(ref, f'{l}_{b}_sanity_I_excluded.csv')) if st == 'I' else ():
            if not (inoise.get(base) and r[2] in inoise[base]):
                finding('b', 'noise', dict(lst=l, scope=b, stokes='I', date=r[2]), 'refused by the I noise rule; today\'s '
                                                                                   'qa3 values do not refuse it')
        seen = set()
        for r in _rows(os.path.join(a['rd'], 'v_deep_ledger.csv')):
            k = tuple(r.values())
            if k in seen or r['stokes'] != st:
                continue
            seen.add(k)
            if r['final'] == '1' or _veto(r) or r.get('veto_reason') == 'v110:marginal':
                ledger.append(r)
        if (l, b) not in tracks:
            skipped = []
            tracks[(l, b)] = RR.jupiter_track(l, os.path.join(base, 'coadd_final', 'weights.csv') if base else None,
                                              skipped=skipped)
            if skipped:                     # the search refuses a track that leaves a night out (B-30, ruled 'refuse')
                find.append(dict(dict.fromkeys(CHECK_COLS, ''), direction='record', rule='jupiter-track', lst=l, scope=b,
                                 words=f'the Jupiter track has no position for {len(skipped)} co-added night(s): {skipped[:3]}'))

    # ---- candidate level: each ledger row through the search's rules in its order
    today, P = {}, {}                               # P: each row's sky position, computed once
    for r in ledger:
        if r.get('veto_reason') == 'v110:marginal':
            finding('b', 'v110', r, 'a v110 marginal detection removed (marginal is kept and labelled, Gregg 2026-09-04)')
            continue
        fr = frames.get(r['stokes'], r['lst'], r['scope'], r['date'])
        if 'error' in fr:
            finding('a' if r['final'] == '1' else 'b', 'frame', r, f'search frame unreadable: {fr["error"]}')
            continue
        pos = P[id(r)] = sky(fr, r)
        el, words = RR.frame_elevation(r['lst'], *pos, fr['header'], epoch=epoch)
        why = RR.elevation_reason(el, words)
        if why:
            today[id(r)] = ('elevation', why)
            continue
        hit = RR.jupiter_track_hit(*pos, r['scope'], tracks[(r['lst'], r['scope'])])
        if hit:
            today[id(r)] = ('jupiter-track', f'within the track of Jupiter on co-added night {hit}')
            continue
        today[id(r)] = ('', '')
    tab_rows = [r for r in ledger if today.get(id(r)) == ('', '') and _veto(r) not in ('elevation', 'jupiter-track')]
    tab = table_decisions(tab_rows, frames, noise, epoch, G, PCT)
    for r in tab_rows:
        if id(r) not in tab:
            finding('a' if r['final'] == '1' else 'b', 'table', r, 'no frame noise recorded for the frame (frame_noise.csv)')
            today[id(r)] = ('?', '')
        elif tab[id(r)][0]:
            today[id(r)] = ('table', RR.table_reason(tab[id(r)][1]))
    epochs = collections.defaultdict(list)          # the row line, among what passes the table today (the search's set)
    for r in tab_rows:
        if today[id(r)] == ('', ''):
            epochs[(r['stokes'], r['lst'], r['scope'], r['date'])].append(r)
    for rs in epochs.values():
        for i in rowline([int(float(r['pix_y'])) for r in rs], sd.ROWLINE_MIN):
            today[id(rs[i])] = ('row-line', f'row-line: >= {sd.ROWLINE_MIN} detections of this epoch on one pixel row +/-1')
    notes, finals = {}, collections.defaultdict(list)
    for r in ledger:
        if id(r) not in today:
            continue
        rule, words = today[id(r)]
        key6 = (r['lst'], r['scope'], r['stokes'], r['date'])
        pos = P[id(r)]
        if r['final'] == '1':
            if (key6 + pos) in p_removed:
                continue                    # the pulsar stage removed it (its record)
            finals[key6].append((r, pos))   # its catalogue row is backed by it whatever a rule says today
            if rule == '?':
                continue                    # its table could not be judged: reported above
            if rule:
                finding('a', rule, r, words, pos)
            else:
                notes[key6 + pos] = tab.get(id(r), (False, ''))[1]
                s_ = abs(RR.f(r['snr']))
                if s_ > sd.SIZE_SNR_GATE:
                    an, sb, se = RR.f(r.get('axis_norm')), RR.f(r.get('size_beams')), RR.f(r.get('size_err_beams'))
                    if np.isfinite(an) and an > sd.AXIS_REMOVE:
                        finding('a', 'axis', r, f'axis ratio {an} x beam > {sd.AXIS_REMOVE} above {sd.SIZE_SNR_GATE:g} sigma', pos)
                    if np.isfinite(sb) and sb - (se if np.isfinite(se) else 0.0) > sd.SIZE_FLOOR_BEAMS:
                        finding('a', 'size', r, f'size (Maj - 1 sigma) {sb - (se if np.isfinite(se) else 0.0):.2f} beams > '
                                                f'{sd.SIZE_FLOOR_BEAMS} above {sd.SIZE_SNR_GATE:g} sigma', pos)
        elif rule != '?':
            if (key6 + pos) in p_restored:
                if rule != 'table':
                    finding('b', 'pulsar', r, f'restored by the pulsar stage as a table cut; today: {rule or "kept"}', pos)
                continue
            if not rule:
                finding('b', _veto(r), r, f'removed by {r["veto_reason"]!r}; today that rule and the ones before it keep it '
                                          f'(it would reach the search\'s later gates)', pos)

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
        r.update(cols_)                                           # (a row the search left without its own size or axis
        if why:                                                   # measure) is a finding, not a silent keep
            finding('a', 'size' if why.startswith('gaussfit') else 'axis', r, why)
        kept.append(r)

    if find:
        with open(os.path.join(CD, CHECK_REPORT), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=CHECK_COLS)
            w.writeheader()
            w.writerows(find)
        n = collections.Counter((x['direction'], x['rule']) for x in find)
        raise Refused(f'conform check: {len(find)} finding(s) - ' + ', '.join(f'{d}:{r} {v}' for (d, r), v in sorted(n.items()))
                      + f' (direction a: a published row or searched frame a rule removes; b: a removal no rule makes today) '
                      f'- {CHECK_REPORT} written, nothing published; re-search the cells')

    # ---- passed: production's outputs, nothing removed
    shutil.copy(merged or src, os.path.join(CD, 'master_catalog_asrun.csv'))
    ocols = cols + ['fp_prediction', 'size_beams', 'size_flag', 'axis_norm', 'axis_flag', 'conform_note']
    with open(src, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=ocols, extrasaction='ignore')
        w.writeheader()
        w.writerows(kept)
    removed = _rows(os.path.join(CD, PULSAR_REMOVED))
    if len(removed) != len(p_removed):
        raise Refused(f'conform check: {PULSAR_REMOVED} holds {len(removed)} rows, the pulsar stage\'s record {len(p_removed)}')
    with open(os.path.join(CD, 'master_catalog_removed.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=ocols + ['removal_reason'], extrasaction='ignore')
        w.writeheader()
        w.writerows(removed)
    if os.path.exists(os.path.join(CD, CHECK_REPORT)):
        os.remove(os.path.join(CD, CHECK_REPORT))       # a passing check leaves no report of an earlier refusal
    out(f'conform check: {len(raw)} rows, every removal rule re-evaluated in both directions ({len(ledger)} ledger rows, '
        f'{len(atts)} attempts): nothing removed, nothing missed -> {len(kept)} conformant (the pulsar stage restored '
        f'{len(p_restored)}, removed {len(p_removed)})', flush=True)
    return dict(raw=len(raw), kept=len(kept), ledger=len(ledger), attempts=len(atts))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not ((len(argv) == 4 and argv[0] == 'pulsars') or (len(argv) == 5 and argv[0] == 'check')):
        raise SystemExit('usage: conform_check pulsars <run dir> <harvest manifest> <label> | '
                         'check <run dir> <harvest manifest> <label> <merged catalogue>')
    from exopipe.phase3.core import env
    env.legacy()
    if argv[0] == 'pulsars':
        pulsars(argv[1], argv[2], argv[3])
    else:
        check(argv[1], argv[2], argv[3], merged=argv[4])


if __name__ == '__main__':
    main()
