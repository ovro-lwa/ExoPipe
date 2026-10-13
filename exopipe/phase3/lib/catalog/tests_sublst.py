"""Content tests of the sub-LST catalogue chain (catalog --tier sublst), registered with core/validate.

Tier-unique classes: catalog_preflight (also the snapshot tier's), tier_rule_removals, window_tie, leak_veto_table,
rejected_record, compare_copy. Sub-LST checks of SHARED classes (whose generic tests are the catalog-deep unit's,
lib/catalog/tests.py) are registered under the shared class name with a 'sublst_' test name and apply only to a sub-LST
product (ctx tier='sublst', or the catalogue's cadence column '10min'); on any other product they pass with value 'n/a':
conformant_catalog (TABLE-NOTES), fitted_catalog (F3 and the blanked deep systematic), selection (split partition, SNR,
subset), source_catalog (one Stokes, unique ids, members, RA range), localization_table (M12, err >= err_raw, S-G19),
merged_record (no row on an excluded epoch), frame_table (keys, DATE-OBS < DATE-END).

Every test reads the file's CONTENT (rows, columns, recomputed rules) and returns (passed, value, detail). Imports are
inside the tests (core/validate imports this module to discover them).
"""
import csv
import glob
import math
import os

from exopipe.phase3.core.validate import content_test

csv.field_size_limit(10 ** 9)
_NA = (True, 'n/a', 'not a sub-LST product')
STATUSES_OK = ('tied', 'untied-few', 'untied-inconsistent')


def _rows(path):
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        return rd.fieldnames or [], list(rd)


def _float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _run_dir(path):
    """The catalogue directory holding master_catalog.csv for a product path (the file's dir or an ancestor, 3 levels)."""
    d = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    for _ in range(3):
        if os.path.exists(os.path.join(d, 'master_catalog.csv')):
            return d
        d = os.path.dirname(d)
    return None


def _is_sublst(path, ctx):
    t = ctx.get('tier')
    if t:
        return t == 'sublst'
    u = ctx.get('run_dir') or _run_dir(path)
    if not u:
        return False
    try:
        with open(os.path.join(u, 'master_catalog.csv'), newline='') as f:
            r = next(csv.DictReader(f), None)
        return bool(r) and (r.get('cadence') or '') == '10min'
    except OSError:
        return False


def _legacy(name):
    import sys
    from exopipe.phase3.config import CFG
    for r in CFG.site.LEGACY_CODE_ROOTS:
        if os.path.basename(r.rstrip('/')) == name:
            if r not in sys.path:
                sys.path.append(r)
            return r
    raise RuntimeError(f'no legacy code root {name!r} in the site file')


# ---------------------------------------------------------------------------------------------------------------------
# tier-unique classes
# ---------------------------------------------------------------------------------------------------------------------

@content_test('catalog_preflight')
def preflight_report(path, **ctx):
    """The preflight report of a catalog tier run: a plan with lines, and no [lit] binding that differs from config."""
    import json
    rep = json.load(open(path))
    plan = rep.get('plan') or []
    mism = [b['key'] for b in rep.get('binding') or [] if b.get('status') == 'MISMATCH']
    if not plan:
        return False, 0, 'empty plan'
    if mism:
        return False, len(mism), f'binding mismatches: {mism[:5]}'
    nb = len(rep.get('binding') or [])
    return True, len(plan), f'{len(plan)} plan lines, {nb} bindings checked, 0 mismatches'


@content_test('tier_rule_removals')
def tier_rules_recomputed(path, **ctx):
    """Tier rules 3/4 and the 10-minute elevation floor recomputed with the in-place functions: every removed row satisfies
    its rule (and carries it), no kept row of master_catalog.csv satisfies any (kept rows at their PEAK position when
    fitted positions were added later)."""
    import numpy as np
    clv, ops = _legacy('clv15'), _legacy('exopipe_ops')
    import phase3_tier_rules as TR
    import gate_inputs as G
    base = os.path.basename(path)
    hdr, rows = _rows(path)
    floor = G.TENMIN_ELEV_FLOOR_DEG
    if base == 'master_catalog_deepsearch_refused.csv':
        bad = [r['cand_id'] for r in rows if not r.get('deep_search_gate')
               or not TR.deep_search_refused(r['lst'], r['date'][:10], r['band'], r['stokes'])]
        return not bad, len(rows), f'{len(rows)} removed by tier rule 3, {len(bad)} not satisfying it' + (
            f' e.g. {bad[:3]}' if bad else '')
    if base == 'master_catalog_fluxscale_refused.csv':
        bad = [r['cand_id'] for r in rows if not r.get('flux_scale_rule')
               or not TR.flux_scale_failed(r['lst'], r['date'][:10], r['band'])]
        return not bad, len(rows), f'{len(rows)} removed by tier rule 4, {len(bad)} not satisfying it' + (
            f' e.g. {bad[:3]}' if bad else '')
    if base == 'master_catalog_elevation_floor.csv':
        bad = []
        for r in rows:
            e = float(G.window_elevation(r['lst'], r['date'], float(r['ra']), float(r['dec'])))
            if not (np.isfinite(e) and e < floor) or abs(round(e, 2) - float(r['window_elevation_deg'])) > 0.011:
                bad.append(r['cand_id'])
        return not bad, len(rows), f'{len(rows)} removed below {floor} deg, {len(bad)} not below it / wrong value' + (
            f' e.g. {bad[:3]}' if bad else '')
    if base == 'master_catalog.csv':
        if not _is_sublst(path, ctx):
            return _NA
        b3 = b4 = be = 0
        ex = []
        rac, decc = ('peak_ra', 'peak_dec') if 'peak_ra' in hdr else ('ra', 'dec')
        for r in rows:
            if TR.deep_search_refused(r['lst'], r['date'][:10], r['band'], r['stokes']):
                b3 += 1
                ex.append(r['cand_id'])
            if TR.flux_scale_failed(r['lst'], r['date'][:10], r['band']):
                b4 += 1
                ex.append(r['cand_id'])
            e = float(G.window_elevation(r['lst'], r['date'], float(r[rac]), float(r[decc])))
            if not (np.isfinite(e) and e >= floor):
                be += 1
                ex.append(r['cand_id'])
        ok = not (b3 or b4 or be)
        return ok, len(rows), (f'{len(rows)} kept rows: {b3} deep-search refused, {b4} failed flux scale, {be} below '
                               f'{floor} deg at {rac}/{decc}' + (f' e.g. {ex[:3]}' if ex else ''))
    return True, len(rows), f'{base}: not a tier-rule file ({len(rows)} rows)'


@content_test('window_tie')
def window_tie_rows(path, **ctx):
    """pertie_detections.csv: one row per cand_id, every row a finite positive systematic with its source and a known tie
    status; every selected detection (selected_[IV]_[sw]b.csv beside the pertie dir) has its row (coverage, T1)."""
    hdr, rows = _rows(path)
    if os.path.basename(path) != 'pertie_detections.csv':
        return bool(hdr), len(rows), f'{len(rows)} rows, {len(hdr)} columns'
    need = ('cand_id', 'tie_status', 'sys_arcmin', 'sys_source')
    miss = [c for c in need if c not in hdr]
    if miss:
        return False, len(rows), f'missing columns {miss}'
    ids = [r['cand_id'] for r in rows]
    dup = len(ids) - len(set(ids))
    badsys = [r['cand_id'] for r in rows if not ((_float(r['sys_arcmin']) or 0) > 0 and r['sys_source'])]
    badst = sorted({r['tie_status'] for r in rows} - set(STATUSES_OK))
    u = ctx.get('selected_dir') or os.path.dirname(os.path.dirname(os.path.abspath(path)))
    sel = set()
    for f in glob.glob(os.path.join(u, 'selected_[IV]_[sw]b.csv')):
        sel |= {r['cand_id'] for r in _rows(f)[1]}
    nocov = sorted(sel - set(ids))
    ok = not (dup or badsys or badst or nocov)
    return ok, len(rows), (f'{len(rows)} rows, {dup} duplicate cand_ids, {len(badsys)} without a positive systematic, '
                           f'unknown statuses {badst}, {len(nocov)} of {len(sel)} selected without a row'
                           + (f' e.g. {nocov[:3]}' if nocov else ''))


@content_test('leak_veto_table')
def leak_veto_rows(path, **ctx):
    """leakveto_<L>_<band>_<S>.csv: veto in {'', elevation, leak-envelope}; an elevation veto only on the affected side below
    the floor, and every clear case (affected side, elevation < floor - 0.1, Master >= MASTER_MIN_JY + 0.001) vetoed;
    rows = the conformant candidates of that (LST, band, Stokes) in master_catalog.csv (when the run dir is beside it)."""
    from exopipe.phase3.config import CFG
    v = CFG.catalog_sublst_veto
    hdr, rows = _rows(path)
    if not rows:
        return False, 0, 'empty veto table (the module writes one only for a pair with kept candidates)'
    lst, band, st = rows[0]['lst'], rows[0]['band'], rows[0]['stokes']
    badv = [r['cand_id'] for r in rows if r['veto'] not in ('', 'elevation', 'leak-envelope')]
    bade, miss = [], []
    for r in rows:
        el, mj, aff = _float(r['elevation_deg']), _float(r['master_jy']), r['affected_side'] in ('1', 'True')
        if r['veto'] == 'elevation' and not (aff and el is not None and el <= v.ELEV_FLOOR_DEG + 0.05):
            bade.append(r['cand_id'])
        if aff and el is not None and el < v.ELEV_FLOOR_DEG - 0.1 and mj is not None and mj >= v.MASTER_MIN_JY + 0.001 \
                and r['veto'] != 'elevation':
            miss.append(r['cand_id'])
    other = [r['cand_id'] for r in rows if (r['lst'], r['band'], r['stokes']) != (lst, band, st)]
    u = ctx.get('run_dir') or _run_dir(path)
    nset = ''
    if u:
        want = {r['cand_id'] for r in _rows(os.path.join(u, 'master_catalog.csv'))[1]
                if (r['lst'], r['band'], r['stokes']) == (lst, band, st)}
        got = {r['cand_id'] for r in rows}
        if want != got:
            nset = f'; rows {len(got)} != conformant candidates {len(want)} ({len(want - got)} missing, {len(got - want)} extra)'
    ok = not (badv or bade or miss or other or nset)
    return ok, len(rows), (f'{lst} {band} {st}: {len(rows)} rows, {sum(1 for r in rows if r["veto"])} vetoed; '
                           f'{len(badv)} unknown veto, {len(bade)} elevation vetoes off the rule, {len(miss)} clear cases not '
                           f'vetoed, {len(other)} rows of another pair{nset}')


@content_test('rejected_record')
def rejected_record_consistent(path, **ctx):
    """rejected.csv (+ rejected_index.npz, detection_ts.csv beside it): every rejected row has a frame time; no served
    cand_id (master_catalog.csv) is rejected; the index has one entry per row and its byte offsets point at each row;
    every served detection has a detection_ts row."""
    import numpy as np
    d = os.path.dirname(os.path.abspath(path))
    hdr, rows = _rows(path)
    nots = [r['cand_id'] for r in rows if not r.get('ts')]
    served = {r['cand_id'] for r in _rows(os.path.join(d, 'master_catalog.csv'))[1]} \
        if os.path.exists(os.path.join(d, 'master_catalog.csv')) else set()
    leak = sorted(served & {r['cand_id'] for r in rows})
    idx = os.path.join(d, 'rejected_index.npz')
    bad_idx = ''
    if not os.path.exists(idx):
        bad_idx = 'no rejected_index.npz'
    else:
        z = np.load(idx)
        if len(z['ra']) != len(rows) or len(z['offset']) != len(rows):
            bad_idx = f'index holds {len(z["ra"])} entries for {len(rows)} rows'
        else:
            with open(path, 'rb') as f:
                step = max(1, len(rows) // 50)
                for i in range(0, len(rows), step):
                    f.seek(int(z['offset'][i]))
                    line = f.readline().decode('utf-8', 'replace')
                    if rows[i]['cand_id'] not in line:
                        bad_idx = f'offset of row {i} does not point at it'
                        break
    dts = os.path.join(d, 'detection_ts.csv')
    nodts = sorted(served - {r['cand_id'] for r in _rows(dts)[1]}) if os.path.exists(dts) else sorted(served)
    ok = not (nots or leak or bad_idx or nodts)
    return ok, len(rows), (f'{len(rows)} rejected rows: {len(nots)} without a frame time, {len(leak)} served cand_ids '
                           f'rejected, index: {bad_idx or "ok"}, {len(nodts)} of {len(served)} served detections without '
                           f'detection_ts')


@content_test('compare_copy')
def compare_copy_equal(path, **ctx):
    """The compare copy (compare_<L>/new_<arm>/catalog): the window-tie tables whole (pertie_detections.csv non-empty); with
    ctx source=<catalogue dir>, every top-level CSV and pertie file equals its source (sha1)."""
    import hashlib

    def sha(p):
        h = hashlib.sha1()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(1 << 20), b''):
                h.update(b)
        return h.hexdigest()
    pd = os.path.join(path, 'pertie', 'pertie_detections.csv')
    if not (os.path.exists(pd) and os.path.getsize(pd) > 0):
        return False, 0, 'pertie/pertie_detections.csv missing or empty in the copy'
    src = ctx.get('source')
    if not src:
        n = len(glob.glob(os.path.join(path, '*.csv')))
        return n > 0, n, f'{n} top-level CSVs, window-tie tables present (no source given: equality not checked)'
    diff, n = [], 0
    for p in sorted(glob.glob(os.path.join(src, '*.csv'))) + sorted(glob.glob(os.path.join(src, 'pertie', '*'))):
        q = os.path.join(path, os.path.relpath(p, src))
        n += 1
        if not os.path.exists(q) or sha(p) != sha(q):
            diff.append(os.path.relpath(p, src))
    return not diff, n, f'{n} files compared with {src}: {len(diff)} differ or missing' + (f' e.g. {diff[:3]}' if diff else '')


# ---------------------------------------------------------------------------------------------------------------------
# sub-LST checks of shared classes
# ---------------------------------------------------------------------------------------------------------------------

@content_test('conformant_catalog', name='sublst_table_notes')
def sublst_table_notes(path, **ctx):
    """S-G13 (7042670, D11): every bright-parent table removal in master_catalog_removed.csv names its elevation band and
    the image-pixel metric (old 10min_* tables applied = failure)."""
    if not _is_sublst(path, ctx):
        return _NA
    from exopipe.phase3.lib.catalog.tier_sublst import table_notes
    rem = os.path.join(os.path.dirname(os.path.abspath(path)), 'master_catalog_removed.csv')
    if not os.path.exists(rem):
        return False, None, f'no {rem}'
    line, n, bad = table_notes(rem)
    return bad == 0, bad, line


@content_test('fitted_catalog', name='sublst_fitted_positions')
def sublst_fitted_positions(path, **ctx):
    """F3 on the written rows (sub-LST): pos_kind gaussian|template|peak on every row; ra/dec = pos_ra/pos_dec; the peak
    kept in peak_ra/peak_dec; finite positive statistical errors; the deep systematic blanked (pos_sys_arcsec,
    pos_err_maj/min_arcsec empty: error model A)."""
    if not _is_sublst(path, ctx):
        return _NA
    hdr, rows = _rows(path)
    if 'pos_kind' not in hdr:
        return False, len(rows), 'no pos_kind column: catalogued before fitted positions (70003ad)'
    kinds, bad = {}, []
    blank = [c for c in ('pos_sys_arcsec', 'pos_err_maj_arcsec', 'pos_err_min_arcsec') if c in hdr]
    for r in rows:
        k = r.get('pos_kind')
        kinds[k] = kinds.get(k, 0) + 1
        if k not in ('gaussian', 'template', 'peak') or r['ra'] != r['pos_ra'] or r['dec'] != r['pos_dec'] \
                or not r.get('peak_ra') or not r.get('peak_dec') \
                or not ((_float(r.get('pos_stat_maj_arcsec')) or 0) > 0 and (_float(r.get('pos_stat_min_arcsec')) or 0) > 0) \
                or any(r.get(c) for c in blank):
            bad.append(r['cand_id'])
    return not bad, len(rows), f'{len(rows)} rows {kinds}; {len(bad)} failing F3/blank' + (f' e.g. {bad[:3]}' if bad else '')


@content_test('selection', name='sublst_selection')
def sublst_selection(path, **ctx):
    """selected_all.csv: |SNR| >= catalog_select.SNR_MIN, cand_ids a subset of master_catalog.csv, the split files partition
    it exactly ('MHz' in band = sb, the wideband groups = wb); a split file: every row of its Stokes and group."""
    if not _is_sublst(path, ctx):
        return _NA
    from exopipe.phase3.config import CFG
    snr_min = CFG.catalog_select.SNR_MIN
    split = CFG.catalog_select.SPLIT['sublst']
    d = os.path.dirname(os.path.abspath(path))
    base = os.path.basename(path)
    hdr, rows = _rows(path)
    if base.startswith('selected_') and base != 'selected_all.csv':
        st, tag = base[len('selected_'):-4].split('_')
        bad = [r['cand_id'] for r in rows if r['stokes'] != st
               or not ((split['sb'] in r['band']) if tag == 'sb' else (r['band'] in split['wb']))]
        return not bad, len(rows), f'{base}: {len(rows)} rows, {len(bad)} of another Stokes/group'
    lowsnr = [r['cand_id'] for r in rows if abs(_float(r.get('snr')) or 0) < snr_min]
    mc = {r['cand_id'] for r in _rows(os.path.join(d, 'master_catalog.csv'))[1]}
    extra = [r['cand_id'] for r in rows if r['cand_id'] not in mc]
    parts = []
    for st in ('V', 'I'):
        for tag in ('sb', 'wb'):
            p = os.path.join(d, f'selected_{st}_{tag}.csv')
            if os.path.exists(p):
                parts += [r['cand_id'] for r in _rows(p)[1]]
    want = sorted(r['cand_id'] for r in rows if r['stokes'] in ('V', 'I')
                  and (split['sb'] in r['band'] or r['band'] in split['wb']))
    part_ok = sorted(parts) == want
    pend = sum(1 for r in rows if (r.get('fp_prediction') or 'pending') == 'pending')
    ok = not lowsnr and not extra and part_ok
    return ok, len(rows), (f'{len(rows)} selected: {len(lowsnr)} below |SNR| {snr_min}, {len(extra)} not in '
                           f'master_catalog.csv, split partition {"exact" if part_ok else "WRONG"} ({len(parts)} vs '
                           f'{len(want)}); {pend} rows fp_prediction pending (no verdict file, Q7)')


@content_test('source_catalog', name='sublst_source_catalog')
def sublst_source_catalog(path, **ctx):
    """source_catalogs_<build>_<S>_<sb|wb>/: one Stokes per directory (every member's cand_id of that Stokes); source_id
    unique per band file; every survivor has >= 1 member; survivors = |best_snr| >= 7 or promoted; RA in [0, 360)."""
    d = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    if not _is_sublst(d, ctx):
        return _NA
    name = os.path.basename(d.rstrip('/'))
    parts = name.split('_')
    st = parts[-2] if len(parts) >= 2 and parts[-2] in ('I', 'V') else None
    surv = sorted(glob.glob(os.path.join(d, '*_sources_survivors.csv')))
    if not surv:
        return False, 0, f'{name}: no survivors file'
    n = 0
    bad = []
    for sp in surv:
        band = os.path.basename(sp)[:-len('_sources_survivors.csv')]
        hdr, rows = _rows(sp)
        n += len(rows)
        ids = [r['source_id'] for r in rows]
        if len(ids) != len(set(ids)):
            bad.append(f'{band}: duplicate source_id')
        mp = os.path.join(d, f'{band}_source_members.csv')
        mem = _rows(mp)[1] if os.path.exists(mp) else []
        have = {m['source_id'] for m in mem}
        nom = [i for i in ids if i not in have]
        if nom:
            bad.append(f'{band}: {len(nom)} survivors without members')
        if st and any(not m.get('cand_id', '').startswith(st) for m in mem):
            bad.append(f'{band}: members of another Stokes')
        for r in rows:
            ra = _float(r.get('ra'))
            if ra is None or not (0.0 <= ra < 360.0):
                bad.append(f'{band}: RA {r.get("ra")} of {r["source_id"]}')
                break
            if 'best_snr' in r and 'promotion' in r and abs(_float(r['best_snr']) or 0) < 7.0 and not r['promotion']:
                bad.append(f'{band}: {r["source_id"]} below 7 sigma unpromoted')
                break
    return not bad, n, f'{name}: {len(surv)} band files, {n} survivors; ' + ('; '.join(bad[:4]) if bad else 'all checks pass')


@content_test('localization_table', name='sublst_localizations')
def sublst_localizations(path, **ctx):
    """localizations_<build>/: each cand_id at most once per identity (M12, 43db654); err >= err_raw (sqrt chi2 widening,
    0.05 arcsec rounding); every identity of the offsets has a localization; non-empty when the build has survivors
    (S-G19)."""
    if not _is_sublst(path, ctx):
        return _NA
    d = os.path.dirname(os.path.abspath(path))
    hdr, rows = _rows(path)
    low = [r['identity_id'] for r in rows if (_float(r.get('err_arcsec')) is not None and _float(r.get('err_raw_arcsec'))
                                              is not None and _float(r['err_arcsec']) < _float(r['err_raw_arcsec']) - 0.05)]
    off = os.path.join(d, 'localization_offsets.csv')
    dup, orphan = 0, 0
    if os.path.exists(off):
        seen = set()
        ids = {r['identity_id'] for r in rows}
        for r in _rows(off)[1]:
            k = (r['identity_id'], r['cand_id'])
            dup += k in seen
            seen.add(k)
            orphan += r['identity_id'] not in ids
    empty = ''
    cats, u = ctx.get('cats'), ctx.get('run_dir')
    if cats and u and glob.glob(os.path.join(u, cats, '*_sources_survivors.csv')) and not rows:
        empty = '; survivors exist but no localizations (S-G19)'
    ok = not (low or dup or orphan or empty)
    return ok, len(rows), (f'{len(rows)} identities: {len(low)} with err < err_raw, {dup} repeated (identity, cand_id), '
                           f'{orphan} offsets without a localization{empty}')


@content_test('merged_record', name='sublst_no_excluded_epoch')
def sublst_no_excluded_epoch(path, **ctx):
    """No row of the sub-LST catalogue on an (LST, band, night) data_decisions.csv excludes now (core/exclusions); > 0 rows."""
    if not _is_sublst(path, ctx):
        return _NA
    from exopipe.phase3.core import exclusions
    ex = exclusions.load()
    hdr, rows = _rows(path)
    bad = sorted({(r['lst'], r['band'], r['date'][:10]) for r in rows if ex.is_excluded(r['lst'], r['band'], r['date'][:10])})
    return bool(rows) and not bad, len(rows), (f'{len(rows)} rows, {len(bad)} (LST, band, night) excluded now'
                                               + (f' e.g. {bad[:3]}' if bad else ''))


@content_test('frame_table', name='sublst_frame_times')
def sublst_frame_times(path, **ctx):
    """frame_times.csv: one row per (Stokes, LST, band, window), DATE-OBS before DATE-END on every row, and every key of
    master_catalog.csv present."""
    if os.path.basename(path) != 'frame_times.csv' or not _is_sublst(path, ctx):
        return _NA
    hdr, rows = _rows(path)
    keys = [(r['stokes'], r['lst'], r['band'], r['date']) for r in rows]
    dup = len(keys) - len(set(keys))
    bad = [k for k, r in zip(keys, rows) if not (r.get('date_obs') and r.get('date_end') and r['date_obs'] < r['date_end'])]
    mc = os.path.join(os.path.dirname(os.path.abspath(path)), 'master_catalog.csv')
    miss = []
    if os.path.exists(mc):
        have = set(keys)
        miss = sorted({(r['stokes'], r['lst'], r['band'], r['date']) for r in _rows(mc)[1]} - have)
    ok = not (dup or bad or miss)
    return ok, len(rows), (f'{len(rows)} frames: {dup} duplicate keys, {len(bad)} with DATE-OBS >= DATE-END or blank, '
                           f'{len(miss)} catalogue keys without a frame time' + (f' e.g. {miss[:2]}' if miss else ''))
