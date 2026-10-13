"""Content tests of the snapshot batch chain (catalog --tier snapshot), registered with core/validate.

Classes: snapshot_run (a chain run directory: R1-R5 and R7 of ops/snapshot_rechain_content_test.py recomputed from the run's
own files, plus no refusal marker and the per-Stokes views agreeing with the union catalogues), tie_cache (one cell's tie
cache file), snapshot_tie_summary (the chain_summary.json a per-cell --stop-after-tie writes). The run content test's
operating points are config (catalog_snapshot.TEST_GATE3D_PCT, TEST_MEMBER_ORDER, GHOST_DATES). R6 (missing panels) is the
pages step's class snapshot_page_run. Every test reads content and returns (passed, value, detail).
"""
import csv
import glob
import json
import math
import os

from exopipe.phase3.core.validate import content_test

csv.field_size_limit(10 ** 9)


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def run_checks(run):
    """R1-R5, R7 and the extras on a run directory: [(name, passed, detail)] (the order of the production test)."""
    c = _cfg().catalog_snapshot
    S = json.load(open(os.path.join(run, 'chain_summary.json')))
    out = []
    pf = S.get('prefit') or {}
    r1 = (pf.get('n_without_fit'), S.get('n_served_with_unfitted_frames'), S.get('n_unfitted_frames_under_served'))
    out.append(('R1', r1 == (0, 0, 0), 'unfitted at the fit step %s of %s; served with unfitted frames %s; unfitted frames '
                'under served %s' % (r1[0], pf.get('n_at_per_frame_fit_step'), r1[1], r1[2])))
    dfiles = sorted(glob.glob(os.path.join(run, 'candidates', '*', '*', 'detections.csv')))
    n_rows = n_bad = n_nocol = 0
    served = set()
    for p in dfiles:
        with open(p, newline='') as f:
            rd = csv.DictReader(f)
            cols = rd.fieldnames or []
            if 'pos_err_stat_arcsec' not in cols or 'pos_err_arcsec' not in cols:
                n_nocol += 1
                continue
            for r in rd:
                n_rows += 1
                served.add(r.get('cand_id', ''))
                a, b = _f(r.get('pos_err_stat_arcsec')), _f(r.get('pos_err_arcsec'))
                if a is None or b is None or a <= 0 or b <= 0:
                    n_bad += 1
    out.append(('R2', bool(dfiles) and n_nocol == 0 and n_bad == 0, '%d detections.csv, %d rows: %d files without the own-'
                'error columns, %d rows without a finite positive error' % (len(dfiles), n_rows, n_nocol, n_bad)))
    gp = os.path.join(run, 'ghost_removed.csv')
    grows = list(csv.DictReader(open(gp))) if os.path.exists(gp) else None
    n_g = (S.get('domain_dropped') or {}).get('ghost')
    has_g = any(x.split('_', 1)[1] in tuple(c.GHOST_DATES) for x in S.get('cells', []))
    leak = sorted({r['cand_id'] for r in (grows or [])} & served)
    out.append(('R3', grows is not None and len(grows) == n_g and (len(grows) > 0 or not has_g) and not leak,
                'ghost_removed.csv rows %s, summary %s, ghost-date cells %s, removed cand_ids served %d' % (
                    None if grows is None else len(grows), n_g, has_g, len(leak))))
    tc = S.get('tie_cache') or {}
    out.append(('R4', tc.get('tied_now') == 0, 'tie cache: %s from the cache, %s tied in this run' % (
        tc.get('from_cache'), tc.get('tied_now'))))
    dd = S.get('domain_dropped') or {}
    r5 = (S.get('gate3d_match_percentile'), S.get('member_order'), (S.get('ghost_gate') or {}).get('dates'),
          any('moon' in k.lower() for k in dd))
    out.append(('R5', r5 == (float(c.TEST_GATE3D_PCT), c.TEST_MEMBER_ORDER, list(c.GHOST_DATES), False),
                '3-D gate percentile %s; member order %s; ghost dates %s; Moon rule %s' % r5))
    n_ra = n_out = 0
    for p in sorted(glob.glob(os.path.join(run, 'source_catalogs*', '*.csv'))) + dfiles:
        with open(p, newline='') as f:
            rd = csv.DictReader(f)
            cols = [x for x in (rd.fieldnames or []) if x == 'ra' or x.startswith('ra_') or x.endswith('_ra')]
            for r in rd if cols else ():
                for x in cols:
                    v = (r.get(x) or '').strip()
                    if not v:
                        continue
                    n_ra += 1
                    fv = _f(v)
                    if fv is None or not (0.0 <= fv < 360.0):
                        n_out += 1
    out.append(('R7', n_ra > 0 and n_out == 0, '%d RA values, %d outside [0, 360)' % (n_ra, n_out)))
    marks = [m for m in ('CHECKSOURCE_MISSING.txt', 'FIT_INCOMPLETE.txt') if os.path.exists(os.path.join(run, m))]
    out.append(('markers', not marks, 'refusal markers present: %s' % (marks or 'none')))
    bad_views = []
    union = {}
    for p in glob.glob(os.path.join(run, 'source_catalogs', '*_sources_survivors.csv')):
        for r in csv.DictReader(open(p)):
            union.setdefault(r.get('stokes', ''), set()).add(r['source_id'])
    for st in ('V', 'I'):
        view = set()
        for p in glob.glob(os.path.join(run, 'source_catalogs_%s' % st, '*_sources_survivors.csv')):
            # a view row is one (identity, band): source_id '<identity>_<band>', identity_id = the union row's source_id
            view |= {(r.get('identity_id') or '').strip() or r['source_id'] for r in csv.DictReader(open(p))}
        # an identity with detections in both Stokes has ONE union row (the Stokes of its best detection) and rows in both
        # views: so every union row of Stokes S is in the S view, and every view identity is a union source (measured on
        # Run_SNAPB1R: 139 of 396 V-view identities have an I union row)
        allu = set().union(*union.values()) if union else set()
        miss, orphan = union.get(st, set()) - view, view - allu
        if miss or orphan:
            bad_views.append('%s: %d union rows not in the view, %d view identities not in the union' % (st, len(miss),
                                                                                                        len(orphan)))
    out.append(('views', not bad_views, 'per-Stokes views cover the union rows of their Stokes and nothing else: %s' % (
        '; '.join(bad_views) or 'yes')))
    return out


@content_test('snapshot_run')
def snapshot_run_content(path, **ctx):
    if not os.path.exists(os.path.join(path, 'chain_summary.json')):
        return False, None, 'no chain_summary.json'
    res = run_checks(path)
    bad = [n for n, ok, _ in res if not ok]
    return not bad, len(bad), '; '.join('%s %s %s' % (n, 'PASS' if ok else 'FAIL', d) for n, ok, d in res)[:1500]


@content_test('tie_cache')
def tie_cache_rows(path, errmodel=None, **ctx):
    """tie_<cell>.csv: the cache columns, every row of one cell, finite tied positions; rows of the installed model (ctx
    errmodel, from the preflight record) present; repeated keys counted (the module keeps the last)."""
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        rows = list(rd)
    need = ('cell', 'sb', 'st', 'tier', 'ts', 'ra_pre', 'dec_pre', 'tie_status', 'ra', 'dec', 'errmodel')
    miss = [x for x in need if x not in cols]
    if miss:
        return False, len(rows), 'missing columns %s' % miss
    cell = os.path.basename(path)[4:-4]
    other = sum(1 for r in rows if r['cell'] != cell)
    keys = [(r['sb'], r['st'], r['tier'], r['ts'], r['ra_pre'], r['dec_pre'], r['errmodel']) for r in rows]
    dup = len(keys) - len(set(keys))
    nonfin = sum(1 for r in rows if _f(r['ra']) is None or _f(r['dec']) is None)
    models = {}
    for r in rows:
        models[r['errmodel']] = models.get(r['errmodel'], 0) + 1
    cur = '' if errmodel is None else ', %d rows of the installed model %s' % (models.get(errmodel, 0), errmodel)
    # a repeated key is a re-tie appended later: phase3_snapshot_tiecache.apply_cached keeps the LAST row (counted only)
    ok = not (other or nonfin) and (errmodel is None or models.get(errmodel, 0) > 0)
    return ok, len(rows), '%s: %d rows %s; %d of another cell, %d repeated keys (last wins), %d non-finite positions%s' % (
        cell, len(rows), models, other, dup, nonfin, cur)


@content_test('snapshot_tie_summary')
def tie_summary(path, **ctx):
    """chain_summary.json of a --stop-after-tie run: every detection at the per-frame fit step fitted (prefit.n_without_fit
    = 0, the precondition of the tie pass) and a tie_cache record with integer counts."""
    S = json.load(open(path))
    pf, tc = S.get('prefit') or {}, S.get('tie_cache') or {}
    ok = pf.get('n_without_fit') == 0 and isinstance(tc.get('from_cache'), int) and isinstance(tc.get('tied_now'), int)
    return ok, tc.get('tied_now'), 'prefit without fit %s of %s; tie cache %s from cache, %s tied now' % (
        pf.get('n_without_fit'), pf.get('n_at_per_frame_fit_step'), tc.get('from_cache'), tc.get('tied_now'))
