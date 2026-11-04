"""A stub catalogue builder carrying the patch points of production's build_source_catalogs_v3.py - the exact texts the
grouping runner replaces (exopipe/phase3/lib/catalog/wraps/builder_grouping.py PATCHES) and its structure census - for the
tests of the runner and of the stages that call it (tests/test_catalog-deep.py, tests/test_catalog_grouping_stage.py). Not a
test file itself (no test_ prefix): imported from the tests directory."""

PATCHABLE_BUILDER = r'''"""STUB catalogue builder WITH the patch points of production's build_source_catalogs_v3.py (the exact texts the grouping
runner replaces, lib/catalog/wraps/builder_grouping.py PATCHES, and its census): a small association (0.1 deg cells per band),
the sigma-stack singleton gate, the multi-integration flag, the repeat rungs, the cross-band identity merge, the 7-8 sigma
gate and the 7-sigma floor - so the runner's passthrough and design-B modes run end to end on synthetic rows."""
import csv, json, math, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _stubcat import rcsv, wcsv

gated_csv, outdir, imgcsv = sys.argv[1:4]
THR = 7.0
RPT_PCT = 99.0
_PERTIE = None
SSTACK_THR = 7.0
SIZE_GATE_7_8 = float(os.environ.get('P3_SIZE_GATE_7_8', '1.7'))   # the census counts this name 6 times (production's
# uses): here, in this line's twin SIZE_GATE_7_8, SIZE_GATE_7_8, SIZE_GATE_7_8 and in the 7-8 sigma gate below
COLS = ['source_id', 'identity_id', 'identity_bands', 'ra', 'dec', 'pos_err_arcsec', 'band', 'lst_cells', 'n_det',
        'multi_integration', 'mi_instants', 'mi_n_cells', 'mi_tiers', 'best_cand_id', 'best_date', 'best_snr', 'promotion',
        'review_dir', 'e_rep']
# census: identity_id x4 (COLS above, the source dict, the merge below twice)

os.makedirs(outdir, exist_ok=True)
rows = list(csv.DictReader(open(gated_csv)))
_in_stokes = {r.get('stokes', '') for r in rows}
SSTACK_GATE = (_in_stokes == {'I'})
print(f'sigma-stack singleton gate: {"ON" if SSTACK_GATE else "off"}', flush=True)


def jname(ra, dec):
    h = ra / 15.0
    return 'J%02d%02d%s%02d%02d' % (int(h), int((h % 1) * 60), '+' if dec >= 0 else '-', int(abs(dec)), int((abs(dec) % 1) * 60))


def _stat_as(r):
    return 30.0


def _sys_det(r):
    return 52.0


def _KP(band):
    return 3.034


def assoc_groups(coords, stat_as, sys_as, K=None):
    g = defaultdict(list)
    for i, (ra, dec) in enumerate(coords):
        g[(round(ra, 1), round(dec, 1))].append(i)
    return list(g.values())


def _sstack_fails(m):
    try:
        ss = float(m.get('sstack') or '')
    except (TypeError, ValueError):
        return False
    return abs(float(m['_snr'])) < SSTACK_THR * max(ss, 1.0)


def _multi_integration(mem, cells, dec, sig_stat):
    inst = sorted({(m.get('lst', ''), m.get('date', '')) for m in mem})
    return dict(multi_integration=int(len(inst) >= 2), mi_instants=len(inst), mi_n_cells=len({c for c, _ in inst}),
                mi_tiers='deep' if len(inst) >= 2 else '')


for r in rows:
    r['_snr'], r['_ra'], r['_dec'] = float(r['snr']), float(r['ra']), float(r['dec'])
byband = defaultdict(list)
for r in rows:
    byband[r['band']].append(r)
allout, allmem = {}, {}
for band in sorted(byband):
    br = byband[band]
    c_all = [(r['_ra'], r['_dec']) for r in br]
    groups = assoc_groups(c_all, [_stat_as(r) for r in br],
                          [_sys_det(r) for r in br], K=_KP(band))
    out, members, ss_removed = [], [], []
    for g in groups:
        mem = [br[i] for i in g]
        if SSTACK_GATE and len({(m['lst'], m['date']) for m in mem}) == 1:
            keep = [m for m in mem if not _sstack_fails(m)]
            ss_removed += [m for m in mem if m not in keep]
            if not keep:
                continue
            mem = keep
        ra = sum(m['_ra'] for m in mem) / len(mem)
        dec = sum(m['_dec'] for m in mem) / len(mem)
        best = max(mem, key=lambda m: abs(m['_snr']))
        cells = sorted({m['lst'] for m in mem})
        sig_stat = 30.0 / math.sqrt(len(mem))
        members.append((jname(ra, dec), band,
                        [(m['cand_id'], m['_ra'], m['_dec'], m['_snr'], m['date']) for m in mem]))
        out.append(dict(
            source_id=jname(ra, dec), identity_id='', identity_bands='',
            ra=round(ra, 5), dec=round(dec, 5), pos_err_arcsec=60.0, band=band, lst_cells='+'.join(cells), n_det=len(mem),
            **_multi_integration(mem, cells, dec, sig_stat),
            best_cand_id=best['cand_id'], best_date=best['date'], best_snr=round(best['_snr'], 2), promotion='',
            review_dir='', _rpt_pct=(50.0 if len(mem) > 1 else ''), e_rep=('0.1' if len(mem) > 1 else ''), _matches=[],
            xm_E_at_snr=''))
    allout[band] = out
    allmem[band] = members
    if SSTACK_GATE:
        wcsv(f'{outdir}/{band}_sigma_stack_singleton_removed.csv', ['cand_id', 'lst', 'date', 'band', 'snr', 'sstack', 'thr'],
             [dict(cand_id=m['cand_id'], lst=m['lst'], date=m['date'], band=band, snr=m['_snr'], sstack=m.get('sstack', ''),
                   thr='') for m in ss_removed])
    with open(f'{outdir}/{band}_source_members.csv', 'w', newline='') as f:
        w_ = csv.writer(f)
        w_.writerow(['source_id', 'band', 'cand_id', 'det_ra', 'det_dec', 'det_snr', 'det_date'])
        for sid, b_, dets in members:
            for cid, dra, ddec, dsnr, ddate in dets:
                w_.writerow([sid, b_, cid, round(dra, 5), round(ddec, 5), round(dsnr, 2), ddate])
for band in allout:
    for s in allout[band]:
        a = abs(s['best_snr'])
        lab = []
        if 6.0 <= a < THR:
            if s['n_det'] >= 2 and s['_rpt_pct'] != '' \
                    and float(s['_rpt_pct']) < RPT_PCT:
                lab.append('RPT')
        try:
            if s['n_det'] >= 2 and s.get('e_rep') not in ('', None) and float(s['e_rep']) < 0.5: lab.append('REP')
            if s['n_det'] >= 2 and s['_matches'] and s.get('xm_E_at_snr') not in ('', None) and float(s['xm_E_at_snr']) < 0.5 \
                    and (s['n_det'] >= 3 or a >= THR): lab.append('XREP')
        except (TypeError, ValueError): pass
        s['promotion'] = '+'.join(lab)
pool = [s for band in allout for s in allout[band]]
if pool:
    _xf = _PERTIE is not None
    g = defaultdict(list)
    for s in pool:
        g[(round(s['ra'], 1), round(s['dec'], 1))].append(s)
    for grp in g.values():
        best = max(grp, key=lambda s: abs(s['best_snr']))
        for s in grp:
            s['identity_id'] = best['source_id']
            s['identity_bands'] = '+'.join(sorted({x['band'] for x in grp}))
    print('identity merge: done', flush=True)
for band in list(allout):
    keep = []
    for s in allout[band]:
        a_ = abs(float(s['best_snr']))
        if s['n_det'] == 1 and not s['promotion'] and not s.get('xm_best') and 7.0 <= a_ < 8.0:
            if SIZE_GATE_7_8 < 1.0:
                continue
        keep.append(s)
    allout[band] = keep
for band, out in allout.items():
    sv = sorted((s for s in out
                 if abs(s['best_snr']) >= THR or s['promotion']),
                key=lambda s: -abs(s['best_snr']))
    for i, s in enumerate(sv):
        s['rank'] = f'rank{i + 1:04d}'
        s['review_dir'] = f"candidates/{band}/{s['rank']}_{band}_{s['source_id']}"
    with open(f'{outdir}/{band}_sources_survivors.csv', 'w', newline='') as f:
        w_ = csv.DictWriter(f, fieldnames=['rank'] + COLS, extrasaction='ignore')
        w_.writeheader()
        for s in sv:
            w_.writerow(s)
    with open(f'{outdir}/{band}_sources_master.csv', 'w', newline='') as f:
        w_ = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
        w_.writeheader()
        for s in sorted(out, key=lambda s: -abs(s['best_snr'])):
            w_.writerow(s)
json.dump({}, open(f'{outdir}/E_params.json', 'w'))
print('DONE', flush=True)
'''
