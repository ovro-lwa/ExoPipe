"""A SYNTHETIC SNAPSHOT CHAIN RECORD for the tests of the combined view and the deep omission (tests/test_snapshot_view.py,
tests/test_deep_omission.py).

Since exopipe_ops dbc40a6 production's combined view RE-JUDGES every batch's served sources on the detections they keep
(snapshot_rejudge.rejudge; the port lib/catalog/rejudge.py) and refuses unless its reconstruction from the batch's per-detection
record reproduces the chain on every served source (mi_tiers, catalog promotion, routes, best member) and the batch's
served_tiers.csv. A synthetic batch therefore needs a record the chain's own functions agree with: write_chain() builds it from
the synthetic identities and their detections and computes the chain's fields WITH THE CHAIN'S FUNCTIONS (clv15
phase3_snapshot_chain route_repeats / reflection_band_source_gate / catalog_route / singleton_floor_for and
phase3_snapshot_tierserve.member, in place, under production's interpreter) - so production's re-judging and the port's both
self-check against what the chain would have written. It writes, into the batch run:
  detection_record.csv.gz   one row per served detection (TS.NEED + assoc0_cluster, gate_3d, cand_id, elev_bin; every row in the
                            pool, fitted, unresolved-constrained, outside the reflection band; run_status complete; elev_bin
                            the detection's own or 2 - production's 1-minute elevation floor, b114ff3, re-measures elev_bin -1
                            rows)
  chain_summary.json        {'cells': [...]} (the chain's floors are its defaults; no footprint / pool-size checks)
  served_tiers.csv          per (identity, tier with members): chain_outcome served, qualifies by the chain's per-tier rule
and returns {(identity, tier): qualifies} - the batch's per-tier catalogues hold exactly the qualifying ones (G4b).
"""
import json
import os
import subprocess

PY36 = '/opt/devel/pipeline/envs/development/bin/python'
CLV15 = '/lustre/gh/main/clv15'

CHILD = r'''
import csv, gzip, json, os, sys
spec = json.load(open(sys.argv[1]))
sys.path.insert(0, spec['clv15'])
import phase3_snapshot_chain as C
import phase3_snapshot_tierserve as TS
from phase3_snapshot_search import ts_to_epoch_seconds
csv.field_size_limit(10 ** 9)


class A(object):
    singleton_floor, singleton_floor_v = C.SINGLETON_FLOOR, C.SINGLETON_FLOOR_V
    repeat_floor, assoc_floor, store_floor, seed_floor = C.REPEAT_FLOOR, C.ASSOC_FLOOR, C.STORE_FLOOR, C.SEED_FLOOR
args = A()
seed_on = args.seed_floor > args.store_floor
rows, by_src = [], []
for i, (J, dets) in enumerate(spec['sources']):
    rr = []
    for j, d in enumerate(dets):
        r = dict(det_uid='%s:%d:%d' % (spec['tag'], i, j), cell='%s_%s' % (d['lst'], d['date']), subband=d['subband'], band=d['band'],
                 stokes=d['stokes'], tier=d['tier'], ts=d['ts'], instant_key=d['ts'] if d['tier'] == '10s' else d['ts'][:13],
                 instant_mid_unix='%.3f' % ts_to_epoch_seconds(d['ts']), snr=d['snr'], ra=d['ra'], dec=d['dec'],
                 ra_peak=d.get('ra_peak') or d['ra'], dec_peak=d.get('dec_peak') or d['dec'], elevation_deg='60.0', pos_source='fit',
                 stat_arcmin='1.0', sys_arcmin='1.0', sys_source='per-case', in_refl_band='0', refl_below_floor='0',
                 fit_verdict='unresolved-constrained', store_fit_verdict='', in_assoc_pool='1', first_removed_by='',
                 src_index=str(i), src_best_det_uid='', src_n_members=str(len(dets)), src_mi_tiers='', catalog_promoted='', routes='',
                 src_outcome='served', identity_jname=J, served='1', run_status='complete', assoc0_cluster='', gate_3d='',
                 cand_id=d.get('cand_id', ''), elev_bin=d.get('elev_bin', '2'))
        rr.append(r)
    rows += rr
    by_src.append((J, rr))
C.CELLS_GLOBAL.clear()
C.CELLS_GLOBAL.update({c: C.cell_centre(c) for c in spec['cells']})
cats = C.load_samples(spec['clv15'])
area = C.footprint_area_deg2(C.CELLS_GLOBAL)
pool = [dict(st=r['stokes'], tier=r['tier'], band=r['band'], snr=float(r['snr']), det_uid=r['det_uid']) for r in rows]


def prepare(mem, full):          # snapshot_rejudge / phase3_snapshot_tierserve prepare (the chain's best: route_repeats' own)
    best = max(mem, key=lambda m: abs(m['snr']))
    if not full and seed_on and not (len(mem) == 1 and abs(mem[0]['snr']) < args.assoc_floor) and abs(best['snr']) < args.seed_floor:
        return dict(qualifies=0, reason='seed', routes=[], best=best, ns=None)
    ns = C.route_repeats([mem], args.repeat_floor)[0]
    kept, _ = C.reflection_band_source_gate([ns], args.repeat_floor)
    if not kept:
        return dict(qualifies=0, reason='refl-floor', routes=[], best=ns['best'], ns=None, ns0=ns)
    return dict(ns=kept[0], best=kept[0]['best'])


def finish(res):
    if res['ns'] is None:
        return res
    ns, b = res['ns'], res['ns']['best']
    routes = (['multi-integration'] if ns['multi_integration'] else []) + list(ns.get('catalog', []))
    if abs(b['snr']) >= C.singleton_floor_for(b, args) and b['verdict'] == 'unresolved-constrained':
        routes.append('singleton')
    res.update(qualifies=int(bool(routes)), routes=routes, best=b)
    return res
full, tier = {}, {}
for J, rr in by_src:
    mem = [TS.member(r) for r in rr]
    full[J] = prepare(mem, True)
    for t in TS.TIERS:
        mt = [m for m in mem if m['tier'] == t]
        if mt:
            tier[(J, t)] = prepare(mt, False)
C.catalog_route([x['ns'] for x in full.values() if x['ns'] is not None] + [x['ns'] for x in tier.values() if x['ns'] is not None],
                pool, cats, area)
for J, rr in by_src:
    f = finish(full[J])
    near = [abs(abs(float(r['snr'])) - fl) < TS.SNR_ROUND for r in rr for fl in (args.singleton_floor, args.singleton_floor_v)]
    if any(near):
        sys.exit('FIXTURE: %s has a member within the record rounding of a singleton floor - choose another S/N' % J)
    for r in rr:
        r.update(src_best_det_uid=f['best']['det_uid'], src_mi_tiers=(f['ns'] or f.get('ns0') or {}).get('mi_tiers') or '',
                 catalog_promoted=';'.join((f['ns'] or {}).get('catalog', [])), routes=';'.join(f['routes']))
cols = list(TS.NEED) + ['assoc0_cluster', 'gate_3d'] + [c for c in ('cand_id', 'elev_bin') if c not in TS.NEED]
with gzip.open(os.path.join(spec['run'], 'detection_record.csv.gz'), 'wt', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
out = {}
with open(os.path.join(spec['run'], 'served_tiers.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['identity_jname', 'chain_outcome', 'tier', 'qualifies', 'routes'])
    for (J, t), x in sorted(tier.items()):
        x = finish(x)
        w.writerow([J, 'served', t, x['qualifies'], ';'.join(x.get('routes') or [])])
        out['%s|%s' % (J, t)] = x['qualifies']
json.dump(dict(cells=spec['cells']), open(os.path.join(spec['run'], 'chain_summary.json'), 'w'))
json.dump(out, open(spec['result'], 'w'))
'''


def write_chain(run, sources, cells, tag=None, py36=PY36, clv15=CLV15):
    """The chain record, summary and served_tiers.csv of a synthetic batch run (module docstring). sources: [(identity, [detection
    dicts with lst, date, subband, band, stokes, tier, ts, snr, ra, dec, ra_peak, dec_peak, cand_id; elev_bin optional])], in the
    record's order; cells: the cells the batch searched. Returns {(identity, tier): qualifies}."""
    os.makedirs(run, exist_ok=True)
    spec_p, res_p, child_p = (os.path.join(run, '.chain_fixture.%s' % x) for x in ('spec.json', 'result.json', 'child.py'))
    with open(child_p, 'w') as f:
        f.write(CHILD)
    with open(spec_p, 'w') as f:
        json.dump(dict(run=run, tag=tag or os.path.basename(run), sources=sources, cells=sorted(cells), clv15=clv15,
                       result=res_p), f)
    r = subprocess.run([py36, child_p, spec_p], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
                       timeout=900, cwd=run)
    for p in (spec_p, child_p):
        os.remove(p)
    if r.returncode:
        raise RuntimeError('snapshot_chain_fixture: %s' % r.stdout[-3000:])
    with open(res_p) as f:
        out = json.load(f)
    os.remove(res_p)
    return {tuple(k.split('|')): v for k, v in out.items()}
