"""Tests of `exopipe3 flow` (unit flow): plain asserts, no pytest, synthetic data only, under 2 minutes.

    NUMPY_MADVISE_HUGEPAGE=0 EXOPIPE3_SITE=/lustre/gh/main/exopipe_ops/exopipe3_site.py \
        /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_flow.py

Everything is written under a fresh /tmp/exopipe3_tests_flow/ (removed at the end). Production files are only READ (the
exclusion list, the ledger, the membership-veto pin). The GPU children are replaced by stand-in scripts that write what the
in-place modules write (the step's orchestration, gates, set comparisons and content tests are what is tested); the deep
prep (process_lst_freq) is not run here - it is production code called in place (parity item P1).
"""
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

ROOT = '/tmp/exopipe3_tests_flow'
PY = sys.executable

import numpy as np                                   # noqa: E402
from astropy.io import fits                          # noqa: E402

from exopipe.phase3 import cli                       # noqa: E402
from exopipe.phase3.config import CFG, ConfigRefusal  # noqa: E402
from exopipe.phase3.core import env, record          # noqa: E402

env.legacy()
from exopipe.phase3.lib.flow import steps_common as C   # noqa: E402
from exopipe.phase3.lib.flow import tests as T          # noqa: E402
from exopipe.phase3.lib.flow import steps_snapshot as SS, steps_sublst as SL  # noqa: E402

N = 64          # synthetic image size (px)


def hdr(**kw):
    h = fits.Header()
    h['NAXIS'] = 2
    for k, v in dict(CTYPE1='RA---SIN', CTYPE2='DEC--SIN', CRVAL1=100.0, CRVAL2=37.2, CRPIX1=N // 2 + 1,
                     CRPIX2=N // 2 + 1, CDELT1=-0.5, CDELT2=0.5, BMAJ=1.0, BMIN=1.0, BPA=0.0).items():
        h[k] = v
    for k, v in kw.items():
        h[k] = v
    return h


def wfits(path, data, **kw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.writeto(path, np.asarray(data, dtype=np.float32), hdr(**kw), overwrite=True)
    return path


def run_cli(argv):
    CFG.reset()
    t0 = time.time()
    rc = cli.main(argv + ['--cores', 'none'])
    print(f'   exopipe3 {" ".join(argv[:4])} ... -> exit {rc} ({time.time() - t0:.1f}s)')
    return rc


# =====================================================================================================================
def test_config_and_stages():
    from exopipe.phase3.steps import flow as F
    secs = CFG.sections()
    for s in ('flow', 'flow_prep', 'flow_snapshot', 'flow_refine'):
        assert s in secs, s
    assert 'flow_reference' not in secs and 'flow_movie' not in secs    # masters / frames own them (_CROSSCHECK)
    assert CFG.flow.DEFAULT_STAGES['snapshot'] == ('flow', 'ship')
    for key in ('flow_snapshot.MASTERS_ROOT', 'flow_refine.WEIGHTS_CSV', 'flow_refine.APPROVED_REFUSALS_CSV'):
        try:
            CFG.set(key, '/tmp/x')
            raise AssertionError(f'{key} should be protected')
        except ConfigRefusal:
            pass
    CFG.reset()
    A = type('A', (), {})
    a = A()
    a.tier, a.stages, a.from_stage, a.until_stage = 'deep', None, None, None
    assert F.resolve_stages(a) == ['stage-in', 'prep', 'solve', 'check']
    a.stages, a.until_stage = None, 'prep'
    assert F.resolve_stages(a) == ['stage-in', 'prep']
    a.stages, a.until_stage, a.from_stage = 'check,solve', None, None
    assert F.resolve_stages(a) == ['solve', 'check']
    a.stages = 'reference'
    try:
        F.resolve_stages(a)
        raise AssertionError('reference is a masters stage now')
    except ValueError:
        pass
    print('ok config + stages')


def test_items():
    def ns(**kw):
        d = dict(cells=None, epochs=None, lsts=None, nights=None, scopes=None, exclude_cells=None, snap_stage_root=None,
                 work='/tmp')
        d.update(kw)
        return type('A', (), d)()
    it = SS.items(ns(epochs='18h_2025-04-11', scopes='36MHz,41MHz'))
    assert it == [('18h_2025-04-11', ['36MHz', '41MHz'], True)], it
    it = SS.items(ns(cells='18h_2025-04-11_36MHz'))
    assert it == [('18h_2025-04-11', ['36MHz'], True)], it
    try:
        SS.items(ns(cells='18h_2025-04-11_Red'))
        raise AssertionError('a band night cell must refuse for flows')
    except ValueError:
        pass
    it = SL.items(ns(cells='09h_55MHz'))
    assert it == [('09h', '55MHz', None)], it
    it = SL.items(ns(epochs='09h_2025-01-03', scopes='18MHz'))
    assert it == [('09h', '18MHz', ['2025-01-03'])], it
    assert len(SL.items(ns(lsts='09h'))) == 15
    try:
        SL.items(ns(cells='09h_Red'))
        raise AssertionError('band scope must refuse for the x2 refine')
    except record.Refusal:
        pass
    assert SL.unit_name('refine', '09h', '55MHz', ['2025-01-03']) == 'refine_09h_55MHz_2025-01-03'
    print('ok items')


# =====================================================================================================================
# snapshot: the per-subband SET comparison (D12, cf72d51) and the whole flow + ship orchestration
# =====================================================================================================================

def make_snapshot_cell(base, cell, sbs, nts=4, zero_vis_v=None, blank_i=None):
    """Staged frames <base>/stage/<cell>/<sb>/ and Masters <base>/masters/<lst>_<sb>/coadd_final/. Returns stage root."""
    lst = cell.split('_')[0]
    stage = os.path.join(base, 'stage')
    for sb in sbs:
        wfits(os.path.join(base, 'masters', f'{lst}_{sb}', 'coadd_final', f'I_deep_Taper_R0_{sb}.fits'),
              np.random.default_rng(1).normal(size=(N, N)))
        lines = []
        for k in range(nts):
            ts = f'20300101_1200{10 * k:02d}'
            di = np.random.default_rng(k).normal(size=(N, N))
            if blank_i and (sb, k) in blank_i:
                di[:] = np.nan
            pi = wfits(os.path.join(stage, cell, sb, f'{sb}-Clean-Snapshot-{ts}-image.fits'), di, WSCNVIS=1000.0)
            zv = 0.0 if zero_vis_v and (sb, k) in zero_vis_v else 900.0
            pv = wfits(os.path.join(stage, cell, sb, f'{sb}-Pilot-Snapshot-{ts}-image.fits'),
                       np.random.default_rng(100 + k).normal(size=(N, N)), WSCNVIS=zv)
            lines += [f'/phase1/x/{os.path.basename(pi)}.fz', f'/phase1/x/{os.path.basename(pv)}.fz']
        with open(os.path.join(stage, cell, sb, '.filelist'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
    return stage


FAKE_FLOW = r'''
import argparse, csv, glob, os, re, sys
import numpy as np
from astropy.io import fits
ap = argparse.ArgumentParser(); ap.add_argument('--stage'); ap.add_argument('--reference'); ap.add_argument('--subband')
ap.add_argument('--out'); ap.add_argument('--frames'); ap.add_argument('--overwrite', action='store_true')
a = ap.parse_args()
skip = os.environ.get('FAKE_SKIP_V', '')
PAT = re.compile(r"(?P<sb>\d+MHz)-(?P<kind>Clean|Pilot)-Snapshot-(?P<ts>\d{8}_\d{6})-image\.fits$")
files = [l.strip() for l in open(a.frames)] if a.frames else sorted(glob.glob(os.path.join(a.stage, '*Clean-Snapshot-*-image.fits')))
os.makedirs(a.out, exist_ok=True)
rows = []
for f in files:
    ts = PAT.search(os.path.basename(f)).group('ts')
    d = fits.getdata(f).astype(np.float32); h = fits.getheader(f)
    fv = f.replace('-Clean-', '-Pilot-')
    fin = np.isfinite(d)
    if fin.mean() < 0.5:
        h['FLOWED'] = (False, 'blank frame written through; no flow'); fits.writeto(f[:-5] + '_flowed.fits', d, h, overwrite=True)
        hv = fits.getheader(fv); hv['FLOWED'] = (True, 'flow of the nearest valid I instant x'); hv['FLOWREF'] = os.path.basename(a.reference)[:60]
        fits.writeto(fv[:-5] + '_flowed.fits', fits.getdata(fv), hv, overwrite=True)
        rows.append([ts, 'blank-I; V flowed from nearest valid instant', '', '', '', '', '', '0.01']); continue
    h['FLOWREF'] = os.path.basename(a.reference)[:60]; h['FLOWED'] = (True, 'x'); h['FLOWBLK'] = (0, 'x')
    fits.writeto(f[:-5] + '_flowed.fits', d, h, overwrite=True)
    if ts != skip:
        hv = fits.getheader(fv); hv['FLOWREF'] = os.path.basename(a.reference)[:60]; hv['FLOWED'] = (True, 'x')
        fits.writeto(fv[:-5] + '_flowed.fits', fits.getdata(fv), hv, overwrite=True)
    off = np.full(d.shape + (2,), 0.25, np.float32)
    np.save(os.path.join(a.out, 'flow_%s_%s.npy' % (a.subband, ts)), off[::8, ::8].astype(np.float16))
    rows.append([ts, 'ok', '0.354', '0.354', '0.354', '0.2500', '0.2500', '0.05'])
with open(os.path.join(a.out, 'flow_%s.csv' % a.subband), 'a') as fh:
    w = csv.writer(fh)
    if fh.tell() == 0:
        w.writerow(['ts', 'state', 'median_flow_px', 'p90_flow_px', 'max_flow_px', 'mean_dx_px', 'mean_dy_px', 'seconds'])
    w.writerows(rows)
print('done %s: %d frames' % (a.subband, len(files)))
'''


def test_set_comparison_d12():
    """The D12 / cf72d51 scenario: a count over the cell passes (an extra flowed zero-vis frame in one subband hides a
    missing V frame in another); the per-subband SET comparison refuses the subband that is incomplete."""
    base = os.path.join(ROOT, 'd12')
    stage = make_snapshot_cell(base, '13h_2030-01-01', ['36MHz', '41MHz'], zero_vis_v={('41MHz', 1)})
    sa, sb = os.path.join(stage, '13h_2030-01-01', '36MHz'), os.path.join(stage, '13h_2030-01-01', '41MHz')
    for p in sorted(os.listdir(sa)):                      # 36 MHz: one V frame never flowed
        if p.endswith('-image.fits') and '120010' not in p or ('Clean' in p and p.endswith('-image.fits')):
            shutil.copy(os.path.join(sa, p), os.path.join(sa, p[:-5] + '_flowed.fits'))
    for p in sorted(os.listdir(sb)):                      # 41 MHz: everything flowed, the zero-vis V frame written through too
        if p.endswith('-image.fits'):
            shutil.copy(os.path.join(sb, p), os.path.join(sb, p[:-5] + '_flowed.fits'))
    staged = sum(1 for d in (sa, sb) for p in os.listdir(d) if C.is_staged_frame(p))
    flowed = sum(1 for d in (sa, sb) for p in os.listdir(d) if p.endswith('_flowed.fits'))
    nzv = 1
    assert flowed >= staged - nzv, (flowed, staged)       # the old count-based gate (expected = staged - zv) passes
    ok_a, n_a, miss_a, _ = T.set_complete_flowed(sa)
    ok_b, n_b, miss_b, info_b = T.set_complete_flowed(sb)
    assert not ok_a and miss_a == ['36MHz-Pilot-Snapshot-20300101_120010-image.fits'], miss_a
    assert ok_b and info_b['zero_vis'] == 1 and n_b == 7, info_b
    r = {x['name']: x for x in __import__('exopipe.phase3.core.validate', fromlist=['run']).run('snapshot_flowed_frame', sa)}
    assert not r['snapshot_flowed_frame.set_complete']['passed']
    print('ok set comparison (D12): count passes, per-subband set refuses 36MHz')


def test_snapshot_flow_and_ship():
    base = os.path.join(ROOT, 'snap')
    cell = '18h_2030-01-02'
    stage = make_snapshot_cell(base, cell, ['36MHz', '41MHz'], zero_vis_v={('36MHz', 2)}, blank_i={('41MHz', 3)})
    rec = os.path.join(base, 'stage_record.json')
    json.dump({'status': 'done', 'unit': f'{cell}_stage', 'meta': {'stage_root': stage}}, open(rec, 'w'))
    fake = os.path.join(base, 'fake_flow.py')
    open(fake, 'w').write(FAKE_FLOW)
    out, work = os.path.join(base, 'out'), os.path.join(base, 'work')
    SS.FLOW_SCRIPT = ('clv15', fake)
    saved = (CFG.flow_snapshot.MASTERS_ROOT, dict(CFG.site.PYTHON))
    CFG.flow_snapshot.MASTERS_ROOT = os.path.join(base, 'masters')
    CFG.site.PYTHON = dict(CFG.site.PYTHON, flow=PY)
    try:
        argv = ['flow', '--tier', 'snapshot', '--cells', cell, '--gpus', '0,1', '--work', work, '--out', out,
                '--snap-stage-record', rec]
        rc = run_cli(argv)
        assert rc == 0, rc
        for sb in ('36MHz', '41MHz'):
            r = record.load_record(out, 'flow', f'flow_{cell}_{sb}')
            assert r['status'] == 'done', (sb, r['status'], r['reason'])
            assert r['meta']['frame_list_source'] == '.filelist'
        r36 = record.load_record(out, 'flow', f'flow_{cell}_36MHz')
        assert r36['meta']['zero_vis']['n'] == 1 and r36['meta']['gpu'] == '0'
        assert record.load_record(out, 'flow', f'flow_{cell}_41MHz')['meta']['gpu'] == '1'     # round-robin
        ship = record.load_record(out, 'flow', f'ship_{cell}')
        assert ship['status'] == 'done', ship['reason']
        assert len(os.listdir(os.path.join(out, 'snapshot_flow', cell))) == 4 + 3           # 41 MHz: one blank I, no field
        assert os.path.exists(os.path.join(out, 'snapshot_fleet_out', cell, 'flow', 'flow_41MHz.csv'))
        # shipping again is a no-op (identical bytes)
        assert run_cli(argv + ['--stages', 'ship']) == 0
        assert record.load_record(out, 'flow', f'ship_{cell}')['meta']['already_identical'] == 4 + 3 + 4
        # a re-run of the flow does not append to flow_<sb>.csv; its new log differs from the shipped one -> G-S9 refuses
        # the re-ship (B-49: never overwrite silently), --fresh --supersede moves the shipped files aside
        rc = run_cli(argv)
        assert rc == 2, rc
        assert 'G-S9' in record.load_record(out, 'flow', f'ship_{cell}')['reason']
        rows = open(os.path.join(work, cell, 'flow', 'flow_36MHz.csv')).read().splitlines()
        assert len(rows) == 1 + 4, rows
        assert run_cli(argv + ['--stages', 'ship', '--fresh', '--supersede', 's1']) == 0
        assert os.path.exists(os.path.join(out, '_superseded', 's1', 'MANIFEST.csv'))
        # G-S6: one V frame not flowed in 41 MHz -> that subband refused by name, 36 MHz done, ship refused
        shutil.rmtree(os.path.join(out, '_records'))
        os.environ['FAKE_SKIP_V'] = '20300101_120010'
        for p in os.listdir(os.path.join(stage, cell, '41MHz')):
            if p.endswith('_flowed.fits'):
                os.remove(os.path.join(stage, cell, '41MHz', p))
        rc = run_cli(argv + ['--scopes', '41MHz', '--stages', 'flow'])
        assert rc == 2, rc
        r = record.load_record(out, 'flow', f'flow_{cell}_41MHz')
        assert r['status'] == 'refused' and 'G-S6' in r['reason'] and '41MHz-Pilot-Snapshot-20300101_120010' in r['reason'], r['reason']
        del os.environ['FAKE_SKIP_V']
        # G-S2: a staged file the list does not name refuses the subband
        wfits(os.path.join(stage, cell, '36MHz', '36MHz-Clean-Snapshot-20300101_130000-image.fits'), np.zeros((N, N)))
        rc = run_cli(argv + ['--scopes', '36MHz', '--stages', 'flow'])
        assert rc == 2
        r = record.load_record(out, 'flow', f'flow_{cell}_36MHz')
        assert 'G-S2' in r['reason'], r['reason']
        # G-S1: no GPU anywhere -> every subband unit refused with the cell's reason
        rc = run_cli(['flow', '--tier', 'snapshot', '--cells', cell, '--work', work, '--out', out, '--snap-stage-record', rec,
                      '--stages', 'flow'])
        assert rc == 2
        assert 'G-S1' in record.load_record(out, 'flow', f'flow_{cell}_41MHz')['reason']
    finally:
        CFG.flow_snapshot.MASTERS_ROOT, CFG.site.PYTHON = saved[0], saved[1]
        SS.FLOW_SCRIPT = ('clv15', 'phase3_snapshot_flow.py')
    print('ok snapshot flow + ship (round-robin, set gate G-S6, G-S2, G-S1, no csv append, idempotent ship)')


# =====================================================================================================================
# sub-LST: the independent verdict (R1/R2/cards) and G-C7 with the approved rows
# =====================================================================================================================

FAKE_REFINE = r'''
import argparse, csv, glob, os, sys
import numpy as np
from astropy.io import fits
ap = argparse.ArgumentParser()
for k in ('--root', '--out', '--work', '--lst', '--subband', '--nights', '--windows'): ap.add_argument(k, default='')
a = ap.parse_args()
wts = {r['date']: float(r['weight']) for r in csv.DictReader(open(os.environ['FAKE_WEIGHTS']))}
skip = os.environ.get('FAKE_SKIP', '')
n_ref = 0
for k in range(6):
    st = {}
    for S in 'IV':
        for p in sorted(glob.glob(os.path.join(a.root, '%s_*' % a.lst, a.subband, S, '*_w%d.fits' % k))):
            st.setdefault(S, {})[p.split(os.sep)[-4].split('_', 1)[1]] = p
    for S, nd in st.items():
        nw = sum(1 for n in nd if wts.get(n, 0) > 0)
        for n, p in nd.items():
            if '%s,%d' % (n, k) == skip:
                if S == 'I': n_ref += 1
                continue
            h = fits.getheader(p); d = fits.getdata(p)
            for c, v in (('SUBTRACT', 'x'), ('LOONCOAD', nw - (1 if wts[n] > 0 else 0)), ('LOOMSTR', 'sublst_master_%s_%s_%s_w%d.fits' % (a.lst, a.subband, S, k)),
                         ('CODE', 'fake'), ('LSTBIN', k), ('DURATION', 600.0), ('REFINE', 'x'), ('REFREF', 'x'), ('FLOWMEDP', 0.1), ('JUPMASK', -1.0)):
                h[c] = v
            od = os.path.join(a.out, '%s_%s' % (a.lst, n), a.subband, S); os.makedirs(od, exist_ok=True)
            fits.writeto(os.path.join(od, os.path.basename(p)[:-5] + '_resid_loo.fits'), d - 0.5, h, overwrite=True)
print('SUBLST-REFINE %s %s aligned=x refused=%d refused_rule=0 refused_night=0' % (a.lst, a.subband, n_ref))
sys.exit(3 if n_ref else 0)
'''


def test_sublst_refine_verdict():
    import phase3_sublst_loo as LOO
    base = os.path.join(ROOT, 'sublst')
    lst, sb = '07h', '46MHz'
    nights = ['2031-02-01', '2031-02-02', '2031-02-03', '2031-02-04']
    stack_root = os.path.join(base, 'sublst')
    for n in nights:
        for S in 'IV':
            for k in range(6):
                wfits(os.path.join(stack_root, lst, f'{lst}_{n}', sb, S, f'sublst_{lst}_{n}_{sb}_{S}_w{k}.fits'),
                      np.random.default_rng(k).normal(size=(N, N)))
    for S in 'IV':
        for k in range(6):
            wfits(os.path.join(stack_root, f'{lst}_masters', sb, S, f'sublst_master_{lst}_{sb}_{S}_w{k}.fits'), np.zeros((N, N)))
    wdir = os.path.join(base, 'weights', lst, sb)
    os.makedirs(wdir, exist_ok=True)
    wcsv = os.path.join(wdir, 'weights.csv')
    with open(wcsv, 'w') as f:
        f.write('date,weight,in_master\n' + ''.join(f'{n},1.0,1\n' for n in nights))
    appr = os.path.join(base, 'approved.csv')
    open(appr, 'w').write('lst,subband,night,window,approved_utc,approval\n')
    fake = os.path.join(base, 'fake_refine.py')
    open(fake, 'w').write(FAKE_REFINE)
    tmpl = os.path.join(base, 'weights', '{lst}', '{sb}', 'weights.csv')
    saved = (CFG.flow_refine.WEIGHTS_CSV, CFG.flow_refine.APPROVED_REFUSALS_CSV, LOO.WEIGHTS_CSV, dict(CFG.site.PYTHON))
    CFG.flow_refine.WEIGHTS_CSV, CFG.flow_refine.APPROVED_REFUSALS_CSV, LOO.WEIGHTS_CSV = tmpl, appr, tmpl
    CFG.site.PYTHON = dict(CFG.site.PYTHON, default=PY)
    SL.REFINE_SCRIPT = ('clv15', fake)
    os.environ['FAKE_WEIGHTS'] = wcsv
    out, work = os.path.join(base, 'out'), os.path.join(base, 'work')
    argv = ['flow', '--tier', 'sublst', '--cells', f'{lst}_{sb}', '--gpus', '1', '--work', work, '--out', out,
            '--sublst-stack-root', stack_root]
    try:
        model = T.sublst_model(lst, sb, stack_root=os.path.join(stack_root, lst), refresh=True)
        assert len(model['expected']) == 2 * 6 * 4 and all(v == 3 for v in model['nloo'].values())
        rc = run_cli(argv)
        assert rc == 0, rc
        r = record.load_record(out, 'flow', f'refine_{lst}_{sb}')
        assert r['status'] == 'done' and r['meta']['verdict']['on_disk'] == 48, (r['status'], r['reason'], r['meta'].get('verdict'))
        assert r['meta']['child_env']['P3_FLOW_LOCK'].endswith('.sublst_refine_gpu1.lock')
        assert r['meta']['child_env']['DEEP_XLA_MEMFRAC'] == '0.5'
        # a re-run without --fresh refuses (G-C5); with --fresh --supersede the residuals are moved aside, then rewritten
        assert run_cli(argv) == 2
        assert 'G-C5' in record.load_record(out, 'flow', f'refine_{lst}_{sb}')['reason']
        # a night with no flow returned (module exit 3): refused unless EXACTLY the approved (night, window) rows
        os.environ['FAKE_SKIP'] = '2031-02-02,4'
        assert run_cli(argv + ['--fresh', '--supersede', 't1']) == 2
        r = record.load_record(out, 'flow', f'refine_{lst}_{sb}')
        assert 'G-C7' in r['reason'] and '2031-02-02' in r['reason'], r['reason']
        assert os.path.exists(os.path.join(out, '_superseded', 't1', 'MANIFEST.csv'))
        with open(appr, 'a') as f:
            f.write(f'{lst},{sb},2031-02-02,4,2031-01-01T00:00Z,"test approval words"\n')
        assert run_cli(argv + ['--fresh', '--supersede', 't2']) == 0
        r = record.load_record(out, 'flow', f'refine_{lst}_{sb}')
        assert r['status'] == 'done' and len(r['refused_items']) == 2 and r['refused_items'][0]['gate'] == 'G-C7', r
        del os.environ['FAKE_SKIP']
        # check alone, and the content test on the x2 directory
        assert run_cli(['flow', '--tier', 'sublst', '--cells', f'{lst}_{sb}', '--stages', 'check', '--out', out,
                        '--sublst-stack-root', stack_root]) == 0
        ok, n, det = T.allowed_and_complete(os.path.join(out, f'{lst}_x2'), subbands=[sb], stack_root=os.path.join(stack_root, lst))
        assert ok and n == 46, (ok, n, det)
        # a residual whose LOONCOAD disagrees with the independent count fails R1
        p = T.residual_path(os.path.join(out, f'{lst}_x2'), lst, '2031-02-01', sb, 'I', 0)
        with fits.open(p, mode='update') as h:
            h[0].header['LOONCOAD'] = 7
        res = {x['name']: x for x in __import__('exopipe.phase3.core.validate', fromlist=['run']).run(
            'sublst_x2_residual', p, stack_root=os.path.join(stack_root, lst))}
        r1 = res['sublst_x2_residual.allowed_and_complete']
        assert not r1['passed'] and 'LOONCOAD 7' in r1['detail'], r1
    finally:
        CFG.flow_refine.WEIGHTS_CSV, CFG.flow_refine.APPROVED_REFUSALS_CSV, LOO.WEIGHTS_CSV, CFG.site.PYTHON = saved
        SL.REFINE_SCRIPT = ('clv15', 'phase3_sublst_refine.py')
        os.environ.pop('FAKE_WEIGHTS', None)
    print('ok sub-LST refine: verdict R1/R2/cards, G-C5, G-C7 exact approved set, check alone')


# =====================================================================================================================
# deep: G-D0 (masters owns the reference), stored-flow stage-in + manifest, check freshness G-D14
# =====================================================================================================================

def test_deep_stagein_check():
    import importlib
    CO = importlib.import_module('exopipe.phase3.lib.masters.coadd')
    base = os.path.join(ROOT, 'deep')
    lst, sc = '05h', '23MHz'
    cell = f'{lst}_{sc}'
    p2 = os.path.join(base, 'phase2')
    nights = ['2031-03-01', '2031-03-02', '2031-03-03']
    imgs = {}
    for n in nights:
        imgs[n] = wfits(os.path.join(p2, lst, n, 'Science_20310101_000000', '23MHz', 'I', 'deep',
                                     f'23MHz-I-Taper-3600s-Robust-0.75-{n.replace("-", "")}_000000-image_dewarped.fits'),
                        np.ones((N, N)))
    out = os.path.join(base, 'out')
    W = os.path.join(out, cell)
    src = os.path.join(base, 'working', cell, 'flow')
    os.makedirs(src, exist_ok=True)
    time.sleep(0.02)
    wfits(os.path.join(W, 'coadd_final', f'I_deep_Taper_R075_{sc}.fits'), np.ones((N, N)), COADDOP='ivw-mean-v3',
          CONSTDIG='0' * 32)
    shutil.copytree(os.path.join(W, 'coadd_final'), os.path.join(W, 'coadd_final_ref'))
    for n in nights:
        np.save(os.path.join(src, f'{n}_scaled_flow.npy'), np.zeros((N, N, 2), np.float32))
        wfits(os.path.join(src, f'{n}_scaled_dewarp.fits'), np.ones((N, N)))
    open(os.path.join(src, '.reference_resolved'), 'w').write(f'MASTER:ivw-mean-v3:{"0" * 32}')
    saved = (CFG.paths.PHASE2_ROOT, CO.PHASE2_ROOT)
    CFG.paths.PHASE2_ROOT = CO.PHASE2_ROOT = p2
    argv = ['flow', '--tier', 'deep', '--cells', cell, '--out', out, '--stages', 'stage-in,check', '--deep-flow-from', src]
    try:
        assert run_cli(argv) == 2                                         # G-D0: no masters stage-in record yet
        assert 'G-D0' in record.load_record(out, 'flow', cell)['reason']
        record.atomic_write_json(record.record_path(out, 'masters', f'stage-in_{cell}'),
                                 {'status': 'done', 'unit': f'stage-in_{cell}', 'meta': {'reference': 'previous-master'}})
        rc = run_cli(argv)
        r = record.load_record(out, 'flow', cell)
        assert rc == 0 and r['status'] == 'done', (rc, r['status'], r['reason'], [t for t in r['tests'] if not t['passed']])
        assert r['meta']['stage_in']['copied'] == 7 and r['meta']['stages_done'] == ['stage-in', 'check']
        assert sorted(r['admitted']) == nights
        assert sum(1 for o in r['outputs'] if o['class'] == 'coadd_flow') == 3
        # an input image newer than its flow -> G-D14 at check
        time.sleep(1.1)
        os.utime(imgs['2031-03-02'])
        rc = run_cli(['flow', '--tier', 'deep', '--cells', cell, '--out', out, '--stages', 'check'])
        assert rc == 2 and 'G-D14' in record.load_record(out, 'flow', cell)['reason']
    finally:
        CFG.paths.PHASE2_ROOT, CO.PHASE2_ROOT = saved
    # flow_prep_scaled: <date>_scaled.fits = input / safe(scale map from _scale.json) (coadd_full_pipeline.py:1772-1798)
    prep = os.path.join(W, 'prep')
    os.makedirs(prep, exist_ok=True)
    src = np.random.default_rng(5).normal(10.0, 1.0, size=(N, N))
    inp = wfits(os.path.join(base, 'input.fits'), src)
    coeffs = [1.2, 0.05, -0.03]
    yy, xx = np.indices((N, N), dtype=float)
    smap = coeffs[0] + coeffs[1] * (2 * xx / N - 1) + coeffs[2] * (2 * yy / N - 1)
    wfits(os.path.join(prep, '2031-03-01_scaled.fits'), (src.astype(np.float32).astype(np.float64) / smap))
    json.dump({'coeffs': coeffs, 'n_sources': 7, 'mean_scale': 1.2, 'tgt_rms': 2.0, 'tgt_rms_scaled': 2.0 / 1.2,
               'weight': 1.2 ** 2 / 4.0}, open(os.path.join(prep, '2031-03-01_scale.json'), 'w'))
    ok, val, det = T.scaled_equals_input_over_scale(os.path.join(prep, '2031-03-01_scaled.fits'), input_image=inp)
    assert ok, (val, det)
    wfits(os.path.join(prep, '2031-03-01_scaled.fits'), src / 1.0)
    assert not T.scaled_equals_input_over_scale(os.path.join(prep, '2031-03-01_scaled.fits'), input_image=inp)[0]
    # flow_reference: prep/ref_master_prev.fits is the byte copy of masters' reference (coadd_final_ref/)
    shutil.copy(os.path.join(W, 'coadd_final_ref', f'I_deep_Taper_R075_{sc}.fits'), os.path.join(prep, 'ref_master_prev.fits'))
    res = {x['name']: x['passed'] for x in __import__('exopipe.phase3.core.validate', fromlist=['run']).run(
        'flow_reference', os.path.join(prep, 'ref_master_prev.fits'))}
    assert all(res.values()), res
    print('ok deep: G-D0, stored-flow stage-in + manifest, check (coadd_flow tests), G-D14, flow_prep_scaled, flow_reference')


def main():
    t0 = time.time()
    shutil.rmtree(ROOT, ignore_errors=True)
    os.makedirs(ROOT)
    try:
        test_config_and_stages()
        test_items()
        test_set_comparison_d12()
        test_snapshot_flow_and_ship()
        test_sublst_refine_verdict()
        test_deep_stagein_check()
    finally:
        shutil.rmtree(ROOT, ignore_errors=True)
    print(f'ALL FLOW TESTS PASSED in {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
