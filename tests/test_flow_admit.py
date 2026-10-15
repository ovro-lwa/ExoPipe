"""D-ADMIT (a), slice 3: `exopipe3 flow --tier deep --stages admit` (lib/flow/steps_deep.run_admit) - the flows of the nights
an installed cell admits, from the pairs of the done `masters --stages admit-prep` record.

No GPU is ever used here: the missing-flow case runs with the GPU gate replaced by a refusal (it must refuse before anything
runs, and nothing is written), and the present-flow case uses a member night's STORED flow copied in (the installed cell
13h_78MHz is not dewarped, so its stored flows align to the installed master's frame). Checks: the pairs come only from a done
admit-prep record; a flow that is present and newer than its input is not recomputed and passes the build's flow content
tests (shape and finite fraction, freshness, the reference signature in admit/flow/.reference_resolved, the warped image's
structure); a flow directory made against another reference refuses; 'admit' never runs with a build's flow stages.
Run: python tests/test_flow_admit.py
"""
import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from exopipe.phase3.core import env as ENV                          # noqa: E402
ENV.legacy()
from exopipe.phase3.config import CFG                               # noqa: E402
from exopipe.phase3.core import record                              # noqa: E402
from exopipe.phase3.core.record import Refusal                      # noqa: E402
from exopipe.phase3.lib.flow import steps_deep as FD                # noqa: E402
from exopipe.phase3.lib.flow import tests as _flow_tests            # noqa: E402,F401 (registers the flow content tests)
from exopipe.phase3.lib.masters import admit as A                   # noqa: E402
from exopipe.phase3.lib.masters import admit_stages as AS           # noqa: E402
from exopipe.phase3.lib.masters import coadd as co                  # noqa: E402

T = '/tmp/exopipe3_tests_flow_admit'
FAILS, N = [], [0]
CELL, SC = '13h_78MHz', '78MHz'


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({detail})'))
    if not ok:
        FAILS.append(name)


def setup():
    """A cell tree with the admission reference and one night's scaled flow input (prep_night, as admit-prep makes them),
    and a done admit-prep record naming the pair. Returns (cell dir, out root, night, pair)."""
    src = os.path.join(CFG.paths.COADD_SB_ROOT, CELL)
    cd = os.path.join(T, 'wt', CELL)
    os.makedirs(os.path.join(cd, 'coadd_final'), exist_ok=True)
    ents = A.installed_reference(CELL, src)
    refs = A.write_reference(cd, SC, ents)
    ref = AS.reference_dict(co, refs['I'], ents[0])
    with contextlib.redirect_stdout(io.StringIO()):
        nights = [x for x in co.discover_dates('13h', '78') if os.path.exists(os.path.join(src, 'flow', f'{x[0]}_scaled_flow.npy'))]
        d, sd, img = [x for x in nights if x[0] >= co.CUTOVER_DATE][-1]
        W = os.path.join(cd, A.ADMIT_DIR)
        os.makedirs(os.path.join(W, 'prep'), exist_ok=True)
        pair, _row = co.prep_night(d, sd, img, ref, W, [True])
    out = os.path.join(T, 'out')
    os.makedirs(os.path.join(out, '_records', 'masters'), exist_ok=True)
    record.atomic_write_json(record.record_path(out, 'masters', f'admit-prep_{CELL}'),
                             dict(status='done', run_id='test', meta=dict(admit=dict(pairs=[pair]))))
    return cd, out, d, pair


def unit(out, name='u'):
    CFG.authorize([], out=out)
    run = record.Run('flow', ['test'], out, CFG, config_sections=())
    run.commit = 'abc1234'
    return run


def args(**kw):
    a = argparse.Namespace(deep_flow_all=False, gpus=None, cells=CELL, nights=None, epochs=None, lsts=None, scopes=None,
                           exclude_cells=None)
    a.__dict__.update(kw)
    return a


def test_missing_flow_refuses_without_a_gpu(cd, out, d, pair):
    run = unit(out)
    real, launched = FD.solve_gpu, []

    def no_gpu(a):
        raise Refusal('G-D18: no GPU id (test)')
    FD.solve_gpu = no_gpu
    real_batch = FD.batch_flow
    FD.batch_flow = lambda *a, **k: launched.append(a)
    try:
        with run.unit(f'admit_{CELL}', tier='deep', stage='admit', cell=CELL) as u:
            FD.run_admit(u, args(), run, CELL)
    finally:
        FD.solve_gpu, FD.batch_flow = real, real_batch
    check('a missing flow needs the GPU: the GPU gate refuses before anything runs, nothing written',
          u.status == 'refused' and 'G-D18' in u.reason and not launched
          and not os.path.exists(os.path.join(pair['out_dir'], f'{d}_scaled_flow.npy')), (u.status, u.reason))


def test_present_flow(cd, out, d, pair):
    src = os.path.join(CFG.paths.COADD_SB_ROOT, CELL, 'flow')
    for f in (f'{d}_scaled_flow.npy', f'{d}_scaled_dewarp.fits'):
        shutil.copy2(os.path.join(src, f), os.path.join(pair['out_dir'], f))
        t = time.time()
        os.utime(os.path.join(pair['out_dir'], f), (t, t))             # newer than its scaled input
    run = unit(out)
    real, calls = FD.solve_gpu, []
    FD.solve_gpu = lambda a: calls.append(a) or (_ for _ in ()).throw(Refusal('must not be called'))
    try:
        with run.unit(f'admit_{CELL}', tier='deep', stage='admit', cell=CELL) as u:
            FD.run_admit(u, args(), run, CELL)
    finally:
        FD.solve_gpu = real
    tests = {t['name']: t['passed'] for t in u.tests}
    sig = open(os.path.join(pair['out_dir'], '.reference_resolved')).read().strip()
    check('a present, fresh flow is not recomputed (no GPU call) and the unit is done', u.status == 'done' and not calls,
          (u.status, u.reason))
    check('the flow passes the build\'s flow content tests (shape/finite, fresh, reference signature, warped image)',
          tests and all(tests.values()) and {'coadd_flow.shape_and_finite', 'coadd_flow.fresh', 'coadd_flow.reference_signature',
                                             'flow_dewarp_image.structure'} <= set(tests), tests)
    check('admit/flow/.reference_resolved names the reference as a build does (MASTER:<COADDOP>:<CONSTDIG>)',
          sig == FD.admit_signature(pair['reference']) and sig.startswith('MASTER:'), sig)
    with open(os.path.join(pair['out_dir'], '.reference_resolved'), 'w') as f:
        f.write('MASTER:ivw-mean-v3:00000000000000000000000000000000')
    run = unit(out)
    with run.unit(f'admit_{CELL}', tier='deep', stage='admit', cell=CELL) as u:
        FD.run_admit(u, args(), run, CELL)
    check('a flow directory made against another reference refuses', u.status == 'refused' and 'reference' in u.reason,
          u.reason)
    with open(os.path.join(pair['out_dir'], '.reference_resolved'), 'w') as f:
        f.write(sig)


def test_record_and_stages(cd, out, d, pair):
    run = unit(os.path.join(T, 'out_none'))
    with run.unit(f'admit_{CELL}', tier='deep', stage='admit', cell=CELL) as u:
        FD.run_admit(u, args(), run, CELL)
    check('no done admit-prep record: refused (G-D0 for admissions)', u.status == 'refused' and 'admit-prep' in u.reason,
          u.reason)
    a = args(flow_stages=['prep', 'admit'])
    try:
        FD.run(a, unit(out))
        check('admit with a build\'s flow stages refuses', False)
    except Refusal as e:
        check('admit with a build\'s flow stages refuses', 'alone' in str(e), str(e))
    try:
        FD.items(args(epochs='13h_2025-01-10'))
        check('--epochs is refused by a build\'s flow stages', False)
    except Refusal as e:
        check('--epochs is refused by a build\'s flow stages, accepted by admit', 'except --stages admit' in str(e)
              and FD.items(args(epochs='13h_2025-01-10'), admit=True) == [CELL], str(e))
    run = unit(out)
    with run.unit(f'admit_{CELL}', tier='deep', stage='admit', cell=CELL) as u:
        FD.run_admit(u, args(epochs='13h_2025-01-10'), run, CELL)
    check('admit --epochs naming a night without a pair: no input', u.status == 'no-input' and '--epochs' in u.reason, u.reason)


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        cd, out, d, pair = setup()
        test_missing_flow_refuses_without_a_gpu(cd, out, d, pair)
        test_present_flow(cd, out, d, pair)
        test_record_and_stages(cd, out, d, pair)
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
