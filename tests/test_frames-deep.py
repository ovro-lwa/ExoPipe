#!/usr/bin/env python3
"""STANDING TEST - `exopipe3 frames --tier deep` (frame-only rebuilds of the deep search frames), on synthetic data plus
read-only real products.

  config     the frames / frames_residuals draft loads; PROTECTED_KEYS refuse; RULED ledger IDs exist; engine values come
             from search's sections (one reader) and the mirrors agree
  seam       lib/frames/deep_seam: the proposed frames_only edit of the REAL engine source keeps the default call
             AST-identical and makes the search unreachable; a tampered default path and a literal fold are detected
  items      per-hour --epochs (core.cells epoch kind) x bands x Stokes, bulk --cells, --nights, unit names, Stokes order
  overrides  an engine literal override refuses; a detection-key override is noted
  child      core.legacy.run plumbing with a FAKE engine child: env from config, inherited P3_* cleared, result mapping
             (ok / refused -> Refusal / policy skip -> NoInput / error / crash -> RuntimeError)
  register   a synthetic attempt (frames, siblings, archive copies, gate refusal, unaccounted epoch, sigma stack) in a
             synthetic cell: every planned output written+tested or refused by a named gate; unit done; verify re-tests
             the recorded outputs; negative copies fail their content tests
  cli        refusals of a real-cell run that would write: no --out (production window), no --deep-rebuild
             (deep-producer), no frames_only keyword (engine-seam) - each exit 2 before any product is written
  real       READ-ONLY: content tests on production frames, siblings, archive copies, sigma stacks and epoch images;
             --deep-adopt of a published attempt (records under the scratch dir); --dry-run on the real cell 07h_18MHz
             and the per-hour item 07h_2025-10-15

Scratch: /tmp/exopipe3_tests_frames-deep/ (removed at the end). Production trees are only read.
Run standalone:  python tests/test_frames-deep.py [test_name ...]
"""
import json
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')
SCR = '/tmp/exopipe3_tests_frames-deep'
PY = sys.executable
REAL_CELL_ROOT = '/data07/gh/phase3/coadd_sb'
REAL_ATT_V = '/lustre/gh/main/phase3_collect/p3s5_collect_lwacalim07/attempts/07h_18MHz_V_20261012T212735'
REAL_ATT_I = '/lustre/gh/main/phase3_collect/p3s5_collect_lwacalim07/attempts/07h_18MHz_I_20261012T214305'
REAL_ARCH = '/lustre/pipeline/exopipe/phase3/07h/2025-01-02/Run_20260909_000000/18MHz/V/deep'
SYN_DATES = ('2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04')    # nights no exclusion row names (asserted)


def _fresh(name):
    d = os.path.join(SCR, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return d


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


class _Args:
    """argparse-like namespace with the step's defaults."""

    def __init__(self, **kw):
        d = dict(out=None, work=None, run_label=None, archive_copy=None, archive_run=None, deep_x2_flows=None,
                 workers=None, deep_rebuild=False, coadd_root=None, resume=False, fresh=False, dry_run=False,
                 epochs=None, cells=None, lsts=None, nights=None, scopes=None, stokes=None, exclude_cells=None,
                 deep_adopt=None, stages=None, from_stage=None, until_stage=None, tier='deep')
        d.update(kw)
        self.__dict__.update(d)


def _cli(argv, timeout=300):
    env = dict(os.environ)
    env['PYTHONPATH'] = REPO
    p = subprocess.run([PY, '-m', 'exopipe.phase3'] + argv, cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, p.stdout, p.stderr


# ---------------------------------------------------------------------------------------------------------------------

def test_config():
    from exopipe.phase3.config import CFG, ConfigRefusal
    from exopipe.phase3.core import ledger
    from exopipe.phase3.lib.frames import deep_engine
    for sec in ('frames', 'frames_residuals'):
        assert sec in CFG.sections(), sec
        s = CFG.sections()[sec]
        for k, ids in (getattr(s, 'RULED', {}) or {}).items():
            for i in (ids if isinstance(ids, tuple) else (ids,)):
                assert ledger.has_id(i, CFG.paths.FIX_LEDGER), f'{sec}.{k}: ledger ID {i} not in the fix ledger'
        assert not [a for a in vars(s) if a in ('protected', 'locked', 'lit', 'mirror')], 'lower-case marker in ' + sec
    for key in ('frames_residuals.PB_FLOOR', 'frames_residuals.DEEP_PRODUCER', 'frames_residuals.LOO_NCOADD_RANGE'):
        try:
            CFG.set(key, '1')
            raise AssertionError(f'{key} override accepted')
        except ConfigRefusal:
            pass
    v = deep_engine.engine_values('deep')
    src = v['_sources']
    for name in ('master_gen', 'min_epochs', 'sanity_i_x', 'beam_gate_ratio', 'streaming', 'run_label', 'stokes_order',
                 'flow_lock'):
        assert src[name].startswith('search'), (name, src[name])
    assert v['pb_floor'] == 0.05 and v['rms_box_half_px'] == 512 and v['loo_ncoadd_range'] == (2, 20)
    assert v['archive_root'] == CFG.paths.PHASE3_ARCHIVE_ROOT and v['archive_run'] == CFG.paths.ARCHIVE_RUN
    assert v['roots_csv'].startswith(CFG.paths.OPS_ROOT) and os.path.exists(v['roots_csv'])
    assert v['refine_x2'] and v['striation'] and v['loo']
    print('config: sections load, protected keys refuse, ruling IDs exist, engine values from search sections')


def test_preview_sanity_minimum():
    """The preview's 10x sanity discard runs with the engine's own minimum, search_detect.SANITY_MIN_NIGHTS (the engine reads
    P3_SANITY_MIN_NIGHTS; review 2026-10-14 m1): 6 nights with a qa3 rms -> the 50x night removed at 5, none at 7."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.frames import deep_cells
    dates = [f'2025-01-0{i}' for i in range(1, 7)]

    class FakeCell:
        lst, scope, final = '07h', '18MHz', os.path.join(SCR, 'no_such_cell', 'coadd_final')

        def epoch_dates(self):
            return list(dates)

        def weights(self):
            return [{'date': d, 'qa3_rms_mjy': q} for d, q in zip(dates, ('10', '10', '10', '10', '10', '500'))]

        def in_master(self):
            return {}

        def master(self, st):
            return os.path.join(SCR, 'no_such_master.fits')

        def loo(self, d, st):
            return os.path.join(SCR, 'no_such_loo.fits')

        def header(self, p):
            return {}

    class NoExcl:
        def is_excluded(self, *a):
            return False
    d = _fresh('preview')
    at5 = deep_cells.preview(FakeCell(), 'V', NoExcl())['sanity']
    CFG.reset()
    CFG.authorize([], out=d)
    CFG.set('search_detect.SANITY_MIN_NIGHTS', '7')
    try:
        at7 = deep_cells.preview(FakeCell(), 'V', NoExcl())['sanity']
    finally:
        CFG.reset()
    assert at5 == ['2025-01-06'] and at7 == [], (at5, at7)
    print('preview: the 10x sanity discard uses search_detect.SANITY_MIN_NIGHTS (5: removed; 7: gate skipped)')


def test_seam():
    import ast
    from exopipe.phase3.lib.frames import deep_engine, deep_seam
    src = open(deep_engine.engine_source()).read()
    if deep_seam.has_keyword(ast.parse(src)):
        rev, orig = deep_seam.committed_without_keyword(deep_engine.engine_source())
        ok, det = deep_seam.default_path_identical(src, orig)
        assert ok, det
        print(f'seam: engine already has frames_only; default path identical to {rev[:10]}')
        return
    patched = deep_seam.propose(src)
    compile(patched, 'patched', 'exec')
    ok, det = deep_seam.default_path_identical(patched, src)
    assert ok, det
    ok2, det2 = deep_seam.frames_only_path_ok(patched)
    assert ok2, det2
    # a default-path change outside the keyword's branches is caught
    bad = patched.replace("    if frames_only:\n        _fo = {'I': [], 'V': []}\n",
                          "    _fo = {'I': [], 'V': []}\n", 1)
    assert bad != patched
    assert not deep_seam.default_path_identical(bad, src)[0]
    # only constants that come from the keyword fold: the literal 0 of `x or 0` is untouched
    toy_o = "def run_deep_cell(a):\n    y = a.get('k') or 0\n    return y\n"
    toy_p = ("def run_deep_cell(a, frames_only=False):\n    if frames_only:\n        return {}\n"
             "    y = a.get('k') or 0\n    return (1 if frames_only else y)\n")
    assert deep_seam.default_path_identical(toy_p, toy_o)[0]
    toy_bad = "def run_deep_cell(a, frames_only=False):\n    y = a.get('k')\n    return y\n"
    assert not deep_seam.default_path_identical(toy_bad, toy_o)[0]
    # an anchor that no longer occurs refuses instead of patching the wrong place
    try:
        deep_seam.propose(src.replace('_archive_frame_publish(vr_path.replace', '_archive_frame_publish( vr_path.replace'))
        raise AssertionError('missing anchor accepted')
    except ValueError:
        pass
    print(f'seam: {det}; {det2}')


def test_items():
    from exopipe.phase3.lib.frames import deep_step
    ps = deep_step.build_plans(_Args(epochs='07h_2025-10-15', scopes='18MHz,Red'))
    assert [p.name for p in ps] == ['07h_18MHz_V_2025-10-15', '07h_18MHz_I_2025-10-15', '07h_Red_V_2025-10-15',
                                    '07h_Red_I_2025-10-15'], [p.name for p in ps]
    assert all(p.mode == 'per-hour' and p.date == '2025-10-15' and p.item == '07h_2025-10-15' for p in ps)
    ps = deep_step.build_plans(_Args(epochs='07h_2025-10-15'))
    assert len(ps) == 2 * (15 + 4), len(ps)                     # every band and subband, both Stokes
    ps = deep_step.build_plans(_Args(cells='07h_18MHz', stokes='I,V'))
    assert [p.name for p in ps] == ['07h_18MHz_V', '07h_18MHz_I'] and ps[0].mode == 'bulk' and ps[0].date is None
    ps = deep_step.build_plans(_Args(cells='07h_18MHz', stokes='V', nights='2025-01-02,2025-01-03'))
    assert ps[0].mode == 'incremental' and ps[0].name.startswith('07h_18MHz_V_n2_')
    for bad in (dict(epochs='07h_2025-10-15', cells='07h_Red'), dict(epochs='07h_2025-10-15_18MHz'), dict()):
        try:
            deep_step.build_plans(_Args(**bad))
            raise AssertionError(f'{bad} accepted')
        except ValueError:
            pass
    print('items: per-hour, bulk, incremental units and refusals of bad item specs')


def test_overrides():
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.frames import deep_engine

    class C:
        def __init__(self, ov):
            self.overrides = [(k, None, None, '--set', None) for k in ov]
    try:
        deep_engine.check_overrides(C(['search_detect.SIGMA_STACK_BUILD_MIN']))
        raise AssertionError('engine literal override accepted')
    except Refusal:
        pass
    notes = deep_engine.check_overrides(C(['search_detect.MSTRUCT_K', 'search_detect.SANITY_X']))
    assert len(notes) == 1 and 'MSTRUCT_K' in notes[0], notes
    # the gates' minimum counts reach the engine (P3_SANITY_MIN_NIGHTS etc.; review 2026-10-14 m1): neither refused nor noted
    notes = deep_engine.check_overrides(C(['search_detect.SANITY_MIN_NIGHTS', 'search_detect.SEARCH_TIER_MIN',
                                           'search_sublst.BEAM_GATE_MIN_NIGHTS']))
    assert not notes, notes
    print('overrides: engine literal refused, detection key noted')


_FAKE_CHILD = r'''
import json, os, sys
spec = json.load(open(sys.argv[1]))
mode = spec['label']
env = {k: v for k, v in os.environ.items() if k.startswith(('P3_', 'DEEP_'))}
res = {'unit': spec['lst'] + '_' + spec['scope'], 'env': env}
if mode == 'CRASH':
    sys.exit(9)
if mode == 'OK':
    res.update(status='ok', **{'return': {'frames': {'V': []}, 'siblings': {'V': []}, 'policy_skip': None}})
elif mode == 'SKIP':
    res.update(status='ok', **{'return': {'policy_skip': 'only 0 epochs (minimum 1, cadence deep)'}})
elif mode == 'REF':
    res.update(status='refused', gate='engine', reason='07h 18MHz: REFUSING to search a superseded master')
else:
    res.update(status='error', reason='RuntimeError: flow batch failed rc=-6')
json.dump(res, open(spec['result_path'], 'w'))
print('fake child', mode)
sys.exit({'OK': 0, 'SKIP': 0, 'REF': 2}.get(mode, 1))
'''


def test_child():
    from exopipe.phase3.core.record import NoInput, Refusal
    from exopipe.phase3.lib.frames import deep_engine
    d = _fresh('child')
    fake = os.path.join(d, 'fakemods')
    os.makedirs(fake)
    with open(os.path.join(fake, 'fake_frames_child.py'), 'w') as f:
        f.write(_FAKE_CHILD)
    old_mod, old_pp, old_stale = deep_engine.CHILD_MODULE, os.environ.get('PYTHONPATH'), os.environ.get('P3_STALE')
    deep_engine.CHILD_MODULE = 'fake_frames_child'
    os.environ['PYTHONPATH'] = fake + (os.pathsep + old_pp if old_pp else '')
    os.environ['P3_STALE'] = 'must-not-reach-the-child'
    try:
        def spec(label):
            return deep_engine.UnitSpec(tier='deep', lst='07h', scope='18MHz', stokes='V', label=label,
                                        attempt=os.path.join(d, 'att_' + label), coadd_root=REAL_CELL_ROOT,
                                        archive_root=os.path.join(d, 'archive'), archive_run='Run_TEST', archive_copy=True,
                                        work_root=os.path.join(d, 'work'))
        res = deep_engine.run_child(spec('OK'), os.path.join(d, 'scr_ok'))
        env = res['env']
        assert 'P3_STALE' not in env, 'inherited P3_* reached the child'
        assert env['P3_COADD_ROOT'] == REAL_CELL_ROOT and env['P3_ARCHIVE_RUN'] == 'Run_TEST'
        assert env['P3_MASTER_GEN'] == 'ivw-mean-v3' and env['P3_DEEP_MIN_EPOCHS'] == '1' and env['P3_MIN_EPOCHS'] == '3'
        assert env['P3_STREAM_RESID'] == '1' and env['DEEP_NPROC_RESID'] == '6' and 'P3_REFINE_OFF' not in env
        assert (env['P3_SANITY_MIN_NIGHTS'], env['P3_SEARCH_TIER_MIN'], env['P3_BEAM_GATE_MIN_NIGHTS']) == ('5', '5', '5'), env
        # an override of a minimum reaches the child (deep: search_detect; 10-minute: search_sublst) - review 2026-10-14 m1
        from exopipe.phase3.config import CFG
        CFG.reset()
        CFG.authorize([], out=d)
        CFG.set('search_detect.SANITY_MIN_NIGHTS', '7')
        CFG.set('search_sublst.SEARCH_TIER_MIN', '6')
        try:
            e7 = deep_engine.child_env(spec('OK'))
            vs = deep_engine.engine_values('sublst')
        finally:
            CFG.reset()
        assert (e7['P3_SANITY_MIN_NIGHTS'], e7['P3_SEARCH_TIER_MIN'], e7['P3_BEAM_GATE_MIN_NIGHTS']) == ('7', '5', '5'), e7
        assert (vs['sanity_min_nights'], vs['search_tier_min'], vs['beam_gate_min_nights']) == (5, 6, 5), vs
        assert env['P3_STAGING_ROOT'] == os.path.join(d, 'att_OK') and env['P3_VRESID_ROOT'].endswith('Run_OK/vresid')
        assert os.path.exists(os.path.join(d, 'att_OK', '07h_18MHz.log'))
        for label, exc in (('SKIP', NoInput), ('REF', Refusal), ('ERR', RuntimeError), ('CRASH', RuntimeError)):
            try:
                deep_engine.run_child(spec(label), os.path.join(d, 'scr_' + label))
                raise AssertionError(f'{label}: no exception')
            except exc as e:
                if label == 'REF':
                    assert 'REFUSING' in str(e)
        s = spec('OK')
        s.archive_copy = False
        assert deep_engine.child_env(s)['P3_ARCHIVE_WRITE'] == '0'
    finally:
        deep_engine.CHILD_MODULE = old_mod
        if old_pp is None:
            os.environ.pop('PYTHONPATH', None)
        else:
            os.environ['PYTHONPATH'] = old_pp
        if old_stale is None:
            os.environ.pop('P3_STALE', None)
    print('child: env from config, P3_* cleared, ok / policy skip / refusal / error / crash mapped')


# ---- synthetic attempt -----------------------------------------------------------------------------------------------

def _syn_cell(root):
    """A synthetic co-add cell 07h_18MHz: V master (NCOADD 18), weights.csv (d1 in the master, d2 outside), PB map."""
    import numpy as np
    from astropy.io import fits
    fd = os.path.join(root, '07h_18MHz', 'coadd_final')
    os.makedirs(fd)
    os.makedirs(os.path.join(root, '07h_18MHz', 'subtracted'))
    h = fits.Header()
    h['NCOADD'], h['COADDOP'], h['CONSTDIG'] = 18, 'ivw-mean-v3', 'x' * 32
    fits.writeto(os.path.join(fd, 'V_deep_Taper_R0_18MHz.fits'), np.zeros((8, 8), np.float32), h)
    with open(os.path.join(fd, 'weights.csv'), 'w') as f:
        f.write('date,weight,qa3_rms_mjy,in_master\n')
        for dd, im in zip(SYN_DATES, (1, 0, 1, 1)):
            f.write(f'{dd},0.1,100.0,{im}\n')
    _syn_nightimage(fd, fail=(SYN_DATES[2],))
    for dd in SYN_DATES:                       # the engine's epoch list is the stage-1 file names (placeholders here)
        open(os.path.join(root, '07h_18MHz', 'subtracted', f'{dd}_I_sub_18MHz.fits'), 'w').close()
    yy, xx = np.mgrid[0:256, 0:256]
    pb = np.exp(-((xx - 128.0) ** 2 + (yy - 128.0) ** 2) / (2 * 60.0 ** 2)).astype(np.float32)
    fits.writeto(os.path.join(fd, 'PB_deep_18MHz.fits'), pb)
    return pb


def _syn_nightimage(final_dir, fail):
    """The cell's night-image gate record (lib/masters/night_image_gate format): the nights in ``fail`` FAIL."""
    import csv
    from exopipe.phase3.lib.masters import night_image_gate as ng
    with open(os.path.join(final_dir, ng.GATE_FILE), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ng.COLS)
        w.writeheader()
        for dd in SYN_DATES:
            w.writerow(dict(date=dd, verdict='FAIL' if dd in fail else 'PASS', method='synthetic'))


def _syn_frame(path, date, loo, rng, template_hdr):
    """A synthetic V search frame with the production header layout (a real frame's header as template, read only)."""
    import numpy as np
    from astropy.io import fits
    from astropy.stats import mad_std
    from exopipe.phase3.core.products import build_name
    d = rng.normal(0, 0.3, (256, 256)).astype(np.float32)
    h = template_hdr.copy()
    half = min(512, 64, 64)
    r = float(round(float(mad_std(d[128 - half:128 + half, 128 - half:128 + half].astype(np.float64))), 9))
    h['THERMNOI'] = r
    h['IMGRMS'] = r
    h['DATE-OBS'] = f'{date}T08:00:00.000'
    mf = f'{date}_V_loo_Taper_R0_18MHz.fits' if loo else 'V_deep_Taper_R0_18MHz.fits'
    h['LOOMSTR'], h['MSTRFILE'], h['MSUBFILE'] = loo, mf, mf
    if loo:
        h['LOOEXCL'] = date
    elif 'LOOEXCL' in h:
        del h['LOOEXCL']
    h['PRODNAME'] = build_name('18MHz', 'V', 'deep', 'Taper', 0.0, proc=['dewarped', 'aligned', 'msub', 'ssub'],
                               lst='07h', epoch=f'{date}T08:00:00')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.writeto(path, d, h)
    return d, h


def _syn_attempt(d):
    """Synthetic engine output for unit 07h_18MHz_V over the four synthetic nights: d1 (LOO) and d2 (outside the master,
    central) have frames + siblings + archive copies; d3 removed by the night-image gate; d4 has nothing (unaccounted)."""
    import numpy as np
    from astropy.io import fits
    from exopipe.phase3.lib.frames import deep_engine
    rng = np.random.default_rng(7)
    root = os.path.join(d, 'coadd')
    pb = _syn_cell(root)
    tpl = fits.getheader(os.path.join(REAL_ATT_V, 'Run_P3S5', 'vresid', '07h_18MHz', '2025-01-02_V_redo_resid_18MHz.fits'))
    spec = deep_engine.UnitSpec(tier='deep', lst='07h', scope='18MHz', stokes='V', label='P3S5',
                                included_dates=None, attempt=os.path.join(d, 'deep_attempts', '07h_18MHz_V_20240101T000000'),
                                coadd_root=root, archive_root=os.path.join(d, 'archive'), archive_run='Run_TEST',
                                archive_copy=True, work_root=os.path.join(d, 'work'))
    d1, d2, d3, d4 = SYN_DATES
    for dd, loo in ((d1, True), (d2, False)):
        data, h = _syn_frame(spec.frame_path(dd, 'V'), dd, loo, rng, tpl)
        pbm = np.where(pb > 0.05, pb, np.nan)
        hs = h.copy()
        hs['PBCORR'], hs['PBFILE'] = True, 'PB_deep_18MHz.fits'
        hs['PRODNAME'] = h['PRODNAME'].replace('_dewarped_', '_pbcorr_dewarped_', 1)
        fits.writeto(spec.frame_path(dd, 'V', True), (data / pbm).astype(np.float32), hs)
        for p in (spec.frame_path(dd, 'V'), spec.frame_path(dd, 'V', True)):
            dst = deep_engine.archive_path(p, spec)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(p, dst)
    os.makedirs(os.path.dirname(spec.refusal_path('nightimage')))
    with open(spec.refusal_path('nightimage'), 'w') as f:
        f.write(f'07h,18MHz,{d3},20240101T000001\n')
    with open(spec.refusal_path('datadecisions'), 'w') as f:   # B-02 step 3c: d4 left out as excluded, admitted since (a
        f.write(f'07h,18MHz,{d4},20240101T000001\n')            # lifted exclusion) - owed now, no frame: unaccounted still
    os.makedirs(os.path.join(d, 'work', '07h_18MHz'))
    for dd in (d1, d2, d3):
        open(os.path.join(d, 'work', '07h_18MHz', f'ep_{dd}_up2_flow.npy'), 'w').close()   # d4: the flow batch dropped it
    res = {'status': 'ok', 'return': {'frames': {'V': [spec.frame_path(d1, 'V'), spec.frame_path(d2, 'V')]},
                                      'siblings': {'V': [spec.frame_path(d1, 'V', True), spec.frame_path(d2, 'V', True)]},
                                      'policy_skip': None}}
    return root, spec, res


def test_register():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import exclusions, record, validate
    from exopipe.phase3.lib.frames import deep_cells, deep_step
    d = _fresh('register')
    root, spec, res = _syn_attempt(d)
    out = os.path.join(d, 'out')
    run = record.Run('frames', ['exopipe3 frames', 'test'], out, CFG)
    args = _Args(out=out, work=os.path.join(d, 'work_root'), deep_rebuild=True)
    ctx = deep_step.Ctx(args, run, ['residuals', 'verify'])
    plan = deep_step.Plan('07h_18MHz', 'V')
    c = deep_cells.Cell('07h_18MHz', root)
    excl = exclusions.load()
    assert not any(excl.is_excluded('07h', '18MHz', dd) for dd in SYN_DATES)
    owed = list(SYN_DATES)
    ctx.flow_root = os.path.join(d, 'work')
    u = run.unit(plan.name, tier='deep', stage='residuals')
    with u:
        u.meta['coadd_root'] = root                              # as _residuals_unit records it (verify reads it back)
        for dd in owed:
            u.plan(spec.frame_path(dd, 'V'), 'search_frame')
            u.plan(spec.frame_path(dd, 'V', True), 'search_frame')
        deep_step.register(u, ctx, plan, c, spec, res, owed, excl)
    bad = [t for t in u.tests if not t['passed']]
    assert u.status == 'done', (u.status, u.reason, bad[:3])
    gates = sorted({r['gate'] for r in u.refused_items})
    assert gates == ['nightimage', 'unaccounted-epoch'], gates
    names = {t['name'] for t in u.tests}
    for n in ('search_frame.pbcorr_matches_plain', 'search_frame.master_used', 'search_frame.copy_identical',
              'refusal_record.parses', 'frame_count', 'archive_copies', 'removed_epochs_have_no_frame.nightimage',
              'engine_reported_set', 'exclusion_conformance'):
        assert n in names, n
    classes = sorted({o['class'] for o in u.outputs})
    assert classes == ['refusal_record', 'search_frame'], classes
    # the record of the nights data_decisions excluded: a frame build's refusal record, registered and content-tested, no
    # search product of a frames_only attempt - and no whole-epoch gate: d4, recorded and owed now, is still an unaccounted
    # epoch (the gates above; as a whole-epoch kind it would be refused as 'datadecisions')
    ddp = os.path.abspath(spec.refusal_path('datadecisions'))
    assert any(o['class'] == 'refusal_record' and o['path'] == ddp for o in u.outputs), ddp
    assert not any('unexpected search products' in n for n in u.notes), u.notes
    assert len([o for o in u.outputs if o['class'] == 'search_frame']) == 8          # 2 frames + 2 siblings + 4 copies
    rec = record.load_record(out, 'frames', plan.name)
    assert rec['status'] == 'done' and len(rec['refused_items']) == 4
    fc = next(t for t in u.tests if t['name'] == 'frame_count')
    assert '1 removed per the gate records' in fc['detail'] and '1 unaccounted' in fc['detail'], fc
    # verify stage re-tests exactly the recorded outputs (and the exclusions now)
    deep_step._verify_unit(ctx, plan)
    v = run.units[-1]
    assert v.status == 'done' and len(v.outputs) == len(rec['outputs']), (v.status, v.reason)
    assert any(t['name'] == 'exclusion_conformance' and t['passed'] for t in v.tests)
    # frame_count is checked against the gate records, not against the engine's own rows: a night the night-image record
    # FAILs although the engine made its frame (and wrote no row) refuses the unit
    _syn_nightimage(os.path.join(root, '07h_18MHz', 'coadd_final'), fail=(SYN_DATES[1], SYN_DATES[2]))
    u2 = run.unit(plan.name + '_neg', tier='deep', stage='residuals')
    with u2:
        for dd in owed:
            u2.plan(spec.frame_path(dd, 'V'), 'search_frame')
            u2.plan(spec.frame_path(dd, 'V', True), 'search_frame')
        deep_step.register(u2, ctx, plan, c, spec, res, owed, excl)
    failed2 = {t['name'] for t in u2.tests if not t['passed']}
    assert u2.status == 'refused' and {'frame_count', 'gate_records_agree'} <= failed2, (u2.status, failed2)
    _syn_nightimage(os.path.join(root, '07h_18MHz', 'coadd_final'), fail=(SYN_DATES[2],))
    # archive-quarantine: archive copies into a production archive root refuse (production quarantines first)
    ctx_p = deep_step.Ctx(_Args(out=out, work=os.path.join(d, 'work_root'), deep_rebuild=True), run, ['residuals'])
    assert ctx_p.archive_refusal() is None
    ctx_p.archive_root = CFG.paths.PHASE3_ARCHIVE_ROOT
    assert ctx_p.archive_refusal().startswith('archive-quarantine'), ctx_p.archive_refusal()
    ctx_p.archive_copy = False
    assert ctx_p.archive_refusal() is None
    # negative copies: each content test fails on a corrupted file
    from astropy.io import fits
    import numpy as np
    f1 = spec.frame_path(SYN_DATES[0], 'V')
    neg = os.path.join(d, 'neg')
    os.makedirs(neg)
    cases = {
        'rms_cards': lambda h, x: h.__setitem__('IMGRMS', 9.0),
        'striwin': lambda h, x: h.__setitem__('STRIWIN', 1.0),
        'own_time': lambda h, x: h.__setitem__('MSTRMJD', h['MJD-MID']),
        'master_used': lambda h, x: h.__setitem__('LOOEXCL', '2024-01-09'),
        'prodname': lambda h, x: h.__setitem__('PRODNAME', 'unknown'),
        'stokes_card': lambda h, x: h.__setitem__('STOKES', 'I'),
    }
    for test, mut in cases.items():
        x = fits.getdata(f1).copy()
        h = fits.getheader(f1).copy()
        mut(h, x)
        p = os.path.join(neg, test, os.path.basename(f1))
        os.makedirs(os.path.dirname(p))
        fits.writeto(p, x, h)
        r = {t['name']: t['passed'] for t in validate.run('search_frame', p, stokes='V')}
        assert r[f'search_frame.{test}'] is False, (test, r)
    sib = spec.frame_path(SYN_DATES[0], 'V', True)
    p = os.path.join(neg, 'sib', os.path.basename(sib))
    os.makedirs(os.path.dirname(p))
    shutil.copy2(f1, os.path.join(neg, 'sib', os.path.basename(f1)))
    x = fits.getdata(sib).copy()
    x[100, 100] *= 1.01
    fits.writeto(p, x, fits.getheader(sib))
    r = {t['name']: t['passed'] for t in validate.run('search_frame', p,
                                                     pb_path=os.path.join(root, '07h_18MHz', 'coadd_final', 'PB_deep_18MHz.fits'))}
    assert r['search_frame.pbcorr_matches_plain'] is False, r
    # noise map: a synthetic sigma stack built the engine's way passes; one with a pixel below the floor fails
    m = np.abs(np.random.default_rng(3).normal(1.0, 0.3, (64, 64)))
    q = 16
    m = m / np.nanmedian(m[q:3 * q, q:3 * q])
    m = np.maximum(m, 1.0).astype(np.float32)
    h = fits.Header()
    h['BUNIT'], h['PRODUCT'], h['STOKES'], h['DATE-OBS'], h['BMAJ'], h['BMIN'] = 'ratio', 'sigstack', 'V', '2024', 0.3, 0.3
    good = os.path.join(neg, '07h_18MHz_V_sigstack.fits')
    fits.writeto(good, m, h)
    assert all(t['passed'] for t in validate.run('noise_map', good)), validate.run('noise_map', good)
    m2 = m.copy()
    m2[0, 0] = 0.5
    badp = os.path.join(neg, 'bad', '07h_18MHz_V_sigstack.fits')
    os.makedirs(os.path.dirname(badp))
    fits.writeto(badp, m2, h)
    assert not {t['name']: t['passed'] for t in validate.run('noise_map', badp)}['noise_map.sigstack_floor']
    # the pbcorr-missing record
    pm = os.path.join(neg, 'pbcorr_missing_07h_18MHz.csv')
    with open(pm, 'w') as f:
        f.write(f'07h 18MHz,{SYN_DATES[0]},{SYN_DATES[0]}_V_redo_resid_18MHz.fits,/nonexistent/PB_deep_18MHz.fits\n')
    assert all(t['passed'] for t in validate.run('frame_refusals', pm))
    with open(pm, 'w') as f:
        f.write('07h 18MHz,2024-01-01,wrong.fits\n')
    assert not all(t['passed'] for t in validate.run('frame_refusals', pm))
    print(f'register: unit done ({len(u.outputs)} outputs, {len(u.tests)} tests, item refusals {gates}); verify re-tests '
          f'the record; negative copies fail their content tests')


def test_cli():
    d = _fresh('cli')
    work = os.path.join(d, 'work')
    rc, o, e = _cli(['frames', '--tier', 'deep', '--cells', '07h_18MHz', '--stokes', 'V', '--deep-rebuild', '--cores',
                     'none', '--work', work])
    assert rc in (1, 2) and 'PRODUCTION_WRITES' in e, (rc, e[-400:])          # cli._guard_out or the step
    rc, o, e = _cli(['frames', '--tier', 'deep', '--cells', '07h_18MHz', '--stokes', 'V', '--cores', 'none', '--out',
                     os.path.join(d, 'out1'), '--work', work])
    assert rc == 2, (rc, e[-400:])
    rec = json.load(open(os.path.join(d, 'out1', '_records', 'frames', '07h_18MHz_V.json')))
    assert rec['status'] == 'refused' and rec['reason'].startswith('deep-producer'), rec['reason']
    rc, o, e = _cli(['frames', '--tier', 'deep', '--cells', '07h_18MHz', '--stokes', 'V', '--deep-rebuild', '--cores',
                     'none', '--out', os.path.join(d, 'out2'), '--work', work])
    rec = json.load(open(os.path.join(d, 'out2', '_records', 'frames', '07h_18MHz_V.json')))
    from exopipe.phase3.lib.frames import deep_engine
    if not deep_engine.engine_has_frames_only():
        assert rc == 2 and rec['status'] == 'refused' and rec['reason'].startswith('engine-seam'), rec['reason']
        assert not os.path.exists(os.path.join(d, 'out2', 'deep_attempts')), 'an attempt was created before the gate'
        passed = [t['name'] for t in rec['tests'] if t['passed']]
        assert 'pregate.D-G3' in passed and 'pregate.D-G12' in passed, passed
    rc, o, e = _cli(['frames', '--tier', 'deep', '--cells', '07h_18MHz', '--deep-rebuild', '--cores', 'none', '--out',
                     os.path.join(d, 'out3'), '--work', work, '--set', 'search_detect.SIGMA_STACK_BUILD_MIN=4'])
    assert rc == 2, (rc, e[-300:])
    print('cli: no --out, no --deep-rebuild, no frames_only keyword, protected literal: each exit 2, nothing written')


def test_real():
    from exopipe.phase3.core import validate
    from exopipe.phase3.lib.frames import deep_cells
    files = [(os.path.join(REAL_ATT_V, 'Run_P3S5/vresid/07h_18MHz/2025-01-02_V_redo_resid_18MHz.fits'), 'search_frame'),
             (os.path.join(REAL_ATT_V, 'Run_P3S5/vresid/07h_18MHz/2024-12-27_V_redo_resid_pbcorr_18MHz.fits'), 'search_frame'),
             (os.path.join(REAL_ATT_I, 'Run_P3S5/iresid/07h_18MHz/2025-01-03_I_redo_resid_18MHz.fits'), 'search_frame'),
             (os.path.join(REAL_ARCH, '18MHz_V_deep_Taper_Robust+0.0_shflux_mdewarp_pbcorr_dewarped_aligned_msub_ssub_LST07h_20250102T080414.fits'),
              'search_frame'),
             (os.path.join(REAL_ATT_V, 'transients/Run_P3S5/sigstack/07h_18MHz_V_sigstack.fits'), 'noise_map'),
             (os.path.join(REAL_ATT_I, 'transients/Run_P3S5/sigstack/07h_18MHz_I_sigstack.fits'), 'noise_map'),
             (os.path.join(REAL_CELL_ROOT, '07h_18MHz/subtracted/2025-01-02_I_epoch_18MHz.fits'), 'epoch_image')]
    lines = []
    c = deep_cells.Cell('07h_18MHz', REAL_CELL_ROOT)
    for p, cls in files:
        if not os.path.exists(p):
            lines.append(f'MISSING {p}')
            continue
        kw = {}
        if cls == 'search_frame' and os.path.basename(p)[:4].isdigit():
            from exopipe.phase3.lib.frames.deep_step import frame_ctx
            st = 'V' if '_V_' in os.path.basename(p) else 'I'
            kw = frame_ctx(c, st, os.path.basename(p)[:10])
        res = validate.run(cls, p, **kw)
        bad = [r['name'] for r in res if not r['passed']]
        lines.append(f"{'PASS' if not bad else 'FAIL ' + ','.join(bad)} {cls} ({len(res)} tests) {os.path.basename(p)}")
        assert not bad, (p, [r for r in res if not r['passed']])
    d = _fresh('real')
    rc, o, e = _cli(['frames', '--tier', 'deep', '--stages', 'verify', '--deep-adopt', REAL_ATT_V, '--cores', 'none',
                     '--out', os.path.join(d, 'adopt')])
    assert rc == 0, (rc, e[-500:])
    rec = json.load(open(os.path.join(d, 'adopt', '_records', 'frames', '07h_18MHz_V_adopt_20261012T212735_verify.json')))
    assert rec['status'] == 'done' and all(t['passed'] for t in rec['tests'])
    assert {'attempt_after_current_masters', 'exclusion_conformance', 'frame_count'} <= {t['name'] for t in rec['tests']}
    # an attempt made before the cell's current masters (2026-10-05 < masters 2026-10-10) is never adopted (fleet D6)
    old_att = REAL_ATT_V.replace('20261012T212735', '20261005T090523')
    if os.path.isdir(old_att):
        rc, o, e = _cli(['frames', '--tier', 'deep', '--stages', 'verify', '--deep-adopt', old_att, '--cores', 'none',
                         '--out', os.path.join(d, 'adopt_old')])
        rec = json.load(open(os.path.join(d, 'adopt_old', '_records', 'frames',
                                          '07h_18MHz_V_adopt_20261005T090523_verify.json')))
        bad = {t['name'] for t in rec['tests'] if not t['passed']}
        assert rc == 2 and rec['status'] == 'refused' and 'attempt_after_current_masters' in bad, (rc, sorted(bad))
        lines.append('adopt 07h_18MHz_V_20261005T090523 (before the current masters): refused attempt_after_current_masters')
    lines.append(f"adopt {os.path.basename(REAL_ATT_V)}: {len(rec['outputs'])} outputs, {len(rec['tests'])} tests pass")
    rc, o, e = _cli(['frames', '--tier', 'deep', '--cells', '07h_18MHz', '--stokes', 'V', '--dry-run', '--cores', 'none',
                     '--out', os.path.join(d, 'dry'), '--work', os.path.join(d, 'work')])
    assert rc == 2 and 'owed frames' in o and 'PASS D-G3' in o and 'REFUSE deep-producer' in o, (rc, o[-800:], e[-300:])
    assert '18 against their LOO V master, 1 against the central master' in o, o
    rc, o, e = _cli(['frames', '--tier', 'deep', '--epochs', '07h_2025-10-15', '--scopes', 'Red', '--stokes', 'V',
                     '--deep-rebuild', '--dry-run', '--cores', 'none', '--out', os.path.join(d, 'dry'), '--work',
                     os.path.join(d, 'work')])
    assert "nights outside the master ['2025-10-15']" in o, o[-800:]
    assert not os.path.exists(os.path.join(d, 'dry')), 'the dry run wrote something'
    lines.append('dry-run 07h_18MHz V: 19 owed frames (18 LOO, 1 central); per-hour 07h_2025-10-15 Red: central master')
    print('real:\n  ' + '\n  '.join(lines))


TESTS = ['test_config', 'test_seam', 'test_items', 'test_overrides', 'test_preview_sanity_minimum', 'test_child', 'test_register', 'test_cli', 'test_real']


def main(argv):
    names = argv or TESTS
    t0 = time.time()
    os.makedirs(SCR, exist_ok=True)
    failed = []
    try:
        for n in names:
            t = time.time()
            try:
                globals()[n]()
                print(f'  ok  {n} ({time.time() - t:.1f} s)')
            except Exception:
                import traceback
                traceback.print_exc()
                failed.append(n)
                print(f'  FAIL {n}')
    finally:
        shutil.rmtree(SCR, ignore_errors=True)
    print(f'{len(names) - len(failed)} of {len(names)} tests passed in {time.time() - t0:.1f} s'
          + (f'; FAILED: {failed}' if failed else ''))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
