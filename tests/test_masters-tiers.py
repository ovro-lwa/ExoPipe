#!/usr/bin/env python3
"""STANDING TEST - unit masters-tiers: `exopipe3 masters --tier sublst|snapshot` (lib/masters/tier_sublst.py, tier_snapshot.py,
tests_sublst.py, tests_snapshot.py, config/steps/masters.py).

Synthetic part (a fresh directory under /tmp/exopipe3_tests_masters-tiers/, removed at the end):
  config      sections load; every RULED ledger ID is a row of the fix ledger; PROTECTED_KEYS refuse an override, an
              unprotected key takes one; interpreters are site keys of the refactor vocabulary; no lower-case markers
  grammar     Selection from --cells / --sublst-worklist (core.cells night cells), admit key, band scope refused
  outcome     done / refused / error classification of engine output (a crash after REFUSED-WINDOW lines is an error)
  prep copy   cpv: copied, identical destination kept
  stack unit  _stack_unit with a FAKE stacker run through core.legacy.run: a done unit whose 39 synthetic products pass
              every content test; a refused unit (both Stokes refused -> unit refused); an engine error (unit error);
              G-S01 (earlier products refuse) and --fresh --supersede (moved with a sha1 manifest)
  loo         G-S05 stack gate (done only, or done/refused with LOO_ADMITS_REFUSED_STACKS); window master + LOO residual
              content tests on products made with the loo formula, and the identity test failing on a perturbed residual
  snapshot    verify-frozen slot and PB-library units on a synthetic frozen tree: done when md5 and links are right,
              refused on an md5 mismatch and on a link into the archive Coadd/ tree; the registry tests re-hash
Read-only part on real production (no write outside the scratch directory):
  the in-place literals equal their config mirrors (both tiers); content tests on production sub-LST products; the
  per-hour interface stack_night() in --dry-run on 09h 2024-12-29 (plan = production's job list for that night).

Run standalone:  python tests/test_masters-tiers.py [test_name ...]
"""
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')
SCRATCH_PARENT = '/tmp/exopipe3_tests_masters-tiers'
PROD_SL = '/lustre/gh/main/phase3/sublst'
T_INT = 10.03102041

from exopipe.phase3.core import env as _env       # noqa: E402
_env.legacy()
from exopipe.phase3.config import CFG, ConfigRefusal   # noqa: E402
from exopipe.phase3.core import record, validate       # noqa: E402
from exopipe.phase3.lib.masters import tier_sublst as TS, tier_snapshot as TN   # noqa: E402

_STATE = {}


def scratch():
    if 'dir' not in _STATE:
        os.makedirs(SCRATCH_PARENT, exist_ok=True)
        _STATE['dir'] = tempfile.mkdtemp(prefix='synth_', dir=SCRATCH_PARENT)
    return _STATE['dir']


class Patch:
    """Set attributes for the duration of a with-block (module functions, config section values)."""

    def __init__(self, *triples):
        self.triples, self.old = triples, []

    def __enter__(self):
        for obj, name, val in self.triples:
            self.old.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)
        return self

    def __exit__(self, *exc):
        for obj, name, val in reversed(self.old):
            setattr(obj, name, val)


def new_run(sub, dry=False):
    out = os.path.join(scratch(), sub)
    os.makedirs(out, exist_ok=True)
    return record.Run('masters', ['test'], out, CFG, dry_run=dry)


# =====================================================================================================================
# config
# =====================================================================================================================

def test_config():
    from exopipe.phase3.core import ledger
    for sec in ('masters_sublst', 'masters_snapshot'):
        s = getattr(CFG, sec)
        assert s.items(), sec
        for k, lid in s.RULED.items():
            assert hasattr(s, k), (sec, k)
            assert ledger.has_id(lid, CFG.paths.FIX_LEDGER), (sec, k, lid)
        for k in s.PROTECTED_KEYS:
            assert hasattr(s, k), (sec, k)
            assert CFG.protected(f'{sec}.{k}'), (sec, k)
        for marker in ('_apply', '_protected', 'protected', 'locked', 'mirror', 'lit', 'window_locked'):
            assert marker not in vars(s), (sec, marker)
    for e in ('stack', 'loo', 'sources', 'test_windows'):
        v = TS.python_key(e)
        assert v in ('default', 'flow', 'py36', 'casa', 'tess'), (e, v)
        assert v in CFG.site.PYTHON, (e, v)
    try:
        CFG.set('masters_sublst.BEAM_AREA_MAX', '2.0')
        raise AssertionError('a protected mirror key took an override')
    except ConfigRefusal:
        pass
    try:
        CFG.set('masters_snapshot.FROZEN_MANIFEST_SHA1', 'x')
        raise AssertionError('a protected pin took an override')
    except ConfigRefusal:
        pass
    CFG.set('masters_sublst.FLUXDC_WAIT_S', '60')
    assert CFG.masters_sublst.FLUXDC_WAIT_S == 60
    CFG.reset()
    assert CFG.masters_sublst.FLUXDC_WAIT_S == 14400
    # one name per shared root: the tier reads paths.*, the frozen tree from masters_snapshot
    assert TS.sublst_root() == CFG.paths.SUBLST_ROOT and TS.cube_root() == CFG.paths.SNAPSHOT_CUBES_ROOT
    assert TS.frozen_masters() == CFG.masters_snapshot.FROZEN_ROOT
    for gone in ('ROOT', 'CUBE_ROOT', 'FROZEN_MASTERS', 'WIDEBAND_GROUPS'):
        assert not hasattr(CFG.masters_sublst, gone), gone


# =====================================================================================================================
# grammar, outcome, prep copy
# =====================================================================================================================

class A:
    def __init__(self, **kw):
        self.__dict__.update(dict(cells=None, epochs=None, lsts=None, nights=None, scopes=None, stokes=None,
                                  sublst_worklist=None, exclude_cells=None), **kw)


def test_grammar():
    sel = TS.Selection.from_args(A(cells='09h_2024-12-29_18MHz,09h_2024-12-29_23MHz,10h_2025-01-01'))
    assert sel.lsts == ['09h', '10h'], sel.lsts
    assert sel.cells_of('09h') == ['09h_2024-12-29'] and sel.scope_ok('09h_2024-12-29', '18MHz')
    assert not sel.scope_ok('09h_2024-12-29', '41MHz') and sel.scope_ok('10h_2025-01-01', '41MHz')
    assert sel.admit_key('09h') == '09h_2024-12-29'
    wl = os.path.join(scratch(), 'wl.txt')
    open(wl, 'w').write('# worklist\n09h 2024-12-29\n09h 2024-12-31\nbad line here\n')
    sel = TS.Selection.from_args(A(sublst_worklist=wl, stokes='V'))
    assert sel.cells_of('09h') == ['09h_2024-12-29', '09h_2024-12-31'] and sel.admit_key('09h') == '09h'
    assert sel.stokes == ['V']
    sel = TS.Selection.from_args(A(sublst_worklist=wl, exclude_cells='09h_2024-12-31'))
    assert sel.cells_of('09h') == ['09h_2024-12-29'] and sel.admit_key('09h') == '09h_2024-12-29'   # one night left
    for bad in ('09h_2024-12-29_Red', '9h_2024-12-29', '09h_20241229'):
        try:
            TS.Selection.from_args(A(cells=bad))
            raise AssertionError(f'{bad} accepted')
        except SystemExit:
            pass


def test_outcome():
    ok = ['WINDOW x', 'SUBLST 09h_2024-12-29 18MHz I frames=358 used=352 windows=6']
    assert TS.outcome(0, ok, r'^SUBLST 09h_2024-12-29 18MHz I ')[0] == 'done'
    st, why = TS.outcome(1, ['WARNING: ErfaWarning', 'REFUSED: only 3 of 300 frames pass'], r'^SUBLST ')
    assert st == 'refused' and why.startswith('REFUSED'), (st, why)
    st, _ = TS.outcome(1, ['RuntimeError: REFUSING: tier-rule-3 pin missing'], r'^SUBLST ')
    assert st == 'refused'
    crash = ['REFUSED-WINDOW 18MHz I w0: 1 night(s)', 'Traceback (most recent call last):', 'MemoryError']
    assert TS.outcome(1, crash, r'^SUBLST-LOO ')[0] == 'error'
    assert TS.outcome(0, ['no completion line'], r'^SUBLST ')[0] == 'error'
    killed = ['WINDOW-MASTER 18MHz I w0 nights=3 residuals=3', 'REFUSED-WINDOW 18MHz I w1: 1 night(s) - no leave-one-out']
    assert TS.outcome(-9, killed, r'^SUBLST-LOO ')[0] == 'error'          # killed after a progress line: not a refusal
    assert TS.outcome(1, ['RuntimeError: REFUSED (--tier1min-only): x'], r'^SUBLST ')[0] == 'refused'
    buffered = ['REFUSED: window 3 stacks differ in shape', 'REFUSED-WINDOW 18MHz I w1: 1 night(s)']   # stdout flushed last
    assert TS.outcome(1, buffered, r'^SUBLST-LOO ') == ('refused', 'REFUSED: window 3 stacks differ in shape')


def test_cpv():
    d = scratch()
    src, dst = os.path.join(d, 'src.bin'), os.path.join(d, 'sub', 'dst.bin')
    open(src, 'wb').write(os.urandom(4096))
    m, what = TS.cpv(src, dst)
    assert what == 'copied' and m == TS.md5_file(dst) and not os.path.exists(dst + '.part')
    assert TS.cpv(src, dst)[1] == 'kept'


# =====================================================================================================================
# stack unit with a fake engine (through core.legacy.run)
# =====================================================================================================================

FAKE_STACKER = r'''
import argparse, csv, json, os, sys
import numpy as np
from astropy.io import fits
from astropy.time import Time
T = 10.03102041
ap = argparse.ArgumentParser()
for k in ('--cube-root', '--cell', '--subband', '--stokes', '--out'):
    ap.add_argument(k)
a = ap.parse_args()
if a.subband == '82MHz':
    sys.exit('REFUSED: synthetic refusal of %s %s' % (a.subband, a.stokes))
if a.subband == '78MHz':
    raise ValueError('synthetic engine crash')
lst, night = a.cell.split('_', 1)
mp = os.environ['FAKE_MASTER']
mh = fits.getheader(mp)
B = float(mh['SHSCALE']) * float(mh['FLUXCORR'])
fdc = 0.8
cols = ['ts', 'ts_int', 'date_obs', 't_mid_unix', 'lst_min', 'window', 'finite_frac', 'beam_area_deg2', 'noise_jyb',
        'sun_elev_deg', 'sun_branch', 'used', 'refused']
od = os.path.join(a.out, a.cell, a.subband, a.stokes)
os.makedirs(od, exist_ok=True)
t0 = Time(night + 'T10:00:00', format='isot', scale='utc').unix
rows = []
rng = np.random.RandomState(1)
for k in range(6):
    sel = []
    for i in range(5):
        ts = t0 + k * 600 + i * T
        do = Time(ts, format='unix', scale='utc').isot[:21]
        tm = float('%.3f' % (Time(do, format='isot', scale='utc').unix + T / 2))
        nz = float('%.6g' % (2.0 + 0.1 * i + 0.01 * k))
        r = dict(ts='x%d%d' % (k, i), ts_int='x', date_obs=do, t_mid_unix='%.3f' % tm, lst_min='%.3f' % (k * 10 + i),
                 window=k, finite_frac='1.000', beam_area_deg2='0.2', noise_jyb='%.6g' % nz, sun_elev_deg='-50.00',
                 sun_branch='morning', used=1, refused='')
        rows.append(r)
        sel.append(r)
    ts_ = np.array([float(r['t_mid_unix']) for r in sel]); nz_ = np.array([float(r['noise_jyb']) for r in sel])
    h = fits.Header()
    h['DATE-OBS'] = sel[0]['date_obs']
    h['DATE-END'] = Time(Time(sel[-1]['date_obs'], format='isot', scale='utc').unix + T, format='unix', scale='utc').isot
    h['MJD-MID'] = Time(float(np.average(ts_, weights=1.0 / nz_ ** 2)), format='unix', scale='utc').mjd
    h['FTBASIS'] = 'DATE-OBS+T/2'; h['LSTHOUR'] = lst; h['LSTWIN'] = k
    h['LSTW0'] = float(k * 10); h['LSTW1'] = float(k * 10 + 4)
    h['NFRAMES'] = 5; h['TEFFSEC'] = round(T * 5, 3)
    h['NOISEMED'] = float(np.median(nz_)) / fdc * B
    h['FLUXDC'] = fdc
    for c in ('SHSCALE', 'FLUXCORR', 'FLUXBAKE'):
        h[c] = mh[c]
    h['PRODUCT'] = 'sublst-v0-test'
    base = os.path.join(od, 'sublst_%s_%s_%s_w%d' % (a.cell, a.subband, a.stokes, k))
    fits.writeto(base + '.fits', rng.normal(size=(16, 16)).astype(np.float32), h, overwrite=True)
    hn = fits.Header()
    for c in ('LSTHOUR', 'LSTWIN', 'LSTW0', 'LSTW1', 'NFRAMES'):
        hn[c] = h[c]
    fits.writeto(base + '_nframes.fits', np.full((16, 16), 5, dtype=np.int16), hn, overwrite=True)
    hp = hn.copy(); hp['PBWIN0'] = h['LSTW0'] - T / 120.0; hp['PBWIN1'] = h['LSTW1'] + T / 120.0
    fits.writeto(base + '_pb.fits', np.linspace(0.05, 0.95, 256, dtype=np.float32).reshape(16, 16), hp, overwrite=True)
with open(os.path.join(od, 'admission_%s_%s_%s.csv' % (a.cell, a.subband, a.stokes)), 'w') as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
if a.stokes == 'I':
    ci = os.path.join(a.cube_root, a.cell, '%s_%s_I_flowcube.fits' % (a.cell, a.subband))
    st, sm = os.stat(ci), os.stat(mp)
    rec = dict(cube_I=ci, cube_I_size=st.st_size, cube_I_mtime=int(st.st_mtime), master=mp, master_size=sm.st_size,
               master_mtime=int(sm.st_mtime), cell=a.cell, subband=a.subband, flux_dc=fdc, n_src=50, n_frames=30, B=B)
    json.dump(rec, open(os.path.join(a.out, a.cell, a.subband, 'I', 'fluxdc_%s_%s.json' % (a.cell, a.subband)), 'w'))
print('unbuffered=%s' % os.environ.get('PYTHONUNBUFFERED', ''))
print('SUBLST %s %s %s frames=30 used=30 blank=0 beam=0 noise=0 outside=0 windows=6' % (a.cell, a.subband, a.stokes))
'''


def _synthetic_inputs(d, lst, night, sbs):
    import numpy as np
    from astropy.io import fits
    cell = f'{lst}_{night}'
    croot = os.path.join(d, 'cubes')
    for sb in sbs:
        for S in ('I', 'V'):
            p = os.path.join(croot, cell, f'{cell}_{sb}_{S}_flowcube.fits')
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, 'wb').write(b'cube')
    mp = os.path.join(d, 'master_I_deep_Taper_R0.fits')
    h = fits.Header()
    h['SHSCALE'], h['FLUXCORR'], h['FLUXBAKE'] = 0.98, 0.94, True
    fits.writeto(mp, np.zeros((16, 16), np.float32), h, overwrite=True)
    wp = os.path.join(d, 'frozen_weights.csv')
    open(wp, 'w').write(f'date,weight,in_master,qa3_scale\n{night},0.2,1,0.9\n')
    fake = os.path.join(d, 'fake_stacker.py')
    open(fake, 'w').write(FAKE_STACKER)
    return croot, mp, wp, fake


def test_stack_unit():
    d = os.path.join(scratch(), 'stack')
    os.makedirs(d, exist_ok=True)
    lst, night = '09h', '2030-01-01'                     # a night data_decisions.csv does not know: never excluded
    cell = f'{lst}_{night}'
    croot, mp, wp, fake = _synthetic_inputs(d, lst, night, ('18MHz', '82MHz', '78MHz'))
    run = new_run('stack_out')
    os.environ['FAKE_MASTER'] = mp
    CFG.set('masters.PYTHON_SITE_KEY[sublst-stack]', 'default')
    try:
        with Patch((TS, 'master_path', lambda l, sb: mp), (TS, 'frozen_weights', lambda l, sb: wp),
                   (TS, 'pins_ok', lambda: ([], [])), (TS, 'script', lambda short: fake)):
            opts = TS.Opts()
            for sb in ('18MHz', '82MHz', '78MHz'):
                try:                       # an engine error is recorded (status error) and re-raised, as for any unit
                    TS._stack_unit(run, opts, run.out_root, lst, cell, sb, ['I', 'V'], croot, [])
                except RuntimeError as e:
                    assert sb == '78MHz' and 'synthetic engine crash' in str(e), (sb, e)
            by = {u.name: u for u in run.units}
            u = by[f'stack_{lst}_{night}_18MHz']
            assert u.status == 'done', (u.status, u.reason, [t for t in u.tests if not t['passed']][:3])
            assert len(u.outputs) == 39, len(u.outputs)
            classes = {o['class'] for o in u.outputs}
            assert classes == {'sublst_window', 'sublst_window_nframes', 'sublst_window_pb', 'sublst_admission',
                               'sublst_fluxdc_record'}, classes
            assert all(t['passed'] for t in u.tests) and len(u.tests) > 39
            assert u.meta.get('legacy_calls') and len(u.meta['legacy_calls']) == 2      # core.legacy.run used, I then V
            log = open(os.path.join(run.out_root, lst, 'logs', f'{cell}_18MHz_I.log')).read()
            assert 'unbuffered=1' in log, log[-300:]                                # production's python -u
            rec = record.load_record(run.out_root, 'masters', f'stack_{lst}_{night}_18MHz')
            assert rec['status'] == 'done' and rec['admitted'] == [f'{night}/18MHz']
            r82 = by[f'stack_{lst}_{night}_82MHz']
            assert r82.status == 'refused' and 'synthetic refusal' in r82.reason, (r82.status, r82.reason)
            assert len(r82.refused_items) == 3                                     # I admission + fluxdc, V admission
            r78 = by[f'stack_{lst}_{night}_78MHz']
            assert r78.status == 'error' and 'synthetic engine crash' in r78.reason, (r78.status, r78.reason)
            # G-S01: earlier products refuse; --fresh --supersede moves them aside with a sha1 manifest
            run2 = new_run('stack_out')
            TS._stack_unit(run2, TS.Opts(), run2.out_root, lst, cell, '18MHz', ['I', 'V'], croot, [])
            assert run2.units[0].status == 'refused' and 'G-S01' in run2.units[0].reason, run2.units[0].reason
            TS._stack_unit(run2, TS.Opts(fresh=True, supersede='t1'), run2.out_root, lst, cell, '18MHz', ['I', 'V'],
                           croot, [])
            assert run2.units[1].status == 'done', (run2.units[1].status, run2.units[1].reason)
            man = os.path.join(run2.out_root, '_superseded', 't1', 'MANIFEST.csv')
            rows = list(csv.DictReader(open(man)))
            assert len(rows) == 39 and all(len(r['sha1']) == 40 for r in rows), len(rows)
            # a content test catches a damaged product: NFRAMES no longer the used admission rows
            from astropy.io import fits
            w = os.path.join(run2.out_root, lst, cell, '18MHz', 'I', f'sublst_{cell}_18MHz_I_w0.fits')
            with fits.open(w, mode='update') as h:
                h[0].header['NFRAMES'] = 4
            res = {r['name']: r['passed'] for r in validate.run('sublst_window', w, master=mp)}
            assert not res['sublst_window.window_matches_admission'] and not res['sublst_window.window_header_census']
    finally:
        CFG.reset()
        os.environ.pop('FAKE_MASTER', None)


def test_stack_gate():
    run = new_run('gate_out')
    jobs = [('09h_2030-01-01', '18MHz', ['I', 'V'], '/x'), ('09h_2030-01-02', '18MHz', ['I', 'V'], '/x')]
    record.atomic_write_json(record.record_path(run.out_root, 'masters', 'stack_09h_2030-01-01_18MHz'), {'status': 'done'})
    record.atomic_write_json(record.record_path(run.out_root, 'masters', 'stack_09h_2030-01-02_18MHz'), {'status': 'refused'})
    bad, refused = TS.stack_gate(run, '09h', '18MHz', jobs)
    assert len(bad) == 2 and refused == [('09h_2030-01-02', 'I'), ('09h_2030-01-02', 'V')], (bad, refused)
    with Patch((CFG.masters_sublst, 'LOO_ADMITS_REFUSED_STACKS', True)):
        bad, refused = TS.stack_gate(run, '09h', '18MHz', jobs)
        assert bad == [] and refused == [('09h_2030-01-02', 'I'), ('09h_2030-01-02', 'V')]
    bad, _ = TS.stack_gate(run, '09h', '23MHz', jobs + [('09h_2030-01-01', '23MHz', ['I'], '/x')])
    assert bad == ['09h_2030-01-01 23MHz I (missing)'], bad
    # review 2026-10-13: a DONE unit whose Stokes V the stacker refused (item refusal) is not complete - production stops
    # the LST when any job lacks its completion line (nocat:208-218)
    adm_v = os.path.join(run.out_root, '09h', '09h_2030-01-03', '27MHz', 'V', 'admission_09h_2030-01-03_27MHz_V.csv')
    record.atomic_write_json(record.record_path(run.out_root, 'masters', 'stack_09h_2030-01-03_27MHz'),
                             {'status': 'done', 'refused_items': [{'path': adm_v, 'reason': 'REFUSED: x', 'gate': 'g'}]})
    j27 = [('09h_2030-01-03', '27MHz', ['I', 'V'], '/x')]
    bad, refused = TS.stack_gate(run, '09h', '27MHz', j27)
    assert bad == ['09h_2030-01-03 27MHz V (refused)'] and refused == [('09h_2030-01-03', 'V')], (bad, refused)
    # a stack unit recorded under the frames step (per hour, stack_night) is found; the newest record wins
    record.atomic_write_json(record.record_path(run.out_root, 'frames', 'stack_09h_2030-01-04_32MHz'),
                             {'status': 'done', 'finished': '2026-10-13T10:00:00Z'})
    record.atomic_write_json(record.record_path(run.out_root, 'masters', 'stack_09h_2030-01-04_32MHz'),
                             {'status': 'refused', 'finished': '2026-10-13T09:00:00Z'})
    bad, _ = TS.stack_gate(run, '09h', '32MHz', [('09h_2030-01-04', '32MHz', ['I', 'V'], '/x')])
    assert bad == [], bad


FAKE_LOO = r'''
import argparse, sys
ap = argparse.ArgumentParser()
for k in ('--root', '--master-root', '--lst', '--subband', '--stokes'):
    ap.add_argument(k)
a = ap.parse_args()
if a.stokes == 'V':
    sys.exit('REFUSED: synthetic loo refusal')
print('SUBLST-LOO %s %s %s windows=0 residuals=0 refused_night=0' % (a.lst, a.subband, a.stokes))
'''


def test_loo_gates():
    """G-S05 admission count refuses BEFORE the engine runs (nocat:210-211); a loo engine refusal of one Stokes refuses the
    unit (nocat:232-233: the LST stops unless all 30 SUBLST-LOO lines), the other Stokes still runs."""
    d = os.path.join(scratch(), 'loo_gates')
    os.makedirs(d, exist_ok=True)
    fake = os.path.join(d, 'fake_loo.py')
    open(fake, 'w').write(FAKE_LOO)
    wp = os.path.join(d, 'w.csv')
    open(wp, 'w').write('date,weight\n')
    run = new_run('loo_gates_out')
    lst, sb, cell = '09h', '18MHz', '09h_2030-01-01'
    sdir = os.path.join(run.out_root, lst, cell, sb)
    for S in ('I', 'V'):
        os.makedirs(os.path.join(sdir, S), exist_ok=True)
        open(os.path.join(sdir, S, f'sublst_{cell}_{sb}_{S}_w0.fits'), 'wb').write(b'x')
    open(os.path.join(sdir, 'I', f'admission_{cell}_{sb}_I.csv'), 'w').write('ts\n')
    record.atomic_write_json(record.record_path(run.out_root, 'masters', f'stack_{lst}_2030-01-01_{sb}'), {'status': 'done'})
    jobs = [(cell, sb, ['I', 'V'], '/x')]
    CFG.set('masters.PYTHON_SITE_KEY[sublst-loo]', 'default')
    try:
        with Patch((TS, 'frozen_weights', lambda l, b: wp), (TS, 'pins_ok', lambda: ([], [])),
                   (TS, 'script', lambda short: fake)):
            TS._loo_unit(run, TS.Opts(), run.out_root, lst, sb, ['I', 'V'], jobs, [])
            u = run.units[-1]
            assert u.status == 'refused' and 'admission files' in u.reason, (u.status, u.reason)
            assert not os.path.exists(os.path.join(run.out_root, lst, 'logs', f'loo_{lst}_{sb}_I.log'))   # engine not run
            open(os.path.join(sdir, 'V', f'admission_{cell}_{sb}_V.csv'), 'w').write('ts\n')
            TS._loo_unit(run, TS.Opts(), run.out_root, lst, sb, ['I', 'V'], jobs, [])
            u = run.units[-1]
            assert u.status == 'refused' and u.reason.startswith('G-S06') and 'V: REFUSED' in u.reason, (u.status, u.reason)
            assert len(u.meta.get('legacy_calls', [])) == 2                  # both Stokes ran
    finally:
        CFG.reset()


def test_sources_prep_waivers():
    """sources: no subband under coadd_<LST> refuses (catpub pcheck exits 1, :60); prep: sources node-local to another
    host refuse (G-S08, sublst_prep_lst.sh:13 as a site check); never-waivable tests refuse a --waive."""
    from exopipe.phase3.core import env
    run = new_run('src_out')
    sel = TS.Selection()
    sel.lsts = ['09h']
    TS.stage_sources(run, TS.Opts(), sel, run.out_root)
    u = run.units[-1]
    assert u.status == 'refused' and 'G-S10' in u.reason, (u.status, u.reason)
    pairs, _, _ = TS.prep_pairs('09h', run.out_root)
    holder = env.node_local_holder(CFG.paths.COADD_SB_ROOT)
    assert holder, 'paths.COADD_SB_ROOT is not node-local in the site file'
    with Patch((env, 'host', lambda: holder)):
        assert TS.prep_locality_problem(pairs) == ''
    with Patch((env, 'host', lambda: 'otherhost')):
        why = TS.prep_locality_problem(pairs)
        assert why.startswith('G-S08') and holder in why, why
    r = record.Run('masters', ['t'], run.out_root, CFG, waivers=[('snapshot.G-D02_md5_equals_manifest', 'x')])
    assert 'never waivable' in TS.waiver_problem(r)
    assert TS.waiver_problem(record.Run('masters', ['t'], run.out_root, CFG, waivers=[('sublst_window.window_siblings', 'x')])) == ''


def test_admit_noinput():
    """G-S03: a night whose every job the rules leave out is NO-INPUT (exit 3; production goes on with 0 jobs), recorded with
    the reasons; the stack stage then records no-input too. A night with jobs writes the plan files (class sublst_job_plan)."""
    from exopipe.phase3.core import exclusions
    d = os.path.join(scratch(), 'admit')
    cubes = os.path.join(d, 'cubes')
    for night in ('2030-01-01', '2030-01-02'):
        os.makedirs(os.path.join(cubes, f'09h_{night}'), exist_ok=True)
        open(os.path.join(cubes, f'09h_{night}', 'MANIFEST.txt'), 'w').write('COMPLETE synthetic\n')

    def fake_plan(lst, cells, sbs=None):
        if cells == ['09h_2030-01-01']:
            return [], [], ['09h_2030-01-01 18MHz (admission rule: 0 of 358 frames pass the blank, beam and night gates)']
        return ['09h_2030-01-02 18MHz I', '09h_2030-01-02 18MHz V'], [], []
    with Patch((TS, 'cube_root', lambda: cubes), (TS, 'admit_plan', fake_plan), (TS, 'pins_ok', lambda: ([], [])),
               (TS, 'exclusion_readers_agree', lambda lst, sb, nights: (exclusions.load(), set()))):
        run = new_run('admit_out')
        sel = TS.Selection.one_night('09h', '2030-01-01', ['18MHz'])
        TS.stage_admit(run, TS.Opts(), sel)
        TS.stage_stack(run, TS.Opts(), sel, run.out_root, planned_now=True, lst_test=False)
        st = {u.name: (u.status, u.reason) for u in run.units}
        assert st['admit_09h_2030-01-01'][0] == 'no-input' and 'admission rule' in st['admit_09h_2030-01-01'][1], st
        assert st['stack_09h_2030-01-01'][0] == 'no-input', st
        assert run.exit_code() == record.EXIT_NO_INPUT
        run2 = new_run('admit_out')
        sel2 = TS.Selection.one_night('09h', '2030-01-02', ['18MHz'])
        TS.stage_admit(run2, TS.Opts(), sel2)
        u = run2.units[0]
        assert u.status == 'done' and len(u.outputs) == 2, (u.status, u.reason)
        jobs, files, how = TS.load_plan(run2, TS.Opts(), '09h', '09h_2030-01-02')
        assert jobs == [('09h_2030-01-02', '18MHz', ['I', 'V'], cubes)] and len(files) == 2, (jobs, files)
        open(files[0], 'a').write('09h_2030-01-02 23MHz I\n')               # the plan changed after its record
        try:
            TS.load_plan(run2, TS.Opts(), '09h', '09h_2030-01-02')
            raise AssertionError('a changed plan file was read')
        except record.Refusal as e:
            assert 'sha1' in str(e)


def test_loo_products():
    """Window master + LOO residuals made with the loo formula (loo:112-147, no Jupiter in the field) pass their content
    tests; a perturbed residual fails the identity test."""
    import numpy as np
    from astropy.io import fits
    d = os.path.join(scratch(), 'loo')
    lst, sb, S, k = '09h', '18MHz', 'I', 0
    nights = ['2030-01-01', '2030-01-02', '2030-01-03', '2030-01-04']
    w = [0.2, 0.3, 0.25, 0.0]                                     # the last night co-add-rejected (weight 0)
    rng = np.random.RandomState(3)
    imgs = [rng.normal(size=(24, 24)) for _ in nights]
    wp = os.path.join(d, 'frozen_weights.csv')
    os.makedirs(d, exist_ok=True)
    with open(wp, 'w') as f:
        f.write('date,weight,in_master\n' + ''.join(f'{n},{x},{1 if x > 0 else 0}\n' for n, x in zip(nights, w)))
    Ssum = sum(x * a for x, a in zip(w, imgs))
    W = sum(w)
    M = (Ssum / W).astype(np.float32)
    mroot = os.path.join(d, f'{lst}_masters')
    mdir = os.path.join(mroot, sb, S)
    os.makedirs(mdir, exist_ok=True)
    mh = fits.Header()
    mh['LSTHOUR'], mh['LSTWIN'], mh['NCOADD'] = lst, k, sum(1 for x in w if x > 0)
    mh['COADDOP'] = CFG.masters_sublst.WINDOW_COADDOP
    mh['JUPMASK'] = -1.0
    mh['CONSTDIG'] = hashlib.md5(','.join(sorted(nights)).encode()).hexdigest()
    for n, x in sorted(zip(nights, w)):
        mh['HISTORY'] = (f'window-Master constituent {n}' if x > 0 else f'NOT in the Master (deep co-add rejected) {n}')
    mp = os.path.join(mdir, f'sublst_master_{lst}_{sb}_{S}_w{k}.fits')
    fits.writeto(mp, M, mh, overwrite=True)
    with open(mp[:-5] + '_weights.csv', 'w') as f:
        f.write('night,weight\n' + ''.join(f'{n},{x}\n' for n, x in zip(nights, w)))
    resid = []
    nw = sum(1 for x in w if x > 0)
    for n, x, a in zip(nights, w, imgs):
        cell = f'{lst}_{n}'
        sdir = os.path.join(d, lst, cell, sb, S)
        os.makedirs(sdir, exist_ok=True)
        base = os.path.join(sdir, f'sublst_{cell}_{sb}_{S}_w{k}')
        h = fits.Header()
        h['DATE-OBS'], h['DATE-END'] = f'{n}T10:00:00.0', f'{n}T10:10:01.8'
        h['FLUXBAKE'] = True
        fits.writeto(base + '.fits', a.astype(np.float32), h, overwrite=True)
        loo = (Ssum - x * a) / (W - x)
        rh = h.copy()
        rh['SUBTRACT'] = 'leave-one-out window Master'
        rh['LOONCOAD'] = nw - (1 if x > 0 else 0)
        rh['LOOMSTR'] = os.path.basename(mp)
        rh['JUPMASK'] = -1.0
        rh['LSTBIN'] = k
        rh['DURATION'] = 601.8
        fits.writeto(base + '_resid_loo.fits', (a.astype(np.float32) - loo).astype(np.float32), rh, overwrite=True)
        resid.append(base + '_resid_loo.fits')
    with Patch((TS, 'frozen_weights', lambda l, b: wp)):
        r = validate.run('sublst_window_master', mp)
        assert all(t['passed'] for t in r), r
        for p in resid:
            r = validate.run('sublst_loo_residual', p, master_root=mroot)
            assert all(t['passed'] for t in r), (p, r)
        with fits.open(resid[0], mode='update') as h:
            h[0].data[:12] += 5.0
        r = validate.run('sublst_loo_residual', resid[0], master_root=mroot)
        assert not r[0]['passed'] and 'identity' in r[0]['detail'], r


# =====================================================================================================================
# snapshot tier on a synthetic frozen tree
# =====================================================================================================================

def _snapshot_tree(d):
    import numpy as np
    from astropy.io import fits
    frozen, links, arch, pblib = (os.path.join(d, x) for x in ('frozen', 'links', 'arch', 'pblib'))
    rows = []
    for slot in ('14h_41MHz', '09h_Red'):
        band = slot.split('_', 1)[1]
        for f in CFG.masters_snapshot.FROZEN_FILES:
            name = f.format(b=band)
            p = os.path.join(frozen, slot, 'coadd_final', name)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, 'wb').write(os.urandom(2048))
            link = ''
            if f in CFG.masters_snapshot.LINKED_FILES:
                link = os.path.join(links, slot, 'coadd_final', name)
                os.makedirs(os.path.dirname(link), exist_ok=True)
                os.symlink(p, link)
            m = TS.md5_file(p)
            rows.append(dict(slot=slot, file=name, source='src', note='link target', md5_source=m, md5_copy=m, link=link,
                             link_before=''))
    man = os.path.join(frozen, 'MANIFEST.csv')
    with open(man, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.makedirs(os.path.join(arch, 'Coadd'), exist_ok=True)
    os.makedirs(pblib, exist_ok=True)
    for f, sb, k in TN.pblib_expected():
        h = fits.Header()
        h['PBBEAM'], h['PBSCOPE'], h['PBLSTBIN'] = CFG.masters_snapshot.PB_BEAM_MODEL, sb, k
        fits.writeto(os.path.join(pblib, f), np.linspace(0, 0.99, 64, dtype=np.float32).reshape(8, 8), h, overwrite=True)
    sha1 = hashlib.sha1(open(man, 'rb').read()).hexdigest()
    return frozen, links, arch, pblib, sha1


def test_snapshot_units():
    d = os.path.join(scratch(), 'snap')
    frozen, links, arch, pblib, sha1 = _snapshot_tree(d)
    s = CFG.masters_snapshot
    with Patch((s, 'FROZEN_ROOT', frozen), (s, 'LINK_ROOT', links), (s, 'FROZEN_MANIFEST_SHA1', sha1),
               (CFG.paths, 'PHASE3_ARCHIVE_ROOT', arch), (CFG.masters_pbmaps, 'PB_LIBRARY', pblib)):
        TN._MAN.clear()
        run = new_run('snap_out')
        opts = TS.Opts()
        for slot in ('14h_41MHz', '09h_Red'):
            TN._slot_unit(run, opts, slot)
        TN._pblib_unit(run, opts)
        st = {u.name: (u.status, u.reason) for u in run.units}
        assert all(v[0] == 'done' for v in st.values()), st
        reg = TN.registry_path(run, '14h_41MHz')
        r = validate.run('snapshot_master_registry', reg)                  # verify mode: re-hashes the frozen files
        assert all(t['passed'] for t in r), r
        r = validate.run('snapshot_pb_library', os.path.join(run.out_root, 'snapshot', 'pb_library_registry.csv'))
        assert all(t['passed'] for t in r), r
        # G-D02: a frozen file changed after the freeze; G-D01: a link re-pointed into the archive Coadd/ tree
        p = os.path.join(frozen, '14h_41MHz', 'coadd_final', 'PB_deep_41MHz.fits')
        open(p, 'ab').write(b'x')
        r = validate.run('snapshot_master_registry', reg)
        assert not r[0]['passed'], r
        link = os.path.join(links, '09h_Red', 'coadd_final', 'V_deep_Taper_R0_Red.fits')
        tgt = os.path.join(arch, 'Coadd', 'V.fits')
        open(tgt, 'wb').write(b'v')
        os.remove(link)
        os.symlink(tgt, link)
        run2 = new_run('snap_out2')
        TN._slot_unit(run2, opts, '14h_41MHz')
        TN._slot_unit(run2, opts, '09h_Red')
        u1, u2 = run2.units
        assert u1.status == 'refused' and 'G-D02' in u1.reason, (u1.status, u1.reason)
        assert u2.status == 'refused' and 'Coadd' in u2.reason, (u2.status, u2.reason)
        # a manifest that is not the pinned one refuses every slot
        with Patch((s, 'FROZEN_MANIFEST_SHA1', '0' * 40)):
            run3 = new_run('snap_out3')
            TN._slot_unit(run3, opts, '09h_Red')
            assert run3.units[0].status == 'refused' and 'sha1' in run3.units[0].reason
        # freeze is plan only (G-D04)
        run4 = new_run('snap_out4')
        TN.stage_freeze(run4, opts)
        assert run4.units[0].status == 'refused' and 'G-D04' in run4.units[0].reason
        TN._MAN.clear()


FAKE_PARENTCAT = r'''
import argparse, hashlib, json, os, sys
import numpy as np
from astropy.io import fits
ap = argparse.ArgumentParser()
for k in ('--out-dir', '--work', '--nproc', '--only'):
    ap.add_argument(k)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
bad = []
for pair in a.only.split(','):
    if pair.endswith('_Red'):
        print('FAILED %s 0 gaussians 0.1 s RuntimeError(synthetic)' % pair); bad.append(pair); continue
    m = os.path.join(os.environ['FAKE_FROZEN'], pair, 'coadd_final', 'I_deep_Taper_R075_%s.fits' % pair.split('_', 1)[1])
    d = np.squeeze(fits.getdata(m)).astype(float); d[~np.isfinite(d)] = 0; d[np.abs(d) > 1e6] = 0
    out = os.path.join(a.out_dir, 'master_R075_%s.gaul.csv' % pair)
    open(out, 'w').write('# Gaussian list\n# Gaus_id, Isl_id, Source_id, Wave_id, RA, E_RA, DEC\n'
                         '0, 0, 0, 0, 10.0, 0.1, 40.0\n1, 1, 1, 0, 11.0, 0.1, 41.0\n')
    json.dump({'master': m, 'master_md5_data': hashlib.md5(np.ascontiguousarray(d.astype(np.float32)).tobytes()).hexdigest(),
               'CONSTDIG': fits.getheader(m).get('CONSTDIG'), 'thresh_pix': 7.0, 'thresh_isl': 4.0, 'atrous_do': False,
               'rows_pybdsf': 3, 'rows_dropped_nonfinite_position': 1}, open(out + '.gen.json', 'w'))
    print('BUILT %s 2 gaussians 0.1 s ' % pair)
print('PARENTCAT-CONTRACT-END built %d, failed %d %s' % (len(a.only.split(',')) - len(bad), len(bad), bad))
sys.exit(1 if bad else 0)
'''


def test_snapshot_sources():
    """M21 carried (review 2026-10-13): the sources stage runs the in-place builder through core.legacy.run, tests every BUILT
    catalogue (class snapshot_parent_catalog: sidecar digest = the zapped master data now, recipe, rows), records a FAILED
    pair as an item refusal, and refuses earlier catalogues under --out (the builder would skip them silently)."""
    import numpy as np
    from astropy.io import fits
    d = os.path.join(scratch(), 'snapsrc')
    frozen = os.path.join(d, 'frozen')
    for slot in ('14h_41MHz', '09h_Red'):
        p = os.path.join(frozen, slot, 'coadd_final', f'I_deep_Taper_R075_{slot.split("_", 1)[1]}.fits')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        a = np.random.RandomState(5).normal(size=(16, 16)).astype(np.float32)
        a[0, 0], a[1, 1] = np.nan, 3e7
        h = fits.Header()
        h['CONSTDIG'] = 'abc123'
        fits.writeto(p, a, h, overwrite=True)
    fake = os.path.join(d, 'fake_parentcat.py')
    open(fake, 'w').write(FAKE_PARENTCAT)
    orig = TN.script
    os.environ['FAKE_FROZEN'] = frozen
    try:
        with Patch((CFG.masters_snapshot, 'FROZEN_ROOT', frozen),
                   (TN, 'script', lambda short: fake if short == 'parentcat' else orig(short))):
            run = new_run('snapsrc_out')
            TN.stage_sources(run, TS.Opts(work=os.path.join(d, 'work')), ['14h_41MHz', '09h_Red'])
            u = run.units[-1]
            assert u.status == 'done' and len(u.outputs) == 1 and len(u.refused_items) == 1, (u.status, u.reason)
            assert all(t['passed'] for t in u.tests), u.tests
            cat = TN.parent_paths(run, '14h_41MHz')[0]
            g = json.load(open(cat + '.gen.json'))
            g['master_md5_data'] = '0' * 32                                  # a catalogue of other data fails
            json.dump(g, open(cat + '.gen.json', 'w'))
            r = validate.run('snapshot_parent_catalog', cat)
            assert not r[0]['passed'] and 'md5' in r[0]['detail'], r
            run2 = new_run('snapsrc_out')
            TN.stage_sources(run2, TS.Opts(work=os.path.join(d, 'work')), ['14h_41MHz'])
            assert run2.units[-1].status == 'refused' and 'G-S01' in run2.units[-1].reason, run2.units[-1].reason
    finally:
        os.environ.pop('FAKE_FROZEN', None)


# =====================================================================================================================
# read-only checks on real production
# =====================================================================================================================

def test_real_mirrors():
    assert TS.mirror_violations() == [], TS.mirror_violations()
    assert TN.mirror_violations() == [], TN.mirror_violations()


def test_real_content():
    w = f'{PROD_SL}/09h/09h_2024-12-29/18MHz/I/sublst_09h_2024-12-29_18MHz_I_w2.fits'
    if not os.path.exists(w):
        print('  (production 09h products not reachable: skipped)')
        return
    checks = [('sublst_window', w), ('sublst_window_nframes', w[:-5] + '_nframes.fits'),
              ('sublst_window_pb', w[:-5] + '_pb.fits'),
              ('sublst_admission', f'{PROD_SL}/09h/09h_2024-12-29/18MHz/I/admission_09h_2024-12-29_18MHz_I.csv'),
              ('sublst_fluxdc_record', f'{PROD_SL}/09h/09h_2024-12-29/18MHz/I/fluxdc_09h_2024-12-29_18MHz.json'),
              ('sublst_window_master', f'{PROD_SL}/09h_masters/18MHz/I/sublst_master_09h_18MHz_I_w2.fits'),
              ('sublst_loo_residual', w[:-5] + '_resid_loo.fits'),
              ('sublst_input_copy', f'{PROD_SL}/coadd_09h/09h_18MHz/coadd_final/I_deep_Taper_R0_18MHz.fits')]
    for cls, p in checks:
        r = validate.run(cls, p)
        assert r and all(t['passed'] for t in r), (cls, p, r)
        print(f'  PASS {cls} on {os.path.basename(p)}: ' + '; '.join(str(t['detail'])[:90] for t in r))


def test_real_snapshot_parents():
    """The installed snapshot parent set (M21, exopipe_ops 3ff6565): a seeded sample of catalogues passes the contract test."""
    import random
    d = CFG.masters_snapshot.PARENT_CATALOG_DIR
    cats = sorted(f for f in os.listdir(d) if f.endswith('.gaul.csv')) if os.path.isdir(d) else []
    if not cats:
        print('  (production snapshot parent catalogues not reachable: skipped)')
        return
    assert len(cats) == CFG.masters_snapshot.N_SLOTS, len(cats)
    for f in random.Random(20261013).sample(cats, 4):
        r = validate.run('snapshot_parent_catalog', os.path.join(d, f))
        assert r and all(t['passed'] for t in r), (f, r)
        print(f'  PASS snapshot_parent_catalog on {f}: {r[0]["detail"][:110]}')


def test_real_stack_night_dry():
    """The per-hour interface frames imports, in --dry-run on a real night: prints the plan, writes nothing."""
    if not os.path.isdir(f'{CFG.paths.SNAPSHOT_CUBES_ROOT}/09h_2024-12-29'):
        print('  (production cubes not reachable: skipped)')
        return
    run = new_run('dry_night', dry=True)
    res = TS.stack_night(run, '09h', '2024-12-29', subbands=['18MHz', '64MHz'])
    assert res['admit_unit'] == 'admit_09h_2024-12-29' and res['would_refuse'] == 0, res
    assert not run.units and not os.listdir(run.out_root), os.listdir(run.out_root)
    main, deep, drop = TS.admit_plan('09h', ['09h_2024-12-29'], ['18MHz', '64MHz'])
    prod = [l.strip() for l in open('/lustre/gh/main/exopipe_ops/sublst_jobs_09h.txt')
            if l.startswith('09h_2024-12-29 ') and l.split()[1] in ('18MHz', '64MHz')]
    assert sorted(main + deep) == sorted(prod), (main, deep, prod)


STACK_BANDS = ['18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz', '64MHz', '69MHz',
               '73MHz', '78MHz', '82MHz']
GUARD_S = 3.0     # s, how long a synthetic unit waits for what a correct budget does at once (only a wrong one pays it)


class StackNode:
    """A synthetic node for the stack-budget tests. free0 GB are free while no unit of the test runs (another job holds
    the rest). A unit TAKES its charge only ramp seconds after its start - until then the free memory still reads high -
    and ends life seconds after its start. Units of several budget_map calls (owners) may run on it. Recorded: what ran,
    the largest sum of the CHARGES of the units running at one moment, the largest number of units at one moment."""

    def __init__(self, charge, free0, ramp=0.04, life=0.06):
        import threading
        self.charge, self.free0, self.ramp, self.life = charge, float(free0), ramp, life
        self.lock, self.live, self.started, self.done, self.peak_mem, self.peak_n = threading.Lock(), {}, [], [], 0.0, 0

    def taken(self, owner=None):
        """GB the running units (of one owner, or of all) have taken by now."""
        import time
        now = time.time()
        with self.lock:
            return sum(self.charge(b) for (o, b), t0 in self.live.items()
                       if (owner is None or o == owner) and now - t0 >= self.ramp)

    def free(self):
        return self.free0 - self.taken()

    def unit(self, owner=None):
        import time

        def fn(b):
            with self.lock:
                self.live[(owner, b)] = time.time()
                self.started.append(b)
                self.peak_mem = max(self.peak_mem, sum(self.charge(x) for _, x in self.live))
                self.peak_n = max(self.peak_n, len(self.live))
            time.sleep(self.life)
            with self.lock:
                del self.live[(owner, b)]
                self.done.append(b)
        return fn


def gather(n_together, hold=0.0):
    """A synthetic unit that ends once n_together units run at one moment (or after GUARD_S) and hold seconds more, and the
    state it keeps: st['peak'] = the most units at one moment, st['n'] = running now, st['first'] = set when one has
    started. What a correct budget starts side by side is seen without relying on sleeps; the hold shows a unit too many."""
    import threading
    import time
    lock, ev, st = threading.Lock(), threading.Event(), dict(n=0, peak=0, started=[], first=threading.Event())

    def fn(b):
        with lock:
            st['n'] += 1
            st['peak'] = max(st['peak'], st['n'])
            st['started'].append(b)
            if st['n'] >= n_together:
                ev.set()
        st['first'].set()
        ev.wait(GUARD_S)
        time.sleep(hold)
        with lock:
            st['n'] -= 1
    return fn, st


def test_stack_budget():
    """The stack stage's memory budget (speed only): every unit runs exactly once; the charges of the units running at one
    moment never exceed the budget and never more than the unit cap run; the largest charges start first; a budget below
    the largest charge, or no memory free on the node now, means one unit at a time; an error is printed and the others go
    on; a SystemExit of one unit is raised after every unit has run, a second one is printed; an interrupt is raised at
    once; --workers and an unreadable MemAvailable keep the fixed pool; the free memory is also bounded by the memory limit
    of the process's cgroups. Review fixes 2 (2026-11-12): the charge follows the
    cube's frame count; a further unit starts only within the free memory LESS what the running units have been charged
    and not taken yet (the free memory reads high while they ramp); the look is repeated every poll."""
    import contextlib
    import io
    import threading
    import time

    def check(what, ok, detail=''):
        assert ok, f'{what}: {detail}'
    sec = CFG.masters_sublst
    bands = list(STACK_BANDS)
    classes = [TS.stack_band_class(b) for b in bands]
    check('5 low, 5 mid, 5 high subbands', classes == ['low'] * 5 + ['mid'] * 5 + ['high'] * 5, str(classes))
    from exopipe.phase3.lib.snapshot_engine import cell as _cell
    check('the same classes as lib/snapshot_engine/cell.band_class', classes == [_cell.band_class(b) for b in bands])
    G = sec.STACK_MEM_GB
    check('config: a charge per class, a fraction in (0, 1], a unit cap of at most 6 (the Lustre-reader rule)',
          set(G) == {'low', 'mid', 'high'} and 0 < sec.STACK_MEM_FRACTION <= 1 and 1 <= sec.STACK_MAX_UNITS <= 6
          and 0 < G['low'] <= G['mid'] <= G['high'], f'{dict(G)} {sec.STACK_MEM_FRACTION} {sec.STACK_MAX_UNITS}')
    check('config: the charges are above the measured peaks (13.0, 28.6, 51.0 GB)',
          G['low'] > 13.0 and G['mid'] > 28.6 and G['high'] > 51.0, str(dict(G)))
    # the charge follows the cube's frame count (review 2026-11-12: archived cubes hold 358 to 681 frames, not one size)
    R, U, C = sec.STACK_REF_FRAMES, sec.STACK_UNKNOWN_FRAMES_FACTOR, TS.stack_charge_gb
    check('config: the charges are those of 358-frame cubes; an unknown frame count is charged at least as the longest '
          'archived cube (681 frames)', R == 358 and U >= 681 / 358.0, f'{R} {U}')
    check('358 frames: the class charge', [C(b, sec, 358) for b in ('18MHz', '41MHz', '73MHz')]
          == [G['low'], G['mid'], G['high']], str([C(b, sec, 358) for b in ('18MHz', '41MHz', '73MHz')]))
    check('680 frames: the class charge x 680/358', abs(C('73MHz', sec, 680) - G['high'] * 680 / 358.0) < 1e-9
          and abs(C('18MHz', sec, 680) - G['low'] * 680 / 358.0) < 1e-9, f"{C('73MHz', sec, 680)} {C('18MHz', sec, 680)}")
    check('fewer than 358 frames: the class charge', C('73MHz', sec, 100) == G['high'] and C('41MHz', sec, 1) == G['mid'])
    check('an unknown frame count (None, not given, 0, negative, no number): the class charge x the unknown factor',
          all(C('73MHz', sec, n) == G['high'] * U for n in (None, 0, -5, 'x', float('nan'), float('inf')))
          and C('73MHz', sec) == G['high'] * U and C('18MHz', sec, None) == G['low'] * U)
    charge = lambda b: C(b, sec, R)                                                          # noqa: E731
    check('a label that is no <n>MHz subband is charged as the largest', charge('Red') == G['high'] == charge('82MHz')
          and C('Red', sec, 2 * R) == 2 * G['high'] and C('Red', sec) == G['high'] * U)
    with tempfile.TemporaryDirectory() as td:               # the frame count: the NFRAMES card of the cube's PRIMARY header
        from astropy.io import fits
        import numpy as np
        ph = fits.Header()
        ph['NFRAMES'] = 681
        cube = os.path.join(td, 'cube.fits')
        fits.HDUList([fits.PrimaryHDU(header=ph), fits.ImageHDU(np.zeros((2, 2), np.float32))]).writeto(cube)
        nocard = os.path.join(td, 'nocard.fits')
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.zeros((2, 2), np.float32), header=ph)]).writeto(nocard)
        nofits = os.path.join(td, 'nofits.fits')
        open(nofits, 'wb').write(b'cube')
        got = [TS.cube_nframes(p) for p in (cube, nocard, nofits, os.path.join(td, 'missing.fits'))]
        check('cube_nframes: the primary NFRAMES card; None without the card (a card of an extension does not count), for a '
              'file that is no FITS, for a missing file', got == [681, None, None, None], str(got))

    def drive(budget, cap, fail=(), leave=(), **kw):
        lock, st = threading.Lock(), dict(mem=0, n=0, peak_mem=0, peak_n=0, started=[], done=[], raised=None)

        def fn(b):
            with lock:
                st['mem'] += charge(b)
                st['n'] += 1
                st['peak_mem'] = max(st['peak_mem'], st['mem'])
                st['peak_n'] = max(st['peak_n'], st['n'])
                st['started'].append(b)
            time.sleep(0.05)
            with lock:
                st['mem'] -= charge(b)
                st['n'] -= 1
                st['done'].append(b)
            if b in fail:
                raise RuntimeError('synthetic failure of ' + b)
            if b in leave:
                raise SystemExit('synthetic refusal of ' + b)
        try:
            TS.budget_map(fn, list(bands), charge, budget, cap, **kw)
        except SystemExit as e:
            st['raised'] = e
        return st
    budget = 3 * G['high'] + G['mid']                         # three high units and one mid unit fit, a fourth high does not
    st = drive(budget, 6)
    check('every unit ran exactly once', sorted(st['done']) == sorted(bands), str(st['done']))
    check('the running charges never exceeded the budget, and never more than 6 units ran',
          st['peak_mem'] <= budget and st['peak_n'] <= 6, f"peak {st['peak_mem']} of {budget} GB, {st['peak_n']} units")
    check('the largest units start first: the first four are three high bands (in the plan order) and the first mid band',
          set(st['started'][:4]) == {'64MHz', '69MHz', '73MHz', '41MHz'}, str(st['started'][:5]))
    st = drive(10000, 6)
    check('without a memory limit the cap of 6 units holds', 2 <= st['peak_n'] <= 6 and len(st['done']) == 15, str(st['peak_n']))
    st = drive(10000, 2)
    check('a cap of 2 units holds', st['peak_n'] == 2 and len(st['done']) == 15, str(st['peak_n']))
    st = drive(G['high'] - 1, 6)
    check('a budget below the largest charge: one unit at a time, every unit still runs',
          st['peak_n'] == 1 and len(st['done']) == 15, f"peak {st['peak_n']}, done {len(st['done'])}")
    st = drive(10000, 6, mem_now_gb=lambda: 0.0, poll_s=0.01)
    check('no memory free on the node now: nothing starts beside a running unit, every unit still runs',
          st['peak_n'] == 1 and len(st['done']) == 15, f"peak {st['peak_n']}, done {len(st['done'])}")
    st = drive(10000, 6, mem_now_gb=lambda: None, poll_s=0.01)
    check('free memory unreadable while running: the budget alone decides', st['peak_n'] >= 2 and len(st['done']) == 15)
    st = drive(budget, 6, fail=('64MHz', '18MHz'))
    check('a failing unit does not stop the others', len(st['done']) == 15 and st['raised'] is None, str(len(st['done'])))
    st = drive(budget, 6, leave=('69MHz',))
    check('a SystemExit of one unit is raised again after every unit has run',
          len(st['done']) == 15 and 'synthetic refusal of 69MHz' in str(st['raised']), f"{len(st['done'])} {st['raised']}")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        st = drive(budget, 6, leave=('69MHz', '18MHz'))
    said = [x for x in buf.getvalue().splitlines() if 'not raised again' in x]
    both = {'synthetic refusal of 69MHz', 'synthetic refusal of 18MHz'}
    check('two units raise SystemExit: every unit runs, the first is raised again, the second is printed (not dropped)',
          len(st['done']) == 15 and str(st['raised']) in both and len(said) == 1
          and (both - {str(st['raised'])}).pop() in said[0], f"{len(st['done'])} {st['raised']!r} {said}")
    lock, seen = threading.Lock(), dict(started=[], ended=[])

    def interrupting(b):
        with lock:
            seen['started'].append(b)
        if b == '64MHz':
            raise KeyboardInterrupt('synthetic interrupt in 64MHz')
        time.sleep(0.05)
        with lock:
            seen['ended'].append(b)
    try:
        TS.budget_map(interrupting, list(bands), charge, 2 * G['high'], 6)
        raised = None
    except KeyboardInterrupt as e:
        raised = e
    check('an interrupt is raised at once, not after the remaining units: no further unit starts (2 of 15 started) and '
          'the running one is waited for', raised is not None and sorted(seen['started']) == ['64MHz', '69MHz']
          and seen['ended'] == ['69MHz'], f'{raised!r} {seen}')
    # the memory the running units have been charged and have not taken yet (review 2026-11-12: the free memory reads
    # high for about a minute after a start, and again when a unit's Stokes V job starts)
    st = drive(10000, 6, mem_now_gb=lambda: 60.0, held_gb=lambda: None, poll_s=0.005)
    check('60 GB read free at every look and nothing known to be held: the charges of the running units never exceed 60 '
          'GB - a started unit has taken nothing yet, in the look that starts it and in the following ones',
          st['peak_mem'] <= 60 and len(st['done']) == 15, f"peak {st['peak_mem']} GB, done {len(st['done'])}")
    node = StackNode(charge, free0=139.0)                     # a 339 GB node on which another job holds 200 GB
    TS.budget_map(node.unit(), list(bands), charge, 271, 6, mem_now_gb=node.free, held_gb=node.taken, poll_s=0.005)
    check('units that take their memory only after a ramp of several polls, on a node where another job took 200 of 339 GB '
          'after the 271 GB budget was set: every unit runs once, and the charges of the running units never exceed the '
          '139 GB the node has, although the free memory still reads high during every ramp',
          sorted(node.done) == sorted(bands) and node.peak_mem <= 139.0, f'peak {node.peak_mem} GB, done {len(node.done)}')
    fn, st = gather(2)

    def holding():                                            # a running unit holds its whole charge from its start
        st['first'].wait(GUARD_S)
        return G['high'] * st['n']
    TS.budget_map(fn, ['64MHz', '69MHz'], charge, 10000, 6, mem_now_gb=lambda: 120.0 - G['high'] * st['n'],
                  held_gb=holding, poll_s=0.005)
    check('memory a running unit HOLDS is no longer "not yet taken": 120 GB free before, a 56 GB unit that holds its 56 GB '
          'leaves 64 GB free, and a second 56 GB unit starts beside it', st['peak'] == 2, str(st['peak']))
    fn, st = gather(2)
    looks = []

    def rising():
        looks.append(1)
        return 0.0 if len(looks) <= 3 else 1000.0
    TS.budget_map(fn, ['64MHz', '69MHz'], charge, 10000, 6, mem_now_gb=rising, poll_s=0.005)
    check('the look is made again every poll_s seconds: the second unit starts when the reading rises (the fourth look), '
          'while the first still runs and without any unit ending', st['peak'] == 2 and len(looks) >= 4,
          f"peak {st['peak']}, {len(looks)} looks")
    with Patch((TS, 'mem_available_gb', lambda: None)):
        check('MemAvailable unreadable: the look returns None (the budget alone decides), never 0',
              TS.stack_free_gb(sec) is None)
    with Patch((TS, 'mem_available_gb', lambda: 100)):
        check('the look: the configured fraction of MemAvailable', TS.stack_free_gb(sec) == sec.STACK_MEM_FRACTION * 100)
    with Patch((TS, 'mem_available_gb', lambda: 0)):
        check('no memory available reads 0 GB, not "unreadable"', TS.stack_free_gb(sec) == 0)
    import subprocess
    kid = subprocess.Popen(['sleep', '30'])
    try:
        h = TS.descendants_held_gb()
    finally:
        kid.kill()
        kid.wait()
    try:
        import psutil  # noqa: F401
        check('the memory resident in the child processes of this process: above 0 GB with a child running',
              isinstance(h, float) and 0 < h < 10000, repr(h))
    except ImportError:
        check('without psutil the memory the units hold is unknown (None)', h is None, repr(h))
    TS.budget_map(lambda b: None, [], charge, 100, 6)
    # the choice between the fixed pool and the budget
    P = TS.stack_parallelism
    check('--workers N: the fixed pool of N', P(3, 400, sec) == ('pool', 3) and P(3, None, sec) == ('pool', 3))
    check("MemAvailable unreadable: the fixed pool of PARALLEL['stack']", P(None, None, sec) == ('pool', sec.PARALLEL['stack']))
    check('otherwise the budget: the configured fraction of the available memory, the configured cap',
          P(None, 100, sec) == ('budget', int(sec.STACK_MEM_FRACTION * 100), sec.STACK_MAX_UNITS), str(P(None, 100, sec)))
    check('never more units than the cores this process may use', P(None, 400, sec, n_cores=2)[2] == 2
          and P(None, 400, sec, n_cores=64)[2] == sec.STACK_MAX_UNITS)
    check('no memory available (0 GB) is a budget of 0, not "unreadable"', P(None, 0, sec)[:2] == ('budget', 0))
    m = TS.mem_available_gb()
    check('MemAvailable of this node reads as a whole number of GB (or None)', m is None or (isinstance(m, int) and m >= 0), repr(m))
    # the memory limit of the cgroups the process runs in (2026-11-12: /user.slice caps all user sessions together at
    # 400 GiB on a node of 503 GiB; MemAvailable does not know that cap)
    G = 2 ** 30
    with tempfile.TemporaryDirectory() as cg:
        root, pc = os.path.join(cg, 'sys'), os.path.join(cg, 'proc_cgroup')

        def level(rel, limit, current=None, stat=None):
            d = os.path.join(root, rel)
            os.makedirs(d, exist_ok=True)
            for name, text in (('memory.max', f'{limit}\n'), ('memory.current', None if current is None else f'{current}\n'),
                               ('memory.stat', None if stat is None else ''.join(f'{k} {v}\n' for k, v in stat.items()))):
                if text is None:
                    if os.path.exists(os.path.join(d, name)):
                        os.remove(os.path.join(d, name))
                else:
                    with open(os.path.join(d, name), 'w') as f:
                        f.write(text)

        def left():
            return TS.cgroup_available_gb(pc, root)
        check('no cgroup file of the process: None', left() is None, repr(left()))
        with open(pc, 'w') as f:
            f.write('0::/user.slice/user-1.slice/session-2.scope\n')
        check('no cgroup tree: None', left() is None, repr(left()))
        level('user.slice/user-1.slice/session-2.scope', 'max')
        level('user.slice/user-1.slice', 'max', 5 * G, dict(inactive_file=G))
        check('no level sets a limit: None', left() is None, repr(left()))
        level('user.slice', 400 * G, 300 * G, dict(anon=150 * G, file=150 * G, shmem=10 * G, inactive_file=60 * G,
                                                    active_file=80 * G))
        check('limit - usage + the file cache that can be dropped (inactive and active; shared memory is not)',
              left() == 400 - 300 + 60 + 80, repr(left()))
        level('user.slice/user-1.slice', 50 * G, 45 * G, dict(inactive_file=2 * G, active_file=0))
        check('the smallest over the levels from the process up', left() == 7.0, repr(left()))
        level('user.slice/user-1.slice', 50 * G, 60 * G, dict(inactive_file=0, active_file=0))
        check('usage above the limit: 0, never negative', left() == 0.0, repr(left()))
        level('user.slice/user-1.slice', 50 * G)
        check('a limited level whose usage cannot be read is skipped', left() == 240.0, repr(left()))
        with open(pc, 'w') as f:
            f.write('12:memory:/user.slice\n')
        check('cgroup v1 (no line of the unified hierarchy): None', left() is None, repr(left()))
    if m is not None and m >= 3:
        with Patch((TS, 'cgroup_available_gb', lambda: 1.9)):
            check('the free memory is what is left under the cgroup limit when that is less than MemAvailable',
                  TS.mem_available_gb() == 1, repr(TS.mem_available_gb()))
        with Patch((TS, 'cgroup_available_gb', lambda: None)):
            big = TS.mem_available_gb()
        with Patch((TS, 'cgroup_available_gb', lambda: 1e9)):
            check('and MemAvailable when the limit leaves more, or there is none', abs(TS.mem_available_gb() - big) <= 2,
                  f'{TS.mem_available_gb()} against {big}')


def test_stack_budget_node():
    """The stack budget on a node that several invocations share (review fixes 2, 2026-11-12). The node ledger - a JSON
    file under a lock, one entry per budget invocation - makes the unit cap and the free memory NODE-wide: an entry counts
    while it is live (a process of that pid and create time, updated within the stale limit) and is dropped otherwise; the
    own entry is rewritten every poll and removed at the end, an interrupt included; two budget_map calls sharing a ledger
    never run more units than the cap together and never exceed the node's memory together; a ledger that cannot be
    created or locked switches itself off with one log line and the call runs as a single invocation, as with no ledger.
    The stage: charges from the cubes' own NFRAMES cards, its log line, the look, the held memory, the ledger from config.
    Everything in a temporary directory; no archive data, nothing under /dev/shm."""
    import contextlib
    import fcntl
    import io
    import subprocess
    import threading
    import time
    import types

    def check(what, ok, detail=''):
        assert ok, f'{what}: {detail}'
    sec = CFG.masters_sublst
    G, R, U, bands = sec.STACK_MEM_GB, sec.STACK_REF_FRAMES, sec.STACK_UNKNOWN_FRAMES_FACTOR, list(STACK_BANDS)
    charge = lambda b: TS.stack_charge_gb(b, sec, R)                                         # noqa: E731
    check('config: the ledger is a per-user file on node-local shared memory',
          sec.STACK_NODE_LEDGER.startswith('/dev/shm/') and '{uid}' in sec.STACK_NODE_LEDGER, sec.STACK_NODE_LEDGER)
    with Patch((sec, 'STACK_NODE_LEDGER', '')):
        check('config: an empty ledger path means no ledger', TS.stack_node_tools(sec)[1] is None)
    try:
        import psutil
    except ImportError:                                       # no ledger without psutil: switched off, one line
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()) as buf:
            L = TS.NodeLedger(os.path.join(td, 'ledger.json'))
            ok = L.begin(60.0) or L.begin(60.0)
        check('without psutil the ledger switches itself off with one line', not ok and L.off
              and buf.getvalue().count('switched off') == 1, buf.getvalue())
        return
    me = psutil.Process()
    pid, ct = me.pid, me.create_time()
    gone = subprocess.Popen(['true'])
    gone.wait()                                               # a pid no process has any more

    def entry(units, charged, held, age=0.0, pid_=pid, ct_=ct):
        return dict(pid=pid_, create_time=ct_, units=units, charged_gb=charged, held_gb=held, t=time.time() - age)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'ledger.json')

        def write(entries):
            with open(path, 'w') as f:
                json.dump(entries, f)

        def read(p=None):
            with open(p or path) as f:
                return json.load(f)

        def quiet(fn_, *a, **k):                              # a budget_map call, its log lines returned
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                fn_(*a, **k)
            return buf.getvalue().splitlines()
        # ---- the ledger alone: entries, liveness, the stale limit ------------------------------------------------------
        L = TS.NodeLedger(path)
        check('a first transaction on a new ledger: no other invocation',
              L.begin(60.0) and (L.others_units, L.others_unmaterialized) == (0, 0.0))
        L.commit((2, 88.0, 30.5))
        own = read().get(L.key) or {}
        main = threading.current_thread() is threading.main_thread()
        check('the entry: keyed <pid>:<process create time> (and :<thread id> outside the main thread), holding pid, '
              'create_time, units, charged_gb, held_gb and the time of the update',
              list(read()) == [L.key] and (L.key == f'{pid}:{ct!r}' if main else L.key.startswith(f'{pid}:{ct!r}:'))
              and set(own) == {'pid', 'create_time', 'units', 'charged_gb', 'held_gb', 't'}
              and (own['pid'], own['create_time'], own['units'], own['charged_gb'], own['held_gb']) == (pid, ct, 2, 88.0, 30.5)
              and abs(own['t'] - time.time()) < 30, f'{L.key} {own}')
        write({L.key: own, 'live': entry(4, 100.0, 30.0), 'holds-more-than-charged': entry(1, 10.0, 25.0),
               'old-50s': entry(1, 20.0, 0.0, age=50.0), 'stale': entry(6, 500.0, 0.0, age=3600.0),
               'dead-pid': entry(6, 500.0, 0.0, pid_=gone.pid), 'reused-pid': entry(6, 500.0, 0.0, ct_=ct - 1000.0),
               'no-fields': {'pid': pid}, 'no-dict': [1, 2]})
        ok = L.begin(60.0)
        check('other entries count while live (4 + 1 + 1 units; charged and not held 70 + 0 + 20 GB): not the own entry, not '
              'a stale one (1 h old), not one of a dead pid, not one of a reused pid (another create time), not a malformed one',
              ok and (L.others_units, L.others_unmaterialized) == (6, 90.0), f'{L.others_units} {L.others_unmaterialized}')
        L.commit((0, 0.0, 0.0))
        check('entries that are not live are dropped when the file is written',
              sorted(read()) == sorted([L.key, 'live', 'holds-more-than-charged', 'old-50s']), str(sorted(read())))
        ok = L.begin(40.0)
        check('the stale limit is the reader\'s: an entry updated 50 s ago is not live within a limit of 40 s',
              ok and (L.others_units, L.others_unmaterialized) == (5, 70.0), f'{L.others_units} {L.others_unmaterialized}')
        L.commit((0, 0.0, 0.0))
        L.leave()
        check('leave removes the own entry and no other', sorted(read()) == ['holds-more-than-charged', 'live']
              and L.key is None, str(sorted(read())))
        # ---- budget_map beside another invocation's entry --------------------------------------------------------------
        write({'other': entry(4, 200.0, 200.0)})
        fn, st = gather(2, hold=0.03)
        TS.budget_map(fn, bands[:6], charge, 10000, 6, node=TS.NodeLedger(path), poll_s=0.005)
        check('the unit cap is node-wide: another invocation runs 4 units, so 6 - 4 = 2 units of this call run at one moment, '
              'never 3; the own entry is gone after the call and the other one is kept',
              st['peak'] == 2 and len(st['started']) == 6 and list(read()) == ['other'], f"peak {st['peak']} {list(read())}")
        write({'other': entry(1, 100.0, 30.0)})               # 70 GB charged to the other invocation and not taken yet
        node = StackNode(charge, 1e9, life=0.03)
        TS.budget_map(node.unit(), ['64MHz', '69MHz', '73MHz'], charge, 10000, 6, node=TS.NodeLedger(path),
                      mem_now_gb=lambda: 150.0, poll_s=0.005)
        check('what another invocation has been charged and has not taken (70 GB) is subtracted from the free memory (150 '
              'GB): one 56 GB unit at a time (56 + 56 > 150 - 70)', node.peak_n == 1 and len(node.done) == 3,
              f'peak {node.peak_n}, done {len(node.done)}')
        write({'other': entry(1, 56.0, 56.0)})
        looks, at_start = [], []

        def rising():
            looks.append(1)
            return 0.0 if len(looks) <= 3 else 1000.0
        said = quiet(TS.budget_map, lambda b: at_start.append(len(looks)), ['64MHz'], charge, 10000, 6,
                     node=TS.NodeLedger(path), mem_now_gb=rising, poll_s=0.005)
        check('another invocation runs a unit: the first unit of this call does NOT start without the look - it starts when '
              'the look allows it (the fourth here); the wait is said once, not at every poll',
              at_start == [4] and len([x for x in said if 'may start yet' in x]) == 1, f'{at_start} {said}')
        write({})
        looks, at_start = [], []
        TS.budget_map(lambda b: at_start.append(len(looks)), ['64MHz'], charge, 10000, 6, node=TS.NodeLedger(path),
                      mem_now_gb=rising, poll_s=0.005)
        check('no other invocation on the node: the first unit starts without the look (progress)', at_start == [0], str(at_start))
        write({'old': entry(5, 0.0, 0.0, age=70.0)})          # an invocation that hung 70 s ago with 5 units
        fn, st = gather(6)
        TS.budget_map(fn, bands[:6], charge, 10000, 6, node=TS.NodeLedger(path), poll_s=0.005)
        check('an entry not rewritten for 70 s is past the stale limit max(60 s, 3 poll_s) = 60 s: it no longer counts (the 6 '
              'units of this call run together) and is dropped', st['peak'] == 6 and read() == {}, f"peak {st['peak']} {read()}")
        write({'old': entry(5, 0.0, 0.0, age=70.0)})
        node = StackNode(charge, 1e9, life=0.01)
        TS.budget_map(node.unit(), bands[:3], charge, 10000, 6, node=TS.NodeLedger(path), poll_s=30.0)
        check('with poll_s = 30 s the stale limit is 90 s: the same entry still counts (5 of the 6 units): one unit at a time',
              node.peak_n == 1 and len(node.done) == 3 and list(read()) == ['old'], f'peak {node.peak_n} {list(read())}')
        write({'stale': entry(2, 0.0, 0.0, age=3600.0), 'dead-pid': entry(2, 0.0, 0.0, pid_=gone.pid),
               'reused-pid': entry(2, 0.0, 0.0, ct_=ct - 1000.0)})
        fn, st = gather(6)
        L = TS.NodeLedger(path)
        TS.budget_map(fn, bands[:6], charge, 10000, 6, node=L, mem_now_gb=lambda: 1000.0, poll_s=0.005)
        check('a stale entry, an entry of a dead pid and one of a reused pid are ignored (the 6 units of this call run '
              'together) and dropped; the own entry is gone after the call',
              st['peak'] == 6 and read() == {} and L.key is None and L.off is None, f"peak {st['peak']} {read()} {L.off}")
        # ---- the own entry while a unit runs, and on an interrupt -------------------------------------------------------
        L = TS.NodeLedger(path)
        beats, shown = set(), set()

        def watching(b):
            t_end = time.time() + GUARD_S
            while len(beats) < 3 and time.time() < t_end:
                e = read().get(L.key)
                if e:
                    beats.add(e['t'])
                    shown.add((e['units'], e['charged_gb'], e['pid']))
                time.sleep(0.002)
        TS.budget_map(watching, ['64MHz'], charge, 10000, 6, node=L, poll_s=0.01)
        check('while a unit runs the own entry shows it (1 unit, its 56 GB charge, this pid) and is rewritten every poll_s '
              'seconds although nothing can start or end (3 updates seen); it is gone after the call',
              len(beats) >= 3 and shown == {(1, G['high'], pid)} and read() == {}, f'{len(beats)} {shown} {read()}')
        L = TS.NodeLedger(path)
        after = []

        def interrupting(b):
            if b == '64MHz':
                raise KeyboardInterrupt('synthetic interrupt in 64MHz')
            seen, t_end = set(), time.time() + GUARD_S        # the unit beside it: this call's entry once 1 unit is left
            while len(seen) < 2 and time.time() < t_end:
                e = read().get(L.key)
                if e and e['units'] == 1:
                    seen.add(e['t'])
                time.sleep(0.002)
            after.append((b, len(seen)))
        try:
            TS.budget_map(interrupting, ['64MHz', '69MHz', '73MHz'], charge, 2 * G['high'], 6, node=L, poll_s=0.01)
            raised = None
        except KeyboardInterrupt as e:
            raised = e
        check('an interrupt with a ledger: raised, no further unit starts; while the running unit ends the entry of this '
              'call stays (1 unit) and is rewritten; it is removed at the end',
              raised is not None and after == [('69MHz', 2)] and read() == {}, f'{raised!r} {after} {read()}')
        # ---- two invocations sharing the node ---------------------------------------------------------------------------
        def two_calls(free0, items_a, items_b):
            write({})
            node = StackNode(charge, free0)
            ledgers, errors = [TS.NodeLedger(path), TS.NodeLedger(path)], []

            def call(owner, items):
                try:
                    TS.budget_map(node.unit(owner), items, charge, 271, 6, node=ledgers[owner], mem_now_gb=node.free,
                                  held_gb=lambda: node.taken(owner), poll_s=0.005)
                except BaseException as e:
                    errors.append(e)
            threads = [threading.Thread(target=call, args=(o, it), daemon=True) for o, it in enumerate((items_a, items_b))]
            with contextlib.redirect_stdout(io.StringIO()):
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(60)
            hung = [t for t in threads if t.is_alive()]
            return node, errors, hung, [x.off for x in ledgers]
        node, errors, hung, off = two_calls(139.0, bands[0::2], bands[1::2])
        check('two budget_map calls sharing one ledger on a 139 GB node, units ramping over several polls: every unit of both '
              'runs once, the charges of the units running at one moment never exceed 139 GB together, never more than 6 '
              'units run together, both entries are gone at the end, the ledger stayed on',
              not hung and not errors and sorted(node.done) == sorted(bands) and node.peak_mem <= 139.0 and node.peak_n <= 6
              and read() == {} and off == [None, None],
              f'hung {len(hung)} errors {errors} done {len(node.done)} peak {node.peak_mem} GB {node.peak_n} units {off}')
        node, errors, hung, off = two_calls(1e9, bands, bands)
        check('two calls of 15 units each with memory to spare: the cap of 6 units holds for the two together',
              not hung and not errors and len(node.done) == 30 and node.peak_n <= 6 and read() == {},
              f'hung {len(hung)} errors {errors} done {len(node.done)} peak {node.peak_n} units')
        # ---- a ledger that cannot be used; no ledger ---------------------------------------------------------------------
        nowhere = TS.NodeLedger(os.path.join(td, 'no_such_directory', 'ledger.json'))
        fn, st = gather(6)
        fn2, st2 = gather(6)
        said = quiet(TS.budget_map, fn, bands[:6], charge, 10000, 6, node=nowhere, poll_s=0.005)
        said += quiet(TS.budget_map, fn2, bands[:6], charge, 10000, 6, node=nowhere, poll_s=0.005)
        lines = [x for x in said if 'switched off' in x]
        check('a ledger file that cannot be created: switched off with ONE log line saying the budget is then per invocation; '
              'this call and the next with the same ledger run as a single invocation',
              len(lines) == 1 and 'PER INVOCATION' in lines[0] and nowhere.off and st['peak'] == 6 and st2['peak'] == 6,
              f"{said} {st['peak']} {st2['peak']}")
        write({})
        fd = os.open(path + '.lock', os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)                        # another invocation stopped inside its transaction
        try:
            held_lock = TS.NodeLedger(path, lock_wait_s=0.05)
            fn, st = gather(6)
            said = quiet(TS.budget_map, fn, bands[:6], charge, 10000, 6, node=held_lock, poll_s=0.005)
        finally:
            os.close(fd)
        check('a ledger that cannot be locked within its wait: switched off with one line, the units run',
              len([x for x in said if 'switched off' in x]) == 1 and 'no lock' in (held_lock.off or '') and st['peak'] == 6,
              f"{said} {held_lock.off} {st['peak']}")
        firsts = []
        for nd in (None, TS.NodeLedger(path), nowhere):
            fn, st = gather(4)
            TS.budget_map(fn, list(bands), charge, 3 * G['high'] + G['mid'], 6, node=nd, poll_s=0.005)
            firsts.append((sorted(st['started'][:4]), len(st['started'])))
        check('a ledger with no other invocation, a ledger switched off and no ledger give the same single-invocation '
              'behavior: the first four units are the three first high bands and the first mid band, all 15 run',
              firsts == [(['41MHz', '64MHz', '69MHz', '73MHz'], 15)] * 3, str(firsts))
        # ---- the stage: charges from the cubes' own NFRAMES cards, its log line, the helpers it hands to budget_map ------
        from astropy.io import fits
        lst, night = '09h', '2030-01-01'
        cell = f'{lst}_{night}'
        croot = os.path.join(td, 'cubes')
        os.makedirs(os.path.join(croot, cell))
        for sb, n in (('18MHz', 358), ('73MHz', 680)):        # 41 MHz has no cube here: its frame count is unknown
            ph = fits.Header()
            ph['NFRAMES'] = n
            fits.PrimaryHDU(header=ph).writeto(os.path.join(croot, cell, f'{cell}_{sb}_I_flowcube.fits'))
        jobs = [(cell, sb, ['I', 'V'], croot) for sb in ('18MHz', '41MHz', '73MHz')]
        ran, calls, real = [], [], TS.budget_map

        def spy(fn_, items, charge_of, budget, cap, run=None, name_of=None, **kw):
            calls.append(dict(charges={it[1]: charge_of(it) for it in items}, budget=budget, cap=cap, kw=kw,
                              free=kw['mem_now_gb'](), names=[name_of(it) for it in items]))
            with Patch((TS, 'mem_available_gb', lambda: None)):
                calls[-1]['free_unreadable'] = kw['mem_now_gb']()
            return real(fn_, items, charge_of, budget, cap, run, name_of, **kw)
        run = types.SimpleNamespace(dry_run=False, units=[], out_root=td, step='masters')
        sel = TS.Selection.one_night(lst, night, ['18MHz', '41MHz', '73MHz'])
        stage_ledger = os.path.join(td, f'stage_{os.getuid()}.json')
        with Patch((TS, 'load_plan', lambda *a, **k: (jobs, [], 'synthetic plan')),
                   (TS, '_stack_unit', lambda run_, opts, root, l_, c, sb, S, cr, files: ran.append((c, sb, tuple(S), cr))),
                   (TS, 'mem_available_gb', lambda: 400), (TS, 'budget_map', spy),
                   (sec, 'STACK_NODE_LEDGER', os.path.join(td, 'stage_{uid}.json'))):
            said = quiet(TS.stage_stack, run, TS.Opts(), sel, td, lst_test=False)
            n_budget = len(calls)
            quiet(TS.stage_stack, run, TS.Opts(workers=2), sel, td, lst_test=False)
        c = calls[0] if calls else {}
        check('the stage ran the three units once under the budget (0.8 of the 400 GB available, at most 6 units and no '
              'more than the cores), and again under --workers 2 without the budget',
              n_budget == 1 and len(calls) == 1 and sorted(ran) == sorted([(cell, sb, ('I', 'V'), croot) for _, sb, _, _ in jobs] * 2)
              and c['budget'] == int(sec.STACK_MEM_FRACTION * 400) and 1 <= c['cap'] <= sec.STACK_MAX_UNITS
              and c['names'] == [f'stack_{lst}_{night}_{sb}' for sb in ('18MHz', '41MHz', '73MHz')], f'{calls} {ran}')
        want = {'18MHz': float(G['low']), '41MHz': G['mid'] * U, '73MHz': G['high'] * 680 / 358.0}
        check('the stage charges each unit by the NFRAMES card of its own Stokes I cube: 358 frames the class charge, 680 '
              'frames x 680/358, no cube (unknown) x the unknown factor',
              set(c['charges']) == set(want) and all(abs(c['charges'][k] - want[k]) < 1e-9 for k in want), str(c['charges']))
        line = [x for x in said if x.startswith(f'[sublst stack {lst}] 3 units')]
        check('the stage log line says how many units are charged above the class charge (1) and how many have an unknown '
              'frame count (1), and names the node ledger',
              len(line) == 1 and '1 unit(s) charged above' in line[0] and '1 with an unknown frame count' in line[0]
              and f'node ledger {stage_ledger}' in line[0], str(said))
        nd = c['kw'].get('node')
        check('the stage hands budget_map the look (the fraction of MemAvailable now; None, not 0, when it is unreadable), the '
              'memory held by its child processes and the node ledger at the configured path with the user id filled in; '
              'the ledger is empty after', c['free'] == sec.STACK_MEM_FRACTION * 400 and c['free_unreadable'] is None
              and c['kw'].get('held_gb') is TS.descendants_held_gb
              and isinstance(nd, TS.NodeLedger) and nd.path == stage_ledger and nd.off is None and read(stage_ledger) == {},
              f"{c['free']} {c['kw']} {getattr(nd, 'path', None)} {getattr(nd, 'off', None)}")


TESTS = [test_config, test_grammar, test_outcome, test_cpv, test_stack_unit, test_stack_budget, test_stack_budget_node,
         test_stack_gate, test_loo_gates,
         test_sources_prep_waivers, test_admit_noinput, test_loo_products,
         test_snapshot_units, test_snapshot_sources, test_real_mirrors, test_real_content, test_real_stack_night_dry,
         test_real_snapshot_parents]


def main(argv):
    want = set(argv)
    failed = 0
    try:
        for t in TESTS:
            if want and t.__name__ not in want:
                continue
            try:
                t()
                print(f'PASS {t.__name__}')
            except Exception as e:
                failed += 1
                import traceback
                print(f'FAIL {t.__name__}: {type(e).__name__}: {e}')
                traceback.print_exc()
    finally:
        if 'dir' in _STATE:
            shutil.rmtree(_STATE['dir'], ignore_errors=True)
            try:
                os.rmdir(SCRATCH_PARENT)
            except OSError:
                pass
    print(f'{len([t for t in TESTS if not want or t.__name__ in want]) - failed} passed, {failed} failed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
