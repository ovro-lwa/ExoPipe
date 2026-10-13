#!/usr/bin/env python3
"""STANDING TEST - unit masters-tiers: `exopipe3 masters --tier sublst|snapshot` (lib/masters/tier_sublst.py, tier_snapshot.py,
tests_sublst.py, tests_snapshot.py, _cfgdraft_masters-tiers.py).

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


TESTS = [test_config, test_grammar, test_outcome, test_cpv, test_stack_unit, test_stack_gate, test_loo_gates,
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
