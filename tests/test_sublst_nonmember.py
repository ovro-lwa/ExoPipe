#!/usr/bin/env python3
"""STANDING TEST (data-free) - the 10-minute residual of ONE night that is not in the window Masters:
`exopipe3 frames --tier sublst --stages nonmember` (lib/masters/sublst_nonmember.py, class sublst_nonmember_residual in
lib/masters/tests_sublst.py, the stage in lib/frames/tier_sublst.py).

Ruling: Gregg 2026-11-12 - nights that are not in a Master are searched as non-members against the frozen Masters.

Everything is synthetic: small arrays in a temporary directory of this test's own (tempfile.mkdtemp, removed at the end;
no fixed /tmp path, no cluster file, no in-place production code). Each test builds an installed <LST>_masters/ with
production's window-Master formula (clv15/phase3_sublst_loo.py:132-135) from member arrays that are NEVER written to disk
- so a stage that read another night's stack could not run - and the new night's stack with its admission record, plan
and stack records.

Asserts:
  residual     residual = float32(window stack - INSTALLED window Master) bit for bit, NaN exactly where either has no
               data; production's residual cards with production's values; the stack's own header kept; BINMSTR removed;
               the unit done and its content test passing; the difference from production's float64-mean form bounded
               by half a float32 step of the Master (the documented cause of non-bit-parity with production)
  membership   a night with weight above 0 in an installed window Master is REFUSED and the refusal names
               `masters --tier sublst --stages loo`; a night listed with weight 0 is served
  masters      no installed Master of a Stokes: unit refused, the batch stage named; one window's Master missing: item
               refusal, the other windows written; no window with a Master: unit refused, nothing moved aside; an
               unusable Master (COADDOP, NCOADD, CONSTDIG, HISTORY, no weights record, an excluded constituent): refused,
               by the same function as the Master's own content test
  content test a residual that fails it: every residual of the unit removed, unit refused (N-12); never waivable
  dry run      agrees with the real run under --resume (a unit valid by its record is skipped, not refused)
  stack        no plan; no done stack record; a window file missing; a sibling missing; no admission record; windows not
               those of the record; 'log' evidence with and without the completion line
  exclusion    a night excluded in data_decisions.csv is refused by the readers the stack stage uses; nothing written
  other gates  differing flux scales; an unreadable time block; earlier residuals (refused; moved aside with --fresh
               --supersede); nothing owed (no-input, never done); a dry run writes nothing
  frames       the stage is in frames_sublst.ORDER after stack, in no default, and runs through lib/frames/tier_sublst.run

Run standalone:  python tests/test_sublst_nonmember.py [test_name ...]
Under pytest:    pytest tests/test_sublst_nonmember.py
"""
import argparse
import csv
import glob
import hashlib
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

# The stage uses the exclusion readers of the stack stage; one of them is the co-add's (lib/masters/coadd.py), which
# imports the Phase 2 stack (reproject, pandas): the site default interpreter has it (py38_orca_nkosogor; the versions the
# CI job installs), py310_orca does not. An interpreter without it cannot run the stage either: UNTESTED-IMPORT, exit 2.
_IMPORT_ERROR = None
try:
    import numpy as np
    from astropy.io import fits

    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import exclusions as EX
    from exopipe.phase3.core import record, validate
    from exopipe.phase3.lib.frames import tier_sublst as FT
    from exopipe.phase3.lib.masters import coadd as CO
    from exopipe.phase3.lib.masters import sublst_nonmember as NM
    from exopipe.phase3.lib.masters import tests_sublst as TST
    from exopipe.phase3.lib.masters import tier_sublst as TS
except Exception as e:  # noqa: BLE001
    _IMPORT_ERROR = e
    if __name__ != '__main__':          # under pytest: a collection error that names the missing module
        raise

LST = '09h'
NEW = '2024-02-01'                      # the night that is not in the Masters (synthetic)
CELL = f'{LST}_{NEW}'
MEMBERS = ['2024-01-01', '2024-01-02', '2024-01-03']        # the Masters' constituent nights (synthetic)
WEIGHTS = [0.8, 1.3, 0.55]              # their night weights in the window Masters
SHAPE = (12, 10)
T_INT = 10.03102041                     # s, one 10-s integration (masters_sublst.INTEGRATION_S)
BATCH = 'masters --tier sublst --stages loo'


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


class Tree:
    """One synthetic sub-LST root in a temporary directory of its own, with its own exclusion list. Use as a context
    manager: the exclusion list is pointed at for both readers (core.exclusions and the co-add's) inside the block."""

    def __init__(self, sbs=('18MHz', '73MHz'), windows=(0, 1, 2, 3), stokes=('I', 'V'), excluded=(), seed=0,
                 records=True, step='frames'):
        self.sbs, self.windows, self.stokes = tuple(sbs), tuple(windows), tuple(stokes)
        self.dir = tempfile.mkdtemp(prefix='sublst_nonmember_')
        self.root = os.path.join(self.dir, 'sublst')
        self.rng = np.random.RandomState(seed)
        self.data = {}                  # (sb, S, k) -> {'stack': float32, 'mean64': float64 window mean, 'master': float32}
        self.decisions = os.path.join(self.dir, 'data_decisions.csv')
        self.write_decisions(excluded)
        for sb in self.sbs:
            for S in self.stokes:
                for k in range(6):
                    self._master(sb, S, k)
                self._stack(sb, S)
        if records:
            self.write_plan(step)
            for sb in self.sbs:
                self.write_stack_record(sb, step)
        self._patch = Patch((CFG.paths, 'DATA_DECISIONS', self.decisions), (CO, 'DECISIONS_CSV', self.decisions))

    def __enter__(self):
        self._patch.__enter__()
        return self

    def __exit__(self, *exc):
        self._patch.__exit__(*exc)
        shutil.rmtree(self.dir, ignore_errors=True)

    # ---- the exclusion list ------------------------------------------------------------------------------------------
    def write_decisions(self, excluded):
        """excluded: [(LST, subband, night)] rows with verdict 'exclude' (plus one unrelated row, so the file is never empty)."""
        with open(self.decisions, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['lst', 'subband', 'date', 'verdict', 'reason'])
            w.writerow(['23h', '82MHz', '2029-12-31', 'exclude', 'synthetic row of another cell'])
            for lst, sb, night in excluded:
                w.writerow([lst, sb, night, 'exclude', 'synthetic exclusion'])
        st = os.stat(self.decisions)                      # the readers cache by mtime: make every rewrite visible
        os.utime(self.decisions, ns=(st.st_atime_ns, st.st_mtime_ns + 1000000 * (1 + len(excluded))))

    # ---- the installed window Masters (production's formula; the member arrays are never written) ----------------------
    def master_paths(self, sb, S, k):
        return NM.master_products(self.root, LST, sb, S, k)

    def _master(self, sb, S, k):
        arrays = [self.rng.normal(5.0, 1.0, SHAPE).astype(np.float32).astype(np.float64) for _ in MEMBERS]
        arrays[0][0, 0] = arrays[1][0, 0] = arrays[2][0, 0] = np.nan       # no member has data here: the Master is NaN
        arrays[1][3, 4] = np.nan                                           # one member lacks this pixel: still a mean
        Ssum, W = np.zeros(SHAPE), np.zeros(SHAPE)
        for a, w in zip(arrays, WEIGHTS):
            v = np.isfinite(a)
            Ssum[v] += w * a[v]
            W[v] += w
        mean64 = np.where(W > 0, Ssum / np.where(W > 0, W, 1.0), np.nan)
        self.data[(sb, S, k)] = {'mean64': mean64, 'master': mean64.astype(np.float32)}
        self.write_master(sb, S, k)

    def write_master(self, sb, S, k, rows=None, **cards):
        """(Re)write one window Master and its weights record. rows: [(night, weight)] (default the members)."""
        rows = list(rows) if rows is not None else list(zip(MEMBERS, WEIGHTS))
        mp, wp = self.master_paths(sb, S, k)
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        h = fits.Header()
        h['LSTHOUR'], h['LSTWIN'] = LST, k
        h['NCOADD'] = sum(1 for _n, w in rows if w > 0)
        h['COADDOP'] = CFG.masters_sublst.WINDOW_COADDOP
        h['JUPMASK'] = 30.52
        h['CONSTDIG'] = hashlib.md5(','.join(sorted(n for n, _w in rows)).encode()).hexdigest()
        h['SHSCALE'], h['FLUXCORR'], h['FLUXBAKE'] = 1.1, 0.9, True
        for n, w in sorted(rows):
            h['HISTORY'] = f'window-Master constituent {n}' if w > 0 else f'NOT in the Master (deep co-add rejected) {n}'
        for key, val in cards.items():
            if val is None:
                h.remove(key, ignore_missing=True)
            else:
                h[key] = val
        fits.writeto(mp, self.data[(sb, S, k)]['master'], h, overwrite=True)
        with open(wp, 'w') as f:
            f.write('night,weight\n' + ''.join(f'{n},{w}\n' for n, w in rows))

    # ---- the new night's stack (windows, siblings, admission record) ---------------------------------------------------
    def stack_dir(self, sb, S):
        return os.path.join(self.root, LST, CELL, sb, S)

    def window(self, sb, S, k):
        return os.path.join(self.stack_dir(sb, S), f'sublst_{CELL}_{sb}_{S}_w{k}.fits')

    def resid(self, sb, S, k):
        return self.window(sb, S, k)[:-5] + '_resid_loo.fits'

    def residuals(self):
        return sorted(glob.glob(os.path.join(self.root, LST, CELL, '*', '*', '*_resid_loo.fits')))

    def _stack(self, sb, S):
        d = self.stack_dir(sb, S)
        os.makedirs(d, exist_ok=True)
        rows = []
        for k in self.windows:
            img = self.rng.normal(5.0, 1.5, SHAPE).astype(np.float32)
            img[5, 5] = np.nan                                              # the night has no data here
            self.data[(sb, S, k)]['stack'] = img
            h = fits.Header()
            h['BUNIT'] = 'Jy/beam'
            h['DATE-OBS'] = (f'{NEW}T10:{k}0:00.000', 'first frame start (its own DATE-OBS)')
            h['DATE-END'] = (f'{NEW}T10:{k}0:50.155', 'last frame end')
            h['LSTHOUR'], h['LSTWIN'], h['NFRAMES'] = LST, k, 5
            h['BMAJ'] = 0.25
            h['SHSCALE'], h['FLUXCORR'], h['FLUXBAKE'] = 1.1, 0.9, True
            h['NONMEMB'] = (True, 'night outside the deep co-add: never a window-Master constituent')
            h['HISTORY'] = 'pixels = weighted mean of frames / FLUXDC x SHSCALE x FLUXCORR'
            h['CODE'] = ('abc1234', 'clv15 phase3_sublst_stack.py')
            h['PRODUCT'] = 'sublst-v0-test'
            if k == self.windows[0]:
                h['BINMSTR'] = 'a card the residual must not carry'
            fits.writeto(self.window(sb, S, k), img, h, overwrite=True)
            fits.writeto(self.window(sb, S, k)[:-5] + '_nframes.fits', np.full(SHAPE, 5, np.int16), overwrite=True)
            fits.writeto(self.window(sb, S, k)[:-5] + '_pb.fits', np.full(SHAPE, 0.5, np.float32), overwrite=True)
            for i in range(5):
                rows.append(dict(ts=f'x{k}{i}', ts_int='x', date_obs=f'{NEW}T10:{k}0:{10 * i:02d}.000', t_mid_unix='0.000',
                                 lst_min=f'{10 * k + i}.000', window=k, finite_frac='1.000', beam_area_deg2='0.2',
                                 noise_jyb='2.0', sun_elev_deg='-50.00', sun_branch='morning', used=1, refused=''))
        rows.append(dict(ts='blank', ts_int='x', date_obs=f'{NEW}T10:59:00.000', t_mid_unix='0.000', lst_min='59.000',
                         window=5, finite_frac='0.100', beam_area_deg2='0.2', noise_jyb='', sun_elev_deg='-50.00',
                         sun_branch='morning', used=0, refused='blank'))
        with open(os.path.join(d, f'admission_{CELL}_{sb}_{S}.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=TST.ADMISSION_COLUMNS)
            w.writeheader()
            w.writerows(rows)

    # ---- the records of the stack stage (the night's plan, the done stack units) ---------------------------------------
    def write_plan(self, step='frames', sbs=None, status='done'):
        jf, jdf = (os.path.join(self.root, '_plans', f'sublst_jobs_{CELL}{sfx}.txt') for sfx in ('', '_deepflow'))
        os.makedirs(os.path.dirname(jf), exist_ok=True)
        with open(jf, 'w') as f:
            f.write(''.join(f'{CELL} {sb} {S}\n' for sb in (self.sbs if sbs is None else sbs) for S in ('I', 'V')))
        open(jdf, 'w').close()
        sha = {p: hashlib.sha1(open(p, 'rb').read()).hexdigest() for p in (jf, jdf)}
        record.atomic_write_json(record.record_path(self.root, step, f'admit_{CELL}'),
                                 {'status': status, 'finished': '2024-02-01T12:00:00Z', 'meta': {'plan_sha1': sha},
                                  'reason': 'G-S03: every (cell, subband) left out by rule' if status == 'no-input' else '',
                                  'outputs': [{'path': p, 'class': 'sublst_job_plan'} for p in (jf, jdf)]})

    def write_stack_record(self, sb, step='frames', status='done', windows=None):
        outs = [{'path': os.path.abspath(self.window(sb, S, k)), 'class': 'sublst_window'}
                for S in self.stokes for k in (self.windows if windows is None else windows)]
        record.atomic_write_json(record.record_path(self.root, step, TS.stack_unit_name(LST, NEW, sb)),
                                 {'status': status, 'finished': '2024-02-01T12:05:00Z', 'outputs': outs})

    def write_log(self, sb, S, line=True, windows=None):
        p = TS.stack_products(self.root, LST, CELL, sb, S)['log']
        os.makedirs(os.path.dirname(p), exist_ok=True)
        n = len(self.windows) if windows is None else windows
        with open(p, 'w') as f:
            f.write('stacker output\n' + (f'SUBLST {CELL} {sb} {S} frames=21 used=20 blank=1 windows={n}\n' if line else ''))

    # ---- running the stage ---------------------------------------------------------------------------------------------
    def run(self, sbs=None, stokes=('I', 'V'), dry=False, step='frames', **opts):
        """Run the stage for the new night; returns (the record.Run, dry-run refusal count)."""
        r = record.Run(step, ['test'], self.root, CFG, dry_run=dry)
        sel = TS.Selection.one_night(LST, NEW, sbs or self.sbs, stokes)
        bad = NM.stage_nonmember(r, TS.Opts(workers=1, **opts), sel, self.root)
        return r, bad


def unit(run, sb=None):
    name = NM.unit_name(LST, NEW, sb) if sb else f'nonmember_{CELL}'
    hits = [u for u in run.units if u.name == name]
    assert len(hits) == 1, (name, [(u.name, u.status) for u in run.units])
    return hits[0]


def refused(run, sb, *needles):
    u = unit(run, sb)
    assert u.status == 'refused', (u.name, u.status, u.reason)
    for n in needles:
        assert n in u.reason, (n, u.reason)
    return u


# =====================================================================================================================
# the residual
# =====================================================================================================================

def test_residual_is_window_minus_installed_master():
    with Tree() as t:
        run, _ = t.run()
        n_differ = n_pix = 0
        for sb in t.sbs:
            u = unit(run, sb)
            assert u.status == 'done', (u.status, u.reason)
            assert len(u.outputs) == 8 and not u.refused_items and all(x['passed'] for x in u.tests), u.tests
            assert u.admitted == [f'{NEW}/{sb}']
            for S in t.stokes:
                for k in t.windows:
                    d = t.data[(sb, S, k)]
                    stack64, master64 = d['stack'].astype(np.float64), d['master'].astype(np.float64)
                    want = np.where(np.isfinite(stack64) & np.isfinite(master64), stack64 - master64, np.nan).astype(np.float32)
                    with fits.open(t.resid(sb, S, k), memmap=False) as hd:
                        got = np.asarray(hd[0].data, dtype=np.float32)
                        h = hd[0].header.copy()
                        assert hd[0].header['BITPIX'] == -32
                    # bit for bit, NaN positions included
                    assert np.array_equal(np.isnan(got), np.isnan(want))
                    assert np.isnan(got[0, 0]) and np.isnan(got[5, 5]) and np.isfinite(got[3, 4])
                    v = np.isfinite(want)
                    assert np.array_equal(got[v].view(np.uint32), want[v].view(np.uint32))
                    # production subtracts the float64 mean before it is rounded for the Master file: the two forms agree to
                    # half a float32 step of the Master value plus the two final roundings, and not always bit for bit
                    prod = np.where(np.isfinite(stack64) & np.isfinite(d['mean64']), stack64 - d['mean64'], np.nan).astype(np.float32)
                    bound = (np.spacing(np.abs(d['master'][v])) / 2 + np.spacing(np.abs(got[v])) / 2
                             + np.spacing(np.abs(prod[v])) / 2).astype(np.float64) + 1e-12
                    assert np.all(np.abs(got[v].astype(np.float64) - prod[v].astype(np.float64)) <= bound)
                    n_differ += int(np.sum(got[v] != prod[v]))
                    n_pix += int(v.sum())
                    # production's cards, production's values (clv15/phase3_sublst_loo.py:169-184)
                    assert h['SUBTRACT'] == 'leave-one-out window Master'
                    assert h['LOONCOAD'] == 3 and h['LOOMSTR'] == os.path.basename(t.master_paths(sb, S, k)[0])
                    assert h['JUPMASK'] == 30.52 and h['LSTBIN'] == k and h['DURATION'] == 50.2
                    assert h.comments['LOONCOAD'] == 'nights in the leave-one-out Master'
                    assert h.comments['CODE'] == NM.CODE_COMMENT and h['CODE'] == run.commit[:68]
                    assert 'BINMSTR' not in h
                    # the stack's own header is kept (its own time block; the stacker's non-member mark is inherited)
                    sh = fits.getheader(t.window(sb, S, k))
                    for key in ('DATE-OBS', 'DATE-END', 'LSTHOUR', 'LSTWIN', 'NFRAMES', 'BMAJ', 'SHSCALE', 'FLUXCORR',
                                'FLUXBAKE', 'NONMEMB', 'PRODUCT', 'BUNIT'):
                        assert h[key] == sh[key] and h.comments[key] == sh.comments[key], key
                    assert list(h['HISTORY']) == list(sh['HISTORY'])
        assert n_pix > 1000
        print(f'    {n_differ} of {n_pix} synthetic pixels differ from the float64-mean form (within the bound)')
        # the record names what was read: the night's stacks and the installed Masters, no other night's file
        ins = unit(run, t.sbs[0]).inputs
        assert any(p.endswith('_weights.csv') for p in ins) and t.decisions in ins
        assert not any(m in p for p in ins for m in MEMBERS), 'an input of another night'
        # the content test fails on a perturbed pixel (exact identity) and on a NaN that should hold data
        p = t.resid('18MHz', 'I', 1)
        with fits.open(p, mode='update') as hd:
            hd[0].data[2, 2] = np.nextafter(hd[0].data[2, 2], np.float32(np.inf))
        r = validate.run('sublst_nonmember_residual', p)
        assert len(r) == 1 and not r[0]['passed'] and '1 pixel(s) differ' in r[0]['detail'], r
        with fits.open(p, mode='update') as hd:
            hd[0].data[2, 2] = np.nan
        r = validate.run('sublst_nonmember_residual', p)
        assert not r[0]['passed'] and '1 NaN position(s) differ' in r[0]['detail'], r


def test_arithmetic_function():
    a = np.array([[1.5, np.nan], [2.0, 4.0]], np.float32)
    m = np.array([[0.25, 1.0], [np.nan, 4.0]], np.float32)
    r = NM.subtract(a, m)
    assert r.dtype == np.float32 and r[0, 0] == np.float32(1.25) and r[1, 1] == 0.0
    assert np.isnan(r[0, 1]) and np.isnan(r[1, 0])
    try:
        NM.subtract(a, m[:1])
        raise AssertionError('shapes that differ were subtracted')
    except record.Refusal:
        pass


# =====================================================================================================================
# membership
# =====================================================================================================================

def test_member_night_is_refused():
    with Tree(sbs=('18MHz',)) as t:
        # the night is a constituent (weight above 0) of ONE installed window Master, in one Stokes
        t.write_master('18MHz', 'V', 2, rows=list(zip(MEMBERS, WEIGHTS)) + [(NEW, 0.7)])
        run, _ = t.run()
        refused(run, '18MHz', 'N-5', 'IS a member', BATCH, 'w2 Stokes V')
        assert t.residuals() == [], 'a residual of a member night was written'
        # a dry run says the same and writes nothing
        _run, bad = t.run(dry=True)
        assert bad == 1 and t.residuals() == []


def test_night_listed_with_weight_zero_is_served():
    with Tree(sbs=('18MHz',), stokes=('I',)) as t:
        for k in range(6):                                 # listed, not a constituent (rejected by the deep co-add)
            t.write_master('18MHz', 'I', k, rows=list(zip(MEMBERS, WEIGHTS)) + [(NEW, 0.0)])
        run, _ = t.run(stokes=('I',))
        u = unit(run, '18MHz')
        assert u.status == 'done' and len(u.outputs) == 4, (u.status, u.reason)
        assert any('weight 0 (not a constituent)' in n for n in u.notes)
        assert fits.getheader(t.resid('18MHz', 'I', 0))['LOONCOAD'] == 3


# =====================================================================================================================
# the installed Masters
# =====================================================================================================================

def test_missing_installed_master():
    with Tree(sbs=('18MHz', '73MHz')) as t:
        # (a) no Master of a Stokes at all: the unit is refused, the batch stage named, nothing written
        shutil.rmtree(os.path.dirname(t.master_paths('18MHz', 'V', 0)[0]))
        # (b) one window's Master missing: that window is an item refusal, the others are written
        os.unlink(t.master_paths('73MHz', 'I', 2)[0])
        run, _ = t.run()
        refused(run, '18MHz', 'N-6', 'no installed window Master', BATCH)
        assert not glob.glob(os.path.join(t.root, LST, CELL, '18MHz', '*', '*_resid_loo.fits'))
        u = unit(run, '73MHz')
        assert u.status == 'done' and len(u.outputs) == 7, (u.status, u.reason)
        assert len(u.refused_items) == 1 and u.refused_items[0]['path'] == os.path.abspath(t.resid('73MHz', 'I', 2))
        assert 'N-7' in u.refused_items[0]['reason'] and BATCH in u.refused_items[0]['reason']
        assert not os.path.exists(t.resid('73MHz', 'I', 2))
    with Tree(sbs=('18MHz',), stokes=('I',), windows=(0, 1)) as t:
        # (c) the Masters exist only for windows the night has no stack in: every window refused -> the unit is refused
        for k in (0, 1):
            os.unlink(t.master_paths('18MHz', 'I', k)[0])
        run, _ = t.run(stokes=('I',))
        refused(run, '18MHz', 'N-7', 'no window of the night has an installed window Master')
        assert t.residuals() == []
        # a unit that can write nothing moves nothing aside either: an earlier residual stays where it is
        with open(t.resid('18MHz', 'I', 0), 'wb') as f:
            f.write(b'residual of an earlier run')
        run, _ = t.run(stokes=('I',), fresh=True, supersede='tag0')
        refused(run, '18MHz', 'N-7')
        assert open(t.resid('18MHz', 'I', 0), 'rb').read() == b'residual of an earlier run'
        assert not os.path.exists(os.path.join(t.root, '_superseded'))


def test_unusable_installed_master():
    for change, needle in ((dict(COADDOP='another-generation'), 'COADDOP'), (dict(NCOADD=7), 'NCOADD'),
                           (dict(JUPMASK=None), 'JUPMASK'), (dict(LSTWIN=4), 'LSTWIN'),
                           (dict(CONSTDIG='0' * 32), 'CONSTDIG')):
        with Tree(sbs=('18MHz',), stokes=('I',)) as t:
            t.write_master('18MHz', 'I', 1, **change)
            run, _ = t.run(stokes=('I',))
            refused(run, '18MHz', 'N-8', needle, BATCH)
            assert t.residuals() == []
    with Tree(sbs=('18MHz',), stokes=('I',)) as t:
        os.unlink(t.master_paths('18MHz', 'I', 3)[1])
        run, _ = t.run(stokes=('I',))
        refused(run, '18MHz', 'N-8', 'no weights record')
    with Tree(sbs=('18MHz',), stokes=('I',)) as t:           # one weighted night only: not a window Master
        t.write_master('18MHz', 'I', 0, rows=[(MEMBERS[0], 0.8), (MEMBERS[1], 0.0)])
        run, _ = t.run(stokes=('I',))
        refused(run, '18MHz', 'N-8', 'fewer than')
    with Tree(sbs=('18MHz',), stokes=('I',)) as t:           # the weights record gained a night after the Master was written
        with open(t.master_paths('18MHz', 'I', 2)[1], 'a') as f:
            f.write('2023-12-31,0.0\n')
        run, _ = t.run(stokes=('I',))
        refused(run, '18MHz', 'N-8', 'CONSTDIG', 'HISTORY')
        assert t.residuals() == []
    with Tree(sbs=('18MHz',), stokes=('I',)) as t:           # the stage's check IS the Master's own content test's
        mp, wp = t.master_paths('18MHz', 'I', 1)
        rows = NM.read_weights(wp)
        assert TST.master_record_problems(fits.getheader(mp), rows, 1) == []
        assert NM.master_problems(fits.getheader(mp), rows, 1, EX.load(), LST, '18MHz') == []
        t.write_master('18MHz', 'I', 1, CONSTDIG='0' * 32)
        shared = TST.master_record_problems(fits.getheader(mp), rows, 1)
        assert shared and NM.master_problems(fits.getheader(mp), rows, 1, EX.load(), LST, '18MHz') == shared
    with Tree(sbs=('18MHz',), stokes=('I',), excluded=[(LST, '18MHz', MEMBERS[1])]) as t:
        run, _ = t.run(stokes=('I',))                        # a constituent of the Master is excluded now
        refused(run, '18MHz', 'N-8', 'excluded in data_decisions.csv now', BATCH)
        assert t.residuals() == []


# =====================================================================================================================
# the night's stack
# =====================================================================================================================

def test_stack_missing_or_incomplete():
    with Tree(sbs=('18MHz',), records=False) as t:           # no plan: the stack stage has not run
        run, _ = t.run()
        u = unit(run)
        assert u.status == 'refused' and 'N-2' in u.reason and NM.STACK_STAGE in u.reason, u.reason
    with Tree(sbs=('18MHz',)) as t:                          # the plan owes the stack, its record is missing
        os.unlink(record.record_path(t.root, 'frames', TS.stack_unit_name(LST, NEW, '18MHz')))
        run, _ = t.run()
        refused(run, '18MHz', 'missing or incomplete', 'N-3', '(missing)', NM.STACK_STAGE)
    with Tree(sbs=('18MHz',)) as t:                          # the stack unit ended refused
        t.write_stack_record('18MHz', status='refused')
        run, _ = t.run()
        refused(run, '18MHz', 'N-3', '(refused)')
    with Tree(sbs=('18MHz',)) as t:                          # a window file is gone: admission and files disagree
        os.unlink(t.window('18MHz', 'V', 2))
        run, _ = t.run()
        refused(run, '18MHz', 'N-4')
        assert t.residuals() == []
    with Tree(sbs=('18MHz',)) as t:                          # a sibling is gone
        os.unlink(t.window('18MHz', 'I', 1)[:-5] + '_pb.fits')
        run, _ = t.run()
        refused(run, '18MHz', 'N-4', 'sibling')
    with Tree(sbs=('18MHz',)) as t:                          # no admission record
        os.unlink(os.path.join(t.stack_dir('18MHz', 'I'), f'admission_{CELL}_18MHz_I.csv'))
        run, _ = t.run()
        refused(run, '18MHz', 'N-4', 'no admission record')
    with Tree(sbs=('18MHz',)) as t:                          # the record names other windows than the disk holds
        t.write_stack_record('18MHz', windows=(0, 1, 2))
        run, _ = t.run()
        refused(run, '18MHz', 'N-4', 'stack record lists')
    with Tree(sbs=('18MHz', '73MHz'), step='masters') as t:  # plan and stack records of the masters step are found
        run, _ = t.run()
        assert unit(run, '18MHz').status == 'done' and unit(run, '73MHz').status == 'done'


def test_stack_log_evidence():
    with Patch((CFG.masters_sublst, 'NONMEMBER_STACK_EVIDENCE', 'log')):
        with Tree(sbs=('18MHz',)) as t:                      # production stacks: no stack record, the stacker's own line
            os.unlink(record.record_path(t.root, 'frames', TS.stack_unit_name(LST, NEW, '18MHz')))
            run, _ = t.run()
            refused(run, '18MHz', 'N-3', 'no completion line')
            for S in ('I', 'V'):
                t.write_log('18MHz', S, line=False)
            run, _ = t.run()
            refused(run, '18MHz', 'N-3', 'no completion line')
            for S in ('I', 'V'):
                t.write_log('18MHz', S, windows=6)           # the line counts windows the disk does not hold
            run, _ = t.run()
            refused(run, '18MHz', 'N-4', 'completion line')
            for S in ('I', 'V'):
                t.write_log('18MHz', S)
            run, _ = t.run()
            u = unit(run, '18MHz')
            assert u.status == 'done' and len(u.outputs) == 8 and u.meta['stack_evidence'] == 'log', (u.status, u.reason)
    with Patch((CFG.masters_sublst, 'NONMEMBER_STACK_EVIDENCE', 'none')):
        with Tree(sbs=('18MHz',)) as t:
            try:
                t.run()
                raise AssertionError('an unknown evidence kind was accepted')
            except record.Refusal:
                pass


# =====================================================================================================================
# exclusion, flux scale, time block, earlier products, nothing owed, dry run
# =====================================================================================================================

def test_excluded_night_is_refused():
    with Tree(excluded=[(LST, '18MHz', NEW)]) as t:
        run, _ = t.run()
        refused(run, '18MHz', 'N-1', 'excluded in data_decisions.csv')
        assert not glob.glob(os.path.join(t.root, LST, CELL, '18MHz', '*', '*_resid_loo.fits'))
        assert unit(run, '73MHz').status == 'done'           # the other subband of the night is not excluded
        # the two readers the stack stage uses must agree (G-M01): the co-add's reader pointed at another list refuses
        with Patch((CO, 'DECISIONS_CSV', os.path.join(t.dir, 'another.csv'))):
            run, _ = t.run(fresh=True, supersede='t1')
            assert unit(run, '73MHz').status == 'refused' and 'paths.DATA_DECISIONS' in unit(run, '73MHz').reason
    with Tree(sbs=('18MHz',), stokes=('I',)) as t:           # a residual of a night excluded SINCE fails its content test
        run, _ = t.run(stokes=('I',))
        assert unit(run, '18MHz').status == 'done'
        t.write_decisions([(LST, '18MHz', NEW)])
        r = validate.run('sublst_nonmember_residual', t.resid('18MHz', 'I', 0))
        assert not r[0]['passed'] and 'excluded in data_decisions.csv' in r[0]['detail'], r


def test_flux_scale_and_time_block():
    with Tree(sbs=('18MHz',), stokes=('I',)) as t:           # the stack carries another flux scale than the Master
        with fits.open(t.window('18MHz', 'I', 3), mode='update') as hd:
            hd[0].header['SHSCALE'] = 1.2
        run, _ = t.run(stokes=('I',))
        refused(run, '18MHz', 'N-9', 'flux scales')
        assert t.residuals() == []
    with Tree(sbs=('18MHz',), stokes=('I',)) as t:           # neither the stack nor the Master carries the correction in
        for q in [t.window('18MHz', 'I', k) for k in t.windows] + [t.master_paths('18MHz', 'I', k)[0] for k in range(6)]:
            with fits.open(q, mode='update') as hd:          # its pixels: the same scale, and still no residual (the
                hd[0].header['FLUXBAKE'] = False             # residual's content test requires FLUXBAKE)
        run, _ = t.run(stokes=('I',))
        refused(run, '18MHz', 'N-9', 'FLUXBAKE')
        assert t.residuals() == []
    with Tree(sbs=('18MHz',), stokes=('I',)) as t:           # no DATE-END: no DURATION is assumed, nothing is written
        with fits.open(t.window('18MHz', 'I', 3), mode='update') as hd:
            del hd[0].header['DATE-END']
        run, _ = t.run(stokes=('I',))
        refused(run, '18MHz', 'N-10', 'time block')
        assert t.residuals() == []


def test_earlier_residuals_and_supersede():
    with Tree(sbs=('18MHz',), stokes=('I',)) as t:
        run, _ = t.run(stokes=('I',))
        assert unit(run, '18MHz').status == 'done'
        first = open(t.resid('18MHz', 'I', 0), 'rb').read()
        run, _ = t.run(stokes=('I',))                        # a second run: the earlier residuals refuse (G-S01)
        refused(run, '18MHz', 'G-S01')
        run, _ = t.run(stokes=('I',), fresh=True, supersede='tag1')
        assert unit(run, '18MHz').status == 'done'
        moved = os.path.join(t.root, '_superseded', 'tag1', LST, CELL, '18MHz', 'I', os.path.basename(t.resid('18MHz', 'I', 0)))
        assert open(moved, 'rb').read() == first and os.path.isfile(os.path.join(t.root, '_superseded', 'tag1', 'MANIFEST.csv'))
        assert open(t.resid('18MHz', 'I', 0), 'rb').read() == first, 'the same inputs gave another residual file'
        # a residual of a window the night has no stack in (left by an earlier run) is never kept
        stale = t.window('18MHz', 'I', 5)[:-5] + '_resid_loo.fits'
        shutil.copy(t.resid('18MHz', 'I', 0), stale)
        run, _ = t.run(stokes=('I',), fresh=True, supersede='tag2')
        assert unit(run, '18MHz').status == 'done' and not os.path.exists(stale)


def test_failed_content_test_leaves_no_residual():
    assert 'sublst_nonmember_residual.nonmember_residual_identity' in TS.NEVER_WAIVABLE
    real = NM.write_residual

    def one_wrong(item, code):                               # window 2 is written with one pixel off by one float32 step
        p = real(item, code)
        if item['k'] == 2:
            with fits.open(p, memmap=False) as hd:
                d, h = np.array(hd[0].data, dtype=np.float32), hd[0].header.copy()
            i = np.flatnonzero(np.isfinite(d))[7]
            d.flat[i] = np.nextafter(d.flat[i], np.float32(np.inf))
            fits.writeto(p, d, h, overwrite=True)
        return p

    with Tree(sbs=('18MHz', '73MHz'), stokes=('I',)) as t:
        with Patch((NM, 'write_residual', one_wrong)):
            run, _ = t.run(stokes=('I',))
        for sb in ('18MHz', '73MHz'):
            u = refused(run, sb, 'N-12', f'1 of {len(t.windows)} residual(s) failed', 'removed')
            assert u.outputs == [] and [x for x in u.tests if not x['passed']], u.tests
        assert t.residuals() == [], 'a residual stayed where the wideband, tree and search stages read it'
        assert run.exit_code() == record.EXIT_REFUSED
        run, _ = t.run(stokes=('I',))                        # nothing was left behind: the next run needs no --supersede
        assert unit(run, '18MHz').status == 'done' and len(t.residuals()) == 2 * len(t.windows)


def test_dry_run_agrees_with_resume():
    with Tree(sbs=('18MHz',), stokes=('I',)) as t:
        run, _ = t.run(stokes=('I',))
        assert unit(run, '18MHz').status == 'done'
        _dry, bad = t.run(stokes=('I',), dry=True)           # without --resume the earlier residuals refuse (N-11)
        assert bad == 1
        for valid in (True, False):                          # --resume: skipped when the record is valid, as the real run
            with Patch((record, 'resume_check', lambda *a, _v=valid, **k: (_v, 'stand-in'))):
                real, _ = t.run(stokes=('I',), resume=True)
                _dry, bad = t.run(stokes=('I',), dry=True, resume=True)
            assert (unit(real, '18MHz').status == 'skipped-valid') == valid, unit(real, '18MHz').status
            assert bad == (0 if valid else 1), (valid, bad)
        real, _ = t.run(stokes=('I',), resume=True)          # and with the real check, whatever this checkout's state
        _dry, bad = t.run(stokes=('I',), dry=True, resume=True)
        assert (bad == 0) == (unit(real, '18MHz').status == 'skipped-valid')


def test_nothing_owed_is_no_input():
    with Tree(sbs=('18MHz',)) as t:
        run, _ = t.run(sbs=('73MHz',))                       # the plan holds no job of this subband
        u = unit(run)
        assert u.status == 'no-input' and run.exit_code() == record.EXIT_NO_INPUT, (u.status, u.reason)
        t.write_plan(status='no-input')                      # the admission left every subband out by rule
        run, _ = t.run()
        assert unit(run).status == 'no-input' and t.residuals() == []


def test_dry_run_writes_nothing():
    with Tree() as t:
        before = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(t.dir) for f in fs)
        run, bad = t.run(dry=True)
        assert bad == 0 and run.units == []
        assert sorted(os.path.join(dp, f) for dp, _, fs in os.walk(t.dir) for f in fs) == before


# =====================================================================================================================
# the frames step
# =====================================================================================================================

def test_frames_step_accepts_supersede():
    # the stage's refusal N-11 names --fresh --supersede TAG: the frames step must accept the flag it names
    p = argparse.ArgumentParser()
    FT.add_arguments(p)
    assert p.parse_args(['--supersede', 'tag1']).supersede == 'tag1' and p.parse_args([]).supersede is None


def test_frames_stage():
    S = CFG.frames_sublst
    order = list(S.ORDER)
    assert order.index('nonmember') == order.index('stack') + 1
    assert 'nonmember' not in S.STAGES['per-hour'] and 'nonmember' not in S.STAGES['intermittent']
    assert CFG.masters_sublst.NONMEMBER_STACK_EVIDENCE == 'record' and CFG.masters_sublst.PARALLEL['nonmember'] <= 6
    with Tree() as t:
        args = argparse.Namespace(epochs=CELL, cells=None, lsts=None, nights=None, scopes='18MHz', stokes=None,
                                  stages='nonmember', from_stage=None, until_stage=None, dry_run=False, workers=1,
                                  fresh=False, supersede=None, resume=False, sublst_groups=None, sublst_mirror=False,
                                  sublst_loo_evidence=None)
        run = record.Run('frames', ['test'], t.root, CFG)
        with Patch((TS, 'mirror_violations', lambda: [])):   # the in-place clv15 literals: not part of a data-free test
            rc = FT.run(args, run)
        assert rc is None and [u.name for u in run.units] == [NM.unit_name(LST, NEW, '18MHz')], [u.name for u in run.units]
        assert run.units[0].status == 'done' and run.exit_code() == record.EXIT_DONE
        assert len(t.residuals()) == 8
        rec = record.load_record(t.root, 'frames', NM.unit_name(LST, NEW, '18MHz'))
        assert rec['status'] == 'done' and {o['class'] for o in rec['outputs']} == {'sublst_nonmember_residual'}
        # without a named night the stage refuses (it never runs over every night of an LST)
        args.epochs, args.lsts = None, LST
        run = record.Run('frames', ['test'], t.root, CFG)
        FT.run(args, run)
        assert [u.status for u in run.units] == ['refused'] and 'ONE named night' in run.units[0].reason


# =====================================================================================================================
# the non-member nights list: one night put on it (stage nonmember-list)
# =====================================================================================================================

class ListInputs:
    """The list step's inputs inside a Tree's directory: the list, frozen deep weights per subband, a cube directory.
    rows: {subband: [(date, qa3_scale, in_master)], or None for no weights file}. A context manager that points the
    step's readers at them."""

    def __init__(self, t, rows, listed=(), cube=True, final_newline=True):
        self.list = os.path.join(t.dir, 'sublst_nonmember_nights.txt')
        text = '# SUB-LST NON-MEMBER NIGHTS (synthetic)\n' + ''.join(f'{l} {n}\n' for l, n in listed)
        with open(self.list, 'w') as f:
            f.write(text if final_newline else text.rstrip('\n'))
        self.wdir = os.path.join(t.dir, 'frozen')
        for sb, rr in rows.items():
            if rr is None:
                continue
            os.makedirs(os.path.join(self.wdir, LST, sb))
            with open(self.weights(sb), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['date', 'qa3_scale', 'in_master', 'weight'])
                w.writerows([d, q, m, 1.0] for d, q, m in rr)
        self.cubes = os.path.join(t.dir, 'cubes')
        if cube:
            os.makedirs(os.path.join(self.cubes, CELL))
        self.root = t.root
        self._patch = Patch((TS, 'nonmember_file', lambda: self.list),
                            (TS, 'frozen_weights', lambda lst, sb: os.path.join(self.wdir, lst, sb, 'weights.csv')),
                            (TS, 'cube_root', lambda: self.cubes),
                            (TS, 'deepflow_root', lambda: os.path.join(t.dir, 'deepflow')),
                            (TS, 'mirror_violations', lambda: []))    # the in-place clv15 literals: not data-free

    def weights(self, sb):
        return os.path.join(self.wdir, LST, sb, 'weights.csv')

    def __enter__(self):
        self._patch.__enter__()
        return self

    def __exit__(self, *exc):
        self._patch.__exit__(*exc)

    def text(self):
        return open(self.list).read()

    def go(self, sbs=('18MHz', '73MHz'), dry=False):
        """Run the step for the new night; returns (the action, the unit or None in a dry run, the record.Run)."""
        r = record.Run('frames', ['test'], self.root, CFG, dry_run=dry)
        res = NM.list_night(r, LST, NEW, subbands=sbs)
        hits = [u for u in r.units if u.name == NM.list_unit_name(LST, NEW)]
        return res['action'], (hits[-1] if hits else None), r


MEMBER_ROWS = [(d, 0.95, 1) for d in MEMBERS]


def production_parser(text, lst):
    """clv15 phase3_sublst_loo.nonmember_nights on a text (the parser of every production reader of the list)."""
    out = set()
    for ln in text.splitlines(True):
        p = ln.split()
        if len(p) >= 2 and not ln.startswith("#") and p[0] == lst:
            out.add(p[1])
    return out


def test_list_appends_a_night_without_a_usable_scale():
    with Tree() as t, ListInputs(t, {'18MHz': MEMBER_ROWS, '73MHz': MEMBER_ROWS + [(NEW, 0.9, 1)]},
                                 listed=[(LST, '2023-12-31'), ('10h', NEW)]) as li:
        before = li.text()
        action, u, run = li.go()
        assert action == 'append' and u.status == 'done', (action, u.status, u.reason)
        after = li.text()
        added = after[len(before):].splitlines()
        assert after.startswith(before) and len(added) == 2, added          # appended, never rewritten
        assert added[0].startswith(f'# {LST} {NEW}: added ') and '18MHz' in added[0] and NM.LIST_RULING in added[0]
        assert added[1] == f'{LST} {NEW}'
        assert production_parser(after, LST) == {'2023-12-31', NEW} == TS.nonmember_nights(LST)
        assert production_parser(after, '10h') == {NEW}                      # the other hour's line is untouched
        rec = record.load_record(t.root, 'frames', NM.list_unit_name(LST, NEW))
        m = rec['meta']['nonmember_list']
        assert m['action'] == 'append' and m['list'] == li.list and m['sha256_before'] != m['sha256_after']
        assert m['sha256_after'] == hashlib.sha256(open(li.list, 'rb').read()).hexdigest() and m['utc'].endswith('Z')
        by = {s['subband']: s for s in m['subbands']}
        assert by['18MHz']['row'] is False and by['18MHz']['usable'] is False and by['18MHz']['in_master'] is None
        assert by['73MHz']['row'] is True and by['73MHz']['usable'] is True and by['73MHz']['in_master'] == '1'
        for sb in ('18MHz', '73MHz'):
            assert by[sb]['weights'] == li.weights(sb)
            assert by[sb]['sha256'] == hashlib.sha256(open(li.weights(sb), 'rb').read()).hexdigest()
        out = NM.list_record_path(t.root, LST, NEW)
        assert [o['path'] for o in rec['outputs'] if o['class'] == NM.LIST_CLASS] == [out]
        assert validate.run(NM.LIST_CLASS, out)[0]['passed']
        action, u, _ = li.go()                                               # once: a second run adds nothing
        assert action == 'already-listed' and u.status == 'done' and li.text() == after
        again = record.load_record(t.root, 'frames', NM.list_unit_name(LST, NEW))['meta']['nonmember_list']
        assert again['action'] == 'already-listed' and again['appended_by']['utc'] == m['utc']    # why it is listed stays
        assert again['appended_by']['sha256_before'] == m['sha256_before']
        # the decision record's content test reads the list as it is NOW: a line removed by hand since fails it
        with open(li.list, 'w') as f:
            f.write(before)
        r = validate.run(NM.LIST_CLASS, out)
        assert not r[0]['passed'] and 'is not on' in r[0]['detail'], r


def test_list_writes_nothing_when_not_needed():
    usable = MEMBER_ROWS + [(NEW, 0.9, 0)]                   # a row with a usable scale, weight 0 in the Master
    with Tree() as t, ListInputs(t, {'18MHz': usable, '73MHz': MEMBER_ROWS + [(NEW, 1.1, 1)]}) as li:
        before = li.text()
        action, u, _ = li.go()
        assert action == 'not-needed' and u.status == 'done' and li.text() == before
    # a listed night whose subbands all have member rows is the normal state of the list (per night; membership is per
    # subband): recorded, never refused
    with Tree() as t, ListInputs(t, {'18MHz': usable, '73MHz': usable}, listed=[(LST, NEW)]) as li:
        before = li.text()
        action, u, _ = li.go()
        assert action == 'not-needed' and u.status == 'done' and li.text() == before
    # the stack admission's own test of 'usable': at or under the floor, not a number, NaN and no weights file all need the list
    floor = CFG.masters_sublst.QA3_ADMIT_FLOOR
    for bad in ([(NEW, floor, 1)], [(NEW, '', 1)], [(NEW, 'nan', 1)], None):
        rows = None if bad is None else MEMBER_ROWS + bad
        with Tree() as t, ListInputs(t, {'18MHz': rows, '73MHz': usable}) as li:
            assert NM.weights_state(LST, '18MHz', NEW)['usable'] is False, bad
            action, u, _ = li.go()
            assert action == 'append' and production_parser(li.text(), LST) == {NEW}, (bad, action)


def test_list_refusals():
    norow = {'18MHz': MEMBER_ROWS, '73MHz': MEMBER_ROWS}
    with Tree(excluded=[(LST, '18MHz', NEW), (LST, '73MHz', NEW)]) as t, ListInputs(t, norow) as li:
        before = li.text()
        action, u, _ = li.go()                               # excluded in every subband: never listed
        assert action == 'refuse' and u.status == 'refused' and 'L-1' in u.reason and li.text() == before, u.reason
    with Tree(excluded=[(LST, '18MHz', NEW)]) as t, ListInputs(t, {'18MHz': MEMBER_ROWS,
                                                                  '73MHz': MEMBER_ROWS + [(NEW, 0.9, 1)]}) as li:
        before = li.text()
        action, u, _ = li.go()                               # an excluded subband is never a reason to list the night
        assert action == 'not-needed' and li.text() == before
        rec = record.load_record(t.root, 'frames', NM.list_unit_name(LST, NEW))
        assert {s['subband']: s['excluded'] for s in rec['meta']['nonmember_list']['subbands']} == {'18MHz': True,
                                                                                                    '73MHz': False}
    with Tree() as t, ListInputs(t, norow, cube=False) as li:
        before = li.text()
        action, u, _ = li.go()                               # no cube of the night: no line for a mistyped night
        assert action == 'refuse' and 'L-3' in u.reason and li.text() == before, u.reason
    with Tree() as t, ListInputs(t, norow) as li:
        before = li.text()
        with Patch((NM, 'write_refusal', lambda path: 'this run may not write it')):
            action, u, _ = li.go()                           # needed, and this run may not write the list
            assert action == 'refuse' and 'L-5' in u.reason and li.text() == before, u.reason
    with Tree() as t, ListInputs(t, norow, listed=[(LST, NEW)]) as li:
        before = li.text()
        with Patch((NM, 'write_refusal', lambda path: 'this run may not write it')):
            action, u, _ = li.go()                           # already listed: nothing to write, so nothing refused
            assert action == 'already-listed' and u.status == 'done' and li.text() == before
    with Tree() as t, ListInputs(t, norow) as li:
        os.remove(li.list)
        action, u, _ = li.go()                               # the list is never created here
        assert action == 'refuse' and 'L-4' in u.reason and not os.path.exists(li.list), u.reason
    # the production list is not written while paths.PRODUCTION_WRITES is False (the refactor window)
    assert CFG.paths.PRODUCTION_WRITES is False
    why = NM.write_refusal(os.path.join(CFG.paths.OPS_ROOT, CFG.masters_sublst.NONMEMBER_NIGHTS_FILE))
    assert why and 'PRODUCTION_WRITES is False' in why, why


def test_list_append_is_locked_and_once():
    import fcntl
    import threading
    import time
    d = tempfile.mkdtemp(prefix='sublst_nonmember_')
    try:
        p = os.path.join(d, 'list.txt')
        with open(p, 'w') as f:
            f.write(f'# header\n{LST} 2023-12-31')            # no final newline: completed before the new lines
        assert NM.append_list_line(p, LST, NEW, 'why 1') is True
        assert open(p).read() == f'# header\n{LST} 2023-12-31\n# why 1\n{LST} {NEW}\n'
        assert NM.append_list_line(p, LST, NEW, 'why 2') is False and 'why 2' not in open(p).read()
        done = []
        with open(p) as held:                                # another writer holds the list's lock
            fcntl.flock(held, fcntl.LOCK_EX)
            th = threading.Thread(target=lambda: done.append(NM.append_list_line(p, LST, '2024-03-03', 'why 3')))
            th.start()
            time.sleep(0.5)
            assert th.is_alive() and not done and 'why 3' not in open(p).read()      # waits, writes nothing
            fcntl.flock(held, fcntl.LOCK_UN)
        th.join(20)
        assert done == [True] and open(p).read().endswith(f'# why 3\n{LST} 2024-03-03\n')
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_list_stage_in_frames():
    S = CFG.frames_sublst
    order = list(S.ORDER)
    assert order.index(NM.LIST_STAGE) < order.index('stack')                 # the list is read by the stack admission
    assert NM.LIST_STAGE not in S.STAGES['per-hour'] and NM.LIST_STAGE not in S.STAGES['intermittent']
    with Tree() as t, ListInputs(t, {'18MHz': MEMBER_ROWS}) as li:
        before = li.text()
        args = argparse.Namespace(epochs=CELL, cells=None, lsts=None, nights=None, scopes='18MHz', stokes=None,
                                  stages=NM.LIST_STAGE, from_stage=None, until_stage=None, dry_run=True, workers=1,
                                  fresh=False, supersede=None, resume=False, sublst_groups=None, sublst_mirror=False,
                                  sublst_loo_evidence=None)
        run = record.Run('frames', ['test'], t.root, CFG, dry_run=True)
        FT.run(args, run)
        assert li.text() == before                                           # a dry run writes nothing
        assert not os.path.exists(record.record_path(t.root, 'frames', NM.list_unit_name(LST, NEW)))
        args.dry_run = False
        run = record.Run('frames', ['test'], t.root, CFG)
        rc = FT.run(args, run)
        assert rc is None and [(u.name, u.status) for u in run.units] == [(NM.list_unit_name(LST, NEW), 'done')], run.units
        assert production_parser(li.text(), LST) == {NEW}
        args.epochs, args.lsts = None, LST                                   # without a named night the stage refuses
        run = record.Run('frames', ['test'], t.root, CFG)
        FT.run(args, run)
        assert [u.status for u in run.units] == ['refused'] and 'ONE named night' in run.units[0].reason


def test_content_tests_are_registered_under_their_names():
    """Every never-waivable '<class>.<test>' of the sub-LST classes is a registered content test of that class, and every
    registered sub-LST test takes the product's path first. (Found 2026-11-13: a helper inserted under the decorator of
    the window Master's test took its registration, so every window Master failed its content test with a TypeError; the
    tests that read real Masters need the site and are not part of the data-free job.)"""
    import inspect
    for full in TS.NEVER_WAIVABLE:
        cls, _, name = full.partition('.')
        if cls.startswith('sublst_'):
            assert name in [n for n, _ in validate.tests_for(cls)], (full, validate.tests_for(cls))
    assert [n for n, _ in validate.tests_for('sublst_window_master')] == ['window_master_record']
    n_tests = 0
    for cls in validate.classes():
        if cls.startswith('sublst_'):
            for name, fn in validate.tests_for(cls):
                n_tests += 1
                assert list(inspect.signature(fn).parameters)[0] == 'path', (cls, name)
    assert n_tests >= 10, n_tests


TESTS = [test_residual_is_window_minus_installed_master, test_arithmetic_function, test_member_night_is_refused,
         test_night_listed_with_weight_zero_is_served, test_missing_installed_master, test_unusable_installed_master,
         test_stack_missing_or_incomplete, test_stack_log_evidence, test_excluded_night_is_refused,
         test_flux_scale_and_time_block, test_earlier_residuals_and_supersede, test_failed_content_test_leaves_no_residual,
         test_dry_run_agrees_with_resume, test_nothing_owed_is_no_input, test_dry_run_writes_nothing,
         test_frames_step_accepts_supersede, test_frames_stage, test_list_appends_a_night_without_a_usable_scale,
         test_list_writes_nothing_when_not_needed, test_list_refusals, test_list_append_is_locked_and_once,
         test_list_stage_in_frames, test_content_tests_are_registered_under_their_names]


def main(argv):
    if _IMPORT_ERROR is not None:
        print(f'UNTESTED-IMPORT: {type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}')
        return 2
    want = set(argv)
    failed = 0
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
    print(f'{len([t for t in TESTS if not want or t.__name__ in want]) - failed} passed, {failed} failed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
