#!/usr/bin/env python3
"""STANDING TEST - `exopipe3 mosaic` (all-sky HEALPix mosaics of the deep masters), synthetic data plus read-only real products.

  config     the five mosaic sections load; every RULED ID is a row of clv15/fix_ledger.csv; protected keys refuse --set;
             G0 passes against the REAL in-place code (engine, renderer, VLSSr check, the two exopipe_ops scripts)
  jobs       job grammar (<scope>_<variant>[_<product>], 'variant band domain' lines), default 76 jobs in scheduling
             order, patterns, @file, --list-jobs
  end2end    the whole step on a synthetic master generation (24 LSTs x Blue x 4 variants, small grids, NSIDE 16): the
             REAL stage script runs in place (ROOT pointed at the synthetic tree, pb_apply.wideband_pb replaced by a
             synthetic PB - a stand-in written into the test scratch), the REAL engine and renderer in place; every
             content test passes; the coadd maps equal a direct call of the engine on the same stage (bit-identical);
             staged names and provenance as the production conformance pass writes them; --resume skips every unit;
             the first run is a --parity run (write guard in the step and in every child)
  gates      G1 (an excluded constituent, also through the weighting-PB reference master), G3 (two candidates), G4
             (--lsts subset; waiver), G7 (P3_PB_* set), G11, G0 (an engine literal changed), G12 (explicit NoTaper conform;
             with the waiver it goes on to staging); end2end also: G13 (a jysr render missing from the render record)
  guard      G9: a CORRUPT-FLAT tile skipped and recorded (skip-and-record), census = engine, partial build for the
             coverage test; TILE_GUARD_ACTION refuse refuses
  qa         the qa stage at NSIDE 2048 on a synthetic noise map: the REAL anchor script and VLSSr check in place; tables
             pass mosaic_qa_table; conform/_tables copies them
  negative   content tests fail on corrupted copies (BUNIT, dec floor, a staged name whose N disagrees, pixels changed)
  real       READ-ONLY: content tests on the production qa table and, when present on this host, a staged weighting PB
             map; MOSAIC_TEST_REAL=1 adds the 200 MB maps, renders and archived files (slow; results in plan/impl/mosaic.md)

Scratch: /tmp/exopipe3_tests_mosaic/ (removed at the end). Nothing outside it is written; production trees are only read.
Run standalone:  python tests/test_mosaic.py [test_name ...]
"""
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
SCRATCH = '/tmp/exopipe3_tests_mosaic'
GEN = os.path.join(SCRATCH, 'Coadd', 'Run_TEST')
DD = os.path.join(SCRATCH, 'data_decisions.csv')
REAL_STAGE = '/lustre/gh/main/exopipe_ops/mosaic_stage_from_lustre.py'
REAL_ROOT = "ROOT = '/lustre/pipeline/exopipe/phase3/Coadd/Run_20260909_000000'"
LSTS = [f'{h:02d}h' for h in range(24)]
TOK = {'Taper_R075': 'Taper_Robust-0.75', 'Taper_R0': 'Taper_Robust+0.0', 'NoTaper_R075': 'NoTaper_Robust-0.75',
       'NoTaper_R0': 'NoTaper_Robust+0.0'}

from exopipe.phase3 import cli                                     # noqa: E402
from exopipe.phase3.config import CFG                              # noqa: E402
from exopipe.phase3.core import record, validate                   # noqa: E402
from exopipe.phase3.lib.mosaic import steps_common as SC           # noqa: E402

_SAVED = {}


def _patch(section, key, value):
    sec = getattr(CFG, section)
    _SAVED.setdefault((section, key), getattr(sec, key))
    setattr(sec, key, value)


def _restore():
    for (section, key), v in _SAVED.items():
        setattr(getattr(CFG, section), key, v)
    _SAVED.clear()
    CFG.reset()
    SC.SCRIPT_PATHS.clear()


def _main(argv):
    """Run `exopipe3 <argv>` in process; returns (rc, stdout, stderr). Overrides of one run never leak into the next."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = cli.main(list(argv))
        except SystemExit as e:                     # cli._guard_out refuses with SystemExit('refused: ...')
            rc = e.code if isinstance(e.code, int) else 2
            if not isinstance(e.code, int):
                err.write(str(e.code))
    CFG.reset()
    return rc, out.getvalue(), err.getvalue()


def _rec(out, unit):
    return record.load_record(out, 'mosaic', unit)


# ---------------------------------------------------------------------------------------------------------------------
# synthetic master generation
# ---------------------------------------------------------------------------------------------------------------------

N, CD = 48, 2.4          # 48 px x 2.4 deg: the whole sky above the horizon of a zenith-centred SIN tile, as real masters


def _master(path, lst, seed, nights=('2025-01-01', '2025-01-02')):
    import numpy as np
    from astropy.io import fits
    rng = np.random.RandomState(seed)
    x = (np.arange(N) + 1 - (N / 2 + 0.5)) * np.radians(CD)
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    d = rng.normal(0, 0.05, (N, N)).astype(np.float32)
    d[N // 2, N // 2 + 3] += 5.0
    d[r2 >= 0.97] = np.nan
    h = fits.Header()
    h['CTYPE1'], h['CTYPE2'], h['CTYPE3'], h['CTYPE4'] = 'RA---SIN', 'DEC--SIN', 'FREQ', 'STOKES'
    h['CRVAL1'], h['CRVAL2'] = (15.0 * (int(lst[:2]) + 0.5)) % 360.0, 37.215849167
    h['CRPIX1'] = h['CRPIX2'] = N / 2 + 0.5
    h['CDELT1'], h['CDELT2'] = -CD, CD
    h['CRVAL3'], h['CDELT3'], h['CRPIX3'] = 75.414192e6, 1e6, 1.0
    h['CRVAL4'], h['CDELT4'], h['CRPIX4'] = 1.0, 1.0, 1.0
    h['RESTFRQ'] = 75.414192e6
    h['BMAJ'], h['BMIN'], h['BPA'] = 0.1 + 0.001 * int(lst[:2]), 0.09, 0.0
    h['PBCORR'], h['PBMODEL'], h['PBCHROM'], h['PBRESID'] = True, 'test', 'bs-sqbp-v1', 'test'
    mjd = 60676.0 + int(lst[:2]) / 24.0
    h['MJD-OBS'], h['MJD-MID'], h['INTTIME'] = mjd, mjd + 0.5, 7200.0
    h['DATE-END'] = '2025-01-02T%02d:00:00.000' % int(lst[:2])
    h['GITTAG'] = 'test-p1'
    h['CONSTDIG'] = f'test{seed}'
    for n in nights:
        h['HISTORY'] = f'COADD constituent {n}'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.writeto(path, d[None, None], h, overwrite=True)


def _name(scope, var, lst, mdewarp=False):
    return (f'{scope}_I_deep_{TOK[var]}_shflux_{"mdewarp_" if mdewarp else ""}pbcorr_dewarped_aligned_coadd_LST{lst}_'
            f'20250101-20250102_N002.fits')


def _make_generation():
    shutil.rmtree(GEN, ignore_errors=True)
    for i, lst in enumerate(LSTS):
        for j, var in enumerate(TOK):
            _master(os.path.join(GEN, lst, 'Blue', _name('Blue', var, lst, mdewarp=(lst == '05h'))), lst, 100 * i + j)
    with open(DD, 'w') as f:
        f.write('lst,subband,date,verdict,reason\n00h,18MHz,2024-12-18,exclude,test row\n')


def _stand_in_stage_script():
    """The REAL stage script source with ROOT pointed at the synthetic generation, and pb_apply.wideband_pb replaced by a
    synthetic PB (the PB library cannot judge synthetic masters). Tests only - written into the test scratch."""
    src = open(REAL_STAGE).read()
    assert REAL_ROOT in src, 'the production stage script no longer holds the expected ROOT line'
    src = src.replace(REAL_ROOT, f'ROOT = {GEN!r}')
    head = (f"# synthetic stand-in of {REAL_STAGE} (tests only)\nROOT = {GEN!r}\n"
            "import numpy as _np\nfrom astropy.io import fits as _fits\n"
            "import exopipe.phase3.lib.masters.pb_apply as _pa\n"
            "def _fake_wideband_pb(scope, lst, master_path, **kw):\n"
            "    h = _fits.getheader(master_path)\n"
            "    x = (_np.arange(h['NAXIS1']) + 1 - h['CRPIX1']) * _np.radians(abs(h['CDELT1']))\n"
            "    y = (_np.arange(h['NAXIS2']) + 1 - h['CRPIX2']) * _np.radians(abs(h['CDELT2']))\n"
            "    r2 = x[None, :] ** 2 + y[:, None] ** 2\n"
            "    return _np.sqrt(_np.sqrt(_np.clip(1 - r2, 0, 1))).astype(_np.float32), "
            "{'used_dates': ['2025-01-01', '2025-01-02'], 'chromatic': 'test', 'residual_pb': 'test'}\n"
            "_pa.wideband_pb = _fake_wideband_pb\n")
    path = os.path.join(SCRATCH, 'stage_stand_in.py')
    with open(path, 'w') as f:
        f.write(head + "_SRC = r'''" + src + "'''\nexec(compile(_SRC, " + repr(REAL_STAGE) + ", 'exec'))\n")
    return path


def _small_world():
    """Synthetic generation + small grids. NSIDE is PROTECTED (the renderer, anchor script and VLSSr check hard-code 2048):
    the test patches it on the class and drops the NSIDE rows of the G0 table, nothing else."""
    _patch('mosaic', 'INPUT_COADD_RUN', GEN)
    _patch('paths', 'DATA_DECISIONS', DD)
    _patch('mosaic_coadd', 'NSIDE', 16)
    SC.SCRIPT_PATHS[SC.STAGE_SCRIPT] = _stand_in_stage_script()
    orig = SC._literal_table
    if not getattr(orig, '_test_wrapped', False):
        def table(C):
            return [r for r in orig(C) if r[0] != 'mosaic_coadd.NSIDE']
        table._test_wrapped = True
        table._orig = orig
        SC._literal_table = table


def _unsmall():
    if getattr(SC._literal_table, '_test_wrapped', False):
        SC._literal_table = SC._literal_table._orig
    _restore()


SMALL = ['--set', 'mosaic_coadd.COMMON_GRID_N_PIX=100', '--set', 'mosaic_coadd.COMMON_GRID_CDELT_DEG=1.2',
         '--set', 'mosaic_render.MOLLWEIDE_WIDTH_PX=360', '--set', 'mosaic_render.CAR_CDELT_ARCMIN=60.0',
         '--cores', 'none', '--workers', '4']


# ---------------------------------------------------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------------------------------------------------

def test_config():
    from exopipe.phase3.core import ledger
    from exopipe.phase3.config import ConfigRefusal
    for sec in ('mosaic', 'mosaic_coadd', 'mosaic_render', 'mosaic_qa', 'mosaic_conform'):
        s = getattr(CFG, sec)
        for key, ids in (getattr(s, 'RULED', None) or {}).items():
            for lid in (ids if isinstance(ids, tuple) else (ids,)):
                assert ledger.has_id(lid, CFG.paths.FIX_LEDGER), f'{sec}.{key}: {lid} not a ledger row'
            assert hasattr(s, key), f'{sec}.RULED names a missing key {key}'
        for key in getattr(s, 'PROTECTED_KEYS', ()):
            assert hasattr(s, key), f'{sec}.PROTECTED_KEYS names a missing key {key}'
            try:
                CFG.set(f'{sec}.{key}', repr(getattr(s, key)))
                raise AssertionError(f'{sec}.{key} accepted an override')
            except ConfigRefusal:
                pass
    assert CFG.mosaic.LSTS is None and CFG.mosaic.REQUIRE_N_LSTS == 24
    assert len(CFG.mosaic.SCOPES) == 19
    probs = SC.g0_check()
    assert probs == [], probs
    CFG.reset()


def test_jobs():
    import argparse
    J = SC.parse_job('27MHz_NoTaper_R0')
    assert (J.scope, J.variant, J.product, J.robust, J.taper) == ('27MHz', 'NoTaper_R0', 'I', 'R0', 'NoTaper')
    assert SC.parse_job('Blue_Taper_R075_I').name == 'Blue_Taper_R075'
    J = SC.parse_job('Taper_R075 Blue jybeam')
    assert J.name == 'Blue_Taper_R075' and J.domains == ('jybeam',)
    for bad in ('Blue', 'Purple_Taper_R0', 'Blue_Taper_R1', 'Taper_R075 Blue jyx'):
        try:
            SC.parse_job(bad)
            raise AssertionError(bad)
        except ValueError:
            pass

    def ns(**kw):
        d = dict(jobs=None, scopes=None, variants=None, products=None, domains=None)
        d.update(kw)
        return argparse.Namespace(**d)
    jobs = SC.select_jobs(ns())
    assert len(jobs) == 76 and jobs[0].name == 'Red_NoTaper_R075' and jobs[-1].name == '82MHz_Taper_R0'
    assert [j.name for j in SC.select_jobs(ns(jobs='*_Taper_R075', scopes='Blue,Red'))] == ['Red_Taper_R075',
                                                                                            'Blue_Taper_R075']
    os.makedirs(SCRATCH, exist_ok=True)
    jf = os.path.join(SCRATCH, 'jobs.txt')
    with open(jf, 'w') as f:
        f.write('Taper_R075 Blue jysr\nTaper_R075 Blue jybeam\nNoTaper_R0 27MHz jybeam\n# comment\n')
    jobs = SC.select_jobs(ns(jobs='@' + jf))
    assert [(j.name, j.domains) for j in jobs] == [('Blue_Taper_R075', ('jysr', 'jybeam')),
                                                   ('27MHz_NoTaper_R0', ('jybeam',))], jobs
    rc, out, _ = _main(['mosaic', '--list-jobs', '--scopes', 'Full', '--variants', 'Taper_R0,NoTaper_R075'])
    assert rc == 0 and out.split() == ['Full_NoTaper_R075', 'Full_Taper_R0'], out


def test_end2end():
    import numpy as np
    from astropy.io import fits
    _make_generation()
    _small_world()
    try:
        out = os.path.join(SCRATCH, 'out')
        work = os.path.join(SCRATCH, 'work')
        shutil.rmtree(out, ignore_errors=True)
        t0 = time.time()
        rc, so, se = _main(['mosaic', '--jobs', 'Blue_Taper_R075', '--stages', 'inputs,coadd,render,conform',
                            '--out', out, '--work', work, '--keep-stage', '--parity'] + SMALL)
        assert rc == 0, (rc, se[-3000:])
        assert not os.path.exists(os.path.join(out, '_records', '_parity_guard_refusals.log')), 'parity guard refused'
        wall = time.time() - t0
        for unit in ('inputs/Blue_Taper_R075', 'coadd/Blue_Taper_R075', 'render/Blue_Taper_R075',
                     'conform/Blue_Taper_R075'):
            r = _rec(out, unit)
            assert r and r['status'] == 'done', (unit, r and r['status'], r and r['reason'])
            assert all(t['passed'] for t in r['tests']), [t for t in r['tests'] if not t['passed']]
        assert _rec(out, 'conform/_tables')['status'] == 'no-input'
        co = _rec(out, 'coadd/Blue_Taper_R075')
        assert len(co['outputs']) == 4 and co['meta']['contributing'] == LSTS and co['admitted'][0] == '00h:2025-01-01'
        assert co['config_overrides'] and all(o[4] is None or o[4].startswith('non-production') for o in
                                              co['config_overrides'])
        # bit-identical to a direct call of the production engine on the same stage (side A of the parity plan, in small)
        stage = [l for l in se.splitlines() if 'stage kept at' in l][0].split('stage kept at ')[1].split()[0]
        from exopipe.phase3.lib.mosaic import coadd_jysr as mcj
        for k, v in SC.engine_globals().items():
            setattr(mcj, k, v)
        mcj.GRID_N, mcj.CDELT = 100, 1.2
        ref = os.path.join(SCRATCH, 'refcoadd')
        import warnings
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for dom in ('jysr', 'jybeam'):
                mcj.coadd_jysr(LSTS, 'Taper_R075', 'Blue', root=os.path.join(stage, 'Blue'), out_dir=ref, domain=dom)
        for o in co['outputs']:
            a = fits.getdata(o['path'], 1)['SIGNAL']
            b = fits.getdata(os.path.join(ref, os.path.basename(o['path'])), 1)['SIGNAL']
            assert np.array_equal(a, b, equal_nan=True), o['path']
        # staged names as the production conformance pass builds them; provenance = 24 tiles in LST order
        stg = os.path.join(out, 'archive_staging', 'mosaic', 'wideband')
        names = sorted(os.listdir(stg))
        want = sorted(['Blue_I_deep_Taper_Robust-0.75_pbcorr_coadd_mosaic_jysr_feather_allsky_20250101-20250102_N024.fits',
                       'Blue_I_deep_Taper_Robust-0.75_pbcorr_coadd_mosaic_feather_psfmap_allsky_20250101-20250102_N024.fits',
                       'Blue_I_deep_Taper_Robust-0.75_coadd_mosaic_jybeam_feather_allsky_20250101-20250102_N024.fits',
                       'Blue_I_deep_Taper_Robust-0.75_coadd_mosaic_feather_pbmap_allsky_20250101-20250102_N024.fits'])
        assert names == want, names
        from exopipe.phase3.core.products import read_provenance
        psf_staged = [n for n in want if 'psfmap' in n][0]
        prov = read_provenance(os.path.join(stg, psf_staged))
        assert len(prov) == 24 and prov[5]['FILE'].endswith(_name('Blue', 'Taper_R075', '05h', mdewarp=True))
        h0 = fits.getheader(os.path.join(stg, psf_staged))
        assert h0['NSTACK'] == 24 and h0['TIMEFIX'] == 'v2-mosaic-constituents' and h0['PBCORR'] is True
        assert abs(h0['BMAJ'] - np.median([0.1 + 0.001 * i for i in range(24)])) < 1e-12 and 'THERMNOI' not in h0
        assert len(os.listdir(os.path.join(out, 'archive_staging', 'mosaic', 'renders', 'wideband', 'mollweide'))) == 2
        # --resume: every unit skipped
        rc, so, se = _main(['mosaic', '--jobs', 'Blue_Taper_R075', '--stages', 'inputs,coadd,render,conform',
                            '--out', out, '--work', work, '--resume'] + SMALL)
        assert rc == 0, se[-2000:]
        if record.git_commit(REPO).endswith('-dirty'):        # other units edit tracked files in this worktree
            assert 'dirty tree' in se and 'never resumed' in se, se[-2000:]
            print('  end2end: worktree dirty - checked that --resume refuses to trust the records (record.py rule)')
        else:
            assert se.count('skipped (resume') >= 3, se[-2000:]
        # G13 (review 2026-10-13): a jysr render missing from the render record refuses the conform unit and stages
        # nothing (it used to be skipped silently while the jybeam renders refused). Records copied into another --out.
        out3 = os.path.join(SCRATCH, 'out_g13')
        rdir = os.path.join(out3, '_records', 'mosaic')
        for st in ('coadd', 'render'):
            os.makedirs(os.path.join(rdir, st), exist_ok=True)
            r = _rec(out, f'{st}/Blue_Taper_R075')
            if st == 'render':
                r['outputs'] = [o for o in r['outputs'] if '_jysr_' not in os.path.basename(o['path'])]
            with open(os.path.join(rdir, st, 'Blue_Taper_R075.json'), 'w') as f:
                json.dump(r, f)
        rc, so, se = _main(['mosaic', '--jobs', 'Blue_Taper_R075', '--stages', 'conform', '--out', out3,
                            '--cores', 'none'] + SMALL)
        r3 = _rec(out3, 'conform/Blue_Taper_R075')
        assert rc == 2 and r3['status'] == 'refused' and 'G13' in r3['reason'] and 'render record' in r3['reason'], r3
        assert not os.listdir(os.path.join(out3, 'archive_staging', 'mosaic', 'wideband')), 'half-staged files left'
        shutil.rmtree(stage, ignore_errors=True)
        print(f'  end2end: {wall:.0f} s for one job (stage, census, 2 co-adds, 2 renders, conform)')
        return out
    finally:
        _unsmall()


def test_gates():
    _make_generation()
    _small_world()
    try:
        out = os.path.join(SCRATCH, 'out_gates')
        base = ['mosaic', '--out', out, '--dry-run', '--cores', 'none']
        rc, so, _ = _main(base + ['--jobs', 'Blue_Taper_R075'])
        assert rc == 0 and 'gates G1/G3/G4/G8 pass' in so, so[-1500:]
        # G1 through the weighting-PB reference: exclude a night of the Taper R0 master's LST -> NoTaper R0 refused too
        with open(DD, 'a') as f:
            f.write('07h,Blue,2025-01-02,exclude,test exclusion\n')
        rc, so, _ = _main(base + ['--jobs', 'Blue_NoTaper_R0'])
        assert rc == 2 and 'G1: 07h' in so and 'weighting PB comes from' in so, so[-1500:]
        _make_generation()
        # G3: two current-master candidates
        _master(os.path.join(GEN, '11h', 'Blue', _name('Blue', 'Taper_R0', '11h', mdewarp=True)), '11h', 999)
        rc, so, _ = _main(base + ['--jobs', 'Blue_Taper_R0'])
        assert rc == 2 and 'G3: 11h: 2 current-master candidates' in so, so[-1500:]
        _make_generation()
        # G4: fewer LSTs than REQUIRE_N_LSTS, then waived
        rc, so, _ = _main(base + ['--jobs', 'Blue_Taper_R075', '--lsts', '00h,01h'])
        assert rc == 2 and 'G4' in so, so[-800:]
        rc, so, _ = _main(base + ['--jobs', 'Blue_Taper_R075', '--lsts', '00h,01h', '--waive', 'mosaic.inputs_complete',
                                  '--ruling', 'test waiver'])
        assert rc == 0 and 'G4 WAIVED' in so, so[-800:]
        # G8: PBCORR not True
        from astropy.io import fits
        p = os.path.join(GEN, '03h', 'Blue', _name('Blue', 'Taper_R075', '03h'))
        with fits.open(p, mode='update') as h:
            h[0].header['PBCORR'] = False
        rc, so, _ = _main(base + ['--jobs', 'Blue_Taper_R075'])
        assert rc == 2 and 'G8: 03h: PBCORR' in so, so[-800:]
        _make_generation()
        # G7: a PB environment variable set
        os.environ['P3_PB_THREE_FACTOR'] = '0'
        try:
            rc, _so, se = _main(base + ['--jobs', 'Blue_Taper_R075'])
        finally:
            os.environ.pop('P3_PB_THREE_FACTOR')
        assert rc == 2 and 'G7' in se, se[-800:]
        # G11: APPLY_FLUX_ANCHOR True (a ruled key: allowed into a non-production --out, then refused by the step)
        rc, _so, se = _main(base + ['--jobs', 'Blue_Taper_R075', '--set', 'mosaic_qa.APPLY_FLUX_ANCHOR=True'])
        assert rc == 2 and 'G11' in se, se[-800:]
        # protected key: refused by config
        rc, _so, se = _main(base + ['--jobs', 'Blue_Taper_R075', '--set', 'mosaic_coadd.MIN_VALID_TILE_PIXELS=50'])
        assert rc == 2 and 'PROTECTED' in se, se[-800:]
        # G0: an engine literal differs from the config (simulated by a patched config value)
        _patch('mosaic_coadd', 'CORRUPT_FLAT_SIGMA_JYBEAM', 2e-4)
        rc, _so, se = _main(base + ['--jobs', 'Blue_Taper_R075'])
        assert rc == 2 and 'G0' in se and 'CORRUPT_FLAT' in se, se[-800:]
        setattr(CFG.mosaic_coadd, 'CORRUPT_FLAT_SIGMA_JYBEAM', _SAVED.pop(('mosaic_coadd', 'CORRUPT_FLAT_SIGMA_JYBEAM')))
        # production --out refused by cli (G15)
        rc, _so, se = _main(['mosaic', '--jobs', 'Blue_Taper_R075', '--stages', 'qa', '--cores', 'none'])
        assert rc == 2 and 'production root' in se, se[-800:]
        # G12: conform of an explicitly named NoTaper job
        out2 = os.path.join(SCRATCH, 'out_g12')
        rc, _so, se = _main(['mosaic', '--jobs', 'Blue_NoTaper_R075', '--stages', 'conform', '--out', out2,
                             '--cores', 'none'])
        r = _rec(out2, 'conform/Blue_NoTaper_R075')
        assert rc == 2 and r['status'] == 'refused' and 'G12' in r['reason'], (rc, r)
        # G12 waived (review 2026-10-13): the job goes on to staging (here refused for its missing coadd record, never
        # silently skipped as before) and the unit carries the waiver
        rc, _so, se = _main(['mosaic', '--jobs', 'Blue_NoTaper_R075', '--stages', 'conform', '--out', out2,
                             '--cores', 'none', '--waive', 'mosaic.conform_taper_only', '--ruling', 'test waiver'])
        r = _rec(out2, 'conform/Blue_NoTaper_R075')
        assert rc == 2 and r['status'] == 'refused' and 'G12' not in r['reason'] and 'coadd record' in r['reason'] \
            and any('G12 WAIVED' in n for n in r['notes']), (rc, r)
    finally:
        _unsmall()


def test_guard():
    """G9 (review 2026-10-13): a CORRUPT-FLAT tile under TILE_GUARD_ACTION 'skip-and-record' (today's behaviour) is skipped
    by the engine and recorded; the census agrees with the engine; the maps count as a partial build in the coverage
    content test (23 contributing tiles). TILE_GUARD_ACTION 'refuse' refuses the job instead."""
    from astropy.io import fits
    _make_generation()
    _small_world()
    try:
        p = os.path.join(GEN, '06h', 'Blue', _name('Blue', 'Taper_R0', '06h'))
        with fits.open(p, mode='update') as h:
            h[0].data[...] = 0.0                       # flat: apparent MAD 0 -> the 1e-6 floor < 1e-4 = CORRUPT-FLAT
        out = os.path.join(SCRATCH, 'out_guard')
        work = os.path.join(SCRATCH, 'work_guard')
        rc, so, se = _main(['mosaic', '--jobs', 'Blue_Taper_R0', '--stages', 'inputs,coadd', '--out', out, '--work', work]
                           + SMALL)
        assert rc == 0, (rc, se[-3000:])
        co = _rec(out, 'coadd/Blue_Taper_R0')
        assert co['status'] == 'done' and co['meta']['skipped'] == [['06h', 'CORRUPT-FLAT']], co['meta'].get('skipped')
        assert len(co['meta']['contributing']) == 23 and '06h' not in co['meta']['contributing']
        assert all(o['ctx']['partial'] is True for o in co['outputs']), [o['ctx'].get('partial') for o in co['outputs']]
        assert any('CORRUPT-FLAT' in n for n in co['notes']), co['notes']
        rc, so, se = _main(['mosaic', '--jobs', 'Blue_Taper_R0', '--stages', 'inputs', '--out', out, '--work', work,
                            '--set', 'mosaic.TILE_GUARD_ACTION=refuse'] + SMALL)
        r = _rec(out, 'inputs/Blue_Taper_R0')
        assert rc == 2 and r['status'] == 'refused' and 'G9' in r['reason'] and '06h CORRUPT-FLAT' in r['reason'], (rc, r)
    finally:
        _unsmall()


def _hpx_map(path, values, bunit, extra):
    from astropy.io import fits
    hdu = fits.BinTableHDU.from_columns([fits.Column(name='SIGNAL', format='E', array=values)])
    for k, v in [('PIXTYPE', 'HEALPIX'), ('ORDERING', 'RING'), ('NSIDE', 2048), ('COORDSYS', 'C'),
                 ('INDXSCHM', 'IMPLICIT'), ('PBCORR', True), ('FEATHER', 'PB 0.15->0.1 cosine weight taper'),
                 ('ALIGNED', False), ('FREQ', 75414192.0), ('BUNIT', bunit)] + list(extra):
        hdu.header[k] = v
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


def test_qa():
    """qa at NSIDE 2048 on a synthetic noise map with the REAL anchor script and VLSSr check, then conform/_tables."""
    import numpy as np
    import healpy as hp
    out = os.path.join(SCRATCH, 'out_qa')
    hpx = os.path.join(out, 'healpix')
    os.makedirs(hpx, exist_ok=True)
    npix = hp.nside2npix(2048)
    rng = np.random.RandomState(7)
    m = (1000.0 + rng.normal(0, 50.0, npix)).astype(np.float32)
    th, _ = hp.pix2ang(2048, np.arange(0, npix, 1))
    m[(90 - np.degrees(th)) < -40] = np.nan
    psf = np.where(np.isfinite(m), 5e-6, np.nan).astype(np.float32)
    del th
    mp = os.path.join(hpx, 'healpix_I_deep_Taper_R075_Blue_jysr_nside2048.fits')
    pp = os.path.join(hpx, 'healpix_PSF_I_deep_Taper_R075_Blue_nside2048.fits')
    _hpx_map(mp, m, 'Jy/sr', [('JY2K', 1e-26 * 2.99792458e8 ** 2 / (2 * 1.380649e-23 * 75414192.0 ** 2))])
    _hpx_map(pp, psf, 'sr', [])
    del m, psf
    # the qa stage reads the coadd record (never a glob): a minimal done record for the synthetic job
    rec = {'status': 'done', 'unit': 'coadd/Blue_Taper_R075', 'outputs': [{'path': mp, 'class': 'mosaic_map'},
                                                                         {'path': pp, 'class': 'mosaic_psfmap'}],
           'meta': {'domains': ['jysr']}}
    record.atomic_write_json(record.record_path(out, 'mosaic', 'coadd/Blue_Taper_R075'), rec)
    rc, so, se = _main(['mosaic', '--jobs', 'Blue_Taper_R075', '--domains', 'jysr', '--stages', 'qa,conform',
                        '--out', out, '--cores', 'none'])
    q = _rec(out, 'qa/_all')
    assert q and q['status'] == 'done', (rc, q and q['reason'], se[-2500:])
    assert all(t['passed'] for t in q['tests']), [t for t in q['tests'] if not t['passed']]
    assert _rec(out, 'conform/_tables')['status'] == 'done'
    # conform of the job is refused (no render record), the tables are staged
    assert _rec(out, 'conform/Blue_Taper_R075')['status'] == 'refused' and rc == 2
    an = [o['path'] for o in q['outputs'] if 'flux_anchor' in o['path']][0]
    rows = open(an).read().splitlines()
    assert rows[0] == 'variant,band,freq_mhz,source,aperture_jy,model_jy,ratio,fwhm_eff_arcmin' and len(rows) == 8, rows
    os.remove(mp)
    os.remove(pp)


def test_negative(out):
    """Content tests fail on corrupted copies of the end2end products."""
    import numpy as np
    from astropy.io import fits
    neg = os.path.join(SCRATCH, 'neg')
    os.makedirs(neg, exist_ok=True)
    _patch('mosaic_coadd', 'NSIDE', 16)
    try:
        src = os.path.join(out, 'healpix', 'healpix_I_deep_Taper_R075_Blue_jysr_nside16.fits')
        assert all(r['passed'] for r in validate.run('mosaic_map', src))
        bad = os.path.join(neg, os.path.basename(src))
        shutil.copy(src, bad)
        with fits.open(bad, mode='update') as h:
            h[1].header['BUNIT'] = 'Jy/beam'
            d = h[1].data['SIGNAL']
            d[-1] = 1.0                                   # a finite pixel at the south pole
        res = {r['name']: r['passed'] for r in validate.run('mosaic_map', bad)}
        assert not res['mosaic_map.domain_cards'] and not res['mosaic_map.dec_floor'], res
        stg = os.path.join(out, 'archive_staging', 'mosaic', 'wideband')
        f = [x for x in os.listdir(stg) if 'jysr' in x][0]
        bad = os.path.join(neg, f.replace('N024', 'N023'))
        shutil.copy(os.path.join(stg, f), bad)
        res = {r['name']: r['passed'] for r in validate.run('mosaic', bad)}
        assert not res['mosaic.cards_vs_name'], res                  # NSTACK 24 vs N023
        bad2 = os.path.join(neg, f)
        shutil.copy(os.path.join(stg, f), bad2)
        with fits.open(bad2, mode='update') as h:
            h[1].data['SIGNAL'][100] = np.float32(7.0)
        res = {r['name']: r['passed'] for r in validate.run('mosaic', bad2, source=src)}
        assert not res['mosaic.pixels_unchanged'], res
    finally:
        _restore()


def test_real():
    """READ-ONLY content tests on production products present on this host."""
    found = 0
    qa = '/lustre/pipeline/exopipe/phase3/mosaic/Run_FEATHER2_20241218-20251029/tables/flux_anchor_measured_20260929.csv'
    if os.path.exists(qa):
        res = validate.run('mosaic_qa_table', qa)
        print('  real', os.path.basename(qa), [(r['name'], r['passed'], r['detail'][:80]) for r in res])
        assert all(r['passed'] for r in res), res
        found += 1
    pb = '/fast/gh/main/phase3/mosaic_v2/masters/00h/PB_deep_Blue.fits'      # lwacalim07 only; staged 2026-09-26..30
    if os.path.exists(pb):
        res = validate.run('mosaic_weight_pb', pb)
        print('  real', pb, [(r['name'].split('.')[-1], r['passed'], r['detail'][:70]) for r in res])
        found += 1
    heavy = os.environ.get('MOSAIC_TEST_REAL') == '1'
    if heavy:
        v2 = '/fast/gh/main/phase3/mosaic_v2'
        arch = '/lustre/pipeline/exopipe/phase3/mosaic/Run_FEATHER2_20241218-20251029/Blue/I'
        for cls, p in (('mosaic_map', f'{v2}/healpix/healpix_I_deep_Taper_R075_Blue_jysr_nside2048.fits'),
                       ('mosaic_map', f'{v2}/healpix/healpix_I_deep_Taper_R075_Blue_jybeam_nside2048.fits'),
                       ('mosaic_psfmap', f'{v2}/healpix/healpix_PSF_I_deep_Taper_R075_Blue_nside2048.fits'),
                       ('mosaic_pbeff', f'{v2}/healpix/healpix_PBeff_I_deep_Taper_R075_Blue_nside2048.fits'),
                       ('mosaic_render', f'{v2}/renders/Blue_Taper_R075_jysr_RADec.fits'),
                       ('mosaic', f'{arch}/healpix/Blue_I_deep_Taper_Robust-0.75_pbcorr_coadd_mosaic_jysr_feather_allsky_'
                                  f'20241218-20251029_N024.fits'),
                       ('mosaic', f'{arch}/radec/Blue_I_deep_Taper_Robust-0.75_coadd_mosaic_jybeam_feather_carrender_allsky_'
                                  f'20241218-20251029_N024.fits')):
            if os.path.exists(p):
                res = validate.run(cls, p)
                print(f'  real {cls} {os.path.basename(p)}:',
                      '; '.join(f"{r['name'].split('.')[-1]} {'PASS' if r['passed'] else 'FAIL'} {r['detail'][:90]}"
                                for r in res))
                found += 1
    print(f'  real: {found} production file(s) checked' + ('' if heavy else ' (MOSAIC_TEST_REAL=1 adds the 200 MB maps)'))


def main(names):
    t0 = time.time()
    shutil.rmtree(SCRATCH, ignore_errors=True)
    os.makedirs(SCRATCH)
    try:
        order = ['config', 'jobs', 'end2end', 'gates', 'guard', 'qa', 'negative', 'real']
        out = None
        for n in order:
            if names and n not in names and not (n == 'end2end' and 'negative' in names):
                continue
            t = time.time()
            if n == 'negative':
                test_negative(out)
            elif n == 'end2end':
                out = test_end2end()
            else:
                globals()[f'test_{n}']()
            print(f'PASS {n} ({time.time() - t:.0f} s)')
    finally:
        _restore()
        shutil.rmtree(SCRATCH, ignore_errors=True)
    print(f'all mosaic tests passed in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main(sys.argv[1:])
