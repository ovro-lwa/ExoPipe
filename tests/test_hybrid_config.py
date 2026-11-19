#!/usr/bin/env python3
"""The hybrid route, pull request PR-1 (plan/specs/hybrid.md section 4): config sections, the step skeleton and its gates, the
token-card rule of the zero-spacing fill, and the names of the archived hybrid products and mosaics.

Data-free (CI):
  1. the config sections of spec 3.3 exist with their keys; RULED and PROTECTED name existing keys; the scope lists agree with
     each other (CAVEAT_SCOPES and STRIATION_SCOPES inside SCOPES; the wideband members are the 15 subbands = hybrid_full.MEMBERS);
     a PROTECTED key cannot be overridden;
  2. the unit plan: stage order, the scopes each stage takes, refusals of an unknown stage, scope, weighting or range;
  3. the stage gates (HG0 COADD_SB absent, HG6 empty PSF registry or a PSF whose md5 differs from its entry, striation selection
     md5, unfilled inputs) and the --out gate; the PSF registry pins every subband x weighting (Q1);
  4. the token-card rule (lib/mosaic/tests.fill_card_problems) on literal headers copied from archived files (read 2026-11-18);
  5. the CLI: `exopipe3 hybrid` refuses every unit of an unported stage with its reason and exits 2; `exopipe3 config --step
     hybrid` lists the sections.
Cluster only (production's tree and the archive present; else UNTESTED-CLUSTER, not a failure):
  6. every value of config/steps/hybrid.py equals production's constant at exopipe 4c7098c (notaper_largescale.py,
     mosaic_tpfill.py) and the TPCAVEAT units of the per-tile fill driver;
  7. every FITS name in the archived hybrid runs (H, HQ11, HQ13 with unfilled/), the v3 / v5 mosaic runs and the v5 quarantines
     (MQ11, MQ13 with unfilled/; MQ11 holds the retired tpfill names, which parse with the retired marker) parses and round-trips
     through core/products (single-directory listings at known depths; no recursive walk of Lustre);
  8. every pinned double-size PSF's md5 equals its registry entry and production's .md5 sidecar.
Run: python tests/test_hybrid_config.py   (or under pytest)
"""
import ast
import os
import shutil
import subprocess
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

_IMPORT_ERROR = None
try:
    from exopipe.phase3.config import CFG, ConfigRefusal
    from exopipe.phase3.core import products as P
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.mosaic.tests import fill_card_problems
    from exopipe.phase3.steps import hybrid as HS
except Exception as e:                                    # pragma: no cover
    _IMPORT_ERROR = f'{e.__class__.__name__}: {e}'

PROD_REPO = '/lustre/gh/main/exopipe'
PROD_COMMIT = '4c7098c'
FILL_DRIVER = '/lustre/gh/main/exopipe_ops/hybrid_master_lsfill_20261111/run_hybrid_master_lsfill_20261111.py'
A = '/lustre/pipeline/exopipe/phase3'
SUBBANDS = ('18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz',
            '64MHz', '69MHz', '73MHz', '78MHz', '82MHz')
# spec 3.3: the keys each section must hold
SPEC_KEYS = {
    'hybrid': ('RECON_GEN', 'HYB_U_IN', 'HYB_U_OUT', 'CAVEAT_SCOPES', 'STRIATION_SCOPES', 'STRIATION_SELECTION',
               'WIDEBAND_MEMBERS', 'ARCHIVE_MASTERS', 'WORK_ROOT'),
    'hybrid_stack': ('COADD_SB', 'MDEWARP', 'WARP_MEDIAN_PX', 'WARP_P90_PX', 'WARP_MIN_SOURCES', 'MAX_UNDERIVABLE',
                     'RAW_NIGHT_ADMIT', 'MIN_NIGHTS_DROP', 'REBUILD_MAX_ABS_JY', 'FLOW_RECOVERED_RATIO_TOL', 'ADMISSION'),
    'hybrid_psf2x': ('REGISTRY',),
    'hybrid_recon': ('SCALES', 'NITER', 'MAXCYC', 'TAU', 'STEP_FWHM_DEG', 'CG_TOL', 'CG_MAX', 'PRIOR_SCALE_RANGE',
                     'PRIOR_RMS_FRAC', 'COMPACT_TOL', 'COMPACT_TOL_LOW', 'LEVEL_LIMITS', 'CLEAN_REGION', 'CASCADE', 'LFSS_DIR',
                     'CATALOGS'),
    'hybrid_unit': ('OWN_BEAM_MAX', 'IDENTITY', 'LARGE_REL', 'SMALL_LIMIT', 'STRIATION_IDENTITY', 'RAW_NIGHT_CAVEAT',
                    'RULED_TAPER_ONLY', 'JUMP10'),
    'hybrid_full': ('MEMBERS', 'REGRID'),
    'hybrid_tilefill': ('TILE_RT_NSIDE', 'TILE_RT_MAX', 'COMPACT_K', 'COMPACT_MED_MAX', 'COMPACT_MAX', 'COMPACT_SIG',
                        'TPCAVEAT_UNITS', 'UNFILLED_INPUTS', 'LFSS_MHZ', 'LMAX', 'APOD_DEG', 'L_LO', 'L_HI', 'TPCAVEAT_TEXT',
                        'REPAIR_KEYS', 'FILLED_MOSAIC_RUN', 'FILLED_MOSAIC_GLOB', 'UNFILLED_MOSAIC_RUNS'),
}


def _args(**kw):
    d = dict(lsts=None, scopes=None, weightings=None, stages=None, stage_from=None, stage_until=None, clean_region=None)
    d.update(kw)
    return types.SimpleNamespace(**d)


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return str(e)
    raise AssertionError(f'{fn.__name__} did not raise {exc.__name__}')


# ---------------------------------------------------------------------------------------------------------------- 1 config
def test_sections_keys_and_consistency():
    secs = CFG.sections()
    for name, keys in SPEC_KEYS.items():
        assert name in secs, f'no config section {name}'
        sec = secs[name]
        miss = [k for k in keys if not hasattr(sec, k)]
        assert not miss, f'{name}: keys missing {miss}'
        for k in list(getattr(sec, 'RULED', {}) or {}) + list(getattr(sec, 'PROTECTED_KEYS', ()) or ()):
            assert hasattr(sec, k), f'{name}: RULED / PROTECTED names no key {k}'
    H = CFG.hybrid
    assert set(H.CAVEAT_SCOPES) <= set(H.SCOPES) and set(H.STRIATION_SCOPES) <= set(H.SCOPES)
    members = [m for w in ('Red', 'Green', 'Blue') for m in H.WIDEBAND_MEMBERS[w]]
    assert tuple(members) == SUBBANDS == tuple(CFG.hybrid_full.MEMBERS), members
    assert tuple(s for s in H.SCOPES if s.endswith('MHz')) == SUBBANDS
    assert H.WEIGHTINGS == ('R075', 'R0')
    assert [c[0] for c in CFG.hybrid_recon.CASCADE] == ['standard', '3x', 'elev10']
    assert CFG.hybrid_recon.CASCADE[0][2] == CFG.hybrid_recon.CLEAN_REGION == 'horizon'


def test_psf2x_registry():
    # Q1 (R-GREGG-20261118-HYBRID-Q1-Q3-GO-WITH-RECOMMENDATIONS): every subband x weighting pinned to one production file
    from exopipe.phase3.config.steps import hybrid as HC
    R = CFG.hybrid_psf2x.REGISTRY
    assert sorted(R) == sorted((b, v) for b in SUBBANDS for v in ('R075', 'R0')), sorted(R)
    for (band, v), (path, md5, ants) in R.items():
        low = int(band[:-3]) <= 59                                   # 18-59 MHz: the 11-18 July-record re-image (PSF2xJ)
        assert os.path.basename(path) == f'PSF2x{"J" if low else "F"}_{band}_{v}-psf.fits', path
        assert path.startswith('/lustre/gh/main/phase3/lsrecon/Run_PSF2X_JULY_20261118/' if low else '/lustre/gh/main/phase3/lsrecon/'), path
        assert len(md5) == 32 and set(md5) <= set('0123456789abcdef'), md5
        assert ants == HC.PSF2X_JULY_LIST, (band, ants)
    assert len({e[0] for e in R.values()}) == len({e[1] for e in R.values()}) == 30


def test_psf2x_files_match_registry():
    R = CFG.hybrid_psf2x.REGISTRY
    if not all(os.path.exists(e[0]) for e in R.values()):
        print('UNTESTED-CLUSTER: the double-size PSFs are absent')
        return
    for (band, v), (path, md5, _a) in R.items():
        assert HS._md5(path) == md5, (band, v, path)
        side = os.path.join(os.path.dirname(path), os.path.basename(path) + '.md5')
        assert open(side).read().split()[0] == md5, (band, v, side)           # production's own sidecar


def test_protected_keys_refuse_override():
    for key in ('hybrid.STRIATION_SELECTION', 'hybrid.STRIATION_SELECTION_MD5', 'hybrid.ARCHIVE_MASTERS',
                'hybrid_psf2x.REGISTRY', 'hybrid_tilefill.UNFILLED_INPUTS', 'hybrid_recon.CATALOGS'):
        _raises((ConfigRefusal, KeyError, ValueError), CFG.set, key, "'x'", source='test')


# ---------------------------------------------------------------------------------------------------------------- 2 plan
def test_plan_units():
    U = HS.plan_units(_args(lsts='18h', scopes='46MHz,Blue', weightings='R075'), CFG)
    assert U == [('stack', '18h', '46MHz', 'R075'), ('recon', '18h', '46MHz', 'R075'), ('wideband', '18h', 'Blue', 'R075'),
                 ('hybrid', '18h', '46MHz', 'R075'), ('hybrid', '18h', 'Blue', 'R075'), ('tilefill', '18h', '46MHz', 'R075'),
                 ('tilefill', '18h', 'Blue', 'R075')], U
    # every LST, scope and weighting: stack 15 x 24 x 2, wideband 3 x 48, hybrid 18 x 48, full 48, tilefill 19 x 48
    n = {}
    for st, *_ in HS.plan_units(_args(), CFG):
        n[st] = n.get(st, 0) + 1
    assert n == {'stack': 720, 'recon': 720, 'wideband': 144, 'hybrid': 864, 'full': 48, 'tilefill': 912}, n
    # stage order whatever the order given; --from / --until
    assert [u[0] for u in HS.plan_units(_args(lsts='00h', scopes='Full', weightings='R0', stages='tilefill,full'), CFG)] == \
        ['full', 'tilefill']
    assert {u[0] for u in HS.plan_units(_args(lsts='00h', stage_from='recon', stage_until='hybrid'), CFG)} == \
        {'recon', 'wideband', 'hybrid'}
    # refusals
    assert 'unknown stage' in _raises(Refusal, HS.plan_units, _args(stages='stack,mosaic'), CFG)
    assert 'not in the hybrid route' in _raises(Refusal, HS.plan_units, _args(scopes='Gold'), CFG)
    assert 'hybrid.WEIGHTINGS' in _raises(Refusal, HS.plan_units, _args(weightings='R05'), CFG)
    assert 'exclusive' in _raises(Refusal, HS.plan_units, _args(stages='stack', stage_from='stack'), CFG)
    assert 'not a stage' in _raises(Refusal, HS.plan_units, _args(stage_from='stak'), CFG)      # a Refusal, not a ValueError
    assert 'not a stage' in _raises(Refusal, HS.plan_units, _args(stage_until='ful'), CFG)
    assert 'comes after' in _raises(Refusal, HS.plan_units, _args(stage_from='full', stage_until='stack'), CFG)
    assert 'are not of' in _raises(Refusal, HS.plan_units, _args(lsts='25h'), CFG)
    # a scope a stage does not take is skipped for that stage: Full alone with stack -> no unit
    assert HS.plan_units(_args(scopes='Full', stages='stack'), CFG) == []


# ---------------------------------------------------------------------------------------------------------------- 3 gates
def test_stage_gates_and_out():
    d = tempfile.mkdtemp(prefix='hy1_', dir=os.environ.get('TMPDIR') or None)
    try:
        sel = os.path.join(d, 'selection.csv')
        with open(sel, 'w') as f:
            f.write('lst,band,variant\n04h,59MHz,Taper_R075\n')
        import hashlib
        md5 = hashlib.md5(open(sel, 'rb').read()).hexdigest()
        cfg = types.SimpleNamespace(
            hybrid_stack=types.SimpleNamespace(COADD_SB=os.path.join(d, 'absent')),
            hybrid_psf2x=types.SimpleNamespace(REGISTRY={}),
            hybrid=types.SimpleNamespace(STRIATION_SELECTION=sel, STRIATION_SELECTION_MD5=md5,
                                         LSRECON_RUNS={'R1': os.path.join(d, 'lsrecon', 'R1')}, ARCHIVE_HYBRID=os.path.join(d, 'H'),
                                         ARCHIVE_MASTERS=os.path.join(d, 'M'), WORK_ROOT=os.path.join(d, 'lsrecon')),
            hybrid_tilefill=types.SimpleNamespace(UNFILLED_INPUTS=None, FILLED_MOSAIC_RUN=os.path.join(d, 'MOS'),
                                                  UNFILLED_MOSAIC_RUNS=(os.path.join(d, 'MQ'),)))
        assert any(p.startswith('HG0') for p in HS.stage_gate_problems('stack', cfg))
        assert any(p.startswith('HG6') for p in HS.stage_gate_problems('recon', cfg))
        assert any(p.startswith('HG6') for p in HS.stage_gate_problems('wideband', cfg))
        for st in ('hybrid', 'full'):                                 # selection md5 equal: only HG0 (they read COADD_SB too)
            assert [p[:3] for p in HS.stage_gate_problems(st, cfg)] == ['HG0'], st
        tf = HS.stage_gate_problems('tilefill', cfg)                 # HG13: nothing pinned, no mosaic runs
        assert any('UNFILLED_INPUTS' in p for p in tf) and sum('mosaic run' in p for p in tf) == 2, tf
        os.mkdir(cfg.hybrid_tilefill.FILLED_MOSAIC_RUN); os.mkdir(cfg.hybrid_tilefill.UNFILLED_MOSAIC_RUNS[0])
        man = os.path.join(d, 'HQ.manifest.csv')
        with open(man, 'w') as f:
            f.write('path,size,mtime_ns,sha256,md5,link_target\n/x.fits,1,1,aa,bb,\n')
        sha = hashlib.sha256(open(man, 'rb').read()).hexdigest()
        cfg.hybrid_tilefill.UNFILLED_INPUTS = {'HQ': (man, sha)}
        HS._ROWS.clear()
        assert HS.stage_gate_problems('tilefill', cfg) == []
        cfg.hybrid_tilefill.UNFILLED_INPUTS = {'HQ': (man, '0' * 64)}  # the manifest changed after it was pinned
        HS._ROWS.clear()
        assert any('HG13' in p and 'sha256' in p for p in HS.stage_gate_problems('tilefill', cfg))
        os.mkdir(cfg.hybrid_stack.COADD_SB)
        for st in ('stack', 'hybrid', 'full'):
            assert HS.stage_gate_problems(st, cfg) == [], st
        psf = os.path.join(d, 'PSF2xF_46MHz_R075-psf.fits')
        with open(psf, 'wb') as f:
            f.write(b'psf')
        cfg.hybrid_psf2x.REGISTRY = {('46MHz', 'R075'): (psf, hashlib.md5(b'psf').hexdigest(), 'night')}
        assert HS.stage_gate_problems('recon', cfg) == []
        assert HS.stage_gate_problems('wideband', cfg) == []
        with open(psf, 'ab') as f:                                    # the PSF changed after it was pinned
            f.write(b'x')
        assert any(p.startswith('HG6') and 'not the pinned' in p for p in HS.stage_gate_problems('recon', cfg))
        assert HS.stage_gate_problems('wideband', cfg) == []          # the wideband stage reads no double-size PSF
        os.remove(psf)
        assert any(p.startswith('HG6') and 'md5 None' in p for p in HS.stage_gate_problems('recon', cfg))
        with open(sel, 'a') as f:                                     # the selection changed after it was pinned
            f.write('05h,59MHz,Taper_R0\n')
        assert any('not the pinned' in p for p in HS.stage_gate_problems('hybrid', cfg))
        os.remove(sel)                                                # missing
        assert any('md5 None' in p for p in HS.stage_gate_problems('hybrid', cfg))
        # --out: never a production work run or an archive run, inside one or above one; never the work root or above it
        L = os.path.join(d, 'lsrecon')
        assert HS.out_problems(os.path.join(L, 'R1'), cfg) and HS.out_problems(os.path.join(L, 'R1', 'x'), cfg)
        assert HS.out_problems(os.path.join(d, 'H'), cfg)
        assert any('contains R1' in p for p in HS.out_problems(d, cfg))            # above the held runs
        assert any('contains ARCHIVE_HYBRID' in p for p in HS.out_problems(d, cfg))
        assert any('work root' in p for p in HS.out_problems(L, cfg))
        assert any('work root' in p for p in HS.out_problems('/', cfg))
        assert HS.out_problems(os.path.join(L, 'R10'), cfg) == []      # a sibling with the same prefix is not inside
        assert HS.out_problems(os.path.join(L, 'new_run'), cfg) == []
        assert HS.out_problems(os.path.join(d, 'elsewhere', 'new_run'), cfg) == []
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------------- 4 cards
# literal cards of archived files (read 2026-11-18 with plain 2,880-byte block reads): proc tokens from the name, the cards that matter
CARD_CASES = [
    # (what, proc, cards, problems expected)
    ('V5 Blue R075 Jy/sr lsfill', ['pbcorr', 'coadd', 'lsrecon', 'mosaic', 'jysr', 'feather', 'lsfill'], {'TPFILL': True}, False),
    ('V5 Blue R075 Jy/beam lsfill', ['coadd', 'lsrecon', 'mosaic', 'jybeam', 'feather', 'lsfill'], {'TPFILL': True}, False),
    ('V5 Blue R075 psfmap lsfill', ['pbcorr', 'coadd', 'lsrecon', 'mosaic', 'feather', 'lsfill', 'psfmap'], {}, False),
    ('V5 Blue R075 pbmap lsfill', ['coadd', 'lsrecon', 'mosaic', 'feather', 'lsfill', 'pbmap'], {}, False),
    ('V5 Blue R075 pkcorr lsfill', ['pbcorr', 'coadd', 'lsrecon', 'mosaic', 'feather', 'lsfill', 'pkcorr'], {}, False),
    ('V5 Blue R075 caveat lsfill', ['pbcorr', 'coadd', 'lsrecon', 'mosaic', 'feather', 'lsfill', 'caveat'], {}, False),
    ('H 18h 46MHz R075 master lsfill', ['shflux', 'mdewarp', 'dewarped', 'aligned', 'coadd', 'lsrecon', 'lsfill'],
     {'LSFILL': True, 'TPFILL': True}, False),
    ('H 18h 46MHz R075 cleanres', ['shflux', 'mdewarp', 'dewarped', 'aligned', 'coadd', 'lsrecon', 'cleanres'], {'LSFILL': True},
     False),
    ('H 18h 46MHz R075 NoTaper reconstruction', ['shflux', 'mdewarp', 'dewarped', 'aligned', 'coadd', 'lsrecon'], {}, False),
    ('V3 Blue Taper (no fill)', ['pbcorr', 'coadd', 'mosaic', 'jysr', 'feather'], {}, False),
    # mutants
    ('Jy/sr named lsfill without TPFILL', ['pbcorr', 'coadd', 'lsrecon', 'mosaic', 'jysr', 'feather', 'lsfill'], {}, True),
    ('Jy/sr named lsfill, TPFILL F', ['pbcorr', 'coadd', 'lsrecon', 'mosaic', 'jysr', 'feather', 'lsfill'], {'TPFILL': False}, True),
    ('master named lsfill, LSFILL only', ['coadd', 'lsrecon', 'lsfill'], {'LSFILL': True}, True),
    ('TPFILL without the name', ['pbcorr', 'coadd', 'lsrecon', 'mosaic', 'jysr', 'feather'], {'TPFILL': True}, True),
    ('psfmap with TPFILL', ['pbcorr', 'coadd', 'lsrecon', 'mosaic', 'feather', 'lsfill', 'psfmap'], {'TPFILL': True}, True),
    ('cleanres with TPFILL', ['coadd', 'lsrecon', 'cleanres'], {'LSFILL': True, 'TPFILL': True}, True),
    # the retired token (parse-only names before 2026-11-11) is a problem whatever the cards
    ('retired tpfill Jy/sr', ['pbcorr', 'coadd', 'lsrecon', 'mosaic', 'jysr', 'feather', 'lsfill', 'tpfill'], {'TPFILL': True},
     True),
    ('retired tpfill without TPFILL', ['pbcorr', 'coadd', 'mosaic', 'jysr', 'feather', 'tpfill'], {}, True),
]


def test_fill_card_rule():
    bad = []
    for what, proc, cards, expect in CARD_CASES:
        got = fill_card_problems(proc, cards)
        if bool(got) != expect:
            bad.append(f'{what}: {got}')
    assert not bad, bad
    # a NEW name derived from a parsed archived tpfill name (the sibling-lookup pattern parse -> change proc -> build_name(**pn))
    # still carries the token, since 'retired' rides along in the dict; the content test refuses it, naming the ruling
    from exopipe.phase3.core.products import build_name, parse_name
    pn = parse_name('Green_I_deep_Taper_Robust+0.0_pbcorr_coadd_lsrecon_mosaic_jysr_feather_lsfill_tpfill_molrender_allsky_'
                    '20241218-20260419_N024.fits')
    pn['proc'] = [t for t in pn['proc'] if t not in ('pbcorr', 'jysr')] + ['jybeam']
    new = parse_name(build_name(**pn))
    got = fill_card_problems(new['proc'], {'TPFILL': True})
    assert got and 'R-GREGG-20261111-KEEP-LSFILL' in got[0], got


# ---------------------------------------------------------------------------------------------------------------- 5 CLI
def _cli(argv, env_extra=None):
    env = dict(os.environ)
    env.update(env_extra or {})
    if argv and argv[0] == 'hybrid':
        argv = argv + ['--cores', 'none']           # no recorder-free core lookup (runner.FREE_CORES_CMD) off the cluster
    p = subprocess.run([sys.executable, '-m', 'exopipe.phase3.cli'] + argv, cwd=REPO, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=600)
    return p.returncode, p.stdout


def test_cli_skeleton_refuses_every_unit():
    d = tempfile.mkdtemp(prefix='hy1_cli_', dir=os.environ.get('TMPDIR') or None)
    try:
        # the stack stage is ported (PR-2): COADD_SB points nowhere, so its unit refuses at HG0 on every host (never a real stack)
        rc, out = _cli(['hybrid', '--lsts', '18h', '--scopes', '46MHz,Blue', '--weightings', 'R075', '--out', os.path.join(d, 'w'),
                        '--set', f'hybrid_stack.COADD_SB={os.path.join(d, "no_coadd_sb")}'])
        assert rc == 2, (rc, out[-2000:])
        lines = [l for l in out.splitlines() if l.strip().startswith('refused ')]
        assert any('stack.18h_46MHz_R075' in l and 'HG0' in l for l in lines), lines
        # recon (ported, PR-3): HG6 where the pinned PSFs are absent (CI: md5 None), else no input (the stack refused, so no
        # stack record); wideband reads no PSF and refuses: its first member holds no passed reconstruction
        rec = [l for l in lines if 'recon.18h_46MHz_R075' in l]
        hg6 = len(rec) == 1 and 'HG6' in rec[0] and 'md5 None' in rec[0]
        assert hg6 or not rec, (rec, out[-3000:])
        # hybrid (ported, PR-4): HG0 refuses it with the stack (COADD_SB points nowhere); in CI the striation-selection gate as well
        hyb = [l for l in lines if l.startswith('  refused hybrid.')]
        assert len(hyb) == 2 and all('HG0' in l for l in hyb), (hyb, out[-3000:])
        # tilefill (ported, PR-5): HG13 where the pinned manifests are absent (CI), else no input (no hybrid record in --out)
        tf = [l for l in lines if l.startswith('  refused tilefill.')]
        hg13 = len(tf) == 2 and all('HG13' in l and 'UNFILLED_INPUTS' in l for l in tf)
        assert hg13 or not tf, (tf, out[-3000:])
        nref, nno = 4 + hg6 + 2 * hg13, int(not hg6) + 2 * int(not hg13)
        assert len(lines) == nref, out[-3000:]
        assert any('wideband.18h_Blue_R075' in l and '64MHz holds no passed reconstruction' in l for l in lines), lines
        assert f'refused {nref}' in out and 'exit 2' in out, out[-500:]
        assert nno == 0 or f'no-input {nno}' in out, out[-800:]
        # no --out: refused before any unit (a new work run is always named)
        rc, out = _cli(['hybrid', '--lsts', '18h', '--scopes', '46MHz'])
        assert rc == 2 and 'needs --out' in out, (rc, out[-800:])
        # --workers > 1 is refused before any unit: the step runs its units one after another (never silently ignored)
        rc, out = _cli(['hybrid', '--lsts', '18h', '--scopes', '46MHz', '--workers', '2', '--out', os.path.join(d, 'w4')])
        assert rc == 2 and '--workers 2' in out and 'disjoint --lsts' in out, (rc, out[-800:])
        assert not os.path.exists(os.path.join(d, 'w4', '_records', 'hybrid', 'stack.18h_46MHz_R075.json')), out[-800:]
        # a protected key cannot be set
        rc, out = _cli(['hybrid', '--lsts', '18h', '--scopes', '46MHz', '--out', os.path.join(d, 'w2'),
                        '--set', 'hybrid.STRIATION_SELECTION=/tmp/x.csv'])
        assert rc == 2 and 'REFUSED' in out, (rc, out[-800:])
        # --clean-region goes through --set's checks before the run record exists (config_overrides): the record carries it
        rc, out = _cli(['hybrid', '--lsts', '18h', '--scopes', '46MHz', '--weightings', 'R075', '--stages', 'recon',
                        '--clean-region', 'elev10', '--out', os.path.join(d, 'w3')])
        assert rc in (2, 3), (rc, out[-800:])             # HG6 refusal (CI: no pinned PSFs) or no input (no stack in w3)
        import json
        rec = out.rsplit('; record ', 1)[-1].strip()
        ov = [o for o in json.load(open(rec))['config_overrides'] if o[0] == 'hybrid_recon.CLEAN_REGION']
        assert ov and ov[0][2] == repr('elev10') and ov[0][3] == '--clean-region', ov
        assert HS.config_overrides(types.SimpleNamespace(clean_region=None)) == []
        # the config listing builds and loads every section
        rc, out = _cli(['config', '--step', 'hybrid'])
        assert rc == 0, (rc, out[-800:])
        for s in SPEC_KEYS:
            assert f'section {s}:' in out, s
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------------- 6 parity
def _module_constants(src):
    """Module-level NAME = literal and A, B = literal, literal assignments; os.environ.get(K, default) gives the default."""
    out = {}
    for node in ast.parse(src).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        t, v = node.targets[0], node.value
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == 'get' and len(v.args) == 2):
            v = v.args[1]
        try:
            val = ast.literal_eval(v)
        except ValueError:
            continue
        if isinstance(t, ast.Name):
            out[t.id] = val
        elif isinstance(t, ast.Tuple) and isinstance(val, tuple) and len(t.elts) == len(val):
            for e, x in zip(t.elts, val):
                if isinstance(e, ast.Name):
                    out[e.id] = x
    return out


def _git_show(path):
    return subprocess.run(['git', '-C', PROD_REPO, 'show', f'{PROD_COMMIT}:{path}'], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True, timeout=300, check=True).stdout


def _cluster():
    return os.path.isdir(os.path.join(PROD_REPO, '.git')) and os.path.isdir(A)


def test_values_equal_production():
    if not _cluster():
        print('UNTESTED-CLUSTER: production tree or archive absent')
        return
    NL = _module_constants(_git_show('exopipe/phase3/notaper_largescale.py'))
    MF = _module_constants(_git_show('exopipe/phase3/mosaic_tpfill.py'))
    C = CFG
    pairs = [
        (C.hybrid.RECON_GEN, NL['RECON_GEN']), (C.hybrid.HYB_U_IN, NL['HYB_U_IN']), (C.hybrid.HYB_U_OUT, NL['HYB_U_OUT']),
        (C.hybrid.CAVEAT_SCOPES, NL['CAVEAT_SCOPES']), (C.hybrid.STRIATION_SCOPES, NL['STRIATION_SCOPES']),
        (C.hybrid.STRIATION_SELECTION, NL['STRIATION_SELECTION']), (C.hybrid.STRIATION_RESTORED, NL['STRIATION_RESTORED']),
        (C.hybrid.WIDEBAND_MEMBERS, NL['WIDEBANDS']), (C.hybrid.ARCHIVE_MASTERS, NL['ARCHIVE_MASTERS']),
        (C.hybrid.ARCHIVE_HYBRID, NL['ARCHIVE_HYBRID']),
        (C.hybrid_stack.COADD_SB, NL['COADD_SB']), (C.hybrid_stack.MDEWARP, NL['MDEWARP']),
        (C.hybrid_recon.SCALES, NL['SCALES']), (C.hybrid_recon.NITER, NL['NITER']), (C.hybrid_recon.GAIN, NL['GAIN']),
        (C.hybrid_recon.MGAIN, NL['MGAIN']), (C.hybrid_recon.BIAS, NL['BIAS']), (C.hybrid_recon.AUTOMASK, NL['AUTOMASK']),
        (C.hybrid_recon.AUTOTHRESH, NL['AUTOTHRESH']), (C.hybrid_recon.PATCH, NL['PATCH']), (C.hybrid_recon.MAXACT, NL['MAXACT']),
        (C.hybrid_recon.MAXLOOP, NL['MAXLOOP']), (C.hybrid_recon.MAXCYC, NL['MAXCYC']), (C.hybrid_recon.TAU, NL['TAU']),
        (C.hybrid_recon.STEP_FWHM_DEG, NL['STEP_FWHM_DEG']), (C.hybrid_recon.CG_TOL, NL['CG_TOL']),
        (C.hybrid_recon.CG_MAX, NL['CG_MAX']), (C.hybrid_recon.LEVEL_LIMITS, NL['LEVEL_LIMITS']),
        (C.hybrid_recon.DEC_BANDS, NL['DEC_BANDS']), (C.hybrid_recon.LFSS_DIR, NL['LFSS_DIR']),
        (C.hybrid_unit.SMALL_LIMIT, NL['HYB_SMALL_LIMIT']), (C.hybrid_unit.RAW_NIGHT_CAVEAT, NL['RAW_NIGHT_CAVEAT']),
        (C.hybrid_unit.RULED_TAPER_ONLY, NL['RULED_TAPER_ONLY']),
        (C.hybrid_tilefill.TILE_RT_NSIDE, MF['TILE_RT_NSIDE']), (C.hybrid_tilefill.TILE_RT_MAX, MF['TILE_RT_MAX']),
        (C.hybrid_tilefill.COMPACT_K, MF['COMPACT_K']), (C.hybrid_tilefill.COMPACT_MED_MAX, MF['COMPACT_MED_MAX']),
        (C.hybrid_tilefill.COMPACT_MAX, MF['COMPACT_MAX']), (C.hybrid_tilefill.COMPACT_SIG, MF['COMPACT_SIG']),
        (C.hybrid_tilefill.LFSS_MHZ, MF['LFSS_MHZ']), (C.hybrid_tilefill.LMAX, MF['LMAX']), (C.hybrid_tilefill.APOD_DEG, MF['APOD']),
        (C.hybrid_tilefill.L_LO, MF['L_LO']), (C.hybrid_tilefill.L_HI, MF['L_HI']),
        (os.path.join(C.hybrid_recon.LFSS_DIR, 'healpix'), MF['LFSS_DIR']),
    ]
    bad = [(i, a, b) for i, (a, b) in enumerate(pairs) if a != b or type(a) is not type(b) and not (
        isinstance(a, (int, float)) and isinstance(b, (int, float)))]
    assert not bad, bad
    # the one value changed on purpose (spec section 7, 2026-11-18): the parent-catalog union rebuilt on the current Masters
    assert NL['CATALOGS'] == '/lustre/gh/main/phase3/deep_parent_catalogs_union_20261014T0203', NL['CATALOGS']
    assert C.hybrid_recon.CATALOGS == '/lustre/gh/main/phase3/deep_parent_catalogs_union_20261118T0130', C.hybrid_recon.CATALOGS
    assert tuple(NL['WIDEBANDS'].get('Full', ())) in ((), SUBBANDS)       # Full is assigned after the literal (NL 947)
    assert C.hybrid_full.MEMBERS == SUBBANDS
    # the TPCAVEAT units of the per-tile fill driver (literal set CAVEAT)
    drv = _module_constants(open(FILL_DRIVER).read())
    assert set(C.hybrid_tilefill.TPCAVEAT_UNITS) == set(drv['CAVEAT']), (C.hybrid_tilefill.TPCAVEAT_UNITS, drv['CAVEAT'])
    assert C.hybrid_tilefill.TPCAVEAT_TEXT == drv['CAVEAT_TXT'], drv.get('CAVEAT_TXT')
    assert (C.hybrid_tilefill.FILLED_MOSAIC_RUN, C.hybrid_tilefill.UNFILLED_MOSAIC_RUNS[-1]) == (drv['MOS'], drv['MQ'])
    # the recon cascade: the tries of fleet-recon (NL 2209: budget x1, x3, then x3 at elev10)
    src = _git_show('exopipe/phase3/notaper_largescale.py')
    assert "(3 * NITER, 3 * MAXCYC, 'elev10')" in src and '(3 * NITER, 3 * MAXCYC, a.clean_region)' in src
    # the pinned striation selection
    assert HS._md5(C.hybrid.STRIATION_SELECTION) == C.hybrid.STRIATION_SELECTION_MD5
    print(f'  {len(pairs)} values equal to production {PROD_COMMIT}; TPCAVEAT units, cascade and selection md5 equal')


# ---------------------------------------------------------------------------------------------------------------- 7 names
def _archive_listings():
    """{run label: [FITS names]} by single-directory listings at known depths."""
    from exopipe.phase3.core import cells
    H = f'{A}/Coadd/Run_NOTAPER_HYBRID_20261028'
    Q11 = f'{A}/Coadd/Run_NOTAPER_HYBRID_QUARANTINE_20261111'
    Q13 = f'{A}/Coadd/Run_NOTAPER_HYBRID_QUARANTINE_20261113'
    V5 = f'{A}/mosaic/Run_HYBRID_20241218-20260419'
    V3 = f'{A}/mosaic/Run_FEATHER3_20241218-20251029'
    scopes = SUBBANDS + ('Red', 'Green', 'Blue', 'Full')
    out = {}

    def ls(d):
        try:
            return sorted(n for n in os.listdir(d) if n.endswith('.fits'))
        except FileNotFoundError:
            return []
    for lab, root in (('H', H), ('HQ11', Q11), ('HQ13', Q13), ('HQ13/unfilled', f'{Q13}/unfilled')):
        out[lab] = [n for l in cells.LSTS for s in scopes for n in ls(f'{root}/{l}/{s}')]
    MQ11 = f'{A}/mosaic/Run_HYBRID_QUARANTINE_20261111'
    MQ13 = f'{A}/mosaic/Run_HYBRID_QUARANTINE_20261113'
    for lab, root, stokes in (('V5', V5, ('I',)), ('V3', V3, ('I', 'V')), ('MQ11', MQ11, ('I',)), ('MQ13', MQ13, ('I',)),
                              ('MQ13/unfilled', f'{MQ13}/unfilled', ('I',))):
        out[lab] = [n for s in scopes for st in stokes for kind in ('healpix', 'mollweide', 'radec')
                    for n in ls(f'{root}/{s}/{st}/{kind}')]
    return out


NOT_GRAMMAR = ('PB_deep_', 'PBdiffuse_deep_', 'PB_eff_')     # PB maps keep their pre-grammar names (renamed next generation)


def test_archive_names_round_trip():
    if not _cluster():
        print('UNTESTED-CLUSTER: production tree or archive absent')
        return
    L = _archive_listings()
    bad, n, skipped, retired = [], 0, 0, 0
    for lab, names in L.items():
        assert names, f'{lab}: no FITS name listed (layout changed?)'
        for nm in names:
            if nm.startswith(NOT_GRAMMAR):
                skipped += 1
                continue
            n += 1
            try:
                g = P.parse_name(nm)
                if g['grammar'] != 'phase3' or P.build_name(**g) != nm:
                    bad.append(f'{lab}: {nm}')
                retired += bool(g.get('retired'))
            except ValueError as e:
                bad.append(f'{lab}: {nm}: {e}')
    assert not bad, f'{len(bad)}/{n} do not round-trip: {bad[:5]}'
    assert retired > 0, 'no archived tpfill name listed (MQ11 holds the pre-2026-11-11 filled mosaics)'
    print(f'  {n} archived names round-trip ({", ".join(f"{k} {len(v)}" for k, v in L.items())}; {skipped} PB maps not in '
          f'the grammar; {retired} with the retired tpfill)')


TESTS = [test_sections_keys_and_consistency, test_psf2x_registry, test_psf2x_files_match_registry, test_protected_keys_refuse_override,
         test_plan_units, test_stage_gates_and_out,
         test_fill_card_rule, test_cli_skeleton_refuses_every_unit, test_values_equal_production, test_archive_names_round_trip]

try:                                                      # pytest collects TestHybridConfig, not the plain functions
    import unittest

    class TestHybridConfig(unittest.TestCase):
        pass
    for _t in TESTS:
        setattr(TestHybridConfig, _t.__name__, (lambda f: lambda self: (self.skipTest(f'UNTESTED-IMPORT: {_IMPORT_ERROR}')
                                                                        if _IMPORT_ERROR else f()))(_t))
    for _t in TESTS:
        globals().pop(_t.__name__, None)
except Exception:                                         # pragma: no cover
    pass


def main():
    if _IMPORT_ERROR is not None:
        print(f'UNTESTED-IMPORT: cannot import dependencies in this interpreter ({_IMPORT_ERROR}) - run under the pipeline env')
        return 1
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f'PASS  {t.__name__}')
        except Exception as e:
            failed += 1
            print(f'FAIL  {t.__name__}: {e.__class__.__name__}: {str(e)[:600]}')
    print(f'{len(TESTS) - failed}/{len(TESTS)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
