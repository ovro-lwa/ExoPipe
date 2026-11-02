#!/usr/bin/env python3
"""STANDING TEST - core/inputs Phase-1 frames: the per-cell RUN PIN, the frame-time constant, the frame identity, and each
night's Phase-1 CODE VERSION in the input record.

  pin        production clv15 facb3e8 (Gregg 2026-11-01, 18h 2025-05-05 "Pin to June run (Recommended)"): a pinned cell
             resolves to its pinned run only, refuses when that run is absent; a cell without a pin is unchanged; the pin file
             is paths.PHASE1_RUN_PINS (protected), re-read when it changes; a file without cell / run columns refuses
  constant   integration_label = production's phase3_snapshot_search.integration_label (its own source, parsed read-only):
             search_snapshot.INTEGRATION_S 10.03102041 s, not instrument.INTEGRATION_S 10.0310 s (a DATE-OBS whose midpoint
             falls between the two gets another label: FOLD_IN 10-28)
  identity   on synthetic cells, phase1_frames returns production discover_frames' (key, path) lists exactly - pinned,
             unpinned (latest run wins per integration), a file-name collision (the later run's frame kept, the lost
             integration listed), a frame without DATE-OBS in a mixed cell refused (production run in a subprocess)
  code       provenance.json pipeline_version per run and subband; a Phase-2 image's Phase-1 runs from its Science log; a unit
             record carries 'phase1_code' per night and per input; a staged copy carries none
  real       (read-only; --no-real skips) 18h 2025-05-05 41 MHz: Stokes I 356 frames pinned (Run_20260603_010316, Orca
             70d5f1c), 358 unpinned (Run_20261020_205211, 908dc6e) - each (key, path) list equal to production's

Run:  python tests/test_core_inputs.py [--no-real]
"""
import ast
import datetime
import json
import os
import random
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

import numpy as np                                                  # noqa: E402
from astropy.io import fits                                         # noqa: E402

from exopipe.phase3.config import CFG, ConfigRefusal                # noqa: E402
from exopipe.phase3.core import inputs as I, record                 # noqa: E402
from exopipe.phase3.core.record import Refusal                      # noqa: E402

T = '/tmp/exopipe3_tests_core_inputs'
CLV = '/lustre/gh/main/clv15'
SS = os.path.join(CLV, 'phase3_snapshot_search.py')
FAILS, N = [], [0]
DT = 10.03102041
EPOCH = datetime.datetime(1970, 1, 1)


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({str(detail)[:600]})'), flush=True)
    if not ok:
        FAILS.append(name)


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    return None


# ---- production's label functions, from its own source text (read only) ------------------------------------------------

def production_label_namespace():
    """dateobs_unix, integration_mid_unix, integration_label and INTEGRATION_S / MAX_FRAMES_PER_HOUR of production's
    phase3_snapshot_search.py, executed from their source (no import of the engine)."""
    src = open(SS).read()
    tree = ast.parse(src)
    want_f = {'dateobs_unix', 'integration_mid_unix', 'integration_label'}
    want_v = {'INTEGRATION_S', '_EPOCH', 'MAX_FRAMES_PER_HOUR'}
    ns = {'datetime': datetime, 'np': np}
    for node in tree.body:                                   # in file order: constants before their users
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in want_v for t in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), SS, 'exec'), ns)
        elif isinstance(node, ast.FunctionDef) and node.name in want_f:
            exec(compile(ast.Module(body=[node], type_ignores=[]), SS, 'exec'), ns)
    return ns


# ---- synthetic Phase-1 trees -----------------------------------------------------------------------------------------------

def dob_str(t):
    """DATE-OBS text of unix time t with one decimal, as WSClean writes it ('2025-05-05T10:58:47.3')."""
    whole = int(t)
    return (EPOCH + datetime.timedelta(seconds=whole)).strftime('%Y-%m-%dT%H:%M:%S') + f'.{int(round((t - whole) * 10)) % 10}'


def label_of(t):
    return (EPOCH + datetime.timedelta(seconds=int(np.floor(t + DT / 2.0)))).strftime('%Y%m%d_%H%M%S')


def write_frame(path, date_obs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h = fits.Header()
    if date_obs is not None:
        h['DATE-OBS'] = date_obs
    fits.PrimaryHDU(np.zeros((4, 4), np.float32), h).writeto(path, overwrite=True)


def build_run(p1, lst, date, run, sb, integrations, t0, version=None, rename=None, no_dateobs=()):
    """One Phase-1 run of ``sb``: new-era I (snapshots_clean) and V (snapshots) frames of the given integration indices, each
    named by its integration label (Phase 1's naming) unless ``rename`` maps an index to another index's label; DATE-OBS = the
    integration start. ``version``: provenance.json pipeline_version (None: no file)."""
    rd = os.path.join(p1, lst, date, run, sb)
    for k in integrations:
        t = t0 + DT * k
        ts = label_of(t0 + DT * rename[k]) if rename and k in rename else label_of(t)
        d = None if k in no_dateobs else dob_str(t)
        write_frame(os.path.join(rd, 'snapshots_clean', f'{sb}-Clean-Snapshot-{ts}-image.fits'), d)
        write_frame(os.path.join(rd, 'snapshots', f'{sb}-Pilot-Snapshot-{ts}-image.fits'), d)
    if version is not None:
        with open(os.path.join(rd, 'provenance.json'), 'w') as f:
            json.dump({'pipeline_version': version, 'pipeline_branch': 'feature/celery-subband-integration',
                       'package_version': '0.3.0', 'started_at': f'{run[4:8]}-{run[8:10]}-{run[10:12]}T00:00:00Z',
                       'subband': sb, 'run_label': run}, f)
    return rd


def write_pins(path, rows):
    with open(path, 'w') as f:
        f.write('cell,run,ruling,reason\n' + ''.join(f'{c},{r},"Gregg test",test\n' for c, r in rows))
    return path


T0 = 1893456000.3            # 2030-01-01T00:00:00.3 UTC


def synthetic_tree():
    p1 = os.path.join(T, 'phase1')
    shutil.rmtree(p1, ignore_errors=True)
    # A: two runs, the newer covers 6 integrations, the older 4 of them (the 18h 2025-05-05 shape)
    build_run(p1, '18h', '2030-01-01', 'Run_20300101_000000', '41MHz', range(4), T0, version='aaaa111')
    build_run(p1, '18h', '2030-01-01', 'Run_20300201_000000', '41MHz', range(6), T0, version='bbbb222')
    # B: a file-name collision - the newer run's integration 10 named with integration 3's second
    build_run(p1, '18h', '2030-01-02', 'Run_20300101_000000', '41MHz', range(4), T0 + 86400, version='aaaa111')
    build_run(p1, '18h', '2030-01-02', 'Run_20300201_000000', '41MHz', [0, 1, 2, 10], T0 + 86400, version='bbbb222',
              rename={10: 3})
    # C: one run only, no provenance.json
    build_run(p1, '18h', '2030-01-03', 'Run_20300101_000000', '41MHz', range(3), T0 + 2 * 86400)
    # D: two runs, a frame of the older run without DATE-OBS
    build_run(p1, '18h', '2030-01-04', 'Run_20300101_000000', '41MHz', range(3), T0 + 3 * 86400, version='aaaa111',
              no_dateobs={1})
    build_run(p1, '18h', '2030-01-04', 'Run_20300201_000000', '41MHz', range(2), T0 + 3 * 86400, version='bbbb222')
    # an index-named CLEAN file (refactor D12): counted, never a frame
    write_frame(os.path.join(p1, '18h', '2030-01-01', 'Run_20300201_000000', '41MHz', 'snapshots_clean',
                             '41MHz-Clean-Snapshot-t0003-image.fits'), dob_str(T0))
    return p1


def production_discovery(p1, cells, pins):
    """production discover_frames (clv15, in a subprocess: P3_SNAP_RUN_PINS = ``pins``) -> {cell: {I, V} or 'REFUSED: ...'}."""
    code = f'''
import json, sys
sys.path.insert(0, {CLV!r})
import phase3_snapshot_search as SS
out = {{}}
for lst, date, sb in {cells!r}:
    try:
        fr = SS.discover_frames(lst, date, sb, phase1_root={p1!r}, flow_select=False)
        out[f'{{lst}}_{{date}}'] = {{st: [list(x) for x in fr[st]] for st in ('I', 'V')}}
    except Exception as e:
        out[f'{{lst}}_{{date}}'] = 'REFUSED: %s: %s' % (type(e).__name__, e)
print('JSON' + json.dumps(out))
'''
    env = dict(os.environ, P3_SNAP_RUN_PINS=pins)
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, timeout=600)
    line = [x for x in r.stdout.splitlines() if x.startswith('JSON')]
    if not line:
        raise RuntimeError(f'production discovery failed: {r.stderr[-2000:]}')
    return json.loads(line[-1][4:])


def refactor_discovery(p1, cells, pins):
    out = {}
    for lst, date, sb in cells:
        try:
            fr = I.phase1_frames(lst, date, sb, phase1_root=p1, run_pins=pins)
            out[f'{lst}_{date}'] = {st: [list(x) for x in fr[st]] for st in ('I', 'V')}
        except Refusal as e:
            out[f'{lst}_{date}'] = f'REFUSED: {e}'
    return out


# =====================================================================================================================

def test_frame_time_constant():
    ns = production_label_namespace()
    check('the frame-time constant is search_snapshot.INTEGRATION_S = production INTEGRATION_S (10.03102041 s)',
          I.frame_integration_s() == CFG.search_snapshot.INTEGRATION_S == ns['INTEGRATION_S'] == DT,
          (I.frame_integration_s(), ns['INTEGRATION_S']))
    check('... and not instrument.INTEGRATION_S (Phase 2 LWA_DT 10.0310 s)',
          CFG.instrument.INTEGRATION_S != I.frame_integration_s(), CFG.instrument.INTEGRATION_S)
    check('max_frames_per_hour = production MAX_FRAMES_PER_HOUR (360)',
          I.max_frames_per_hour() == ns['MAX_FRAMES_PER_HOUR'] == 360, (I.max_frames_per_hour(), ns['MAX_FRAMES_PER_HOUR']))
    # the boundary case of the FOLD_IN 10-28 finding: a midpoint between the two constants' half-integrations
    d = '2025-05-05T10:58:47.9844975'      # + 5.0155 s = :52.9999975, + 5.015510205 s = :53.000007705 (margins >> 1 ulp)
    old = (EPOCH + datetime.timedelta(seconds=int(np.floor(ns['dateobs_unix'](d) + CFG.instrument.INTEGRATION_S / 2.0)))
           ).strftime('%Y%m%d_%H%M%S')
    check(f'boundary DATE-OBS {d}: label {I.integration_label(d)} = production {ns["integration_label"](d)} '
          f'(the 10.0310 s constant gave {old})',
          I.integration_label(d) == ns['integration_label'](d) == '20250505_105853' and old == '20250505_105852',
          (I.integration_label(d), ns['integration_label'](d), old))
    rng = random.Random(7)
    base = 1735689600.0            # 2025-01-01
    bad = []
    for i in range(20000):
        t = base + rng.uniform(0, 3e7)
        digits = rng.choice((0, 1, 3, 6, 7))
        frac = f'{rng.random():.{digits}f}'[1:] if digits else ''          # '.3', '.984', '.9844975', ...
        s = (EPOCH + datetime.timedelta(seconds=int(t))).strftime('%Y-%m-%dT%H:%M:%S') + frac
        if I.integration_label(s) != ns['integration_label'](s):
            bad.append(s)
    check('integration_label = production on 20,000 random DATE-OBS (0-7 decimals)', not bad, bad[:3])
    unread = [x for x in ('', '2025-05-05 10:58:47.3', '2025-05-05T10:58', 'garbage')
              if raises((ValueError, TypeError), I.integration_label, x) is None]
    check('an unreadable DATE-OBS (empty, space separator, short, garbage) raises ValueError, as production', not unread, unread)


def test_run_pin_and_identity():
    p1 = synthetic_tree()
    nopins = os.path.join(T, 'no_pins.csv')                       # absent: no pins
    pins = write_pins(os.path.join(T, 'pins.csv'), [('18h_2030-01-01', 'Run_20300101_000000')])
    cells = [('18h', '2030-01-01', '41MHz'), ('18h', '2030-01-02', '41MHz'), ('18h', '2030-01-03', '41MHz'),
             ('18h', '2030-01-04', '41MHz')]
    # unpinned and pinned: identical to production's discover_frames on the same tree and pin file
    for label, pf in (('unpinned', nopins), ('pinned', pins)):
        prod, ref = production_discovery(p1, cells, pf), refactor_discovery(p1, cells, pf)
        for (lst, date, _sb) in cells:
            c = f'{lst}_{date}'
            a, b = prod.get(c), ref.get(c)
            if isinstance(a, str) or isinstance(b, str):
                ok = isinstance(a, str) and isinstance(b, str)
                check(f'{label} {c}: both refuse ({str(b)[:80]})', ok, (a, b))
            else:
                check(f'{label} {c}: (key, path) lists = production discover_frames (I {len(b["I"])}, V {len(b["V"])})',
                      a == b, (a, b))
    rep = {}
    fr = I.phase1_frames('18h', '2030-01-01', '41MHz', phase1_root=p1, run_pins=nopins, report=rep)
    check('unpinned: 6 frames per Stokes, all from the latest run (mixed, 4 older copies dropped, index file counted)',
          len(fr['I']) == len(fr['V']) == 6 and {p.split('/')[-4] for _, p in fr['I']} == {'Run_20300201_000000'}
          and rep['mixed'] and rep['dropped'] == {'I': 4, 'V': 4} and rep['pin'] is None and rep['index_named'] == 1,
          (len(fr['I']), rep))
    rep = {}
    fr = I.phase1_frames('18h', '2030-01-01', '41MHz', phase1_root=p1, run_pins=pins, report=rep)
    check('pinned: 4 frames per Stokes, all from the pinned run, not mixed; the report names the pin and its file sha1',
          len(fr['I']) == len(fr['V']) == 4 and {p.split('/')[-4] for _, p in fr['I'] + fr['V']} == {'Run_20300101_000000'}
          and not rep['mixed'] and rep['pin'] and rep['pin']['run'] == 'Run_20300101_000000'
          and rep['pin']['file'] == pins and len(rep['pin']['sha1']) == 40 and list(rep['runs']) == ['Run_20300101_000000'],
          rep)
    check('pinned: phase1_code names the pinned run only (aaaa111)',
          {r: c['pipeline_version'] for r, c in rep['phase1_code'].items()} == {'Run_20300101_000000': 'aaaa111'},
          rep['phase1_code'])
    other_np = I.phase1_frames('18h', '2030-01-03', '41MHz', phase1_root=p1, run_pins=nopins)
    other_p = I.phase1_frames('18h', '2030-01-03', '41MHz', phase1_root=p1, run_pins=pins)
    check('a cell without a pin is unchanged by the pin file', other_np == other_p and len(other_p['I']) == 3, other_p)
    absent = write_pins(os.path.join(T, 'pins_absent.csv'), [('18h_2030-01-01', 'Run_20991231_000000')])
    e = raises(Refusal, I.phase1_frames, '18h', '2030-01-01', '41MHz', phase1_root=p1, run_pins=absent)
    check('a pinned run absent from the archive refuses, naming the pin', e is not None and 'pinned to Phase-1 run '
          'Run_20991231_000000' in str(e), e)
    # re-read on change (production reads once per process): a pin added later applies to the next call
    live = write_pins(os.path.join(T, 'pins_live.csv'), [])
    n0 = len(I.phase1_frames('18h', '2030-01-01', '41MHz', phase1_root=p1, run_pins=live)['I'])
    with open(live, 'a') as f:
        f.write('18h_2030-01-01,Run_20300101_000000,"Gregg test","added later, a longer row"\n')
    n1 = len(I.phase1_frames('18h', '2030-01-01', '41MHz', phase1_root=p1, run_pins=live)['I'])
    check('the pin file is re-read when it changes (6 frames, then 4 after the row is added)', (n0, n1) == (6, 4), (n0, n1))
    later = write_pins(os.path.join(T, 'pins_later_row.csv'), [('18h_2030-01-01', 'Run_20991231_000000'),
                                                               ('18h_2030-01-01', 'Run_20300101_000000')])
    check('a later row of a cell wins (as production: dict assignment in file order)',
          I.phase1_run_pin('18h', '2030-01-01', later) == 'Run_20300101_000000')
    empty = write_pins(os.path.join(T, 'pins_empty_run.csv'), [('18h_2030-01-01', '')])
    check('an empty run is no pin', I.phase1_run_pin('18h', '2030-01-01', empty) is None)
    bad = os.path.join(T, 'pins_bad.csv')
    with open(bad, 'w') as f:
        f.write('cell,pinned_run\n18h_2030-01-01,Run_20300101_000000\n')
    e = raises(Refusal, I.phase1_frames, '18h', '2030-01-01', '41MHz', phase1_root=p1, run_pins=bad)
    check('a pin file without cell / run columns refuses', e is not None and 'no cell / run column' in str(e), e)
    # the identity rule: a file-name collision keeps the later run's frame and lists the lost integration
    rep = {}
    fr = I.phase1_frames('18h', '2030-01-02', '41MHz', phase1_root=p1, run_pins=nopins, report=rep)
    lost = rep['lost']['I']
    check('file-name collision: 4 frames, the later run\'s frame kept under the shared second, the older run\'s integration '
          'listed as lost (production 4fe1121)', len(fr['I']) == 4 and rep['ts_collisions'] == {'I': 1, 'V': 1}
          and len(lost) == 1 and lost[0]['run'] == 'Run_20300101_000000' and lost[0]['file_ts'] == label_of(T0 + 86400 + 3 * DT)
          and rep['disagree']['I'] and rep['integrations']['I'] == 5, rep)
    e = raises(Refusal, I.phase1_frames, '18h', '2030-01-04', '41MHz', phase1_root=p1, run_pins=nopins)
    check('a frame without DATE-OBS in a cell of several runs refuses (production: Phase1DiscoveryRefused)',
          e is not None and 'no readable DATE-OBS' in str(e), e)
    # the default pin file: the production file, protected (no --set / --config changes it)
    check('paths.PHASE1_RUN_PINS = exopipe_ops/snapshot_phase1_run_pins.csv (production RUN_PINS default) and protected',
          CFG.paths.PHASE1_RUN_PINS == '/lustre/gh/main/exopipe_ops/snapshot_phase1_run_pins.csv'
          and CFG.protected('paths.PHASE1_RUN_PINS'), CFG.paths.PHASE1_RUN_PINS)
    e = raises(ConfigRefusal, CFG.set, 'paths.PHASE1_RUN_PINS', '/tmp/other_pins.csv')
    check('--set paths.PHASE1_RUN_PINS refuses (PROTECTED)', e is not None and 'PROTECTED' in str(e), e)
    CFG.reset()
    prod_src = open(SS).read()
    check('production still reads the same file by default (RUN_PINS literal in clv15 phase3_snapshot_search.py)',
          f'"P3_SNAP_RUN_PINS", "{CFG.paths.PHASE1_RUN_PINS}"' in prod_src)


def test_phase1_code_recorded():
    p1 = synthetic_tree()
    p2 = os.path.join(T, 'phase2')
    shutil.rmtree(p2, ignore_errors=True)
    sci = os.path.join(p2, '18h', '2030-01-01', 'Science_20300301_000000')
    os.makedirs(os.path.join(sci, 'Green', 'I', 'deep'))
    os.makedirs(os.path.join(sci, '41MHz', 'I', 'deep'))
    build_run(p1, '18h', '2030-01-01', 'Run_20300201_000000', '46MHz', range(2), T0, version='bbbb222')
    with open(os.path.join(sci, 'phase2_18h_2030-01-01_Science_20300301_000000.log'), 'w') as f:
        f.write('2030-03-01 00:00:00,000 - [PHASE2] - INFO - Found 2 run(s) in ...:\n'
                '2030-03-01 00:00:00,001 - [PHASE2] - INFO -   41MHz: found in Run_20300201_000000 (supersedes Run_20300101_000000)\n'
                'Discovered 2 subbands:\n'
                '2030-03-01 00:00:00,002 - [PHASE2] - INFO -    41MHz  Run_20300101_000000 (84 FITS files)\n'
                '2030-03-01 00:00:00,003 - [PHASE2] - INFO -    46MHz  Run_20300201_000000 (84 FITS files)\n')
    img_g = os.path.join(sci, 'Green', 'I', 'deep', 'Green_I_deep_Taper_Robust-0.75_20300101_LST18h.fits')
    img_41 = os.path.join(sci, '41MHz', 'I', 'deep', '41MHz-I-Taper-3591s-Robust-0.75-20300101_003000-image.fits')
    for p in (img_g, img_41):
        write_frame(p, '2030-01-01T00:00:00.3')
    frames = I.phase1_frames('18h', '2030-01-01', '41MHz', phase1_root=p1, run_pins=os.path.join(T, 'no_pins.csv'))
    older = os.path.join(p1, '18h', '2030-01-01', 'Run_20300101_000000', '41MHz', 'snapshots_clean')
    older_frame = os.path.join(older, sorted(os.listdir(older))[0])
    nopro = os.path.join(p1, '18h', '2030-01-03', 'Run_20300101_000000', '41MHz', 'snapshots_clean')
    nopro_frame = os.path.join(nopro, sorted(os.listdir(nopro))[0])
    staged = os.path.join(T, 'staged', os.path.basename(older_frame))
    os.makedirs(os.path.dirname(staged), exist_ok=True)
    shutil.copyfile(older_frame, staged)
    out = os.path.join(T, 'recout')
    shutil.rmtree(out, ignore_errors=True)
    saved = (CFG.paths.PHASE1_ROOT, CFG.paths.PHASE2_ROOT)
    CFG.paths.PHASE1_ROOT, CFG.paths.PHASE2_ROOT = p1, p2
    try:
        check('phase1_code_of: a Phase-1 frame -> its run and subband', [(n, sb, c['run'], c['pipeline_version'])
              for n, sb, c in I.phase1_code_of(older_frame)] == [('18h_2030-01-01', '41MHz', 'Run_20300101_000000', 'aaaa111')])
        g = I.phase1_code_of(img_g)
        check('phase1_code_of: a Phase-2 Green image -> its five subbands, each with the run the Science log names (later '
              'summary lines win over the "found in" line); unnamed subbands recorded missing',
              [(sb, c['run'], c['pipeline_version']) for _n, sb, c in g] ==
              [('41MHz', 'Run_20300101_000000', 'aaaa111'), ('46MHz', 'Run_20300201_000000', 'bbbb222'),
               ('50MHz', None, None), ('55MHz', None, None), ('59MHz', None, None)]
              and all('names no Phase-1 run' in c['missing'] for _n, sb, c in g[2:]), g)
        nc = I.phase1_code_of(nopro_frame)
        check('a Phase-1 run without provenance.json: pipeline_version None and the reason (never guessed)',
              nc and nc[0][2]['pipeline_version'] is None and 'no provenance.json' in nc[0][2]['missing'], nc)
        check('a staged copy (outside PHASE1_ROOT / PHASE2_ROOT) has no Phase-1 code', I.phase1_code_of(staged) == [])
        run = record.Run('test', ['test_core_inputs'], out, CFG)
        run.commit = 'abc1234'                     # a clean-tree commit, so resume_check judges the inputs (test_core_v2)
        with run.unit('u1') as u:
            for _ts, p in frames['I']:
                u.add_input(p)
            for p in (older_frame, img_g, img_41, nopro_frame, staged):
                u.add_input(p)
            dst = os.path.join(out, 'x.csv')
            os.makedirs(out, exist_ok=True)
            with open(dst, 'w') as f:
                f.write('a\n1\n')
            u.add_output(dst, 'csv_table')
        rec = record.load_record(out, 'test', 'u1')
        pc = rec.get('phase1_code') or {}
        n1 = pc.get('18h_2030-01-01', {})
        check('the unit record carries each night\'s Phase-1 code: 41MHz from both runs (frames of the latest, the older '
              'frame, the 41 MHz Phase-2 image), 46MHz, and the missing ones',
              {r: v.get('pipeline_version') for r, v in n1.get('41MHz', {}).items()} ==
              {'Run_20300101_000000': 'aaaa111', 'Run_20300201_000000': 'bbbb222'}
              and {r: v.get('pipeline_version') for r, v in n1.get('46MHz', {}).items()} == {'Run_20300201_000000': 'bbbb222'}
              and set(n1.get('50MHz', {})) == {'unknown'}
              and pc.get('18h_2030-01-03', {}).get('41MHz', {}).get('Run_20300101_000000', {}).get('missing'), pc)
        ins = rec['inputs']
        check('each input entry names its Phase-1 run and version; the staged copy none',
              ins[os.path.abspath(older_frame)].get('phase1_code') == {'41MHz': {'run': 'Run_20300101_000000',
                                                                                  'pipeline_version': 'aaaa111'}}
              and set(ins[os.path.abspath(img_g)].get('phase1_code', {})) == {'41MHz', '46MHz', '50MHz', '55MHz', '59MHz'}
              and 'phase1_code' not in ins[os.path.abspath(staged)], {k: v.get('phase1_code') for k, v in ins.items()})
        ok, why = record.resume_check(out, 'test', 'u1')
        check('the unit is done and resumes: the recorded phase1_code is no part of an input\'s identity',
              rec['status'] == 'done' and ok, (rec['status'], why))
        run2 = record.Run('test', ['test_core_inputs'], out, CFG)
        run2.commit = 'abc1234'
        with run2.unit('u2') as u:
            u.add_input(staged)
            u.add_output(dst, 'csv_table')
        check('a unit with no Phase-1 / Phase-2 input writes no phase1_code', 'phase1_code' not in
              record.load_record(out, 'test', 'u2'))
    finally:
        CFG.paths.PHASE1_ROOT, CFG.paths.PHASE2_ROOT = saved


def test_real():
    """Read-only, real archive: the pinned cell (acceptance) and the Phase-1 code of each run."""
    rep_p, rep_u = {}, {}
    pinned = I.phase1_frames('18h', '2025-05-05', '41MHz', report=rep_p)
    unpinned = I.phase1_frames('18h', '2025-05-05', '41MHz', report=rep_u, run_pins=os.path.join(T, 'no_pins.csv'))
    check(f'real 18h 2025-05-05 41MHz Stokes I: {len(pinned["I"])} frames pinned (Run_20260603_010316), '
          f'{len(unpinned["I"])} unpinned (Run_20261020_205211)',
          len(pinned['I']) == 356 and len(unpinned['I']) == 358
          and {p.split('/')[-4] for _, p in pinned['I'] + pinned['V']} == {'Run_20260603_010316'}
          and {p.split('/')[-4] for _, p in unpinned['I']} == {'Run_20261020_205211'}
          and rep_p['pin']['run'] == 'Run_20260603_010316', (len(pinned['I']), len(unpinned['I']), rep_p.get('pin')))
    check('real Phase-1 code: the pinned (June) run 70d5f1c, the October run 908dc6e',
          {r: c['pipeline_version'] for r, c in rep_p['phase1_code'].items()} == {'Run_20260603_010316': '70d5f1c'}
          and {r: c['pipeline_version'] for r, c in rep_u['phase1_code'].items()} == {'Run_20261020_205211': '908dc6e'},
          (rep_p['phase1_code'], rep_u['phase1_code']))
    cells = [('18h', '2025-05-05', '41MHz')]
    pins = CFG.paths.PHASE1_RUN_PINS
    for label, pf, mine in (('pinned', pins, pinned), ('unpinned', os.path.join(T, 'no_pins.csv'), unpinned)):
        prod = production_discovery(CFG.paths.PHASE1_ROOT, cells, pf)['18h_2025-05-05']
        ok = not isinstance(prod, str) and all(prod[st] == [list(x) for x in mine[st]] for st in ('I', 'V'))
        check(f'real {label}: the (key, path) lists equal production discover_frames (I and V)', ok,
              prod if isinstance(prod, str) else (len(prod['I']), len(mine['I'])))
    g = I.phase1_code_of('/lustre/pipeline/exopipe/phase2/18h/2025-05-05/Science_20260610_063632/Green/I/deep/x.fits')
    check('real Phase-2 18h 2025-05-05 Green (Science_20260610_063632): its log names the June run for all five subbands',
          [(sb, c['run'], c['pipeline_version']) for _n, sb, c in g] ==
          [(sb, 'Run_20260603_010316', '70d5f1c') for sb in ('41MHz', '46MHz', '50MHz', '55MHz', '59MHz')], g)


def test_run_pins_env_override():
    """P3_SNAP_RUN_PINS: production's discover_frames reads its pin file from that variable at the module's import (clv15
    facb3e8), so a stale shell variable would override the protected paths.PHASE1_RUN_PINS - the frames step's default
    (legacy) discovery refuses it, as it refuses P3_SNAP_FLOW; the curves lock binds RUN_PINS too."""
    from types import SimpleNamespace
    from exopipe.phase3.core import env as E
    E.legacy()
    import phase3_snapshot_search as SS
    from exopipe.phase3.lib.frames import tier_snapshot as FT
    from exopipe.phase3.lib.curves import lock
    ctx = SimpleNamespace(discovery='legacy')
    saved = SS.RUN_PINS
    try:
        SS.RUN_PINS = os.path.join(T, 'stale_pins.csv')
        e = raises(Refusal, FT.discover, ctx, '18h', '2025-05-05', '41MHz')
        check('frames (legacy discovery): a P3_SNAP_RUN_PINS naming another file refuses before discovery', e is not None
              and 'P3_SNAP_RUN_PINS' in str(e) and 'PHASE1_RUN_PINS' in str(e), e)
    finally:
        SS.RUN_PINS = saved
    if '--no-real' not in sys.argv:
        paths, _rep = FT.discover(ctx, '18h', '2025-05-05', '41MHz')
        check('frames (legacy discovery) with the protected pin file: the pinned cell gives 356 + 356 frames of the June run',
              len(paths) == 712 and all('/Run_20260603_010316/' in p for p in paths), len(paths))
    out = os.path.join(T, 'pins_cli')
    env = dict(os.environ, P3_SNAP_RUN_PINS=os.path.join(T, 'stale_pins.csv'), PYTHONPATH=REPO)
    r = subprocess.run([sys.executable, '-m', 'exopipe.phase3', 'frames', '--tier', 'snapshot', '--cells',
                        '18h_2025-05-05_41MHz', '--stages', 'stage', '--dry-run', '--out', out, '--work',
                        os.path.join(T, 'pins_work'), '--cores', 'none'], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=600)
    check('CLI: `P3_SNAP_RUN_PINS=<other file> exopipe3 frames --tier snapshot ... --dry-run` refuses (exit 2)',
          r.returncode == 2 and 'P3_SNAP_RUN_PINS' in r.stdout + r.stderr, (r.returncode, (r.stdout + r.stderr)[-500:]))
    clv = E.legacy_root('clv15')
    ok, probs, files = lock.check(clv, CFG.curves, CFG.paths)
    check('curves lock: the real clv15 RUN_PINS default = paths.PHASE1_RUN_PINS', not [p for p in probs if 'RUN_PINS' in p],
          probs)
    old = os.environ.get('P3_SNAP_RUN_PINS')
    os.environ['P3_SNAP_RUN_PINS'] = os.path.join(T, 'stale_pins.csv')
    try:
        ok, probs, _ = lock.check(clv, CFG.curves, CFG.paths)
    finally:
        if old is None:
            os.environ.pop('P3_SNAP_RUN_PINS', None)
        else:
            os.environ['P3_SNAP_RUN_PINS'] = old
    check('curves lock: a P3_SNAP_RUN_PINS naming another file in the environment refuses (G13)', not ok
          and any('P3_SNAP_RUN_PINS=' in p for p in probs), probs)
    root = os.path.join(T, 'lockcopy')
    os.makedirs(root, exist_ok=True)
    for f in files:
        shutil.copy(f, root)
    ssp = os.path.join(root, 'phase3_snapshot_search.py')
    src = open(ssp).read()
    assert CFG.paths.PHASE1_RUN_PINS in src
    open(ssp, 'w').write(src.replace(CFG.paths.PHASE1_RUN_PINS, '/somewhere/else/pins.csv'))
    ok, probs, _ = lock.check(root, CFG.curves, CFG.paths)
    check('curves lock: an extractor whose RUN_PINS default moved refuses (G13)', not ok
          and any('RUN_PINS' in p and '/somewhere/else' in p for p in probs), probs)

def main(argv):
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    tests = [test_frame_time_constant, test_run_pin_and_identity, test_phase1_code_recorded, test_run_pins_env_override]
    if '--no-real' not in argv:
        tests.append(test_real)
    for t in tests:
        try:
            t()
        except Exception as e:                     # noqa: BLE001 - a crashing test is a failure, the rest still run
            import traceback
            traceback.print_exc()
            check(f'{t.__name__} raised {type(e).__name__}: {e}', False)
    shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed' + (f'; FAILED: {FAILS}' if FAILS else ''))
    return 1 if FAILS else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
