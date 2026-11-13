#!/usr/bin/env python3
"""STANDING TEST - the scopes of an exclusion read (production exopipe a8b56d3 + 36351ca, clv15 12b9a71; 2026-11-13).

A data_decisions.csv row whose reason carries '[SNAPSHOT-TIERS-ONLY' excludes in the snapshot tiers only (10-s, 1-minute,
10-minute): the default scope (co-adds, deep tier) skips it, scopes 'snapshot' and 'snapshot-gpu' honor it, and the snapshot
tiers read per night ('snapshot-gpu' on a GPU-frame cell, 'snapshot' on every other night). Asserts, for both v2 readers
(core.exclusions and lib/masters/coadd, the port of production's coadd_full_pipeline):
  (i)    the default scope skips a tagged row: its key is absent and the key set equals the untagged list's;
  (ii)   'snapshot' and 'snapshot-gpu' honor it: the key is present, and it is the only key added;
  (iii)  the band rule on the tagged 82 MHz row: Blue and Full excluded, Red and Green not (default: none of them);
  (iv)   an unknown scope or tier refuses;
  (v)    core.exclusions.for_tier: the default for the deep tier, the per-night read for the snapshot tiers;
  (vi)   the two v2 readers agree, per scope and per night, over every (LST, night) of the list and every subband and band;
  (vii)  the 10-minute search's read (production's three checks of 36351ca): every GPU-readmit row on a GPU-frame cell is
         admitted; no default key of a non-GPU night is missed; the tagged row is excluded at 10 minutes for 82 MHz and Blue,
         and not by the default; lib/search/deep_search.run_deep_cell reads it at the 10-minute cadence only;
  (viii) config constants equal the co-add port's;
  (ix)   on the cluster: production's coadd_full_pipeline (run read-only in a child process) gives the same key sets and the
         same per-night verdicts as the v2 co-add port.

Inputs: a COPY of the real /lustre/gh/main/clv15/data_decisions.csv (read only) and of the GPU-frame cell list when they exist,
else small synthetic lists (CI); the tagged row (23h, 82 MHz, test-only night 2099-12-30) is appended to the copy only. Scratch: a fresh
directory under $EXOPIPE3_TEST_SCRATCH, else the system temp directory, removed at exit.

Run standalone:  python tests/test_exclusion_scope.py
Under pytest:    pytest tests/test_exclusion_scope.py
"""
import atexit
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

_IMPORT_ERROR = None
try:
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import exclusions as E
    from exopipe.phase3.core.record import Refusal
    import exopipe.phase3.lib.masters.coadd as CO
except Exception as e:                                    # pragma: no cover
    _IMPORT_ERROR = f'{e.__class__.__name__}: {e}'

REAL_CSV = '/lustre/gh/main/clv15/data_decisions.csv'
REAL_GPU = '/lustre/gh/main/exopipe_ops/snapshot_gpu_frame_cells.txt'
PRODUCTION = '/lustre/gh/main/exopipe'
KEY = ('23h', '82', '2099-12-30')   # a night no real row names (2025-10-29 became a real snapshot-only row 2026-11-13)
TAG = '[SNAPSHOT-TIERS-ONLY'
GPU_TAG = '[SNAPSHOT-GPU-READMITTED'
SCOPES = (None, 'snapshot', 'snapshot-gpu')
SUBBANDS = ('18', '23', '27', '32', '36', '41', '46', '50', '55', '59', '64', '69', '73', '78', '82')
BANDS = ('Red', 'Green', 'Blue', 'Full')
HEADER = ['lst', 'subband', 'date', 'verdict', 'reason', 'score_thermal_noise', 'score_flux_scale', 'score_ionosphere']
SYNTH_ROWS = [['23h', '64MHz', '2025-10-28', 'exclude', 'synthetic whole-band row'],
              ['01h', '18MHz', '2024-12-18', 'exclude', GPU_TAG + '] synthetic GPU re-admission'],
              ['01h', '46MHz', '2024-12-19', 'exclude', '[PERBAND] synthetic per-image row'],
              ['05h', '23MHz', '2025-01-02', 'keep', 'synthetic readmission']]

_FX = {}


def _fixture():
    """{dir, untagged, tagged, gpu, real} - built once."""
    if _FX:
        return _FX
    root = os.environ.get('EXOPIPE3_TEST_SCRATCH') or tempfile.gettempdir()
    os.makedirs(root, exist_ok=True)
    d = tempfile.mkdtemp(prefix='excl_scope_', dir=root)
    atexit.register(shutil.rmtree, d, True)
    un, tg, gp = (os.path.join(d, n) for n in ('untagged.csv', 'tagged.csv', 'gpu_cells.txt'))
    real = os.path.exists(REAL_CSV)
    if real:
        shutil.copyfile(REAL_CSV, un)
        with open(REAL_CSV, 'rb') as a, open(un, 'rb') as b:
            assert a.read() == b.read(), 'the copy of data_decisions.csv differs from the file'
    else:
        with open(un, 'w', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(HEADER)
            for r in SYNTH_ROWS:
                w.writerow(r + ['', '', ''])
    if os.path.exists(REAL_GPU):
        shutil.copyfile(REAL_GPU, gp)
    else:
        with open(gp, 'w') as f:
            f.write('# synthetic GPU-frame cells\n01h 2024-12-18\n01h 2024-12-19\n')
    with open(un, 'rb') as f:
        raw = f.read()
    eol = '\r\n' if b'\r\n' in raw else '\n'
    with open(tg, 'wb') as f:
        f.write(raw if raw.endswith(b'\n') else raw + eol.encode())
    with open(tg, 'a', newline='') as f:
        csv.writer(f, lineterminator=eol).writerow(
            ['23h', '82MHz', '2099-12-30', 'exclude', TAG + '] test row (tests/test_exclusion_scope.py), never in the real list',
             '', '', ''])
    _FX.update(dir=d, untagged=un, tagged=tg, gpu=gp, real=real)
    return _FX


class _CoaddFiles:
    """The co-add port reading ``csv_path`` and ``gpu_path`` (module attributes patched in-process, restored on exit)."""

    def __init__(self, csv_path, gpu_path=None):
        self.csv_path, self.gpu_path = csv_path, gpu_path

    def __enter__(self):
        self.was = (CO.DECISIONS_CSV, CO.GPU_FRAME_CELLS_FILE)
        CO.DECISIONS_CSV = self.csv_path
        if self.gpu_path:
            CO.GPU_FRAME_CELLS_FILE = self.gpu_path
        return CO

    def __exit__(self, *a):
        CO.DECISIONS_CSV, CO.GPU_FRAME_CELLS_FILE = self.was


def _keys(path, scope):
    return set(E.Exclusions(path, scope=scope).rows)


def _grid(path):
    """[(lst, scope, date)] over every (LST, night) of the list x every subband and band."""
    cells = set()
    with open(path) as f:
        for r in csv.DictReader(f):
            cells.add((r['lst'].strip(), r['date'].strip()))
    return [(l, s, d) for (l, d) in sorted(cells) for s in SUBBANDS + BANDS]


def _gpu_cells():
    return E.gpu_frame_cells(_fixture()['gpu'])


# ---- tests ------------------------------------------------------------------------------------------------------------
def test_default_skips_tagged_row():
    fx = _fixture()
    a, b = _keys(fx['untagged'], None), _keys(fx['tagged'], None)
    assert KEY not in b, 'core: the default scope counts the [SNAPSHOT-TIERS-ONLY row'
    assert a == b and len(a) == len(b), f'core: default key sets differ ({len(a)} untagged, {len(b)} tagged)'
    with _CoaddFiles(fx['untagged']):
        ca = CO.load_exclusions()
    with _CoaddFiles(fx['tagged']):
        cb = CO.load_exclusions()
    assert KEY not in cb and ca == cb, f'co-add port: default key sets differ ({len(ca)} untagged, {len(cb)} tagged)'
    print(f'      default: {len(b)} keys, tagged copy = untagged copy, key absent (core and co-add port)')


def test_snapshot_scopes_honor_tagged_row():
    fx = _fixture()
    for sc in ('snapshot', 'snapshot-gpu'):
        a, b = _keys(fx['untagged'], sc), _keys(fx['tagged'], sc)
        assert KEY in b and b - a == {KEY} and a <= b, f'core {sc}: added keys {sorted(b - a)}'
        with _CoaddFiles(fx['tagged']):
            cb = CO.load_exclusions(scope=sc)
        assert KEY in cb and cb == b, f'co-add port {sc}: key present {KEY in cb}, equal to core {cb == b}'
        print(f'      {sc}: {len(b)} keys = untagged {len(a)} + the tagged key')


def test_band_rule():
    fx = _fixture()
    l, s, d = KEY
    for sc in SCOPES:
        un, tg = E.Exclusions(fx['untagged'], scope=sc), E.Exclusions(fx['tagged'], scope=sc)
        assert not any(un.is_excluded(l, b, d) for b in BANDS + (s,)), f'{sc}: {l} {d} already excluded without the row'
        got = {b: tg.is_excluded(l, b, d) for b in BANDS + (s,)}
        want = ({'Red': False, 'Green': False, 'Blue': False, 'Full': False, s: False} if sc is None else
                {'Red': False, 'Green': False, 'Blue': True, 'Full': True, s: True})
        assert got == want, f'core {sc}: {got}'
        with _CoaddFiles(fx['tagged']):
            ex = CO.load_exclusions(scope=sc)
            cgot = {b: bool(CO.is_excluded(l, b, d, ex)) for b in BANDS + (s,)}
        assert cgot == want, f'co-add port {sc}: {cgot}'
    print(f'      {l} {d}: default excludes none of 82 MHz/Red/Green/Blue/Full; snapshot scopes exclude 82 MHz, Blue, Full')


def test_unknown_scope_refuses():
    fx = _fixture()
    for bad in ('deep', 'snapshot_gpu', ''):
        try:
            E.Exclusions(fx['tagged'], scope=bad)
        except Refusal:
            pass
        else:
            raise AssertionError(f'core: scope {bad!r} did not refuse')
        with _CoaddFiles(fx['tagged']):
            try:
                CO.load_exclusions(scope=bad)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f'co-add port: scope {bad!r} did not refuse')
    for bad in ('Deep', 'sub-lst', 'forced'):
        try:
            E.for_tier(bad, path=fx['tagged'])
        except Refusal:
            pass
        else:
            raise AssertionError(f'for_tier({bad!r}) did not refuse')


def test_for_tier():
    fx = _fixture()
    for t in (None, 'deep'):
        ex = E.for_tier(t, fresh=True, path=fx['tagged'])
        assert isinstance(ex, E.Exclusions) and ex.scope is None and KEY not in ex.rows, f'for_tier({t!r})'
    for t in CFG.exclusions.SNAPSHOT_SCOPE_TIERS:
        ex = E.for_tier(t, fresh=True, path=fx['tagged'])
        assert isinstance(ex, E.PerNight) and ex.is_excluded(*KEY), f'for_tier({t!r})'
    assert set(CFG.exclusions.SNAPSHOT_SCOPE_TIERS) == {'snapshot', 'sublst', '10s', '1min', '10min'}
    for bad in ('forced', 'tenmin', ''):
        try:
            E.for_tier(bad, fresh=True, path=fx['tagged'])
        except Refusal:
            pass
        else:
            raise AssertionError(f'for_tier({bad!r}) did not refuse')


def test_core_and_coadd_agree():
    fx = _fixture()
    grid = _grid(fx['tagged'])
    for sc in SCOPES:
        core = E.Exclusions(fx['tagged'], scope=sc)
        with _CoaddFiles(fx['tagged']):
            ex = CO.load_exclusions(scope=sc)
            bad = [(l, s, d) for (l, s, d) in grid if core.is_excluded(l, s, d) != bool(CO.is_excluded(l, s, d, ex))]
        assert set(core.rows) == ex, f'{sc}: key sets differ'
        assert not bad, f'{sc}: {len(bad)} verdicts differ, e.g. {bad[:3]}'
    pn = E.PerNight(fx['tagged'], gpu_cells=_gpu_cells(), fresh=True)
    with _CoaddFiles(fx['tagged'], fx['gpu']):
        x10 = CO.snapshot_exclusion_test()
        bad = [(l, s, d) for (l, s, d) in grid if pn.is_excluded(l, s, d) != x10(l, s, d)]
    assert not bad, f'per night: {len(bad)} verdicts differ, e.g. {bad[:3]}'
    print(f'      {len(grid)} (LST, night, subband or band) verdicts agree per scope and per night')


def test_tenmin_search_read():
    fx = _fixture()
    gpu = _gpu_cells()
    with open(fx['tagged']) as f:
        rows = [r for r in csv.DictReader(f) if (r.get('verdict') or '').strip() == 'exclude']
    readmit = [(r['lst'].strip(), r['subband'].replace('MHz', '').strip(), r['date'].strip()) for r in rows
               if GPU_TAG in (r.get('reason') or '')]
    on_gpu = [k for k in readmit if (k[0], k[2]) in gpu]
    pn = E.PerNight(fx['tagged'], gpu_cells=gpu, fresh=True)
    with _CoaddFiles(fx['tagged'], fx['gpu']):
        default = CO.load_exclusions()
        x10 = CO.snapshot_exclusion_test()
        n_adm = sum(1 for (l, s, d) in on_gpu if not x10(l, s, d) and not pn.is_excluded(l, s, d))
        missed = [(l, s, d) for (l, s, d) in default if (l, d) not in gpu and not (x10(l, s, d) and pn.is_excluded(l, s, d))]
        l, s, d = KEY
        tenmin = {b: (x10(l, b, d), pn.is_excluded(l, b, d)) for b in (s, 'Blue')}
        ex = CO.load_exclusions()
        by_default = {b: bool(CO.is_excluded(l, b, d, ex)) for b in (s, 'Blue')}
    print(f'      GPU-readmit rows admitted on GPU-frame cells: {n_adm} of {len(on_gpu)} ({len(readmit)} such rows in all); '
          f'default keys of non-GPU nights missed: {len(missed)}; tagged row at 10 min (82 MHz, Blue): {tenmin}, '
          f'by default: {by_default}')
    assert on_gpu and n_adm == len(on_gpu), f'{n_adm} of {len(on_gpu)} GPU-readmit rows on GPU-frame cells admitted'
    if fx['real']:
        assert len(readmit) == len(on_gpu), f'{len(readmit) - len(on_gpu)} GPU-readmit rows on cells not in the GPU list'
    assert not missed, f'{len(missed)} default keys of non-GPU nights missed, e.g. {missed[:3]}'
    assert all(a and b for a, b in tenmin.values()), f'tagged row at 10 min: {tenmin}'
    assert not any(by_default.values()), f'tagged row by default: {by_default}'
    src = open(os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'search', 'deep_search.py')).read()
    i = src.index('def run_deep_cell(')
    j = min([k for k in (src.find('\ndef ', i + 1), src.find('\nclass ', i + 1)) if k > 0] or [len(src)])
    body = src[i:j]
    assert ("if cadence == '10min':\n        from exopipe.phase3.core.exclusions import refuse_frame_overrides\n"
            "        from exopipe.phase3.lib.masters.coadd import snapshot_exclusion_test\n"
            "        refuse_frame_overrides()" in body
            and "        _xtest = snapshot_exclusion_test()\n    else:\n        _excl = load_exclusions()\n" in body), \
        'run_deep_cell: the 10-minute read is not snapshot_exclusion_test (deep: the default)'
    assert 'is_excluded(lst, scope, _kd(d), _excl)' not in body, 'run_deep_cell: a night test bypasses the cadence read'
    for rel, call, old_ in (('tenmin_frames.py', '    xtest = snapshot_exclusion_test()', 'load_exclusions()'),
                            ('phase3_tenmin.py', '    _xtest = snapshot_exclusion_test()', 'load_exclusions()')):
        s_ = open(os.path.join(REPO, 'exopipe', 'phase3', rel)).read()
        assert call in s_ and old_ not in s_ and 'is_excluded(' not in s_, \
            f'{rel}: a 10-minute night read is not snapshot_exclusion_test (the default scope keeps a snapshot-only night)'


def test_constants_mirror():
    x = CFG.exclusions
    assert (x.SNAPSHOT_ONLY_TAG, x.GPU_READMIT_TAG, tuple(x.SCOPES)) == \
        (CO.SNAPSHOT_ONLY_EXCLUDE_TAG, CO.SNAPSHOT_GPU_READMIT_TAG, CO.EXCLUSION_SCOPES)
    assert x.SNAPSHOT_ONLY_TAG == TAG and x.GPU_READMIT_TAG == GPU_TAG
    if 'P3_SNAP_GPU_CELLS' not in os.environ:
        assert CFG.paths.SNAPSHOT_GPU_FRAME_CELLS == CO.GPU_FRAME_CELLS_FILE == REAL_GPU
    eng = open(os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'snapshot_engine', 'phase3_snapshot_search.py')).read()
    assert repr(REAL_GPU).replace("'", '"') in eng, 'the vendored engine reads another GPU-frame cell list'
    from exopipe.phase3.config import shared
    assert 'paths.SNAPSHOT_GPU_FRAME_CELLS' in shared.PROTECTED, 'the GPU-frame cell list is not a protected key'


_CHILD = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import exopipe.phase3.coadd_full_pipeline as C
assert C.__file__.startswith(sys.argv[1]), C.__file__
C.DECISIONS_CSV, C.GPU_FRAME_CELLS_FILE = sys.argv[2], sys.argv[3]
grid = json.load(open(sys.argv[4]))
out = {}
for sc in (None, 'snapshot', 'snapshot-gpu'):
    ex = C.load_exclusions(scope=sc)
    out[str(sc)] = sorted(list(k) for k in ex)
    out['v_' + str(sc)] = [bool(C.is_excluded(l, s, d, ex)) for l, s, d in grid]
x10 = C.snapshot_exclusion_test()
out['x10'] = [bool(x10(l, s, d)) for l, s, d in grid]
json.dump(out, sys.stdout)
'''


def test_production_agrees():
    fx = _fixture()
    if not os.path.exists(os.path.join(PRODUCTION, 'exopipe', 'phase3', 'coadd_full_pipeline.py')):
        print('      UNTESTED here: no production tree at ' + PRODUCTION + ' (cluster only)')
        return
    grid = _grid(fx['tagged'])
    gp = os.path.join(fx['dir'], 'grid.json')
    with open(gp, 'w') as f:
        json.dump(grid, f)
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    r = subprocess.run([sys.executable, '-c', _CHILD, PRODUCTION, fx['tagged'], fx['gpu'], gp], cwd=fx['dir'], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=900)
    assert r.returncode == 0, f'production child failed: {r.stderr[-800:]}'
    prod = json.loads(r.stdout)
    with _CoaddFiles(fx['tagged'], fx['gpu']):
        for sc in SCOPES:
            ex = CO.load_exclusions(scope=sc)
            assert {tuple(k) for k in prod[str(sc)]} == ex, f'{sc}: key sets differ from production'
            v = [bool(CO.is_excluded(l, s, d, ex)) for l, s, d in grid]
            bad = [g for g, a, b in zip(grid, v, prod['v_' + str(sc)]) if a != b]
            assert not bad, f'{sc}: {len(bad)} verdicts differ from production, e.g. {bad[:3]}'
        x10 = CO.snapshot_exclusion_test()
        bad = [g for g, b in zip(grid, prod['x10']) if bool(x10(*g)) != b]
    assert not bad, f'per night: {len(bad)} verdicts differ from production, e.g. {bad[:3]}'
    print(f'      production coadd_full_pipeline: same key sets per scope and same {len(grid)} verdicts per scope and per night')


def test_overrides_and_fingerprint():
    """An environment override of the GPU-frame cell list or frame root refuses the per-night read (the in-place engines
    would follow it, core.exclusions would not); the per-night fingerprint changes with the cell set."""
    fx = _fixture()
    saved = {k: os.environ.pop(k, None) for k in ('P3_SNAP_GPU_CELLS', 'P3_SNAP_PHASE1_ROOT')}
    try:
        for var, val in (('P3_SNAP_GPU_CELLS', fx['gpu']),
                         ('P3_SNAP_PHASE1_ROOT', '/lustre/pipeline/snapshots_gpu_pipeline')):
            os.environ[var] = val
            try:
                E.PerNight(fx['tagged'], fresh=True)
            except Refusal:
                pass
            else:
                raise AssertionError(f'{var}={val}: the per-night read did not refuse')
            del os.environ[var]
        os.environ['P3_SNAP_GPU_CELLS'] = CFG.paths.SNAPSHOT_GPU_FRAME_CELLS     # the configured list itself: no refusal
        E.refuse_frame_overrides()
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    cells = sorted(_gpu_cells())
    a = E.PerNight(fx['tagged'], gpu_cells=cells, fresh=True).fingerprint()
    b = E.PerNight(fx['tagged'], gpu_cells=cells[1:], fresh=True).fingerprint()
    snap = E.Exclusions(fx['tagged'], scope='snapshot').fingerprint()
    assert all(a[k] == b[k] == snap[k] for k in snap), 'the per-night fingerprint lost the list fingerprint'
    assert a['gpu_frame_cells'] != b['gpu_frame_cells'], 'the cell set is not fingerprinted'
    assert a['n_gpu_frame_cells'] == len(cells), a
    assert a == E.PerNight(fx['tagged'], gpu_cells=list(reversed(cells)), fresh=True).fingerprint(), 'order-dependent'
    print(f'      overrides refuse; fingerprint covers the {len(cells)} GPU-frame cells')


TESTS = [test_default_skips_tagged_row, test_snapshot_scopes_honor_tagged_row, test_band_rule, test_unknown_scope_refuses,
         test_for_tier, test_core_and_coadd_agree, test_tenmin_search_read, test_constants_mirror, test_production_agrees,
         test_overrides_and_fingerprint]


def main():
    if _IMPORT_ERROR is not None:
        print(f'UNTESTED-IMPORT: cannot import dependencies in this interpreter ({_IMPORT_ERROR}) - run under the pipeline env')
        return 1
    failed = 0
    fx = _fixture()
    print(f'inputs: {"copy of " + REAL_CSV if fx["real"] else "synthetic list (no " + REAL_CSV + ")"}, '
          f'{"copy of " + REAL_GPU if os.path.exists(REAL_GPU) else "synthetic GPU-frame cell list"}; scratch {fx["dir"]}')
    for t in TESTS:
        try:
            t()
            print(f'PASS  {t.__name__}')
        except AssertionError as e:
            failed += 1
            print(f'FAIL  {t.__name__}: {e}')
    print(f'{len(TESTS) - failed}/{len(TESTS)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
