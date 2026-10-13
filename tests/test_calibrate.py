#!/usr/bin/env python3
"""STANDING TEST - `exopipe3 calibrate` (versioned reference tables), synthetic data plus READ-ONLY checks of real products.

  config          sections load, every RULED ledger ID is a fix_ledger row, tables.* protected, CONFIG declared
  lock            the in-place post_process is locked against EXCESS_GUARD_CANDIDATES / MONOTONE_CAP (refuses on a change)
  key_loops       ported build loops (deep band / all-elevation, multiples, sub-LST) on synthetic derivations; G6/G7/G8
  merge           install merge rules per class; families outside the class untouched (G10)
  config_patch    pointer consistency over every config source, scoped provenance edits, a K entry edit (text only)
  deep_e2e        CLI: build -> diff (table-only, waiver) -> install (ruling, G22) -> verify; fresh / resume; negatives
  served_replay   diff with the conformance table step replayed (fake gate inputs): newly removed, served, judged, tally
  orchestration   derive inputs + curves with the in-place call faked: harvest frozen, excluded rows counted (G23),
                  production environment and arguments of every sub-run; multiples exclude lists; build from the record
  sublst          sub-LST whole-file build and install (G21 sublst_present)
  scale           scale_table_v4.py IN PLACE on synthetic ratios (also under --parity); G11; recompute test
  fluxcorr        k x g table in process; G12; band-vector iteration (the in-place check faked) converges / G13
  template_k      summary of synthetic trials; diff; install G16 / G17 / G18; K patch
  real            READ ONLY (lwacalim07): `calibrate all verify` on every installed table; parity rebuilds from production's
                  own derivations (deep band 114 keys, multiples 89 keys = installed records; fluxcorr v2 byte-identical;
                  scale v4 identical but measured_at; deep K = config); `--dry-run` of a real deep derivation

Scratch: /tmp/exopipe3_tests_calibrate/ (removed at the end). Production files are only read.
Run standalone:  python tests/test_calibrate.py [test_name ...]
"""
import contextlib
import csv
import io
import json
import os
import shutil
import socket
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')
SCRATCH = '/tmp/exopipe3_tests_calibrate'
RULING = 'R-DEEP-BRIGHT-SOURCE-TABLES-ELEVATION-BANDS-AND-FRAME-S'


def _fresh(name):
    d = os.path.join(SCRATCH, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return d


@contextlib.contextmanager
def patched(obj, **kw):
    """Test-only: set attributes of a config section (or module) in memory; restored afterwards."""
    old = {k: getattr(obj, k) for k in kw}
    try:
        for k, v in kw.items():
            setattr(obj, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


@contextlib.contextmanager
def synthetic_tables(d, **values):
    """Test-only: point config section `tables` at synthetic files - in memory (the step reads CFG) AND in a synthetic config
    source parsed first by the install's config patch (lib/calibrate/common.config_sources)."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.calibrate import common as K
    src = os.path.join(d, '_cfgdraft_synthetic_tables.py')
    open(src, 'w').write('from exopipe.phase3.config import _Section\n\n\nclass tables(_Section):\n'
                         + ''.join(f'    {k} = {v!r}\n' for k, v in values.items()))
    real = K.config_sources
    with patched(CFG.tables, **values), patched(K, config_sources=lambda: [src] + real()):
        yield


def cli(*argv):
    """exopipe3 in process; returns (exit code, stderr+stdout text)."""
    from exopipe.phase3 import cli as C
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
        try:
            rc = C.main(list(argv) + (['--cores', 'none'] if '--help' not in argv else []))
        except SystemExit as e:              # argparse (--help, a bad flag)
            rc = e.code
    return rc, buf.getvalue()


def rec(out, unit):
    p = os.path.join(out, '_records', 'calibrate', f'{unit}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def failed(r):
    return [t['name'] for t in r['tests'] if not t['passed'] and not t.get('waived')]


def wjson(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(obj, open(p, 'w'), indent=1)
    return p


def wcsv(p, fields, rows, lineterminator='\n'):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator=lineterminator)
        w.writeheader()
        w.writerows(rows)
    return p


# ---------------------------------------------------------------------------------------------------------------------
# synthetic tables
# ---------------------------------------------------------------------------------------------------------------------

def curve(bins, n=1000, cells=3):
    """bins: [(S_lo, S_hi, R2, marginal_excess)] ascending S."""
    bl = [dict(S_lo=a, S_hi=b, R2=r, R15=r, marginal_excess=e, n_parents=10, capped=False) for a, b, r, e in bins]
    return dict(metric='image-pixel', n_candidates=n, n_cells=cells, union_kill_fraction=0.5, null_area_fraction=0.1,
                bins=bl, Smin=None, R_cap=40.0, table=[[a, b, r] for a, b, r, e in bins if r > 0])


def derivation(curves, ratio, eb='20-25,25-35,35-50,50-90', mult=None, cadence='deep'):
    return {'meta': dict(derived_utc='2030-01-01T00:00:00Z', floor=5.0, by='subband', ratio=ratio, multiplicity=mult,
                         match_beams=1.0, metric='image-pixel', elevation_bands=eb, cadence=cadence,
                         parents='synthetic parents', beam_unit='synthetic beam'), 'curves': curves}


BINS_A = [(20, 35, 0.5, 0.05), (35, 60, 1.0, 0.05), (60, 100, 1.5, 0.05), (200, 1e6, 2.0, 0.05)]   # x n 1000 = 50 >= 20
BINS_B = [(35, 60, 0.75, 0.05), (100, 200, 2.0, 0.01), (200, 1e6, 3.0, 0.05)]                    # 100-200: 10 < 20 dropped


def synthetic_installed(d):
    """A deep installed file: singles all-elevation + band keys, multiples, retired 10min keys, the three meta records."""
    T = {}
    for S, b, ratio in (('I', '18MHz', '3.0'), ('V', '18MHz', '2.0'), ('I', '41MHz', '3.0'), ('V', '41MHz', '3.0')):
        T[f'deep_{S}_{b}'] = dict(stokes=S, band=b, ratio=ratio, metric='image-pixel', table=[[200, 1e6, 1.0]],
                                  raw_table=[[200, 1e6, 1.0]], n_candidates=500, n_cells=3, provisional=False,
                                  derived_utc='2029-01-01', source='old.json', parents='P', beam_unit='BU')
        T[f'deep_{S}_{b}_el50-90'] = dict(T[f'deep_{S}_{b}'], table=[[200, 1e6, 1.25]], elevation_band=[50, 90])
        T[f'deep_{S}-multi_{b}'] = dict(T[f'deep_{S}_{b}'], stokes=f'{S}-multi', table=[[100, 1e6, 0.5]])
    T['10min_I_18MHz'] = dict(stokes='I', band='18MHz', table=[[50, 1e6, 2.0]])
    return wjson(os.path.join(d, 'installed', 'parent_curve_tables.json'),
                 {'meta': {'method': 'synthetic'}, 'tables': T, 'meta_10min': {'a': 1}, 'meta_deep': {'2029-01-01T00:00:00Z': {'x': 1}}})


def synthetic_derivations(d):
    r3 = {}
    r2 = {}
    for S, b in (('I', '18MHz'), ('V', '18MHz'), ('I', '41MHz'), ('V', '41MHz')):
        k = f'deep_{S}_{b}'
        r3[k] = curve(BINS_A)
        r3[k + '_el20-25'] = curve(BINS_A)
        r3[k + '_el25-35'] = curve(BINS_B)
        r3[k + '_el35-50'] = curve([(200, 1e6, 0.0, 0.0)])         # no radius: no band table (falls back)
        r3[k + '_el50-90'] = curve(BINS_A)
        if S == 'V':
            r2[k] = curve(BINS_B)
            r2[k + '_el50-90'] = curve(BINS_B)
            r2[k + '_el25-35'] = curve(BINS_A)
    p3 = wjson(os.path.join(d, 'deriv', 'deep_elev_r3.json'), derivation(r3, 3.0))
    p2 = wjson(os.path.join(d, 'deriv', 'deep_elev_r2V.json'), derivation(r2, 2.0))
    return p3, p2


def synthetic_multiples(d):
    out = []
    for name, S, ratio in (('multi_r3_I', 'I', 3.0), ('multi_r3_V', 'V', 3.0), ('multi_r2V', 'V', 2.0)):
        cv = {}
        for b in ('18MHz', '41MHz'):
            k = f'deep_{S}-multi_{b}'
            cv[k] = curve(BINS_A if ratio == 3.0 else BINS_B)
            cv[k + '_el20-25'] = curve(BINS_A)
            cv[k + '_el50-90'] = curve(BINS_B)
            cv[k + '_el35-50'] = curve([(200, 1e6, 0.0, 0.0)])
        out.append(wjson(os.path.join(d, 'deriv', f'{name}.json'), derivation(cv, ratio, mult='harvest')))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------------------------------------------------

def test_config():
    from exopipe.phase3.config import CFG, ConfigRefusal
    from exopipe.phase3.core import ledger
    from exopipe.phase3.steps import calibrate as step
    secs = ('tables', 'calibrate', 'calibrate_bright_parent', 'calibrate_multiples', 'calibrate_scale', 'calibrate_fluxcorr',
            'calibrate_template_k')
    for s in secs:
        assert s in CFG.sections(), s
        for k, v in (getattr(CFG._section(s), 'RULED', None) or {}).items():
            for lid in (v if isinstance(v, tuple) else (v,)):
                assert ledger.has_id(lid, CFG.paths.FIX_LEDGER), f'{s}.{k}: {lid} is not a fix_ledger row'
        for k in getattr(CFG._section(s), 'PROTECTED_KEYS', ()):
            assert hasattr(CFG._section(s), k), f'{s}.PROTECTED_KEYS names a missing key {k}'
    for k, _v in CFG.tables.items():
        try:
            CFG.set(f'tables.{k}', 'x')
            raise AssertionError(f'tables.{k} accepted an override')
        except ConfigRefusal:
            pass
    assert CFG.production_root_of(CFG.calibrate.OUT_ROOT) is not None        # refused in the window without --out
    assert set(step.CONFIG) >= set(secs)
    rc, txt = cli('calibrate', '--help')
    assert rc == 0 and 'Failure modes' in txt and '--install-ruling' in txt, txt[:300]


def test_lock():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.calibrate import bright_parent as BP
    BP._PP.clear()
    assert BP.post_process()(BP.LOCK_RECORD) == [[35, 60, 1.5], [100, 200, 3.0], [200, 1e6, 3.0]]
    for kw in ({'EXCESS_GUARD_CANDIDATES': 25}, {'MONOTONE_CAP': False}):
        BP._PP.clear()
        with patched(CFG.calibrate_bright_parent, **kw):
            try:
                BP.post_process()
                raise AssertionError(f'lock did not refuse {kw}')
            except Refusal as e:
                assert 'lock' in str(e)
    BP._PP.clear()


def test_key_loops():
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.calibrate import bright_parent as BP
    from exopipe.phase3.lib.calibrate import common as K
    d = _fresh('loops')
    inst = K.read_json(synthetic_installed(d))['tables']
    p3, p2 = synthetic_derivations(d)
    D3, D2 = K.read_json(p3), K.read_json(p2)
    T, L = BP.build_deep(D3, D2, p3, p2, inst, ('elevation-bands',))
    # I keys from r3; V 18 MHz from r2 (low-ratio band), V 41 MHz from r3; 20-25 report only; 35-50 no radius -> no key
    assert set(T) == {f'deep_{S}_{b}_{e}' for S in 'IV' for b in ('18MHz', '41MHz') for e in ('el25-35', 'el50-90')}, sorted(T)
    assert T['deep_V_18MHz_el50-90']['ratio'] == '2.0' and T['deep_V_18MHz_el50-90']['source'] == 'deep_elev_r2V.json'
    assert T['deep_V_41MHz_el50-90']['ratio'] == '3.0'
    assert T['deep_I_18MHz_el25-35']['table'] == [[35, 60, 0.75], [200, 1e6, 3.0]]      # 100-200 below the guard
    assert T['deep_I_18MHz_el50-90']['elevation_band'] == [50, 90]                       # ints, as production writes
    assert T['deep_I_18MHz_el50-90']['parents'] == 'P'                                    # copied from the installed key
    assert any('REPORT ONLY' in x for x in L)
    T2, _ = BP.build_deep(D3, D2, p3, p2, inst, ('all-elevation',))
    assert set(T2) == {f'deep_{S}_{b}' for S in 'IV' for b in ('18MHz', '41MHz')}
    assert T2['deep_I_18MHz']['table'] == [[20, 35, 0.5], [35, 60, 1.0], [60, 100, 1.5], [200, 1e6, 2.0]]
    bad = dict(D3, meta=dict(D3['meta'], elevation_bands=''))
    for args in ((bad, D2, p3, p2, inst, ('elevation-bands',)),):
        try:
            BP.build_deep(*args)
            raise AssertionError('G6 not refused')
        except Refusal:
            pass
    m = synthetic_multiples(d)
    Cm, Lm = BP.build_multiples(*(K.read_json(p) for p in m), m, {'I': '/x/I.txt', 'V': '/x/V.txt'})
    assert 'deep_I-multi_18MHz' in Cm and 'deep_I-multi_18MHz_el20-25' not in Cm
    assert Cm['deep_V-multi_18MHz']['ratio'] == '2.0' and Cm['deep_V-multi_18MHz']['source'] == 'multi_r2V.json'
    assert Cm['deep_V-multi_41MHz']['source'] == 'multi_r3_V.json' and Cm['deep_I-multi_18MHz']['excluded_cells'] == '/x/I.txt'
    assert 'deep_I-multi_18MHz_el35-50' not in Cm
    nm = dict(K.read_json(m[0]))
    nm['meta'] = dict(nm['meta'], multiplicity=None)
    try:
        BP.build_multiples(nm, K.read_json(m[1]), K.read_json(m[2]), m, {})
        raise AssertionError('G7 not refused')
    except Refusal:
        pass
    sub = derivation({'sublst_V_18MHz': curve(BINS_A), 'sublst_V_18MHz_el50-90': curve(BINS_B),
                      'sublst_I_Full_el25-35': curve([(200, 1e6, 0.0, 0.0)])}, 2.0, eb='25-35,35-50,50-90', cadence='10min')
    F, rep = BP.build_sublst(sub, '/x/sub.json')
    assert len([k for k in F['tables'] if '_el' not in k]) == 38                         # 19 bands x 2 Stokes
    assert F['tables']['sublst_I_18MHz']['note'].startswith('no harvest') and F['tables']['sublst_I_18MHz']['table'] == []
    assert 'sublst_V_18MHz_el50-90' in F['tables'] and 'sublst_I_Full_el25-35' not in F['tables']
    for badsub in (derivation({'deep_V_18MHz': curve(BINS_A)}, 2.0, cadence='10min'),
                   derivation({'sublst_V_18MHz': dict(curve(BINS_A), metric='sky')}, 2.0, cadence='10min')):
        try:
            BP.build_sublst(badsub, '/x')
            raise AssertionError('G8 not refused')
        except Refusal:
            pass


def test_merge():
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.calibrate import bright_parent as BP
    from exopipe.phase3.lib.calibrate import common as K
    d = _fresh('merge')
    base = K.read_json(synthetic_installed(d))
    cand = {'meta': {}, 'tables': {'deep_I_18MHz_el25-35': {'table': [[1, 2, 0.5]]}}}
    new, rep, rem = BP.merge('bright-parent', 'deep', base, cand)
    assert rep == ['deep_I_18MHz_el25-35'] and not rem and len(new['tables']) == len(base['tables']) + 1
    assert new['tables']['10min_I_18MHz'] == base['tables']['10min_I_18MHz'] and 'deep_I_18MHz_el25-35' not in base['tables']
    mc = {'meta': {}, 'tables': {'deep_I-multi_18MHz': {'table': []}}}
    new, rep, rem = BP.merge('multiples', 'deep', base, mc)
    assert rem == sorted(k for k in base['tables'] if '-multi_' in k and k != 'deep_I-multi_18MHz')
    assert not any('-multi_' in k for k in new['tables'] if k != 'deep_I-multi_18MHz')
    for cls, c in (('multiples', cand), ('bright-parent', mc)):
        try:
            BP.merge(cls, 'deep', base, c)
            raise AssertionError('family not refused')
        except Refusal:
            pass
    sc = {'meta': {'source': 's', 'derivation': {}, 'rule': 'r'}, 'tables': {'sublst_I_18MHz': {'table': []}}}
    new, rep, rem = BP.merge('bright-parent', 'sublst', {'tables': {'sublst_I_41MHz': {}}}, sc)
    assert set(new['meta']) == {'installed_utc', 'source', 'derivation', 'rule'} and rem == ['sublst_I_41MHz']


def test_config_patch():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.calibrate import common as K
    d = _fresh('patch')
    newf = os.path.join(d, 'new.json')
    open(newf, 'w').write('{}')
    old = CFG.tables.BRIGHT_PARENT_DEEP
    patch, changed = K.config_patch(K.pointer_replacements(old, newf, old, newf))
    keys = {f'{s}.{k}' for s, k in changed}
    assert {'tables.BRIGHT_PARENT_DEEP', 'tables.BRIGHT_PARENT_DEEP_SHA256'} <= keys, keys
    assert newf in patch and K.sha256(newf) in patch and '_DERIVATIONS' not in ''.join(keys)
    # scoped: a provenance key changes only by name
    patch2, ch2 = K.config_patch({}, scoped=[('tables', 'SCALE_TABLE_PREVIOUS', {CFG.tables.SCALE_TABLE_PREVIOUS: '/x/prev.csv'})])
    assert ch2 == [('tables', 'SCALE_TABLE_PREVIOUS')] and "+    SCALE_TABLE_PREVIOUS = '/x/prev.csv'" in patch2
    # one K entry
    rx = r"(\(\s*'deep'\s*,\s*'I'\s*\)\s*:\s*\()([^)]*)(\))"
    patch3, ch3 = K.config_patch({}, entry_edits=[('fitting', 'K_TEMPLATE', rx, r'\g<1>0.999, 0.888\g<3>')])
    assert ('fitting', 'K_TEMPLATE') in ch3 and '0.999, 0.888' in patch3 and 'config.py' in patch3
    for f in K.config_sources():                         # the patch never touches the sources
        assert '0.999, 0.888' not in open(f).read()


def _deep_setup(name):
    from exopipe.phase3.lib.calibrate import common as K
    d = _fresh(name)
    inst = synthetic_installed(d)
    p3, p2 = synthetic_derivations(d)
    return d, inst, p3, p2, K.sha256(inst)


def test_deep_e2e():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.calibrate import common as K
    d, inst, p3, p2, sha = _deep_setup('deep_e2e')
    out = os.path.join(d, 'out')
    with synthetic_tables(d, BRIGHT_PARENT_DEEP=inst, BRIGHT_PARENT_DEEP_SHA256=sha):
        rc, txt = cli('calibrate', 'bright-parent', 'derive', '--tier', 'deep', '--stages', 'build', '--curves', f'{p3},{p2}',
                      '--version', 't1', '--out', out)
        assert rc == 0, txt
        r = rec(out, 'bright-parent.deep.derive-build.t1')
        assert r['status'] == 'done' and not failed(r)
        cand = r['outputs'][0]['path']
        assert K.read_json(cand)['meta']['calibrate']['derivations'] == [p3, p2]
        assert open(cand.replace('deep_elev_candidate.json', 'deep_elev_report.txt')).read().strip().endswith(cand)
        # the stage directory is never overwritten; --resume skips a current record; --fresh supersedes
        rc, txt = cli('calibrate', 'bright-parent', 'derive', '--tier', 'deep', '--stages', 'build', '--curves', f'{p3},{p2}',
                      '--version', 't1', '--out', out)
        assert rc == 2 and 'never overwritten' in txt, txt
        rc, txt = cli('calibrate', 'bright-parent', 'derive', '--tier', 'deep', '--stages', 'build', '--curves', f'{p3},{p2}',
                      '--version', 't1', '--out', out, '--resume')
        assert rc == 0 and 'skipped-valid 1' in txt, txt
        # install before a diff: G22
        rc, txt = cli('calibrate', 'bright-parent', 'install', '--tier', 'deep', '--version', 't1', '--install-ruling', RULING,
                      '--out', out)
        assert rc == 2 and 'diff' in txt, txt
        # table-only diff without a waiver is refused; with one it is done
        rc, txt = cli('calibrate', 'bright-parent', 'diff', '--tier', 'deep', '--version', 't1', '--table-only', '--out', out)
        assert rc == 2 and rec(out, 'bright-parent.deep.diff.t1')['status'] == 'refused', txt
        rc, txt = cli('calibrate', 'bright-parent', 'diff', '--tier', 'deep', '--version', 't1', '--table-only', '--fresh',
                      '--waive', 'calibrate.diff.served', '--ruling', 'test: no served replay', '--out', out)
        assert rc == 0, txt
        ds = K.read_json(rec(out, 'bright-parent.deep.diff.t1')['outputs'][0]['path'])
        assert ds['tables'] == {'added': 4, 'changed': 4}, ds['tables']
        # install: needs a ruling that is a ledger row
        for extra, why in (([], 'install-ruling'), (['--install-ruling', 'NOT-A-LEDGER-ROW-XYZ'], 'fix_id')):
            rc, txt = cli('calibrate', 'bright-parent', 'install', '--tier', 'deep', '--version', 't1', '--out', out, *extra)
            assert rc == 2 and why in txt, txt
        rc, txt = cli('calibrate', 'bright-parent', 'install', '--tier', 'deep', '--version', 't1', '--install-ruling', RULING,
                      '--install-words', 'Yes on 1 and 2', '--out', out)
        assert rc == 0, txt
        r = rec(out, 'bright-parent.deep.install.t1')
        assert r['status'] == 'done' and not failed(r), failed(r)
        man = K.read_json(r['outputs'][0]['path'])
        mir = K.read_json(man['mirror']['path'])
        base = K.read_json(inst)
        assert len(mir['tables']) == len(base['tables']) + 4 and mir['tables']['10min_I_18MHz'] == base['tables']['10min_I_18MHz']
        assert len(mir['meta_deep']) == 2 and K.sha256(man['versioned']['path']) == man['mirror']['sha256']
        assert 'tables.BRIGHT_PARENT_DEEP' in man['patch_changes'] and K.sha256(inst) == sha        # production-style file untouched
        assert RULING in open(man['ledger_draft']).read()
        # negatives: a corrupted mirror fails mirror_equals; a corrupted candidate fails monotone / equals_derivation
        from exopipe.phase3.core import validate
        m2 = dict(mir)
        m2['tables'] = dict(mir['tables'], **{'10min_I_18MHz': {'table': [[1, 2, 9.0]]}})
        wjson(man['mirror']['path'], m2)
        res = {x['name']: x['passed'] for x in validate.run('install_bundle', r['outputs'][0]['path'])}
        assert not res['install_bundle.files_sha'] and not res['install_bundle.mirror_equals'], res
        c = K.read_json(cand)
        k = 'deep_I_18MHz_el25-35'
        c['tables'][k]['table'] = [[35, 60, 4.0], [200, 1e6, 3.0]]
        bad = wjson(os.path.join(d, 'bad_candidate.json'), c)
        res = {x['name']: x['passed'] for x in validate.run('bright_parent_table', bad)}
        assert not res['bright_parent_table.monotone_guard'] and not res['bright_parent_table.equals_derivation'], res
        rc, txt = cli('calibrate', 'bright-parent', 'verify', '--tier', 'deep', '--candidate', bad, '--out', out)
        assert rc == 2, txt
        rc, txt = cli('calibrate', 'bright-parent', 'verify', '--tier', 'deep', '--candidate', cand, '--out', out)
        assert rc == 0, txt
        # a changed installed file refuses (pin, G19)
        with patched(CFG.tables, BRIGHT_PARENT_DEEP_SHA256='0' * 64):
            rc, txt = cli('calibrate', 'bright-parent', 'diff', '--tier', 'deep', '--version', 't1', '--table-only', '--fresh',
                          '--out', out)
            assert rc == 2 and 'G19' in txt, txt


class FakeG:
    """Gate inputs of one synthetic cell: one bright parent at (RA0, DEC0), frame beam 0.5 deg."""
    RA0, DEC0, BEAM = 67.5, 37.5, 0.5

    def parents(self, lst, band):
        import numpy as np
        return np.array([[self.RA0, self.DEC0, 2.0]])

    def deep_target_beam(self, lst, band, night, stokes):
        return self.BEAM

    def target_beam(self, lst, band, key, stokes):
        return self.BEAM

    def phase_centre(self, lst, band):
        return (self.RA0, self.DEC0)


def test_served_replay():
    import numpy as np
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.calibrate import bright_parent as BP
    from exopipe.phase3.lib.calibrate import common as K
    from exopipe.phase3.lib.search import parent_curve_table as P
    d = _fresh('served')
    # a position with no ATNF pulsar near it (the pulsar exemption would change the verdicts)
    pt, _ = P._psr_targets()
    for dec in (37.5, 41.5, 45.5, 49.5, 33.5):
        if pt is None or not pt.query_ball_point(P._uv(np.array([FakeG.RA0]), np.array([dec]))[0], 2 * np.sin(np.radians(3.0) / 2)):
            FakeG.DEC0 = dec
            break
    base = {'meta': {}, 'tables': {'deep_V_18MHz': dict(stokes='V', band='18MHz', ratio='2.0', metric='image-pixel',
                                                        table=[[100, 1e6, 1.0]])}, 'meta_deep': {}}
    inst = wjson(os.path.join(d, 'installed.json'), base)
    cand = wjson(os.path.join(d, 'cand.json'), {'meta': {}, 'tables': {'deep_V_18MHz_el50-90': dict(
        stokes='V', band='18MHz', ratio='2.0', metric='image-pixel', table=[[100, 1e6, 3.0]], elevation_band=[50, 90])}})
    run = os.path.join(d, 'Run_X')
    rows = []
    for i, off in enumerate((0.25, 1.6, 1.6, 4.0)):        # beam widths: 0.25 (both remove), 1.6 (only the candidate), 4 (neither)
        rows.append(dict(cand_id=f'C{i}', lst='04h', date='2030-01-0%d' % (1 + (i == 2)), band='18MHz', stokes='V',
                         ra=f'{FakeG.RA0:.5f}', dec=f'{FakeG.DEC0 + off * FakeG.BEAM:.5f}', peak_jy='0.05', global_snr='5.0',
                         snr='5.0'))
    wcsv(os.path.join(run, 'master_catalog_asrun.csv'), list(rows[0]), rows)
    wcsv(os.path.join(run, 'source_catalogs_V', '18MHz_sources_survivors.csv'), ['source_id', 'identity_id'],
         [dict(source_id='S1', identity_id='J0430+40')])
    wcsv(os.path.join(run, 'source_catalogs_V', '18MHz_source_members.csv'), ['source_id', 'cand_id'],
         [dict(source_id='S1', cand_id='C1')])
    os.makedirs(os.path.join(run, 'judgements_V'))
    open(os.path.join(run, 'judgements_V', 'judgements.jsonl'), 'w').write(
        json.dumps({'action': 'classify', 'source_id': 'J0430+40', 'judge_confidence': 'probably-real'}) + '\n')
    out = os.path.join(d, 'out')
    parents = _fresh('served/parents')
    with patched(CFG.tables, BRIGHT_PARENT_DEEP=inst, BRIGHT_PARENT_DEEP_SHA256=K.sha256(inst)), \
            patched(CFG.calibrate, DIFF_LEDGER_LABEL={'deep': None, 'sublst': None}), \
            patched(BP, _gate_inputs=lambda ctx, parents, lst=None: (FakeG(), {'fake': True})):
        rc, txt = cli('calibrate', 'bright-parent', 'diff', '--tier', 'deep', '--candidate', cand, '--served-run', run,
                      '--parents', parents, '--stokes', 'V', '--out', out)
        assert rc == 0, txt
    r = rec(out, f'bright-parent.deep.diff.cand-{K.sha256(cand)[:12]}')
    assert r['status'] == 'done' and not failed(r), r['reason']
    s = K.read_json(r['outputs'][0]['path'])['served']
    assert s['n_rows'] == 4 and s['removed_installed'] == 1 and s['removed_candidate'] == 3 and s['changed'] == 2, s
    ch = {x['cand_id']: x for x in K.read_csv(s['rows_csv'])}
    assert set(ch) == {'C1', 'C2'} and ch['C1']['change'] == 'newly removed' and ch['C1']['served_identity'] == 'J0430+40'
    assert ch['C1']['judged'] == 'probably-real' and '[el50-90]' in ch['C1']['band_note'] and ch['C1']['elev_band'] == '50-90'
    assert s['served_identities_losing'] == {'V:J0430+40': 'probably-real'} and s['tally'] == {'V|newly removed|50-90': 2}
    # elevation mode (the 2026-10-06 replay): the installed side without elevations
    with patched(CFG.tables, BRIGHT_PARENT_DEEP=inst, BRIGHT_PARENT_DEEP_SHA256=K.sha256(inst)), \
            patched(CFG.calibrate, DIFF_LEDGER_LABEL={'deep': None, 'sublst': None}), \
            patched(BP, _gate_inputs=lambda ctx, parents, lst=None: (FakeG(), {'fake': True})):
        rc, txt = cli('calibrate', 'bright-parent', 'diff', '--tier', 'deep', '--candidate', cand, '--served-run', run,
                      '--parents', parents, '--stokes', 'V', '--diff-mode', 'elevation', '--fresh', '--out', out)
        assert rc == 0, txt


def test_orchestration():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.calibrate import common as K
    d = _fresh('orch')
    state = os.path.join(d, 'state')
    os.makedirs(state)
    root = os.path.join(d, 'collect')
    # one attempt per Stokes; 21h 2025-08-06 is excluded in data_decisions.csv (all subbands; Gregg 2026-10-13)
    for S in ('V', 'I'):
        open(os.path.join(state, f'published_{S}.csv'), 'w').write(f'21h_46MHz,nodeX,a{S}\n')
        led = os.path.join(root, 'p3s5_collect_nodeX', 'attempts', f'21h_46MHz_{S}_a{S}', 'transients', 'Run_P3S5', 'v_deep_ledger.csv')
        wcsv(led, ['lst', 'scope', 'stokes', 'date', 'ra', 'dec', 'snr'],
             [dict(lst='21h', scope='46MHz', stokes=S, date='2025-08-06', ra='1', dec='2', snr='6.0'),
              dict(lst='21h', scope='46MHz', stokes=S, date='2025-08-06', ra='1', dec='2', snr='4.0'),     # below the floor
              dict(lst='21h', scope='46MHz', stokes=S, date='2025-08-05', ra='1', dec='2', snr='6.0')])
    parents = _fresh('orch/parents')
    calls = []

    def fake_run(u, target, args, env=None, log=None, cwd=None, module=False, timeout=None):
        calls.append((os.path.basename(str(target)), list(args), dict(env or {})))
        outp = args[0]
        ratio = float(env['P3S3_CURVE_RATIO'])
        mult = 'harvest' if '--multiplicity' in args else None
        st = args[args.index('--stokes') + 1:]
        st = [x for x in st if x in ('I', 'V')][:2] if '--multiplicity' not in args else [args[args.index('--stokes') + 1]]
        cv = {}
        for S in st:
            for b in ('18MHz', '41MHz'):
                k = f'deep_{S}-multi_{b}' if mult else f'deep_{S}_{b}'
                cv[k] = curve(BINS_A)
                cv[k + '_el50-90'] = curve(BINS_B)
        wjson(outp, derivation(cv, ratio, eb=env['P3_CURVE_ELEV_BANDS'], mult=mult))
        return type('R', (), {'tail': '', 'ok': True})()

    excl = {S: wcsv(os.path.join(d, f'excl_{S}.txt'), ['cell'], []) for S in ('I', 'V')}
    out = os.path.join(d, 'out')
    inst = synthetic_installed(d)
    with patched(CFG.calibrate_bright_parent, COLLECT_ROOTS=(root,)), patched(K, run_legacy=fake_run), \
            patched(CFG.calibrate_multiples, EXCLUDE_CELLS=excl), \
            patched(CFG.tables, BRIGHT_PARENT_DEEP=inst, BRIGHT_PARENT_DEEP_SHA256=K.sha256(inst)):
        rc, txt = cli('calibrate', 'bright-parent', 'derive', '--tier', 'deep', '--harvest-state', state, '--parents', parents,
                      '--version', 'o1', '--out', out)
        assert rc == 0, txt
        r = rec(out, 'bright-parent.deep.derive-inputs.o1')
        man = K.read_json(r['outputs'][0]['path'])
        assert man['excluded_rows']['n'] == 2 and man['excluded_rows']['by_cell'] == {'21h_46MHz': 2}, man['excluded_rows']
        assert man['parents_dir'] == parents and os.path.dirname(next(iter(man['files']))) == man['state_dir']
        assert [c[0] for c in calls] == ['p3s3_curve_derive.py'] * 2
        e3 = [c for c in calls if c[1][0].endswith('deep_elev_r3.json')][0]
        assert e3[2]['P3S3_CURVE_RATIO'] == '3' and e3[2]['P3_PARENT_DIR'] == parents and e3[2]['P3_GATE_METRIC'] == 'image-pixel'
        assert e3[2]['P3S3_STATE_DIR'] == man['state_dir'] and e3[2]['P3_CURVE_ELEV_BANDS'] == '20-25,25-35,35-50,50-90'
        assert e3[1][1:] == ['--by', 'subband', '--floor', '5', '--stokes', 'V', 'I']
        e2 = [c for c in calls if c[1][0].endswith('deep_elev_r2V.json')][0]
        assert e2[2]['P3S3_CURVE_RATIO'] == '2' and e2[1][-2:] == ['--stokes', 'V']
        rb = rec(out, 'bright-parent.deep.derive-build.o1')
        assert rb['status'] == 'done' and not failed(rb), rb['reason']
        with patched(CFG.calibrate_bright_parent, EXCLUDED_ROWS='refuse'):
            rc, txt = cli('calibrate', 'bright-parent', 'derive', '--tier', 'deep', '--stages', 'inputs', '--harvest-state', state,
                          '--parents', parents, '--version', 'o2', '--out', out)
            assert rc == 2 and 'G23' in txt, txt
        calls.clear()
        rc, txt = cli('calibrate', 'multiples', 'derive', '--tier', 'deep', '--harvest-state', state, '--parents', parents,
                      '--version', 'm1', '--out', out)
        assert rc == 0, txt
        names = sorted(os.path.basename(c[1][0]) for c in calls)
        assert names == ['multi_r2V.json', 'multi_r3_I.json', 'multi_r3_V.json'], names
        mi = [c for c in calls if c[1][0].endswith('multi_r3_I.json')][0]
        assert mi[2]['P3_DERIVE_EXCLUDE_CELLS'].endswith('exclude_I.txt') and '--multiplicity' in mi[1] and 'multi' in mi[1]
        rb = rec(out, 'multiples.deep.derive-build.m1')
        assert rb['status'] == 'done' and not failed(rb), rb['reason']
        cm = K.read_json(rb['outputs'][0]['path'])['tables']
        assert cm['deep_I-multi_18MHz']['excluded_cells'].endswith('exclude_I.txt')
        # a subset derivation (--cells) is recorded and never installed (G24)
        rc, txt = cli('calibrate', 'bright-parent', 'derive', '--tier', 'deep', '--harvest-state', state, '--parents', parents,
                      '--cells', '21h_46MHz', '--version', 's1', '--out', out)
        assert rc == 0, txt
        rc, txt = cli('calibrate', 'bright-parent', 'diff', '--tier', 'deep', '--version', 's1', '--table-only', '--waive',
                      'calibrate.diff.served', '--ruling', 'test', '--out', out)
        assert rc == 0, txt
        rc, txt = cli('calibrate', 'bright-parent', 'install', '--tier', 'deep', '--version', 's1', '--install-ruling', RULING,
                      '--out', out)
        assert rc == 2 and 'G24' in txt, txt


def test_sublst():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.calibrate import common as K
    d = _fresh('sublst')
    sub = derivation({'sublst_V_18MHz': curve(BINS_A), 'sublst_V_18MHz_el50-90': curve(BINS_B),
                      'sublst_I_Red': curve(BINS_A)}, 2.0, eb='25-35,35-50,50-90', cadence='10min')
    sp = wjson(os.path.join(d, 'sublst_tables_20300101T0000.json'), sub)
    inst = wjson(os.path.join(d, 'installed_sublst.json'), {'meta': {'x': 1}, 'tables': {'sublst_I_18MHz': {'table': []}}})
    out = os.path.join(d, 'out')
    with synthetic_tables(d, BRIGHT_PARENT_SUBLST=inst, BRIGHT_PARENT_SUBLST_SHA256=K.sha256(inst)):
        rc, txt = cli('calibrate', 'bright-parent', 'derive', '--tier', 'sublst', '--stages', 'build', '--curves', sp,
                      '--version', 'u1', '--out', out)
        assert rc == 0, txt
        r = rec(out, 'bright-parent.sublst.derive-build.u1')
        assert not failed(r), failed(r)
        rc, txt = cli('calibrate', 'bright-parent', 'diff', '--tier', 'sublst', '--version', 'u1', '--table-only', '--waive',
                      'calibrate.diff.served', '--ruling', 'test', '--out', out)
        assert rc == 0, txt
        rc, txt = cli('calibrate', 'bright-parent', 'install', '--tier', 'sublst', '--version', 'u1', '--install-ruling',
                      'R-SUBLST-TABLES-GO-ELEVATION-BANDS', '--out', out)
        assert rc == 0, txt
        man = K.read_json(rec(out, 'bright-parent.sublst.install.u1')['outputs'][0]['path'])
        mir = K.read_json(man['mirror']['path'])
        assert len(mir['tables']) == 39 and set(mir['meta']) == {'installed_utc', 'source', 'derivation', 'rule'}
        assert man['mirror']['path'].endswith(inst.lstrip('/'))
        # sub-LST derive needs exactly one LST
        rc, txt = cli('calibrate', 'bright-parent', 'derive', '--tier', 'sublst', '--stages', 'inputs', '--version', 'u2',
                      '--out', out)
        assert rc == 2 and '--lsts' in txt, txt


def _scale_inputs(d, missing_lst=None, n=30):
    import numpy as np
    rng = np.random.default_rng(3)
    old = [dict(lst=f'{h:02d}h', scale=f'{1 + 0.01 * h:.4f}', err='0.02', nsrc='700', median_ratio='1.0', window='dec +20..+60 deg',
                ref='synthetic', measured_at='2030-01-01T00:00:00Z') for h in range(24)]
    po = wcsv(os.path.join(d, 'old_scale.csv'), list(old[0]), old)
    rows = []
    for h in range(24):
        if f'{h:02d}h' == missing_lst:
            continue
        for i in range(n):
            rows.append(dict(lst=f'{h:02d}h', ra='10.0', dec=f'{rng.uniform(15, 65):.3f}', s_lwa_jy='5.0', s_vlssr_jy='5.0',
                             ratio=f'{rng.normal(1.02, 0.05):.4f}', pb='0.5', sep_arcmin='1.0'))
    pr = wcsv(os.path.join(d, 'ratios_73MHz.csv'), list(rows[0]), rows)
    return pr, po


def test_scale():
    from exopipe.phase3.lib.calibrate import common as K
    from exopipe.phase3.core import validate
    d = _fresh('scale')
    pr, po = _scale_inputs(d)
    out = os.path.join(d, 'out')
    rc, txt = cli('calibrate', 'scale', 'derive', '--stages', 'table', '--ratios', pr, '--old-scale-table', po, '--version',
                  'v9-20300101', '--out', out, '--parity')
    assert rc == 0, txt
    r = rec(out, 'scale.deep.derive-table.v9-20300101')
    assert r['status'] == 'done' and not failed(r), r['reason']
    t = r['outputs'][0]['path']
    assert os.path.basename(t) == 'vlssr_scale_table_v9_20300101.csv'
    assert all('re-measured 2030-01-01' in x['ref'] for x in K.read_csv(t))           # amendment A3
    assert r['meta']['legacy_calls'][0]['argv'][1].endswith('scale_table_v4.py')
    rows = K.read_csv(t)
    rows[3]['scale'] = '9.99999'
    bad = wcsv(os.path.join(d, 'bad', os.path.basename(t)), list(rows[0]), rows)
    shutil.copy(t + '.provenance.json', bad + '.provenance.json')
    assert not {x['name']: x['passed'] for x in validate.run('scale_table', bad)}['scale_table.recompute']
    d2 = _fresh('scale2')
    pr2, po2 = _scale_inputs(d2, missing_lst='05h')
    rc, txt = cli('calibrate', 'scale', 'derive', '--stages', 'table', '--ratios', pr2, '--old-scale-table', po2, '--version',
                  'v9-20300101', '--out', os.path.join(d2, 'out'))
    assert rc == 2 and 'REFUSED' in txt, txt                                            # G11


def _fc_inputs(d, ok=True):
    import numpy as np
    from exopipe.phase3.config import CFG
    sb = CFG.calibrate_fluxcorr.SUBBANDS
    rng = np.random.default_rng(5)
    vec = wcsv(os.path.join(d, 'band_vector.csv'), ['subband', 'factor'],
               [dict(subband=b, factor=('1.000000' if b == '73MHz' else f'{rng.uniform(0.95, 1.05):.6f}')) for b in sb])
    sc = [dict(lst=f'{h:02d}h', scale='1.0', err='0.01', nsrc='700', median_ratio='1.0', scale_old='1.0',
               rescale=f'{rng.uniform(0.97, 1.03):.5f}', window='w', ref='r', measured_at='2030-01-02T00:00:00Z') for h in range(24)]
    scale = wcsv(os.path.join(d, 'scale.csv'), list(sc[0]), sc)
    cols = ['lst', 'scope', 'status', 'scale', 'err', 'nsrc', 'srcs', 'freq_hz', 'freq_src', 'rms_jy', 'n_written', 'detail',
            'measured_at']
    for k in range(2):
        rows = [dict(lst=f'{h:02d}h', scope=sb[(h + k) % 15], status=('OK' if ok and h % 3 else 'REFUSED no calibrator'),
                     scale=f'{rng.normal(1.03, 0.02):.5f}', err=f'{rng.uniform(0.005, 0.02):.5f}', nsrc='1', srcs='3C48',
                     freq_hz='1', freq_src='x', rms_jy='0.1', n_written='0', detail='', measured_at='t') for h in range(24)]
        wcsv(os.path.join(d, 'stageA', f'shflux_measure_{k}.csv'), cols, rows)
    return vec, scale, os.path.join(d, 'stageA', 'shflux_measure_*.csv')


def test_fluxcorr():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.calibrate import common as K
    from exopipe.phase3.lib.calibrate import fluxscale as FS
    d = _fresh('fluxcorr')
    vec, scale, glob_ = _fc_inputs(d)
    out = os.path.join(d, 'out')
    rc, txt = cli('calibrate', 'fluxcorr', 'derive', '--stages', 'table', '--stagea', glob_, '--band-vector', vec,
                  '--scale-candidate', scale, '--version', 'v9-20300101', '--out', out)
    assert rc == 0, txt
    r = rec(out, 'fluxcorr.deep.derive-table.v9-20300101')
    assert r['status'] == 'done' and not failed(r), r['reason']
    rows = K.read_csv(r['outputs'][0]['path'])
    assert rows[0]['version'].startswith('fluxcorr-v9-20300101 (') and 'on the 2030-01-02 per-LST VLSSr scale' in rows[0]['version']
    assert open(r['outputs'][0]['path'], newline='').read().count('\r\n') == 20            # build_table line ends
    d2 = _fresh('fluxcorr2')
    vec2, scale2, glob2 = _fc_inputs(d2, ok=False)
    rc, txt = cli('calibrate', 'fluxcorr', 'derive', '--stages', 'table', '--stagea', glob2, '--band-vector', vec2,
                  '--scale-candidate', scale2, '--version', 'v9-20300101', '--out', os.path.join(d2, 'out'))
    assert rc == 2 and 'G12' in txt, txt
    # band-vector iteration with the in-place VLSSr check faked: medians shrink by 10x per step
    sb = CFG.calibrate_fluxcorr.SUBBANDS
    state = {'n': 0}

    def fake_run(u, target, args, env=None, log=None, cwd=None, module=False, timeout=None):
        state['n'] += 1
        m = 0.02 / (10 ** (state['n'] - 1)) if state.get('fast', True) else 0.02 / (1.2 ** state['n'])
        wcsv(os.path.join(env['VLSSR_OUT'], 'smooth_by_subband_powerlaw.csv'),
             ['subband', 'freq_mhz', 'n', 'raw_median_pct', 'raw_scatter_pct', 'sh_median_pct', 'sh_scatter_pct'],
             [dict(subband=b, freq_mhz='1', n='10', raw_median_pct=f'{100 * m * (1 if i % 2 else -1):.2f}', raw_scatter_pct='1',
                   sh_median_pct='', sh_scatter_pct='') for i, b in enumerate(sb)])
        return type('R', (), {'tail': '', 'ok': True})()
    with patched(K, run_legacy=fake_run):
        rc, txt = cli('calibrate', 'fluxcorr', 'derive', '--stages', 'bandvec', '--band-vector-start', vec, '--version', 'b1',
                      '--out', out)
        assert rc == 0, txt
        r = rec(out, 'fluxcorr.deep.derive-bandvec.b1')
        assert r['status'] == 'done' and not failed(r) and state['n'] == 2, (r['reason'], state)   # worst 2% then 0.2% < 0.3% (:27)
        state.update(n=0, fast=False)
        rc, txt = cli('calibrate', 'fluxcorr', 'derive', '--stages', 'bandvec', '--band-vector-start', vec, '--version', 'b2',
                      '--out', out)
        assert rc == 2 and 'G13' in txt and state['n'] == CFG.calibrate_fluxcorr.BANDVEC_MAX_STEPS, txt
    worst, m = FS._loo_arith(vec, os.path.join(out, 'calibrate', 'fluxcorr', 'deep', 'b1', 'bandvec', 'loo1',
                                               'smooth_by_subband_powerlaw.csv'), os.path.join(d, 'nv.csv'), '73MHz')
    assert abs(worst - 0.02) < 1e-12 and float(K.read_csv(os.path.join(d, 'nv.csv'))[12]['factor']) == 1.0
    # stage A needs an explicit cell list
    rc, txt = cli('calibrate', 'fluxcorr', 'derive', '--stages', 'stageA', '--version', 'a1', '--out', out)
    assert rc == 2 and '--cells' in txt, txt


def _trials(d, k_true=(0.5, 0.45), stokes=('I', 'V'), n=200):
    import numpy as np
    rng = np.random.default_rng(11)
    cols = ['cell', 'stokes', 'frame', 'snr_true', 'sign', 'x_true', 'y_true', 'x_tgt', 'y_tgt', 'start_off_beams', 'bmaj_deg',
            'bmin_deg', 'bpa_deg', 'urms', 'lrms', 'tf_status', 'tf_snr', 'tf_x', 'tf_y', 'err_maj_deg', 'err_min_deg', 'pos_kind',
            'pos_reason']
    rows = []
    for S in stokes:
        for i in range(n):
            s = float(rng.choice([5.0, 5.5, 6.0, 6.5]))
            rows.append(dict(cell='04h_18MHz', stokes=S, frame=f'f{i % 3}', snr_true=s, sign=1, bmaj_deg=0.3, bmin_deg=0.2,
                             tf_status='ok' if i % 50 else 'at-bound', tf_snr=s, err_maj_deg=rng.normal(0, k_true[0] * 0.3 / s),
                             err_min_deg=rng.normal(0, k_true[1] * 0.2 / s)))
    return wcsv(os.path.join(d, 'trials', 'trials_0.csv'), cols, [{c: r.get(c, '') for c in cols} for r in rows])


def test_template_k():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.calibrate import common as K
    d = _fresh('k')
    _trials(d)
    out = os.path.join(d, 'out')
    rc, txt = cli('calibrate', 'template-k', 'derive', '--tier', 'deep', '--stages', 'summary', '--trials',
                  os.path.join(d, 'trials'), '--version', 'k1', '--out', out)
    assert rc == 0, txt
    r = rec(out, 'template-k.deep.derive-summary.k1')
    assert r['status'] == 'done' and not failed(r), r['reason']
    kr = K.read_json(r['outputs'][0]['path'])
    assert kr['orientation'] == 'wcs' and kr['K']['I']['n_ok'] == 196 and 0.3 < kr['K']['I']['major'] < 0.8
    rc, txt = cli('calibrate', 'template-k', 'diff', '--tier', 'deep', '--version', 'k1', '--out', out)
    assert rc == 0, txt
    rc, txt = cli('calibrate', 'template-k', 'install', '--tier', 'deep', '--version', 'k1', '--out', out)
    assert rc == 2 and 'install-ruling' in txt, txt                                     # G18
    rc, txt = cli('calibrate', 'template-k', 'install', '--tier', 'deep', '--version', 'k1', '--install-ruling',
                  'RULINGS-GREGG-20261013B-MIRROR-EXCLUSIONS-ALL-APPROVED', '--out', out)
    assert rc == 0, txt
    man = K.read_json(rec(out, 'template-k.deep.install.k1')['outputs'][0]['path'])
    km, kn = man['k_new']['I']
    assert f'{km:.3f}, {kn:.3f}' in open(man['patch']).read() and man['patch_changes'] == ['fitting.K_TEMPLATE']
    for edit, gate in (({'orientation': 'legacy-mirrored'}, 'G17'), ({'K': dict(kr['K'], V=dict(kr['K']['V'], major=7.0))}, 'G16'),
                       ({'K': {'I': kr['K']['I']}}, 'G16')):
        bad = wjson(os.path.join(d, f'bad_{gate}_{len(edit)}_{time.time()}.json'), dict(kr, **edit))
        rc, txt = cli('calibrate', 'template-k', 'diff', '--tier', 'deep', '--candidate', bad, '--out', out)
        assert rc == 0, txt
        rc, txt = cli('calibrate', 'template-k', 'install', '--tier', 'deep', '--candidate', bad, '--install-ruling',
                      'RULINGS-GREGG-20261013B-MIRROR-EXCLUSIONS-ALL-APPROVED', '--out', out)
        assert rc == 2 and gate in txt, (gate, txt)
    rc, txt = cli('calibrate', 'template-k', 'derive', '--tier', 'snapshot', '--out', out)
    assert rc == 2, txt
    rc, txt = cli('calibrate', 'snapshot-curve', 'verify', '--out', out)
    assert rc == 2 and 'not implemented' in txt, txt


# ---------------------------------------------------------------------------------------------------------------------
# real products (READ ONLY; outputs only under the scratch root)
# ---------------------------------------------------------------------------------------------------------------------

def test_real():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.calibrate import common as K
    if socket.gethostname().split('.')[0] != 'lwacalim07' or not os.path.exists(CFG.tables.BRIGHT_PARENT_DEEP_DERIVATIONS['multiples'][0]):
        print('  real: SKIPPED (the derivations are lwacalim07 node-local)')
        return
    out = _fresh('real')
    rc, txt = cli('calibrate', 'all', 'verify', '--out', out)
    assert rc == 0, txt
    units = [u for u in os.listdir(os.path.join(out, '_records', 'calibrate')) if u.endswith('.json')]
    assert len(units) == 7, units
    for u in units:
        r = json.load(open(os.path.join(out, '_records', 'calibrate', u)))
        assert r['status'] == 'done' and not failed(r), (u, r['reason'])
    eq = [t for t in rec(out, 'bright-parent.deep.verify-installed.installed')['tests'] if t['name'].endswith('equals_derivation')][0]
    assert eq['value'] == 241, eq                                                         # 38 + 114 + 89 keys recomputed
    inst = K.read_json(CFG.tables.BRIGHT_PARENT_DEEP)['tables']
    dv = CFG.tables.BRIGHT_PARENT_DEEP_DERIVATIONS
    rc, txt = cli('calibrate', 'bright-parent', 'derive', '--tier', 'deep', '--stages', 'build', '--curves',
                  ','.join(dv['elevation-bands']), '--version', 'p1', '--out', out)
    assert rc == 0, txt
    c = K.read_json(rec(out, 'bright-parent.deep.derive-build.p1')['outputs'][0]['path'])['tables']
    assert len(c) == 114 and all(K.canon(v) == inst[k] for k, v in c.items())            # every record = the installed one
    rc, txt = cli('calibrate', 'multiples', 'derive', '--tier', 'deep', '--stages', 'build', '--curves', ','.join(dv['multiples']),
                  '--version', 'p1', '--out', out)
    assert rc == 0, txt
    c = K.read_json(rec(out, 'multiples.deep.derive-build.p1')['outputs'][0]['path'])['tables']
    assert set(c) == {k for k in inst if '-multi_' in k} and all(K.canon(v) == inst[k] for k, v in c.items())
    T = CFG.tables
    rc, txt = cli('calibrate', 'fluxcorr', 'derive', '--stages', 'table', '--stagea', T.FLUXCORR_INPUTS['stagea_glob'],
                  '--band-vector', T.FLUXCORR_INPUTS['band_vector'], '--version', 'v2-20261001', '--out', out)
    assert rc == 0, txt
    f = rec(out, 'fluxcorr.deep.derive-table.v2-20261001')['outputs'][0]['path']
    assert K.sha256(f) == T.FLUXCORR_TABLE_SHA256 and K.sha256(f.replace('.csv', '_shcells.csv')) == T.FLUXCORR_SHCELLS_SHA256
    rc, txt = cli('calibrate', 'scale', 'derive', '--stages', 'table', '--ratios', T.SCALE_TABLE_INPUTS['ratios'],
                  '--old-scale-table', T.SCALE_TABLE_INPUTS['previous'], '--version', 'v4-20261001', '--out', out)
    assert rc == 0, txt
    s = rec(out, 'scale.deep.derive-table.v4-20261001')['outputs'][0]['path']
    strip = lambda p: [ln.rsplit(',', 1)[0] for ln in open(p).read().splitlines()]       # every column but measured_at
    assert strip(s) == strip(T.SCALE_TABLE)
    rc, txt = cli('calibrate', 'template-k', 'derive', '--tier', 'deep', '--stages', 'summary', '--version', 'p1', '--out', out)
    assert rc == 0, txt
    kr = K.read_json(rec(out, 'template-k.deep.derive-summary.p1')['outputs'][0]['path'])
    assert all((round(kr['K'][S]['major'], 3), round(kr['K'][S]['minor'], 3)) == CFG.fitting.K_TEMPLATE[('deep', S)] for S in 'IV')
    rc, txt = cli('calibrate', 'bright-parent', 'derive', '--tier', 'deep', '--dry-run', '--out', out)
    assert rc == 0 and 'p3s3_curve_derive.py' in txt and '"P3S3_CURVE_RATIO": "2"' in txt, txt


TESTS = ['test_config', 'test_lock', 'test_key_loops', 'test_merge', 'test_config_patch', 'test_deep_e2e', 'test_served_replay',
         'test_orchestration', 'test_sublst', 'test_scale', 'test_fluxcorr', 'test_template_k', 'test_real']


def main(names):
    t0 = time.time()
    shutil.rmtree(SCRATCH, ignore_errors=True)
    os.makedirs(SCRATCH)
    n_fail = 0
    try:
        for name in names or TESTS:
            t = time.time()
            try:
                globals()[name]()
                print(f'PASS {name} ({time.time() - t:.1f} s)')
            except Exception:                     # noqa: BLE001 - report every test
                n_fail += 1
                import traceback
                print(f'FAIL {name}\n{traceback.format_exc()}')
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    print(f'{len(names or TESTS) - n_fail} of {len(names or TESTS)} passed in {time.time() - t0:.1f} s')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
