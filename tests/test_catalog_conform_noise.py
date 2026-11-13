"""The table-noise runner of the conformance (lib/catalog/wraps/conform_noise.py): the ONE statistic of the population of
catalog rows in exopipe_ops p3s2_conform.py - per (LST, band, Stokes), the median of |peak flux / global S/N| - recorded by
the run that computes it and read frozen by a per-hour run (Gregg 2026-11-12, R-GREGG-20261112-PER-HOUR-STEPS-SIX-RULINGS
(2): "Approved." - frozen statistics from earlier nights everywhere).

Checks, data-free (a stand-in module that holds the patched line and writes what its table step would use):
  passthrough   the stand-in's output equals the unpatched stand-in's, byte for byte; no record
  record        the same output, and the record: one row per key, n = finite values, the median reads back to the float
                the module used; a module that exits early writes no record and leaves the earlier one
  frozen        a run over ONE night's rows uses the recorded medians (not its own); a key the run has no row of answers
                as LST-wide; refusals: no record, a record of another module version, a module outside the accepted
                versions, no accepted versions named, a key of the run's rows that the record lacks, a malformed or
                duplicated record
  patch         refuses a module without the line, with the line twice, or with another use of the name
  wiring        the 10-minute conform stage runs the runner in record mode under catalog_conform.TABLE_NOISE_RECORD
                ['sublst'] and registers the record; the config names the ruling and the checked module version
Real module (read only; skipped where exopipe_ops is not mounted): the patch applies to the live p3s2_conform.py, the
patched source compiles, and the module's path-normalized sha1 is the one config accepts for the frozen mode.
Run: python tests/test_catalog_conform_noise.py
"""
import csv
import math
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from exopipe.phase3.lib.catalog.wraps import conform_noise as CN     # noqa: E402

T = f'/tmp/exopipe3_tests_catalog_conform_noise_{os.getpid()}'
RUNNER = os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'catalog', 'wraps', 'conform_noise.py')
LIVE = '/lustre/gh/main/exopipe_ops/p3s2_conform.py'
N, FAILS = [0], []

# a stand-in for the module: the lines around the patched one as production has them (the initialisation, the per-row
# append, the median, three look-ups), reading rows.csv of its run directory and writing what the table step would use
STANDIN = '''import csv, os, sys
import numpy as np
U = sys.argv[1]
def f(x):
    try: return float(x)
    except Exception: return np.nan
raw = list(csv.DictReader(open(f'{U}/rows.csv')))
if os.environ.get('STANDIN_EXIT_EARLY'):
    sys.exit('conform: stand-in REFUSING before the table step')
_PAR, _BEAM, _RMS = {}, {}, {}
for _r in raw:
    try: _RMS.setdefault((_r['lst'], _r['band'], _r['stokes']), []).append(abs(f(_r['peak_jy'])) / abs(f(_r['global_snr'])))
    except Exception: pass
_RMS = {k: float(np.nanmedian(v)) for k, v in _RMS.items()}
with open(f'{U}/used.csv', 'w') as _o:
    for _r in raw:
        _a = _RMS.get((_r['lst'], _r['band'], _r['stokes']), 0.0)
        _b = _RMS.get((_r['lst'], _r['band'], 'V'), 0.0)
        _c = _RMS.get((_r['lst'], '99MHz', _r['stokes']), 0.0)
        _o.write(f"{_r['lst']},{_r['band']},{_r['stokes']},{_r['date']},{_a!r},{_b!r},{_c!r}\\n")
print('conform: stand-in done', flush=True)
'''

ROWS = [   # lst, band, stokes, date, peak_jy, global_snr
    ('01h', '41MHz', 'I', '2025-01-01T00', '2.0', '10'), ('01h', '41MHz', 'I', '2025-01-02T00', '3.0', '10'),
    ('01h', '41MHz', 'I', '2025-01-03T01', '-8.0', '-20'), ('01h', '41MHz', 'V', '2025-01-01T00', '0.9', '9'),
    ('01h', '41MHz', 'V', '2025-01-02T02', '1.1', '10'), ('01h', '55MHz', 'I', '2025-01-02T00', '5.0', '0'),     # S/N 0: skipped
    ('01h', '55MHz', 'I', '2025-01-03T00', 'x', '7'),                                                         # unreadable: nan
    ('01h', '73MHz', 'I', '2025-01-01T03', '7.0', '10'), ('01h', '73MHz', 'I', '2025-01-09T00', '70.0', '10')]
NIGHT = '2025-01-09'            # the "new night": one row of 73 MHz I, whose own median (7.0) is not the LST-wide one (3.85)


def check(name, ok, detail=''):
    N[0] += 1
    print(f'{"PASS" if ok else "FAIL"}  {name}' + (f'  ({detail})' if detail and not ok else ''))
    if not ok:
        FAILS.append(name)


def rundir(name, rows, src=STANDIN, tree='tree'):
    """A run directory with rows.csv, and the stand-in module at <T>/<tree>/exopipe_ops/p3s2_conform.py."""
    d = os.path.join(T, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'rows.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['lst', 'band', 'stokes', 'date', 'peak_jy', 'global_snr'])
        w.writerows(rows)
    m = os.path.join(T, tree, 'exopipe_ops', 'p3s2_conform.py')
    os.makedirs(os.path.dirname(m), exist_ok=True)
    with open(m, 'w') as f:
        f.write(src)
    return d, m


def child(module, d, mode=None, record=None, accepted=None, extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith('EXOPIPE3_CONFORM_NOISE_')}
    env.pop('STANDIN_EXIT_EARLY', None)
    if mode is None:
        cmd = [sys.executable, module, d]
    else:
        cmd = [sys.executable, RUNNER, module, d]
        env['EXOPIPE3_CONFORM_NOISE_MODE'] = mode
        if record:
            env['EXOPIPE3_CONFORM_NOISE_RECORD'] = record
        if accepted is not None:
            env['EXOPIPE3_CONFORM_NOISE_SHA1'] = ','.join(accepted)
    env.update(extra or {})
    r = subprocess.run(cmd, env=env, cwd=d, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    return r.returncode, r.stdout


def used(d):
    with open(os.path.join(d, 'used.csv')) as f:
        return f.read()


def norm_of(module):
    with open(module) as f:
        return CN.normalized_sha1(module, f.read())


def test_passthrough_and_record():
    d0, m = rundir('plain', ROWS)
    rc0, out0 = child(m, d0)
    check('the stand-in module runs unpatched', rc0 == 0 and 'stand-in done' in out0, out0[-300:])
    d1, _ = rundir('pass', ROWS)
    rc1, out1 = child(m, d1, 'passthrough')
    check('passthrough: rc 0, start and done lines', rc1 == 0 and 'CONFORM-NOISE start' in out1 and 'CONFORM-NOISE done' in out1,
          out1[-300:])
    check('passthrough: the module\'s output equals the unpatched run\'s, byte for byte', used(d1) == used(d0))
    check('passthrough: no record written', not os.path.exists(os.path.join(d1, CN.FILE)))
    d2, _ = rundir('record', ROWS)
    rec = os.path.join(d2, CN.FILE)
    rc2, out2 = child(m, d2, 'record', rec)
    check('record: rc 0 and the done line names the record', rc2 == 0 and f'recorded in {rec}' in out2, out2[-300:])
    check('record: the module\'s output equals the unpatched run\'s, byte for byte', used(d2) == used(d0))
    got, shas, _sha = CN.read_record(rec)
    check('record: one row per (LST, band, Stokes) of the rows', sorted(got) == [
        ('01h', '41MHz', 'I'), ('01h', '41MHz', 'V'), ('01h', '55MHz', 'I'), ('01h', '73MHz', 'I')], sorted(got))
    check('record: n counts the finite values (S/N 0 skipped, an unreadable flux not counted)',
          got[('01h', '41MHz', 'I')][0] == 3 and got[('01h', '55MHz', 'I')][0] == 0 and got[('01h', '73MHz', 'I')][0] == 2, got)
    check('record: the medians are the module\'s (0.3, 0.105, nan, 3.85 Jy)',
          got[('01h', '41MHz', 'I')][1] == 0.3 and abs(got[('01h', '41MHz', 'V')][1] - 0.105) < 1e-12
          and math.isnan(got[('01h', '55MHz', 'I')][1]) and got[('01h', '73MHz', 'I')][1] == 3.85, got)
    line = [ln for ln in used(d2).splitlines() if ln.startswith('01h,73MHz,I,2025-01-09')][0]
    check('record: the recorded median reads back to the float the module used', line.split(',')[4] == repr(got[('01h', '73MHz', 'I')][1]),
          line)
    check('record: the rows carry the module\'s path-normalized sha1', shas == {norm_of(m)}, shas)
    before = open(rec).read()
    rc3, out3 = child(m, d2, 'record', rec, extra={'STANDIN_EXIT_EARLY': '1'})
    check('record: a module that exits early ends non-zero and leaves the earlier record', rc3 != 0 and open(rec).read() == before,
          out3[-200:])
    d4, _ = rundir('record_none', ROWS)
    rc4, out4 = child(m, d4, 'record', os.path.join(d4, CN.FILE), extra={'STANDIN_EXIT_EARLY': '1'})
    check('record: a module that exits early writes no record', rc4 != 0 and not os.path.exists(os.path.join(d4, CN.FILE)), out4[-200:])
    rc5, out5 = child(m, d1, 'record')
    check('record: refuses without a record path', rc5 == 2 and 'CONFORM-NOISE REFUSING' in out5 and 'RECORD is not set' in out5,
          out5[-200:])
    rc6, out6 = child(m, d1, 'other')
    check('an unknown mode refuses', rc6 == 2 and 'is not one of' in out6, out6[-200:])
    return rec, m, used(d0)


def test_frozen(rec, m, used_all):
    acc = [norm_of(m)]
    night = [r for r in ROWS if r[3].startswith(NIGHT)]
    dn, _ = rundir('night_alone', night)
    rc, out = child(m, dn)
    alone = [ln for ln in used(dn).splitlines()][0]
    check('a night judged alone uses its own median (7.0 Jy) - the defect the frozen mode removes',
          rc == 0 and alone.split(',')[4] == '7.0', alone)
    df, _ = rundir('night_frozen', night)
    rc, out = child(m, df, 'frozen', rec, acc)
    check('frozen: rc 0, the done line names the record', rc == 0 and 'read frozen from' in out, out[-400:])
    want = [ln for ln in used_all.splitlines() if ln.startswith('01h,73MHz,I,' + NIGHT)][0]
    check('frozen: the night\'s row is given the median the LST-wide run gave it (3.85 Jy, not its own 7.0 Jy)',
          used(df).strip() == want,
          f'{used(df).strip()} / {want}')
    check('frozen: the run\'s own median is printed beside the frozen one, not used',
          'recorded median 3.85 Jy over 2 rows (used)' in out and "own 1 rows give 7.0 Jy (not used)" in out, out[-500:])
    # a night with a row of 41 MHz V only: its look-up of ('01h', '41MHz', 'V') and the I key answer as LST-wide
    nv = [('01h', '41MHz', 'V', '2025-01-09T00', '5.0', '10')]
    dv, _ = rundir('night_v', nv)
    rc, out = child(m, dv, 'frozen', rec, acc)
    g = used(dv).strip().split(',')
    check('frozen: a key this run has no row of answers as LST-wide (the record is used whole)',
          rc == 0 and abs(float(g[4]) - 0.105) < 1e-12 and abs(float(g[5]) - 0.105) < 1e-12 and g[6] == '0.0', used(dv))
    # refusals
    rc, out = child(m, df, 'frozen', os.path.join(T, 'no_such.csv'), acc)
    check('frozen: no record refuses, naming the LST-wide conformance', rc == 2 and 'no table-noise record' in out
          and '--stages conform' in out, out[-300:])
    rc, out = child(m, df, 'frozen', rec, [])
    check('frozen: no accepted module versions named refuses', rc == 2 and 'EXOPIPE3_CONFORM_NOISE_SHA1 is empty' in out, out[-300:])
    rc, out = child(m, df, 'frozen', rec, ['0' * 40])
    check('frozen: a module outside the accepted versions refuses', rc == 2 and 'is not one the frozen mode was checked for' in out,
          out[-300:])
    d2, m2 = rundir('night_other_module', night, src=STANDIN + '# another version\n', tree='tree2')
    rc, out = child(m2, d2, 'frozen', rec, [norm_of(m2)])
    check('frozen: a record written by another module version refuses', rc == 2 and 'computed by other code' in out, out[-300:])
    new_band = [('01h', '82MHz', 'I', '2025-01-09T00', '5.0', '10')]
    d3, _ = rundir('night_new_band', new_band)
    rc, out = child(m, d3, 'frozen', rec, acc)
    check('frozen: a (LST, band, Stokes) of the run\'s rows that the record lacks refuses, naming it',
          rc == 2 and "('01h', '82MHz', 'I')" in out and 'no earlier night gives it' in out, out[-300:])
    nan_band = [('01h', '55MHz', 'I', '2025-01-09T00', '5.0', '10')]
    d4, _ = rundir('night_nan_band', nan_band)
    rc, out = child(m, d4, 'frozen', rec, acc)
    check('frozen: a key recorded with no finite value refuses as a missing one', rc == 2 and "('01h', '55MHz', 'I')" in out,
          out[-300:])
    body = open(rec).read()
    bad = os.path.join(T, 'dup.csv')
    with open(bad, 'w') as f:
        f.write(body + body.splitlines()[1] + '\n')
    rc, out = child(m, df, 'frozen', bad, acc)
    check('frozen: a record with a key twice refuses', rc == 2 and 'twice' in out, out[-300:])
    with open(bad, 'w') as f:
        f.write(body.replace('median_jy', 'median'))
    rc, out = child(m, df, 'frozen', bad, acc)
    check('frozen: a record with other columns refuses', rc == 2 and 'not a table-noise record' in out, out[-300:])
    with open(bad, 'w') as f:
        f.write(body.replace(',3,0.3,', ',3,-0.3,'))
    rc, out = child(m, df, 'frozen', bad, acc)
    check('frozen: a record with a non-positive median refuses', rc == 2 and 'inconsistent' in out, out[-300:])
    # a frozen tree's copy of the same module gives the same path-normalized sha1
    d5, m5 = rundir('night_tree3', night, src=STANDIN.replace("U = sys.argv[1]", "U = sys.argv[1]   # /lustre/gh/main/exopipe_ops"),
                    tree='tree3')
    d6, m6 = rundir('night_tree4', night, src=STANDIN.replace("U = sys.argv[1]", f"U = sys.argv[1]   # {T}/tree4/exopipe_ops"),
                    tree='tree4')
    check('the path-normalized sha1 of a tree\'s own copy equals the live-rooted one', norm_of(m5) == norm_of(m6)
          and norm_of(m5) != norm_of(m), f'{norm_of(m5)} {norm_of(m6)}')


def test_patch():
    def refuses(src, what):
        try:
            CN.patch(src)
        except CN.Refused as e:
            return what in str(e)
        return False
    line = CN.PATCHES[0][1]
    check('patch: applies to the stand-in (one patch, the hook\'s call in place of the line)',
          CN.patch(STANDIN)[1] == ['P1-table-noise'] and '__CONFORM_NOISE__.medians(_RMS, np)' in CN.patch(STANDIN)[0]
          and line not in CN.patch(STANDIN)[0])
    check('patch: a module without the line refuses', refuses(STANDIN.replace(line, '_RMS = {}\n'), 'occurs 0 times'))
    check('patch: a module with the line twice refuses', refuses(STANDIN.replace(line, line + line), 'occurs 2 times'))
    check('patch: another use of the name refuses (census)', refuses(STANDIN + 'print(len(_RMS))\n', "census '_RMS'"))
    check('patch: another look-up of the name refuses (census)',
          refuses(STANDIN.replace("print('conform: stand-in done'", "_RMS.get(1); print('conform: stand-in done'"), 'census'))


def test_wiring():
    with open(os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'catalog', 'tier_sublst.py')) as f:
        src = f.read()
    i = src.index('def st_conform(')
    body = src[i:src.index('\ndef ', i + 10)]
    check('wiring: the conform stage reads catalog_conform.TABLE_NOISE_RECORD[\'sublst\']',
          "shared('catalog_conform', 'TABLE_NOISE_RECORD', 'sublst')" in body)
    check('wiring: record mode through the runner, the record registered, a missing done line or record refuses',
          "EXOPIPE3_CONFORM_NOISE_MODE='record'" in body and 'conform_noise_runner(), [script, U]' in body
          and "s.out(rec, 'csv_table')" in body and "last_line(res.tail, r'^CONFORM-NOISE done')" in body
          and body.index('need_ok(res') < body.index("last_line(res.tail, r'^CONFORM-NOISE done')"))
    check('wiring: with the key off the module runs as before', "res = call(ctx, s, script, [U], env=ctx.env(**env_x), log=ctx.log.path, cwd=U)"
          in body)
    from exopipe.phase3.lib.catalog import tier_sublst as TS
    check('wiring: the runner and the record name are the wrap\'s', TS.conform_noise_runner() == RUNNER
          and os.path.basename(TS.conform_noise_record('/x')) == CN.FILE)
    from exopipe.phase3.core import env as _env
    _env.legacy()
    from exopipe.phase3.config import CFG
    try:
        cc = CFG.sections()['catalog_conform']
        check('config: TABLE_NOISE_RECORD on for the 10-minute tier only, ruled',
              cc.TABLE_NOISE_RECORD == {'deep': False, 'sublst': True, 'fork': False}
              and cc.RULED.get('TABLE_NOISE_RECORD') == 'R-GREGG-20261112-PER-HOUR-STEPS-SIX-RULINGS', cc.TABLE_NOISE_RECORD)
        check('config: NOISE_MODULE_SHA1 names 40-character sha1s', len(cc.NOISE_MODULE_SHA1) >= 1
              and all(len(x) == 40 and set(x) <= set('0123456789abcdef') for x in cc.NOISE_MODULE_SHA1))
        if os.path.isfile(LIVE):
            with open(LIVE) as f:
                live = f.read()
            try:
                patched, names = CN.patch(live)
                compile(patched, LIVE, 'exec')
                ok, det = True, ''
            except Exception as e:                                   # noqa: BLE001
                ok, det = False, f'{type(e).__name__}: {e}'
            check('real module: the patch applies to the live p3s2_conform.py and the patched source compiles', ok, det)
            check('real module: its path-normalized sha1 is accepted for the frozen mode (config catalog_conform.NOISE_MODULE_SHA1)',
                  CN.normalized_sha1(LIVE, live) in cc.NOISE_MODULE_SHA1, CN.normalized_sha1(LIVE, live))
        else:
            print(f'SKIP  real module: {LIVE} is not on this machine')
    finally:
        CFG.reset()


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        rec, m, used_all = test_passthrough_and_record()
        test_frozen(rec, m, used_all)
        test_patch()
        test_wiring()
    finally:
        if not os.environ.get('EXOPIPE3_TEST_KEEP'):
            shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
