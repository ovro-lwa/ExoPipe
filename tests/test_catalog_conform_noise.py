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
  evidence      the pulsar step's rows of other nights: the LST-wide run records every row within its frame beam of a
                pulsar (beam, daytime verdict, position error, the pulsar its note names); a night judged alone loses
                the group, the frozen run of the night gives its row the verdicts the LST-wide run gave it (from every
                pulsar the LST-wide run visited), with no evidence row left in the module's rows; a cell excluded since
                the record leaves the group; refusals: no evidence record, a malformed one, one of another module
                version, a needed evidence row without a position error, a row of another night, no night, another
                cadence
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
if os.path.isfile('/lustre/gh/main/exopipe_ops/exopipe3_site.py'):       # absent on a data-free machine: no site file is loaded
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
# the pulsar step, its lines as production has them where the patches touch (the loop over the pulsars the table notes name,
# the rows near a pulsar, the group of 2+ with each row's position error, the verdicts applied after the loop)
import collections, math, types
_TABLE_CUT = {_i: (_r.get('cut') == '1', _r.get('note') or '') for _i, _r in enumerate(raw) if _r.get('note')}
_DAY = {(_r['stokes'], _r['lst'], _r['band'], _r['date']): (_r.get('day') or '') for _r in raw}
_EXCL = set(filter(None, os.environ.get('STANDIN_EXCLUDED', '').split(';')))
def _is_excl(lst, band, date): return f'{lst},{band},{date}' in _EXCL
_G = types.SimpleNamespace(target_beam=lambda lst, band, date, stokes: 0.5)
_PSR_POS = {'J0001+01': (10.0, 20.0), 'J0002+02': (10.3, 20.0), 'J0003+03': (50.0, -5.0)}
_TIE = {_r['cand_id']: float(_r['sys']) for _r in csv.DictReader(open(f'{U}/tie.csv'))} if os.path.exists(f'{U}/tie.csv') else {}
def _sig_row(r):
    return math.hypot(10.0 / max(abs(f(r['snr'])), 5.0), _TIE[r['cand_id']])
def _unit(ra_, de_):
    return np.array([math.cos(math.radians(de_)) * math.cos(math.radians(ra_)), math.cos(math.radians(de_)) * math.sin(math.radians(ra_)),
                     math.sin(math.radians(de_))])
_psr_rows = collections.defaultdict(list)
for _i, (_c, _nt) in _TABLE_CUT.items():
    if _nt.startswith('PSR '): _psr_rows[_nt[4:]].append(_i)
_PSR_VERDICTS = collections.defaultdict(list)
if os.environ.get('P3_CADENCE', 'deep') not in ('deep', '10min'):
    _psr_rows = {}
_RA_ = np.array([f(r.get('ra')) for r in raw]); _DE_ = np.array([f(r.get('dec')) for r in raw])
_RAWU = np.nan_to_num(np.stack([np.cos(np.radians(_DE_)) * np.cos(np.radians(_RA_)), np.cos(np.radians(_DE_)) * np.sin(np.radians(_RA_)),
                                np.sin(np.radians(_DE_))], -1))
for _pn, _idx0 in _psr_rows.items():
    _pra, _pdec = _PSR_POS[_pn]; _pu = _unit(_pra, _pdec)
    near = []
    for _i in np.where(_RAWU @ _pu > math.cos(math.radians(1.5)))[0]:
        r = raw[_i]
        _sep = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(_pu, _unit(f(r['ra']), f(r['dec']))))))))
        if _sep > _G.target_beam(r['lst'], r['band'], r['date'], r['stokes']): continue
        if _is_excl(r['lst'], r['band'], r['date']) or _DAY.get((r['stokes'], r['lst'], r['band'], r['date']), 'x'): continue
        near.append(_i)
    if len(near) < 2: continue
    dets = {i: dict(ra=f(raw[i]['ra']), dec=f(raw[i]['dec']), sig=_sig_row(raw[i])) for i in near}
    _w = math.fsum(1.0 / d['sig'] ** 2 for d in dets.values())
    for i in near:
        if i in _TABLE_CUT: _PSR_VERDICTS[i].append((_pn, len(near), repr(_w)))
_out = []
for i, _vs in _PSR_VERDICTS.items():
    _out.append(raw[i]['cand_id'] + '|' + ';'.join(f'{_a},{_b},{_c}' for _a, _b, _c in sorted(_vs)))
with open(f'{U}/psr.csv', 'w') as _o:
    _o.write(''.join(_x + '\\n' for _x in sorted(_out)) + f'rows {len(raw)}\\n')
print(f'conform: stand-in pulsars {len(_psr_rows)}', flush=True)
print('conform: stand-in done', flush=True)
'''

ROWS = [   # lst, band, stokes, date, peak_jy, global_snr
    ('01h', '41MHz', 'I', '2025-01-01T00', '2.0', '10'), ('01h', '41MHz', 'I', '2025-01-02T00', '3.0', '10'),
    ('01h', '41MHz', 'I', '2025-01-03T01', '-8.0', '-20'), ('01h', '41MHz', 'V', '2025-01-01T00', '0.9', '9'),
    ('01h', '41MHz', 'V', '2025-01-02T02', '1.1', '10'), ('01h', '55MHz', 'I', '2025-01-02T00', '5.0', '0'),     # S/N 0: skipped
    ('01h', '55MHz', 'I', '2025-01-03T00', 'x', '7'),                                                         # unreadable: nan
    ('01h', '73MHz', 'I', '2025-01-01T03', '7.0', '10'), ('01h', '73MHz', 'I', '2025-01-09T00', '70.0', '10')]
NIGHT = '2025-01-09'            # the "new night": one row of 73 MHz I, whose own median (7.0) is not the LST-wide one (3.85)

# the pulsar step's rows (all 01h 41 MHz I; frame beam 0.5 deg): pulsars J0001+01 at (10.0, 20.0), J0002+02 0.28 deg from it,
# J0003+03 at (50.0, -5.0)
ECOLS = ['lst', 'band', 'stokes', 'date', 'peak_jy', 'global_snr', 'cand_id', 'ra', 'dec', 'snr', 'note', 'cut', 'day']


def erow(cand, date, ra, dec, snr, note='', cut='', day=''):
    return dict(lst='01h', band='41MHz', stokes='I', date=date, peak_jy='2.0', global_snr='10', cand_id=cand, ra=str(ra),
                dec=str(dec), snr=str(snr), note=note, cut=cut, day=day)


EROWS = [erow('a1', '2025-01-01T00', 10.01, 20.0, 10, 'PSR J0001+01', '1'),      # another night, its note names the first pulsar
         erow('a2', '2025-01-02T00', 10.02, 20.01, 8),                            # another night, no note
         erow('a3', '2025-01-03T01', 10.0, 20.02, 12, 'PSR J0002+02', '1'),       # another night, its note names the SECOND pulsar
         erow('n1', NIGHT + 'T00', 10.015, 20.005, 9, 'PSR J0001+01', '1'),       # the night's row
         erow('d1', '2025-01-04T00', 10.0, 20.0, 9, day='daytime: stand-in'),     # another night, its frame daytime
         erow('x1', '2025-01-05T00', 10.0, 19.99, 9),                             # another night, its cell excluded LATER
         erow('u1', '2025-01-06T00', 50.0, -5.0, 9),                              # near the third pulsar, no window tie
         erow('far', '2025-01-02T00', 100.0, 0.0, 9)]                             # near no pulsar
TIE = {'a1': 0.5, 'a2': 0.7, 'a3': 0.4, 'n1': 0.6, 'd1': 0.5, 'x1': 0.9, 'far': 0.5, 'n2': 0.5}


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
        if rows and isinstance(rows[0], dict):              # the pulsar step's rows, with the window-tie stand-in
            w = csv.DictWriter(f, ECOLS)
            w.writeheader()
            w.writerows(rows)
            with open(os.path.join(d, 'tie.csv'), 'w') as g:
                g.write('cand_id,sys\n' + ''.join(f'{k},{v}\n' for k, v in TIE.items()))
        else:
            w = csv.writer(f)
            w.writerow(['lst', 'band', 'stokes', 'date', 'peak_jy', 'global_snr'])
            w.writerows(rows)
    m = os.path.join(T, tree, 'exopipe_ops', 'p3s2_conform.py')
    os.makedirs(os.path.dirname(m), exist_ok=True)
    with open(m, 'w') as f:
        f.write(src)
    return d, m


def child(module, d, mode=None, record=None, accepted=None, extra=None, evidence=None, night=NIGHT):
    """Run the module (mode None) or the runner. The pulsar-evidence record defaults to the table-noise record's sibling,
    the night to NIGHT, the cadence to the 10-minute one."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(('EXOPIPE3_CONFORM_NOISE_', 'STANDIN_'))}
    env['P3_CADENCE'] = '10min'
    if mode is None:
        cmd = [sys.executable, module, d]
    else:
        cmd = [sys.executable, RUNNER, module, d]
        env['EXOPIPE3_CONFORM_NOISE_MODE'] = mode
        if record:
            env['EXOPIPE3_CONFORM_NOISE_RECORD'] = record
            env['EXOPIPE3_CONFORM_NOISE_EVIDENCE'] = evidence or os.path.join(os.path.dirname(record), CN.EVIDENCE_FILE)
        if mode == 'frozen' and night:
            env['EXOPIPE3_CONFORM_NOISE_NIGHT'] = night
        if accepted is not None:
            env['EXOPIPE3_CONFORM_NOISE_SHA1'] = ','.join(accepted)
    env.update(extra or {})
    r = subprocess.run(cmd, env=env, cwd=d, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    return r.returncode, r.stdout


def used(d):
    """What the stand-in wrote: its table step's values, then its pulsar step's verdicts and its number of rows."""
    with open(os.path.join(d, 'used.csv')) as f, open(os.path.join(d, 'psr.csv')) as g:
        return f.read() + g.read()


def psr(d):
    """{cand_id: verdicts} and the stand-in's number of rows after its pulsar step."""
    with open(os.path.join(d, 'psr.csv')) as f:
        lines = f.read().splitlines()
    return dict(ln.split('|') for ln in lines[:-1]), lines[-1]


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
    check('record: the pulsar-evidence record is written beside it, empty when no row lies near a pulsar',
          CN.read_evidence(os.path.join(d2, CN.EVIDENCE_FILE))[0] == [] and 'recorded in ' + os.path.join(d2, CN.EVIDENCE_FILE) in out2)
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
    check('record: a module that exits early writes no record', rc4 != 0 and not os.path.exists(os.path.join(d4, CN.FILE))
          and not os.path.exists(os.path.join(d4, CN.EVIDENCE_FILE)), out4[-200:])
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
    alone = used(dn).splitlines()[0]
    check('a night judged alone uses its own median (7.0 Jy) - the defect the frozen mode removes',
          rc == 0 and alone.split(',')[4] == '7.0', alone)
    df, _ = rundir('night_frozen', night)
    rc, out = child(m, df, 'frozen', rec, acc)
    check('frozen: rc 0, the done line names the record', rc == 0 and 'read frozen from' in out, out[-400:])
    want = [ln for ln in used_all.splitlines() if ln.startswith('01h,73MHz,I,' + NIGHT)][0]
    check('frozen: the night\'s row is given the median the LST-wide run gave it (3.85 Jy, not its own 7.0 Jy)',
          used(df).splitlines()[0] == want,
          f'{used(df).splitlines()[0]} / {want}')
    check('frozen: the run\'s own median is printed beside the frozen one, not used',
          'recorded median 3.85 Jy over 2 rows (used)' in out and "own 1 rows give 7.0 Jy (not used)" in out, out[-500:])
    # a night with a row of 41 MHz V only: its look-up of ('01h', '41MHz', 'V') and the I key answer as LST-wide
    nv = [('01h', '41MHz', 'V', '2025-01-09T00', '5.0', '10')]
    dv, _ = rundir('night_v', nv)
    rc, out = child(m, dv, 'frozen', rec, acc)
    g = used(dv).splitlines()[0].split(',')
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
    # the reader takes back every row the writer can write; whether a recorded noise can judge the night is asked for the
    # keys of the night's rows only (code review of pull request 36)
    with open(bad, 'w') as f:
        f.write(body.replace(',2,3.85,', ',2,0.0,'))
    rc, out = child(m, df, 'frozen', bad, acc, evidence=os.path.join(os.path.dirname(rec), CN.EVIDENCE_FILE))
    check('frozen: a recorded median of 0.0 Jy for a key of the night\'s rows refuses as no usable noise; the record reads back',
          rc == 2 and 'not a finite positive number' in out and "('01h', '73MHz', 'I')" in out
          and CN.read_record(bad)[0][('01h', '73MHz', 'I')] == (2, 0.0), out[-300:])
    with open(bad, 'w') as f:
        f.write(body.replace(',3,0.3,', ',3,0.0,'))
    rc, out = child(m, df, 'frozen', bad, acc, evidence=os.path.join(os.path.dirname(rec), CN.EVIDENCE_FILE))
    check('frozen: a recorded median of 0.0 Jy for a key the night has no row of does not stop the night',
          rc == 0 and 'read frozen from' in out, out[-300:])
    for old, new, what in ((',3,0.3,', ',3,nan,', 'no median over 3 finite values'),
                           (',0,nan,', ',0,1.5,', 'a finite median over no finite value'),
                           (',3,0.3,', ',-1,0.3,', 'a negative count')):
        with open(bad, 'w') as f:
            f.write(body.replace(old, new))
        rc, out = child(m, df, 'frozen', bad, acc)
        check(f'frozen: a record row with {what} refuses as inconsistent', old in body and rc == 2 and 'inconsistent' in out,
              out[-300:])
    # a frozen tree's copy of the same module gives the same path-normalized sha1
    d5, m5 = rundir('night_tree3', night, src=STANDIN.replace("U = sys.argv[1]", "U = sys.argv[1]   # /lustre/gh/main/exopipe_ops"),
                    tree='tree3')
    d6, m6 = rundir('night_tree4', night, src=STANDIN.replace("U = sys.argv[1]", f"U = sys.argv[1]   # {T}/tree4/exopipe_ops"),
                    tree='tree4')
    check('the path-normalized sha1 of a tree\'s own copy equals the live-rooted one', norm_of(m5) == norm_of(m6)
          and norm_of(m5) != norm_of(m), f'{norm_of(m5)} {norm_of(m6)}')


def test_evidence():
    """The pulsar step's rows of other nights (patches P2 - P5)."""
    d0, m = rundir('ev_plain', EROWS)
    rc0, out0 = child(m, d0)
    all_v, all_n = psr(d0)
    check('evidence: unpatched, all nights - the night\'s row takes a verdict from BOTH pulsars the notes name (groups of 5: the '
          'daytime row is out)', rc0 == 0 and [v.split(',')[:2] for v in all_v.get('n1', '').split(';')]
          == [['J0001+01', '5'], ['J0002+02', '5']] and all_n == 'rows 8', f'{all_v} {all_n} {out0[-300:]}')
    d1, _ = rundir('ev_pass', EROWS)
    rc1, out1 = child(m, d1, 'passthrough')
    check('evidence: passthrough equals the unpatched run, byte for byte', rc1 == 0 and used(d1) == used(d0), out1[-300:])
    d2, _ = rundir('ev_record', EROWS)
    rec, evr = os.path.join(d2, CN.FILE), os.path.join(d2, CN.EVIDENCE_FILE)
    rc2, out2 = child(m, d2, 'record', rec)
    check('evidence: record mode equals the unpatched run, byte for byte', rc2 == 0 and used(d2) == used(d0), out2[-300:])
    rows, shas, _sha = CN.read_evidence(evr) if os.path.isfile(evr) else ([], set(), '')
    by = {r['cand_id']: r for r in rows}
    check('evidence: the record holds every row within its frame beam of a pulsar, in the module\'s order, and no other',
          [r['cand_id'] for r in rows] == ['a1', 'a2', 'a3', 'n1', 'd1', 'x1', 'u1'], [r['cand_id'] for r in rows])
    check('evidence: the record carries the catalog\'s own strings, the frame beam, the daytime verdict, the note\'s pulsar',
          bool(by) and by['a1']['ra'] == '10.01' and by['a1']['beam_deg'] == '0.5' and by['a1']['note_pulsar'] == 'J0001+01'
          and by['a3']['note_pulsar'] == 'J0002+02' and by['a2']['note_pulsar'] == '' and by['d1']['day'] == 'daytime: stand-in'
          and by['a2']['day'] == '' and shas == {norm_of(m)}, by.get('a1'))
    check('evidence: the recorded position error is the module\'s own (repr), empty for a row without a window tie',
          bool(by) and by['a1']['sig_arcsec'] == repr(math.hypot(10.0 / 10, 0.5)) and by['u1']['sig_arcsec'] == '', by.get('u1'))
    acc = [norm_of(m)]
    night = [r for r in EROWS if r['date'].startswith(NIGHT)]
    dn, _ = rundir('ev_night_alone', night)
    rc, out = child(m, dn)
    check('evidence: the night judged alone loses its group - no verdict for its row (the defect)', rc == 0 and psr(dn) == ({}, 'rows 1'),
          psr(dn))
    df, _ = rundir('ev_night_frozen', night)
    rc, out = child(m, df, 'frozen', rec, acc)
    fv, fn = psr(df) if rc == 0 else ({}, '')
    check('evidence: frozen - the night\'s row takes the verdicts the all-nights run gave it, from both pulsars, value for value',
          rc == 0 and fv == {'n1': all_v['n1']}, f'{fv} / {all_v.get("n1")} {out[-400:]}')
    check('evidence: frozen - no evidence row is left in the module\'s rows after its pulsar step', fn == 'rows 1', fn)
    check('evidence: frozen - the done line counts the recorded rows read, joined, and the pulsar only other nights name',
          '6 recorded detection(s) of other nights' in out and '4 joined a group' in out and 'in 2 pulsar look-up(s)' in out
          and '1 pulsar(s) named only by other nights' in out, out[-500:])
    rc, out = child(m, df, 'frozen', rec, acc, extra={'STANDIN_EXCLUDED': '01h,41MHz,2025-01-05T00'})
    fv, fn = psr(df) if rc == 0 else ({}, '')
    check('evidence: frozen - a cell excluded since the record leaves the groups (exclusion is tested live): groups of 4',
          rc == 0 and [v.split(',')[:2] for v in fv.get('n1', '').split(';')] == [['J0001+01', '4'], ['J0002+02', '4']], f'{fv} {out[-300:]}')
    # refusals
    two = night + [erow('n2', NIGHT + 'T01', 50.0, -5.01, 9, 'PSR J0003+03', '1')]
    d3, _ = rundir('ev_night_untied', two)
    rc, out = child(m, d3, 'frozen', rec, acc)
    check('evidence: frozen - a group that needs an evidence row without a position error refuses, naming the row',
          rc == 2 and 'u1 joins a pulsar\'s group and holds no recorded position error' in out, out[-300:])
    d4, _ = rundir('ev_two_nights', night + [EROWS[0]])
    rc, out = child(m, d4, 'frozen', rec, acc)
    check('evidence: frozen - a row of another night in the run refuses', rc == 2 and 'rows of 1 other night(s)' in out
          and '2025-01-01' in out, out[-300:])
    rc, out = child(m, df, 'frozen', rec, acc, night='')
    check('evidence: frozen - no night named refuses', rc == 2 and 'is not a night YYYY-MM-DD' in out, out[-300:])
    rc, out = child(m, df, 'frozen', rec, acc, evidence=os.path.join(T, 'no_such_evidence.csv'))
    check('evidence: frozen - no evidence record refuses, naming the LST-wide conformance', rc == 2
          and 'no pulsar-evidence record' in out and '--stages conform' in out, out[-300:])
    body = open(evr).read()
    bad = os.path.join(T, 'bad_evidence.csv')
    with open(bad, 'w') as f:
        f.write(body.replace('sig_arcsec', 'sig'))
    rc, out = child(m, df, 'frozen', rec, acc, evidence=bad)
    check('evidence: frozen - a record with other columns refuses', rc == 2 and 'not a pulsar-evidence record' in out, out[-300:])
    with open(bad, 'w') as f:
        f.write(body + body.splitlines()[1] + '\n')
    rc, out = child(m, df, 'frozen', rec, acc, evidence=bad)
    check('evidence: frozen - a record with a row twice (a cell executed twice, as the LST-wide pulsar step saw it) is read whole',
          rc == 0 and len(CN.read_evidence(bad)[0]) == len(body.splitlines()) and 'pulsar evidence:' in out, out[-300:])
    with open(bad, 'w') as f:
        f.write(body + ',' + body.splitlines()[1].split(',', 1)[1] + '\n')
    rc, out = child(m, df, 'frozen', rec, acc, evidence=bad)
    check('evidence: frozen - a row without a cand_id refuses', rc == 2 and 'holds no cand_id' in out, out[-300:])
    with open(bad, 'w') as f:
        f.write(body.replace(',0.5,', ',-0.5,', 1))
    rc, out = child(m, df, 'frozen', rec, acc, evidence=bad)
    check('evidence: frozen - a record with a non-positive beam refuses', rc == 2 and 'not a finite positive number' in out, out[-300:])
    with open(bad, 'w') as f:
        f.write(body.replace(norm_of(m), '0' * 40, 1))
    rc, out = child(m, df, 'frozen', rec, acc, evidence=bad)
    check('evidence: frozen - a record written by another module version refuses', rc == 2 and 'recorded by other code' in out,
          out[-300:])
    for mode in ('record', 'frozen'):
        d5, _ = rundir(f'ev_cadence_{mode}', night)
        rc, out = child(m, d5, mode, rec if mode == 'frozen' else os.path.join(d5, CN.FILE), acc, extra={'P3_CADENCE': 'deep'})
        check(f'evidence: {mode} - another cadence than the 10-minute one refuses', rc == 2 and 'cadence only' in out, out[-300:])
    env = {k: v for k, v in os.environ.items() if not k.startswith('EXOPIPE3_CONFORM_NOISE_')}
    env.update(P3_CADENCE='10min', EXOPIPE3_CONFORM_NOISE_MODE='record', EXOPIPE3_CONFORM_NOISE_RECORD=os.path.join(d5, CN.FILE))
    r = subprocess.run([sys.executable, RUNNER, m, d5], env=env, cwd=d5, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True)
    check('evidence: record - no evidence path refuses', r.returncode == 2 and 'EXOPIPE3_CONFORM_NOISE_EVIDENCE is not set' in r.stdout,
          r.stdout[-300:])


def test_patch():
    def refuses(src, what):
        try:
            CN.patch(src)
        except CN.Refused as e:
            return what in str(e)
        return False
    line = CN.PATCHES[0][1]
    names = ['P1-table-noise', 'P2-pulsar-loop', 'P3-pulsar-near', 'P4-pulsar-error', 'P5-pulsar-release']
    check('patch: applies to the stand-in (five patches, the hook\'s calls in place of the lines)',
          CN.patch(STANDIN)[1] == names and '__CONFORM_NOISE__.medians(_RMS, np)' in CN.patch(STANDIN)[0]
          and line not in CN.patch(STANDIN)[0] and all(CN.patch(STANDIN)[0].count(f'__CONFORM_NOISE__.{h}(') == 1
                                                         for h in ('evidence', 'pulsars', 'near', 'sig', 'release')))
    for nm, old, _new in CN.PATCHES[1:]:
        check(f'patch: a module without the {nm} text refuses', refuses(STANDIN.replace(old, old.replace(old.strip()[:6], old.strip()[:6].upper(), 1)),
                                                                      f'{nm}: its text occurs 0 times'))
    check('patch: a second call of the position-error function refuses (census)',
          refuses(STANDIN + '_sig_row(raw[0])\n', "census '_sig_row('"))
    check('patch: a second reader of the pulsars the notes name refuses (census)',
          refuses(STANDIN + 'print(len(_psr_rows))\n', "census '_psr_rows'"))
    check('patch: a module without the line refuses', refuses(STANDIN.replace(line, '_RMS = {}\n'), 'occurs 0 times'))
    check('patch: a module with the line twice refuses', refuses(STANDIN.replace(line, line + line), 'occurs 2 times'))
    check('patch: another use of the name refuses (census)', refuses(STANDIN + 'print(len(_RMS))\n', "census '_RMS'"))
    check('patch: a comment that mentions the names is no use of them (the census counts the code)',
          CN.patch(STANDIN + '# _RMS, _RMS.get(, _psr_rows, _PSR_VERDICTS and _sig_row( in a comment\nx = 1   # _RMS = 2\n')[1]
          == names)
    check('patch: a `#` inside a string starts no comment (the use after it is counted)',
          refuses(STANDIN + "print('# ' + str(len(_RMS)))\n", "census '_RMS'"))
    check('patch: a module that does not tokenize refuses', refuses(STANDIN + 'x = (1,\n', 'does not tokenize'))
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
    check('wiring: record mode writes the pulsar-evidence record beside the noise record and registers it; a missing one refuses',
          'EXOPIPE3_CONFORM_NOISE_EVIDENCE=evr' in body and 'evr = conform_pulsar_evidence(U)' in body
          and "s.out(evr, 'csv_table')" in body and 'not os.path.isfile(evr)' in body)
    check('wiring: the night pass reads the pulsar evidence of the frozen-from directory for its night; a missing record refuses '
          'before the module runs',
          'conform_pulsar_evidence(ctx.frozen_from)' in body and 'EXOPIPE3_CONFORM_NOISE_EVIDENCE=evid' in body
          and 'EXOPIPE3_CONFORM_NOISE_NIGHT=ctx.night' in body and 's.inp(frozen, evid)' in body
          and body.index('if not os.path.isfile(evid):') < body.index("EXOPIPE3_CONFORM_NOISE_MODE='frozen'")
          and "'pulsar evidence:' not in done" in body)
    check('wiring: the night pass reads the two records only under the LST-wide directory\'s own conformance record',
          'frozen_records_refusal(ctx.frozen_from, ctx.unit, (frozen, evid))' in body
          and body.index('if not os.path.isfile(evid):') < body.index('frozen_records_refusal(')
          < body.index("EXOPIPE3_CONFORM_NOISE_MODE='frozen'"))
    check('wiring: with the key off the module runs as before', "res = call(ctx, s, script, [U], env=ctx.env(**env_x), log=ctx.log.path, cwd=U)"
          in body)
    from exopipe.phase3.lib.catalog import tier_sublst as TS
    check('wiring: the runner and the record name are the wrap\'s', TS.conform_noise_runner() == RUNNER
          and os.path.basename(TS.conform_noise_record('/x')) == CN.FILE
          and os.path.basename(TS.conform_pulsar_evidence('/x')) == CN.EVIDENCE_FILE)
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


def test_frozen_records_gate():
    """tier_sublst.frozen_records_refusal: the two records judge a night only when the LST-wide directory's own record of
    its conformance stage (same LST and arm) is done and lists both files as they are now."""
    from exopipe.phase3.lib.catalog import tier_sublst as TS
    from exopipe.phase3.core import record as R
    d = os.path.join(T, 'lstwide_gate')
    os.makedirs(d)
    a, b = TS.conform_noise_record(d), TS.conform_pulsar_evidence(d)
    for p, text in ((a, 'noise\n'), (b, 'evidence\n')):
        with open(p, 'w') as f:
            f.write(text)

    def stage(status='done', paths=(a, b), unit='01h_B.conform'):
        p = R.record_path(d, 'catalog', unit)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        R.atomic_write_json(p, dict(status=status, reason='S-G13: stand-in', finished='2026-11-13T00:00:00Z',
                                    meta={'output_fingerprints': TS.output_fingerprints(paths)}))
    why = TS.frozen_records_refusal(d, '01h_B', (a, b))
    check('gate: a directory without a record of its conformance stage refuses', why[1] is None
          and 'holds no record of the conformance stage 01h_B.conform' in why[0], why)
    stage()
    check('gate: a done conformance that lists both records as they are passes, with its finished time',
          TS.frozen_records_refusal(d, '01h_B', (a, b)) == (None, '2026-11-13T00:00:00Z'))
    why = TS.frozen_records_refusal(d, '01h_A', (a, b))
    check('gate: the directory of another arm refuses (no record of this arm\'s stage)',
          'holds no record of the conformance stage 01h_A.conform' in str(why[0]), why)
    stage(status='refused')
    why = TS.frozen_records_refusal(d, '01h_B', (a, b))
    check('gate: a later conformance that refused makes the records an earlier run\'s: refuses',
          'is refused' in str(why[0]) and 'earlier run' in str(why[0]), why)
    stage(paths=(a,))
    why = TS.frozen_records_refusal(d, '01h_B', (a, b))
    check('gate: a record file the conformance did not write refuses', 'is not an output' in str(why[0])
          and os.path.basename(b) in str(why[0]), why)
    stage()
    with open(b, 'a') as f:
        f.write('x\n')
    why = TS.frozen_records_refusal(d, '01h_B', (a, b))
    check('gate: a record file changed since the conformance recorded it refuses', 'changed since' in str(why[0])
          and os.path.basename(b) in str(why[0]), why)


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        rec, m, used_all = test_passthrough_and_record()
        test_frozen(rec, m, used_all)
        test_evidence()
        test_patch()
        test_wiring()
        test_frozen_records_gate()
    finally:
        if not os.environ.get('EXOPIPE3_TEST_KEEP'):
            shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
