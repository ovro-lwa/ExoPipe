"""B-02 step 1: lib/search/removal_rules.py against production's conformance (exopipe_ops/p3s2_conform.py).

Each comparison gives production's script and the port their own sandbox copy of a run directory - the merged rows, the
gate ledger, the run's three frame caches (copies), the archived search frames and v110 verdicts (read through symlinks)
- with the environment of Run_P3S5's 2026-10-14 pass, and compares every file both write, byte for byte, and their
printed conform lines:
  (1) eight Run_P3S5 cells, their as-run rows and ledger rows, changed so that every rule removes or flags at least one row
      (a duplicate, an excluded night, a frame whose times cannot be read, I and V noise epochs, a row-line, an unbacked
      row, axis ratios, sizes and v110 labels to compute, a v110 marginal to re-admit) - daytime, twilight, the Jupiter
      track, the tables and the pulsar groups fire on the real rows;
  (2) the same with the size step on (production's module default);
  (--full) every row of Run_P3S5 (about an hour each; by hand, not in the suite).
Unit checks: production's own small functions (taken from its source) against the port's; the frame-mid elevation; the
refusal of every mode the port is not (the fork's, the 10-minute tier's).
Run: python tests/test_removal_rules.py [--full]
"""
import ast
import csv
import filecmp
import glob
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from exopipe.phase3.core import env as ENV                          # noqa: E402
ENV.legacy()
from exopipe.phase3.config import CFG                               # noqa: E402
from exopipe.phase3.lib.search import removal_rules as R            # noqa: E402

T = '/tmp/exopipe3_tests_removal_rules'
RUN = os.path.realpath('/data07/gh/phase3/transients/Run_P3S5')
PROD = '/lustre/gh/main/exopipe_ops/p3s2_conform.py'
PAR = '/fast/gh/main/phase3/p3s5_parents_union_20261014T0203'     # the 2026-10-14 pass's parent snapshot (pass_fork_20261014T0203)
PY = '/opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3'
CELLS = (('09h', 'Red'), ('20h', '18MHz'), ('22h', 'Red'), ('22h', 'Full'), ('21h', 'Red'), ('08h', '55MHz'), ('15h', 'Full'),
         ('16h', 'Red'))
OUTPUTS = ('master_catalog.csv', 'master_catalog_asrun.csv', 'master_catalog_removed.csv', 'frame_beam_ratio.csv',
           'frame_beam_bmaj.csv', 'frame_times.csv')
CACHES = ('frame_beam_ratio.csv', 'frame_beam_bmaj.csv', 'frame_times.csv')
FAILS, N = [], [0]

_PORT = r'''
import sys
sys.path.insert(0, sys.argv[1])
from exopipe.phase3.core import env
env.legacy()
from exopipe.phase3.config import CFG
CFG.authorize(out=sys.argv[2])                                  # production's conventions: the deep elevation at mid-hour
CFG.set('catalog_conform.DEEP_ELEV_EPOCH', 'mid-hour')          # (the ruled 'frame-mid' since B-02 step 3b; the parity file)
from exopipe.phase3.lib.search import removal_rules as R
R.conform(sys.argv[2], size_step=sys.argv[3] == '1')
'''


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({detail})'))
    if not ok:
        FAILS.append(name)


def prod_functions():
    """bclass, _qflag and _deep_mid_el as production's source defines them (with SUB; _QUAR filled by the caller)."""
    tree = ast.parse(open(PROD).read())
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in ('bclass', '_qflag', '_deep_mid_el'))
            or (isinstance(n, ast.Assign) and any(getattr(t, 'id', '') == 'SUB' for t in n.targets))]
    import numpy as np
    ns = {'np': np, '_QUAR': {}}
    exec(compile(ast.Module(body=keep, type_ignores=[]), PROD, 'exec'), ns)
    return ns


def test_units():
    ns = prod_functions()
    bands = [b for subs in CFG.catalog.BAND_CLASS.values() for b in subs] + ['Red', 'Green', 'Blue', 'Full', '13MHz', 'x']
    check('band_class is production\'s bclass on every band (and an unknown name)',
          all(R.band_class(b) == ns['bclass'](b) for b in bands), [(b, R.band_class(b), ns['bclass'](b)) for b in bands
                                                                   if R.band_class(b) != ns['bclass'](b)])
    quar = {('09h', '2025-04-21'): {'18', '64'}, ('10h', '2025-01-01'): {'41'}}
    ns['_QUAR'] = quar
    cases = [(l, b, d) for (l, d) in list(quar) + [('09h', '2025-04-22')] for b in bands]
    check('quarantined is production\'s _qflag (a subband by itself, a wideband by any constituent, Full by any)',
          all(R.quarantined(quar, *k) == ns['_qflag'](*k) for k in cases),
          [k for k in cases if R.quarantined(quar, *k) != ns['_qflag'](*k)][:5])
    pts = [('05h', 100.0, 20.0), ('23h', 359.5, -30.0), ('00h', 0.2, 89.0), ('12h', 180.0, 37.2398)]
    check('the mid-hour elevation is production\'s _deep_mid_el, bit for bit',
          all(R.mid_hour_elevation(*p) == ns['_deep_mid_el'](*p) for p in pts))
    from astropy.coordinates import SkyCoord, AltAz
    from astropy.time import Time
    import astropy.units as u
    t = ('2025-04-21T03:32:00.000', '2025-04-21T04:10:00.000')
    el, words = R.deep_elevation('09h', 140.0, 30.0, epoch='frame-mid', frame_time=t)
    tm = Time(t[0]) + (Time(t[1]) - Time(t[0])) * 0.5
    ref = SkyCoord(140.0 * u.deg, 30.0 * u.deg).transform_to(AltAz(obstime=tm, location=R._ovro())).alt.deg
    check('frame-mid: the elevation at the frame\'s mid-time (within 0.6 deg of astropy\'s full transform - the rule\'s '
          'formula takes no precession or refraction, as production\'s)', words == 'at the frame mid-time'
          and abs(el - ref) < 0.6, (el, ref))
    for bad in (dict(epoch='frame-mid'), dict(epoch='noon')):
        try:
            R.deep_elevation('09h', 140.0, 30.0, **bad)
            ok = False
        except ValueError:
            ok = True
        check(f'deep_elevation refuses {bad} (no time, an unknown convention)', ok)
    for k, v in (('P3_CONFORM_FORK', '1'), ('P3_CONFORM_TABLE', 'multi'), ('P3_CADENCE', '10min'), ('P3_TENMIN_PERTIE', '1'),
                 ('P3_GATE_TARGET_BEAM', '0'), ('P3_TABLE_PREFIX', 'sublst'), ('P3_CONFORM_SIZE', '1')):
        old = os.environ.get(k)
        os.environ[k] = v
        try:
            R._refuse_other_modes(size_step=False)
            ok = False
        except SystemExit as e:
            ok = k in str(e)
        finally:
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
        check(f'the port refuses {k}={v} (not the deep main chain, or not the size step it was asked for)', ok)


def _rows(p):
    with open(p, newline='') as f:
        rd = csv.DictReader(f)
        return rd.fieldnames, list(rd)


def _write(p, cols, rows):
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def subset():
    """(catalogue columns, as-run rows of CELLS, ledger columns, ledger rows of CELLS) from Run_P3S5."""
    cells = set(CELLS)
    cols, rows = _rows(os.path.join(RUN, 'master_catalog_asrun.csv'))
    rows = [r for r in rows if (r['lst'], r['band']) in cells]
    lrows = []
    with open(os.path.join(RUN, 'v_deep_ledger.csv'), newline='') as f:
        rd = csv.DictReader(f)
        lcols = rd.fieldnames
        for r in rd:
            if (r['lst'], r['scope']) in cells:
                lrows.append(r)
    return cols, rows, lcols, lrows


def mutate(rows, lrows, kept_ids):
    """Change the subset so that every rule fires, each on rows the unchanged subset keeps. Returns (rows, lrows, the
    frame-time and frame changes, expectations [(what, cand_id, the reason it must get - or 'kept:<column>' for a
    conformant row whose column must be set)])."""
    import numpy as np
    out = dict(times_add={}, times_drop=set(), omit=set())
    f = R.f
    kept = [r for r in rows if r['cand_id'] in kept_ids]
    used, expect = set(), []

    def take(pred, n=1):
        got = [r for r in kept if pred(r) and r['cand_id'] not in used][:n]
        assert len(got) == n, 'no row for a mutation'
        used.update(r['cand_id'] for r in got)
        return got
    per = {}
    for x in kept:
        per.setdefault((x['lst'], x['band'], x['stokes'], x['date']), []).append(x)
    # duplicate: an exact copy of a conformant row, after it
    dup = take(lambda r: True)[0]
    rows.append(dict(dup))
    expect.append(('a duplicate', dup['cand_id'], 'duplicate'))
    # excluded: a 21h Red row moved to 2025-08-06 (excluded since 2026-10-14 04:27; its searched frames are archived)
    r = take(lambda r: r['lst'] == '21h')[0]
    r['date'] = '2025-08-06'
    expect.append(('an excluded night', r['cand_id'], 'excluded epoch'))
    # I noise: an 08h 55MHz I row moved to 2024-12-27 (qa3 4.7x the cell median; its I frame archived, the search refused
    # the epoch), with the archived frame's own times
    r = take(lambda r: (r['lst'], r['band'], r['stokes']) == ('08h', '55MHz', 'I'))[0]
    r['date'] = '2024-12-27'
    out['times_add'][('I', '08h', '55MHz', '2024-12-27')] = ('2024-12-27T09:27:38.000', '2024-12-27T10:27:29.060')
    expect.append(('an I noise epoch', r['cand_id'], 'epoch refused: I'))
    # frame times unreadable: one conformant epoch whose residual frame and cache row are both gone
    k = next(kk for kk, v in sorted(per.items()) if kk[:2] == ('22h', 'Full') and not any(x['cand_id'] in used for x in v))
    out['times_drop'].add((k[2], k[0], k[1], k[3]))
    out['omit'].add((k[2], k[0], k[1], k[3]))
    for x in per[k]:
        used.add(x['cand_id'])
        expect.append(('a frame without times', x['cand_id'], 'daytime gate: frame'))
    # unbacked: every final of an epoch holding one conformant row is removed from the ledger
    k1 = next(kk for kk, v in sorted(per.items()) if len(v) == 1 and v[0]['cand_id'] not in used)
    used.add(per[k1][0]['cand_id'])
    expect.append(('an unbacked row', per[k1][0]['cand_id'], 'no final gate-ledger'))
    lrows[:] = [x for x in lrows if not ((x['lst'], x['scope'], x['stokes'], x['date']) == k1 and x['final'] == '1')]
    # row-line: five conformant rows of one epoch; their nearest finals moved onto one pixel row, the epoch's other finals
    # off it
    k5 = next(kk for kk, v in sorted(per.items()) if len(v) >= 5 and not any(x['cand_id'] in used for x in v))
    used.update(x['cand_id'] for x in per[k5])
    fin = [x for x in lrows if (x['lst'], x['scope'], x['stokes'], x['date']) == k5 and x['final'] == '1']
    on = set()
    for x in per[k5][:5]:
        d = [np.hypot((f(y['ra']) - f(x['ra'])) * np.cos(np.radians(f(x['dec']))), f(y['dec']) - f(x['dec'])) for y in fin]
        j = int(np.argmin(d))
        fin[j]['pix_y'] = '1234'
        on.add(j)
        expect.append(('a row-line', x['cand_id'], 'row-line'))
    for j, y in enumerate(fin):
        if j not in on and abs(int(float(y['pix_y'])) - 1234) <= 1:
            y['pix_y'] = '3000'
    # axis ratio above 8 sigma: one removed (> 2.0), one flagged (1.5-2.0); one row's axis_norm and one's size computed
    big = lambda r: abs(f(r['snr'])) > CFG.catalog_conform.SIZE_SNR and r['axis_norm'] not in ('', 'nan')
    r = take(big)[0]; r['axis_norm'] = '2.6'; expect.append(('an axis ratio 2.6', r['cand_id'], 'axis ratio'))
    r = take(big)[0]; r['axis_norm'] = '1.8'; expect.append(('an axis ratio 1.8', r['cand_id'], 'kept:axis_flag'))
    r = take(lambda r: r['axis_ratio'] not in ('', 'nan'))[0]; r['axis_norm'] = ''
    expect.append(('an axis ratio over the frame beam', r['cand_id'], 'kept:axis_norm'))
    r = take(lambda r: r['fwhm_maj_arcmin'] not in ('', 'nan'))[0]; r['size_beams'] = ''
    expect.append(('a size over the frame BMAJ', r['cand_id'], 'kept:size_beams'))
    # size step (the size-on run only): one above 2.5 beams, one 2.0-2.5, both above 8 sigma
    sz = lambda r: abs(f(r['snr'])) > CFG.catalog_conform.SIZE_SNR and r['size_beams'] not in ('', 'nan')
    r = take(sz)[0]; r['size_beams'] = '2.7'; expect.append(('size 2.7 beams (size step on)', r['cand_id'], 'size:gaussfit size'))
    r = take(sz)[0]; r['size_beams'] = '2.2'; expect.append(('size 2.2 beams (size step on)', r['cand_id'], 'size:kept:size_flag'))
    # v110 label: a V row without its in-chain verdict
    r = take(lambda r: r['stokes'] == 'V' and r['fp_prediction'])[0]; r['fp_prediction'] = ''
    expect.append(('a V row without its v110 verdict', r['cand_id'], 'kept:fp_prediction'))
    # v110 marginal: a V final copied 0.3 deg away as a marginal veto (re-admitted, labelled)
    m = dict(next(x for x in lrows if x['stokes'] == 'V' and x['final'] == '1'))
    m.update(final='0', veto_reason='v110:marginal', dec=repr(f(m['dec']) + 0.3))
    lrows.append(m)
    expect.append(('a v110 marginal', f'VM_{m["lst"]}_{m["date"]}_{m["scope"]}_', 'kept:marginal'))
    return rows, lrows, out, expect


def fates(U, expect, size):
    """[(what, cand_id, wanted, got)] of the expectations that do not hold in the run of U."""
    rem = {}
    for r in _rows(os.path.join(U, 'master_catalog_removed.csv'))[1]:
        rem.setdefault(r['cand_id'], []).append(r['removal_reason'])
    kept = {}
    for r in _rows(os.path.join(U, 'master_catalog.csv'))[1]:
        kept.setdefault(r['cand_id'], r)
    bad = []
    for what, cid, want in expect:
        if want.startswith('size:'):
            if not size:
                continue
            want = want[5:]
        if want == 'kept:marginal':
            ok = any(k.startswith(cid) and r['fp_prediction'] == 'marginal' for k, r in kept.items())
            got = 'missing' if not ok else 'kept'
        elif want.startswith('kept:'):
            r = kept.get(cid)
            ok = bool(r and r.get(want[5:]) not in ('', None))
            got = (rem.get(cid) or ['?'])[0][:60] if not r else f'{want[5:]}={r.get(want[5:])!r}'
        else:
            ok = any(x.startswith(want) for x in rem.get(cid, []))
            got = rem.get(cid) or ('kept' if cid in kept else 'missing')
        if not ok:
            bad.append((what, cid, want, got))
    return bad


def sandbox(name, cols, rows, lcols, lrows, mut):
    """A run directory for one conform call. rows/lrows None: the run's own as-run rows (copied) and gate ledger (a
    symlink), and its whole frame trees (symlinks)."""
    U = os.path.join(T, name)
    os.makedirs(U)
    if rows is None:
        shutil.copy(os.path.join(RUN, 'master_catalog_asrun.csv'), os.path.join(U, 'master_catalog.csv'))
        os.symlink(os.path.join(RUN, 'v_deep_ledger.csv'), os.path.join(U, 'v_deep_ledger.csv'))
    else:
        _write(os.path.join(U, 'master_catalog.csv'), cols, rows)
        _write(os.path.join(U, 'v_deep_ledger.csv'), lcols, lrows)
    for c in CACHES:
        shutil.copy(os.path.join(RUN, c), os.path.join(U, c))
    tp = os.path.join(U, 'frame_times.csv')
    tc, tr = _rows(tp)
    tr = [r for r in tr if (r['stokes'], r['lst'], r['band'], r['date']) not in mut['times_drop']]
    for (st, l, b, d), (t0, t1) in mut['times_add'].items():
        tr.append(dict(stokes=st, lst=l, band=b, date=d, date_obs=t0, date_end=t1))
    _write(tp, tc, tr)
    for sub in ('v110_products', 'parent_catalogs'):
        os.symlink(os.path.join(RUN, sub), os.path.join(U, sub))
    if rows is None:
        for fr in ('iresid', 'vresid'):
            os.symlink(os.path.join(RUN, fr), os.path.join(U, fr))
        return U
    omit = {f'{st}/{l}_{b}/{d}_{st}_redo_resid_{b}.fits' for (st, l, b, d) in mut['omit']}
    for st, fr in (('I', 'iresid'), ('V', 'vresid')):
        for l, b in CELLS:
            src = os.path.join(RUN, fr, f'{l}_{b}')
            dst = os.path.join(U, fr, f'{l}_{b}')
            os.makedirs(dst)
            for p in sorted(glob.glob(os.path.join(src, '*.fits'))):
                if f'{st}/{l}_{b}/{os.path.basename(p)}' not in omit:
                    os.symlink(p, os.path.join(dst, os.path.basename(p)))
    return U


ARCHIVE = '/lustre/pipeline/exopipe/phase3'
# 2026-10-17: the I-noise mutation's night (08h 55MHz 2024-12-27, an epoch the search refused) kept its searched I frame in the
# archive until the leftover frames of refused nights were retired (Gregg 2026-10-17, Option 1; ledger clv15 8bb419a)
RETIRED = (f'{ARCHIVE}/Run_REFUSED_FRAME_QUARANTINE_20261017', '08h/2024-12-27/Run_20260909_000000/55MHz/I/deep')


def archive_overlay():
    """A read-only view of the deep archive in which the retired frame of the I-noise night is where the conform's target-beam
    lookup (gate_inputs.deep_target_beam, under P3_ARCHIVE_ROOT) found it before the retirement: every entry a symlink to the
    archive's own, except the directories on the retired frame's path, rebuilt with symlinks to their entries plus the frame."""
    O = os.path.join(T, 'archive_overlay')
    if os.path.isdir(O):
        return O
    qroot, rel = RETIRED
    parts = rel.split('/')
    src, dst = ARCHIVE, O
    for i in range(len(parts) + 1):
        os.makedirs(dst, exist_ok=True)
        nxt = parts[i] if i < len(parts) else None
        for e in (sorted(os.listdir(src)) if os.path.isdir(src) else []):
            if e != nxt:
                os.symlink(os.path.join(src, e), os.path.join(dst, e))
        if nxt is None:
            break
        src, dst = os.path.join(src, nxt), os.path.join(dst, nxt)
    for f in os.listdir(os.path.join(qroot, rel)):                  # the retired frames of that night, at their archive path
        if not os.path.lexists(os.path.join(dst, f)):
            os.symlink(os.path.join(qroot, rel, f), os.path.join(dst, f))
    return O


def pass_env(size):
    """The 2026-10-14 pass's environment for its conform call (fork_serve_p3s5_20261008.sh:31, p3s3_pass.sh:42), with the
    archive seen through archive_overlay (the I-noise night's retired frame in place)."""
    e = {k: v for k, v in os.environ.items() if not k.startswith(('P3_', 'P3S'))}
    e.update(P3S_LABEL='P3S5', P3_CADENCE='deep', P3_TABLE_ELEV='1', P3_PARENT_DIR=PAR, NUMPY_MADVISE_HUGEPAGE='0',
             P3_CONFORM_SIZE='1' if size else '0', P3_ARCHIVE_ROOT=archive_overlay())
    return e


def run_both(tag, cols, rows, lcols, lrows, mut, size):
    Ua = sandbox(f'{tag}_production', cols, rows, lcols, lrows, mut)
    Ub = sandbox(f'{tag}_port', cols, rows, lcols, lrows, mut)
    e = pass_env(size)
    pa = subprocess.run(['nice', PY, PROD, Ua], env=e, cwd=Ua, capture_output=True, text=True, timeout=7200)
    pb = subprocess.run(['nice', PY, '-c', _PORT, REPO, Ub, '1' if size else '0'], env=e, cwd=Ub, capture_output=True,
                        text=True, timeout=7200)
    la = [x for x in pa.stdout.splitlines() if x.startswith('conform:')]
    lb = [x for x in pb.stdout.splitlines() if x.startswith('conform:')]
    same = {n: filecmp.cmp(os.path.join(Ua, n), os.path.join(Ub, n), shallow=False)
            if os.path.exists(os.path.join(Ua, n)) and os.path.exists(os.path.join(Ub, n)) else None for n in OUTPUTS}
    check(f'{tag}: production and the port both finish', pa.returncode == 0 and pb.returncode == 0,
          (pa.returncode, pa.stderr[-500:], pb.returncode, pb.stderr[-800:]))
    check(f'{tag}: every file both write is the same, byte for byte ({", ".join(OUTPUTS)})', all(same.values()), same)
    check(f'{tag}: the same printed conform lines ({len(la)})', la == lb and la,
          [(a, b) for a, b in zip(la, lb) if a != b][:2] or (len(la), len(lb)))
    return Ua, la


def kinds(U):
    c = {}
    for r in _rows(os.path.join(U, 'master_catalog_removed.csv'))[1]:
        s = r['removal_reason']
        k = next((n for n, p in (('duplicate', 'duplicate'), ('excluded', 'excluded epoch'), ('elevation', 'elevation'),
                                  ('times unreadable', 'daytime gate: frame'), ('daytime', 'daytime:'), ('twilight', 'twilight'),
                                  ('jupiter-track', 'jupiter-track'), ('pulsar group', 'per-band per-Stokes table: table: multi'),
                                  ('table', 'per-band'), ('noise I', 'epoch refused: I'), ('noise V', 'epoch refused: V'),
                                  ('row-line', 'row-line'), ('unbacked', 'no final gate-ledger'), ('size', 'gaussfit size'),
                                  ('axis', 'axis ratio')) if s.startswith(p)), s[:30])
        c[k] = c.get(k, 0) + 1
    kept = _rows(os.path.join(U, 'master_catalog.csv'))[1]
    c['axis flag'] = sum(1 for r in kept if r['axis_flag'])
    c['size flag'] = sum(1 for r in kept if r['size_flag'])
    c['marginal re-admitted'] = sum(1 for r in kept if r['cand_id'].startswith('VM_'))
    c['pulsar-consistent kept'] = sum(1 for r in kept if 'pulsar-consistent' in (r.get('conform_note') or ''))
    return c


def test_subset():
    cols, rows, lcols, lrows = subset()
    nomut = dict(times_add={}, times_drop=set(), omit=set())
    Ub = sandbox('baseline', cols, [dict(r) for r in rows], lcols, lrows, nomut)
    pb = subprocess.run(['nice', PY, '-c', _PORT, REPO, Ub, '0'], env=pass_env(False), cwd=Ub, capture_output=True,
                        text=True, timeout=7200)
    check('the unchanged subset conforms (port)', pb.returncode == 0, pb.stderr[-600:])
    kept_ids = {r['cand_id'] for r in _rows(os.path.join(Ub, 'master_catalog.csv'))[1]}
    rows, lrows, mut, expect = mutate([dict(r) for r in rows], [dict(r) for r in lrows], kept_ids)
    print(f'  subset: {len(rows)} rows of {len(CELLS)} cells, {len(lrows)} ledger rows, {len(expect)} expectations')
    for size in (False, True):
        tag = 'subset_size' if size else 'subset'
        Ua, _ = run_both(tag, cols, rows, lcols, lrows, mut, size=size)
        k = kinds(Ua)
        bad = fates(Ua, expect, size)
        check(f'{tag}: every changed row meets its rule ({len(expect)} expectations); the rules that fired: '
              + json.dumps(k, sort_keys=True), not bad, bad[:6])
        if not size:
            want = ('elevation', 'daytime', 'twilight', 'jupiter-track', 'table', 'pulsar group', 'noise V',
                    'pulsar-consistent kept')
            check('the real rows exercise the rest: ' + ', '.join(want), all(k.get(w) for w in want),
                  [w for w in want if not k.get(w)])


def test_full():
    """Every row of Run_P3S5: its as-run rows and ledger as they are (the 2026-10-14 pass's last conform removed 17,231 of
    158,988 - today's inputs may differ: the data decisions, the masters)."""
    Ua, la = run_both('full', None, None, None, None, dict(times_add={}, times_drop=set(), omit=set()), size=False)
    print('  ' + (la[-1] if la else ''))
    print('  ' + json.dumps(kinds(Ua), sort_keys=True))


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        test_units()
        if '--full' in sys.argv:
            test_full()
        else:
            test_subset()
    finally:
        if os.environ.get('KEEP_T') != '1':                 # KEEP_T=1 keeps the sandboxes for a look
            shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
