#!/usr/bin/env python3
"""Tests of the deep search's recording of its removals (2026-10-17; the deep chat's removal inventory; Gregg: "It would obviously
be most expedient if all the information to do any additional removal/filtering was fully available to you *without* any redo of
the search!"): plain asserts, runnable with python, well under a minute.

    EXOPIPE3_SITE=/lustre/gh/main/exopipe_ops/exopipe3_site.py NUMPY_MADVISE_HUGEPAGE=0 \
        /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_search_record.py

1. parent_curve_table.table_inputs agrees with removals on every candidate, on the installed tables (the deep I and V tables, image-
   pixel metric; with and without elevations, so the elevation-band tables are exercised): removals cuts => margin < 1; margin < 1
   and not cut => the note is an exemption's; margin >= 1 => no cut and no note. A scaled filter (margin < k) is monotonic in k.
2. deep_search._append_locked refuses to append rows to a ledger written with another header (an older run resumed).
3. identification.run_identification_v(return_all=True) returns every candidate with its status; the default the PASS ones only.
4. The ledger content test accepts both headers (GLED_COLS_V1 of every run before 2026-10-17, GLED_COLS after) and flags a 'table'
   row whose recorded margin is >= 1.
Synthetic data in a fresh /tmp/exopipe3_tests_search_record/ (removed at the end).
"""
import csv
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
TMP = '/tmp/exopipe3_tests_search_record'

from exopipe.phase3.core import env as _env   # noqa: E402
_env.apply()
_env.legacy()
import numpy as np   # noqa: E402

FAILS = []
N_CHECKS = [0]


def check(what, ok, detail=''):
    N_CHECKS[0] += 1
    if not ok:
        FAILS.append(f'{what}: {detail}')
        print('FAIL ', what, detail)


def test_table_inputs():
    from exopipe.phase3.lib.search import parent_curve_table as P
    rng = np.random.default_rng(20261017)
    pc = (180.0, 37.0)
    # parents: bright and faint sources scattered within 30 deg of the phase centre (peaks in Jy)
    npar = 400
    pra = pc[0] + rng.uniform(-30, 30, npar) / np.cos(np.radians(pc[1]))
    pdec = np.clip(pc[1] + rng.uniform(-30, 30, npar), -89, 89)
    pfl = 10 ** rng.uniform(-0.5, 2.5, npar)
    par = np.stack([pra % 360, pdec, pfl], 1)
    # candidates: some near parents (inside and outside their zones), some far
    k = rng.integers(0, npar, 600)
    off = rng.uniform(0.0, 2.0, 600)                       # deg from a parent
    ang = rng.uniform(0, 2 * np.pi, 600)
    cra = (par[k, 0] + off * np.cos(ang) / np.cos(np.radians(par[k, 1]))) % 360
    cdec = np.clip(par[k, 1] + off * np.sin(ang), -89, 89)
    el = rng.uniform(25, 89, 600)
    beam = 0.3
    n_keys = 0
    for stokes in ('I', 'V'):
        for band in ('23MHz', '50MHz', '78MHz', 'Red', 'Full'):
            if not P.tables().get(f'deep_{stokes}_{band}'):
                continue
            n_keys += 1
            for elev in (None, el):
                for rms in (0.5, 0.05):
                    cut, note = P.removals(stokes, band, par, rms, beam, cra, cdec, elev=elev, phase_centre=pc)
                    ti = P.table_inputs(stokes, band, par, rms, beam, cra, cdec, elev=elev, phase_centre=pc)
                    m = np.array([np.inf if t['table_margin'] is None else t['table_margin'] for t in ti])
                    tag = f'{stokes} {band} elev={"yes" if elev is not None else "no"} rms={rms}'
                    bad_cut = [j for j in range(len(m)) if cut[j] and not m[j] < 1]
                    bad_exempt = [j for j in range(len(m)) if m[j] < 1 and not cut[j] and 'exempt' not in note[j]]
                    bad_pass = [j for j in range(len(m)) if m[j] >= 1 and (cut[j] or note[j])]
                    check(f'table_inputs agrees with removals ({tag}): cut -> margin < 1', not bad_cut, bad_cut[:3])
                    check(f'table_inputs agrees with removals ({tag}): margin < 1 and kept -> an exemption note', not bad_exempt,
                          [(m[j], note[j][:60]) for j in bad_exempt[:3]])
                    check(f'table_inputs agrees with removals ({tag}): margin >= 1 -> neither cut nor note', not bad_pass,
                          [(m[j], note[j][:60]) for j in bad_pass[:3]])
                    check(f'table_inputs: some candidates on each side of the table ({tag})',
                          (m < 1).any() and (m >= 1).any() if np.isfinite(m).any() else True, (int((m < 1).sum()), len(m)))
                    keys = {t['table_key'] for t in ti}
                    check(f'table_inputs: keys are the deep tables of {stokes} {band} ({tag})',
                          all(x.startswith(f'deep_{stokes}_{band}') for x in keys), sorted(keys)[:3])
                    fin = np.isfinite(m)
                    ok_fields = all(t['table_parent_S'] is not None and t['table_R'] and t['table_bin'] for t, f in zip(ti, fin) if f)
                    check(f'table_inputs: the binding bin is recorded wherever a margin is ({tag})', ok_fields)
                    check(f'table_inputs: the frame rms is recorded ({tag})', all(t['table_frame_rms_jy'] == rms for t in ti))
                    n1, n2 = int((m < 1.0).sum()), int((m < 1.5).sum())
                    check(f'table_inputs: a looser filter (k=1.5) removes a superset ({tag})', n2 >= n1, (n1, n2))
    check('table_inputs: deep tables found for I and V', n_keys >= 4, n_keys)
    ti = P.table_inputs('I', 'Red', np.zeros((0, 3)), 0.5, beam, cra[:3], cdec[:3], phase_centre=pc)
    check('table_inputs: no parents -> no margin', all(t['table_margin'] is None for t in ti), ti[:1])
    ti = P.table_inputs('I', 'Red', par, 0.0, beam, cra[:3], cdec[:3], phase_centre=pc)
    check('table_inputs: no frame rms -> no margin', all(t['table_margin'] is None and t['table_frame_rms_jy'] is None for t in ti))


def test_append_refuses_other_header():
    os.makedirs(TMP, exist_ok=True)              # a test function run alone (pytest) has no main() to make it
    from exopipe.phase3.lib.search import deep_search as E
    p = os.path.join(TMP, 'v_deep_ledger.csv')
    with open(p, 'w', newline='') as f:
        csv.writer(f).writerow(E.GLED_COLS_V1)
    try:
        E._append_locked(p, E.GLED_COLS, [[''] * len(E.GLED_COLS)])
        refused = False
    except RuntimeError as e:
        refused = 'REFUSING' in str(e)
    check('_append_locked refuses rows under another header (a pre-2026-10-17 ledger)', refused)
    check('... and the ledger is left as it was', open(p).read().strip() == ','.join(E.GLED_COLS_V1))
    q = os.path.join(TMP, 'v_deep_ledger_new.csv')
    E._append_locked(q, E.GLED_COLS, [['x'] * len(E.GLED_COLS)])
    E._append_locked(q, E.GLED_COLS, [['y'] * len(E.GLED_COLS)])
    rows = list(csv.reader(open(q)))
    check('_append_locked appends under its own header', rows[0] == E.GLED_COLS and len(rows) == 3, len(rows))
    row = E._gled_row('05h', '78MHz', 'V', '2025-01-01', dict(ra=1.0, dec=2.0, snr=4.2, flux=0.1, table_margin=0.81,
                                                               table_key='deep_V_78MHz', parent_jy=12.5), 1, reason='table')
    check('_gled_row: one value per GLED_COLS column', len(row) == len(E.GLED_COLS), (len(row), len(E.GLED_COLS)))
    d = dict(zip(E.GLED_COLS, row))
    check('_gled_row: the table inputs land in their columns', d['table_margin'] == 0.81 and d['table_key'] == 'deep_V_78MHz'
          and d['parent_jy'] == 12.5 and d['table_R'] == '', {k: d[k] for k in E.GLED_COLS_TABLE})


def test_v_return_all():
    from exopipe.phase3.lib.search import identification as ID, search_v as sv2
    rng = np.random.default_rng(7)
    img = rng.normal(0, 1.0, (400, 400)).astype(np.float32)
    cands = [dict(pix_x=100, pix_y=100, flux=12.0, snr=12.0, status='PASS', reject_reason=''),
             dict(pix_x=300, pix_y=300, flux=3.0, snr=6.0, status='PASS', reject_reason=''),
             dict(pix_x=200, pix_y=120, flux=-9.0, snr=-9.0, status='PASS', reject_reason='')]
    saved = (sv2.make_masks, sv2.get_uniform_rms, sv2.extract_v_peaks)
    no = np.zeros(img.shape, bool)
    sv2.make_masks = lambda hdr, shape, zenith_time=None: (no, no, no, None, None, None)
    sv2.get_uniform_rms = lambda data, mask: 1.0
    sv2.extract_v_peaks = lambda *a, **k: [dict(c) for c in cands]
    try:
        import astropy.io.fits as fits
        h = fits.Header({'NAXIS': 2, 'NAXIS1': 400, 'NAXIS2': 400, 'CTYPE1': 'RA---SIN', 'CTYPE2': 'DEC--SIN',
                         'CRVAL1': 180.0, 'CRVAL2': 37.0, 'CRPIX1': 200.5, 'CRPIX2': 200.5, 'CDELT1': -0.05, 'CDELT2': 0.05})
        allv = ID.run_identification_v(img, img, h, 3.0, None, sigma_thresh=5.0, return_all=True)
        passv = ID.run_identification_v(img, img, h, 3.0, None, sigma_thresh=5.0)
    finally:
        sv2.make_masks, sv2.get_uniform_rms, sv2.extract_v_peaks = saved
    st = sorted((c['pix_x'], c['status']) for c in allv)
    check('run_identification_v(return_all=True): every candidate with its status', len(allv) == 3
          and ('REJECT_LOCAL_RMS' in {c['status'] for c in allv}), st)
    check('run_identification_v(): the PASS candidates only, as before', [c['status'] for c in passv] == ['PASS'] * len(passv)
          and len(passv) == sum(c['status'] == 'PASS' for c in allv), (len(passv), st))


def test_content_test_headers():
    os.makedirs(TMP, exist_ok=True)              # a test function run alone (pytest) has no main() to make it
    from exopipe.phase3.lib.search import tests as T
    for cols, name in ((T.GLED_COLS_V1, 'v1'), (T.GLED_COLS, 'v2')):
        d = os.path.join(TMP, f'ledger_{name}')
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, 'v_deep_ledger.csv')
        rows = [dict(dict.fromkeys(cols, ''), lst='05h', scope='78MHz', stokes='V', date='2025-01-01', ra='1.0', dec='2.0',
                     snr='9.0', veto_reason='table', final='0', mstruct='1.0')]
        if name == 'v2':
            rows[0]['table_margin'] = '0.7'
            rows.append(dict(rows[0], ra='3.0', table_margin='1.3'))
        with open(p, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        ok, n, msg = T.complete.__wrapped__(p) if hasattr(T.complete, '__wrapped__') else T.complete(p)[:3]
        if name == 'v1':
            check('ledger content test: the pre-2026-10-17 header is accepted', 'header' not in str(msg), msg)
        else:
            check("ledger content test: a 'table' row with margin >= 1 is flagged", 'table_margin 1.3' in str(msg), msg)
            check("ledger content test: the 'table' row with margin < 1 is not", 'table_margin 0.7' not in str(msg), msg)


def test_refusal_columns():
    os.makedirs(TMP, exist_ok=True)              # a test function run alone (pytest) has no main() to make it
    import re
    from exopipe.phase3.lib.search import deep_search as E
    src = open(E.__file__).read()
    written = set()
    for suf in re.findall(r"_refdir\}/\{lst\}_\{scope\}(_[a-zA-Z_{}]+_excluded\.csv)", src):
        written |= {suf.replace('{st}', 'I'), suf.replace('{st}', 'V')} if '{st}' in suf else {suf}
    named = {x for x, _ in E.REFUSAL_COLUMNS}
    check('refusals/COLUMNS.csv names every refusal file deep_search writes', written <= named and len(written) >= 11,
          sorted(written - named))
    d = os.path.join(TMP, 'refusals')
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    E._refusal_columns(d)
    rows = list(csv.reader(open(os.path.join(d, 'COLUMNS.csv'))))
    check('COLUMNS.csv: a header and one row per file kind', rows[0] == ['file_suffix', 'columns']
          and len(rows) == 1 + len(E.REFUSAL_COLUMNS), len(rows))
    open(os.path.join(d, 'COLUMNS.csv'), 'a').write('kept,as is\n')
    E._refusal_columns(d)
    check('COLUMNS.csv: an existing one is left as it is', open(os.path.join(d, 'COLUMNS.csv')).read().endswith('kept,as is\n'))


def main():
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)
    try:
        test_table_inputs()
        test_append_refuses_other_header()
        test_v_return_all()
        test_content_test_headers()
        test_refusal_columns()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f'{N_CHECKS[0] - len(FAILS)} of {N_CHECKS[0]} checks passed')
    return 1 if FAILS else 0


def test_all():
    assert main() == 0


if __name__ == '__main__':
    sys.exit(main())
