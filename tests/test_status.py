"""Standing test of `exopipe3 status` (plain asserts, no pytest):

    NUMPY_MADVISE_HUGEPAGE=0 EXOPIPE3_SITE=/lustre/gh/main/exopipe_ops/exopipe3_site.py \
        /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_status.py

Synthetic data in a fresh directory under /tmp/exopipe3_tests_status/ (removed at the end):
- a synthetic survey WORLD (data_decisions.csv + dated backups, processing ledger, deep working tree with masters /
  weights.csv / residuals, Phase 1/2 trees with INTTIME images, archive generation, hub registry, deep / sub-LST run dirs,
  snapshot fleet state, QA directory). Protected config keys (paths.DATA_DECISIONS, status.QA_DIR, status.LEDGER,
  serve.HUB_DIR, ...) are pointed at it directly for the test and restored afterwards.
- FAKE ENGINES for every production tool the step runs in place: a fake exopipe_ops root (site LEGACY_CODE_ROOTS for the
  test; first on sys.path for the in-process imports) holds stand-ins of conformance_check, build_qa_pages,
  build_data_processing_page, build_ratified_register, _inventory_p1p2 / _build_ledger, solar_contamination_audit,
  candidate_provenance_audit / perframe_solar_check (in process: attr / remap), build_qa_table, beam_audit,
  qa_noise_vs_freq_plot, bandpass_phase_warp, coadd_exclusion_conformance, snapshot_batch_audit, snapshot_batch_lists
  (children through core.legacy.run), p3s3_sources / p3s3_stale_exclusions / snapshot_frametime_worklist (imports). Each
  writes what the production tool writes, to the production tool's LITERAL path where it has one, so the step's path
  redirection, write guard, records and content tests run end to end. phase_status.py (branch copy, identical bytes)
  and snapshot_serve_exclusion_gate.keys_of run for real.
- every stage of every action, the gates S-G1, S-G2, S-G3, S-G5, S-G6, S-G7, S-G8, S-G10, S-G11, S-G12, S-G14 and the
  tier checks, item lists usable as --cells @FILE, and NEGATIVE tests: every content test fails on a corrupted copy.
Read only on production: content tests of the installed solar audit, the membership conformance table and its page, the
bandpass phase QA table and a snapshot eligibility table (+ each fails on a corrupted /tmp copy); one --dry-run on a real
deep cell (writes nothing); `status ledger --dry-run` against the production QA directory (S-G2 in the plan); the session
summary on real data under --parity (no write refused: nothing outside --out).
"""
import calendar
import contextlib
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from exopipe.phase3 import cli                        # noqa: E402
from exopipe.phase3.config import CFG                 # noqa: E402
from exopipe.phase3.core import env, validate, record  # noqa: E402

env.legacy()
REAL_OPS = env.legacy_root('exopipe_ops')            # before the test points the site's legacy roots at the fakes
BASE = '/tmp/exopipe3_tests_status'
os.makedirs(BASE, exist_ok=True)
TMP = tempfile.mkdtemp(prefix='run_', dir=BASE)
W = os.path.join(TMP, 'world')
FAKE = os.path.join(TMP, 'fake', 'exopipe_ops')
T0 = time.time()
RESULTS = []
NEG_TESTS = set()      # (class, test name or the tuple of tests that failed) exercised by a negative test
SUBS = ['18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz', '64MHz', '69MHz',
        '73MHz', '78MHz', '82MHz']
CELLS = '05h_18MHz,05h_Full'


def step(argv, quiet=True):
    """Run `exopipe3 <argv>` in this process; returns (rc, stdout+stderr text)."""
    CFG.reset()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            rc = cli.main(list(argv))
        except SystemExit as e:                      # argparse (--help, usage errors) and SystemExit errors of the step
            rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
            if not isinstance(e.code, int) and e.code is not None:
                buf.write(str(e.code) + '\n')
    txt = buf.getvalue()
    if not quiet:
        print(txt)
    return rc, txt


def check(name, cond, detail=''):
    RESULTS.append((name, bool(cond)))
    print(f"{'ok  ' if cond else 'FAIL'} {name}" + (f'  [{str(detail)[-900:]}]' if detail and not cond else ''))
    assert cond, f'{name}: {detail}'


def rec(out, unit):
    p = os.path.join(out, '_records', 'status', f'{unit}.json')
    return json.load(open(p)) if os.path.exists(p) else {}


def rows(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def wcsv(p, cols, rs):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rs:
            w.writerow(r)


def wtxt(p, s):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        f.write(s)


def sha(p):
    return hashlib.sha1(open(p, 'rb').read()).hexdigest()


def failed(cls, path, **ctx):
    """[names of failing tests] of class ``cls`` on ``path``."""
    return [r['name'] + ': ' + str(r['detail'])[:200] for r in validate.run(cls, path, **ctx) if not r['passed']]


def neg(name, cls, src, mutate, ctx=None, siblings=(), subdir='', expect=None):
    """A copy of ``src`` (placed in ``subdir``; sibling files copied unchanged, at paths relative to the copy root) passes
    every test of ``cls`` (control), and fails one once ``mutate`` corrupted it (the test named ``expect``, if given)."""
    d = os.path.join(TMP, 'neg', name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, subdir))
    dst = os.path.join(d, subdir, os.path.basename(src))
    (shutil.copytree if os.path.isdir(src) else shutil.copyfile)(src, dst)
    for s in siblings:
        rel, path = s if isinstance(s, tuple) else (os.path.join(subdir, os.path.basename(s)), s)
        os.makedirs(os.path.dirname(os.path.join(d, rel)), exist_ok=True)
        (shutil.copytree if os.path.isdir(path) else shutil.copyfile)(path, os.path.join(d, rel))
    ok = failed(cls, dst, **(ctx or {}))
    mutate(dst)
    bad = failed(cls, dst, **(ctx or {}))
    names = [b.split(':')[0].split('.')[-1] for b in bad]
    hit = bad if expect is None else [b for b, n in zip(bad, names) if n == expect]
    NEG_TESTS.update((cls, n) for n in ([expect] if expect else names))
    check(f'negative: {name} fails {cls}{" " + expect if expect else ""} (its unmodified copy passes)', hit and not ok,
          f'control failures: {ok}' if ok else f'the corrupted copy passed {"the named test" if expect else "every test"} '
                                               f'(failures: {bad})')


def drop_last_line(p):
    lines = open(p).readlines()
    wtxt(p, ''.join(lines[:-1]))


def edit(old, new, count=1):
    def m(p):
        s = open(p).read()
        assert old in s, (old, p)
        open(p, 'w').write(s.replace(old, new, count))
    return m


# =====================================================================================================================
# the synthetic world
# =====================================================================================================================

NIGHTS = [('05h', '2099-01-01'), ('05h', '2099-01-02'), ('05h', '2099-01-03'), ('05h', '2099-01-04'), ('06h', '2099-01-01')]
LEDGER_COLS = ['lst', 'date', 'p1', 'p1_nsub', 'p1_peel', 'p1_npeel', 'p2', 'p2_nsub', 'p2_nwide', 'p2_state',
               'dd_excl_subbands', 'dd_status', 'dd_reason', 'qa_verdict', 'qa_basis', 'p1_missing', 'p2_missing']


def ledger_rows(extra_night=False):
    """Ledger rows consistent with _build_ledger.py's rule over the synthetic data_decisions."""
    spec = {('05h', '2099-01-01'): (0, 'none', '', 'needs-QA', 'unverified (no exclusion, no positive QA)', 'none', 0, 'complete'),
            ('05h', '2099-01-02'): (1, 'partial-excluded', 'peeling divergence', 'needs-QA', 'partial subband exclusion',
                                    'none', 0, 'complete'),
            ('05h', '2099-01-03'): (15, 'all-excluded', 'bad bandpass', 'exclude', 'bad bandpass', 'full', 15, 'complete'),
            ('05h', '2099-01-04'): (0, 'none', '', 'needs-QA', 'unverified (no exclusion, no positive QA)', 'none', 0, 'complete'),
            ('06h', '2099-01-01'): (0, 'none', '', 'incomplete', 'awaiting Phase 2', 'none', 0, 'none')}
    if extra_night:
        spec[('06h', '2099-01-02')] = (0, 'none', '', 'incomplete', 'awaiting Phase 2', 'none', 0, 'none')
    out = []
    for (l, d), (ne, st, rsn, qv, qb, peel, npeel, p2) in sorted(spec.items()):
        out.append(dict(lst=l, date=d, p1='complete', p1_nsub=15, p1_peel=peel, p1_npeel=npeel, p2=p2,
                        p2_nsub=15 if p2 == 'complete' else 0, p2_nwide=4 if p2 == 'complete' else 0, p2_state='',
                        dd_excl_subbands=ne, dd_status=st, dd_reason=rsn, qa_verdict=qv, qa_basis=qb, p1_missing='',
                        p2_missing=''))
    return out


def fits_header_file(path, cards=(), history=()):
    from astropy.io import fits
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h = fits.Header()
    for k, v in cards:
        h[k] = v
    for t in history:
        h['HISTORY'] = t
    fits.PrimaryHDU(header=h).writeto(path, overwrite=True)


def build_world():
    import numpy as np                                                      # noqa: F401
    dd_cols = ['lst', 'subband', 'date', 'verdict', 'reason', 'score_thermal_noise', 'score_flux_scale', 'score_ionosphere']
    allx = [dict(lst='05h', subband=s, date='2099-01-03', verdict='exclude', reason='bad bandpass') for s in SUBS]
    # dated backup (BREACH vs STALE history): its date and the masters' mtimes are in the past (a later mtime is clamped at
    # 2038-01-19 by 32-bit file systems; a future one would hide the masters re-read); NIGHTS are 2099 dates (never real)
    wcsv(f'{W}/clv15/data_decisions.csv.bak_20250101', dd_cols, allx)
    wcsv(f'{W}/clv15/data_decisions.csv', dd_cols, allx + [
        dict(lst='05h', subband='18MHz', date='2099-01-02', verdict='exclude', reason='peeling divergence'),
        dict(lst='05h', subband='46MHz', date='2099-01-04', verdict='keep', reason='readmitted'),
        dict(lst='06h', subband='46MHz', date='2099-01-05', verdict='keep', reason='readmitted (date-outlier rule test)')])
    wcsv(f'{W}/clv15/processing_ledger.csv', LEDGER_COLS, ledger_rows())
    # the newest production re-adjudication list (the first summary's baseline): 05h 2099-01-03 has advanced since
    rcols = list(CFG.status_report.READJUDICATION_COLUMNS)
    wcsv(f'{W}/QA/readjudication_list_20990101.csv', rcols, [
        dict(lst='05h', date='2099-01-03', dd_status='all-excluded', excluded_subbands=15, reason='bad bandpass',
             p1_peel='none', p2='partial', go_no_go=CFG.status_report.READJUDICATION_GO),
        dict(lst='05h', date='2099-01-02', dd_status='partial-excluded', excluded_subbands=1, reason='peeling divergence',
             p1_peel='none', p2='complete', go_no_go=CFG.status_report.READJUDICATION_GO)])
    wtxt(f'{W}/fast/.data_decisions.fingerprint', '0.0\n')
    # deep working tree (5 cells with residuals_provisional for the outlier rule; 2 cells with masters and records)
    sci = 'Science_20990101_000000'
    for l, d in NIGHTS[:4]:
        for sc in ('18MHz', 'Full'):
            name = (f'{sc}-I-Taper-10s-Robust-0.75-x-image_dewarped.fits' if sc.endswith('MHz')
                    else f'{sc}_I_deep_Taper_Robust-0.75_dewarped_x_LST{l}.fits')
            fits_header_file(f'{W}/phase2/{l}/{d}/{sci}/{sc}/I/deep/{name}', [('INTTIME', 1000.0)])
    os.makedirs(f'{W}/phase1/05h/2099-01-01/Run_20990102_000000', exist_ok=True)     # Phase 1 newer than Phase 2
    when = calendar.timegm((2025, 6, 1, 0, 0, 0))
    for cell, cons, op in (('05h_18MHz', ('2099-01-01', '2099-01-02'), None),
                           ('05h_Full', ('2099-01-01', '2099-01-03'), 'ivw-mean-v3')):
        sc = cell.split('_')[1]
        m = f'{W}/coadd_sb/{cell}/coadd_final/I_deep_Taper_R075_{sc}.fits'
        fits_header_file(m, [('INTTIME', 2000.0), ('NCOADD', 2)] + ([('COADDOP', op)] if op else []),
                         [f'COADD constituent {d}' for d in cons])
        os.utime(m, (when, when))
    # a .done marker beside the master WITHOUT COADDOP: never counted as a master (only the file's COADDOP card is)
    wtxt(f'{W}/coadd_sb/05h_18MHz/coadd_final/.done', '')
    wcsv(f'{W}/coadd_sb/05h_18MHz/coadd_final/weights.csv', ['date', 'sci_dir', 'in_master'],
         [dict(date='2099-01-01', sci_dir=f'{W}/phase2/05h/2099-01-01/{sci}', in_master=1)])
    wcsv(f'{W}/coadd_sb/05h_Full/coadd_final/weights.csv', ['date', 'sci_dir', 'in_master'],
         [dict(date=d, sci_dir=f'{W}/phase2/05h/{d}/{sci}', in_master=im) for d, im in
          (('2099-01-01', 1), ('2099-01-02', 0), ('2099-01-04', 1))])
    for cell, dates in (('05h_18MHz', ('2099-01-01',)), ('05h_Full', ('2099-01-01', '2099-01-04'))):
        for d in dates:
            wtxt(f'{W}/coadd_sb/{cell}/subtracted/{d}_I_sub_{cell.split("_")[1]}.fits', '')
    wtxt(f'{W}/coadd_sb/05h_Full/VIOLATION', 'flux scale not baked\n')               # for the fake conformance check
    for cell, rej in (('05h_18MHz', True), ('05h_Full', True), ('06h_18MHz', True), ('06h_Full', False),
                      ('07h_Red', True)):
        wcsv(f'{W}/coadd_sb/{cell}/residuals_provisional.csv', ['date', 'outlier'],
             [dict(date='2099-01-05', outlier=str(rej)), dict(date='2099-01-01', outlier='False')])
    # archive generation: 05h 2099-01-01 18 MHz fully archived (census 'ok')
    a = f'{W}/arch/05h/2099-01-01/Run_TESTARCH/18MHz'
    for p in (f'{a}/I/deep/e.fits', f'{a}/I/deep/e_msub.fits', f'{a}/V/deep/e_msub.fits'):
        wtxt(p, '')
    os.makedirs(f'{W}/arch_coadd', exist_ok=True)
    # QA directory
    wcsv(f'{W}/QA/membership_conformance_latest.csv', list(CFG.status_conformance.COLUMNS), [
        dict(cell='05h_18MHz', status='PASS', n_violations=0, in_rebuild_list=0, first_violation='',
             checked_utc='2099-01-05T00:00:00Z'),
        dict(cell='05h_Full', status='FAIL', n_violations=2, in_rebuild_list=1, first_violation='flux scale not baked',
             checked_utc='2099-01-05T00:00:00Z')])
    wcsv(f'{W}/QA/solar_contamination_audit.csv', SOLAR_COLS, SOLAR_ROWS[:2])     # the "installed" table
    os.makedirs(f'{W}/qa_assembly', exist_ok=True)
    for i in range(5):                                                              # calibration tables (bandpass)
        os.makedirs(f'{W}/cal/{i}/x/successful/r{i}/tables/calibration_2099-01-0{i + 1}_05h.B.flagged', exist_ok=True)
    # hub registry: deep run Run_TESTA, sub-LST cumulative Run_SLSTB (LSTs 06h), snapshot none
    json.dump({'V': {'1-hour': {'run': 'Run_TESTA', 'port': 18766, 'served': 4},
                     '10-min': {'run': 'Run_SLSTB', 'port': 18787, 'generation': 'fitted positions; LSTs: 06h'}},
               'I': {'1-hour': {'run': 'Run_TESTA', 'port': 18768, 'served': 3}}},
              open(_mk(f'{W}/hub/registry.json'), 'w'))
    # deep runs A (served) and B; cand_id = <lst>_<date>_<band>_<S>_<tier>_<ts>
    c1, c2, c3 = '05h_2099-01-01_18MHz_I_sb_1', '05h_2099-01-01_Full_I_wb_2', '05h_2099-01-04_Red_I_wb_3'
    c4, c5 = '05h_2099-01-04_32MHz_I_sb_4', '05h_2099-01-03_Full_V_wb_5'          # c5: on an excluded cell-band
    A, B = f'{W}/transients/Run_TESTA', f'{W}/transients/Run_TESTB'
    wcsv(f'{A}/selected_all.csv', ['cand_id', 'stokes', 'lst', 'date', 'band'],
         [dict(cand_id=c, stokes=c.split('_')[3], lst='05h', date=c.split('_')[1], band=c.split('_')[2])
          for c in (c1, c2, c3, c5)])
    wcsv(f'{A}/source_catalogs/pages_input_I_sb.csv', ['stokes', 'lst', 'date', 'band'],
         [dict(stokes='I', lst='05h', date='2099-01-01', band='18MHz')])
    wcsv(f'{A}/master_catalog.csv', ['cand_id'], [dict(cand_id=c) for c in (c1, c2, c3, c5)])
    wcsv(f'{A}/source_catalogs_I/t_sources_survivors.csv', ['source_id', 'identity_id'],
         [dict(source_id='s1', identity_id='J1'), dict(source_id='s2', identity_id='J2')])
    wcsv(f'{A}/source_catalogs_I/t_source_members.csv', ['source_id', 'cand_id'],
         [dict(source_id='s1', cand_id=c1), dict(source_id='s1', cand_id=c2), dict(source_id='s2', cand_id=c3)])
    wtxt(f'{A}/judgements_I/judgements.jsonl', json.dumps({'action': 'classify', 'source_id': 'J1', 'label': 'real'}) + '\n')
    wcsv(f'{A}/judgements/skipped.csv', ['cand_id'], [dict(cand_id=c5)])          # judgements/ is never audited
    wcsv(f'{B}/master_catalog.csv', ['cand_id'], [dict(cand_id=c) for c in (c1, c2, c4)])
    wcsv(f'{B}/source_catalogs_I/t_sources_survivors.csv', ['source_id', 'identity_id'],
         [dict(source_id='s1b', identity_id='J1'), dict(source_id='s3', identity_id='J3')])
    wcsv(f'{B}/source_catalogs_I/t_source_members.csv', ['source_id', 'cand_id'],
         [dict(source_id='s1b', cand_id=c1), dict(source_id='s3', cand_id=c4)])
    os.makedirs(f'{W}/transients/Run_EMPTY', exist_ok=True)
    # sub-LST: 05h published before fitted positions (no pos_kind), 06h current
    for hh, cols in (('05', 'cand_id,ra,dec'), ('06', 'cand_id,ra,dec,pos_kind')):
        wtxt(f'{W}/transients/Run_SLST{hh}B/master_catalog.csv', cols + '\n')
        wtxt(f'{W}/transients/Run_SLST{hh}B/pertie_detections.csv', 'cand_id\n')
    wtxt(f'{W}/transients/Run_SLSTB/master_catalog.csv', 'cand_id,lst,date,band\n05h_2099-01-03_Full_I_10m_1,05h,2099-01-03,Full\n')
    wtxt(f'{W}/sublst/tenmin_root_x2/05h_18MHz/frames.csv', 'f\n')
    for S in ('V', 'I'):
        wtxt(f'{W}/ops/sublst_search_state_B/done/05h_18MHz:{S}', '')
        wtxt(f'{W}/sublst/search_B/05h_18MHz_{S}/05h_18MHz.log',            # the search RECORD (shipped log)
             f'05h 18MHz: 3 epochs -> 2 candidates\nSUBLST-SEARCH-DONE 05h_18MHz {S}\n')
    wtxt(f'{W}/ops/sublst_search_state_B/done/05h_23MHz:V', '')            # a done marker without a search record
    wtxt(f'{W}/sublst/search_B/05h_27MHz_I/05h_27MHz.log', 'Traceback (most recent call last):\n  boom\n'
         'SUBLST-SEARCH-DONE 05h_27MHz I\n')                                # a log that fails the acceptance rule
    # snapshot fleet
    wtxt(f'{W}/ops/snapshot_fleet_worklist_v9_20261005.txt', '18h 2099-02-01\n18h 2099-02-02\n')
    wtxt(f'{W}/ops/snapshot_fleet_claims/18h_2099-02-01/done', 'synthetic\n')
    wtxt(f'{W}/fleet/18h_2099-02-01/STALE_MARK', '')
    for n in ('admission_18MHz.csv', 'runinfo_18MHz_I_median.json', 'nightrefused_18MHz_V_median.json'):
        wtxt(f'{W}/fleet/18h_2099-02-01/{n}', '{}\n')                       # searched: runinfo OR night refusal per Stokes
    os.makedirs(f'{W}/fleet/18h_2099-02-03', exist_ok=True)
    for n in ('admission_18MHz.csv', 'runinfo_18MHz_I_median.json'):        # V unaccounted: not searched
        wtxt(f'{W}/fleet/18h_2099-02-03/{n}', '{}\n')
    wtxt(f'{W}/ops/snapshot_fleet_claims/18h_2099-02-03/done', 'synthetic\n')   # a claim 'done' without the records
    wtxt(f'{W}/snap/order.txt', '18h 2099-02-01\n18h 2099-02-02\n18h 2099-02-03\n18h 2099-02-04\n')
    wtxt(f'{W}/snap/base.txt', '18h 2099-02-01\n')
    wtxt(f'{W}/snap/cells.txt', '18h_2099-02-01\n18h_2099-02-02\n18h_2099-02-03\n')
    # published deep attempts (fake p3s3_sources) and their searched images
    att = []
    for cell, S, dates in (('05h_18MHz', 'V', ('2099-01-01', '2099-01-02')), ('05h_Full', 'I', ('2099-01-01',))):
        dd = f'{W}/att/{cell}_{S}'
        wcsv(f'{dd}/transients/Run_P3S5/v_deep_images.csv', ['date', 'lst', 'scope', 'stokes'],
             [dict(date=d, lst='05h', scope=cell.split('_')[1], stokes=S) for d in dates])
        att.append(dict(cell=cell, stokes=S, harvested=True, dir=dd, node='n1', att=1))
    json.dump(att, open(_mk(f'{W}/attempts.json'), 'w'))
    # audit slices of the register
    rc = list(CFG.status_pages.REGISTER_COLUMNS)[:-1]
    wcsv(f'{W}/audit/audit_A.csv', rc, [dict(ruling_id='R-B', date='2099-01-02', status='enforced'),
                                       dict(ruling_id='R-A', date='2099-01-01', status='not-enforced')])
    wcsv(f'{W}/audit/audit_B.csv', rc, [dict(ruling_id='R-C', date='2099-01-01', status='SUPERSEDED')])
    # run records for --stale record
    R = f'{TMP}/recs/_records/frames'
    inp = _mk(f'{TMP}/recs/in.txt')
    wtxt(inp, 'input v1\n')
    outp = _mk(f'{TMP}/recs/out.txt')
    wtxt(outp, 'output\n')
    base = dict(step='frames', config_sections=['instrument'], config_hash=CFG.hash(['instrument']), code={},
                outputs=[{'path': outp, 'class': 'csv_table'}], inputs={inp: record.fingerprint(inp)})
    recs = {'05h_18MHz': dict(base, status='done', commit='abc1234', admitted=['2099-01-01', '2099-01-02']),
            '05h_Full': dict(base, status='done', commit='abc1234', inputs={inp: {'size': 1, 'mtime_ns': 1, 'sha1': 'x'}}),
            '05h_Red': dict(base, status='done', commit='abc1234', admitted=['2099-01-01']),
            '05h_Green': dict(base, status='done', commit='abc1234-dirty'),
            '05h_Blue': dict(base, status='refused', commit='abc1234', reason='gate X')}
    for u, r in recs.items():
        json.dump(dict(r, unit=u), open(_mk(f'{R}/{u}.json'), 'w'))
    os.makedirs(f'{TMP}/empty_reports', exist_ok=True)
    os.makedirs(f'{TMP}/jobs', exist_ok=True)


def _mk(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


SOLAR_COLS = ['lst', 'date', 'subband', 'R_f', 'R_46', 'S_f', 'sun_el_deg', 'verdict', 'cause']
SOLAR_ROWS = [dict(lst='05h', date='2099-01-01', subband='18MHz', R_f='1.000', R_46='1.000', S_f='1.000', sun_el_deg='-30.0',
                   verdict='ok', cause='not-solar'),
              dict(lst='05h', date='2099-01-01', subband='23MHz', R_f='4.000', R_46='1.000', S_f='4.000', sun_el_deg='-15.0',
                   verdict='suspect', cause='twilight'),
              dict(lst='05h', date='2099-01-02', subband='27MHz', R_f='12.000', R_46='1.000', S_f='12.000',
                   sun_el_deg='-5.0', verdict='QUARANTINE', cause='solar')]

# ---------------------------------------------------------------------------------------------------------------------
# fake engines (each mimics its production tool's interface; @W@ = the world root)
# ---------------------------------------------------------------------------------------------------------------------
FAKES = {
    'conformance_check.py': '''
import csv, glob, os, time
ROOT = '/nonexistent/production/coadd_sb'; QA = '/nonexistent/QA'
LATEST = QA + '/membership_conformance_latest.csv'; HIST = QA + '/membership_conformance_history.csv'
PAGE = '/nonexistent/hub/conformance.html'; REBUILD = '/nonexistent/rebuild_list.txt'
def main():
    t = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    reb = {l.strip() for l in open(REBUILD) if l.strip()} if os.path.exists(REBUILD) else set()
    rows = []
    for cd in sorted(glob.glob(ROOT + '/[0-2][0-9]h_*')):
        if not os.path.exists(cd + '/coadd_final/weights.csv'):
            continue
        cell = os.path.basename(cd)
        v = [open(cd + '/VIOLATION').read().strip()] * 2 if os.path.exists(cd + '/VIOLATION') else []
        rows.append(dict(cell=cell, status='PASS' if not v else 'FAIL', n_violations=len(v), in_rebuild_list=int(cell in reb),
                         first_violation=(v[0] if v else '')[:300], checked_utc=t))
    with open(LATEST + '.new', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    os.replace(LATEST + '.new', LATEST)
    npass = sum(r['status'] == 'PASS' for r in rows)
    new = not os.path.exists(HIST)
    with open(HIST, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['checked_utc', 'cells', 'pass', 'fail', 'fail_not_in_rebuild_list'])
        w.writerow([t, len(rows), npass, len(rows) - npass, 0])
    open(PAGE, 'w').write('<html><h2>%d of %d cells PASS</h2></html>' % (npass, len(rows)))
    print('CONFORMANCE %s: %d/%d PASS' % (t, npass, len(rows)))
''',
    'build_qa_pages.py': '''
import csv, os, sys
HUB = '/nonexistent/hub'; LEDGER = '/nonexistent/ledger.csv'; DD = ''; ROOT = ''; QA = ''
def qa3_window(band):
    from exopipe.phase3.lib.masters import membership as M
    lo, hi = M.qa3_window(band)
    return (lo + 0.01, hi) if os.environ.get('FAKE_QA3_DRIFT') else (lo, hi)
def _nights():
    return [(r['lst'], r['date']) for r in csv.DictReader(open(LEDGER))]
def main():
    only = set(sys.argv[sys.argv.index('--only') + 1:]) if '--only' in sys.argv else None
    os.makedirs(HUB + '/nights', exist_ok=True); os.makedirs(HUB + '/dates', exist_ok=True)
    n = 0
    for l, d in _nights():
        if only and l + '_' + d not in only:
            continue
        open('%s/dates/%s.html' % (HUB, d), 'w').write('<html>date %s</html>' % d)
        open('%s/nights/%s_%s.html.part' % (HUB, l, d), 'w').write('<html><a href="../dates/%s.html">d</a></html>' % d)
        os.replace('%s/nights/%s_%s.html.part' % (HUB, l, d), '%s/nights/%s_%s.html' % (HUB, l, d))
        n += 1
    open(HUB + '/qa_overview_fragment.html', 'w').write('<p>fragment</p>')
    print('QA pages: %d night pages' % n)
def check():
    miss = ['%s_%s' % k for k in _nights() if not os.path.exists('%s/nights/%s_%s.html' % (HUB, k[0], k[1]))]
    print('CHECK: %d ledger nights, %d without a page' % (len(_nights()), len(miss)))
    print('CHECK PASS' if not miss else 'CHECK FAIL')
    return 0 if not miss else 1
''',
    'build_data_processing_page.py': '''
import csv, os
OUT = '/nonexistent/hub/data_processing.html'; LEDGER = ''; DD = ''; ROOT = ''; QA = ''
def main():
    frag = open('@W@/hub/qa_overview_fragment.html').read()          # production reads the hub fragment by literal path
    links = ''.join('<a href="nights/%s_%s.html">n</a>' % (r['lst'], r['date']) for r in csv.DictReader(open(LEDGER)))
    open(OUT + '.part', 'w').write('<html>' + frag + links + '</html>')
    os.replace(OUT + '.part', OUT)
    print('data page -> ' + OUT)
''',
    'build_ratified_register.py': '''
import collections, csv, os
REG = '/nonexistent/RATIFIED_STEPS_REGISTER.csv'; PAGE = '/nonexistent/ratified_steps.html'
COLS = ['ruling_id', 'date', 'gregg_words', 'step', 'requirement', 'superseded_by', 'enforced_where', 'evidence', 'status',
        'note', 'source']
ORDER = ['NOT-ENFORCED', 'SIDE-SCRIPT-ONLY', 'DATA-FILE-ONLY', 'DOC-CODE-DISCREPANCY', 'UNCLEAR', 'ENFORCED', 'SUPERSEDED',
         'DRAFT-NOT-RATIFIED', 'OUT-OF-TIER']
def main(paths):
    rows = []
    for p in paths:
        for r in csv.DictReader(open(p)):
            r = {k: (r.get(k) or '').strip() for k in COLS}
            r['source'] = os.path.basename(p).replace('.csv', ''); r['status'] = r['status'].upper(); rows.append(r)
    rows.sort(key=lambda r: (ORDER.index(r['status']) if r['status'] in ORDER else 99, r['date'], r['ruling_id']))
    with open(REG + '.new', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    os.replace(REG + '.new', REG)
    cnt = collections.Counter(r['status'] for r in rows)
    open(PAGE, 'w').write('<html><table><tr>' + ''.join('<th>%s</th>' % s for s in ORDER if cnt.get(s)) + '<th>total</th></tr><tr>'
                          + ''.join('<td class="x">%d</td>' % cnt[s] for s in ORDER if cnt.get(s)) + '<td>%d</td></tr></table>'
                          % len(rows) + '</html>')
    print('register: %d rows' % len(rows))
''',
    '_inventory_p1p2.py': '''
import json
rows = json.load(open('@W@/inventory_rows.json'))
json.dump(rows, open('@HANDOFF@', 'w'))                  # production: df.to_pickle('/tmp/_inventory_p1p2.pkl')
print('inventory: %d (LST,date) cells' % len(rows))
''',
    '_build_ledger.py': '''
import csv, json
rows = json.load(open('@HANDOFF@'))
cols = list(rows[0].keys())
with open('@W@/clv15/processing_ledger.csv', 'w', newline='') as f:   # production: the clv15 ledger literal (:59)
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
print('WROTE ledger rows=', len(rows))
''',
    'solar_contamination_audit.py': '''
import csv, os
LOW = [18, 23, 27, 32, 36, 41]
REF = 46
S_WARN, S_BAD = 3.0, 10.0
OUT = '@W@/QA/solar_contamination_audit.csv'
import json
rows = json.load(open('@W@/solar_rows.json'))
with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
if os.environ.get('FAKE_SOLAR_ESCAPE'):
    open('/proc/exopipe3_status_test_escape', 'w').write('x')   # a write outside --out: the guard must refuse it
print('solar audit: %d blocks' % len(rows))
''',
    'candidate_provenance_audit.py': '''
import csv, sys
RUN = sys.argv[1]
OUT = '@W@/QA/candidate_provenance_audit.csv'
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['cand_id', 'lst', 'band', 'date', 'excluded_data', 'solar'])
    w.writerow(['05h_2099-01-03_Full_V_wb_5', '05h', 'Full', '2099-01-03', 1, ''])
    w.writerow(['05h_2099-01-01_18MHz_I_sb_1', '05h', '18MHz', '2099-01-01', 0, 'SUN-UP'])
print('provenance: ' + RUN)
''',
    'perframe_solar_check.py': '''
import csv, sys
OUT = '@W@/QA/perframe_solar_check.csv'
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['cand_id', 'sun_min', 'sun_max', 'verdict'])
    w.writerow(['05h_2099-01-01_18MHz_I_sb_1', 5.0, 10.0, 'SUN-UP'])
    w.writerow(['05h_2099-01-01_Full_I_wb_2', -30.0, -25.0, 'night'])
''',
    'build_qa_table.py': '''
import csv, os, sys
COLS = ['lst', 'date', 'subband', 'verdict'] + ['p3g_in_record', 'p3g_in_master', 'p3g_nightgate', 'p3g_epochgate',
                                               'p3s_searched_V', 'p3s_searched_I', 'p3s_refused_V', 'p3s_refused_I', 'p3_status']
OUT = sys.argv[1]
if (os.environ.get('P3_ARCHIVE_RUN'), os.environ.get('P3_QA_SEARCH_LABEL')) != ('Run_TESTARCH', 'P3S5'):
    sys.exit('env not set from config: %r' % ((os.environ.get('P3_ARCHIVE_RUN'), os.environ.get('P3_QA_SEARCH_LABEL')),))
base = dict(verdict='', p3g_in_record='1', p3g_in_master='1', p3g_nightgate='PASS', p3g_epochgate='PASS', p3s_searched_V='1',
            p3s_searched_I='1', p3s_refused_V='', p3s_refused_I='')
rows = [dict(base, lst='05h', date='2099-01-01', subband='18MHz', p3_status='in master; searched V+I'),
        dict(base, lst='05h', date='2099-01-03', subband='18MHz', verdict='exclude', p3_status='excluded (data_decisions)')]
with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
''',
    'beam_audit.py': '''
import csv, os, sys
outdir = sys.argv[1]; nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 16
os.makedirs(outdir, exist_ok=True)
rows = []
for i, (a, b, m) in enumerate(((100.0, 80.0, 8000.0), (130.0, 90.0, 8000.0), (100.0, 82.0, 8100.0))):
    rows.append(dict(lst='05h', date='2099-01-0%d' % (i + 1), scope='18MHz', stokes='I', bmaj_as=a, bmin_as=b,
                     cell_median_area=m, area_ratio=round(a * b / m, 3), err='', path='/x/e%d.fits' % i))   # beam_audit.py:73-79
for name, rs in (('beam_audit_all.csv', rows), ('beam_audit_flagged.csv', [r for r in rows if r['area_ratio'] >= 1.2])):
    with open(os.path.join(outdir, name), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rs)
''',
    'qa_noise_vs_freq_plot.py': '''
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT = sys.argv[1]
fig, ax = plt.subplots(figsize=(17, 7.5))
ax.semilogy([18, 46, 82], [3.0, 1.0, 0.5], 'o-')
fig.savefig(OUT, dpi=120)
print('noise plot ' + OUT)
''',
    'bandpass_phase_warp.py': '''
import csv, sys
a = sys.argv[1:]
out = a[a.index('--csv') + 1]; i = int(a[a.index('--shard') + 1]); n = int(a[a.index('--shard') + 2]); lst = a[-1]
tables = [l.strip() for l in open(lst) if l.strip()][i::n]
cols = ['cal_date', 'cal_lst', 'cal_run', 'n', 'n_skipped', 'res_in', 'res_hi', 'ratio', 'jump', 'rad75', 'bad', 'offset_deg',
        'error', 'table']
rows = []
for t in tables:
    rec = dict(cal_date=t.split('calibration_')[1][:10], cal_lst='05h', cal_run='r', table=t, error='')
    if t.endswith('2099-01-05_05h.B.flagged'):
        rec['error'] = 'no solutions'
    else:
        m = dict(n=100, n_skipped=0, res_in=0.0298123, res_hi=0.0639456, ratio=0.0639456 / 0.0298123, jump=0.1, rad75=1.0,
                 bad=0, offset_deg=0.0)
        rec.update({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in m.items()})
    rows.append(rec)
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
''',
    'coadd_exclusion_conformance.py': '''
import csv, os, sys
ROOT = sys.argv[1]; OUT = os.environ['CONF_OUT']
rows = [dict(lst='05h', band='Full', kind='I', product='05h_Full_I_deep_coadd.fits', date='2099-01-03', verdict='BREACH',
             note='product written 20990105')]
with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('1 co-add products in ' + ROOT); print('  BREACH 1')
sys.exit(1)
''',
    'snapshot_batch_audit.py': '''
import argparse, csv
ST = {'18h_2099-02-01': ('1', '1', '1', 'PREFIT-STALE'), '18h_2099-02-02': ('1', '1', '0', 'PREFIT-DONE'),
      '18h_2099-02-03': ('1', '0', '0', 'TO-PREFIT'), '18h_2099-02-04': ('1', '1', '0', 'PREFIT-DONE')}
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--cells-file', nargs='*')
ap.add_argument('--exclude-file'); ap.add_argument('--no-store', action='store_true'); a = ap.parse_args()
cells = []
for f in a.cells_file or ['@W@/snap/cells.txt']:
    for l in open(f):
        p = l.split()
        if p and not p[0].startswith('#'):
            cells.append(p[0] if len(p) == 1 else '%s_%s' % (p[0], p[1]))
rows = []
for c in cells:
    sh, pd, stale, s = ST.get(c, ('0', '0', '0', 'NOT-SHIPPED'))
    rows.append(dict(cell=c, shipped=sh, dirty=0, n_flow_unregistered=0, n_tier1min_paused=0, n_missing_1min=0, frametime_job=0,
                     lightcurves_missing='', old_stage_active=0, prefit_active=0, prefit_done=pd, prefit_stale=stale, status=s,
                     in_exclude=0))
with open(a.out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('AUDIT table: %s' % a.out)
''',
    'snapshot_batch_lists.py': '''
import argparse, csv, hashlib, subprocess, sys
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--base'); ap.add_argument('--order')
for k in ('--max-new', '--max-gate-gb', '--lc-cache', '--base-detections', '--base-loaded', '--new-detections', '--new-loaded'):
    ap.add_argument(k, type=int)
ap.add_argument('--all', action='store_true'); a = ap.parse_args()
def cells_of(path):
    out = []
    for l in open(path):
        p = l.split()
        if p and not p[0].startswith('#'):
            out.append(p[0] if len(p) == 1 else '%s_%s' % (p[0], p[1]))
    return out
base = cells_of(a.base)
cand = [c for c in cells_of(a.order) if c not in set(base)]
open(a.out + '.candidates.txt', 'w').write('\\n'.join(cand) + '\\n')
subprocess.run([sys.executable, '@FAKE@/snapshot_batch_audit.py', '--cells-file', a.out + '.candidates.txt', '--no-store',
                '--out', a.out + '.audit.csv'], check=True, stdout=subprocess.DEVNULL)
st = {r['cell']: r['status'] for r in csv.DictReader(open(a.out + '.audit.csv'))}
ok = ('PREFIT-DONE', 'TO-PREFIT', 'PREFIT-STALE') if a.all else ('PREFIT-DONE',)
new, left = [], []
for c in cand:
    s = st.get(c, '?')
    if s not in ok:
        left.append((c, s)); continue
    if not a.all and len(new) >= a.max_new:
        left.append((c, s + ' (over the batch budget)')); continue
    new.append(c)
open(a.out, 'w').write('\\n'.join(base + new) + '\\n')
print('LIST %s: %d cells = %d base + %d new' % (a.out, len(base) + len(new), len(base), len(new)))
print('CELLS_FILE_SHA256=%s' % hashlib.sha256(open(a.out, 'rb').read()).hexdigest())
open(a.out + '.left_out.txt', 'w').write(''.join('%s %s\\n' % x for x in left))
''',
    'p3s3_sources.py': '''
import json, os
LABEL = os.environ.get('P3S_LABEL', 'P3S3')
STATE = '@W@/ops/%s_state' % LABEL.lower()
def published_attempts(stokes=('V', 'I')):
    return [a for a in json.load(open('@W@/attempts.json')) if a['stokes'] in stokes]
''',
    'p3s3_stale_exclusions.py': '''
import csv, os
def stale_attempts(atts):
    from exopipe.phase3.core import exclusions as E
    ex = E.load(); out = []
    for a in atts:
        if not a.get('harvested'):
            continue
        p = a['dir'] + '/transients/Run_' + os.environ.get('P3S_LABEL', 'P3S3') + '/v_deep_images.csv'
        lst, scope = a['cell'].split('_', 1)
        searched = sorted({r['date'] for r in csv.DictReader(open(p)) if r.get('lst') == lst and r.get('scope') == scope
                           and r.get('stokes') == a['stokes']})
        bad = [d for d in searched if ex.is_excluded(lst, scope, d)]
        if bad:
            out.append((a, bad))
    return out
def refuse_and_unmark(stale, uni):
    raise RuntimeError('S-G13: status must never call refuse_and_unmark')
''',
    'snapshot_frametime_worklist.py': '''
import os
def needs(celldir):
    if os.path.exists(os.path.join(celldir, 'STALE_MARK')):
        return {'frame_time_1min'}, ['18MHz']
    return set(), []
''',
}
FAKE_MODULES = ('conformance_check', 'build_qa_pages', 'build_data_processing_page', 'build_ratified_register',
                'p3s3_sources', 'p3s3_stale_exclusions', 'snapshot_frametime_worklist')


def write_fakes():
    handoff = f'{W}/tmp_handoff/_inventory_p1p2.pkl'
    for name, code in FAKES.items():
        wtxt(f'{FAKE}/{name}', code.replace('@W@', W).replace('@FAKE@', FAKE).replace('@HANDOFF@', handoff).lstrip())
    json.dump(SOLAR_ROWS, open(f'{W}/solar_rows.json', 'w'))
    json.dump(ledger_rows(extra_night=True), open(f'{W}/inventory_rows.json', 'w'))
    return handoff


@contextlib.contextmanager
def world():
    """Point every root the step reads at the synthetic world and the fake engines; restore everything afterwards."""
    from exopipe.phase3.lib.status import steps_exclconf as XC
    build_world()
    handoff = write_fakes()
    sets = {('paths', 'DATA_DECISIONS'): f'{W}/clv15/data_decisions.csv', ('status', 'QA_DIR'): f'{W}/QA',
            ('status', 'LEDGER'): f'{W}/clv15/processing_ledger.csv', ('serve', 'HUB_DIR'): f'{W}/hub',
            ('status', 'REPORT_ROOT'): f'{TMP}/empty_reports', ('runner', 'JOBS_ROOT'): f'{TMP}/jobs',
            ('status_report', 'DECISIONS_FINGERPRINT'): f'{W}/fast/.data_decisions.fingerprint',
            ('status_report', 'COADD_ROOTS'): (f'{W}/coadd_sb',), ('status_report', 'RETIRED_COADD'): f'{W}/retired',
            ('status_report', 'CANON_CHECK'): 'echo canon-check-test-output',
            ('status_report', 'REGISTER'): f'{W}/clv15/RATIFIED_STEPS_REGISTER.csv',
            ('status_ledger', 'LEGACY_HANDOFF'): handoff,
            ('paths', 'COADD_SB_ROOT'): f'{W}/coadd_sb', ('paths', 'PHASE1_ROOT'): f'{W}/phase1',
            ('paths', 'PHASE2_ROOT'): f'{W}/phase2', ('paths', 'PHASE3_ARCHIVE_ROOT'): f'{W}/arch',
            ('paths', 'ARCHIVE_RUN'): 'Run_TESTARCH', ('paths', 'TRANSIENTS_ROOT'): f'{W}/transients',
            ('paths', 'SNAPSHOT_FLEET_OUT'): f'{W}/fleet', ('paths', 'OPS_ROOT'): f'{W}/ops',
            ('paths', 'SUBLST_ROOT'): f'{W}/sublst', ('paths', 'SNAPSHOT_REVIEW_ROOT'): f'{W}/snapreview',
            ('status_conformance', 'DD_HISTORY_GLOB'): f'{W}/clv15/data_decisions.csv.bak_20*',
            ('status_conformance', 'MASTERS_ARCHIVE_ROOT'): f'{W}/arch_coadd',
            ('status_conformance', 'ROUND_LIST'): f'{W}/rebuild_list.txt',
            ('status_qa', 'QA_ASSEMBLY'): f'{W}/qa_assembly',
            ('status_qa', 'BANDPASS_TABLES_GLOB'): f'{W}/cal/*/*/successful/*/tables/*.B.flagged',
            ('status_qa', 'BANDPASS_SHARDS'): 3,
            ('status_snapshot', 'ORDER_LIST'): f'{W}/snap/order.txt',
            ('site', 'LEGACY_CODE_ROOTS'): [env.legacy_root('clv15'), FAKE]}
    wtxt(f'{W}/rebuild_list.txt', '05h_Full\n')
    saved = {}
    for (sec, key), v in sets.items():
        s = getattr(CFG, sec)
        saved[(sec, key)] = getattr(s, key)
        setattr(s, key, v)
    sys.path.insert(0, FAKE)
    for m in FAKE_MODULES:
        sys.modules.pop(m, None)
    XC._SNAPS = None
    try:
        yield
    finally:
        for (sec, key), v in saved.items():
            setattr(getattr(CFG, sec), key, v)
        if FAKE in sys.path:
            sys.path.remove(FAKE)
        for m in FAKE_MODULES:
            sys.modules.pop(m, None)
        XC._SNAPS = None
        CFG.reset()


# =====================================================================================================================
# tests on the synthetic world
# =====================================================================================================================

def test_config_and_help():
    rc, txt = step(['status', '--help'])
    check('--help prints the docstring, actions and every gate', rc == 0 and 'S-G14' in txt and 'batch-list' in txt, txt[:400])
    rc, txt = step(['status', '--set', 'status.QA_DIR=/tmp/x', '--out', f'{TMP}/o_cfg', '--dry-run'])
    check('protected key status.QA_DIR refuses an override', rc == 2 and 'PROTECTED' in txt, txt[-300:])
    from exopipe.phase3.core import ledger
    for sec in ('status_rebuild',):
        for k, lids in (getattr(CFG, sec).RULED or {}).items():
            for lid in ((lids,) if isinstance(lids, str) else lids):
                check(f'ruled key {sec}.{k} names a ledger row ({lid})', ledger.has_id(lid, CFG.paths.FIX_LEDGER))
    S = CFG.status_snapshot
    check('batch-list selection values live in config = snapshot_batch_lists.py defaults (:70-76)',
          (S.MAX_NEW, S.MAX_GATE_GB, S.LC_CACHE, S.BASE_DETECTIONS, S.BASE_LOADED, S.NEW_DETECTIONS, S.NEW_LOADED) ==
          (100, 300, 4000, 11530, 213270, 25624, 314396))
    import ast
    src = open(os.path.join(REAL_OPS, 'snapshot_batch_lists.py')).read()
    dflt = {}
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Call) and getattr(n.func, 'attr', '') == 'add_argument' and n.args and \
                isinstance(n.args[0], ast.Constant) and str(n.args[0].value).startswith('--'):
            for kw in n.keywords:
                if kw.arg == 'default' and isinstance(kw.value, ast.Constant):
                    dflt[n.args[0].value] = kw.value.value
    from exopipe.phase3.steps import status as STEP
    check('every batch-list config value equals the production script default (ast read)',
          all(dflt.get(f) == getattr(S, k) for f, k in STEP._BATCH_FLAGS), dflt)


def test_summary():
    out = f'{TMP}/o_summary'
    rc, txt = step(['status', '--out', out, '--cores', 'none'])
    r = rec(out, 'report_summary_deep')
    check('summary done (phase_status in place + CLAUDE.md 1 sections); every content test passes',
          rc == 0 and r.get('status') == 'done' and all(t['passed'] for t in r['tests']), txt[-1500:])
    check('legacy text is phase_status.py run in place on the synthetic ledger',
          '[ledger] 5 (LST,date) cells' in txt and 'canon-check-test-output' in txt, txt[:1200])
    check('masters counted from the files carrying COADDOP (1 of 2), never the .done marker; newest master = that file',
          'carrying COADDOP): 1 {\'ivw-mean-v3\': 1}; 1 without COADDOP' in txt and '.done markers' not in txt and
          'newest deep master' in txt and '(05h_Full)' in txt, [ln for ln in txt.splitlines() if 'coadd' in ln.lower()])
    lst = os.path.join(out, 'QA', f'readjudication_list_{time.strftime("%Y%m%d", time.gmtime())}.csv')
    rr = rows(lst)
    check('re-adjudication: the excluded night whose (p1_peel, p2) advanced since the production list is proposed',
          [(x['lst'], x['date']) for x in rr] == [('05h', '2099-01-03')], rr)
    fp = f'{W}/fast/.data_decisions.fingerprint'
    check("fingerprint mode 'read': the production fingerprint file is never written", open(fp).read() == '0.0\n')
    j = [p for p in os.listdir(os.path.join(out, 'status')) if p.endswith('.json')][0]
    J = json.load(open(os.path.join(out, 'status', j)))
    m = J['masters'][f'{W}/coadd_sb']
    check('served runs read from the hub registry; masters on the configured roots: 2 build records, 1 master carrying '
          'COADDOP (its file and mtime), 1 without',
          {p['run'] for p in J['served_runs']['pages']} == {'Run_TESTA', 'Run_SLSTB'} and m['with_weights_record'] == 2
          and (m['masters_coaddop'], m['coaddop'], m['masters_without_coaddop']) == (1, {'ivw-mean-v3': 1}, 1) and
          m['newest_master_utc'] == '2025-06-01T00:00:00Z', (J['served_runs'], m))
    first = os.path.join(TMP, 'readjudication_first', os.path.basename(lst))
    os.makedirs(os.path.dirname(first))
    shutil.copyfile(lst, first)
    # a second run compares with the first run's record: nothing advanced -> empty list (empty allowed, absent not)
    rc, txt = step(['status', '--out', out, '--cores', 'none', '--no-legacy-text'])
    r = rec(out, 'report_summary_deep')
    check('second summary compares with the previous record: 0 rows, still done', rc == 0 and r['status'] == 'done' and
          'record ' in str(r['outputs'][-2].get('ctx', {}).get('previous_source', '')), r.get('outputs'))
    # negative: the readjudication list with the advanced row removed fails against the run's context
    first_ctx = {'ledger': CFG.status.LEDGER, 'ledger_sha1': sha(CFG.status.LEDGER),
                 'previous': {'05h_2099-01-03': ['none', 'partial'], '05h_2099-01-02': ['none', 'complete']}}
    lst = first
    neg('readjudication_missing_row', 'readjudication_list', lst, lambda p: wcsv(
        p, list(CFG.status_report.READJUDICATION_COLUMNS), []), first_ctx)
    neg('readjudication_wrong_status', 'readjudication_list', lst, edit('all-excluded', 'none'))
    jp = os.path.join(out, 'status', j)
    neg('report_json_section_absent', 'status_report', jp, lambda p: json.dump(
        {k: v for k, v in json.load(open(p)).items() if k != 'canon_check'}, open(p + '.x', 'w')) or os.replace(p + '.x', p))
    neg('report_json_counts', 'status_report', jp, lambda p: json.dump(
        dict(json.load(open(p)), phase_completion=dict(J['phase_completion'], counts={'p1': {'complete': 1}})),
        open(p + '.x', 'w')) or os.replace(p + '.x', p))
    neg('report_text_count', 'status_report', jp[:-5] + '.txt', edit('[ledger] 5 (LST, night)', '[ledger] 7 (LST, night)'),
        siblings=[jp])
    neg('report_text_masters', 'status_report', jp[:-5] + '.txt', edit('1 masters carrying COADDOP', '2 masters carrying COADDOP'),
        siblings=[jp])
    neg('report_json_masters', 'status_report', jp, lambda p: json.dump(
        dict(json.load(open(p)), masters={f'{W}/coadd_sb': dict(m, masters_coaddop=2, coaddop={'ivw-mean-v3': 2})}),
        open(p + '.x', 'w')) or os.replace(p + '.x', p))
    it = os.path.join(out, 'items', 'report_summary_deep.txt')
    neg('item_list_unsorted', 'item_list', it, lambda p: open(p, 'a').write('05h_2099-01-01\n04h_2099-01-01\n'),
        expect='items_parse')


def test_coverage_gaps():
    out = f'{TMP}/o_cov'
    em = f'{TMP}/gaps_cells.txt'
    rc, txt = step(['status', '--gaps', '--cells', CELLS, '--emit', em, '--out', out, '--cores', 'none'])
    u = [f for f in os.listdir(os.path.join(out, '_records', 'status')) if f.startswith('report_gaps_deep_')]
    r = rec(out, u[0][:-5]) if u else {}
    check('gaps done; coverage table + item list pass their content tests', rc == 0 and r.get('status') == 'done', txt[-1200:])
    tab = [o['path'] for o in r['outputs'] if o['class'] == 'coverage_table'][0]
    t = {(x['scope'], x['date']): x for x in rows(tab)}
    check('rows: one per (LST, scope, night) with a Phase 2 product (8)', len(t) == 8, sorted(t))
    check('excluded by core.exclusions: 18 MHz 01-02 and both scopes 01-03; 18-only exclusion keeps Full (carve-out)',
          [k for k, x in sorted(t.items()) if x['excluded'] == '1'] == [('18MHz', '2099-01-02'), ('18MHz', '2099-01-03'),
                                                                     ('Full', '2099-01-03')], t)
    check('gaps = not excluded, no residual: 18 MHz 01-04 and Full 01-02',
          sorted(k for k, x in t.items() if x['gap'] == '1') == [('18MHz', '2099-01-04'), ('Full', '2099-01-02')])
    check('census classes: archived = ok, residual only = recover-local, partial exclusion flagged',
          t[('18MHz', '2099-01-01')]['census_class'] == 'ok' and t[('Full', '2099-01-01')]['census_class'] == 'recover-local'
          and t[('Full', '2099-01-02')]['census_class'].endswith('+partial-exclusion'), {k: x['census_class'] for k, x in t.items()})
    check('searched columns from the published attempts (fake p3s3_sources); Phase 1 newer than Phase 2 flagged',
          t[('18MHz', '2099-01-02')]['searched_V'] == '1' and t[('Full', '2099-01-01')]['searched_I'] == '1' and
          t[('18MHz', '2099-01-01')]['p1_newer_than_p2'] == '1', t[('18MHz', '2099-01-02')])
    from exopipe.phase3.core import cells
    check('--emit FILE: an item list usable as --cells @FILE', cells.expand('@' + em) == ['05h_18MHz', '05h_Full'], open(em).read())
    rc, txt = step(['status', '--coverage', '--cells', '@' + em, '--out', f'{TMP}/o_cov2', '--cores', 'none',
                    '--set', 'status_coverage.ARCHIVE_COLUMNS=False'])
    check('the emitted list drives the next run (--cells @FILE)', rc == 0 and '8 rows, 2 gaps' in txt, txt[-600:])
    em2 = f'{TMP}/gaps_epochs.txt'
    rc, txt = step(['status', '--gaps', '--cells', CELLS, '--emit-kind', 'epoch', '--emit', em2, '--out', out, '--cores', 'none'])
    check('--emit-kind epoch: the gap hours for the per-hour chain', rc == 0 and
          cells.expand_epochs('@' + em2) == ['05h_2099-01-02', '05h_2099-01-04'], open(em2).read())
    rc, txt = step(['status', '--coverage', '--lsts', '05h', '--out', f'{TMP}/o_cov3', '--cores', 'none',
                    '--set', 'status_coverage.SEARCHED_COLUMNS=False'])
    check('--lsts 05h narrows the deep cells to that LST (19 cells, 8 rows with Phase 2 products)',
          rc == 0 and 'for 19 cells' in txt and '8 rows' in txt, txt[-600:])
    rc, txt = step(['status', '--coverage', '--epochs', '05h_2099-01-02', '--out', f'{TMP}/o_cov4', '--cores', 'none'])
    check('--epochs: the hour\'s LST only, rows of that night only', rc == 0 and 'for 19 cells' in txt and '2 rows, 1 gaps' in txt,
          txt[-600:])
    rc, txt = step(['status', '--gaps', '--strict', '--cells', CELLS, '--out', f'{TMP}/o_cov5', '--cores', 'none'])
    check('S-G14 --strict: gaps found -> exit 2', rc == 2 and 'S-G14' in txt, txt[-400:])
    rc, txt = step(['status', '--coverage', '--tier', 'sublst', '--lsts', '05h', '--out', f'{TMP}/o_cov6', '--cores', 'none'])
    tab = os.path.join(TMP, 'o_cov6', 'QA', [f for f in os.listdir(os.path.join(TMP, 'o_cov6', 'QA'))][0])
    g = [x for x in rows(tab) if x['gap'] == '1']
    check('sub-LST coverage: 05h 18 MHz windows searched, LST run of the pre-fitted generation, not served -> gap',
          rc == 0 and len(g) == 1 and 'older generation' in g[0]['reason'] and 'not in the served' in g[0]['reason'], g)
    sl = {x['scope']: x for x in rows(tab)}
    check('sub-LST search done from its RECORD (shipped log), never the done/ marker: a marker without a record is not '
          'done and is named; a log with a Traceback is not done',
          (sl['18MHz']['search_V'], sl['18MHz']['search_I']) == ('done', 'done') and sl['23MHz']['search_V'] == '' and
          sl['23MHz']['done_marker_V'] == '1' and 'done marker without a search record' in sl['23MHz']['marker_disagrees']
          and sl['27MHz']['search_I'] == '', {k: (v['search_V'], v['search_I'], v['marker_disagrees']) for k, v in sl.items()
                                              if k in ('18MHz', '23MHz', '27MHz')})
    rc, txt = step(['status', '--coverage', '--tier', 'snapshot', '--out', f'{TMP}/o_cov7', '--cores', 'none'])
    tab7 = os.path.join(TMP, 'o_cov7', 'QA', os.listdir(os.path.join(TMP, 'o_cov7', 'QA'))[0])
    check('snapshot coverage: the worklist cell not shipped is the gap (night_cell item)', rc == 0 and
          [x['cell'] for x in rows(tab7) if x['gap'] == '1'] == ['18h_2099-02-02'] and
          open(os.path.join(TMP, 'o_cov7', 'items', 'report_coverage_snapshot.txt')).read().strip().endswith('18h_2099-02-02'),
          txt[-500:])
    sn = {x['cell']: x for x in rows(tab7)}
    check('snapshot searched from its RECORDS (runinfo or night refusal per admitted subband and Stokes), never the '
          'fleet claim done: a claim done with V unaccounted is not shipped and is named',
          sn['18h_2099-02-01']['shipped'] == '1' and sn['18h_2099-02-03']['shipped'] == '0' and
          sn['18h_2099-02-03']['claim_done'] == '1' and sn['18h_2099-02-03']['unaccounted'] == '18MHz_V' and
          'without complete search records' in sn['18h_2099-02-03']['marker_disagrees'], sn)
    neg('coverage_snapshot_shipped_flag', 'coverage_table', tab7, edit('18h_2099-02-03,18h,2099-02-03,0,0,',
                                                                       '18h_2099-02-03,18h,2099-02-03,0,1,'))
    neg('coverage_sublst_search_record', 'coverage_table', tab, lambda p: wcsv(p, list(rows(p)[0].keys()), [
        dict(x, search_I='done') if x['scope'] == '27MHz' else x for x in rows(p)]))
    neg('coverage_excluded_flag', 'coverage_table', [o['path'] for o in r['outputs'] if o['class'] == 'coverage_table'][0],
        edit('05h,18MHz,2099-01-02,05h_18MHz,1,1', '05h,18MHz,2099-01-02,05h_18MHz,1,0'))
    neg('coverage_sublst_gap_reason', 'coverage_table', tab, lambda p: wcsv(p, list(rows(p)[0].keys()), [dict(x, reason='')
                                                                                                         for x in rows(p)]))
    it = [o['path'] for o in r['outputs'] if o['class'] == 'item_list'][0]
    neg('item_list_missing_qualifying_item', 'item_list', it, drop_last_line, expect='items_qualify')


def test_rebuild_due():
    out = f'{TMP}/o_rb'
    rc, txt = step(['status', '--rebuild-due', '--cells', CELLS, '--out', out, '--cores', 'none'])
    u = [f[:-5] for f in os.listdir(os.path.join(out, '_records', 'status')) if f.startswith('report_rebuilddue')][0]
    r = rec(out, u)
    tab = [o['path'] for o in r['outputs'] if o['class'] == 'rebuild_due'][0]
    t = {x['cell']: x for x in rows(tab)}
    check('rebuild-due: 18 MHz due (1000 s new >= 0.5 x 1000 s), Full not due (no admitted new night)',
          rc == 0 and t['05h_18MHz']['status'] == 'due' and t['05h_Full']['status'] == 'not-due' and
          t['05h_18MHz']['new_nights'] == '2099-01-04', t)
    check('rebuild-due is a report: on-disk new nights (no D-ADMIT admission record yet), the record says no trigger',
          all(x['new_source'] == 'on-disk' for x in t.values()) and 'never a rebuild trigger' in t['05h_18MHz']['measure']
          and 'none: a report' in r['meta']['trigger'] and 'report only' in txt, r['meta'].get('trigger'))
    rc, txt = step(['status', '--rebuild-due', '--tier', 'snapshot', '--out', out, '--cores', 'none'])
    check('rebuild-due --tier snapshot refused', rc == 2 and 'snapshot masters' in txt, txt[-300:])
    rc, txt = step(['status', '--rebuild-due', '--cells', CELLS, '--out', out, '--cores', 'none',
                    '--set', 'status_rebuild.NEW_SET_SOURCE=admission-record'])
    check("D-ADMIT hook: 'admission-record' without status_rebuild.ADMISSION_RECORD refused (no fallback)", rc == 2 and
          'S-G11' in txt and 'ADMISSION_RECORD' in txt, txt[-400:])
    sci = 'Science_20990101_000000'
    wcsv(f'{W}/coadd_sb/05h_18MHz/coadd_final/admitted_test.csv', ['date', 'sci_dir'],
         [dict(date='2099-01-04', sci_dir=f'{W}/phase2/05h/2099-01-04/{sci}')])
    rc, txt = step(['status', '--rebuild-due', '--cells', CELLS, '--out', f'{TMP}/o_rb2', '--cores', 'none',
                    '--set', 'status_rebuild.NEW_SET_SOURCE=admission-record',
                    '--set', 'status_rebuild.ADMISSION_RECORD=admitted_test.csv'])
    u2 = [f[:-5] for f in os.listdir(os.path.join(TMP, 'o_rb2', '_records', 'status')) if f.startswith('report_rebuilddue')][0]
    r2 = rec(f'{TMP}/o_rb2', u2)
    tab2 = [o['path'] for o in r2['outputs'] if o['class'] == 'rebuild_due'][0]
    t2 = {x['cell']: x for x in rows(tab2)}
    check('D-ADMIT hook: new nights = the admission record beside weights.csv (18 MHz: 2099-01-04 admitted -> due; Full: '
          'no record -> nothing admitted, not due)', rc == 0 and r2['status'] == 'done' and
          (t2['05h_18MHz']['status'], t2['05h_18MHz']['new_nights'], t2['05h_18MHz']['new_source']) ==
          ('due', '2099-01-04', 'admission-record') and t2['05h_Full']['status'] == 'not-due', (txt[-400:], t2))
    CFG.status_rebuild.ADMISSION_RECORD = 'admitted_test.csv'
    try:
        neg('rebuild_due_not_admitted', 'rebuild_due', tab2, edit(',2099-01-04,admission-record,', ',2099-01-05,admission-record,'))
    finally:
        CFG.status_rebuild.ADMISSION_RECORD = None
    os.remove(f'{W}/coadd_sb/05h_18MHz/coadd_final/admitted_test.csv')
    neg('rebuild_due_flag', 'rebuild_due', tab, edit(',due,1,', ',not-due,0,'))


def test_exclusion_conformance():
    out = f'{TMP}/o_xc'
    rc, txt = step(['status', '--exclusion-conformance', '--cells', CELLS, '--out', out, '--cores', 'none'])
    u = [f[:-5] for f in os.listdir(os.path.join(out, '_records', 'status')) if f.startswith('report_exclusionconformance')][0]
    r = rec(out, u)
    check('exclusion-conformance done (archive audit exit 1 = breaches recorded, not inherited without --strict)',
          rc == 0 and r['status'] == 'done' and r['meta']['archive_audit']['rc'] == 1, txt[-1200:])
    tab = [o['path'] for o in r['outputs'] if o['class'] == 'exclusion_conformance' and 'coadd_' not in o['path']][0]
    check('a narrowed run (--cells) writes its own table (selection suffix), never over the full run\'s',
          os.path.basename(tab) == f"exclusion_conformance_deep{u[len('report_exclusionconformance_deep'):]}.csv" and
          u != 'report_exclusionconformance_deep', tab)
    t = [(x['part'], x['lst'], x['band'], x['date'], x['verdict']) for x in rows(tab)]
    check('working-tree masters: 18 MHz constituent excluded after the build = STALE; Full constituent excluded before = '
          'BREACH (dated data_decisions backups)', ('master-working', '05h', '18MHz', '2099-01-02', 'STALE') in t and
          ('master-working', '05h', 'Full', '2099-01-03', 'BREACH') in t, t)
    sv = [x for x in rows(tab) if x['part'] == 'served-run']
    check('served run: rows on the excluded cell-band keyed by the production gate keys_of (judgements/ skipped)',
          len(sv) == 2 and {x['file'] for x in sv} == {'selected_all.csv', 'master_catalog.csv'}, sv)
    check('archive part through the in-place script (fake): its table written under --out',
          ('master-archive', '05h', 'Full', '2099-01-03', 'BREACH') in t and
          os.path.exists(os.path.join(out, 'QA', 'coadd_exclusion_conformance.csv')))
    rc, txt = step(['status', '--exclusion-conformance', '--strict', '--cells', CELLS, '--out', out, '--cores', 'none'])
    check('S-G14 --strict: breaches / served rows on excluded cell-bands -> exit 2', rc == 2, txt[-300:])
    bad = f'{W}/transients/Run_TESTA/broken.csv'
    open(bad, 'wb').write(b'cand_id\n\xff\xfe05h_2099-01-01_18MHz\n')
    rc, txt = step(['status', '--exclusion-conformance', '--cells', CELLS, '--out', f'{TMP}/o_xc2', '--cores', 'none'])
    os.remove(bad)
    check('an unreadable served CSV refuses the audit (the serve gate refuses it, :83-85)', rc == 2 and
          'served_files_readable' in txt, txt[-500:])
    rc, txt = step(['status', '--exclusion-conformance', '--tier', 'sublst', '--out', f'{TMP}/o_xc3', '--cores', 'none'])
    check('sub-LST tier: the served cumulative run audited (row on the excluded band)', rc == 0 and
          len(rows(os.path.join(TMP, 'o_xc3', 'QA', 'exclusion_conformance_sublst.csv'))) == 1, txt[-500:])
    neg('exclusion_row_not_excluded', 'exclusion_conformance', tab, edit(',2099-01-02,18MHz,STALE', ',2099-01-01,18MHz,STALE'))
    neg('exclusion_verdict', 'exclusion_conformance', tab, edit('18MHz,STALE', '18MHz,BREACH'))
    neg('exclusion_archive_not_excluded', 'exclusion_conformance', os.path.join(out, 'QA', 'coadd_exclusion_conformance.csv'),
        edit('2099-01-03', '2099-01-01'))


def test_violations_stale():
    out = f'{TMP}/o_st'
    rc, txt = step(['status', '--violations', '--out', out, '--cores', 'none'])
    r = rec(out, 'report_violations_deep')
    check('violations: FAIL cells of the membership table, listed for masters', rc == 0 and
          open(os.path.join(out, 'items', 'report_violations_deep.txt')).read().strip().endswith('05h_Full'), txt[-500:])
    rc, txt = step(['status', '--stale', 'record', '--records', f'{TMP}/recs', '--stale-step', 'frames', '--out', out,
                    '--cores', 'none'])
    r = rec(out, 'report_stale_record_deep')
    tab = [o['path'] for o in r['outputs'] if o['class'] == 'stale_table'][0]
    t = {x['item']: x for x in rows(tab)}
    check('--stale record: admitted night excluded now, changed input, dirty tree = stale; refused = listed unfinished; '
          'a clean record not listed', rc == 0 and set(t) == {'05h_18MHz', '05h_Full', '05h_Green', '05h_Blue'} and
          'excluded now' in t['05h_18MHz']['reason'] and 'input changed' in t['05h_Full']['reason'] and
          'dirty' in t['05h_Green']['reason'] and t['05h_Blue']['status'] == 'refused', t)
    it = [o['path'] for o in r['outputs'] if o['class'] == 'item_list'][0]
    from exopipe.phase3.core import cells
    check('stale record list (kind unit) holds the stale units only', cells.expand('@' + it) == ['05h_Full', '05h_Green',
                                                                                                 '05h_18MHz'] or
          sorted(cells.expand('@' + it)) == ['05h_18MHz', '05h_Full', '05h_Green'], open(it).read())
    neg('stale_row_without_reason', 'stale_table', tab, lambda p: wcsv(p, list(rows(p)[0].keys()),
                                                                       [dict(x, reason='') for x in rows(p)]))
    rc, txt = step(['status', '--stale', 'search_attempt,master', '--cells', CELLS, '--out', out, '--cores', 'none'])
    u = [f[:-5] for f in os.listdir(os.path.join(out, '_records', 'status')) if f.startswith('report_stale_search_attempt')][0]
    t = rows([o['path'] for o in rec(out, u)['outputs'] if o['class'] == 'stale_table'][0])
    check('--stale search_attempt (fake stale_attempts in place, never refuse_and_unmark) and master (STALE masters)',
          rc == 0 and {(x['class'], x['item']) for x in t} == {('search_attempt', '05h_18MHz'), ('master', '05h_18MHz')}
          and 'data_decisions.csv' in [x for x in t if x['class'] == 'search_attempt'][0]['detail'], t)
    rc, txt = step(['status', '--stale', 'sublst_generation', '--tier', 'sublst', '--out', out, '--cores', 'none'])
    check('--stale sublst_generation: the LST run without pos_kind', rc == 0 and open(
        os.path.join(out, 'items', 'report_stale_sublst_generation_sublst.txt')).read().strip().endswith('05h'), txt[-400:])
    rc, txt = step(['status', '--stale', 'frame_time', '--tier', 'snapshot', '--out', out, '--cores', 'none'])
    check('--stale frame_time (fake needs() in place): the pre-fix shipped cell', rc == 0 and open(
        os.path.join(out, 'items', 'report_stale_frame_time_snapshot.txt')).read().strip().endswith('18h_2099-02-01'),
          txt[-400:])
    rc, txt = step(['status', '--stale', 'qa_night_pages', '--out', out, '--cores', 'none'])
    check('--stale qa_night_pages: every ledger night without a hub page (P1)', rc == 0 and 'stale (qa_night_pages): 5 stale'
          in txt, txt[-300:])
    rc, txt = step(['status', '--stale', 'prefit', '--tier', 'snapshot', '--cells-file', f'{W}/snap/cells.txt', '--out', out,
                    '--cores', 'none'])
    check('--stale prefit: the eligibility audit (fake, child) PREFIT-STALE cell', rc == 0 and open(
        os.path.join(out, 'items', 'report_stale_prefit_snapshot.txt')).read().strip().endswith('18h_2099-02-01'), txt[-400:])
    rc, txt = step(['status', '--stale', 'ledger', '--out', out, '--cores', 'none'])
    check('--stale ledger: a fresh inventory (fakes, remapped) differs by one night', rc == 0 and
          'stale (ledger): 1 stale' in txt, txt[-400:])


def test_served_diff_cell_load():
    out = f'{TMP}/o_sd'
    rc, txt = step(['status', '--served-diff', 'TESTB', '--out', out, '--cores', 'none'])
    tab = os.path.join(out, 'status', 'served_diff_Run_TESTA_Run_TESTB_rows.csv')
    t = {(x['side'], x['source']): x for x in rows(tab)} if os.path.exists(tab) else {}
    check('served-diff A (registry) -> B: J1 reduced (a member removed by B\'s gates), J2 in a re-searched cell, J3 new',
          rc == 0 and t[('A', 'J1')]['status'] == 'reduced' and t[('A', 'J2')]['status'] == 're-searched cell' and
          t[('B', 'J3')]['status'] == 'new' and t[('A', 'J1')]['label'] == 'real', txt[-800:])
    rc, txt = step(['status', '--served-diff', 'TESTA', 'SLST05B', '--out', out, '--cores', 'none'])
    check('S-G7: runs of different tiers refused', rc == 2 and 'S-G7' in txt, txt[-300:])
    rc, txt = step(['status', '--served-diff', 'TESTA', 'EMPTY', '--out', out, '--cores', 'none'])
    check('S-G7: a run without served catalogues = no input (exit 3)', rc == 3, txt[-300:])
    neg('served_diff_status', 'served_diff', tab, edit('re-searched cell', 'unchanged'),
        siblings=[tab[:-len('_rows.csv')] + '_summary.json'])
    rc, txt = step(['status', '--cell-load', '--run', 'TESTA', '--out', out, '--cores', 'none'])
    cl = os.path.join(out, 'status', 'cell_load_review_TESTA.csv')
    t = rows(cl)
    check('cell-load: per (Stokes, LST, night, band) detections and served sources (p3s3_pass.sh:213-221), content test',
          rc == 0 and len(t) == 4 and sum(int(x['detections_5sig']) for x in t) == 4 and
          sum(int(x['served_sources']) for x in t) == 1, txt[-400:])
    neg('cell_load_counts', 'cell_load_review', cl, edit('05h,2099-01-01,18MHz,1,1', '05h,2099-01-01,18MHz,2,1'),
        {'run_dir': f'{W}/transients/Run_TESTA'})


def test_qa():
    out = f'{TMP}/o_qa'
    dd0 = sha(CFG.paths.DATA_DECISIONS)
    inst0 = sha(f'{W}/QA/solar_contamination_audit.csv')
    rc, txt = step(['status', 'qa', '--stages', 'qa-table,solar,beam,noise,outliers,bandpass', '--out', out, '--cores', 'none'])
    recs = {s: rec(out, f'qa_{s.replace("-", "")}_none') for s in ('qa-table', 'solar', 'beam', 'noise', 'outliers', 'bandpass')}
    check('qa stages done (fake engines; every content test passes)', rc == 0 and
          all(r.get('status') == 'done' for r in recs.values()), {s: (r.get('status'), r.get('reason')) for s, r in recs.items()})
    check('qa-table child got P3_ARCHIVE_RUN / P3_QA_SEARCH_LABEL from config', recs['qa-table']['meta']['env_set'] ==
          {'P3_ARCHIVE_RUN': 'Run_TESTARCH', 'P3_QA_SEARCH_LABEL': 'P3S5'})
    check('S-G3/S-G4: data_decisions and the installed solar audit unchanged; the candidate written under --out',
          sha(CFG.paths.DATA_DECISIONS) == dd0 and sha(f'{W}/QA/solar_contamination_audit.csv') == inst0 and
          any(f.startswith('solar_contamination_audit.candidate_') for f in os.listdir(os.path.join(out, 'QA'))))
    diff = [o['path'] for o in recs['solar']['outputs'] if o['path'].endswith('.diff.csv')][0]
    check('solar diff against the installed table: 1 block added', [x['change'] for x in rows(diff)] == ['added'], rows(diff))
    op = os.path.join(out, 'QA', 'date_outlier_proposals.csv')
    check('outliers: 2099-01-05 rejected in 4 of 5 cells, leaked at 06h -> 14 proposals (46 MHz has a keep row, never '
          're-proposed), data_decisions never written', len(rows(op)) == 14 and {x['lst'] for x in rows(op)} == {'06h'}
          and '46MHz' not in {x['subband'] for x in rows(op)}, rows(op)[:2])
    bp = os.path.join(out, 'QA', 'bandpass_phase_qa.csv')
    check('bandpass: 3 shards merged, one row per table, error row kept', len(rows(bp)) == 5 and
          sum(1 for x in rows(bp) if x['error']) == 1)
    os.environ['FAKE_SOLAR_ESCAPE'] = '1'
    rc, txt = step(['status', 'qa', '--stages', 'solar', '--out', f'{TMP}/o_qa2', '--cores', 'none'])
    os.environ.pop('FAKE_SOLAR_ESCAPE')
    check('a production tool writing outside --out is refused by the write guard (ParityWriteRefused), stage refused',
          rc == 2 and 'ParityWriteRefused' in txt and not os.path.exists('/proc/exopipe3_status_test_escape'), txt[-600:])
    rc, txt = step(['status', 'qa', '--stages', 'provenance', '--out', out, '--cores', 'none'])
    check('S-G12: provenance without --run refused', rc == 2 and 'S-G12' in txt, txt[-300:])
    rc, txt = step(['status', 'qa', '--stages', 'provenance', '--run', 'TESTA', '--out', out, '--cores', 'none'])
    check('provenance: both in-process scripts, literal outputs redirected under --out', rc == 0 and
          os.path.exists(os.path.join(out, 'QA', 'perframe_solar_check.csv')) and
          not os.path.exists(f'{W}/QA/candidate_provenance_audit.csv'), txt[-600:])
    Q = os.path.join(out, 'QA')
    cand = [os.path.join(Q, f) for f in os.listdir(Q) if f.startswith('solar_contamination_audit.candidate_') and
            not f.endswith('.diff.csv')][0]
    neg('qa_table_p3_status', 'qa_table', os.path.join(Q, 'qa_all_phases.csv'), edit('searched V+I', 'searched V'))
    neg('solar_verdict', 'solar_audit', cand, edit('suspect', 'ok'))
    neg('beam_ratio', 'beam_audit', os.path.join(Q, 'beam_audit', 'beam_audit_all.csv'), edit(',8100.0,1.012,', ',8100.0,1.112,'),
        siblings=[os.path.join(Q, 'beam_audit', 'beam_audit_flagged.csv')])
    neg('beam_flag_missing', 'beam_audit', os.path.join(Q, 'beam_audit', 'beam_audit_all.csv'),
        lambda p: wcsv(os.path.join(os.path.dirname(p), 'beam_audit_flagged.csv'), list(rows(p)[0].keys()), []),
        siblings=[os.path.join(Q, 'beam_audit', 'beam_audit_flagged.csv')])
    neg('noise_plot_size', 'noise_plot', os.path.join(Q, 'median_noise_vs_freq_all_lsts_v2.png'), _small_png)
    neg('outlier_proposal_already_excluded', 'date_outlier_proposals', op, lambda p: open(p, 'a').write(
        '05h,18MHz,2099-01-02,exclude,x,4,5,06h\n'), {'recompute': False})
    neg('bandpass_ratio', 'bandpass_phase_qa', bp, edit('2.1449', '2.1549'))
    neg('bandpass_missing_table', 'bandpass_phase_qa', bp, lambda p: wcsv(p, list(rows(p)[0].keys()), rows(p)[1:]),
        {'tables_list': [o for o in recs['bandpass']['outputs'] if o['class'] == 'bandpass_phase_qa'][0]['ctx']['tables_list']})
    neg('provenance_excluded_flag', 'provenance_audit', os.path.join(Q, 'candidate_provenance_audit.csv'),
        edit('2099-01-03,1,', '2099-01-03,0,'))
    neg('perframe_verdict', 'provenance_audit', os.path.join(Q, 'perframe_solar_check.csv'), edit('night', 'SUN-UP'))


def _small_png(p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4, 3))
    fig.savefig(p, dpi=50)


def test_conformance_pages_ledger_register():
    out = f'{TMP}/o_cp'
    rc, txt = step(['status', 'conformance', '--out', out, '--cores', 'none'])
    r = rec(out, 'conformance_membership_deep')
    check('conformance (fake conformance_check, attr) done; S-G6 passes on the real modules; table, history, page tested',
          rc == 0 and r['status'] == 'done' and len(r['outputs']) == 3, txt[-800:])
    lat = os.path.join(out, 'QA', 'membership_conformance_latest.csv')
    check('in_rebuild_list from --round default (config)', {x['cell']: x['in_rebuild_list'] for x in rows(lat)} ==
          {'05h_18MHz': '0', '05h_Full': '1'}, rows(lat))
    rc, txt = step(['status', '--violations', '--out', out, '--cores', 'none'])
    check('violations read this --out\'s fresh membership table first', rc == 0 and lat in txt, txt[-300:])
    rc, txt = step(['status', 'conformance', '--cells', '05h_Full', '--out', out, '--cores', 'none'])
    check('conformance with --cells refused (whole tree only)', rc == 2, txt[-300:])
    drift = f'{TMP}/membership_drift/membership.py'
    s = open(CFG.status_conformance.PRODUCTION_MEMBERSHIP).read().splitlines()
    i = next(k for k, ln in enumerate(s) if ln.startswith('def '))
    wtxt(drift, '\n'.join(s[:i] + ['EXTRA_RULE = 1'] + s[i:]) + '\n')
    rc, txt = step(['status', 'conformance', '--out', f'{TMP}/o_cp_drift', '--cores', 'none',
                    '--set', f'status_conformance.PRODUCTION_MEMBERSHIP={drift}'])
    check('S-G6: the branch membership module differing from production (non-import line) refuses', rc == 2 and 'S-G6' in txt,
          txt[-400:])
    neg('membership_status', 'membership_conformance', lat, edit('05h_Full,FAIL,2', '05h_Full,PASS,2'),
        siblings=[os.path.join(out, 'QA', 'membership_conformance_history.csv')])
    page = os.path.join(out, 'review_hub', 'conformance.html')
    neg('conformance_page_count', 'status_page', page, edit('1 of 2 cells PASS', '2 of 2 cells PASS'),
        siblings=[('QA/membership_conformance_latest.csv', lat)], subdir='review_hub')
    # pages
    rc, txt = step(['status', 'pages', '--out', out, '--cores', 'none'])
    rn, rd = rec(out, 'pages_nights_deep'), rec(out, 'pages_dataprocessing_deep')
    check('pages nights + data-processing done; P1-P4 (fake check in place) and P5 (stage-3 windows = membership) pass',
          rc == 0 and rn['status'] == 'done' and rd['status'] == 'done', txt[-800:])
    check('the data page read the hub fragment through the path redirect (not the production hub)',
          not os.path.exists(f'{W}/hub/qa_overview_fragment.html') and
          'fragment' in open(os.path.join(out, 'review_hub', 'data_processing.html')).read())
    rc, txt = step(['status', 'pages', '--stages', 'nights', '--check', '--out', out, '--cores', 'none'])
    check('pages --check: only the night-page content test on the written hub', rc == 0, txt[-300:])
    hub = os.path.join(out, 'review_hub')
    neg('night_pages_missing_page', 'qa_night_pages', hub, lambda p: os.remove(os.path.join(p, 'nights', '05h_2099-01-02.html')))
    os.environ['FAKE_QA3_DRIFT'] = '1'
    bad = failed('qa_night_pages', hub)
    os.environ.pop('FAKE_QA3_DRIFT')
    check('negative: a stage-3 window drawn other than membership.qa3_window fails P5', any('P5' in b or 'stage-3' in b
                                                                                         for b in bad), bad)
    rc, txt = step(['status', 'pages', '--stages', 'nights', '--nights', '05h_2099-01-01', '--out', f'{TMP}/o_cp2',
                    '--cores', 'none'])
    check('pages nights --nights N: --only N, requested page checked', rc == 0 and
          os.listdir(os.path.join(TMP, 'o_cp2', 'review_hub', 'nights')) == ['05h_2099-01-01.html'], txt[-400:])
    neg('data_page_broken_link', 'status_page', os.path.join(hub, 'data_processing.html'),
        edit('</html>', '<a href="nights/99h_2099-01-01.html">x</a></html>'),
        siblings=[('review_hub/nights', os.path.join(hub, 'nights'))], subdir='review_hub')
    # ledger
    rc, txt = step(['status', 'ledger', '--cells', '05h_Full', '--out', out, '--cores', 'none'])
    check('S-G10: ledger with --cells refused', rc == 2 and 'S-G10' in txt, txt[-300:])
    led0 = sha(CFG.status.LEDGER)
    rc, txt = step(['status', 'ledger', '--out', out, '--cores', 'none'])
    led = os.path.join(out, 'clv15', 'processing_ledger.csv')
    check('ledger: inventory -> ledger in place (runpy), both literal paths redirected under --out, content test passes',
          rc == 0 and len(rows(led)) == 6 and sha(CFG.status.LEDGER) == led0 and
          not os.path.exists(CFG.status_ledger.LEGACY_HANDOFF), txt[-600:])
    neg('ledger_dd_count', 'processing_ledger', led, edit('2099-01-02,complete,15,none,0,complete,15,4,,1,',
                                                          '2099-01-02,complete,15,none,0,complete,15,4,,2,'))
    # register
    rc, txt = step(['status', 'register', f'{W}/audit/audit_A.csv', f'{W}/audit/audit_B.csv', '--out', out, '--cores', 'none'])
    reg = os.path.join(out, 'clv15', 'RATIFIED_STEPS_REGISTER.csv')
    check('register: slices merged, statuses upper-cased, sorted (fake builder, attr); register + page tested',
          rc == 0 and [x['ruling_id'] for x in rows(reg)] == ['R-A', 'R-B', 'R-C'], txt[-400:])
    rc, txt = step(['status', 'register', '--out', out, '--cores', 'none'])
    check('register without audit slices refused', rc == 2, txt[-300:])
    neg('register_unsorted', 'ratified_register', reg, lambda p: wcsv(p, list(rows(p)[0].keys()), rows(p)[::-1]))
    neg('register_page_count', 'status_page', os.path.join(out, 'review_hub', 'ratified_steps.html'),
        edit('<td>3</td></tr></table>', '<td>4</td></tr></table>'),
        siblings=[('clv15/RATIFIED_STEPS_REGISTER.csv', reg)], subdir='review_hub')


def test_snapshot():
    out = f'{TMP}/o_sn'
    rc, txt = step(['status', 'eligibility', '--tier', 'snapshot', '--cells-file', f'{W}/snap/cells.txt', '--no-store',
                    '--out', out, '--cores', 'none'])
    el = os.path.join(out, 'snapshot', 'eligibility.csv')
    check('eligibility (fake audit, child py36) done; status precedence re-derived', rc == 0 and len(rows(el)) == 3, txt[-500:])
    rc, txt = step(['status', 'eligibility', '--out', out, '--cores', 'none'])
    check('eligibility is snapshot-only', rc == 2 and 'snapshot-tier' in txt, txt[-300:])
    rc, txt = step(['status', 'batch-list', '--tier', 'snapshot', '--list-out', f'{out}/list.txt', '--out', out,
                    '--cores', 'none'])
    check('S-G8: batch-list without --base refused (the script would ssh to the review host)', rc == 2 and 'S-G8' in txt,
          txt[-300:])
    L = f'{out}/list.txt'
    rc, txt = step(['status', 'batch-list', '--tier', 'snapshot', '--base', f'{W}/snap/base.txt', '--list-out', L,
                    '--max-new', '1', '--out', out, '--cores', 'none'])
    r = rec(out, 'batchlist_batchlist_snapshot')
    check('batch-list: base + 1 new PREFIT-DONE cell (--max-new 1 recorded as an override); sha256 printed = file',
          rc == 0 and open(L).read().split() == ['18h_2099-02-01', '18h_2099-02-02'] and
          any(o[0] == 'status_snapshot.MAX_NEW' for o in r.get('config_overrides', [])), txt[-900:])
    ctx = [o for o in r['outputs'] if o['class'] == 'snapshot_batch_list'][0]['ctx']
    neg('batch_list_status_not_ok', 'snapshot_batch_list', L, lambda p: open(p, 'a').write('18h_2099-02-03\n'), ctx,
        siblings=[L + '.audit.csv', L + '.candidates.txt', L + '.left_out.txt'])
    neg('eligibility_status', 'snapshot_eligibility', el, edit('PREFIT-DONE', 'TO-PREFIT'))


def test_gates_and_dry_run_world():
    out = f'{TMP}/o_g'
    rc, txt = step(['status', '--coverage', '--cells', CELLS, '--out', out, '--dry-run', '--cores', 'none'])
    check('--dry-run plans and writes nothing', rc == 0 and 'PLAN' in txt and not os.path.exists(out), txt[-600:])
    saved = CFG.site.NODE_LOCAL_ROOTS
    CFG.site.NODE_LOCAL_ROOTS = {'otherhost-test': [f'{W}/coadd_sb']}
    try:
        rc, txt = step(['status', '--coverage', '--cells', CELLS, '--out', out, '--cores', 'none'])
    finally:
        CFG.site.NODE_LOCAL_ROOTS = saved
    check('S-G5: input node-local to another host refuses even though the path exists here (a different disk)',
          rc == 2 and 'S-G5' in txt and 'otherhost-test' in txt, txt[-400:])
    rc, txt = step(['status', '--coverage', '--cells', CELLS, '--out', out, '--cores', 'none', '--set', 'paths.COADD_SB_ROOT='])
    check('S-G11: an empty root key refuses (no fallback root)', rc == 2 and 'S-G11' in txt, txt[-400:])
    real_dd = CFG.paths.DATA_DECISIONS
    CFG.paths.DATA_DECISIONS = f'{W}/missing.csv'
    try:
        res = {a: step(['status'] + list(a) + ['--out', out, '--cores', 'none'])
               for a in (('--no-legacy-text',), ('--violations',), ('--cell-load', '--run', 'TESTA'))}
    finally:
        CFG.paths.DATA_DECISIONS = real_dd
    check('S-G1: a missing data_decisions.csv refuses every stage (summary, violations, cell-load)',
          all(rc == 2 and 'C-09' in t for rc, t in res.values()), {k[0]: v[1][-200:] for k, v in res.items()})
    rc, txt = step(['status', 'qa', '--stages', 'outliers', '--out', '/lustre/pipeline/exopipe/phase3/QA/status_test_x',
                    '--cores', 'none'])
    # cli._guard_out (shared cli.py:123-130) raises SystemExit(text): exit 1, not the Refusal exit 2 (reported)
    check('S-G2: an --out under a production root refused before anything runs', rc in (1, 2) and 'production root' in txt and
          not os.path.exists('/lustre/pipeline/exopipe/phase3/QA/status_test_x'), txt[-300:])
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.status import steps_common as K
    for p, gate in (('/lustre/gh/main/clv15/processing_ledger.csv', 'S-G2'), ('/data07/gh/phase3/coadd_sb/x.csv', 'S-G2')):
        try:
            K.check_write(p)
            got = 'not refused'
        except Refusal as e:
            got = str(e)
        check(f'{gate}: planned output {p} refused in the window', got.startswith(gate), got)
    CFG.paths.PRODUCTION_WRITES = True
    try:
        K.check_write('/lustre/gh/main/clv15/data_decisions.csv')
        got = 'not refused'
    except Refusal as e:
        got = str(e)
    finally:
        CFG.paths.PRODUCTION_WRITES = False
    check('S-G3: data_decisions.csv is never written, even after cutover', got.startswith('S-G3'), got)
    rc, txt = step(['status', '--violations', '--tier', 'sublst', '--out', out, '--cores', 'none'])
    check('deep-only stage with --tier sublst refused', rc == 2 and 'deep-only' in txt, txt[-300:])
    rc, txt = step(['status', '--coverage', '--violations', '--emit', f'{TMP}/x.txt', '--out', out, '--cores', 'none'])
    check('--emit with two listing stages is an error (one list per file)', rc != 0 and '--emit' in txt, txt[-300:])


# =====================================================================================================================
# read-only checks on production products and real dry runs
# =====================================================================================================================

def test_real_products():
    Q = '/lustre/pipeline/exopipe/phase3/QA'
    items = [('solar_audit', f'{Q}/solar_contamination_audit.csv', edit('QUARANTINE', 'ok')),
             ('membership_conformance', f'{Q}/membership_conformance_latest.csv', edit(',PASS,0,', ',FAIL,0,')),
             ('bandpass_phase_qa', f'{Q}/bandpass_phase_qa.csv', None),
             ('snapshot_eligibility', '/lustre/gh/main/exopipe_ops/snapshot_batch/batch_audit_20261008T0830.csv',
              edit('TO-PREFIT', 'PREFIT-DONE'))]
    for cls, p, mut in items:
        if not os.path.exists(p):
            print(f'skip {cls}: {p} not found on this host')
            continue
        bad = failed(cls, p)
        check(f'real product passes {cls}: {os.path.basename(p)}', not bad, bad)
        if mut is None:                                    # bandpass: one ratio moved by 0.01 (beyond its rounding bound)
            def mut(q):
                rs = rows(q)
                k = next(i for i, x in enumerate(rs) if x['ratio'] and not x['error'])
                rs[k]['ratio'] = f"{float(rs[k]['ratio']) + 0.01:.4f}"
                wcsv(q, list(rs[0].keys()), rs)
        neg(f'real_{cls}', cls, p, mut)
    page = '/data07/gh/phase3/transients/review_hub/conformance.html'
    if env.readable_here(page) and os.path.exists(page):
        check('real conformance.html summary = the latest membership table', not failed('status_page', page),
              failed('status_page', page))


def test_real_dry_runs():
    out = f'{TMP}/o_real_dry'
    rc, txt = step(['status', '--coverage', '--rebuild-due', '--cells', '04h_Full', '--out', out, '--dry-run', '--cores', 'none'])
    check('real dry run (04h_Full coverage + rebuild-due): plan printed, every stage can run, nothing written',
          rc == 0 and 'every stage can run' in txt and not os.path.exists(out), txt[-1200:])
    print('\n'.join('    ' + ln for ln in txt.splitlines() if ln.strip().startswith(('PLAN', '[', 'gate', 'note')))[:1500])
    qa = CFG.status.QA_DIR
    before = sorted(os.listdir(qa))
    rc, txt = step(['status', 'ledger', '--dry-run', '--cores', 'none'])
    check('ledger --dry-run with the production default --out: the plan shows S-G2, exit 2, nothing written',
          rc == 2 and 'S-G2' in txt and sorted(os.listdir(qa)) == before, txt[-600:])


def test_real_coverage_records():
    """Sub-LST and snapshot coverage on the real state (read only; tables under /tmp): completion comes from the search
    RECORDS, and today they agree with every production done marker (536 sub-LST jobs, 422 snapshot cells, 2026-10-14)."""
    for tier in ('sublst', 'snapshot'):
        out = f'{TMP}/o_real_cov_{tier}'
        rc, txt = step(['status', '--coverage', '--tier', tier, '--out', out, '--cores', 'none'])
        if rc == 2 and 'S-G5' in txt:
            print(f'skip real {tier} coverage: {txt.strip().splitlines()[-1][:200]}')
            continue
        tab = os.path.join(out, 'QA', f'coverage_{tier}.csv')
        rs = rows(tab) if os.path.exists(tab) else []
        dis = [x for x in rs if x['marker_disagrees']]
        done = sum(x['search_V'] == 'done' for x in rs) + sum(x['search_I'] == 'done' for x in rs) if tier == 'sublst' \
            else sum(x['shipped'] == '1' for x in rs)
        check(f'real {tier} coverage done, content test passes; completion from the records ({done}) agrees with every '
              f'done marker ({len(dis)} disagreements)', rc == 0 and rs and not dis and
              rec(out, f'report_coverage_{tier}').get('status') == 'done', (txt[-500:], dis[:3]))


def test_real_summary_parity():
    """The session report on real data under --parity: every write outside --out would be refused and logged."""
    from exopipe.phase3.core import guard
    out = f'{TMP}/o_parity'
    keep = {k: os.environ.get(k) for k in ('PYTHONPATH', guard.ENV_ROOTS, guard.ENV_LOG)}
    fp = CFG.status_report.DECISIONS_FINGERPRINT
    fp0 = (os.path.getmtime(fp), sha(fp)) if os.path.exists(fp) else None
    try:
        rc, txt = step(['status', '--out', out, '--parity', '--cores', 'none'])
    finally:
        guard.uninstall()
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    log = os.path.join(out, '_records', '_parity_guard_refusals.log')
    refused = open(log).read() if os.path.exists(log) else ''
    check('real summary under --parity: done, no write outside --out attempted (guard log empty), production fingerprint '
          'untouched', rc == 0 and not refused.strip() and
          (fp0 is None or (os.path.getmtime(fp), sha(fp)) == fp0), (txt[-600:], refused[:400]))


def test_every_content_test_has_a_negative():
    """Every content test lib/status/tests.py registers was made to fail by at least one negative test above."""
    validate.classes()
    want = {(cls, name) for cls, tests in validate._TESTS.items() for name, fn in tests
            if fn.__module__ == 'exopipe.phase3.lib.status.tests'}
    missing = sorted(want - NEG_TESTS)
    check(f'every status content test ({len(want)}) has a negative test that makes it fail', not missing, missing)


def main():
    with world():
        for t in (test_config_and_help, test_summary, test_coverage_gaps, test_rebuild_due, test_exclusion_conformance,
                  test_violations_stale, test_served_diff_cell_load, test_qa, test_conformance_pages_ledger_register,
                  test_snapshot, test_gates_and_dry_run_world):
            t0 = time.time()
            print(f'--- {t.__name__}')
            t()
            print(f'    ({time.time() - t0:.1f} s)')
    for t in (test_real_products, test_real_dry_runs, test_real_coverage_records, test_real_summary_parity,
              test_every_content_test_has_a_negative):
        t0 = time.time()
        print(f'--- {t.__name__}')
        t()
        print(f'    ({time.time() - t0:.1f} s)')
    n = len(RESULTS)
    print(f'\n{sum(ok for _, ok in RESULTS)} of {n} checks passed in {time.time() - T0:.0f} s')


if __name__ == '__main__':
    try:
        main()
    finally:
        if not os.environ.get('EXOPIPE3_TEST_KEEP'):        # debugging only: keep the scratch tree
            shutil.rmtree(BASE, ignore_errors=True)
