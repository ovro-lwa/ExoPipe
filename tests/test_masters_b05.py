"""B-05 (a): the epoch gate on the build's own master, bounded rebuild passes (lib/masters/epoch_gate_passes.py).

Synthetic cells, no co-add: the final co-add is a fake that builds the master from the members the epoch-gate record
leaves (master value = the number of members, residuals per night), epoch_gate.evaluate_cell is replaced by a scenario
that fails a night against a given master, and the membership check is membership.verify's two epoch-gate rules
(membership.py:226-233: a member the record FAILs; a non-member with no other reason that passes it). Each check would fail
if the loop used a carried record, did not re-evaluate, did not restore the stage-in reference before a rerun, let a removed
night back in, accepted a removed night that passes the master built without it, ran past its bound, mixed two builds'
passes, or evaluated in another frame than the baked generation's.
Run: python tests/test_masters_b05.py
"""
import csv
import os
import shutil
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from astropy.io import fits                                          # noqa: E402
from exopipe.phase3.core.record import Refusal                       # noqa: E402
from exopipe.phase3.lib.masters import epoch_gate as EG              # noqa: E402
from exopipe.phase3.lib.masters import epoch_gate_passes as P        # noqa: E402

T = '/tmp/exopipe3_tests_masters_b05'
OUT = os.path.join(T, 'out')
SC = '23MHz'
NIGHTS = ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05']
FAILS, N = [], [0]


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({detail})'))
    if not ok:
        FAILS.append(name)


class Unit:
    def __init__(self):
        self.meta, self.notes = {}, []

    def note(self, s):
        self.notes.append(s)


def wfits(p, val):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    h = fits.Header()
    h['COADDOP'], h['CONSTDIG'] = 'ivw-mean-v3', f'dig{val}'
    fits.writeto(p, np.full((8, 8), float(val), np.float32), h, overwrite=True)


def make_cell(name):
    cd = os.path.join(OUT, name)
    if os.path.exists(cd):                        # stage-in --fresh: a new cell directory, earlier passes stay where they are
        shutil.rmtree(cd)
    ref = 100.0                                   # the stage-in reference (never a build product)
    os.makedirs(f'{cd}/coadd_final', exist_ok=True)
    wfits(f'{cd}/coadd_final_ref/I_deep_Taper_R075_{SC}.fits', ref)
    wfits(f'{cd}/coadd_final_ref/V_deep_Taper_R0_{SC}.fits', ref)
    shutil.copy2(f'{cd}/coadd_final_ref/I_deep_Taper_R075_{SC}.fits', f'{cd}/coadd_final/I_deep_Taper_R075_{SC}.fits')
    shutil.copy2(f'{cd}/coadd_final_ref/V_deep_Taper_R0_{SC}.fits', f'{cd}/coadd_final/V_deep_Taper_R0_{SC}.fits')
    with open(f'{cd}/coadd_final/stage2_verdict.csv', 'w') as f:
        f.write('date,verdict\n' + ''.join(f'{d},pass\n' for d in NIGHTS))
    return cd


def fake_final(cd, seen_refs, never=()):
    """The final co-add: members = every night the record does not FAIL and not in ``never`` (kept out for another reason,
    e.g. a stage-2 outlier); master value = len(members)."""
    def run():
        fd = f'{cd}/coadd_final'
        # the alignment / flux reference the co-add would read (coadd._select_reference)
        seen_refs.append(float(fits.getdata(f'{fd}/I_deep_Taper_R075_{SC}.fits')[0, 0]))
        try:
            bad = EG.failed_dates(fd, check_master=False)
        except FileNotFoundError:
            bad = set()
        members = [d for d in NIGHTS if d not in bad and d not in never]
        wfits(f'{fd}/I_deep_Taper_R075_{SC}.fits', len(members))
        wfits(f'{fd}/V_deep_Taper_R0_{SC}.fits', len(members))
        with open(f'{fd}/weights.csv', 'w') as f:
            f.write('date,in_master,flux_dc\n' + ''.join(f'{d},{int(d in members)},1.0\n' for d in NIGHTS))
        open(f'{fd}/.done', 'w').close()
        for d in NIGHTS:
            wfits(f'{cd}/subtracted/{d}_I_sub_{SC}.fits', 0.0)
        return {'n_good': len(members), 'n_total': len(NIGHTS)}
    return run


def fake_membership(cd, reasons=None, calls=None):
    """membership.verify's epoch-gate rules on the synthetic cell; ``reasons`` = {date: another reason it is out}."""
    reasons = reasons or {}

    def run():
        if calls is not None:
            calls.append(1)
        fd = f'{cd}/coadd_final'
        E = {r['date']: r['verdict'] for r in csv.DictReader(open(f'{fd}/{EG.GATE_FILE}'))}
        out = []
        for r in csv.DictReader(open(f'{fd}/weights.csv')):
            d, member = r['date'], r['in_master'] == '1'
            why = ([reasons[d]] if d in reasons else []) + (['epoch-image check FAIL'] if E.get(d) == 'FAIL' else [])
            if member and why:
                out.append(f'{d}: IN the master but ' + '; '.join(why))
            elif not member and not why:
                out.append(f'{d}: LEFT OUT of the master but passes every membership ruling')
        return out
    return run


def scenario(fail_by_master):
    """evaluate_cell replacement: fail_by_master[master value] = nights that FAIL against that master."""
    def ev(final_dir, sub_dir, fl):
        m = float(fits.getdata(f'{final_dir}/I_deep_Taper_R075_{fl}.fits')[0, 0])
        md5 = EG.master_md5(f'{final_dir}/I_deep_Taper_R075_{fl}.fits')
        bad = fail_by_master.get(m, set())
        return {d: dict(i_ratio='0.5' if d in bad else '1.0', n_src=10, verdict='FAIL' if d in bad else 'pass',
                        reason='synthetic' if d in bad else '', master_md5=md5, method='synthetic') for d in NIGHTS}
    return ev


def run(name, fail_by_master, max_passes, run_id='masters_RUN1', never=(), membership=None):
    cd = make_cell(name)
    refs, calls = [], []
    rf = fake_final(cd, refs, never)
    first = rf()
    mc = membership or fake_membership(cd, {d: 'stage-2 outlier' for d in never}, calls)
    EG.evaluate_cell, saved = scenario(fail_by_master), EG.evaluate_cell
    u = Unit()
    try:
        res = P.run_passes(u, rf, cd, SC, OUT, first, run_id, mc, max_passes=max_passes)
        err = None
    except Refusal as e:
        res, err = None, str(e)
    finally:
        EG.evaluate_cell = saved
    return cd, u, res, err, refs, calls


def weights(cd):
    return {r['date']: r for r in csv.DictReader(open(f'{cd}/coadd_final/weights.csv'))}


def frame_cell(cd, factor=None):
    """A 64x64 cell whose 25 compact master sources are 30 Jy (one night; epoch = master, residual = epoch - master).
    With ``factor`` the products are baked as shflux.apply bakes them (float32 x float32(factor))."""
    rng = np.random.default_rng(5)
    m = rng.normal(0, 0.1, (64, 64))
    for y in (12, 22, 32, 42, 52):
        for x in (12, 22, 32, 42, 52):
            m[y, x] += 30.0
    e = m * 1.02
    imgs = {f'coadd_final/I_deep_Taper_R075_{SC}.fits': m, f'subtracted/2025-01-01_I_epoch_{SC}.fits': e,
            f'subtracted/2025-01-01_I_sub_{SC}.fits': e - m}
    for rel, d in imgs.items():
        d = np.asarray(d, dtype=np.float32)
        if factor:
            d = np.asarray(d, dtype=np.float32) * np.float32(factor)
        os.makedirs(os.path.dirname(f'{cd}/{rel}'), exist_ok=True)
        fits.writeto(f'{cd}/{rel}', d, overwrite=True)
    return f'{cd}/coadd_final', f'{cd}/subtracted'


def main():
    if os.path.exists(T):
        shutil.rmtree(T)
    os.makedirs(T)
    try:
        # 1. clean first pass: one evaluation, record of this master written, membership verified once, nothing moved
        cd, u, res, err, refs, calls = run('c1', {}, 3)
        rec = EG.read_record(f'{cd}/coadd_final')
        check('clean build: one pass, record of this master, membership verified with it, nothing moved aside',
              err is None and len(u.meta['epoch_gate_passes']) == 1 and calls == [1]
              and not os.path.exists(f'{OUT}/_epoch_gate_passes/c1')
              and {r['master_md5'] for r in rec.values()} == {EG.master_md5(f'{cd}/coadd_final/I_deep_Taper_R075_{SC}.fits')},
              (err, u.meta, calls))

        # 2. 01-03 fails against the 5-member master and still fails against the 4-member master: pass 2 is final
        cd, u, res, err, refs, calls = run('c2', {5.0: {'2025-01-03'}, 4.0: {'2025-01-03'}}, 3)
        p = u.meta.get('epoch_gate_passes', [])
        w = weights(cd)
        check('a night failing its own master leaves membership in pass 2; pass 2 clean, membership verified',
              err is None and len(p) == 2 and p[0]['fail_in_master'] == ['2025-01-03'] and p[1]['fail_in_master'] == []
              and w['2025-01-03']['in_master'] == '0' and res['n_good'] == 4 and calls == [1], (err, p, calls))
        check('each pass recorded with its own master md5', len({x['master_md5'] for x in p}) == 2, p)
        check('the rerun aligned to the stage-in reference, not to pass 1 master (reference restored)',
              refs == [100.0, 100.0], refs)
        a = f'{OUT}/_epoch_gate_passes/c2/masters_RUN1/pass1'
        check('pass 1 moved aside under the build\'s run id, never deleted: masters, weights, subtracted, evaluation',
              os.path.exists(f'{a}/coadd_final/I_deep_Taper_R075_{SC}.fits') and os.path.exists(f'{a}/coadd_final/weights.csv')
              and os.path.isdir(f'{a}/subtracted') and os.path.exists(f'{a}/epoch_gate_eval.csv')
              and os.path.exists(f'{a}/coadd_final/.done')
              and float(fits.getdata(f'{a}/coadd_final/I_deep_Taper_R075_{SC}.fits')[0, 0]) == 5.0)
        check('provisional record kept in place across passes', os.path.exists(f'{cd}/coadd_final/stage2_verdict.csv'))
        rec = EG.read_record(f'{cd}/coadd_final')
        check('final record = plain evaluation against the FINAL master (01-03 FAIL against it, a non-member)',
              rec['2025-01-03']['verdict'] == 'FAIL' and {r['master_md5'] for r in rec.values()} == {p[1]['master_md5']},
              rec['2025-01-03'])

        # 3. the oscillation (17h 18 MHz 2025-05-05): 01-03 fails inside the master and passes the master built without it
        cd, u, res, err, refs, calls = run('c3', {5.0: {'2025-01-03'}}, 3)
        p = u.meta.get('epoch_gate_passes', [])
        rec = EG.read_record(f'{cd}/coadd_final')
        check('a removed night that passes the master built without it REFUSES the cell, naming it for the rulings table',
              err is not None and "['2025-01-03']" in err and 'named_out_of_master.csv' in err and 'LEFT OUT' in err
              and len(p) == 2 and p[1]['pass_after_removal'] == ['2025-01-03'] and p[1]['membership_violations'], (err, p))
        check('the refused cell holds pass 2 with the record of its own master; a copy beside the passes',
              {r['master_md5'] for r in rec.values()} == {p[1]['master_md5']} and rec['2025-01-03']['verdict'] == 'pass'
              and os.path.exists(f'{OUT}/_epoch_gate_passes/c3/masters_RUN1/pass2_epoch_gate_eval.csv')
              and os.path.isdir(f'{OUT}/_epoch_gate_passes/c3/masters_RUN1/pass1'), rec)

        # 4. monotone: 01-03 out in pass 1, 01-04 fails against the 4-member master; removed nights stay out (cumulative)
        cd, u, res, err, refs, calls = run('c4', {5.0: {'2025-01-03'}, 4.0: {'2025-01-03', '2025-01-04'},
                                                  3.0: {'2025-01-03', '2025-01-04'}}, 3)
        p = u.meta.get('epoch_gate_passes', [])
        w = weights(cd)
        check('monotone: nights removed in earlier passes stay out (3 passes, 01-03 and 01-04 out, 3 members)',
              err is None and len(p) == 3 and res['n_good'] == 3 and w['2025-01-03']['in_master'] == '0'
              and w['2025-01-04']['in_master'] == '0' and p[2]['removed_before'] == ['2025-01-03', '2025-01-04'],
              (err, p))
        # 4b. the same, but 01-03 passes the 3-member master: refused at the clean pass (the gate cannot settle it)
        cd, u, res, err, refs, calls = run('c4b', {5.0: {'2025-01-03'}, 4.0: {'2025-01-03', '2025-01-04'},
                                                   3.0: {'2025-01-04'}}, 3)
        check('a night removed two passes back that passes the final master refuses the cell',
              err is not None and "['2025-01-03']" in err and len(u.meta.get('epoch_gate_passes', [])) == 3, err)

        # 5. bound: a new in-master failure every pass -> refused at the bound; the cell holds the last pass and its record
        cd, u, res, err, refs, calls = run('c5', {5.0: {'2025-01-01'}, 4.0: {'2025-01-01', '2025-01-02'}}, 2)
        p = u.meta.get('epoch_gate_passes', [])
        rec = EG.read_record(f'{cd}/coadd_final')
        check('refused at EPOCH_GATE_MAX_PASSES with an in-master FAIL left; earlier passes kept; the cell\'s record is '
              'its last pass\'s evaluation',
              err is not None and 'after 2 pass' in err and "['2025-01-02']" in err
              and os.path.isdir(f'{OUT}/_epoch_gate_passes/c5/masters_RUN1/pass1') and len(p) == 2
              and {r['master_md5'] for r in rec.values()} == {p[1]['master_md5']}
              and os.path.exists(f'{OUT}/_epoch_gate_passes/c5/masters_RUN1/pass2_epoch_gate_eval.csv') and calls == [], err)

        # 6. a FAIL night outside the master for another reason never triggers a pass and breaks no membership rule
        cd, u, res, err, refs, calls = run('c6', {5.0: {'2025-01-05'}, 4.0: {'2025-01-05'}}, 3, never=('2025-01-05',))
        p = u.meta.get('epoch_gate_passes', [])
        check('a FAIL night outside the master: recorded, no rebuild pass', err is None and len(p) == 1
              and p[0]['fail'] == ['2025-01-05'] and p[0]['fail_in_master'] == [] and calls == [1], (err, p))

        # 7. any membership violation with the own record refuses (membership.verify decides, not the loop's bookkeeping)
        cd, u, res, err, refs, calls = run('c7', {}, 3, membership=lambda: ['23h 23MHz 2025-01-02: IN the master but x'])
        check('a clean pass whose record breaks membership refuses with the violation', err is not None
              and 'IN the master but x' in err, err)

        # 8. a rebuild of the same cell (new run id) never meets the earlier build's passes; the same run id refuses
        cd, u, res, err, refs, calls = run('c2', {5.0: {'2025-01-03'}, 4.0: {'2025-01-03'}}, 3, run_id='masters_RUN2')
        check('a second build of a cell keeps its passes apart (per run id); the first build\'s pass 1 stays',
              err is None and os.path.isdir(f'{OUT}/_epoch_gate_passes/c2/masters_RUN2/pass1')
              and os.path.isdir(f'{OUT}/_epoch_gate_passes/c2/masters_RUN1/pass1'), err)
        cd, u, res, err, refs, calls = run('c2', {5.0: {'2025-01-03'}, 4.0: {'2025-01-03'}}, 3, run_id='masters_RUN2')
        check('a pass directory that exists is never overwritten (refused)', err is not None and 'exists' in err, err)

        # 9. the frame: evaluate() with the bake factor = evaluate_cell of the generation shflux.apply bakes
        rows0 = EG.evaluate_cell(*frame_cell(f'{T}/f0'), SC)
        check('frame: no factor = evaluate_cell exactly', P.evaluate(*frame_cell(f'{T}/f0'), SC, None) == rows0, rows0)
        for f in (1.0, 1.3, 2.0):
            got = P.evaluate(*frame_cell(f'{T}/f0'), SC, f)
            want = EG.evaluate_cell(*frame_cell(f'{T}/fb{f}', f), SC)
            strip = lambda rows: {d: {k: v for k, v in r.items() if k not in ('method', 'master_md5')} for d, r in rows.items()}
            check(f'frame x{f}: evaluate() = evaluate_cell of the baked products (verdict, ratio, sources)',
                  strip(got) == strip(want) and 'flux-bake frame' in got['2025-01-01']['method']
                  and got['2025-01-01']['master_md5'] == rows0['2025-01-01']['master_md5'], (got, want))
        check('frame matters: x2 lifts the 30 Jy sources over the 50 Jy cap - the unbaked test passes, the baked one fails',
              rows0['2025-01-01']['verdict'] == 'pass' and P.evaluate(*frame_cell(f'{T}/f0'), SC, 2.0)['2025-01-01']['verdict']
              == 'FAIL', rows0)
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
