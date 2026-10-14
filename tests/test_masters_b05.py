"""B-05 (a): the epoch gate on the build's own master, bounded rebuild passes (lib/masters/epoch_gate_passes.py).

Synthetic cells, no co-add: the final co-add is a fake that builds the master from the members the epoch-gate record
leaves (master value = the number of members, residuals per night), and epoch_gate.evaluate_cell is replaced by a scenario
that fails a night against a given master. Each check would fail if the loop used a carried record, did not re-evaluate,
did not restore the stage-in reference before a rerun, let a removed night back in, or ran past its bound.
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
SC = '23MHz'
NIGHTS = ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05']
FAILS = []


def check(name, ok, detail=''):
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
    cd = os.path.join(T, 'out', name)
    ref = 100.0                                   # the stage-in reference (never a build product)
    os.makedirs(f'{cd}/coadd_final', exist_ok=True)
    wfits(f'{cd}/coadd_final_ref/I_deep_Taper_R075_{SC}.fits', ref)
    wfits(f'{cd}/coadd_final_ref/V_deep_Taper_R0_{SC}.fits', ref)
    shutil.copy2(f'{cd}/coadd_final_ref/I_deep_Taper_R075_{SC}.fits', f'{cd}/coadd_final/I_deep_Taper_R075_{SC}.fits')
    shutil.copy2(f'{cd}/coadd_final_ref/V_deep_Taper_R0_{SC}.fits', f'{cd}/coadd_final/V_deep_Taper_R0_{SC}.fits')
    with open(f'{cd}/coadd_final/stage2_verdict.csv', 'w') as f:
        f.write('date,verdict\n' + ''.join(f'{d},pass\n' for d in NIGHTS))
    return cd


def fake_final(cd, seen_refs):
    """The final co-add: members = every night the record does not FAIL; master value = len(members)."""
    def run():
        fd = f'{cd}/coadd_final'
        # the alignment / flux reference the co-add would read (coadd._select_reference)
        seen_refs.append(float(fits.getdata(f'{fd}/I_deep_Taper_R075_{SC}.fits')[0, 0]))
        try:
            bad = EG.failed_dates(fd, check_master=False)
        except FileNotFoundError:
            bad = set()
        members = [d for d in NIGHTS if d not in bad]
        wfits(f'{fd}/I_deep_Taper_R075_{SC}.fits', len(members))
        wfits(f'{fd}/V_deep_Taper_R0_{SC}.fits', len(members))
        with open(f'{fd}/weights.csv', 'w') as f:
            f.write('date,in_master,flux_dc\n' + ''.join(f'{d},{int(d in members)},1.0\n' for d in NIGHTS))
        open(f'{fd}/.done', 'w').close()
        for d in NIGHTS:
            wfits(f'{cd}/subtracted/{d}_I_sub_{SC}.fits', 0.0)
        return {'n_good': len(members), 'n_total': len(NIGHTS)}
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


def run(name, fail_by_master, max_passes):
    cd = make_cell(name)
    refs = []
    rf = fake_final(cd, refs)
    first = rf()
    EG.evaluate_cell, saved = scenario(fail_by_master), EG.evaluate_cell
    u = Unit()
    try:
        res = P.run_passes(u, rf, cd, SC, os.path.join(T, 'out'), first, max_passes=max_passes)
        err = None
    except Refusal as e:
        res, err = None, str(e)
    finally:
        EG.evaluate_cell = saved
    return cd, u, res, err, refs


def main():
    if os.path.exists(T):
        shutil.rmtree(T)
    os.makedirs(T)
    try:
        # 1. clean first pass: one evaluation, record of this master written, nothing moved
        cd, u, res, err, refs = run('c1', {}, 3)
        rec = EG.read_record(f'{cd}/coadd_final')
        check('clean build: one pass, record of this master, nothing moved aside',
              err is None and len(u.meta['epoch_gate_passes']) == 1 and not os.path.exists(f'{T}/out/_epoch_gate_passes/c1')
              and {r['master_md5'] for r in rec.values()} == {EG.master_md5(f'{cd}/coadd_final/I_deep_Taper_R075_{SC}.fits')},
              (err, u.meta))
        # 2. night 01-03 fails against the 5-member master -> pass 2 without it (4 members) is clean
        cd, u, res, err, refs = run('c2', {5.0: {'2025-01-03'}}, 3)
        p = u.meta.get('epoch_gate_passes', [])
        w = {r['date']: r for r in csv.DictReader(open(f'{cd}/coadd_final/weights.csv'))}
        check('a night failing its own master leaves membership in pass 2; pass 2 clean',
              err is None and len(p) == 2 and p[0]['fail_in_master'] == ['2025-01-03'] and p[1]['fail_in_master'] == []
              and w['2025-01-03']['in_master'] == '0' and res['n_good'] == 4, (err, p))
        check('each pass recorded with its own master md5', len({x['master_md5'] for x in p}) == 2, p)
        check('the rerun aligned to the stage-in reference, not to pass 1 master (reference restored)',
              refs == [100.0, 100.0], refs)
        a = f'{T}/out/_epoch_gate_passes/c2/pass1'
        check('pass 1 moved aside, never deleted: its masters, weights, subtracted and evaluation',
              os.path.exists(f'{a}/coadd_final/I_deep_Taper_R075_{SC}.fits') and os.path.exists(f'{a}/coadd_final/weights.csv')
              and os.path.isdir(f'{a}/subtracted') and os.path.exists(f'{a}/epoch_gate_eval.csv')
              and float(fits.getdata(f'{a}/coadd_final/I_deep_Taper_R075_{SC}.fits')[0, 0]) == 5.0)
        check('provisional record kept in place across passes', os.path.exists(f'{cd}/coadd_final/stage2_verdict.csv'))
        rec = EG.read_record(f'{cd}/coadd_final')
        check('final record = plain evaluation against the FINAL master (01-03 passes it, stays a non-member)',
              rec['2025-01-03']['verdict'] == 'pass' and w['2025-01-03']['in_master'] == '0'
              and {r['master_md5'] for r in rec.values()} == {p[1]['master_md5']}, rec['2025-01-03'])
        # 3. monotone: 01-03 out in pass 1; against the 4-member master 01-04 fails; against the 3-member master 01-03
        #    would pass again - it must stay out (the loop ends)
        cd, u, res, err, refs = run('c3', {5.0: {'2025-01-03'}, 4.0: {'2025-01-04'}}, 3)
        p = u.meta.get('epoch_gate_passes', [])
        w = {r['date']: r for r in csv.DictReader(open(f'{cd}/coadd_final/weights.csv'))}
        check('monotone: nights removed in earlier passes stay out (3 passes, 01-03 and 01-04 out, 3 members)',
              err is None and len(p) == 3 and res['n_good'] == 3 and w['2025-01-03']['in_master'] == '0'
              and w['2025-01-04']['in_master'] == '0' and p[2]['removed_before'] == ['2025-01-03', '2025-01-04'],
              (err, p))
        # 4. bound: a new in-master failure every pass -> refused at the bound, every pass kept
        cd, u, res, err, refs = run('c4', {5.0: {'2025-01-01'}, 4.0: {'2025-01-02'}}, 2)
        check('refused at EPOCH_GATE_MAX_PASSES with an in-master FAIL left; passes kept',
              err is not None and 'after 2 pass' in err and os.path.isdir(f'{T}/out/_epoch_gate_passes/c4/pass1')
              and len(u.meta.get('epoch_gate_passes', [])) == 2, err)
        # 5. a non-member FAIL (not in the master) never triggers a pass
        cd = make_cell('c5')
        refs = []
        rf = fake_final(cd, refs)
        first = rf()
        with open(f'{cd}/coadd_final/weights.csv', 'w') as f:      # 01-05 outside the master (e.g. a stage-2 outlier)
            f.write('date,in_master,flux_dc\n' + ''.join(f'{d},{int(d != "2025-01-05")},1.0\n' for d in NIGHTS))
        EG.evaluate_cell, saved = scenario({5.0: {'2025-01-05'}}), EG.evaluate_cell
        u = Unit()
        try:
            P.run_passes(u, rf, cd, SC, os.path.join(T, 'out'), first, max_passes=3)
            err = None
        except Refusal as e:
            err = str(e)
        finally:
            EG.evaluate_cell = saved
        p = u.meta.get('epoch_gate_passes', [])
        check('a FAIL night outside the master: recorded, no rebuild pass', err is None and len(p) == 1
              and p[0]['fail'] == ['2025-01-05'] and p[0]['fail_in_master'] == [], (err, p))
    finally:
        shutil.rmtree(T, ignore_errors=True)
    n = 11
    print(f'{n - len(FAILS)} of {n} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
