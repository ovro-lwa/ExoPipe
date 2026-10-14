"""The epoch gate on the build's OWN master, with bounded rebuild passes (Gregg 2026-10-13, B-05 option a; ledger
R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32): "the masters step builds, tests every night against its own
new master, and rebuilds without any failing in-master night (a bounded number of passes, each recorded); no carried
record". Design: plan/wave2/B05_DESIGN.md.

Production today runs this loop by hand: the co-add drops the FAIL nights of a CARRIED record (coadd.py:3496-3503, read
without the master check), the install evaluates the new generation and refuses a FAIL night inside the master
(install.py:306-316), and a person carries that record and rebuilds. Here, after the final co-add of pass k:
  1. every night with a Stokes I residual is evaluated against pass k's master (epoch_gate.evaluate_cell);
  2. the in-master FAIL nights are install's own rule (install.py:311): FAIL, weights.csv in_master 1, not a night-image-gate
     FAIL;
  3. none -> the evaluation is the record of this master (coadd_final/epoch_gate.csv) and the build is final;
  4. else pass k's final products move aside (never deleted) to <out>/_epoch_gate_passes/<cell>/pass<k>/, the stage-in
     reference is restored into coadd_final/ (so coadd._select_reference sees the same signature and reuses the stored
     flows - no GPU), a record with every night removed so far marked FAIL is written, and the final co-add runs again
     (the provisional stage reads no epoch-gate record, so its outputs are those of a full rerun);
  5. a night removed in one pass stays out for the rest of the build (monotone: the loop ends), and at
     masters_build.EPOCH_GATE_MAX_PASSES with a FAIL night still in the master the cell is refused, every pass kept.
The record left in coadd_final/ is the plain evaluation against the FINAL master, as the install writes it.
"""
import csv
import os
import shutil

from exopipe.phase3.core.record import Refusal

GATE_FILE = 'epoch_gate.csv'
KEEP_IN_FINAL = ('stage2_verdict.csv',)          # provisional's record (written before the final stage, never by it)
REFERENCE_VARS = ('I_deep_Taper_R075', 'V_deep_Taper_R0')


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def in_master_failures(final_dir, rows):
    """(sorted in-master FAIL nights, sorted FAIL nights) of an evaluation: install.py:309-311's rule."""
    from exopipe.phase3.lib.masters import night_image_gate as NG
    wrows = {r['date']: r for r in csv.DictReader(open(os.path.join(final_dir, 'weights.csv')))}
    try:
        gfail = NG.failed_dates(final_dir)
    except FileNotFoundError:
        gfail = set()
    fails = sorted(d for d, r in rows.items() if r['verdict'] == 'FAIL')
    return sorted(d for d in fails if wrows.get(d, {}).get('in_master') == '1' and d not in gfail), fails


def move_pass_aside(cell_dir, sc, k, out_root, rows):
    """Move pass k's final-stage products out of the cell (to <out>/_epoch_gate_passes/<cell>/pass<k>/) and keep its
    evaluation there; returns that directory. coadd_final/ keeps only provisional's records and the stage-in references
    (restored by restore_reference); subtracted/ moves whole."""
    from exopipe.phase3.lib.masters import epoch_gate as EG
    cell = os.path.basename(cell_dir.rstrip('/'))
    dest = os.path.join(out_root, '_epoch_gate_passes', cell, f'pass{k}')
    if os.path.exists(dest):
        raise Refusal(f'B-05: {dest} exists - an earlier build of this cell left its passes there; move it aside first')
    os.makedirs(os.path.join(dest, 'coadd_final'))
    fd = os.path.join(cell_dir, 'coadd_final')
    for name in sorted(os.listdir(fd)):
        if name in KEEP_IN_FINAL:
            continue
        shutil.move(os.path.join(fd, name), os.path.join(dest, 'coadd_final', name))
    sd = os.path.join(cell_dir, 'subtracted')
    if os.path.isdir(sd):
        shutil.move(sd, os.path.join(dest, 'subtracted'))
    EG.write_record(dest, rows)
    os.replace(os.path.join(dest, GATE_FILE), os.path.join(dest, 'epoch_gate_eval.csv'))
    return dest


def restore_reference(cell_dir, sc):
    """coadd_final/{I_deep_Taper_R075,V_deep_Taper_R0}_<sc>.fits from coadd_final_ref/ (the stage-in copies): the co-add's
    alignment and flux-scale reference of the next pass is the same file as pass 1's. Returns the restored names."""
    done = []
    for var in REFERENCE_VARS:
        src = os.path.join(cell_dir, 'coadd_final_ref', f'{var}_{sc}.fits')
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(cell_dir, 'coadd_final', f'{var}_{sc}.fits'))
            done.append(os.path.basename(src))
    return done


def next_record(rows, removed, k):
    """The record pass k+1's final co-add reads (without the master check): pass k's evaluation with every night removed
    so far (cumulative) marked FAIL."""
    out = {}
    for d, r in rows.items():
        if d in removed and r['verdict'] != 'FAIL':
            r = dict(r, verdict='FAIL', reason=f'removed from membership in an earlier pass (B-05; passes {k} and before)')
        out[d] = r
    return out


def run_passes(u, run_final, cell_dir, sc, out_root, first, max_passes=None):
    """The loop after the first final co-add (``first`` = its result dict). ``run_final()`` reruns the final co-add and
    returns its result dict (None or a 'skipped' status is a refusal of the caller). Returns the last final's result."""
    from exopipe.phase3.lib.masters import epoch_gate as EG
    max_passes = int(max_passes or _cfg().masters_build.EPOCH_GATE_MAX_PASSES)
    if max_passes < 1:
        raise Refusal(f'masters_build.EPOCH_GATE_MAX_PASSES {max_passes}: at least 1')
    fd, sd = os.path.join(cell_dir, 'coadd_final'), os.path.join(cell_dir, 'subtracted')
    removed, passes, fin = set(), [], first
    for k in range(1, max_passes + 1):
        rows = EG.evaluate_cell(fd, sd, sc)
        inm, fails = in_master_failures(fd, rows)
        mmd5 = EG.master_md5(os.path.join(fd, f'I_deep_Taper_R075_{sc}.fits'))
        passes.append(dict(pass_number=k, master_md5=mmd5, nights_evaluated=len(rows), fail=fails, fail_in_master=inm,
                           removed_before=sorted(removed), n_good=(fin or {}).get('n_good'),
                           n_total=(fin or {}).get('n_total')))
        u.meta['epoch_gate_passes'] = passes
        if not inm:
            EG.write_record(fd, rows)
            u.note(f'B-05 epoch gate on the build\'s own master: pass {k} clean (master md5 {mmd5[:8]}; FAIL nights '
                   f'{fails or "none"}, none in the master)' + (f'; removed in earlier passes: {sorted(removed)}'
                                                                if removed else ''))
            return fin
        if k == max_passes:
            raise Refusal(f'B-05: night(s) {inm} in the master still fail the epoch gate after {k} pass(es) '
                          f'(masters_build.EPOCH_GATE_MAX_PASSES {max_passes}; removed so far {sorted(removed)}) - '
                          f'every pass is kept under {os.path.join(out_root, "_epoch_gate_passes")}')
        removed |= set(inm)
        aside = move_pass_aside(cell_dir, sc, k, out_root, rows)
        restored = restore_reference(cell_dir, sc)
        EG.write_record(fd, next_record(rows, removed, k))
        u.note(f'B-05 pass {k}: night(s) {inm} in the master fail the epoch gate against it (master md5 {mmd5[:8]}) - '
               f'leave membership; pass {k} moved to {aside}; reference restored ({", ".join(restored) or "none - fixed-table night"}); '
               f'final co-add pass {k + 1}')
        fin = run_final()
    return fin
