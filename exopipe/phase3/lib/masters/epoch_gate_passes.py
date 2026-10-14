"""The epoch gate on the build's OWN master, with bounded rebuild passes (Gregg 2026-10-13, B-05 option a; ledger
R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32): "the masters step builds, tests every night against its own
new master, and rebuilds without any failing in-master night (a bounded number of passes, each recorded); no carried
record; a night kept out by name goes to the rulings table". Design: exopipe_ops/refactor_20261012/plan/wave2/B05_DESIGN.md.

Production today runs this loop by hand: the co-add drops the FAIL nights of a CARRIED record (coadd.py:3496-3503, read
without the master check), the install evaluates the new generation and refuses a FAIL night inside the master
(install.py:306-316), and a person carries that record and rebuilds. Here, after the final co-add of pass k:
  1. every night with a Stokes I residual is evaluated against pass k's master - in the flux-bake frame (master, epoch and
     residual x the cell's SHSCALE x FLUXCORR, float32 as shflux.apply bakes them), the frame of the masters verify T6 and of
     the install: the epoch test is scale-free except its absolute 50 Jy/beam source cap (coadd.py:3169);
  2. the in-master FAIL nights are install's own rule (install.py:309-311): FAIL, weights.csv in_master 1, not a night-image
     gate FAIL;
  3. none -> the evaluation is the record of this master (coadd_final/epoch_gate.csv) and membership is verified again
     with it (membership.verify, as run_final does at its end with the record it read): a night removed in an earlier pass
     that now PASSES against this master (it fails inside the master and passes outside it - an oscillation the gate cannot
     settle; production met it at 17h 18 MHz 2025-05-05 and Gregg ruled the night out by name,
     R-GREGG-20261011-17H18-0505-OPTION-B) is "LEFT OUT of the master but passes every membership ruling": the cell refuses
     naming it - "a night kept out by name goes to the rulings table" (named_out_of_master.csv), or Gregg admits it;
  4. else pass k's final products move aside (never deleted) to <out>/_epoch_gate_passes/<cell>/<run id>/pass<k>/, the
     stage-in reference is restored into coadd_final/ (coadd._select_reference sees the same signature and reuses the
     stored flows - no GPU), a record with every night removed so far marked FAIL is written, and the final co-add runs
     again (the provisional stage reads no epoch-gate record, so its outputs are those of a full rerun);
  5. a removed night stays out for the rest of the build (monotone: the loop ends), and at
     masters_build.EPOCH_GATE_MAX_PASSES with a FAIL night still in the master the cell refuses.
The master dewarp frame cannot be reproduced at build time (the screen is measured after the build): the install's own
evaluation of the dewarped generation stays the cross-check, and its verdict decides (Gregg 2026-10-10, ledger 2140).
"""
import csv
import glob
import os
import shutil

from exopipe.phase3.core.record import Refusal

GATE_FILE = 'epoch_gate.csv'
# coadd_final/ files a rerun keeps: the stage-2 record (the final stage rewrites it from the provisional's top-level copy,
# coadd.py:3805, with the same content)
KEEP_IN_FINAL = ('stage2_verdict.csv',)
REFERENCE_VARS = ('I_deep_Taper_R075', 'V_deep_Taper_R0')


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def evaluate(final_dir, sub_dir, fl, factor=None):
    """epoch_gate.evaluate_cell, with every image multiplied by ``factor`` (float32, as shflux.apply bakes the products)
    when one is given: the evaluation of the baked generation, made before the bake. Same rows as evaluate_cell."""
    from exopipe.phase3.lib.masters import epoch_gate as EG
    if not factor:
        return EG.evaluate_cell(final_dir, sub_dir, fl)
    import numpy as np
    from astropy.io import fits
    from exopipe.phase3.lib.masters import coadd as C
    f32 = np.float32(factor)

    def baked(p):
        return (np.squeeze(fits.getdata(p)).astype(np.float32) * f32).astype(np.float64)
    mp = f'{final_dir}/I_deep_Taper_R075_{fl}.fits'
    m, mmd5 = baked(mp), EG.master_md5(mp)
    rows = {}
    for p in sorted(glob.glob(f'{sub_dir}/*_I_sub_{fl}.fits')):
        g = EG._SUB_RE.match(os.path.basename(p))
        if not g:
            continue
        d, r = g['date'], baked(p)
        ep = f'{sub_dir}/{d}_I_epoch_{fl}.fits'
        if os.path.exists(ep):
            ok, why, med, n = C._epoch_content_arrays(baked(ep), m, r, 'I')
            src = 'epoch file'
        else:
            ok, why, med, n = C._epoch_content_arrays(r + m, m, None, 'I')
            src = 'residual + master'
        rows[d] = dict(i_ratio='' if med is None else f'{med:.4f}', n_src=n, verdict='pass' if ok else 'FAIL',
                       reason='' if ok else why, master_md5=mmd5,
                       method=f'epoch content test ({src}; flux-bake frame x{factor:.5f})')
    return rows


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


def passes_dir(out_root, cell, run_id):
    """<out>/_epoch_gate_passes/<cell>/<run id>/: one directory per build, so a later build of the cell never meets it."""
    return os.path.join(out_root, '_epoch_gate_passes', cell, run_id)


def move_pass_aside(cell_dir, k, dest_root, rows):
    """Move pass k's final-stage products out of the cell to <dest_root>/pass<k>/ and keep its evaluation beside them;
    returns that directory. coadd_final/ keeps KEEP_IN_FINAL and gets the stage-in references back (restore_reference);
    subtracted/ moves whole."""
    dest = os.path.join(dest_root, f'pass{k}')
    if os.path.exists(dest):
        raise Refusal(f'B-05: {dest} exists - refusing to mix two builds\' passes')
    os.makedirs(os.path.join(dest, 'coadd_final'))
    fd = os.path.join(cell_dir, 'coadd_final')
    for name in sorted(os.listdir(fd)):
        if name in KEEP_IN_FINAL:
            continue
        shutil.move(os.path.join(fd, name), os.path.join(dest, 'coadd_final', name))
    sd = os.path.join(cell_dir, 'subtracted')
    if os.path.isdir(sd):
        shutil.move(sd, os.path.join(dest, 'subtracted'))
    save_evaluation(dest, 'epoch_gate_eval.csv', rows)
    return dest


def save_evaluation(dest, name, rows):
    from exopipe.phase3.lib.masters import epoch_gate as EG
    os.makedirs(dest, exist_ok=True)
    EG.write_record(dest, rows)
    os.replace(os.path.join(dest, GATE_FILE), os.path.join(dest, name))
    return os.path.join(dest, name)


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


def run_passes(u, run_final, cell_dir, sc, out_root, first, run_id, membership_check, factor=None, max_passes=None):
    """The loop after the first final co-add (``first`` = its result dict). ``run_final()`` reruns the final co-add and
    returns its result dict; ``membership_check()`` = membership.verify of the cell with the exclusions the co-add used
    ([violation lines]). ``factor`` = the cell's flux-bake factor (None: evaluate the unbaked products). Returns the last
    final's result; raises Refusal at the bound or on a membership violation of the cell's own record."""
    from exopipe.phase3.lib.masters import epoch_gate as EG
    max_passes = int(max_passes or _cfg().masters_build.EPOCH_GATE_MAX_PASSES)
    if max_passes < 1:
        raise Refusal(f'masters_build.EPOCH_GATE_MAX_PASSES {max_passes}: at least 1')
    cell = os.path.basename(cell_dir.rstrip('/'))
    keep = passes_dir(out_root, cell, run_id)
    fd, sd = os.path.join(cell_dir, 'coadd_final'), os.path.join(cell_dir, 'subtracted')
    frame = f'flux-bake frame x{factor:.5f}' if factor else 'unbaked products (no bake factor)'
    removed, passes, fin = set(), [], first
    for k in range(1, max_passes + 1):
        rows = evaluate(fd, sd, sc, factor)
        inm, fails = in_master_failures(fd, rows)
        mmd5 = EG.master_md5(os.path.join(fd, f'I_deep_Taper_R075_{sc}.fits'))
        lost = sorted(removed - set(rows))
        if lost:                    # the residual loop covers every night with flow parameters: a removed night keeps one
            raise Refusal(f'B-05 pass {k}: night(s) {lost} removed in an earlier pass have no Stokes I residual against '
                          f'this master - they cannot be tested again; passes kept under {keep}')
        back = sorted(d for d in removed if rows[d]['verdict'] != 'FAIL')
        rec = dict(pass_number=k, master_md5=mmd5, nights_evaluated=len(rows), fail=fails, fail_in_master=inm,
                   removed_before=sorted(removed), pass_after_removal=back, n_good=(fin or {}).get('n_good'),
                   n_total=(fin or {}).get('n_total'), frame=frame)
        passes.append(rec)
        u.meta['epoch_gate_passes'] = passes
        if not inm or k == max_passes:
            EG.write_record(fd, rows)               # the record of the master the cell holds (also when it refuses)
        if not inm:
            viol = list(membership_check())
            rec['membership_violations'] = viol
            if viol:
                ev = save_evaluation(keep, f'pass{k}_epoch_gate_eval.csv', rows)
                raise Refusal(f'B-05 pass {k}: with the record of its own master the cell breaks membership ({len(viol)}): '
                              + ' | '.join(viol[:5]) + (f' - night(s) {back} fail the epoch gate inside the master and '
                                                         f'pass against the master built without them, an oscillation the '
                                                         f'gate cannot settle' if back else '')
                              + f'. Gregg decides: name them out of the master (named_out_of_master.csv, as '
                                f'R-GREGG-20261011-17H18-0505-OPTION-B) or admit them. The cell holds pass {k} (record '
                                f'coadd_final/{GATE_FILE}, copy {ev}); earlier passes under {keep}')
            u.note(f'B-05 epoch gate on the build\'s own master: pass {k} clean (master md5 {mmd5[:8]}; FAIL nights '
                   f'{fails or "none"}, none in the master; {frame}; membership verified with this record)'
                   + (f'; removed in earlier passes and still failing: {sorted(removed)}' if removed else ''))
            return fin
        if k == max_passes:
            ev = save_evaluation(keep, f'pass{k}_epoch_gate_eval.csv', rows)
            raise Refusal(f'B-05: night(s) {inm} in the master fail the epoch gate against it after {k} pass(es) '
                          f'(masters_build.EPOCH_GATE_MAX_PASSES {max_passes}; removed in earlier passes {sorted(removed)}). '
                          f'The cell holds pass {k} (record coadd_final/{GATE_FILE}, copy {ev}); earlier passes under {keep}')
        removed |= set(inm)
        aside = move_pass_aside(cell_dir, k, keep, rows)
        restored = restore_reference(cell_dir, sc)
        EG.write_record(fd, next_record(rows, removed, k))
        u.note(f'B-05 pass {k}: night(s) {inm} in the master fail the epoch gate against it (master md5 {mmd5[:8]}; '
               f'{frame}) - they leave membership; pass {k} moved to {aside}; reference restored '
               f'({", ".join(restored) or "none - fixed-table night"}); final co-add pass {k + 1}')
        fin = run_final()
    return fin
