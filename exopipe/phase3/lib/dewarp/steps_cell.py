"""`exopipe3 dewarp` per-cell stages 'apply', 'verify', 'record': the cell unit <LST>_<scope> (spec 1.3, 7.1 items 5-13).

Ports dewarp_lst.sh:36-94 for ONE routed cell, after its LST's measurement:
  acceptance  steps_common.accepted on the cell's FINAL table row (dewarp_lst.sh:37-52; G-A1)
  apply       accepted: phase3_master_dewarp.apply_saved IN PLACE (the saved screen, cubic spline) into
              <GEN_ROOT>/<cell>/gen_mdewarp_rbf (dewarp_lst.sh:56-60; G-A2..G-A5); an exception is a refusal (G-R2: an
              accepted cell whose apply refused is refused, never sent to the not-accepted path as dewarp_lst.sh:63 does).
              G-A12 before and after: the screen and the master it dewarps are the bytes the measurement keyed (md5); a
              measurement without keys is noted and applied (dewarp.SCREEN_KEY_REQUIRED True refuses it); then the
              apply's MDWSTAT record (applied_mdwstat: 'applied' and each new master's md5, size and mtime)
  verify      content test D1-D6 (D7 with --prior-gen-root) = check_mdewarp_gen.py IN PLACE, on the generation itself so its
              INSTALL_CONTENT_OK / APPLIED_SCREEN.json land where the install reads them, plus the install's cubic,
              applied-screen and product gates and the tested-pixel count (G-A6..G-A10); pass -> meta.install
              {kind 'mdewarp', gen} (dewarp_lst.sh:66-68 -> install_cell.sh <cell> mdewarp <gen>)
  record      not accepted: G-R4 (measured on these masters), G-R5 (each leave-one-out master is its parent's), MDWSTAT
              on the promoted masters and their leave-one-out masters, verified (dewarp_lst.sh:119-131; G-W, G-R3; the
              LOO masters as production's header repair stamped them), then the applied=0 row of the archived screen
              record through phase3_install._screen_row IN PLACE (dewarp_lst.sh:72-90; G-R1)
              -> meta.install {kind 'rebuild', gen <REBUILD_ROOT>/<cell>} (dewarp_lst.sh:92)
The promoted marker removal (dewarp_lst.sh:94) is replaced by the unit record. Each attempt keeps the cell's last done
record that recorded MDWSTAT as meta.last_done (keep_last_done), so a rerun that refuses or fails does not erase it.
"""
import os
import shutil

from exopipe.phase3.lib.dewarp import steps_common as SC


def _screen_nanchor(path):
    from astropy.io import fits
    return fits.getheader(path, 'SCREEN_X').get('NANCHOR')


def _tree_size(d):
    n = b = 0
    for dp, _, fs in os.walk(d):
        for f in fs:
            n += 1
            try:
                b += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return n, b


def pb_weights_gate(u, cell, groot):
    """B-30 (review 2026-10-14 m3): a master-dewarped cell's PB maps are its source generation's, resampled (master_dewarp
    apply_saved), and each recomputed pbcorr sibling keeps its source sibling's PBNDATE / PBMISS - so the source siblings' cards
    are the PB weights this apply installs. Their nights without a Phase-2 noise file are counted and a wideband over
    masters_pbmaps.PB_QA_MISSING_MAX_FRAC refuses (lib/masters/pb_qa.py), as the masters pbmaps stage does for a plain cell,
    whose refusal never reaches a dewarped one (G-M21)."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.masters import pb_qa
    scope = cell.split('_', 1)[1]
    fd = os.path.join(groot, 'coadd_final')
    pm = CFG.masters_pbmaps
    plains = [v for v in pm.PB_VARIANTS if os.path.exists(os.path.join(fd, f'{v}_{scope}.fits'))]
    pb_qa.pb_qa_gate(u, cell, scope, pb_qa.pb_qa_missing(fd, scope, plains), pm.PB_QA_MISSING_MAX_FRAC)


def screen_key_gate(u, run, meas, cell, groot):
    """G-A12 - A KEYED SCREEN IS APPLIED ONLY TO THE EXACT MASTER IT WAS MEASURED ON (FOLD_IN 2026-10-26 'STALE SCREENS'; keyed
    by content, steps_common.SCREEN_KEYS): when the measurement keyed its screens, the scope's key row must exist and the saved
    screen and the generation's two measured files (I Taper R-0.75 plain + pbcorr sibling, the ones MD resamples and
    recomputes from) must have the md5 the key recorded - a mismatch always refuses (no setting, no waiver). A measurement that
    keyed no screen (made before the keys existed: every production measurement) is applied as production applies it, with the
    note NO_KEY_NOTE recorded (G-A11's time check still applies to a --measurement), unless dewarp.SCREEN_KEY_REQUIRED, which
    refuses it. Returns (key or None, plain, pbcorr)."""
    lst, scope = cell.split('_', 1)
    scr = meas.screen(scope)
    plain, pbcorr = SC.measured_files(os.path.join(groot, 'coadd_final'), scope)
    key = meas.screen_key(scope)
    if key is None:
        why = (f'{SC.NO_KEY_NOTE} ({meas.dir} has no {os.path.basename(meas.keys())}), so {os.path.basename(scr)} cannot be '
               f'shown to belong to {plain}')
        if SC.C().SCREEN_KEY_REQUIRED:
            u.refuse(f'G-A12: {why} - REFUSED (dewarp.SCREEN_KEY_REQUIRED True)')
        u.meta['screen_key'] = {'status': SC.NO_KEY_NOTE}
        u.note(f'G-A12: {why}; applied without a content check, as production applies it (dewarp.SCREEN_KEY_REQUIRED False)')
        return None, plain, pbcorr
    if not key:
        u.refuse(f'G-A12: {meas.keys()} keys other screens of {lst} but not {scope}: {scr} is not a screen that measurement '
                 f'wrote - REFUSED')
    probs = SC.screen_key_problems(key, scr, plain, pbcorr)
    if probs:
        u.refuse('G-A12: the screen is not keyed to the master it would dewarp, or changed since it was measured ('
                 + '; '.join(probs[:3]) + ') - a mismatch REFUSES, never applies')
    u.meta['screen_key'] = {k: key.get(k) for k in ('screen', 'screen_md5', 'master', 'master_md5', 'pbcorr', 'pbcorr_md5',
                                                     'measured_at')}
    u.note(f'G-A12: screen {os.path.basename(scr)} md5 {key["screen_md5"]} keyed to the master it dewarps ({plain} md5 '
           f'{key["master_md5"]}, pbcorr md5 {key["pbcorr_md5"]})')
    return key, plain, pbcorr


def apply(u, run, roots, cell, meas, row):
    """G-A2, G-A3, G-A12, the PB weights, G-R5 on the source (loo_source_gate) (+ --fresh removal, recorded), then
    phase3_master_dewarp.apply_saved in place, then G-A12 again (the screen and the master unchanged during the apply), then
    the apply's MDWSTAT record (applied_mdwstat). Returns the generation."""
    from exopipe.phase3.lib.dewarp import master_dewarp as MD
    c = SC.C()
    lst, scope = cell.split('_', 1)
    scr = meas.screen(scope)
    if not os.path.isfile(scr):
        u.refuse(f'G-A2: no saved screen {scr} for the accepted cell {cell}')
    try:
        nanc = int(_screen_nanchor(scr))
        ok = nanc == int(row['n_anchors'])
    except (TypeError, ValueError, KeyError):
        nanc, ok = None, False
    if not ok:
        u.refuse(f"G-A2: saved screen {scr} has NANCHOR {nanc}, the final row {row.get('n_anchors')!r} - not this "
                 f"measurement's screen")
    gen = roots.gen_of(cell)
    celldir = os.path.dirname(gen)
    if os.path.isdir(gen) and os.listdir(gen):
        if not run.fresh:
            u.refuse(f'G-A3: {gen} already holds files - refusing to overwrite a written generation (pass --fresh to '
                     f'remove <GEN_ROOT>/{cell} first, as dewarp_lst.sh:57 does)')
    # the source generation and its PB weights BEFORE --fresh removes anything (a refusal keeps the previous generation)
    groot = MD._gen_root(roots.working, roots.rebuild, lst, scope)
    if groot is None:
        u.refuse(f'G-A2: {cell} has no generation root under {roots.rebuild} or {roots.working}')
    key, plain, pbcorr = screen_key_gate(u, run, meas, cell, groot)
    pb_weights_gate(u, cell, groot)
    loo_source_gate(u, cell, groot)
    if run.fresh and os.path.isdir(celldir):
        if os.path.abspath(celldir) != os.path.abspath(os.path.join(roots.gen, cell)):
            raise RuntimeError(f'refusing to remove {celldir}: not <GEN_ROOT>/{cell}')
        n, b = _tree_size(celldir)
        shutil.rmtree(celldir)
        u.meta['removed_previous'] = {'path': celldir, 'files': n, 'bytes': b}
        u.note(f'--fresh: removed the previous {celldir} ({n} files, {b} bytes; dewarp_lst.sh:57)')
    src = []
    for sub in ('coadd_final',) + (('subtracted',) if c.APPLY_RESIDUALS else ()):
        for dp, _, fs in os.walk(os.path.join(groot, sub)):
            src += [os.path.join(dp, f) for f in fs if f.endswith('.fits')]
    for p in sorted(src):
        u.add_input(p)
    u.meta['apply_source_root'] = groot
    u.meta['apply_sources'] = len(src)
    old_env, old_nores = os.environ.get('MDEWARP_OUT_ROOT'), MD.NO_RESIDUALS
    os.environ['MDEWARP_OUT_ROOT'] = roots.gen           # read at MD:264
    MD.NO_RESIDUALS = not c.APPLY_RESIDUALS              # = MD main --no-residuals (MD:441)
    try:
        n = MD.apply_saved(lst, scope, roots.working, roots.rebuild, meas.table, [meas.dir])
    except RuntimeError as e:                            # MD raises RuntimeError on every refusal (G-A2..G-A5)
        u.refuse(f'G-R2: the apply of the ACCEPTED cell {cell} refused ({str(e)[:240]}) - refused, not routed to the '
                 f'not-accepted path (dewarp_lst.sh:63, finding F-1)')
    finally:
        MD.NO_RESIDUALS = old_nores
        if old_env is None:
            os.environ.pop('MDEWARP_OUT_ROOT', None)
        else:
            os.environ['MDEWARP_OUT_ROOT'] = old_env
    if key:                                              # G-A12 again: what the apply read is still what the key names
        probs = SC.screen_key_problems(key, scr, plain, pbcorr)
        if probs:
            u.refuse('G-A12: the screen or the master changed while the apply ran (' + '; '.join(probs[:3]) + ') - the '
                     'generation written is not installable')
        u.check('dewarp.screen_key', True, value=key['master_md5'],
                detail=f'screen md5 {key["screen_md5"]} measured on master md5 {key["master_md5"]} / pbcorr md5 '
                       f'{key["pbcorr_md5"]}; both unchanged through the apply')
    u.meta['n_written'] = n
    u.note(f'apply: {n} files written into {gen} ({c.RESAMPLE_METHOD}; from {groot})')
    applied_mdwstat(u, cell, gen)
    print(f'{lst} {scope}: APPLIED - {n} files written', flush=True)
    return gen


def applied_mdwstat(u, cell, gen):
    """The apply's MDWSTAT record (2026-11-25, PR 176 decision 1: one rule for accepted and not-accepted cells): every
    master of the new generation (steps_common.master_files of <gen>/coadd_final - the masters the install takes) and
    every leave-one-out master (steps_common.loo_files: <gen>/coadd_final/loo, gate G-R5 as the record stage) must carry
    MDWSTAT 'applied' as master_dewarp._apply wrote it (header only, check 'dewarp.mdwstat'); then meta.mdwstat =
    {value 'applied', files [{path, before None, md5, size, mtime_ns}] of the masters, loo_files [the same] of the LOO
    masters}, as the record stage records its stamp, and each is an output of class 'mdwstat_master', so --resume
    re-tests it. The install takes MDWSTAT from here and the md5 of ``files`` is its currency rule (lib/install/mdwstat).
    A generation without a master, a LOO master that is not its parent's, or a file without the card, refuses."""
    from exopipe.phase3.core.record import fits_primary_cards
    value = 'applied'
    fd = os.path.join(gen, 'coadd_final')
    files = SC.master_files(fd)
    if not files:
        u.refuse(f'dewarp.mdwstat: {cell}: the apply wrote no master (coadd._MASTER_RE) into {gen}/coadd_final: the '
                 f'install would take no MDWSTAT')
    loo = SC.loo_files(fd)
    loo_gate(u, cell, fd, value, loo, 'the generation is written, but this unit is refused, so the install takes none of it')

    def entries(paths):
        out = []
        for p in paths:
            got = fits_primary_cards(p).get('MDWSTAT')
            u.check('dewarp.mdwstat', got == value, value=got, path=p,
                    detail=f'{os.path.basename(p)}: MDWSTAT {got!r}' + ('' if got == value else
                                                                       f' (the apply writes {value!r})'))
            if got != value:
                u.refuse(f'dewarp.mdwstat: {cell}: {p} carries MDWSTAT {got!r} after the apply, not {value!r}')
            st = os.stat(p)
            out.append({'path': p, 'before': None, 'md5': SC.md5(p), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns})
        return out
    done, done_loo = entries(files), entries(loo)
    u.meta['mdwstat'] = {'value': value, 'files': done, 'loo_files': done_loo}
    for d in done + done_loo:
        u.add_output(d['path'], 'mdwstat_master', test=False, mdwstat=value, md5=d['md5'])


def loo_source_gate(u, cell, groot):
    """Gate G-R5 on an accepted cell's apply source <groot>/coadd_final, before the apply writes anything: each
    leave-one-out master is its parent master's (lib/masters/loo_link, the search's D-G16 rule; NCOADD; LOOEXCL) with its
    parent's MDWARP, and a parent that needs LOO masters has some (steps_common.loo_problems, ``value`` None). The apply
    resamples coadd_final/loo with the masters, so a stale LOO master here would become a dewarped stale one: refused
    before the resample - no generation written, the previous one (--fresh) kept."""
    fd = os.path.join(groot, 'coadd_final')
    loo_gate(u, cell, fd, None, SC.loo_files(fd), 'refused before the apply: no generation written, the previous one kept')


def loo_gate(u, cell, fd, value, loo, when):
    """Gate G-R5 (steps_common.loo_problems): every leave-one-out master ``loo`` of the coadd_final directory ``fd`` is its
    parent master's (lib/masters/loo_link) and takes ``value``, its masters' MDWSTAT, by the rule production's header
    repair stamped the archived ones with (check 'dewarp.loo_mdwstat'), and each parent that needs LOO masters has some;
    ``value`` None: an apply's source (check 'dewarp.loo_source': MDWARP as the parent's). Else the unit refuses, naming
    each file. ``when``: what the refusal leaves behind."""
    probs, checked = SC.loo_problems(fd, value, loo)
    u.check('dewarp.loo_mdwstat' if value is not None else 'dewarp.loo_source', not probs, value=f'{len(loo)} LOO master(s)',
            path=fd, detail=('; '.join(probs[:6]) + (f'; ... {len(probs) - 6} more' if len(probs) > 6 else '')) if probs
            else checked + (f'; each takes {value!r}' if value is not None else ''))
    if probs:
        what = (f'so MDWSTAT {value!r} cannot be the value of every one' if value is not None else
                'the apply\'s source: the generation would carry a leave-one-out master that is not its parent\'s')
        u.refuse(f'{SC.LOO_GATE}: {cell}: {len(probs)} leave-one-out master problem(s) in {fd}, {what} '
                 f'({"; ".join(probs[:3])}): {when}')


def keep_last_done(run, u, cell):
    """A rerun of the cell that refuses or fails must not erase its last finished record (PR 176 review item 4): the unit
    record of every attempt is written at the same path, so the attempt keeps, as meta.last_done, the newest earlier done
    attempt that recorded MDWSTAT - the earlier record itself when it is such a one, else the meta.last_done it carried -
    with its run, times, meta.mdwstat (the stamp md5 included) and meta.install. The install uses it only while the newest
    attempt is not done or recorded no MDWSTAT of its own, and only while the masters keep its md5
    (lib/install/mdwstat)."""
    from exopipe.phase3.core import record as R
    try:
        prev = R.load_record(run.out_root, 'dewarp', cell)
    except Exception as e:                         # an unreadable record is not kept; the note says so
        u.note(f'the earlier dewarp record of {cell} could not be read ({type(e).__name__}: {e}); no meta.last_done kept')
        return
    if not prev:
        return
    m = prev.get('meta') or {}
    if prev.get('status') == 'done' and m.get('mdwstat') is not None:
        last = {k: prev.get(k) for k in ('run_id', 'commit', 'started', 'finished')}
        last.update(status='done', mdwstat=m.get('mdwstat'), install=m.get('install'))
    else:
        last = m.get('last_done')
    if last:
        u.meta['last_done'] = last


def verify(u, run, args, roots, cell, meas, gen):
    """The dewarped_master content tests on the generation (CMG in 'write' mode) + per-test checks; sets meta.install."""
    from exopipe.phase3.lib.dewarp import tests as T
    c = SC.C()
    prior = None
    if args.prior_gen_root:
        prior = os.path.join(args.prior_gen_root, cell, c.GEN_DIR)
        if not os.path.isdir(prior):
            u.refuse(f'--prior-gen-root: no earlier generation {prior} of {cell}')
        u.meta['prior_gen'] = prior
    # sidecars of an earlier verify are this step's own: removed (and recorded) so a failed CMG cannot leave a stale
    # INSTALL_CONTENT_OK naming this generation for the install's G1
    removed = {}
    for name in (SC.CONTENT_OK, SC.APPLIED_SCREEN):
        p = os.path.join(os.path.dirname(gen), name)
        if os.path.isfile(p):
            removed[p] = open(p).read()[:2000]
            os.remove(p)
    if removed:
        u.meta['removed_sidecars'] = removed
        u.note(f'verify: removed the earlier sidecars {sorted(removed)}')
    waivers = {t.split(':', 1)[1]: r for t, r in run.waivers if t.startswith('dewarped_master.D6_product:')}
    script = SC.content_check_script()
    u.meta['content_check'] = {'script': script, 'sha1': SC.sha1(script)}
    ctx = dict(cell=cell, table=meas.table, meas_root=meas.dir, working_root=roots.working, rebuild_root=roots.rebuild,
               prior_gen=prior, d6_waivers=waivers, mode='write', log_dir=os.path.join(roots.records, 'logs'))
    e = u.add_output(gen, 'dewarped_master', **ctx)
    stored = dict(ctx, mode='scratch')                  # stored for `verify` / --resume: a lookup never rewrites sidecars
    stored.pop('log_dir')
    e['ctx'] = SC.jsonable(stored)
    res = T.cmg_result(gen)
    if res:
        u.meta['content_check'].update(rc=res['rc'], seconds=res['seconds'], log=res.get('log'), argv=res.get('argv'),
                                       child_mismatch=res.get('child_mismatch'),
                                       child_unchecked=res.get('child_unchecked'),
                                       child_branch_modules=res.get('child_branch_modules'))
        u.meta.setdefault('legacy_code', {}).update(res.get('fingerprint') or {})    # as core.legacy.run(unit=u) records
        u.meta.setdefault('legacy_calls', []).append({'argv': res.get('argv'), 'rc': res['rc'],
                                                      'wall_s': res['seconds'], 'log': res.get('log')})
        u.check('dewarp.G-D0b_child_code', not res.get('child_mismatch'),
                value=f"{len(res.get('fingerprint') or {})} child modules",
                detail='; '.join(res.get('child_mismatch') or []) or 'the content-test child ran the code the step '
                       'applied with (production modules equal to the branch, fingerprint checked)',
                cls='dewarped_master', path=gen)
        for t, (ok, msg) in sorted(res['lines'].items()):
            u.check(f'dewarped_master.{t}', ok, value='PASS' if ok else 'FAIL', detail=msg[:400], cls='dewarped_master',
                    path=gen)
        for f in res['d6_waived']:
            u.check(f'dewarped_master.D6_product:{f}', False, value='WAIVED by CMG',
                    detail='per-product D6 failure waived (--waive dewarped_master.D6_product:<file> --ruling ...)',
                    cls='dewarped_master', path=gen)
        for f in res['d6_failed']:
            if f not in res['d6_waived']:
                u.check(f'dewarped_master.D6_product:{f}', False, value='FAIL', detail='per-product D6 failure',
                        cls='dewarped_master', path=gen)
    if not u.failed_tests():
        u.meta['install'] = {'kind': 'mdewarp', 'gen': gen}
        print(f'{cell}: re-measured dewarp applied ({c.RESAMPLE_METHOD}); D1-D{7 if prior else 6} PASS', flush=True)
    else:
        print(f'{cell}: dewarp content test FAILED: ' + '; '.join(t['name'] for t in u.failed_tests()[:4]), flush=True)


def record(u, roots, cell, row, meas=None):
    """G-R1, G-W of the promoted rebuild, G-R3 and G-R4 (the final row was measured on these masters: steps_common.
    measured_master_identity), G-R5 (every leave-one-out master of the promoted coadd_final/loo is its parent master's,
    steps_common.loo_problems), then MDWSTAT on the cell's promoted masters and their leave-one-out masters (the
    same value, as production's header repair stamped the archived LOO masters), each verified (stamp_mdwstat),
    then the applied=0 row (phase3_install._screen_row in place, flock-guarded). The row ends the cell's routing (G-D2), so
    it is written only after every master carries a verified card: a refusal or failure before it leaves the cell routed
    for a rerun (dewarp_lst.sh:119-131 writes the row first, then the card). The record file is install's class
    'screen_table' (crosscheck 4.2: one class, owned by install); this unit's own row is checked inline
    ('dewarp.record_row': exactly one row for the cell, applied=0, the status as written). The stamped masters are outputs
    of class 'mdwstat_master' with the value and md5 the stamp left, so --resume re-tests them (lib/dewarp/tests.py)."""
    import csv
    from exopipe.phase3.core.record import utcnow
    from exopipe.phase3.lib.install import install as INS
    lst, scope = cell.split('_', 1)
    if not roots.archive_run:
        u.refuse('G-R1: paths.ARCHIVE_RUN is empty')
    sdir = INS._screen_dir(roots.archive, roots.archive_run)
    if os.path.abspath(sdir) != os.path.abspath(roots.record_dir):
        u.refuse(f'G-R1: phase3_install._screen_dir gives {sdir}, the section names {roots.record_dir}')
    probe = sdir
    while not os.path.isdir(probe):
        probe = os.path.dirname(probe)
    if not os.access(probe, os.W_OK):
        u.refuse(f'G-R1: the archive record directory {sdir} is not writable ({probe})')
    gw = SC.gate_write_roots(roots, ['record'], stamp=True)     # the run checks it up front only from a --measurement
    if gw:
        u.refuse('G-W: ' + '; '.join(gw))
    files = SC.mdwstat_files(roots, cell)
    if not files:
        u.refuse(f'G-R3: no master (coadd._MASTER_RE) in {os.path.join(roots.rebuild, cell, "coadd_final")}: the masters '
                 f'the install takes would carry no MDWSTAT')
    try:
        value = SC.mdwstat_not_accepted(row)
    except ValueError as e:
        u.refuse(f'G-R3: {cell}: {e}')
    ident, probs = SC.measured_master_identity(meas, roots, cell)
    if probs:
        u.check('dewarp.measured_master', False, value=None, detail='; '.join(probs))
        u.refuse(f'G-R4: {cell}: the final row was measured on another master ({"; ".join(probs)}): a gap that needs a '
                 f're-measure; no MDWSTAT written, no applied=0 row')
    if ident is None:
        u.note(f'G-R4: {cell}: the measurement {getattr(meas, "dir", None)} records no identity of the master it measured '
               f'(no screen key for {scope}): the value rests on G-A11\'s time check alone')
    else:
        u.check('dewarp.measured_master', True, value=ident['master']['md5'],
                detail='; '.join(f'{k} {os.path.basename(v["path"])} = the measured bytes ({v["matched_by"]})'
                                 for k, v in ident.items()))
    u.meta['measured_master'] = ident
    fd = os.path.join(roots.rebuild, cell, 'coadd_final')
    loo = SC.loo_files(fd)
    loo_gate(u, cell, fd, value, loo, 'no MDWSTAT written, no applied=0 row')
    done, done_loo = stamp_mdwstat(u, cell, value, files, loo)
    u.meta['mdwstat'] = {'value': value, 'files': done, 'loo_files': done_loo}
    # the record file is this stage's OUTPUT (one row replaced, INS._screen_row): it is not fingerprinted as an input (it
    # would make the unit stale the moment it is written); the row it replaces is kept in the record instead
    u.meta['screen_row_replaced'] = SC.routing_row(roots, cell)
    r = SC.not_accepted_row(lst, scope, row, utcnow())
    INS._screen_row(sdir, r)
    try:
        from exopipe.phase3.lib.install import tests as _IT  # noqa: F401  (registers install's screen_table class)
    except Exception as ex:
        u.note(f'lib/install/tests.py did not import ({type(ex).__name__}: {ex})')
    u.add_output(roots.record_csv, 'screen_table')      # a class without a registered test raises: never silent
    with open(roots.record_csv, newline='') as f:
        mine = [x for x in csv.DictReader(f) if (x['lst'], x['scope']) == (lst, scope)]
    ok = len(mine) == 1 and mine[0]['applied'] == '0' and mine[0]['status'] == r['status']
    u.check('dewarp.record_row', ok, value=f'{len(mine)} row(s)',
            detail=(f'{cell}: applied=0, status as written' if ok else
                    f'{cell}: rows {[(x["applied"], x["status"][:40]) for x in mine]} != applied=0 {r["status"][:40]!r}'),
            cls='screen_table', path=roots.record_csv)
    u.meta['screen_row'] = SC.jsonable(r)
    print(f'{lst}_{scope}: screen record set applied=0 - {r["status"]}', flush=True)
    for d in done + done_loo:   # verified inline (header only), md5 taken in the write: the class test runs at --resume
        u.add_output(d['path'], 'mdwstat_master', test=False, mdwstat=value, md5=d['md5'])
    u.meta['install'] = SC.install_meta_rebuild(roots, cell)


def _header_block(f, path, max_blocks=400):
    """The primary header of the open file ``f`` (at offset 0) as bytes: its 2880-byte blocks up to the one holding END
    (core.record.fits_primary_cards's reading). A file that does not start with SIMPLE, or has no END, raises ValueError."""
    raw = []
    for n in range(max_blocks):
        blk = f.read(2880)
        if len(blk) < 2880 or (n == 0 and not blk.startswith(b'SIMPLE')):
            break
        raw.append(blk)
        if any(blk[i:i + 8] == b'END     ' for i in range(0, 2880, 80)):
            return b''.join(raw)
    raise ValueError(f'{path}: no primary header ending in END within {max_blocks} blocks')


def _cards(block):
    """The 80-character card images of a header block, up to and without END and the blank fill."""
    out = []
    for i in range(0, len(block), 80):
        c = block[i:i + 80]
        if c[:8] == b'END     ':
            break
        out.append(c)
    while out and not out[-1].strip():
        out.pop()
    return out


def _stage_mdwstat(src, tmp, value, comment):
    """Write a copy of ``src`` to the new file ``tmp`` (a sibling in the same directory) with MDWSTAT = ``value``,
    ``comment`` in its primary header, as fits.setval sets it (astropy Header.set on the parsed header: a card already there
    takes the new value in place, else it is appended; the comment cut to what the card holds); every byte after the
    primary header is copied verbatim with ordinary reads (no memory map). The md5 of the new file is taken from the bytes
    as they are written (no second read). Returns (md5, the original header block, the new header block)."""
    import hashlib
    import warnings
    from astropy.io import fits
    h = hashlib.md5()
    with open(src, 'rb') as f:
        old = _header_block(f, src)
        hdr = fits.Header.fromstring(old.decode('ascii'))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', fits.verify.VerifyWarning)
            hdr.set('MDWSTAT', value, comment)
            new = hdr.tostring(sep='', endcard=True, padding=True).encode('ascii')
        with open(tmp, 'xb') as o:
            o.write(new)
            h.update(new)
            while True:
                b = f.read(16 << 20)
                if not b:
                    break
                o.write(b)
                h.update(b)
            o.flush()
            os.fsync(o.fileno())
    return h.hexdigest(), old, new


def stamp_mdwstat(u, cell, value, files, loo):
    """MDWSTAT = ``value`` (steps_common.mdwstat_not_accepted of the final row), comment dewarp.MDWSTAT_COMMENT, on each of
    the masters ``files`` (steps_common.mdwstat_files: the promoted rebuild of a routed cell that is NOT accepted) and of
    their leave-one-out masters ``loo`` (steps_common.loo_files, cleared by gate G-R5), before the install takes them
    (R-GREGG-20261111-MDWSTAT-CARD; dewarp_lst.sh:119-131). No file is rewritten in place (a concurrent reader keeps the
    file it opened): each is written to a temporary sibling in its own directory (_stage_mdwstat: the card set, the data
    copied verbatim, the md5 taken from the write) and verified from that sibling's header (check 'dewarp.mdwstat'): the
    card reads back as written (one card: a CONTINUE would read back otherwise), every other card is byte-identical to the
    original's, the standard header is valid (validate=True; validate_header reads headers only) and the size is the
    original's. Only when every file of the cell is staged and verified are they placed together by
    core.placement.replace_many (the products' group, mode and ACL; one rename each); a failure before placement removes
    every temporary sibling and leaves every original byte-identical (no half-stamped cell) and refuses the unit - record()
    has written nothing else yet, so the cell stays routed and a rerun stamps every file again. Returns two lists, of the
    masters and of the LOO masters: [{path, before (the value the file held), md5, size (bytes), mtime_ns (after
    placement)}] - the install pre-checks size and mtime at its plan and decides on the md5 at staging
    (lib/install/mdwstat)."""
    from exopipe.common.validate_fits_headers import validate_header
    from exopipe.phase3.core import placement, tempnames
    from exopipe.phase3.core.record import fits_primary_cards
    c = SC.C()
    paths = [(p, False) for p in files] + [(p, True) for p in loo]
    placement.sweep([p for p, _l in paths], tag='mdwstat')    # temporary siblings a killed stamp left (older than an hour)
    staged = []                                 # [(tmp, dst, entry, is_loo)]
    try:
        for p, is_loo in paths:
            name = os.path.basename(p)
            tmp = tempnames.name(p, 'mdwstat')
            try:
                before = fits_primary_cards(p).get('MDWSTAT')
                size = os.path.getsize(p)
                staged.append((tmp, p, None, is_loo))
                md5, old, new = _stage_mdwstat(p, tmp, value, c.MDWSTAT_COMMENT)
            except Exception as e:
                u.check('dewarp.mdwstat', False, value=None, path=p,
                        detail=f'{name}: the stamped copy could not be made ({type(e).__name__}: {e})')
                u.refuse(f'dewarp.mdwstat: {cell}: the copy of {p} with MDWSTAT could not be made ({type(e).__name__}: '
                         f'{e}); none of the {len(paths)} files changed, no applied=0 row written: the cell stays routed')
            with open(tmp, 'rb') as f:                      # read-back: the header of the staged file itself
                back_blk = _header_block(f, tmp)
            back = fits_primary_cards(tmp).get('MDWSTAT')
            mcards = [x for x in _cards(back_blk) if x[:8] == b'MDWSTAT ']
            others = [x for x in _cards(back_blk) if x[:8] != b'MDWSTAT ']
            same = others == [x for x in _cards(old) if x[:8] != b'MDWSTAT '] and back_blk == new
            rep = validate_header(tmp, level='standard')
            size2 = os.path.getsize(tmp)
            ok = rep.ok and back == value and len(mcards) == 1 and same and size2 == size
            u.check('dewarp.mdwstat', ok, value=back, path=p,
                    detail=(f'{name}: MDWSTAT {value!r} staged, every other card unchanged, standard header valid' if ok
                            else f'{name}: MDWSTAT read back {back!r} in {len(mcards)} card(s) (wrote {value!r}); other '
                                 f'cards unchanged {same}; header errors {list(rep.errors)[:3]}; size {size} -> {size2} '
                                 f'bytes'))
            if not ok:
                u.refuse(f'dewarp.mdwstat: {cell}: the stamped copy of {p} failed its check (read back {back!r} in '
                         f'{len(mcards)} card(s), wrote {value!r}; other cards unchanged {same}; header errors '
                         f'{list(rep.errors)[:2]}; size {size} -> {size2} bytes); none of the {len(paths)} files changed, '
                         f'no applied=0 row written: the cell stays routed')
            staged[-1] = (tmp, p, {'path': p, 'before': before, 'md5': md5, 'size': size2}, is_loo)
    except BaseException:
        for tmp, _p, _e, _l in staged:
            try:
                os.remove(tmp)
            except FileNotFoundError:
                pass
        raise
    try:
        placement.replace_many([(tmp, p) for tmp, p, _e, _l in staged])
    except placement.PlacementRefusal as e:
        u.check('dewarp.mdwstat', False, value=None, path=os.path.dirname(paths[0][0]), detail=f'placement: {e}'[:400])
        u.refuse(f'dewarp.mdwstat: {cell}: the stamped files could not be placed ({e}); no applied=0 row written: the cell '
                 f'stays routed')
    done, done_loo = [], []
    for _tmp, p, e, is_loo in staged:
        e['mtime_ns'] = os.stat(p).st_mtime_ns
        (done_loo if is_loo else done).append(e)
    print(f'{cell}: MDWSTAT = {value!r} on {len(done)} master(s) and {len(done_loo)} leave-one-out master(s) of the '
          f'promoted rebuild, placed together', flush=True)
    return done, done_loo


def _resume(run, cell, meas):
    """(ok, why) for --resume: the core record check plus 'the record used THIS measurement' (a re-measured LST - new
    directory, new table - makes every cell record of the LST stale even when the old table is unchanged on disk)."""
    from exopipe.phase3.core import record as R
    if meas is None or not meas.exists():
        return False, 'no current measurement'
    rec = R.load_record(run.out_root, 'dewarp', cell)
    if rec and (rec.get('meta') or {}).get('measurement') != meas.dir:
        return False, f"record used the measurement {(rec.get('meta') or {}).get('measurement')}, this run {meas.dir}"
    return run.resume_check(cell, None, None)


def run_cell(run, args, roots, lst, cell, meas, stages, route_src=None, resume_only=None):
    """The cell unit: acceptance, then apply/verify (accepted) or record (not accepted). ``resume_only`` (the G-D2 reason):
    the cell is no longer routed and is planned only because this step's own record of it is done (--resume): skipped
    when that record is current, refused with the G-D2 reason otherwise."""
    from exopipe.phase3.core import record as R, validate
    c = SC.C()
    scope = cell.split('_', 1)[1]
    if run.resume and not run.fresh:
        ok, why = _resume(run, cell, meas)
        if ok:
            u = run.unit(cell, cell=cell, lst=lst, kind='cell')
            u.status, u.reason = 'skipped-valid', why
            print(f'{cell}: resumed ({why})')
            return u
        if resume_only:
            with run.unit(cell, cell=cell, lst=lst, kind='cell') as u:
                keep_last_done(run, u, cell)
                u.refuse(f'{resume_only}; its earlier dewarp record is not current ({why})')
            return u
    with run.unit(cell, cell=cell, lst=lst, kind='cell') as u:
        keep_last_done(run, u, cell)
        u.meta['stages'] = [s for s in stages if s in SC.CELL_STAGES]
        if meas is None:
            u.refuse(f'measurement unit {lst} not done')
        if not meas.exists():
            u.refuse(f'no measurement table {meas.table}')
        u.add_input(meas.table)
        if os.path.isfile(meas.keys()):                  # the screens' content keys (G-A12)
            u.add_input(meas.keys())
        u.meta['route_source'] = c.ROUTE_SOURCE
        if c.ROUTE_SOURCE == 'accepted_list' and route_src and os.path.isfile(route_src):
            u.add_input(route_src)                       # a hand list: its fingerprint is the routing evidence
        elif route_src:                                  # the shared screen record: the cell's row is the evidence
            u.meta['routing_row'] = SC.routing_row(roots, cell)
        u.meta['measurement'] = meas.dir
        if getattr(args, 'measurement', None):        # an earlier measurement is an input: it must pass its own test
            for r in validate.run('dewarp_table', meas.table, lst=lst):
                u.check(r['name'], r['passed'], value=r['value'], detail=r['detail'], cls='dewarp_table', path=meas.table)
            if u.failed_tests():
                u.refuse(f'--measurement table {meas.table} fails dewarp_table.rows: {u.failed_tests()[0]["detail"]}')
        row = meas.final_row(scope)
        if getattr(args, 'measurement', None):
            why = SC.measurement_currency(roots, cell, row)
            if why:
                u.refuse(why)
        acc = SC.accepted(row, c.HELDOUT_MIN_GAIN_AS)
        u.meta.update(final_row=SC.jsonable(row), accepted=acc)
        verdict = (f"final row {('status ' + row['status'][:60]) if row else 'missing'}, gain "
                   f"{row.get('anchor_off_med_as', '?') if row else '?'} - {row.get('cv_resid_med_as', '?') if row else '?'}"
                   f" arcsec: {'ACCEPTED' if acc else 'not accepted'} (a)+(b') >= {c.HELDOUT_MIN_GAIN_AS:g}")
        u.note(f'G-A1: {verdict}')
        print(f'{cell}: {verdict}', flush=True)
        if acc:
            gen = roots.gen_of(cell)
            scr = meas.screen(scope)
            if os.path.isfile(scr):
                u.add_input(scr)
            if 'apply' in stages:
                gen = apply(u, run, roots, cell, meas, row)
            elif not (os.path.isdir(gen) and os.listdir(gen)):
                u.refuse(f'no generation {gen} to verify (run the apply stage)')
            if 'verify' in stages:
                verify(u, run, args, roots, cell, meas, gen)
            else:
                u.add_output(gen, 'dewarped_master', test=False)
                u.meta['install'] = None
                u.note('generation written but NOT content-tested (--stages without verify): not installable')
        elif 'record' in stages:
            record(u, roots, cell, row, meas)
        else:
            raise R.NoInput(f'{cell}: not accepted ({verdict}); the record stage was not selected')
    return u
