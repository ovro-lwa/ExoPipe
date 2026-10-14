"""`exopipe3 dewarp` per-cell stages 'apply', 'verify', 'record': the cell unit <LST>_<scope> (spec 1.3, 7.1 items 5-13).

Ports dewarp_lst.sh:36-94 for ONE routed cell, after its LST's measurement:
  acceptance  steps_common.accepted on the cell's FINAL table row (dewarp_lst.sh:37-52; G-A1)
  apply       accepted: phase3_master_dewarp.apply_saved IN PLACE (the saved screen, cubic spline) into
              <GEN_ROOT>/<cell>/gen_mdewarp_rbf (dewarp_lst.sh:56-60; G-A2..G-A5); an exception is a refusal (G-R2: an
              accepted cell whose apply refused is refused, never sent to the not-accepted path as dewarp_lst.sh:63 does)
  verify      content test D1-D6 (D7 with --prior-gen-root) = check_mdewarp_gen.py IN PLACE, on the generation itself so its
              INSTALL_CONTENT_OK / APPLIED_SCREEN.json land where the install reads them, plus the install's cubic,
              applied-screen and product gates and the tested-pixel count (G-A6..G-A10); pass -> meta.install
              {kind 'mdewarp', gen} (dewarp_lst.sh:66-68 -> install_cell.sh <cell> mdewarp <gen>)
  record      not accepted: the applied=0 row of the archived screen record through phase3_install._screen_row IN PLACE
              (dewarp_lst.sh:72-90; G-R1) -> meta.install {kind 'rebuild', gen <REBUILD_ROOT>/<cell>} (dewarp_lst.sh:92)
The promoted marker removal (dewarp_lst.sh:94) is replaced by the unit record.
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


def apply(u, run, roots, cell, meas, row):
    """G-A2, G-A3 (+ --fresh removal, recorded), then phase3_master_dewarp.apply_saved in place. Returns the generation."""
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
    pb_weights_gate(u, cell, groot)
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
    u.meta['n_written'] = n
    u.note(f'apply: {n} files written into {gen} ({c.RESAMPLE_METHOD}; from {groot})')
    print(f'{lst} {scope}: APPLIED - {n} files written', flush=True)
    return gen


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


def record(u, roots, cell, row):
    """G-R1, then the applied=0 row (phase3_install._screen_row in place, flock-guarded). The record file is install's class
    'screen_table' (crosscheck 4.2: one class, owned by install); this unit's own row is checked inline ('dewarp.record_row':
    exactly one row for the cell, applied=0, the status as written)."""
    import csv
    from exopipe.phase3.core.record import utcnow
    from exopipe.phase3.lib.install import install as INS
    c = SC.C()
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
    u.meta['install'] = SC.install_meta_rebuild(roots, cell)
    print(f'{lst}_{scope}: screen record set applied=0 - {r["status"]}', flush=True)


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
            u = run.unit(cell, lst=lst, kind='cell')
            u.status, u.reason = 'skipped-valid', why
            print(f'{cell}: resumed ({why})')
            return u
        if resume_only:
            with run.unit(cell, lst=lst, kind='cell') as u:
                u.refuse(f'{resume_only}; its earlier dewarp record is not current ({why})')
            return u
    with run.unit(cell, lst=lst, kind='cell') as u:
        u.meta['stages'] = [s for s in stages if s in SC.CELL_STAGES]
        if meas is None:
            u.refuse(f'measurement unit {lst} not done')
        if not meas.exists():
            u.refuse(f'no measurement table {meas.table}')
        u.add_input(meas.table)
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
            record(u, roots, cell, row)
        else:
            raise R.NoInput(f'{cell}: not accepted ({verdict}); the record stage was not selected')
    return u
