"""`exopipe3 dewarp` stages 'stage' and 'measure': the measurement unit <LST> (spec 1.2-1.3, 7.1 items 2-4, 7.2).

Ports dewarp_lst.sh:18-35 and measure_on_09.sh: stage the two measured files of every scope of the LST (I_deep_Taper_R075 and
its pbcorr sibling, from the first of REBUILD_ROOT / paths.COADD_SB_ROOT holding the plain file, per FILE) to node-local scratch,
md5-verified (G-D6); then phase3_master_dewarp.run_lst IN PLACE on the staged copies, measurement only, into a fresh
<measure root>/<run_id>/ (tables/table_<LST>.csv + <LST>/maps/<scope>_warp_screen.fits + diagnostics). The collection copy
(dewarp_lst.sh:33-35) is dropped: the cell units read the measurement directory it was written to.
Gates here: G-D3 pinned VLSSr files, G-D8 single-thread BLAS, G-D5 excluded constituents, G-D0 in-place values, G-D4 VLSSr
catalogue loads, G-D6 staged copies verified.
"""
import logging
import os
import shutil
import tempfile

from exopipe.phase3.lib.dewarp import steps_common as SC


def sources(roots, lst):
    """[(scope, source coadd_final dir or None, [plain, pbcorr])] for the 19 scopes. STAGE_INPUTS True: the staging rule
    (first root holding the plain file, per FILE: dewarp_lst.sh:24-26). False: phase3_master_dewarp._gen_root's directory rule
    (first of REBUILD_ROOT, paths.COADD_SB_ROOT whose <cell>/coadd_final exists, MD:133-138) - what MD then reads in place."""
    from exopipe.phase3.core import cells
    c = SC.C()
    out = []
    for s in cells.scopes():
        if c.STAGE_INPUTS:
            d = roots.source_dir(lst, s)
        else:
            d = next((os.path.join(r, f'{lst}_{s}', 'coadd_final') for r in (roots.rebuild, roots.working)
                      if os.path.isdir(os.path.join(r, f'{lst}_{s}', 'coadd_final'))), None)
        out.append((s, d, [os.path.join(d, n.format(scope=s)) for n in c.MEASURE_FILES] if d else []))
    return out


def admitted(lst, srcs):
    """G-D5: the constituent nights of every measured master (pb_apply.constituent_dates, in place) against core/exclusions.
    Returns (admitted entries '<scope>:<date>', violations, {scope: n constituents}, exclusions fingerprint)."""
    from exopipe.phase3.core import exclusions
    from exopipe.phase3.lib.masters import pb_apply as PB
    ex = exclusions.load()
    adm, bad, by = [], [], {}
    for s, d, files in srcs:
        if not d or not os.path.isfile(files[0]):
            continue
        dates = PB.constituent_dates(files[0])
        x = ex.excluded_constituents(lst, s, dates)
        if x:
            bad.append(f'G-D5: {lst}_{s} master {files[0]} holds constituent night(s) excluded by data_decisions: {x}')
        by[s] = len(dates)
        adm += [f'{s}:{d_}' for d_ in dates if d_ not in x]
    return sorted(adm), bad, by, ex.fingerprint()


def stage(u, lst, srcs, work):
    """Copy the measured files to node-local scratch, md5-verified (G-D6; dewarp_lst.sh:24-30 checked only rsync's exit
    status). Returns the staging root (<root>/<LST>_<scope>/coadd_final/...)."""
    root = SC.scratch_root(work)
    os.makedirs(root, exist_ok=True)
    sdir = tempfile.mkdtemp(prefix=f'dewarp_stage_{lst}_', dir=root)
    n = 0
    try:
        for s, d, files in srcs:
            if not d:
                u.note(f'stage: {lst}_{s}: no measured master in REBUILD_ROOT or paths.COADD_SB_ROOT (MD records the scope '
                       f'as REFUSED no generation root)')
                continue
            miss = [f for f in files if not os.path.isfile(f)]
            if miss:
                raise SC.refusal(f'G-D6: staging {lst}_{s}: {miss[0]} missing (dewarp_lst.sh:28-29 stops the LST)')
            dst = os.path.join(sdir, f'{lst}_{s}', 'coadd_final')
            os.makedirs(dst)
            for f in files:
                m0 = SC.md5(f)
                t = os.path.join(dst, os.path.basename(f))
                shutil.copy2(f, t)
                if SC.md5(t) != m0:
                    raise SC.refusal(f'G-D6: the staged copy of {f} differs from its source (md5)')
                n += 1
    except BaseException:
        shutil.rmtree(sdir, ignore_errors=True)       # a refused staging leaves no scratch behind
        raise
    u.note(f'stage: {n} files copied to {sdir}, md5-verified')
    return sdir


def _measure_in_place(u, lst, root, input_root, mdir, vl):
    """phase3_master_dewarp.run_lst on one LST, measurement only (measure_on_09.sh:13-14 = MD main :455-457)."""
    import threadpoolctl
    from exopipe.phase3.lib.dewarp import master_dewarp as MD
    c = SC.C()
    # a fresh measurement directory per run (never overwrite a screen); several LSTs of one run share it (one table and
    # one <LST>/ per LST, MD's layout), so the LST's own table and screen directory must not exist yet
    os.makedirs(os.path.join(mdir, 'tables'), exist_ok=True)
    table = os.path.join(mdir, 'tables', f'table_{lst}.csv')
    if os.path.exists(table) or os.path.exists(os.path.join(mdir, lst)):
        raise SC.refusal(f'{lst}: {table} or {os.path.join(mdir, lst)} already exists in this run\'s measurement '
                         f'directory - a measurement is never appended to or overwritten')
    saved = {k: os.environ.get(k) for k in ('BDSF_NCORES',)}
    os.environ['BDSF_NCORES'] = str(c.BDSF_NCORES)     # read by cascade_dewarp.extract_sources (CD:186-187)
    old_meas = MD.MEAS_ROOT
    MD.MEAS_ROOT = mdir                                 # module global read at MD:339 (set from the environment today)
    logging.basicConfig(level=logging.WARNING, format='%(message)s')     # as MD main (MD:432)
    u.note(f'measure: BDSF_NCORES={c.BDSF_NCORES}, MD.MEAS_ROOT={mdir}, root={root}, input_root={input_root}')
    try:
        with threadpoolctl.threadpool_limits(limits=c.MEASURE_THREADS):
            u.meta['threadpools'] = [{k: d.get(k) for k in ('internal_api', 'num_threads', 'filepath')}
                                     for d in threadpoolctl.threadpool_info()]
            MD.run_lst(lst, root, input_root, False, set(), table, vl)
    finally:
        MD.MEAS_ROOT = old_meas
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return table


def run_measure(run, args, roots, lst, cells_, stages):
    """The measurement unit <LST>. Returns the Measurement when the unit is done (or validly resumed), else None."""
    from exopipe.phase3.core import record
    c = SC.C()
    srcs = sources(roots, lst)
    if run.resume and not run.fresh:
        from exopipe.phase3.lib.dewarp import master_dewarp  # noqa: F401  (code identity: resume compares loaded modules)
        from exopipe.phase3.lib.masters import pb_apply  # noqa: F401
        inputs = [f for _, d, files in srcs if d for f in files] + [c.VLSSR_PEAKS, c.VLSSR_CATALOG]
        adm, bad, _, _ = admitted(lst, srcs)
        ok, why = run.resume_check(lst, inputs, adm)
        if ok and not bad:
            rec = record.load_record(run.out_root, 'dewarp', lst)
            mdir = (rec.get('meta') or {}).get('measure_dir')
            if mdir and os.path.isdir(mdir):
                u = run.unit(lst, kind='measure', cells=list(cells_))
                u.status, u.reason = 'skipped-valid', why
                print(f'{lst}: measurement resumed from {mdir} ({why})')
                return SC.Measurement(mdir, lst)
    meas = None
    with run.unit(lst, kind='measure', cells=list(cells_)) as u:
        u.meta.update(stages=list(stages), measure_dir=roots.measure_dir, stage_inputs=c.STAGE_INPUTS)
        bad = SC.gate_pinned_inputs()
        if bad:
            u.refuse('; '.join(bad))
        u.add_input(c.VLSSR_PEAKS)
        u.add_input(c.VLSSR_CATALOG)
        bad = SC.gate_threads()
        if bad:
            u.refuse('; '.join(bad))
        have = [(s, d, files) for s, d, files in srcs if d]
        if not have:
            raise record.NoInput(f'{lst}: no measured master (I_deep_Taper_R075_<scope>.fits) of any scope under '
                                 f'{roots.rebuild} or {roots.working}')
        for _, _, files in have:
            for f in files:
                if os.path.isfile(f):
                    u.add_input(f)
        u.meta['sources'] = {s: (d if d else None) for s, d, _ in srcs}
        adm, bad, by, exfp = admitted(lst, srcs)
        u.set_admitted(adm)
        u.meta.update(exclusions=exfp, constituents_by_scope=by)
        if bad:
            u.refuse('; '.join(bad[:5]))
        from exopipe.phase2 import cascade_dewarp as CD
        bad, n = SC.gate_in_place(import_heavy=True)
        if bad:
            u.refuse('; '.join(bad[:5]))
        u.note(f'G-D0: {n} in-place values equal the section')
        vl = CD.load_vlssr_catalog(c.VLSSR_CATALOG, 'VLSSr')
        if vl is None:
            u.refuse(f'G-D4: the VLSSr catalogue {c.VLSSR_CATALOG} failed to load (phase3_master_dewarp.py:452-454)')
        staging = None
        try:
            if c.STAGE_INPUTS:
                staging = stage(u, lst, srcs, roots.work)
                root, input_root = staging, None
            else:
                root, input_root = roots.working, roots.rebuild
            table = _measure_in_place(u, lst, root, input_root, roots.measure_dir, vl)
        except BaseException:
            if staging:                         # node-local scratch never outlives a refused or failed measurement
                shutil.rmtree(staging, ignore_errors=True)
            raise
        meas = SC.Measurement(roots.measure_dir, lst)
        e = u.add_output(table, 'dewarp_table', lst=lst)
        e['ctx'] = {'lst': lst}
        masters = {s: files[0] for s, d, files in srcs if d}
        for scr in meas.screens():
            scope = os.path.basename(scr)[:-len('_warp_screen.fits')]
            ctx = {'table': table, 'master': masters.get(scope)}
            e = u.add_output(scr, 'dewarp_screen', **ctx)
            e['ctx'] = SC.jsonable(ctx)
        diag = sorted(os.path.relpath(os.path.join(dp, f), roots.measure_dir)
                      for dp, _, fs in os.walk(os.path.join(roots.measure_dir, lst)) for f in fs
                      if not f.endswith('_warp_screen.fits'))
        u.meta['diagnostics'] = diag            # analysis output (MD:123 'not a product'): listed, not outputs (spec Q9)
        if staging:
            shutil.rmtree(staging, ignore_errors=True)
            u.note(f'stage: scratch {staging} removed')
    return meas if u.status == 'done' else None
