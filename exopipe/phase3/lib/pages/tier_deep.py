"""Deep tier of `exopipe3 pages` (production: the page part of ops/p3s3_pass.sh, launched by fork_serve_p3s5_20261008.sh).

Unit of work (one record each): <label>:stagein (only with --from-run), <label>:tess, <label>:input:<K>, <label>:build:<K>,
<label>:merge, <label>:strat (on request); K = page set V_wb V_sb I_wb I_sb V_fwb V_fsb I_fwb I_fsb. Run directory
U = <out>/Run_<label> (production CFG.paths.TRANSIENTS_ROOT/Run_P3S5 on lwacalim07).

Stages (in-place calls through core.legacy.run; the order is fixed):
  tess    ops/tess/tess_coverage.py --run U (site PYTHON['tess']); an annotation, never a gate (p3s3_pass.sh:62-63)
  input   singles: lib/pages/pages_input.singles (port of p3s3_pass.sh:137-169); fork: ops/p3s3_tier_pages_input.py U K
          'multiples fork' (p3s3_pass.sh:174-175); a fork set without its catalogue directory is NO-INPUT (p3s3_pass.sh:174)
  build   ops/build_i14_pages_std7.py source_catalogs/pages_input_<K>.csv U <shard> <N>, N processes at once (p3s3_pass.sh:
          180-190) with the production pass environment; queue_i14.txt renamed queue_p3s3_<K>.txt (singles); the panel plan
          (lib/pages/panels.py) is the unit's planned outputs. Incremental by construction: build_i14 keeps a page whose
          fingerprint is unchanged (renamed to its new rank, touched) and renders only new or changed pages; --full (or
          pages.PAGE_FINGERPRINT_SKIP False) removes the set's sidecars first so every page renders.
  stale   ops/p3s3_stale_panels.py U <pass start> in place (p3s3_pass.sh after every set's pages, before the merge; exopipe_ops
          8826836): in each page folder the pass wrote, panels older than the pass start are moved aside, never deleted, with a
          sha256 manifest - the server shows EVERY panel in a folder (2026-10-14: 32 panels of since-excluded nights served).
          The pass start is the earliest recorded start of the current build units (a --resume keeps an earlier invocation's)
  merge   lib/pages/merge.run (ops/p3s3_merge_review_dir.py, pages.REVIEW_DIR_MERGE['deep']; or the M14 keyed merge)
  strat   ops/deep_stratified_queue.py <S> <per stratum> <seed> (on request; the script hard-codes the production run dir)
Localization tables are catalog's (catalog --stages localize, crosscheck decision 6): build reads them (gate P4) and draws
the astrometry figures (inside build_i14).
"""
import glob
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from exopipe.phase3.lib.pages import common, mirror, panels, pages_input as PI

TIER = 'deep'


def add_arguments(p):
    g = p.add_argument_group('deep tier (--tier deep)')
    g.add_argument('--deep-parent-dir', metavar='DIR',
                   help='P3_PARENT_DIR of the pass: the parent-catalogue snapshot the catalog stage used (production: '
                        'fork_serve_p3s5_20261008.sh:15, /fast/gh/main/phase3/p3s5_parents_union_<UTC>); required by build '
                        '(the astrometry figures read it) unless pages_deep.PARENT_DIR is set')
    g.add_argument('--deep-roots-map', metavar='CSV',
                   help='masters map the builder reads (P3S2_ROOTS_CSV; default pages.MASTER_ROOTS_MAP); a parity mirror '
                        'passes its own')
    g.add_argument('--deep-coadd-root', metavar='DIR',
                   help='P3_COADD_ROOT for gate_inputs.parents\' master digest check in the figures (default: not set = '
                        'gate_inputs.py:20 default = CFG.paths.COADD_SB_ROOT)')
    g.add_argument('--deep-strat-per-stratum', type=int, metavar='N', help='strat: identities drawn per stratum (required)')
    g.add_argument('--deep-strat-seed', type=int, metavar='N', help='strat: random seed (default pages_deep.STRAT_SEED)')


class Ctx:
    pass


def context(args, run):
    C, D = common.CFG().pages, common.CFG().pages_deep
    label = getattr(args, 'run_label', None)
    if not label:
        raise common.Refusal('P1: --run <label> is required for the deep tier (no default label; ledger 966, E-6)')
    if not re.match(r'^[A-Za-z0-9_]+$', label):
        raise common.Refusal(f'--run {label!r}: a run label is letters, digits and underscores (P3S5)')
    c = Ctx()
    c.label, c.out = label, run.out_root
    c.U = os.path.join(run.out_root, C.RUN_DIR['deep'].format(label=label))
    c.roots = common.legacy_roots(args.legacy_root)
    c.stages = common.select_stages(args, TIER)
    c.sets = common.select_sets(args, TIER)
    c.t0 = int(time.time())          # P3S3_PASS_T0: ONE value for every set of this invocation (ledger 1457)
    c.layout = C.PANEL_DIR_LAYOUT
    if c.layout not in ('shared', 'per-set'):
        raise common.Refusal(f'pages.PANEL_DIR_LAYOUT {c.layout!r}: shared or per-set')
    if c.layout == 'per-set' and C.DEEP_MERGE != 'keyed':
        raise common.Refusal('pages.PANEL_DIR_LAYOUT per-set needs pages.DEEP_MERGE keyed (M14 items 1 and 2 go together)')
    c.shards = args.shards or C.DEEP_PAGE_SHARDS
    c.full = bool(args.full) or not C.PAGE_FINGERPRINT_SKIP
    c.epochs = common.epochs_of(args)
    c.cells = None
    if getattr(args, 'cells', None):
        from exopipe.phase3.core import cells as _cells
        c.cells = _cells.expand_cells(args.cells)
    c.only = None
    if args.only_pages:
        with open(args.only_pages) as f:
            c.only = set(f.read().split())
    c.parent = args.deep_parent_dir or D.PARENT_DIR
    c.roots_map = args.deep_roots_map or C.MASTER_ROOTS_MAP
    c.coadd_root = args.deep_coadd_root
    c.from_run = args.from_run
    if c.from_run and 'stagein' not in c.stages:
        c.stages = ['stagein'] + c.stages
    return c


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------

def unit_name(c, stage, K=None):
    return f'{c.label}:{stage}' + (f':{K}' if K else '')


def build_env(c, K, only_file=None):
    C, D = common.CFG().pages, common.CFG().pages_deep
    env = dict(D.PASS_ENV)
    env.update(P3S_LABEL=c.label, P3S3_PASS_T0=str(c.t0), P3S2_ROOTS_CSV=c.roots_map,
               P3S3_QUEUE_NAME=(C.QUEUE_NAMES['fork'].format(K=K) if K in C.FORK_SETS else C.QUEUE_NAMES['singles_build']))
    if c.parent:
        env['P3_PARENT_DIR'] = c.parent
    if C.PAGE_SORT_KEY == 'snr':
        env['P3_SORT_SNR'] = '1'
    if not C.PAGE_CTX_IN_TEXT:
        env['P3_PAGES_CTX_IN_TEXT'] = '0'
    if c.coadd_root:
        env['P3_COADD_ROOT'] = c.coadd_root
    if only_file:
        env['P3_PAGES_ONLY_FILE'] = only_file
    return env


def build_env_key(c, K):
    """The build environment that identifies a build unit for --resume: build_env minus the per-invocation pass start and
    the partial-render file (P3_PARENT_DIR, the roots map, the coadd root, the label, sort and queue switches stay)."""
    return {k: v for k, v in build_env(c, K).items() if k not in ('P3S3_PASS_T0', 'P3_PAGES_ONLY_FILE')}


def sample_paths():
    """The cross-match sample CSVs build_i14 reads: the footer samples (CATS, build_i14:35-36, 723-736) and the distance table
    (build_i14:471-477, the same ten names as build_tenmin's CATS = pages.XMATCH_SAMPLES['sublst']). Existing files only: a
    missing file never matches a fingerprint (record.same_fingerprint), and the builder skips an absent sample."""
    C = common.CFG().pages
    names = list(dict.fromkeys(tuple(C.XMATCH_SAMPLES['deep']) + tuple(C.XMATCH_SAMPLES['sublst'])))
    return [p for p in (os.path.join(C.XMATCH_SAMPLE_DIR, f'{n}.csv') for n in names) if os.path.isfile(p)]


def build_resume(c, run_, K):
    """(ok, why): may the build unit of set K be kept under --resume / --epochs? common.resume_skip (record done; recorded
    inputs, step code, in-place code, config unchanged; outputs pass now) AND what the record's file list cannot show:
    the same build environment (build_env_key: a new --deep-parent-dir or roots map re-renders, as every production pass
    does) and a FULL render on both sides (a --cells / --only-pages record never stands for the set; --full never resumes)."""
    from exopipe.phase3.core import record
    if c.full or c.only is not None or c.cells:
        return False, 'partial (--cells / --only-pages) or --full build: never resumed'
    name = unit_name(c, 'build', K)
    ok, why = common.resume_skip(run_, name, enabled=getattr(c, 'build_resume', None))
    if not ok:
        return ok, why
    meta = (record.load_record(run_.out_root, 'pages', name) or {}).get('meta') or {}
    if meta.get('partial'):
        return False, 'the record is of a partial render (--cells / --only-pages)'
    if meta.get('build_env') != build_env_key(c, K):
        return False, 'build environment changed (P3_PARENT_DIR / roots map / coadd root / switches)'
    return True, why


def would_render(c, K):
    """The build of set K would run build_i14 (not NO-INPUT): its catalogue directory exists and its pages input has a row."""
    return (os.path.isdir(os.path.join(c.U, f'source_catalogs_{K}'))
            and common.n_rows(os.path.join(c.U, 'source_catalogs', f'pages_input_{K}.csv')) >= 1)


def shared_resume_guard(c, run_):
    """Shared layout (pages.PANEL_DIR_LAYOUT 'shared', production): every set writes into candidates/I/<page id> and
    build_i14 deletes each same-(LST, night, band) panel older than P3S3_PASS_T0 (build_i14:978-987), so ONE pass start must
    cover every set that is rendered: a set kept by --resume while another is rebuilt would lose its shadowed panels and,
    when the rebuilt set comes EARLIER in the build order, the server would show the rebuilt set's panel where a full
    production pass shows the later set's (classify_transients.py:595-616, newest wins). So under --resume / --epochs the
    build units are kept all together or not at all: when any set would render, every selected set is rebuilt
    (unchanged pages are kept by build_i14's fingerprint skip and touched, as every production pass does)."""
    c.build_resume, c.resume_withheld = None, ''
    if c.layout != 'shared' or not run_.resume or run_.fresh:
        return
    keep = [K for K in c.sets if build_resume(c, run_, K)[0]]
    render = [K for K in c.sets if K not in keep and would_render(c, K)]
    if keep and render:
        c.build_resume = False
        c.resume_withheld = (f'--resume withheld for every build unit: {",".join(render)} must render and the shared layout '
                             f'needs one pass start for every set ({",".join(keep)} would otherwise keep panels older than '
                             f'P3S3_PASS_T0 that build_i14 deletes; build_i14:978-987)')


def loc_tables(c, base=None):
    return [os.path.join(base or c.U, t) for t in common.CFG().pages_deep.LOC_TABLES]


def loc_ndet(c):
    p = os.path.join(c.U, 'source_catalogs', 'localizations.csv')
    out = {}
    if os.path.exists(p):
        for r in common.read_csv(p)[1]:
            try:
                out[r['identity_id']] = int(r.get('n_det') or 0)
            except ValueError:
                pass
    return out


def set_inputs(c, K):
    ins = [os.path.join(c.U, 'master_catalog.csv')]
    ins += sorted(glob.glob(os.path.join(c.U, f'source_catalogs_{K}', '*_sources_survivors.csv')))
    return [p for p in ins if os.path.exists(p)]


def only_ids(c, plan_full):
    """The page ids of a partial re-render (--only-pages FILE, or --cells: pages with a panel in those cells), or None."""
    if c.only is not None:
        return set(c.only)
    if c.cells:
        return set(plan_full.pages_in_cells(c.cells))
    return None


def _dones(log):
    rendered = kept = 0
    for line in common_tail_lines(log):
        m = re.search(r'DONE shard \d+/\d+ \((\d+) pages: (\d+) rendered, (\d+) unchanged and kept\)', line)
        if m:
            rendered, kept = int(m.group(2)), int(m.group(3))
    return rendered, kept


def common_tail_lines(path, n=20):
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 16384))
            return f.read().decode('utf-8', 'replace').splitlines()[-n:]
    except OSError:
        return []


# ---------------------------------------------------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------------------------------------------------

def stage_stagein(c, run, done):
    """--from-run SRC: the run directory under --out from an existing run (copies of the small files the stages rewrite,
    links to the read-only trees); refuses to mix into an existing run directory unless --fresh."""
    C = common.CFG().pages
    name = unit_name(c, 'stagein')
    src = os.path.abspath(c.from_run)
    with run.unit(name, tier=TIER, stage='stagein', source=src) as u:
        if not os.path.isdir(src):
            raise common.Refusal(f'--from-run {src}: not a directory')
        if os.path.realpath(src) == os.path.realpath(c.U):
            raise common.Refusal('--from-run names the run directory itself')
        todo = [(e, 'copy') for e in C.FROM_RUN_COPY] + [(e, 'link') for e in C.FROM_RUN_LINK]
        present = [e for e, _ in todo if os.path.lexists(os.path.join(c.U, e))]
        if present and not run.fresh:
            raise common.Refusal(f'--from-run: {len(present)} entries already exist under {c.U} (e.g. {present[:3]}); pass '
                                 f'--fresh to replace them')
        os.makedirs(c.U, exist_ok=True)
        for e, how in todo:
            s, d = os.path.join(src, e), os.path.join(c.U, e)
            if not os.path.lexists(s):
                u.note(f'{e}: absent in the source run')
                continue
            u.add_input(s)
            if os.path.lexists(d):
                shutil.rmtree(d) if (os.path.isdir(d) and not os.path.islink(d)) else os.remove(d)
            if how == 'link':
                os.symlink(os.path.realpath(s), d)
            elif os.path.isdir(s):
                shutil.copytree(s, d, symlinks=True)
            else:
                shutil.copy2(s, d)
            u.add_output(d, 'pages_run_copy', source=s)
        done[name] = 'done'


def stage_tess(c, run, done):
    D = common.CFG().pages_deep
    name = unit_name(c, 'tess')
    ok, why = common.resume_skip(run, name)
    if ok:
        common.mark_skipped(run, name, why)
        done[name] = 'resumed'
        return
    with run.unit(name, tier=TIER, stage='tess') as u:
        if not D.TESS_ENABLED:
            raise common.NoInput('pages_deep.TESS_ENABLED is False')
        mirror.gate(u, c.roots, TIER, 'tess')
        script = common.legacy_script(c.roots, 'ops', 'tess/tess_coverage.py')
        # inputs: the detections, their frame times and the cached TESS orbit table beside the script (tess_coverage.py:38;
        # the script refreshes it when older than 7 days) - a refreshed table changes the coverage, so it is resume identity
        for p in (os.path.join(c.U, 'master_catalog.csv'), os.path.join(c.U, 'frame_times.csv'),
                  os.path.join(os.path.dirname(script), 'TESS_orbit_times.csv')):
            if os.path.exists(p):
                u.add_input(p)
        if not common.interpreter_or_none('tess'):
            msg = ("TESS COVERAGE NOT COMPUTED: the site PYTHON['tess'] interpreter is not on this host (the TESS environment "
                   "is node-local, p3s3_pass.sh:62) - the pages say 'coverage not computed'")
            if D.TESS_IS_GATE:
                raise common.Refusal(msg)
            raise common.NoInput(msg)
        args = ['--run', c.U] + ([] if D.TESS_STOKES == 'I,V' else ['--stokes', D.TESS_STOKES])
        res = common.lrun(u, script, args, python='tess', cwd=c.U, log=os.path.join(c.U, 'pages_tess.log'))
        if not res.ok:
            msg = f"TESS COVERAGE FAILED - the pages will say 'coverage not computed' ({res.reason})"
            if D.TESS_IS_GATE:
                raise common.Refusal(msg)
            raise common.NoInput(msg)
        u.add_output(os.path.join(c.U, 'tess_coverage.csv'), 'tess_coverage')
        done[name] = 'done'


def stage_input(c, run, done, K):
    C, D = common.CFG().pages, common.CFG().pages_deep
    name = unit_name(c, 'input', K)
    ok, why = common.resume_skip(run, name)
    if ok:
        common.mark_skipped(run, name, why)
        done[name] = 'resumed'
        return
    dst = os.path.join(c.U, 'source_catalogs', f'pages_input_{K}.csv')
    with run.unit(name, tier=TIER, stage='input', set=K) as u:
        if not os.path.exists(os.path.join(c.U, 'master_catalog.csv')):
            raise common.Refusal(f'P6: no {c.U}/master_catalog.csv (catalog writes it)')
        common.FPCache().add(u, set_inputs(c, K))
        if K in C.FORK_SETS:
            if not os.path.isdir(os.path.join(c.U, f'source_catalogs_{K}')):
                raise common.NoInput(f'no source_catalogs_{K}/ (the multiples fork wrote no {K} tier; p3s3_pass.sh:174)')
            mirror.gate(u, c.roots, TIER, 'input')
            script = common.legacy_script(c.roots, 'ops', 'p3s3_tier_pages_input.py')
            res = common.lrun(u, script, [c.U, K, D.FORK_PAGES_LABEL], env={'P3S_LABEL': c.label}, cwd=c.U,
                              log=os.path.join(c.U, f'pages_input_{K}.log'))
            if not res.ok:
                raise common.Refusal(f'P6 pages input {K} failed: {res.reason}')
            kept, dropped = PI.fork_rows(c.U, K, D.FORK_PAGES_LABEL, c.label)
        else:
            os.makedirs(os.path.join(c.U, 'source_catalogs'), exist_ok=True)
            kept, dropped = PI.singles_rows(c.U, K, c.label)
            PI.write_rows(dst, kept, atomic=D.ATOMIC_WRITE['singles'])
        u.meta.update(rows=len(kept), dropped_without_date_or_lst=dropped, output_fp=common.output_fps([dst]))
        if dropped:
            u.note(f'{dropped} rows without a date or an LST dropped (as p3s3_pass.sh:165; B-30 open)')
        common.exclusion_bookkeeping(u, [(r['lst'], r['band'], r['date']) for r in kept])
        if not kept:
            raise common.NoInput(f'set {K}: no served source with a date and an LST (header-only pages input written)')
        u.add_output(dst, 'pages_input', tier=TIER)
        done[name] = 'done'


def stage_build(c, run, done, K):
    C, D = common.CFG().pages, common.CFG().pages_deep
    name = unit_name(c, 'build', K)
    ok, why = build_resume(c, run, K)
    if ok:
        common.mark_skipped(run, name, why)
        done[name] = 'resumed'
        return
    pin = os.path.join(c.U, 'source_catalogs', f'pages_input_{K}.csv')
    with run.unit(name, tier=TIER, stage='build', set=K, layout=c.layout, pass_t0=c.t0) as u:
        if getattr(c, 'resume_withheld', ''):
            u.note(c.resume_withheld)
        prov = common.prerequisite(run, done, unit_name(c, 'input', K), [pin])
        if prov == 'no-input' or common.n_rows(pin) < 1:
            raise common.NoInput(f'set {K}: no pages-input row (p3s3_pass.sh:178-179)')
        u.meta['input_provenance'] = prov
        u.meta['test_ctx'] = {'roots': c.roots, 'tier': TIER, 'layout': c.layout, 'set': K}
        u.meta['build_env'] = build_env_key(c, K)          # resume identity beside the recorded files (build_resume)
        miss = [p for p in loc_tables(c) if not (os.path.exists(p) and os.path.getsize(p) > 0)]
        if miss:
            raise common.Refusal(f'P4: localization tables missing {miss} - catalog --stages localize writes them before '
                                 f'any page (p3s3_pass.sh:132-134)')
        if not c.parent:
            raise common.Refusal('P3_PARENT_DIR unknown: pass --deep-parent-dir (the parent snapshot of the catalog stage; '
                                 'unset, gate_inputs.py:19 reads the retired Run_P3S4 parents)')
        if not os.path.isdir(c.parent):
            raise common.Refusal(f'--deep-parent-dir {c.parent}: not a directory')
        mirror.gate(u, c.roots, TIER, 'build')
        script = common.build_i14_path(c.roots)
        plan_full = panels.Plan(c.U, K, C.PAGE_SORT_KEY, c.layout)
        sel = only_ids(c, plan_full)
        plan = plan_full if sel is None else panels.Plan(c.U, K, C.PAGE_SORT_KEY, c.layout, only=sel)
        if sel is not None and not plan.panels:
            raise common.NoInput(f'set {K}: none of the selected pages is a page of this set')
        fpc = common.FPCache()
        fpc.add(u, [pin] + loc_tables(c) + set_inputs(c, K)
                + sorted(glob.glob(os.path.join(c.U, f'source_catalogs_{K}', '*_source_members.csv')))
                + [p for p in (os.path.join(c.U, 'v_deep_images.csv'), os.path.join(c.U, 'v_deep_ledger.csv'),
                               os.path.join(c.U, f'source_catalogs_{K}', 'E_params.json'), c.roots_map) if os.path.exists(p)]
                + panels.cell_inputs(c.U, plan.cells()) + panels.master_paths(c.roots_map, plan.cells())
                + sample_paths())
        common.exclusion_bookkeeping(u, plan.panel_keys())
        for png, cls in ((p, 'review_page') for p in plan.panels):
            u.plan(png, cls)
        for p in plan.index.values():
            u.plan(p, 'page_index')
        is_fork = K in C.FORK_SETS
        queue = os.path.join(c.U, 'judgements', (C.QUEUE_NAMES['fork'] if is_fork else C.QUEUE_NAMES['singles']).format(K=K))
        if sel is None:
            u.plan(queue, 'review_queue')
        if c.epochs:
            ep = plan.epoch_panels([f'{l}_{d}' for l, d in c.epochs])
            u.meta['epoch_panels'] = len(ep)
            u.meta['epoch_new_pages'] = len({v[0] for p, v in ep.items() if not os.path.isdir(os.path.dirname(p))})
        if c.full:
            n = 0
            for png, (pid, lst, date, band, _cid) in plan.panels.items():
                for sc in glob.glob(os.path.join(os.path.dirname(png), f'rank*_{lst}_{date}_{band}.page.json')):
                    os.remove(sc)
                    n += 1
            u.note(f'full render: {n} sidecars removed so every page of the set renders')
        rundir = c.U
        if c.layout == 'per-set':
            from exopipe.phase3.lib.pages import m14
            rundir = m14.overlay(c.U, K)
        only_file = None
        if sel is not None:
            fd, only_file = tempfile.mkstemp(prefix=f'exopipe3_pages_only_{K}_', suffix='.txt')
            with os.fdopen(fd, 'w') as f:
                f.write('\n'.join(sorted(sel)) + '\n')
        env = build_env(c, K, only_file)
        rel = os.path.relpath(pin, c.U)
        logs = [os.path.join(c.U, f'pages_{K}_{sh}.log') for sh in range(c.shards)]
        for lg in logs:
            if os.path.exists(lg):
                os.remove(lg)
        try:
            with ThreadPoolExecutor(max_workers=c.shards) as ex:
                futs = [ex.submit(common.lrun, u, script, [rel, rundir, sh, c.shards], 'default', env, rundir, logs[sh])
                        for sh in range(c.shards)]
                results = [f.result() for f in futs]
        finally:
            if only_file:
                os.remove(only_file)
        with open(os.path.join(c.U, f'pages_{K}.log'), 'w') as f:       # p3s3_pass.sh:188
            for lg in logs:
                if os.path.exists(lg):
                    f.write(open(lg, errors='replace').read())
        bad = [(sh, r) for sh, r in enumerate(results) if not r.ok]
        if bad:
            raise common.Refusal(f'P7: {len(bad)} page shard(s) of {K} failed (D10), e.g. shard {bad[0][0]}: {bad[0][1].reason}')
        counts = [_dones(lg) for lg in logs]
        u.meta.update(shards=c.shards, rendered=sum(a for a, _ in counts), kept_unchanged=sum(b for _, b in counts),
                      pages=len(plan.pages), panels=len(plan.panels), partial=sel is not None, full=c.full)
        if sel is None and not is_fork:
            qb = os.path.join(c.U, 'judgements', C.QUEUE_NAMES['singles_build'])
            if os.path.exists(qb):
                os.replace(qb, queue)                                      # p3s3_pass.sh:190
        if c.layout == 'per-set' and sel is None:
            from exopipe.phase3.lib.pages import m14
            m14.rewrite_index(plan.index.values(), K)
        code = common.page_code(c.roots)
        shadowed = 0
        for png in plan.panels:
            if os.path.exists(png):
                n0 = len(u.tests)
                u.add_output(png, 'review_page', pass_t0=c.t0, tier=TIER, set=K, layout=c.layout, code=code, roots=c.roots)
                shadowed += any(t.get('value') == 'shadowed' for t in u.tests[n0:])
        u.meta['panels_shadowed_m14'] = shadowed
        for b, p in plan.index.items():
            if os.path.exists(p):
                u.add_output(p, 'page_index', expected_ids=[r['_pid'] for r in plan.rows if r['band'] == b],
                             subdir=plan.subdir, roots=c.roots)
        if sel is None and os.path.exists(queue):
            u.add_output(queue, 'review_queue', expected_ids=plan.queue_ids)
        figs(c, u, plan)
        u.meta['output_fp'] = common.output_fps(list(plan.index.values()) + ([queue] if sel is None else []))
        done[name] = 'done'


def figs(c, u, plan):
    """Astrometry figures of the set's identities with more than one detection (drawn by build_i14 into
    localization_figs/<identity>.png; build_i14:951-977). Missing ones are counted (FIGURE_FAILURE 'record') or refuse."""
    D = common.CFG().pages_deep
    nd = loc_ndet(c)
    want = sorted(i for i in plan.identities() if nd.get(i, 1) >= D.FIGURE_MIN_DETECTIONS)
    missing = []
    for ident in want:
        p = os.path.join(c.U, 'localization_figs', f'{ident}.png')
        if os.path.exists(p):
            u.add_output(p, 'astrometry_figure')
        else:
            missing.append(ident)
    u.meta['figures'] = dict(wanted=len(want), missing=len(missing), missing_ids=missing[:20])
    if missing:
        if D.FIGURE_FAILURE == 'refuse':
            raise common.Refusal(f'{len(missing)} of {len(want)} astrometry figures missing, e.g. {missing[:3]}')
        u.note(f'{len(missing)} of {len(want)} astrometry figures missing (pages_deep.FIGURE_FAILURE record = production: '
               f'swallowed as build_i14:966-970)')


def stage_stale(c, run, done):
    """Panels of earlier passes in the page folders this pass wrote, moved aside - never deleted - by ops/p3s3_stale_panels.py
    in place (production p3s3_pass.sh, after every set's pages and before the merge: exopipe_ops 8826836; the deep chat
    2026-10-14 on the refactor chat's finding that classify_transients shows EVERY panel in a page folder, so a detection
    that left a source - a since-excluded night - kept its panel on the served page).
    Pass start = the EARLIEST recorded start (meta pass_t0) of the done build units of the sets that have pages: every
    current panel was rendered or re-timed by its set's build at or after that start, while a build kept by --resume (or
    run by an earlier invocation) predates this one - production's P3_PAGES_RESUME_SINCE. A set with pages but no done build
    record refuses the stage: its panels share the page folders (shared layout) and would be moved as stale."""
    from exopipe.phase3.core import record
    C = common.CFG().pages
    name = unit_name(c, 'stale')
    with run.unit(name, tier=TIER, stage='stale') as u:
        t0s, missing = {}, []
        for K in C.PAGE_SETS[TIER]:
            if not would_render(c, K):
                continue                                   # no pages input: the set writes no page folder
            rec = record.load_record(run.out_root, 'pages', unit_name(c, 'build', K)) or {}
            t0 = (rec.get('meta') or {}).get('pass_t0')
            if rec.get('status') != 'done' or not t0:
                missing.append(f'{K} ({rec.get("status") or "no record"})')
            else:
                t0s[K] = int(t0)
        if missing:
            raise common.Refusal(f'stale panels: set(s) {", ".join(missing)} have pages but no done build - their panels share '
                                 f'the page folders and would be moved aside as stale; build every set first')
        if not t0s:
            raise common.NoInput('stale panels: no page set has a done build (no page folder to clean)')
        T0 = min(t0s.values())
        u.meta.update(pass_t0=T0, build_pass_t0=t0s)
        script = common.legacy_script(c.roots, 'ops', 'p3s3_stale_panels.py')
        res = common.lrun(u, script, [c.U, str(T0)], python='default', cwd=c.U, log=os.path.join(c.U, 'pages_stale.log'))
        lines = [l for l in common_tail_lines(os.path.join(c.U, 'pages_stale.log'), 40)
                 if l.startswith(('stale panels:', 'STALE-PANELS-OK', 'STALE-PANELS CONTENT TEST FAIL'))]
        for l in lines[-2:]:
            u.note(l)
        if not res.ok:
            raise common.Refusal(f'stale panels FAILED - pass aborted (D10): {(lines[-1] if lines else res.reason)[:300]}')
        q = os.path.join(c.U, f"stale_panels_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(T0))}")
        if not os.path.exists(os.path.join(q, 'MANIFEST.sha256')):
            raise common.NoInput(f'stale panels: none older than {T0} in the page folders the builds wrote - nothing moved')
        u.meta['quarantine'] = q
        u.add_output(os.path.join(q, 'MANIFEST.sha256'), 'stale_panels_manifest', run=c.U)
        done[name] = 'done'


def stage_merge(c, run, done):
    from exopipe.phase3.lib.pages import merge
    C, D = common.CFG().pages, common.CFG().pages_deep
    name = unit_name(c, 'merge')
    ok, why = common.resume_skip(run, name)
    if ok:
        common.mark_skipped(run, name, why)
        done[name] = 'resumed'
        return
    with run.unit(name, tier=TIER, stage='merge', mode=C.DEEP_MERGE) as u:
        prov = {}
        for K in D.SINGLES_SETS:            # every singles set with a catalogue must have a current build (its index)
            if PI.survivors_files(c.U, K):
                prov[K] = common.prerequisite(run, done, unit_name(c, 'build', K))
        u.meta['build_provenance'] = prov
        merged = merge.run(u, c.U, TIER, c.roots, log=os.path.join(c.U, 'pages_merge.log'))
        merge.add_outputs(u, merged)
        u.meta['merge_line'] = merged['tail']
        done[name] = 'done'


def stage_strat(c, run, done, stokes_list, per, seed):
    from exopipe.phase3.config import CFG
    name = unit_name(c, 'strat')
    with run.unit(name, tier=TIER, stage='strat') as u:
        prod = os.path.join(CFG.paths.TRANSIENTS_ROOT, f'Run_{c.label}')
        if os.path.realpath(c.U) != os.path.realpath(prod):
            raise common.Refusal(f'strat: ops/deep_stratified_queue.py hard-codes U=/data07/gh/phase3/transients/Run_<label> '
                                 f'(:24) - it can only run on the production run directory {prod}, never on {c.U} '
                                 f'(port it into lib/pages at absorption to run elsewhere)')
        if not per:
            raise common.Refusal('strat: --deep-strat-per-stratum N is required (deep_stratified_queue.py:22 has no default)')
        mirror.gate(u, c.roots, TIER, 'strat')
        script = common.legacy_script(c.roots, 'ops', 'deep_stratified_queue.py')
        env = {'P3S_LABEL': c.label}
        if c.parent:
            env['P3_PARENT_DIR'] = c.parent
        for S in stokes_list:
            res = common.lrun(u, script, [S, per, seed], env=env, cwd=c.U, log=os.path.join(c.U, f'pages_strat_{S}.log'))
            if not res.ok:
                raise common.Refusal(f'strat {S} failed: {res.reason}')
            u.add_output(os.path.join(c.U, 'judgements', f'queue_strat_{S}.txt'), 'strat_queue')
        done[name] = 'done'


# ---------------------------------------------------------------------------------------------------------------------
# run and dry run
# ---------------------------------------------------------------------------------------------------------------------

def run(args, run_):
    c = context(args, run_)
    if args.dry_run:
        return dry_run(c, args, run_)
    C, D = common.CFG().pages, common.CFG().pages_deep
    if c.epochs and C.EPOCH_IMPLIES_RESUME:
        run_.resume = True
    done = {}
    os.makedirs(c.U, exist_ok=True)
    with common.Lease(run_.out_root, c.U, C.LEASE_WAIT_S):
        for st in c.stages:
            if st == 'stagein':
                stage_stagein(c, run_, done)
            elif st == 'tess':
                stage_tess(c, run_, done)
            elif st == 'input':
                for K in c.sets:
                    stage_input(c, run_, done, K)
            elif st == 'build':
                shared_resume_guard(c, run_)           # one pass start for every set rendered (shared layout)
                for K in c.sets:
                    stage_build(c, run_, done, K)
            elif st == 'stale':
                stage_stale(c, run_, done)
            elif st == 'merge':
                stage_merge(c, run_, done)
            elif st == 'strat':
                st_list = [s for s in ('V', 'I') if not args.stokes or s in args.stokes.split(',')]
                stage_strat(c, run_, done, st_list, args.deep_strat_per_stratum or D.STRAT_PER_STRATUM,
                            args.deep_strat_seed or D.STRAT_SEED)
            if any(u.status in ('refused', 'error') for u in run_.units):
                break                                   # D10: a failed stage stops the pass (p3s3_pass.sh)
    return None


def _rec_status(run_, name):
    return _rec(run_, name).get('status', 'none')


def _rec(run_, name):
    from exopipe.phase3.core import record
    return record.load_record(run_.out_root, 'pages', name) or {}


def dry_run(c, args, run_):
    C, D = common.CFG().pages, common.CFG().pages_deep
    lines, refuse = [], []
    say = lines.append
    # with --from-run the stagein stage builds the run directory from the source run first: plan from the source (what the
    # stages would read after the copy), never from the absent / stale target (a false P6 / P4 'WOULD REFUSE')
    R = os.path.abspath(c.from_run) if (c.from_run and 'stagein' in c.stages) else c.U
    say(f'exopipe3 pages --tier deep --run {c.label}: DRY RUN (nothing written)')
    say(f'  run directory  {c.U}' + ('' if os.path.isdir(c.U) else '  (absent)')
        + (f'; inputs planned from the --from-run source {R}' if R != c.U else ''))
    say(f'  legacy roots   ' + ', '.join(f'{k}={v}' for k, v in sorted(c.roots.items()))
        + (f'  (overridden: {sorted(common.overridden(c.roots))})' if common.overridden(c.roots) else ''))
    say(f'  stages {",".join(c.stages)}; sets {",".join(c.sets)}; {c.shards} shards per set; layout {c.layout}; merge '
        f'{C.DEEP_MERGE}; fingerprint skip {"OFF (full render)" if c.full else "on (incremental)"}; P3S3_PASS_T0 {c.t0}')
    if c.epochs:
        say(f'  per-hour epochs {", ".join(f"{l}_{d}" for l, d in c.epochs)} (units unchanged are kept: '
            f'pages.EPOCH_IMPLIES_RESUME {C.EPOCH_IMPLIES_RESUME})')
    for st in c.stages:
        say(f'  stage {st}')
        if st == 'stagein':
            say(f'    from {c.from_run}: copy {len(C.FROM_RUN_COPY)} entries, link {len(C.FROM_RUN_LINK)} trees into {c.U}')
        elif st == 'tess':
            py = common.interpreter_or_none('tess')
            say(f"    ops/tess/tess_coverage.py --run {c.U}  interpreter PYTHON['tess'] = {py or 'NOT SET on this host'}")
            say(f'    -> {c.U}/tess_coverage.csv [tess_coverage]' + ('' if py else '  (would end NO-INPUT: coverage not computed)'))
            lines.extend(mirror.plan_lines(c.roots, TIER, 'tess'))
        elif st == 'input':
            mc = os.path.join(R, 'master_catalog.csv')
            if not os.path.exists(mc):
                refuse.append(f'input: P6 no {mc}')
            for K in c.sets:
                if K in C.FORK_SETS and not os.path.isdir(os.path.join(R, f'source_catalogs_{K}')):
                    say(f'    {K}: no source_catalogs_{K}/ -> NO-INPUT')
                    continue
                kept, dropped = ((PI.fork_rows(R, K, D.FORK_PAGES_LABEL, c.label) if K in C.FORK_SETS
                                  else PI.singles_rows(R, K, c.label)) if os.path.exists(mc) else ([], 0))
                how = 'ops/p3s3_tier_pages_input.py (in place)' if K in C.FORK_SETS else 'port of p3s3_pass.sh:137-169'
                say(f'    {K}: {len(PI.survivors_files(R, K))} survivors files -> {kept and len(kept) or 0} rows '
                    f'({dropped} without date/LST dropped) -> source_catalogs/pages_input_{K}.csv [pages_input]; {how}')
            lines.extend(mirror.plan_lines(c.roots, TIER, 'input'))
        elif st == 'build':
            miss = [p for p in loc_tables(c, R) if not os.path.exists(p)]
            say(f'    gate P4 localization tables (catalog --stages localize): ' + ('present' if not miss else f'MISSING {miss}'))
            if miss:
                refuse.append(f'build: P4 missing {miss}')
            say(f'    P3_PARENT_DIR {c.parent or "NOT GIVEN (--deep-parent-dir)"}' +
                ('' if (c.parent and os.path.isdir(c.parent)) else '  -> WOULD REFUSE'))
            if not (c.parent and os.path.isdir(c.parent)):
                refuse.append('build: no --deep-parent-dir directory')
            say(f'    roots map {c.roots_map}; env {build_env(c, "V_wb")}')
            lines.extend(mirror.plan_lines(c.roots, TIER, 'build'))
            for K in c.sets:
                pin = os.path.join(R, 'source_catalogs', f'pages_input_{K}.csv')
                if not os.path.exists(pin):
                    say(f'    {K}: no pages_input_{K}.csv yet (input stage writes it)')
                    continue
                pf = panels.Plan(R, K, C.PAGE_SORT_KEY, c.layout)
                sel = only_ids(c, pf)
                pl = pf if sel is None else panels.Plan(R, K, C.PAGE_SORT_KEY, c.layout, only=sel)
                have = sum(1 for p in pl.panels if os.path.exists(p))
                rec = _rec_status(run_, unit_name(c, 'input', K))
                extra = ''
                if c.epochs:
                    ep = pl.epoch_panels([f'{l}_{d}' for l, d in c.epochs])
                    extra = f'; {len(ep)} panels in the epochs ({sum(1 for p in ep if not os.path.exists(p))} not on disk)'
                say(f'    {K}: {len(pl.rows)} rows, {len(pl.pages)} pages, {len(pl.panels)} planned panels '
                    f'({have} on disk) under candidates/{pl.subdir}/; {len(pl.index)} index files; queue '
                    f'{"(none: partial)" if sel is not None else (C.QUEUE_NAMES["fork"] if K in C.FORK_SETS else C.QUEUE_NAMES["singles"]).format(K=K)}'
                    f'; P2 input record: {rec}{extra}')
            say(f'    command per shard: build_i14_pages_std7.py source_catalogs/pages_input_<K>.csv {c.U} <shard> {c.shards}')
        elif st == 'stale':
            t0s = {K: (_rec(run_, unit_name(c, 'build', K)).get('meta') or {}).get('pass_t0') for K in C.PAGE_SETS[TIER]
                   if would_render(c, K)}
            say(f'    ops/p3s3_stale_panels.py {c.U} <earliest build start> (in place): build starts {t0s} - a set with pages '
                f'and no done build refuses (this invocation\'s builds record their start when they run)')
        elif st == 'merge':
            idx = glob.glob(os.path.join(R, 'source_catalogs', '*_pages_index*.csv'))
            sv = glob.glob(os.path.join(R, 'source_catalogs', '*_sources_survivors.csv'))
            say(f'    {len(sv)} union survivors files, {len(idx)} page indexes; mode {C.DEEP_MERGE}; env '
                f'{C.REVIEW_DIR_MERGE["deep"]} -> review_dir column [survivors_review_dir]')
            say(f'    P2 build records: ' + ', '.join(f'{K} {_rec_status(run_, unit_name(c, "build", K))}' for K in D.SINGLES_SETS))
        elif st == 'strat':
            prod = os.path.join(common.CFG().paths.TRANSIENTS_ROOT, f'Run_{c.label}')
            say('    ops/deep_stratified_queue.py runs only on the production run directory (hard-coded :24)'
                + ('' if os.path.realpath(c.U) == os.path.realpath(prod) else f' -> WOULD REFUSE ({c.U} is not {prod})'))
            if os.path.realpath(c.U) != os.path.realpath(prod):
                refuse.append(f'strat: the in-place script writes only {prod}')
    refuse += [l.strip() for l in lines if 'gate P3 mirror: WOULD REFUSE' in l]
    print('\n'.join(lines))
    for r in refuse:
        print(f'  WOULD REFUSE: {r}')
    return 2 if refuse else 0
