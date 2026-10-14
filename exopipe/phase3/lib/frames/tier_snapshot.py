"""exopipe3 frames --tier snapshot - per-frame snapshot products of one LST hour of one night (a night cell).

Item: a night cell <LST>_<date> (all subbands) or <LST>_<date>_<subband> (--cells; --epochs takes <LST>_<date>); --scopes
narrows the subbands. Stages (always in the order of frames_snapshot.ORDER; default frames_snapshot.DEFAULT_STAGES =
stage,regrid). The snapshot chain runs `frames --stages stage,regrid` -> `flow --tier snapshot` (GPU, per-frame flow to the
multi-night master) -> `frames --stages cube,movies` -> search ..., all on the node that holds the staging (per hour,
DESIGN 12.1; the same item kind in bulk).
  stage   unit <cell>_stage: Phase-1 10-s frames copied to node-local disk and decompressed (port of
          snapshot_fleet_node_v9.sh:92-122 / snapshot_cube_pass.sh:39-64): discovery per subband (frames_snapshot.DISCOVERY:
          'legacy' = clv15 phase3_snapshot_search.discover_frames, today's; 'integration' = core.inputs.phase1_frames, B-15),
          copy without overwrite (COPY_PROCS at a time), funpack -S to <o>.tmp then rename (FUNPACK_PROCS at a time, every
          exit status checked). Writes <stage root>/<cell>/<sb>/*-image.fits + .filelist [snapshot_stage]; the record lists
          per subband the staged I/V counts, the sha1 of the staged name set and the frames whose header says WSCNVIS = 0
          (flow compares sets, not counts). Gates: N-G1 free disk < MIN_FREE_GB[chain]; N-G2 staged set != discovery set
          per subband and Stokes, or .fs/.fz/.tmp left, or 0 frames; N-G3 fewer than MIN_FRAMES[chain] frames (tested here,
          before the GPU flow - production tests it after the flow).
  regrid  unit <cell>_regrid: N-G4 a broken Master link refuses the cell (v9:146-151). Chain 'search' (fleet, v9:124-138):
          a subband whose first frame has NAXIS1 != ON_GRID_NAXIS (1508) is regridded IN THE STAGING onto its OWN Master
          (clv15/phase3_snapshot_regrid.py in place, py36) or left on its grid when it has no Master (REGRID-SKIPPED,
          noted). Chain 'cube' (cube pass:66-81): frames off their own Master's grid refuse the cell. Outputs: the regridded
          subband directories [snapshot_regrid]; nothing to regrid = no-input.
  wideband  not offered in the window (frames.md B9: production builds the 10-s Red/Green/Blue co-adds in memory inside the
          group search and persists only wideband_members_<S>.csv; persisting them is new behaviour for the snapshot chat).
  cube    unit <cell>_cube (a COMPLETE cell of this unit's own done record is skipped, as cube pass :35; any other
          earlier cubes refuse without --fresh): one flowed-frame cube per (subband, Stokes):
          clv15/phase3_snapshot_cube.py write (flow interpreter, astropy >= 5.3) then verify (py36), CUBE_WRITE_PROCS at a
          time, into node-local
          <scratch>/cube_pass/<cell>/cubes; shipped by copy + md5 to <cubes root>/<cell>/ with MANIFEST.txt (one line per cube:
          md5, copy-OK, its VERIFY line; last line COMPLETE <cell> cubes=N subbands=M node=... code=...), cube_write.log,
          cube_verify.log [snapshot_cube]. Gates: N-G4 broken link, N-G6 a staged subband without Master, no flow fields,
          N-G7 cubes != 2 x staged subbands or a verify not OK (nothing shipped), N-G8 shipped md5 != local md5, N-G9 the
          writer's astropy (the engine refuses). The flow fields are read from --snap-flow-dir, else the flow step's
          node-local <scratch>/<cell>/flow, else its shipped copy <paths.SNAPSHOT_FLOW_ROOT>/<cell>; copying the fields to
          the flow tree is the flow step's `ship` (not done here).
  movies  unit <cell>_movies: with frames_snapshot.FLOW_MOVIES first (v9:277), per staged subband the flow movie + series
          (clv15/phase3_snapshot_flow_movie.py, FLOW_MOVIE_PROCS at a time, header = the first flowed Stokes I frame, whose
          |CDELT2| is the arcmin scale - M31; a subband without one gets no movie, item refused) shipped as production does
          (cp -n) to <paths.SNAPSHOT_FLEET_OUT>/<cell>/flow/ [flow_movie] - an earlier shipped file is kept and the item
          refused by name unless --fresh; then the 19 frame movies (clv15/phase3_snapshot_frame_movie.py in place, flow
          interpreter) from the staged flowed frames (--snap-movie-source staged) or the shipped cubes (cube) into
          <movies root>/<cell>/ (frames_<cell>_<band>_I.mp4, movies.log; MOVIES_DONE touched as today for its current
          readers) [frame_movie]. N-G10: a frame-movie failure refuses this unit only.
Engine arguments: a config value equal to the in-place engine's own default is not passed (a default run has the
production command line); an overridden key is passed; a config value that differs from the engine default without an
override refuses as drift (engine_flags).

Roots: staging and cube work are node-local under the scratch root (--work, else the site SCRATCH_ROOT of this host;
--snap-stage-root names an existing staging root to read). Products go to the production roots (paths.SNAPSHOT_CUBES_ROOT,
frames_snapshot.MOVIES_ROOT, paths.SNAPSHOT_FLEET_OUT; refused while paths.PRODUCTION_WRITES is False) or, with --out D, to
D/snapshot_cubes, D/snapshot_movies, D/snapshot_fleet_out (same relative layout). Records: <out>/_records/frames/.
The cube and movies stages need the done stage record of the cell written on THIS node (or an explicit --snap-stage-root).

Exclusions: as today, every discovered subband is staged and cubed whatever data_decisions says (snapshot_cube_pass.sh:46-58
has no exclusion test; question B13 for Gregg); excluded (LST, subband, night) are reported in the record, not refused.

Failure modes (exit 2 = refused, the gate named; 3 = nothing to do): N-G1..N-G10 above; no Phase-1 run for the cell;
discovery refused (core.inputs content test); staging of an earlier run present (pass --fresh); no done stage record on
this node; a site PYTHON key or funpack missing; a production root while paths.PRODUCTION_WRITES is False; an override of a
PROTECTED key; the wideband stage requested.
"""
import glob
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time

from exopipe.phase3.lib.frames.tier_sublst import (cfg, plan, say, split_list, requested_stages, refuse_production_writes,
                                                   check_overrides, interpreter_key, legacy_script, check_interpreter,
                                                   engine, refusal_line, try_resume, parallel, exclusions, report,
                                                   unit_refused, production_root_of)

TIER = 'snapshot'
SUMMARY = 'snapshot: staging, pilot regrid, flowed-frame cubes, frame and flow movies (one night cell)'
_FRAME_RE = re.compile(r'^(?P<sb>\d+MHz)-(?P<kind>Clean|Pilot)-Snapshot-(?P<ts>\d{8}_\d{6})(?:-(?P<st>[IV]))?-image\.fits$')
_INDEX_RE = re.compile(r'-t\d{4}-image\.fits(\.f[sz])?$')
_ROOT_KEYS = {'snapshot_cubes': 'SNAPSHOT_CUBES_ROOT', 'snapshot_fleet_out': 'SNAPSHOT_FLEET_OUT'}


def frame_stokes(name):
    """Stokes of a staged frame name (both naming eras: Clean = I, Pilot = V; pilot era -I-/-V-), or None."""
    m = _FRAME_RE.match(name)
    if not m:
        return None
    return m.group('st') or ('I' if m.group('kind') == 'Clean' else 'V')


def decompressed(name):
    for sfx in ('.fs', '.fz'):
        if name.endswith(sfx):
            return name[:-len(sfx)]
    return name


def raw_cards(path):
    from exopipe.phase3.core.record import fits_primary_cards
    try:
        return fits_primary_cards(path)
    except OSError:
        return {}


def md5_file(path, bs=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(bs), b''):
            h.update(b)
    return h.hexdigest()


def engine_defaults(script):
    """{'--flag': default} of the in-place engine's argparse calls with a literal default (a module-level constant name is
    resolved), read from its source with ast - nothing imported. The engine's own defaults are the production values."""
    import ast
    with open(script) as f:
        tree = ast.parse(f.read())
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    out = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, 'attr', None) == 'add_argument' and node.args
                and isinstance(node.args[0], ast.Constant) and str(node.args[0].value).startswith('--')):
            continue
        for kw in node.keywords:
            if kw.arg == 'default':
                if isinstance(kw.value, ast.Name) and kw.value.id in consts:
                    out[node.args[0].value] = consts[kw.value.id]
                else:
                    try:
                        out[node.args[0].value] = ast.literal_eval(kw.value)
                    except ValueError:
                        pass
    return out


def engine_flags(script, spec, section='frames_snapshot'):
    """The engine arguments for [(flag, KEY, config value)]: none when the value equals the engine's own default (a
    default run has exactly the production command line), the flag when the key was overridden (--set / --config).
    A config value that differs from the engine default WITHOUT an override is drift (the engine changed in production,
    or the config is wrong): refused - the config stays the one place the value is read from."""
    from exopipe.phase3.core.record import Refusal
    dflt = engine_defaults(script)
    over = {o[0].split('[', 1)[0] for o in cfg().overrides}
    args = []
    for flag, key, val in spec:
        if flag not in dflt:
            raise Refusal(f'{os.path.basename(script)} has no literal default for {flag} (engine changed?): '
                          f'{section}.{key} cannot be checked against it')
        d = dflt[flag]
        if isinstance(val, float) and isinstance(d, (int, float)):
            same = abs(float(d) - val) <= 1e-12 * max(1.0, abs(val))
        else:
            same = val == d
        if same:
            continue
        if f'{section}.{key}' not in over:
            raise Refusal(f'config drift: {os.path.basename(script)} {flag} defaults to {d!r}, {section}.{key} = {val!r} '
                          f'without an override - update config/steps/frames.py to the engine (production) value, or pass '
                          f'--set {section}.{key}=... to choose it')
        args += [flag, str(val)]
    return args


def add_arguments(p):
    S = cfg().frames_snapshot
    g = p.add_argument_group('tier snapshot (lib/frames/tier_snapshot.py; config frames_snapshot)',
                             f'--stages of --tier snapshot, in order: {",".join(S.ORDER)}; default '
                             f'{",".join(S.DEFAULT_STAGES)} (then `flow --tier snapshot`, then --stages cube,movies); '
                             f'items: --cells <LST>_<date>[_<subband>] or --epochs <LST>_<date>')
    g.add_argument('--snap-chain', choices=('search', 'cube'),
                   help="whose rules apply (default frames_snapshot.CHAIN): 'search' = the fleet (disk 800 GB, >= 1000 frames, "
                        "pilot frames regridded), 'cube' = the cube pass (disk 700 GB, frames off their Master's grid refuse)")
    g.add_argument('--snap-discovery', choices=('legacy', 'integration'),
                   help='Phase-1 frame discovery (default frames_snapshot.DISCOVERY = legacy, today; integration = '
                        'core.inputs.phase1_frames, one frame per integration - batch row B-15)')
    g.add_argument('--snap-stage-root', metavar='DIR',
                   help='staging root holding <cell>/<sb>/ (default <scratch>/frames_snapshot.STAGE_SUBDIR); an existing root '
                        'given here is read by cube/movies without a stage record')
    g.add_argument('--snap-flow-dir', metavar='DIR', help='cube/movies: the cell\'s flow fields flow_<sb>_<ts>.npy (default: '
                                                          'node-local <scratch>/<cell>/flow, else paths.SNAPSHOT_FLOW_ROOT/<cell>)')
    g.add_argument('--snap-movie-source', choices=('staged', 'cube'),
                   help='movies: from the staged flowed frames or the shipped cubes (default frames_snapshot.MOVIE_SOURCE)')
    g.add_argument('--snap-no-flow-movies', action='store_true', help='movies: frame movies only (frames_snapshot.FLOW_MOVIES off)')
    g.add_argument('--snap-clean-stage', action='store_true',
                   help='remove the node-local staging of each cell at the end (default frames_snapshot.CLEAN_AFTER)')


def default_out(args):
    return cfg().paths.SNAPSHOT_CUBES_ROOT


class _Ctx:
    def __init__(self, args, run):
        self.args, self.run, self.C = args, run, cfg()
        self.S = self.C.frames_snapshot
        self.dry = bool(getattr(args, 'dry_run', False))
        self.chain = getattr(args, 'snap_chain', None) or self.S.CHAIN
        if self.chain not in ('search', 'cube'):
            raise ValueError(f'frames_snapshot.CHAIN must be search or cube, not {self.chain!r}')
        self.discovery = getattr(args, 'snap_discovery', None) or self.S.DISCOVERY
        if self.discovery not in ('legacy', 'integration'):
            raise ValueError(f'frames_snapshot.DISCOVERY must be legacy or integration, not {self.discovery!r}')
        self.movie_source = getattr(args, 'snap_movie_source', None) or self.S.MOVIE_SOURCE
        self.flow_movies = bool(self.S.FLOW_MOVIES) and not getattr(args, 'snap_no_flow_movies', False)
        self.scopes = split_list(getattr(args, 'scopes', None))
        self.workers = getattr(args, 'workers', None)
        self.out = os.path.abspath(args.out) if getattr(args, 'out', None) else None
        self.would_refuse = 0
        self.planned = 0
        self.stages = ()
        self.explicit_stage_root = getattr(args, 'snap_stage_root', None)
        self._scratch, self.scratch_err = None, None
        try:
            from exopipe.phase3.core.fitsio import scratch_root
            self._scratch = os.path.abspath(scratch_root(getattr(args, 'work', None)))
        except Exception as e:
            self.scratch_err = str(e)

    # ---- roots ---------------------------------------------------------------------------------------------------
    def scratch(self):
        from exopipe.phase3.core.record import Refusal
        if self._scratch is None:
            raise Refusal(self.scratch_err or 'no node-local scratch root')
        return self._scratch

    def stage_root(self):
        if self.explicit_stage_root:
            return os.path.abspath(self.explicit_stage_root)
        return os.path.join(self.scratch(), self.S.STAGE_SUBDIR)

    def stage_dir(self, cell):
        return os.path.join(self.stage_root(), cell)

    def product_root(self, key):
        prod = getattr(self.C.paths, _ROOT_KEYS[key]) if key in _ROOT_KEYS else self.S.MOVIES_ROOT
        return os.path.join(self.out, key) if self.out else prod

    def production_cubes(self):
        return self.C.paths.SNAPSHOT_CUBES_ROOT

    def master(self, lst, sb):
        return self.S.MASTER_FILE.format(links=self.C.masters_snapshot.LINK_ROOT, lst=lst, sb=sb)

    def gate(self, name, ok, detail):
        if self.dry:
            plan(f'    gate {name}: {"pass" if ok else "WOULD REFUSE"} - {detail}')
            if not ok:
                self.would_refuse += 1
        return ok


def parse_items(args):
    """[(cell, lst, date, [subbands])] from --cells (night cells, optional subband) and --epochs."""
    from exopipe.phase3.core import cells
    S = cfg().frames_snapshot
    scopes = split_list(getattr(args, 'scopes', None))
    toks = []
    if getattr(args, 'epochs', None):
        toks += cells.expand_epochs(args.epochs)
    if getattr(args, 'cells', None):
        toks += cells.expand(args.cells, kind='night cell')
    if getattr(args, 'lsts', None):
        raise ValueError('tier snapshot works on night cells: give --cells <LST>_<date>[_<subband>] or --epochs <LST>_<date>')
    by = {}
    for t in toks:
        lst, date, sb = cells.parse_night_cell(t)
        if sb and sb not in S.SUBBANDS:
            raise ValueError(f'{t}: {sb} is not a snapshot subband (frames_snapshot.SUBBANDS)')
        cell = f'{lst}_{date}'
        e = by.setdefault(cell, [lst, date, set()])
        e[2].update([sb] if sb else [s for s in S.SUBBANDS if not scopes or s in scopes])
    return [(c, v[0], v[1], [s for s in S.SUBBANDS if s in v[2]]) for c, v in sorted(by.items())]


def run(args, run):
    import exopipe.phase3.lib.frames.tests_snapshot  # noqa: F401  (registers this tier's content tests)
    try:
        ctx = _Ctx(args, run)
        items = parse_items(args)
        if not items:
            raise ValueError('tier snapshot needs --cells <LST>_<date>[_<subband>] or --epochs <LST>_<date>')
        stages = requested_stages(args, ctx.S.ORDER, ctx.S.DEFAULT_STAGES)
        ctx.stages = stages
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        return 1
    check_overrides(('frames_snapshot',))
    exclusions()                                       # C-09
    plan(f'frames --tier snapshot (chain {ctx.chain}): stages {",".join(stages)}; cells {",".join(c for c, *_ in items)}; '
         f'staging {ctx.stage_root() if ctx._scratch or ctx.explicit_stage_root else "(no scratch root: " + str(ctx.scratch_err) + ")"}; '
         f'products under {ctx.out or "the PRODUCTION roots"}' + ('; DRY RUN (nothing written)' if ctx.dry else ''))
    if not ctx.dry:
        refuse_production_writes([ctx.product_root(k) for k in ('snapshot_cubes', 'snapshot_movies', 'snapshot_fleet_out')])
    for cell, lst, date, sbs in items:
        for st in stages:
            if st == 'stage':
                stage_stage(ctx, cell, lst, date, sbs)
            elif st == 'regrid':
                stage_regrid(ctx, cell, lst, date, sbs)
            elif st == 'wideband':
                unit_refused(run, f'{cell}_wideband', 'not offered in the window (frames.md B9): production builds the 10-s '
                             'Red/Green/Blue co-adds in memory inside the group search and persists only '
                             'wideband_members_<S>.csv; persisting them is new behaviour (snapshot chat, B-16)',
                             ctx.dry, ctx, tier=TIER)
            elif st == 'cube':
                stage_cube(ctx, cell, lst, date, sbs)
            elif st == 'movies':
                stage_movies(ctx, cell, lst, date, sbs)
        if (getattr(args, 'snap_clean_stage', False) or ctx.S.CLEAN_AFTER) and not ctx.explicit_stage_root:
            clean_stage(ctx, cell)
    if ctx.dry:
        plan(f'plan: {ctx.planned} unit(s); {ctx.would_refuse} gate(s) would refuse')
        return 2 if ctx.would_refuse else (0 if ctx.planned else 3)
    return None


def clean_stage(ctx, cell):
    try:
        d = ctx.stage_dir(cell)
    except Exception:
        return
    if ctx.dry:
        plan(f'  clean: would remove the staging {d}')
        return
    if os.path.isdir(d) and production_root_of(d) is None:
        shutil.rmtree(d, ignore_errors=True)
        say(f'frames: staging {d} removed (--snap-clean-stage / frames_snapshot.CLEAN_AFTER)')


# ---------------------------------------------------------------------------------------------------------------------
# stage: stage
# ---------------------------------------------------------------------------------------------------------------------

def discover(ctx, lst, date, sb):
    """The source frame paths of (lst, date, sb) in the order the fleet's heredoc prints them (I then V), and a report."""
    from exopipe.phase3.core.record import Refusal
    if ctx.discovery == 'integration':
        from exopipe.phase3.core import inputs
        rep = {}
        fr = inputs.phase1_frames(lst, date, sb, report=rep)
        return [p for st in ('I', 'V') for _, p in fr[st]], rep
    import phase3_snapshot_search as SS            # clv15, in place (site LEGACY_CODE_ROOTS on sys.path)
    if SS.FLOW_MODE:
        raise Refusal('P3_SNAP_FLOW=1 in this environment: discover_frames would substitute flowed frames; the fleet\'s '
                      'staging heredoc runs without it (v9:106-111) - unset it')
    try:
        fr = SS.discover_frames(lst, date, sb)
    except RuntimeError as e:
        raise Refusal(f'discovery: {e}')
    return [p for st in ('I', 'V') for _, p in fr[st]], {}


def free_gb(path):
    d = path
    while d and not os.path.exists(d):
        d = os.path.dirname(d)
    st = os.statvfs(d or '/')
    return st.f_bavail * st.f_frsize / 1e9


def funpack_bin():
    from exopipe.phase3.core.record import Refusal
    C = cfg()
    fp = C.frames_snapshot.FUNPACK or getattr(C.site, 'FUNPACK', None)
    if not fp:
        py36 = (C.site.PYTHON or {}).get('py36')
        fp = os.path.join(os.path.dirname(py36), 'funpack') if py36 else None
    if not fp or not os.access(fp, os.X_OK):
        raise Refusal(f'funpack not found ({fp}): set frames_snapshot.FUNPACK or site FUNPACK')
    return fp


def staged_names(d):
    return sorted(f for f in os.listdir(d) if f.endswith('-image.fits')) if os.path.isdir(d) else []


def stage_stage(ctx, cell, lst, date, sbs):
    from exopipe.phase3.core.record import Refusal
    S = ctx.S
    name = f'{cell}_stage'
    if ctx.dry:
        plan(f'  STAGE stage {cell}: {len(sbs)} subband(s); discovery {ctx.discovery}; chain {ctx.chain}')
        try:
            d = ctx.stage_dir(cell)
            ctx.gate('node-local scratch', True, d)
            ctx.gate('N-G1 disk', free_gb(d) >= S.MIN_FREE_GB[ctx.chain],
                     f'{free_gb(d):.0f} GB free, need {S.MIN_FREE_GB[ctx.chain]}')
            ctx.gate('earlier staging', not staged_names_any(d) or getattr(ctx.args, 'fresh', False),
                     f'{d} ' + ('holds frames (pass --fresh)' if staged_names_any(d) else 'empty or absent'))
        except Refusal as e:
            ctx.gate('node-local scratch', False, str(e))
        total = 0
        try:
            for sb in sbs:
                paths, rep = discover(ctx, lst, date, sb)
                nI = sum(1 for p in paths if frame_stokes(decompressed(os.path.basename(p))) == 'I')
                total += len(paths)
                plan(f'    {sb}: {len(paths)} frames (I {nI}, V {len(paths) - nI})'
                     + (f'; runs {rep.get("runs")} mixed={rep.get("mixed")}' if rep else ''))
        except Refusal as e:
            ctx.gate('discovery', False, str(e))
        ctx.gate('N-G3 minimum frames', total >= S.MIN_FRAMES[ctx.chain], f'{total} discovered, need {S.MIN_FRAMES[ctx.chain]}')
        try:
            funpack_bin()
            ctx.gate('funpack', True, funpack_bin())
        except Refusal as e:
            ctx.gate('funpack', False, str(e))
        plan(f'    UNIT {name}')
        ctx.planned += 1
        return
    d = ctx.stage_dir(cell)
    if staged_names_any(d) and not getattr(ctx.args, 'fresh', False):
        try:
            stage_input(ctx, cell)                     # the done stage record of this unit, its file set intact
            from exopipe.phase3.core import record
            held = set(((record.load_record(ctx.run.out_root, ctx.run.step, name) or {}).get('meta') or {})
                       .get('subbands') or {})
            if not set(sbs) <= held:                   # and it holds every subband this run asks for
                raise Refusal(f'the staging holds {sorted(held)}, this run asks {sbs}')
            u = ctx.run.unit(name)
            u.status, u.reason = 'skipped-valid', (f'the staging {d} of this unit\'s done record is intact (same file set); '
                                                   f'pass --fresh to stage again')
            say(f'frames: {name} skipped ({u.reason})')
            return
        except Refusal:
            pass                                       # not this unit's intact staging: the unit refuses below
    with ctx.run.unit(name, tier=TIER, stage='stage', cell=cell, chain=ctx.chain, discovery=ctx.discovery) as u:
        refuse_production_writes([d])
        ex = exclusions()
        u.set_admitted([f'{date}/{sb}' for sb in sbs if not ex.is_excluded(lst, sb, date)])
        excl = [sb for sb in sbs if ex.is_excluded(lst, sb, date)]
        if excl:
            u.note(f'excluded in data_decisions but staged as today (cube pass has no exclusion test; B13): {excl}')
        fg = free_gb(d)
        if fg < S.MIN_FREE_GB[ctx.chain]:
            u.refuse(f'N-G1 only {fg:.0f} GB free under {d} (need {S.MIN_FREE_GB[ctx.chain]}, chain {ctx.chain}; '
                     f'INCIDENT-FLEET-V4-SHIPPED-A-PARTIAL-CELL-ON-A-FULL-DISK)')
        if staged_names_any(d):
            if not getattr(ctx.args, 'fresh', False):
                u.refuse(f'staging of an earlier run present under {d} that no intact done stage record of this unit lists '
                         f'- pass --fresh to remove it and stage again')
            shutil.rmtree(d)
        fp = funpack_bin()
        lists = {}
        for sb in sbs:
            paths, rep = discover(ctx, lst, date, sb)
            if rep:
                u.meta.setdefault('discovery_report', {})[sb] = rep
            if not paths and ctx.chain == 'cube':
                continue                                   # cube pass: an empty list drops the subband (:56)
            lists[sb] = paths
            os.makedirs(os.path.join(d, sb), exist_ok=True)
            with open(os.path.join(d, sb, '.filelist'), 'w') as f:
                f.write(''.join(p + '\n' for p in paths))
        for p in [p for ps in lists.values() for p in ps]:
            u.add_input(p)
        errors = []

        def copy(job):
            src, dst = job
            if os.path.lexists(dst):
                return                                      # cp -n
            try:
                shutil.copyfile(src, dst)
            except OSError as e:
                errors.append(f'copy {src}: {e}')
        jobs = [(p, os.path.join(d, sb, os.path.basename(p))) for sb, ps in lists.items() for p in ps]
        parallel(copy, jobs, S.COPY_PROCS)

        def unpack(f):
            o = decompressed(f)
            with open(o + '.tmp', 'wb') as out:
                rc = subprocess.run([fp, '-S', f], stdout=out, stderr=subprocess.DEVNULL).returncode
            if rc == 0:
                os.replace(o + '.tmp', o)
                os.remove(f)
            else:
                errors.append(f'funpack rc={rc} {f}')
        comp = sorted(p for sb in lists for p in glob.glob(os.path.join(d, sb, '*.f[sz]')))
        parallel(unpack, comp, S.FUNPACK_PROCS)
        # N-G2: per subband and Stokes the staged set equals the discovery set; no leftovers; > 0 frames
        bad, total, summary = [], 0, {}
        for sb, ps in lists.items():
            sd = os.path.join(d, sb)
            want = sorted(decompressed(os.path.basename(p)) for p in ps)
            have = staged_names(sd)
            left = sorted(f for f in os.listdir(sd) if f.endswith(('.fs', '.fz', '.tmp')))
            for st in ('I', 'V'):
                w = {x for x in want if frame_stokes(x) == st}
                h = {x for x in have if frame_stokes(x) == st}
                if w != h:
                    bad.append(f'{sb} {st}: staged {len(h)} != listed {len(w)} (missing {sorted(w - h)[:2]}, extra '
                               f'{sorted(h - w)[:2]})')
            if left:
                bad.append(f'{sb}: {len(left)} compressed/tmp leftovers, e.g. {left[0]}')
            zero, unread = [], []
            for f in have:
                c = raw_cards(os.path.join(sd, f))
                if not c:
                    unread.append(f)
                elif c.get('WSCNVIS') is not None and float(c.get('WSCNVIS')) == 0:
                    zero.append(f)
            total += len(have)
            summary[sb] = {'I': sum(1 for x in have if frame_stokes(x) == 'I'),
                           'V': sum(1 for x in have if frame_stokes(x) == 'V'),
                           'names_sha1': hashlib.sha1('\n'.join(have).encode()).hexdigest(),
                           'wscnvis0': zero, 'unreadable_header': unread,
                           'index_named': sum(1 for p in ps if _INDEX_RE.search(p))}
        u.meta['stage_dir'] = d
        u.meta['subbands'] = summary
        u.meta['host'] = ctx.run.host
        for e in errors[:20]:
            u.note(e)
        if bad:
            u.refuse('N-G2 staging incomplete: ' + '; '.join(bad[:6]) + (f' (+{len(errors)} copy/funpack errors)'
                                                                       if errors else ''))
        if total == 0:              # cube pass :61 STAGE-REFUSED at 0 frames; the fleet refuses it as < MIN_FRAMES (v9:199)
            u.refuse(f'N-G2 no Phase-1 frame staged for {cell} in {",".join(sbs)} (discovery listed none)')
        u.check('stage.min_frames', total >= S.MIN_FRAMES[ctx.chain], value=total,
                detail=f'N-G3: {total} decompressed frames, need {S.MIN_FRAMES[ctx.chain]} (chain {ctx.chain}; v9:199-201)')
        for sb in lists:
            if summary[sb]['I'] + summary[sb]['V']:
                u.add_output(os.path.join(d, sb), 'snapshot_stage', cell=cell)
            else:
                u.note(f'{sb}: no Phase-1 frames discovered (empty .filelist kept, as the fleet keeps the empty directory)')
    report(u)


def staged_names_any(d):
    return bool(glob.glob(os.path.join(d, '*', '*-image.fits*')))


# ---------------------------------------------------------------------------------------------------------------------
# the stage record (cube, movies, regrid read only what it lists - DESIGN 11.5)
# ---------------------------------------------------------------------------------------------------------------------

def stage_input(ctx, cell, u=None):
    """(stage dir, {sb: staged names}) from the done stage record of this cell written on THIS host; an explicit
    --snap-stage-root without a record is accepted (noted). Refuses otherwise or when the directory no longer matches."""
    from exopipe.phase3.core import record
    from exopipe.phase3.core.record import Refusal
    d = ctx.stage_dir(cell)
    rec = record.load_record(ctx.run.out_root, ctx.run.step, f'{cell}_stage')
    if rec is None and ctx.explicit_stage_root:
        sbs = {sb: staged_names(os.path.join(d, sb)) for sb in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, sb))} \
            if os.path.isdir(d) else {}
        if u is not None:
            u.note(f'staging {d} given by --snap-stage-root without a stage record: {sum(map(len, sbs.values()))} frames')
        return d, {k: v for k, v in sbs.items() if v}
    if rec is None or rec.get('status') != 'done':
        raise Refusal(f'no done stage record for {cell} under {ctx.run.out_root} (status '
                      f'{None if rec is None else rec.get("status")}) - run --stages stage on this node first')
    meta = rec.get('meta') or {}
    if meta.get('host') not in (None, ctx.run.host):
        raise Refusal(f'the staging of {cell} is on {meta.get("host")}, not this node ({ctx.run.host}): node-local data')
    if os.path.realpath(meta.get('stage_dir', d)) != os.path.realpath(d):
        raise Refusal(f'the stage record names {meta.get("stage_dir")}, this run reads {d}')
    out = {}
    for sb, info in (meta.get('subbands') or {}).items():
        names = staged_names(os.path.join(d, sb))
        if hashlib.sha1('\n'.join(names).encode()).hexdigest() != info.get('names_sha1'):
            raise Refusal(f'staged {sb} files of {cell} differ from the stage record (set changed since staging)')
        if names:
            out[sb] = names
    if u is not None:                                   # the chain of records (DESIGN 11.5): this unit read that record
        u.add_input(record.record_path(ctx.run.out_root, ctx.run.step, f'{cell}_stage'), extra={'role': 'stage record'})
    return d, out


def dry_stage_input(ctx, cell, required):
    """Dry-run gate: the stage record this stage will read (written by this invocation's own stage stage, or earlier)."""
    from exopipe.phase3.core.record import Refusal
    if 'stage' in ctx.stages:
        plan(f'    stage record: written by this invocation\'s stage stage')
        return
    try:
        d, staged = stage_input(ctx, cell)
        plan(f'    staging {d}: {len(staged)} subband(s) (stage record)')
    except Refusal as e:
        if required:
            ctx.gate('stage record', False, str(e))
        else:
            plan(f'    no staging on this node: flow movies would be skipped ({e})')


# ---------------------------------------------------------------------------------------------------------------------
# stage: regrid
# ---------------------------------------------------------------------------------------------------------------------

def broken_links(ctx, lst, sbs):
    return [sb for sb in sbs if os.path.islink(ctx.master(lst, sb)) and not os.path.exists(ctx.master(lst, sb))]


def stage_regrid(ctx, cell, lst, date, sbs):
    from exopipe.phase3.core.record import Refusal, NoInput
    S = ctx.S
    name = f'{cell}_regrid'
    mode = S.OFF_GRID[ctx.chain]
    if ctx.dry:
        plan(f'  STAGE regrid {cell}: chain {ctx.chain} ({mode}); masters under {ctx.C.masters_snapshot.LINK_ROOT}')
        br = broken_links(ctx, lst, sbs)
        ctx.gate('N-G4 Master links', not br, f'broken: {br}' if br else 'no broken link')
        miss = [sb for sb in sbs if not os.path.exists(ctx.master(lst, sb))]
        if miss:
            plan(f'    no own Master for {miss} (regrid skipped there; flow / cube refuse them, N-G6)')
        ok, py = check_interpreter(interpreter_key('frames_snapshot', 'regrid'))
        ctx.gate('interpreter', ok, py)
        plan(f'    UNIT {name}: subbands whose first staged frame has NAXIS1 != {S.ON_GRID_NAXIS} -> phase3_snapshot_regrid.py '
             f'<stage>/<sb> <own Master> --nproc {S.REGRID_NPROC}')
        ctx.planned += 1
        return
    with ctx.run.unit(name, tier=TIER, stage='regrid', cell=cell, chain=ctx.chain) as u:
        d, staged = stage_input(ctx, cell, u)
        refuse_production_writes([d])
        u.set_admitted([f'{date}/{sb}' for sb in staged])
        # the fleet tests every subband of its run ($SUBBANDS, v9:147), the cube pass the staged ones ($SBS, :68-76)
        br = broken_links(ctx, lst, sbs if mode == 'regrid' else [sb for sb in staged if sb in sbs])
        if br:
            u.refuse(f'N-G4 broken Master link(s) {br} under {ctx.C.masters_snapshot.LINK_ROOT} (a fault, never "no Master"; '
                     f'v9:146-151)')
        script = legacy_script('clv15', 'phase3_snapshot_regrid.py')
        py = interpreter_key('frames_snapshot', 'regrid')
        done = 0
        for sb in sorted(staged, key=lambda s: int(s[:-3])):
            if sb not in sbs:
                continue
            first = os.path.join(d, sb, staged[sb][0])
            nax = raw_cards(first).get('NAXIS1')
            m = ctx.master(lst, sb)
            if mode == 'refuse':
                if os.path.exists(m):                  # an unreadable header on either side refuses (cube pass :79-81)
                    mnax = raw_cards(m).get('NAXIS1')
                    if nax is None or mnax is None or nax != mnax:
                        u.refuse(f'N-G5 {sb}: frames {nax} px, own Master {mnax} px - off their Master\'s grid (cube pass '
                                 f':78-81; OFF_GRID[cube] = refuse, B-22)')
                continue
            if nax is None or nax == S.ON_GRID_NAXIS:          # the fleet counts an unreadable header as on grid (v9:129-130)
                continue
            if not os.path.exists(m):
                u.note(f'REGRID-SKIPPED {cell} {sb} no Master of its own - frames left on their grid (v9:133)')
                continue
            u.add_input(m)
            args_ = [os.path.join(d, sb), m, '--nproc', str(S.REGRID_NPROC)]       # v9:137 passes --nproc 8
            res, lines = engine(script, args_, py, os.path.join(d, f'regrid_{sb}.log'), unit=u)
            fin = [l for l in lines if l.startswith('REGRID-DONE ')]
            if not res.ok or not fin:
                why = refusal_line(lines, res)
                if res.refused:
                    u.refuse(f'regrid {sb}: {why}')
                raise RuntimeError(f'regrid {sb} failed: {why}')
            u.note(fin[-1][:200])
            u.add_output(os.path.join(d, sb), 'snapshot_regrid', master=m)
            done += 1
        if not done:
            raise NoInput(f'no staged subband of {cell} needs a regrid (chain {ctx.chain})')
    report(u)


# ---------------------------------------------------------------------------------------------------------------------
# stage: cube
# ---------------------------------------------------------------------------------------------------------------------

def flow_dir(ctx, cell):
    from exopipe.phase3.core.record import Refusal
    if getattr(ctx.args, 'snap_flow_dir', None):
        return os.path.abspath(ctx.args.snap_flow_dir), '--snap-flow-dir'
    if ctx._scratch:
        d = ctx.S.FLOW_DIR.format(scratch=ctx._scratch, cell=cell)
        if os.path.isdir(d):
            return d, 'node-local flow work dir'
    d = os.path.join(ctx.C.paths.SNAPSHOT_FLOW_ROOT, cell)
    if os.path.isdir(d):
        return d, 'shipped flow fields (paths.SNAPSHOT_FLOW_ROOT)'
    raise Refusal(f'no flow fields for {cell}: no --snap-flow-dir, no node-local flow dir, no {d}')


def cubes_complete(dest):
    """The cell's shipped MANIFEST.txt ends with its COMPLETE line (cube pass :35)."""
    p = os.path.join(dest, 'MANIFEST.txt')
    try:
        with open(p) as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
    except OSError:
        return False
    return bool(lines) and lines[-1].startswith('COMPLETE ')


def _date_u():
    t = time.gmtime()
    return time.strftime('%a %b ', t) + f'{t.tm_mday:2d}' + time.strftime(' %H:%M:%S UTC %Y', t)


def stage_cube(ctx, cell, lst, date, sbs):
    from exopipe.phase3.core.record import Refusal
    S = ctx.S
    name = f'{cell}_cube'
    dest = os.path.join(ctx.product_root('snapshot_cubes'), cell)
    if ctx.dry:
        plan(f'  STAGE cube {cell}: write (flow) + verify (py36) {S.CUBE_WRITE_PROCS} at a time; ship to {dest}')
        dry_stage_input(ctx, cell, True)
        br = broken_links(ctx, lst, sbs)
        ctx.gate('N-G4 Master links', not br, f'broken: {br}' if br else 'no broken link')
        miss = [sb for sb in sbs if not os.path.exists(ctx.master(lst, sb))]
        ctx.gate('N-G6 own Masters', not miss, f'no Master: {miss}' if miss else 'every subband has its Master')
        try:
            fd, how = flow_dir(ctx, cell)
            ctx.gate('flow fields', True, f'{fd} ({how})')
        except Refusal as e:
            ctx.gate('flow fields', False, str(e))
        for k in ('cube_write', 'cube_verify'):
            ok, py = check_interpreter(interpreter_key('frames_snapshot', k))
            ctx.gate(f'interpreter {k}', ok, py)
        try:
            q = engine_flags(legacy_script('clv15', 'phase3_snapshot_cube.py'), [('--quant', 'CUBE_QUANT', S.CUBE_QUANT)])
            ctx.gate('engine defaults', True, f'cube write extra arguments: {q or "none (production command line)"}')
        except Refusal as e:
            ctx.gate('engine defaults', False, str(e))
        ex = os.path.exists(os.path.join(dest, 'MANIFEST.txt'))
        ctx.gate('earlier cubes', not ex or getattr(ctx.args, 'fresh', False),
                 f'{dest}/MANIFEST.txt ' + ('exists (pass --fresh; a COMPLETE cell of this unit\'s own done record is '
                                            'skipped)' if ex else 'absent'))
        plan(f'    UNIT {name}: {2 * len(sbs)} cubes planned')
        ctx.planned += 1
        return
    if not getattr(ctx.args, 'fresh', False) and cubes_complete(dest):
        # cube pass :35: a cell whose MANIFEST ends COMPLETE is not cubed again (a re-write changes the bytes: RICE dither).
        # This unit's own done record + a MANIFEST that passes its test now = skipped-valid; any other COMPLETE cell refuses
        # below (cubes of an earlier run, never overwritten without --fresh).
        from exopipe.phase3.core import record, validate
        rec = record.load_record(ctx.run.out_root, ctx.run.step, name)
        man = os.path.join(dest, 'MANIFEST.txt')
        if rec and rec.get('status') == 'done' and man in {o['path'] for o in rec.get('outputs', [])} \
                and all(r['passed'] for r in validate.run('snapshot_cube', man)):
            u = ctx.run.unit(name)
            u.status, u.reason = 'skipped-valid', (f'{man} is COMPLETE, written by this unit\'s done record and passes its '
                                                   f'content test (cube pass :35 skips a COMPLETE cell); --fresh rebuilds')
            say(f'frames: {name} skipped ({u.reason})')
            return
    with ctx.run.unit(name, tier=TIER, stage='cube', cell=cell) as u:
        d, staged = stage_input(ctx, cell, u)
        staged = {sb: v for sb, v in staged.items() if sb in sbs}
        u.set_admitted([f'{date}/{sb}' for sb in staged])
        if not staged:
            u.refuse(f'no staged subband of {cell} under {d}')
        br = broken_links(ctx, lst, list(staged))
        if br:
            u.refuse(f'N-G4 broken Master link(s) {br}')
        miss = [sb for sb in staged if not os.path.exists(ctx.master(lst, sb))]
        if miss:
            u.refuse(f'N-G6 no Taper R0 Master for staged subband(s) {miss} (the hourly-screen fallback is retired, '
                     f'Gregg 2026-10-04 "go on all three"; cube pass :83-92)')
        fd, how = flow_dir(ctx, cell)
        u.note(f'flow fields from {fd} ({how})')
        u.meta['flow_dir'] = fd
        work = os.path.join(ctx.scratch(), S.CUBE_WORK_SUBDIR, cell)
        cub = os.path.join(work, 'cubes')
        refuse_production_writes([work, dest])
        earlier = os.path.isdir(dest) and (os.path.exists(os.path.join(dest, 'MANIFEST.txt'))
                                          or any(f.endswith('.fits') for f in os.listdir(dest)))
        if earlier and not getattr(ctx.args, 'fresh', False):
            u.refuse(f'cubes of an earlier run present in {dest} - pass --fresh to replace them')
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(cub)
        jobs = [(sb, st) for sb in sorted(staged, key=lambda s: int(s[:-3])) for st in ('I', 'V')]
        for sb, st in jobs:
            u.plan(os.path.join(dest, f'{cell}_{sb}_{st}_flowcube.fits'), 'snapshot_cube')
        wpy, vpy = interpreter_key('frames_snapshot', 'cube_write'), interpreter_key('frames_snapshot', 'cube_verify')
        script = legacy_script('clv15', 'phase3_snapshot_cube.py')
        qargs = engine_flags(script, [('--quant', 'CUBE_QUANT', S.CUBE_QUANT)])
        results = {}

        def one(job):
            sb, st = job
            cube = os.path.join(cub, f'{cell}_{sb}_{st}_flowcube.fits')
            wargs = ['write', '--stage', os.path.join(d, sb), '--cell', cell, '--subband', sb, '--stokes', st,
                     '--flows', fd, '--out', cub] + qargs
            wres, wl = engine(script, wargs, wpy, os.path.join(work, f'cube_write_{sb}_{st}.log'), unit=u)
            if not wres.ok:
                results[job] = ('refused' if wres.refused else 'error', refusal_line(wl, wres), None)
                return
            vres, vl = engine(script, ['verify', cube, '--stage', os.path.join(d, sb)], vpy,
                              os.path.join(work, f'cube_verify_{sb}_{st}.log'), unit=u, refusal_codes=())
            vline = next((l.strip() for l in vl if l.startswith('VERIFY ')), '')
            results[job] = ('ok' if (vres.rc == 0 and vline.endswith('-> OK')) else 'verify', vline or vres.reason, vline)
        parallel(one, jobs, ctx.workers or S.CUBE_WRITE_PROCS)
        for log in ('cube_write', 'cube_verify'):           # production's two logs, in (subband, Stokes) order
            with open(os.path.join(work, f'{log}.log'), 'w') as f:
                for sb, st in jobs:
                    p = os.path.join(work, f'{log}_{sb}_{st}.log')
                    if os.path.exists(p):
                        f.writelines(l for l in open(p, errors='replace') if not l.startswith('===== '))
        bad = [f'{sb} {st}: {results.get((sb, st), ("missing", "no result"))[0]} {results.get((sb, st), ("", ""))[1][:200]}'
               for sb, st in jobs if results.get((sb, st), ('missing',))[0] != 'ok']
        ncube = len(glob.glob(os.path.join(cub, '*_flowcube.fits')))
        if bad or ncube != S.CUBES_PER_SUBBAND * len(staged):
            if any(results.get(j, ('',))[0] == 'error' for j in jobs) and not any(results.get(j, ('',))[0] == 'refused'
                                                                                  for j in jobs):
                raise RuntimeError(f'cube write/verify failed for {cell}: ' + '; '.join(bad[:4]) + f' (logs in {work})')
            u.refuse(f'N-G7 cubes={ncube} of {S.CUBES_PER_SUBBAND * len(staged)}, not OK: ' + '; '.join(bad[:4])
                     + f' - NOT shipped (logs in {work})')
        # ship: copy (tmp + rename), md5 of the shipped copy against the local one, MANIFEST lines as cube pass :118-132
        os.makedirs(dest, exist_ok=True)
        lines, nbad, md5s = [], 0, {}
        for sb, st in jobs:
            b = f'{cell}_{sb}_{st}_flowcube.fits'
            src, dst = os.path.join(cub, b), os.path.join(dest, b)
            shutil.copyfile(src, dst + '.part')
            os.replace(dst + '.part', dst)
            m1, m2 = md5_file(src), md5_file(dst)
            md5s[dst] = m2
            nbad += m1 != m2
            lines.append(f'{b} md5={m2} {"copy-OK" if m1 == m2 else "COPY-MISMATCH"} | {results[(sb, st)][2]}')
        if nbad:
            u.refuse(f'N-G8 SHIP-FAILED {cell}: {nbad} cube(s) differ on the destination')
        size = subprocess.run(['du', '-sh', dest], capture_output=True, text=True).stdout.split('\t')[0].strip()
        code = subprocess.run(['git', '-C', os.path.dirname(script), 'rev-parse', '--short', 'HEAD'], capture_output=True,
                              text=True).stdout.strip() or 'unknown'
        lines.append(f'COMPLETE {cell} cubes={ncube} subbands={len(staged)} node={ctx.run.host} {_date_u()} code={code} '
                     f'size={size}')
        man = os.path.join(work, 'MANIFEST.txt')
        with open(man, 'w') as f:
            f.write(''.join(l + '\n' for l in lines))
        for fn in ('MANIFEST.txt', 'cube_write.log', 'cube_verify.log'):
            shutil.copyfile(os.path.join(work, fn), os.path.join(dest, fn))
        for dst, m in md5s.items():
            u.add_output(dst, 'snapshot_cube', md5=m)
        u.add_output(os.path.join(dest, 'MANIFEST.txt'), 'snapshot_cube', staged_subbands=len(staged))
        shutil.rmtree(work, ignore_errors=True)
    report(u)


# ---------------------------------------------------------------------------------------------------------------------
# stage: movies
# ---------------------------------------------------------------------------------------------------------------------

def first_flowed_I(sd):
    c = sorted(glob.glob(os.path.join(sd, '*Clean-Snapshot-*-image_flowed.fits')) +
               glob.glob(os.path.join(sd, '*Pilot-Snapshot-*-I-image_flowed.fits')))
    return c[0] if c else None


def strip_legacy_header(log):
    """core.legacy.run opens a child's log with a blank line and a '===== <UTC> <argv>' line; a log that is a product
    (movies.log, movie_<sb>.log) keeps the engine's own lines only, as production writes it ('> log')."""
    try:
        with open(log, errors='replace') as f:
            lines = f.read().splitlines(True)
    except OSError:
        return
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and lines[i].startswith('===== '):
        with open(log + '.tmp', 'w') as f:
            f.writelines(lines[i + 1:])
        os.replace(log + '.tmp', log)


def movie_skips(cell, lines):
    """B-30 (Gregg 2026-10-14: the minimum kept, every skip recorded, never refused): a band the frame-movie engine made no
    movie of - fewer than frames_snapshot.MIN_FRAMES_PER_BAND staged flowed frames, its line 'too few staged flowed frames for
    <subband>' (phase3_snapshot_frame_movie.py:56-57) - is COUNTED (fallbacks.FRAME_MOVIE_SKIP_FALLBACK). Returns them."""
    few = [l.rsplit(' ', 1)[-1] for l in lines if l.startswith('too few staged flowed frames for ')]
    if few:
        from exopipe.phase3.core import fallback
        fallback.site('fallbacks.FRAME_MOVIE_SKIP_FALLBACK', f'{cell}: no frame movie of {few} - fewer than '
                      f'{cfg().frames_snapshot.MIN_FRAMES_PER_BAND} staged flowed frames', n=len(few))
    return few


def frame_movie_args(ctx, script):
    S = ctx.S
    fargs = []
    if tuple(S.FRAME_BANDS) != tuple(S.SUBBANDS) + ('Red', 'Green', 'Blue', 'Full'):
        fargs += ['--bands', ','.join(S.FRAME_BANDS)]
    return fargs + engine_flags(script, [('--k', 'STRETCH_K', S.STRETCH_K), ('--size', 'MOVIE_WIDTH_PX', S.MOVIE_WIDTH_PX),
                                         ('--fps', 'MOVIE_FPS', S.MOVIE_FPS), ('--crf', 'MOVIE_CRF', S.MOVIE_CRF)])


def flow_movies(ctx, u, cell, d, staged, ship):
    """Per staged subband the flow movie + series (v9:277: phase3_snapshot_flow_movie.py, FLOW_MOVIE_PROCS at a time, the
    header frame = the first flowed Stokes I frame, whose |CDELT2| sets the arcmin scale - M31, clv15 98595b3), rendered
    in node-local scratch and shipped as production does (cp -n, v9:346). A subband without a flowed I frame gets no movie
    (production's call fails without --header-fits; the engine would otherwise assume the 18 MHz pixel). A shipped file of
    an earlier run is not overwritten (cp -n) and the item is refused by name unless --fresh replaces it."""
    S = ctx.S
    fd, how = flow_dir(ctx, cell)
    mwork = os.path.join(ctx.scratch(), 'snapshot_movies_work', cell)
    shutil.rmtree(mwork, ignore_errors=True)
    os.makedirs(mwork)
    fscript = legacy_script('clv15', 'phase3_snapshot_flow_movie.py')
    dargs = engine_flags(fscript, [('--duration-s', 'FLOW_MOVIE_DURATION_S', S.FLOW_MOVIE_DURATION_S)])
    fres, hdrs = {}, {}
    order = sorted(staged, key=lambda s: int(s[:-3]))
    for sb in order:
        hdrs[sb] = first_flowed_I(os.path.join(d, sb))

    def fm(sb):
        out = os.path.join(mwork, f'flow_{sb}.mp4')
        a = ['--flows', fd, '--subband', sb, '--header-fits', hdrs[sb], '--out', out] + dargs
        fres[sb] = engine(fscript, a, interpreter_key('frames_snapshot', 'flow_movies'),
                          os.path.join(mwork, f'movie_{sb}.log'), unit=u, refusal_codes=())[0]
    parallel(fm, [sb for sb in order if hdrs[sb]], S.FLOW_MOVIE_PROCS)
    os.makedirs(ship, exist_ok=True)
    fresh = bool(getattr(ctx.args, 'fresh', False))
    for sb in order:
        target = os.path.join(ship, f'flow_{sb}.mp4')
        mp4 = os.path.join(mwork, f'flow_{sb}.mp4')
        if not hdrs[sb]:
            u.refuse_item(target, f'flow movie {sb}: no flowed Stokes I frame in {os.path.join(d, sb)} for --header-fits '
                                  f'(its |CDELT2| is the pixel scale, M31)', gate='N-G10 flow movie')
            continue
        if sb not in fres or fres[sb].rc != 0 or not os.path.exists(mp4):
            u.refuse_item(target, f'flow movie {sb}: {fres[sb].reason if sb in fres else "not run"} '
                                  f'(log {mwork}/movie_{sb}.log)', gate='N-G10 flow movie')
            continue
        strip_legacy_header(os.path.join(mwork, f'movie_{sb}.log'))
        kept = []
        for fn in (f'flow_{sb}.mp4', f'flow_{sb}_series.png', f'flow_{sb}_series.csv', f'movie_{sb}.log'):
            s_, t_ = os.path.join(mwork, fn), os.path.join(ship, fn)
            if not os.path.exists(s_):
                continue
            if os.path.exists(t_) and not fresh:
                kept.append(fn)                                    # cp -n keeps the earlier file (v9:346)
                continue
            shutil.copyfile(s_, t_ + '.part')
            os.replace(t_ + '.part', t_)
        if kept:
            u.refuse_item(target, f'flow movie {sb}: earlier file(s) {kept} in {ship} not overwritten (cp -n, v9:346) - '
                                  f'move them aside or pass --fresh', gate='N-G10 flow movie ship')
            continue
        u.add_output(target, 'flow_movie', flow_dir=fd, header_fits=hdrs[sb])
    shutil.rmtree(mwork, ignore_errors=True)


def stage_movies(ctx, cell, lst, date, sbs):
    from exopipe.phase3.core.record import Refusal
    S = ctx.S
    name = f'{cell}_movies'
    mv = os.path.join(ctx.product_root('snapshot_movies'), cell)
    ship = os.path.join(ctx.product_root('snapshot_fleet_out'), cell, 'flow')
    if ctx.dry:
        plan(f'  STAGE movies {cell}: frame movies from the {ctx.movie_source} frames -> {mv}'
             + (f'; flow movies -> {ship}' if ctx.flow_movies else ''))
        if ctx.movie_source == 'staged' or ctx.flow_movies:
            dry_stage_input(ctx, cell, ctx.movie_source == 'staged')
        if ctx.movie_source == 'cube':
            ok = os.path.exists(os.path.join(ctx.product_root('snapshot_cubes'), cell, 'MANIFEST.txt')) or \
                os.path.exists(os.path.join(ctx.production_cubes(), cell, 'MANIFEST.txt'))
            ctx.gate('shipped cubes', ok, f'{cell}/MANIFEST.txt')
        if ctx.flow_movies:
            try:
                fd, how = flow_dir(ctx, cell)
                ctx.gate('flow fields', True, f'{fd} ({how})')
            except Refusal as e:
                ctx.gate('flow fields', False, str(e))
        for k in ('frame_movies', 'flow_movies'):
            ok, py = check_interpreter(interpreter_key('frames_snapshot', k))
            ctx.gate(f'interpreter {k}', ok, py)
        fargs = []
        try:
            fargs = frame_movie_args(ctx, legacy_script('clv15', 'phase3_snapshot_frame_movie.py'))
            if ctx.flow_movies:
                engine_flags(legacy_script('clv15', 'phase3_snapshot_flow_movie.py'),
                             [('--duration-s', 'FLOW_MOVIE_DURATION_S', S.FLOW_MOVIE_DURATION_S)])
            ctx.gate('engine defaults', True, 'config = the engines\' defaults, or overridden')
        except Refusal as e:
            ctx.gate('engine defaults', False, str(e))
        plan(f'    UNIT {name}: ' + ('flow movies per staged subband, then ' if ctx.flow_movies else '')
             + f'phase3_snapshot_frame_movie.py {cell} {mv} '
             + (f'--frames-root <staging>' if ctx.movie_source == 'staged' else '--cube-root <cubes root>')
             + (' ' + ' '.join(fargs) if fargs else ''))
        ctx.planned += 1
        return
    with ctx.run.unit(name, tier=TIER, stage='movies', cell=cell, source=ctx.movie_source) as u:
        refuse_production_writes([mv] + ([ship] if ctx.flow_movies else []))
        d, staged = None, {}
        if ctx.movie_source == 'staged' or ctx.flow_movies:
            try:
                d, staged = stage_input(ctx, cell, u)
            except Refusal:
                if ctx.movie_source == 'staged':
                    raise
                u.note('no staging on this node: flow movies skipped (the cube pass makes none)')
        staged = {sb: v for sb, v in staged.items() if sb in sbs}
        u.set_admitted([f'{date}/{sb}' for sb in (staged or sbs)])
        script = legacy_script('clv15', 'phase3_snapshot_frame_movie.py')
        fargs = frame_movie_args(ctx, script)
        if ctx.movie_source == 'staged':
            src_args = ['--frames-root', d]
        else:
            croot = ctx.product_root('snapshot_cubes')
            if not os.path.exists(os.path.join(croot, cell, 'MANIFEST.txt')):
                croot = ctx.production_cubes()
            if not os.path.exists(os.path.join(croot, cell, 'MANIFEST.txt')):
                u.refuse(f'no shipped cubes of {cell} (MANIFEST.txt) under {ctx.product_root("snapshot_cubes")} or {croot}')
            src_args = ['--cube-root', croot]
            u.add_input(os.path.join(croot, cell, 'MANIFEST.txt'))
        log = os.path.join(mv, 'movies.log')
        if os.path.exists(log) and not getattr(ctx.args, 'fresh', False):
            from exopipe.phase3.core import record
            rec = record.load_record(ctx.run.out_root, ctx.run.step, name)
            if not (rec and log in {o['path'] for o in rec.get('outputs', [])}):
                u.refuse(f'movies of an earlier run present in {mv} - pass --fresh to replace them')
        # the fleet makes the flow movies first (v9:277), then the frame movies (v9:285); a frame-movie failure does not
        # take the flow movies with it (N-G10 refuses this unit after both)
        if ctx.flow_movies and staged:
            flow_movies(ctx, u, cell, d, staged, ship)
        os.makedirs(mv, exist_ok=True)
        if os.path.exists(log):
            os.remove(log)                                   # production truncates it ('> movies.log')
        t0 = time.time()
        res, lines = engine(script, [cell, mv] + src_args + fargs,
                            interpreter_key('frames_snapshot', 'frame_movies'), log, unit=u, refusal_codes=())
        strip_legacy_header(log)
        movie_skips(cell, lines)
        total = [l for l in lines if l.startswith('TOTAL')]
        if res.rc != 0 or not total:
            u.refuse(f'N-G10 FRAMEMOVIES-FAILED {cell}: {refusal_line(lines, res)} (see {log})')
        open(os.path.join(mv, 'MOVIES_DONE'), 'a').close()   # legacy marker for its current readers (B-49)
        os.utime(os.path.join(mv, 'MOVIES_DONE'))
        u.note(f'FRAMEMOVIES {cell} {total[-1]}')
        expect = [l.split(':')[0].split()[-1] for l in lines if re.match(r'^\S+ \S+: \d+ frames, ', l)]
        for p in sorted(glob.glob(os.path.join(mv, f'frames_{cell}_*_I.mp4'))):
            if os.path.getmtime(p) >= t0 - 2:
                u.add_output(p, 'frame_movie', cell=cell)
        u.add_output(log, 'frame_movie', cell=cell, bands=expect)
    report(u)
