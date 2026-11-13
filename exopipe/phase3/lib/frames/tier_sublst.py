"""exopipe3 frames --tier sublst - sub-LST (10-minute sidereal window) frames.

Stages (always run in the order of frames_sublst.ORDER; defaults per operating mode in frames_sublst.STAGES):
  stack     PER HOUR (--epochs <LST>_<date>): the night's window stacks through the ONE wrapper of lib/masters/tier_sublst.py
            (crosscheck decision 6; config section masters_sublst): its 'admit' (the job-list rules of
            sublst_run_lst_nocat.sh:82-186) then its 'stack' (clv15/phase3_sublst_stack.py in place, Stokes I then V).
            Called as tier_sublst.stack_night(run, lst, night, ...): units, products and content tests are the masters
            wrapper's (admit_<LST>_<night> with its plan under <out>/_plans/, stack_<LST>_<night>_<sb>; classes
            sublst_job_plan, sublst_window*, sublst_admission, sublst_fluxdc_record), recorded under <out>/_records/frames/.
  nonmember-list  PER HOUR, on request (in no default), before stack: puts ONE night on the non-member nights list
            (<paths.OPS_ROOT>/sublst_nonmember_nights.txt, the one switch production's readers and the stack admission
            read) when a subband of the night has no usable qa3_scale in the frozen deep weights - locked, once, recorded;
            nothing is written when not needed or already listed. lib/masters/sublst_nonmember.list_night (rules
            L-1..L-6 there), unit nonmember-list_<LST>_<night>.
  nonmember PER HOUR, on request (in no default): the arm-A residuals of ONE night that is NOT in the installed window
            Masters (Gregg 2026-11-12: nights that are not in a Master are searched as non-members against the frozen
            Masters) - residual = the night's window stack minus the installed window Master <root>/<LST>_masters/...,
            no other night's stack read. Through lib/masters/sublst_nonmember.nonmember_night (unit of work, inputs,
            outputs and failure modes N-1..N-12 in that module): units nonmember_<LST>_<night>_<sb>, one per subband of
            the night's plan, outputs <root>/<LST>/<LST>_<night>/<sb>/<S>/..._w<k>_resid_loo.fits
            [sublst_nonmember_residual], recorded under <out>/_records/frames/. A night that IS in the Masters refuses
            (its residual is `masters --tier sublst --stages loo`), as does a missing installed Master or an incomplete
            stack of the night.
  wideband  Red/Green/Blue windows = inverse-variance stacks of the member subband windows, both residual sets:
            clv15/phase3_sublst_wideband.py in place (site PYTHON['default']), one process per (group, night),
            frames_sublst.WIDEBAND_PARALLEL at a time. Gates: arm A complete for every member subband (the done record of
            `masters --tier sublst --stages loo`, unit loo_<LST>_<sb>; frames_sublst.LOO_EVIDENCE='log' reads production's
            SUBLST-LOO lines instead) and the x2 refine complete (S-G8: <LST>_x2/logs/<LST>_<sb>.log newer than the
            subband's newest window stack ends 'refused=0', or exopipe_ops/sublst_refine_approved.py accepts its refusals).
            Per hour with ONE night, arm A of a member subband is judged by the record of the night's own unit
            nonmember_<LST>_<night>_<sb> (stage nonmember, under this root) where this root holds one: done passes, any
            other status refuses; a subband without such a record (the night is in the window Masters there, or the
            batch stage wrote its residuals) keeps the LST-wide evidence above (night_arm_a_complete, wideband_arm_a).
            Outputs <root>/<LST>/<LST>_<night>/<G>/<S>/sublst_<LST>_<night>_<G>_<S>_w<k>{,_pb,_nframes,_resid_loo}.fits and
            <root>/<LST>_x2/.../..._resid_loo.fits [sublst_wideband]; logs <root>/<LST>/logs/wideband_<LST>_<G>_<night>.log.
            Per hour: the epoch's night; intermittent: every night (or --nights), and a group with no window over the LST
            refuses (S-G10). A (LST, group, night) excluded under core.exclusions is not passed to the engine.
            REQUIRED MEMBERS (the engine since clv15 582c92e, 2026-10-23; Gregg 2026-10-23 "If a subband is missing the
            wideband does not get created", 18 MHz the one exception): per Stokes a window is built only when every member
            but 18 MHz has that Stokes' five products and its V stack; a refused (window, Stokes) is the engine's
            REFUSED-INCOMPLETE line, recorded as an item refusal (gate S-G9), and a window built in one Stokes only
            ('(Stokes X only)') records that Stokes' products. Every engine log is reconciled with its final count line
            (SUBLST-WIDEBAND ... refused_incomplete=; wideband_outcome).
  tree      tree B for the 10-minute search: clv15/phase3_sublst_tenmin_tree.py in place (site PYTHON['py36']), one process
            per band (15 subbands + Red/Green/Blue). Outputs <root>/tenmin_root_x2/<LST>_<band>/{frames.csv, subtracted/
            links, bin_masters/PB_bin<k>_<band>.fits} or NO_WINDOWS.txt [tenmin_tree].
  tree-nonmember  PER HOUR, on request (in no default), after the night's x2 refine (`flow --tier sublst --stages
            refine-nonmember`): the entry of ONE night of co-add weight 0 in the tree of each band that has an x2 residual
            of the night - its links, its rows in frames.csv (the builder's order, the other rows byte for byte), PBNNIGHT
            (and the header, when the night is the earliest of a window) of the window primary-beam maps, whose pixels a
            weight-0 night does not change. lib/frames/sublst_tree_nonmember.py (unit of work, inputs, outputs and failure
            modes T-1..T-8 in that module): units <LST>_tree-nonmember_<band>_<night> [tenmin_tree]. A member night
            refuses and names the stage `tree`. One writer per band's tree: the lease <tree>/.leases/<LST>_<band>, which
            the stage `tree` takes too.
  searchframes  optional: REFUSED in the window (crosscheck decision 6, group 14: `search --tier sublst` stays the producer
            of the sub-LST search frames; a frames-only call needs the frames_only seam, frames.md B1/B15).
  verify    content test (b) of exopipe_ops/sublst_reserve_test_residuals.py, ported (R1 per-night rule + LOONCOAD, R2
            completeness, R3 wideband members, plus the cards of every residual) [sublst_residual_set], and every band's
            tree [tenmin_tree].

Per hour vs intermittent (DESIGN 12): --epochs 09h_2025-01-01 (per hour; default stages: stack) or --cells with night cells;
--lsts 09h (intermittent; default stages: wideband,tree,verify; --nights narrows wideband). The per-hour chain is
`frames --stages stack` -> `flow --tier sublst` (x2 refine, GPU) -> `frames --stages wideband,tree,verify`; arm A
(window masters + leave-one-out residuals) is `masters --tier sublst --stages loo` (intermittent, LST-wide).

Roots: the sub-LST root is --out (default paths.SUBLST_ROOT, production; refused while paths.PRODUCTION_WRITES is False):
<root>/<LST>/ (stacks, arm-A residuals, wideband), <root>/<LST>_x2/ (arm B, wideband set B), <root>/<LST>_masters/,
<root>/tenmin_root_x2/, as production. A non-production --out holds only what earlier steps wrote into it (masters stacks
and loo, flow refine) - nothing is read from production implicitly. --sublst-mirror (frames-only parity runs on production
inputs) first symlinks the production inputs a stage reads but the root lacks (one link per file, never over a real file;
symlinks in a directory an engine writes into are removed first, so no engine writes through a link into production);
use it with --parity.

Failure modes (exit 2 = refused, the gate named): a site PYTHON key missing; an override of a PROTECTED key; a production
root while paths.PRODUCTION_WRITES is False; exclusion list missing (C-09); masters wrapper refusals (G-S01..G-S09);
the nonmember stage's refusals (N-1..N-12 of lib/masters/sublst_nonmember.py: excluded night, no plan or incomplete stack
of the night, a member night, a missing or unusable installed window Master, differing flux scales, earlier residuals);
arm A incomplete (no masters loo record / log line); S-G8 x2 refine incomplete or older than the stacks; wideband engine
REFUSED (S-G9: mixed or disagreeing flux scales, shapes); S-G10 no window of a group over the LST; tree engine REFUSED
(S-G11); S-G12 (production's run-starts-clean rule) products of an earlier run where a unit writes, no record of this unit
lists them and no --fresh, or - after the wideband engine - any file in the unit's write directories this run did not write;
searchframes requested; any content test failed (R1-R3 included). Exit 1 also when a wideband log disagrees with its count
line. Exit 3: every unit had nothing to do.
"""
import glob
import os
import re
import sys
import threading
import time

TIER = 'sublst'
SUMMARY = 'sub-LST: night window stacks (per hour, via masters), wideband windows, tree B, content test (b)'
_GROUPS_ALL = ('Red', 'Green', 'Blue', 'Full')
_PRINT_LOCK = threading.Lock()


# =====================================================================================================================
# Helpers shared with tier_snapshot.py
# =====================================================================================================================

def cfg():
    from exopipe.phase3.config import CFG
    return CFG


def say(msg):
    with _PRINT_LOCK:
        print(msg, file=sys.stderr, flush=True)


def plan(msg):
    with _PRINT_LOCK:
        print(msg, flush=True)


def split_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            out += split_list(x)
        return out
    return [x.strip() for x in str(v).split(',') if x.strip()]


def requested_stages(args, order, default):
    """The stages to run, in ``order``: --stages (or ``default``), cut by --from / --until (steps/frames.py adds the flags,
    dest from_stage / until_stage)."""
    order = list(order)
    req = split_list(getattr(args, 'stages', None)) or list(default)
    frm, until = getattr(args, 'from_stage', None), getattr(args, 'until_stage', None)
    unknown = [s for s in req + [x for x in (frm, until) if x] if s not in order]
    if unknown:
        raise ValueError(f'unknown stage(s) {unknown}; stages in order: {", ".join(order)}')
    lo = order.index(frm) if frm else 0
    hi = order.index(until) if until else len(order) - 1
    sel = [s for i, s in enumerate(order) if s in req and lo <= i <= hi]
    if not sel:
        raise ValueError(f'no stage selected (stages in order: {", ".join(order)})')
    return sel


def production_roots():
    C = cfg()
    extra = tuple(getattr(getattr(C, 'frames', None), 'EXTRA_PRODUCTION_ROOTS', ()) or ())
    p = C.paths
    return tuple(p.PRODUCTION_ROOTS) + extra + (p.SUBLST_ROOT, p.SNAPSHOT_CUBES_ROOT, p.SNAPSHOT_FLEET_OUT,
                                                p.SNAPSHOT_FLOW_ROOT, p.OPS_ROOT)


def production_root_of(path):
    rp = os.path.realpath(path)
    for root in production_roots():
        r = os.path.realpath(root)
        if rp == r or rp.startswith(r.rstrip('/') + '/'):
            return root
    return None


def refuse_production_writes(paths):
    """README rule 10: while paths.PRODUCTION_WRITES is False nothing is written under a production root (incl. the
    tiers' roots that paths.PRODUCTION_ROOTS does not list yet: OPS_ROOT, SNAPSHOT_FLEET_OUT; spec B8)."""
    from exopipe.phase3.core.record import Refusal
    if cfg().paths.PRODUCTION_WRITES:
        return
    for p in paths:
        root = production_root_of(p)
        if root:
            raise Refusal(f'this run would write under the production root {root} ({p}) and paths.PRODUCTION_WRITES is '
                          f'False (refactor window): pass --out (and --work) to your own roots')


def check_overrides(sections):
    """config.py refuses --set/--config on a PROTECTED_KEYS key already; checked again here so a programmatic CFG.set
    (tests, wrappers) cannot reach an engine literal either."""
    from exopipe.phase3.core.record import Refusal
    C = cfg()
    for o in C.overrides:
        sec, _, rest = o[0].partition('.')
        key = re.split(r'\[', rest, 1)[0]
        if sec in sections and key in (getattr(getattr(C, sec), 'PROTECTED_KEYS', ()) or ()):
            raise Refusal(f'{sec}.{key} is protected (an engine literal or a pinned input): no override can change it')


def interpreter_key(section, engine_name):
    return getattr(cfg(), section).INTERPRETER[engine_name]


def legacy_script(root_name, fname):
    from exopipe.phase3.core import env
    from exopipe.phase3.core.record import Refusal
    try:
        root = env.legacy_root(root_name)
    except KeyError as e:
        raise Refusal(str(e))
    p = os.path.join(root, fname)
    if not os.path.isfile(p):
        raise Refusal(f'in-place code {p} not found (refactor window, DESIGN 11.1)')
    return p


def check_interpreter(key):
    """(ok, detail) for a site PYTHON key (dry-run gate)."""
    from exopipe.phase3.core import legacy
    try:
        py = legacy.interpreter(key)
    except Exception as e:
        return False, str(e)
    return os.access(py, os.X_OK), py


def engine(script, args_, python, log, unit=None, env=None, refusal_codes=(2,), refusal_text=None):
    """One in-place engine run through core.legacy.run (the one child runner). Returns (res, lines) where lines are the
    lines THIS call appended to ``log`` (legacy.run appends, with a header line). refusal_text: what in the output marks a
    failed call as a refusal (default legacy.run's). A failed call is judged by its OWN last lines (judge_own_lines)."""
    from exopipe.phase3.core import legacy
    os.makedirs(os.path.dirname(log), exist_ok=True)
    off = os.path.getsize(log) if os.path.exists(log) else 0
    kw = {} if refusal_text is None else {'refusal_text': tuple(refusal_text)}
    res = legacy.run(script, list(args_), python=python, env=env or {}, log=log, unit=unit, refusal_codes=refusal_codes,
                     **kw)
    with open(log, errors='replace') as f:
        f.seek(off)
        lines = f.read().splitlines()
    judge_own_lines(res, lines, script, refusal_codes, refusal_text)
    return res, lines


def judge_own_lines(res, lines, script, refusal_codes, refusal_text=None):
    """legacy.run judges a call that failed outside its refusal codes (refused or error) by the last lines of the WHOLE log,
    and these logs are appended across calls: an earlier call's refusal line would make this call's crash a refusal. Re-judge
    it by THIS call's own last lines - legacy.run's rule, texts and window (its refusal_text and tail_lines defaults, read from
    its signature) - and the reason as legacy.run words it. Exit 0, an exit in refusal_codes and a B-30 fallback refusal
    (sticky) are left as legacy.run judged them. (The fix in core would be legacy.run reading its tail from after its own
    header line; then this is a no-op.)"""
    import inspect
    from exopipe.phase3.core import legacy
    if res.rc == 0 or res.rc in tuple(refusal_codes) or res.fallback_refusals:
        return
    par = inspect.signature(legacy.run).parameters
    texts = tuple(refusal_text) if refusal_text is not None else tuple(par['refusal_text'].default)
    own = lines
    for i, l in enumerate(lines[:3]):                 # this call's text starts with legacy.run's '===== <UTC> <argv>' line
        if l.startswith('===== '):
            own = lines[i + 1:]
            break
    tail = own[-int(par['tail_lines'].default):]
    hit = next((t for t in texts if any(t in l for l in tail)), None)
    if (hit is not None) == bool(res.refused):
        return
    res.refused = hit is not None
    line = next((l.strip()[:200] for l in reversed(tail) if hit and hit in l), '')
    res.reason = f'{os.path.basename(script)} exit {res.rc}' + (f' ({hit}: {line})' if hit else ' - error')


def refusal_line(lines, res):
    ref = [l.strip() for l in lines if re.match(r'^(?:\w+Error: )?REFUS(?:ED|ING)\b', l.strip())]
    if ref:
        return ref[-1][:600]
    return res.reason or f'exit {res.rc}'


def unlink_symlinks(d, pattern='*'):
    n = 0
    for p in glob.glob(os.path.join(d, pattern)):
        if os.path.islink(p):
            os.unlink(p)
            n += 1
    return n


def link_tree(src, dst, include):
    """Symlink mirror: dst/rel -> src/rel for every file under src with include(rel) true, unless dst/rel exists."""
    n = 0
    if not os.path.isdir(src) or os.path.realpath(src) == os.path.realpath(dst):
        return 0
    for dp, _, fns in os.walk(src):
        rel = os.path.relpath(dp, src)
        for fn in fns:
            r = os.path.normpath(os.path.join(rel, fn))
            if not include(r):
                continue
            t = os.path.join(dst, r)
            if os.path.lexists(t):
                continue
            os.makedirs(os.path.dirname(t), exist_ok=True)
            os.symlink(os.path.join(src, r), t)
            n += 1
    return n


def clean_start(run, args, unit, present):
    """S-G12 (nocat:62-68): real files of an earlier run where this unit writes refuse, unless an earlier record of this
    same unit lists them (rebuilt by this step) or --fresh is given (the engine overwrites them)."""
    from exopipe.phase3.core import record
    from exopipe.phase3.core.record import Refusal
    present = sorted(p for p in present if os.path.exists(p) and not os.path.islink(p))
    if not present or getattr(args, 'fresh', False):
        return
    rec = record.load_record(run.out_root, run.step, unit)
    mine = {o['path'] for o in (rec or {}).get('outputs', [])}
    foreign = [p for p in present if os.path.abspath(p) not in mine]
    if foreign:
        raise Refusal(f'S-G12 products of an earlier run present ({len(foreign)}, e.g. {foreign[0]}) that no record of this '
                      f'unit lists - move them aside, or pass --fresh to let the engine overwrite them')


def try_resume(run, args, name, inputs, admitted):
    if not getattr(args, 'resume', False) or getattr(args, 'fresh', False):
        return False
    ok, why = run.resume_check(name, inputs=inputs, admitted=admitted)
    if ok:
        u = run.unit(name)
        u.status, u.reason = 'skipped-valid', why
        say(f'frames: {name} skipped (--resume: {why})')
    return ok


def parallel(fn, items, workers):
    """fn(item) for every item, ``workers`` at a time (threads: the work is in engine subprocesses). A unit's error is in
    its record already; it is printed and the others go on."""
    from concurrent.futures import ThreadPoolExecutor
    if not items:
        return
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers or 1), len(items)))) as ex:
        futs = [ex.submit(fn, it) for it in items]
        for f in futs:
            try:
                f.result()
            except Exception as e:
                say(f'frames: unit error: {type(e).__name__}: {str(e)[:300]}')


def exclusions():
    """The 10-minute tier's read, per night (core.exclusions.for_tier; the in-place tree and wideband read the same way,
    clv15 3e64f79)."""
    from exopipe.phase3.core import exclusions as ex
    return ex.for_tier('sublst')


def report(u):
    say(f'frames: {u.name} -> {u.status}' + (f' ({u.reason[:300]})' if u.reason else ''))


def mtime(p, default=0.0):
    try:
        return os.stat(p).st_mtime
    except OSError:
        return default


def newest_mtime(paths, threads=16):
    """max mtime of many files (stat in threads: Lustre metadata latency dominates a serial scan)."""
    from concurrent.futures import ThreadPoolExecutor
    if not paths:
        return 0.0
    with ThreadPoolExecutor(max_workers=min(threads, len(paths))) as ex:
        return max(ex.map(mtime, paths))


def unit_refused(run, name, reason, dry, counter=None, **meta):
    if dry:
        plan(f'  UNIT {name}: WOULD REFUSE - {reason}')
        if counter is not None:
            counter.would_refuse += 1
        return
    with run.unit(name, **meta) as u:
        u.refuse(reason)
    report(u)


# =====================================================================================================================
# tier sublst
# =====================================================================================================================

def add_arguments(p):
    S = cfg().frames_sublst
    g = p.add_argument_group('tier sublst (lib/frames/tier_sublst.py; config frames_sublst)',
                             f'--stages of --tier sublst, in order: {",".join(S.ORDER)}; default per hour (--epochs) '
                             f'{",".join(S.STAGES["per-hour"])}, intermittent (--lsts) {",".join(S.STAGES["intermittent"])}')
    g.add_argument('--sublst-groups', metavar='G[,G]',
                   help='wideband groups (default frames_sublst.GROUPS = Red,Green,Blue; Full is not built today)')
    g.add_argument('--sublst-mirror', action='store_true',
                   help='non-production --out: link the production inputs a stage needs but the root lacks (frames-only '
                        'parity runs on production stacks / residuals; spec 7.3)')
    g.add_argument('--supersede', metavar='TAG',
                   help='stages stack and nonmember, with --fresh: move a unit\'s earlier outputs aside under '
                        '<out>/_superseded/<TAG>/ with a sha1 manifest instead of refusing (as `masters --supersede`; the '
                        'stages read it already, the flag was missing here)')
    g.add_argument('--sublst-loo-evidence', choices=('record', 'log'),
                   help="what proves arm A complete before wideband: 'record' = the done masters loo record (default "
                        "frames_sublst.LOO_EVIDENCE), 'log' = production's SUBLST-LOO log lines (parity on production inputs)")


def default_out(args):
    return cfg().paths.SUBLST_ROOT


class _Ctx:
    def __init__(self, args, run):
        self.args, self.run, self.C = args, run, cfg()
        self.S = self.C.frames_sublst
        self.dry = bool(getattr(args, 'dry_run', False))
        self.root = os.path.abspath(run.out_root)
        self.prod = os.path.abspath(self.C.paths.SUBLST_ROOT)
        self.mirror = os.path.realpath(self.root) != os.path.realpath(self.prod) and bool(getattr(args, 'sublst_mirror', False))
        self.scopes = split_list(getattr(args, 'scopes', None))
        self.stokes = [s for s in ('I', 'V') if s in (split_list(getattr(args, 'stokes', None)) or ['I', 'V'])]
        self.workers = getattr(args, 'workers', None)
        self.would_refuse = 0
        self.planned = 0
        self.stages = ()              # the stages of this invocation, set by run()

    def lst_dir(self, lst, base=None):
        return os.path.join(base or self.root, lst)

    def x2_dir(self, lst, base=None):
        return os.path.join(base or self.root, f'{lst}_x2')

    def tree_dir(self, base=None):
        return os.path.join(base or self.root, self.S.TREE_SUBDIR)

    def input_base(self, sub):
        """For a dry run / verify: the root that holds an input directory (this root; production when mirroring and absent)."""
        return self.prod if (self.mirror and not os.path.isdir(os.path.join(self.root, sub))) else self.root

    def groups(self):
        g = split_list(getattr(self.args, 'sublst_groups', None)) or list(self.S.GROUPS)
        bad = [x for x in g if x not in _GROUPS_ALL]
        if bad:
            raise ValueError(f'--sublst-groups: unknown group(s) {bad}')
        if self.scopes and any(s in _GROUPS_ALL for s in self.scopes):
            g = [x for x in g if x in self.scopes]
        return g

    def loo_evidence(self):
        v = getattr(self.args, 'sublst_loo_evidence', None) or self.S.LOO_EVIDENCE
        if v not in ('record', 'log'):
            raise ValueError(f'frames_sublst.LOO_EVIDENCE must be record or log, not {v!r}')
        return v

    def gate(self, name, ok, detail):
        if self.dry:
            plan(f'    gate {name}: {"pass" if ok else "WOULD REFUSE"} - {detail}')
            if not ok:
                self.would_refuse += 1
        return ok


def parse_items(args):
    """({lst: [nights]} from --epochs / --cells (night cells), [lsts] from --lsts)."""
    from exopipe.phase3.core import cells
    by = {}
    toks = []
    if getattr(args, 'epochs', None):
        toks += cells.expand_epochs(args.epochs)
    if getattr(args, 'cells', None):
        toks += cells.expand(args.cells, kind='night cell')
    for t in toks:
        lst, night, scope = cells.parse_night_cell(t)
        if scope:
            raise ValueError(f'{t}: a sub-LST frames item is a whole night (<LST>_<date>); narrow bands with --scopes')
        by.setdefault(lst, [])
        if night not in by[lst]:
            by[lst].append(night)
    lsts = [cells.parse_lst(x) for x in cells.expand(getattr(args, 'lsts', None), universe=cells.LSTS, kind='LST')] \
        if getattr(args, 'lsts', None) else []
    return by, lsts


def run(args, run):
    import exopipe.phase3.lib.frames.tests_sublst  # noqa: F401  (registers this tier's content tests)
    ctx = _Ctx(args, run)
    try:
        by_night, lsts = parse_items(args)
        if by_night and lsts:
            raise ValueError('give --epochs/--cells (per hour: nights) or --lsts (intermittent: every night), not both')
        if not by_night and not lsts:
            raise ValueError('tier sublst needs --epochs <LST>_<date> (per hour) or --lsts <LST> (intermittent)')
        mode = 'per-hour' if by_night else 'intermittent'
        stages = requested_stages(args, ctx.S.ORDER, ctx.S.STAGES[mode])
        ctx.stages = tuple(stages)
        ctx.groups()
        ctx.loo_evidence()
        from exopipe.phase3.core import cells
        nights_sel = cells.expand(args.nights) if getattr(args, 'nights', None) else []
        for n in nights_sel:
            cells.parse_date(n)
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        return 1
    check_overrides(('frames_sublst',))
    exclusions()                                      # C-09: a missing exclusion list refuses before anything runs
    if not ctx.dry:
        refuse_production_writes([ctx.root])
    work = {lst: sorted(ns) for lst, ns in by_night.items()}
    for lst in lsts:
        work[lst] = sorted(nights_sel)
    plan(f'frames --tier sublst ({mode}): stages {",".join(stages)}; LSTs {",".join(sorted(work))}; sub-LST root '
         f'{ctx.root}' + (f' (inputs it lacks are linked from {ctx.prod})' if ctx.mirror else
                          (' (PRODUCTION root)' if os.path.realpath(ctx.root) == os.path.realpath(ctx.prod) else ''))
         + ('; DRY RUN (nothing written)' if ctx.dry else ''))
    for lst in sorted(work):
        ns = work[lst]
        for st in stages:
            if st == 'stack':
                stage_stack(ctx, lst, ns)
            elif st == 'nonmember-list':
                stage_nonmember_list(ctx, lst, ns)
            elif st == 'nonmember':
                stage_nonmember(ctx, lst, ns)
            elif st == 'wideband':
                stage_wideband(ctx, lst, ns or None, per_hour=(mode == 'per-hour'))
            elif st == 'tree':
                stage_tree(ctx, lst)
            elif st == 'tree-nonmember':
                from exopipe.phase3.lib.frames import sublst_tree_nonmember
                sublst_tree_nonmember.stage(ctx, lst, ns)
            elif st == 'searchframes':
                unit_refused(run, f'{lst}_searchframes', 'not offered in the window: `search --tier sublst` '
                             '(run_deep_cell(cadence=\'10min\')) stays the producer of the sub-LST search frames '
                             '(crosscheck decision 6, group 14; frames.md B1/B15)', ctx.dry, ctx, tier=TIER)
            elif st == 'verify':
                stage_verify(ctx, lst)
    if ctx.dry:
        plan(f'plan: {ctx.planned} unit(s) of this tier; {ctx.would_refuse} gate(s) would refuse')
        return 2 if ctx.would_refuse else (0 if ctx.planned else 3)
    return None


# ---------------------------------------------------------------------------------------------------------------------
# stage: stack (per hour) - the masters wrapper (crosscheck decision 6)
# ---------------------------------------------------------------------------------------------------------------------

def stage_stack(ctx, lst, nights):
    """The night's window stacks through lib/masters/tier_sublst.stack_night (the ONE wrapper, config masters_sublst):
    its admit unit (admit_<LST>_<night>: the job-list rules of nocat:82-186) and one stack unit per admitted subband
    (stack_<LST>_<night>_<sb>: clv15/phase3_sublst_stack.py in place, Stokes I then V), recorded under this run."""
    from exopipe.phase3.core.record import Refusal
    if not nights:
        unit_refused(ctx.run, f'{lst}_stack', 'the stack stage stacks ONE night per hour: give --epochs <LST>_<date> (or '
                     '--lsts with --nights); every night of an LST is `masters --tier sublst --stages admit,stack`',
                     ctx.dry, ctx, tier=TIER)
        return
    from exopipe.phase3.lib.masters import tier_sublst as MT
    sbs = [s for s in ctx.C.instrument.ALL_SUBBANDS if not ctx.scopes or s in ctx.scopes] or None
    for n in nights:
        if ctx.dry:
            plan(f'  STAGE stack {lst}_{n}: lib/masters/tier_sublst.stack_night (admit + stack; config masters_sublst) '
                 f'into {ctx.root}')
        try:
            res = MT.stack_night(ctx.run, lst, n, sublst_root=ctx.root, subbands=sbs, stokes=tuple(ctx.stokes),
                                 workers=ctx.workers, fresh=bool(getattr(ctx.args, 'fresh', False)),
                                 supersede=getattr(ctx.args, 'supersede', None),
                                 resume=bool(getattr(ctx.args, 'resume', False)))
        except (Refusal, SystemExit) as e:
            unit_refused(ctx.run, f'{lst}_{n}_stack', f'masters sub-LST wrapper: {e}', ctx.dry, ctx, tier=TIER)
            continue
        if ctx.dry:
            ctx.planned += 1
            ctx.would_refuse += int(res.get('would_refuse') or 0)
        else:
            say(f'frames: stack {lst}_{n}: {len(res.get("stack_units", []))} stack unit(s), {len(res.get("jobs", []))} '
                f'job(s), left out: {"; ".join(res.get("left_out") or []) or "none"}')


# ---------------------------------------------------------------------------------------------------------------------
# stage: nonmember (per hour) - the masters wrapper's residual of a night that is not in the window Masters
# ---------------------------------------------------------------------------------------------------------------------

def stage_nonmember_list(ctx, lst, nights):
    """Put the named night(s) on the non-member nights list when the stack admission needs them there, through
    lib/masters/sublst_nonmember.list_night (rules L-1 .. L-6 there): one unit nonmember-list_<LST>_<night>. Writes nothing
    when every subband has a usable qa3_scale in the frozen deep weights, or when the night is listed already. A line of
    the list switches the admission of the night in EVERY subband, so the step always decides from every subband:
    --scopes does not narrow it."""
    from exopipe.phase3.core.record import Refusal
    if not nights:
        unit_refused(ctx.run, f'{lst}_nonmember-list', 'the nonmember-list stage lists ONE named night: give --epochs '
                     '<LST>_<date> (or --lsts with --nights)', ctx.dry, ctx, tier=TIER)
        return
    from exopipe.phase3.lib.masters import sublst_nonmember as NM
    if ctx.scopes:
        say(f'frames: nonmember-list {lst}: --scopes is not applied (a line of the list is per LST and night: every '
            f'subband is read)')
    for n in nights:
        try:
            res = NM.list_night(ctx.run, lst, n)
        except (Refusal, SystemExit) as e:
            unit_refused(ctx.run, NM.list_unit_name(lst, n), f'masters sub-LST wrapper: {e}', ctx.dry, ctx, tier=TIER)
            continue
        if ctx.dry:
            ctx.planned += 1
            ctx.would_refuse += int(res.get('would_refuse') or 0)
        else:
            say(f'frames: nonmember-list {lst}_{n}: {res.get("action")}')


def stage_nonmember(ctx, lst, nights):
    """The arm-A residuals of the named night(s) against the INSTALLED window Masters, through
    lib/masters/sublst_nonmember.nonmember_night (config masters_sublst): one unit nonmember_<LST>_<night>_<sb> per subband
    of the night's plan, recorded under this run. A night that is in the Masters refuses there and names the batch stage."""
    from exopipe.phase3.core.record import Refusal
    if not nights:
        unit_refused(ctx.run, f'{lst}_nonmember', 'the nonmember stage subtracts ONE named night against the installed '
                     'window Masters: give --epochs <LST>_<date> (or --lsts with --nights); the residuals of every night '
                     'of an LST are `masters --tier sublst --stages loo`', ctx.dry, ctx, tier=TIER)
        return
    from exopipe.phase3.lib.masters import sublst_nonmember as NM
    sbs = [s for s in ctx.C.instrument.ALL_SUBBANDS if not ctx.scopes or s in ctx.scopes] or None
    for n in nights:
        if ctx.dry:
            plan(f'  STAGE nonmember {lst}_{n}: lib/masters/sublst_nonmember.nonmember_night (night minus the installed '
                 f'window Masters under {os.path.join(ctx.root, lst + "_masters")}; config masters_sublst) into {ctx.root}')
        try:
            res = NM.nonmember_night(ctx.run, lst, n, sublst_root=ctx.root, subbands=sbs, stokes=tuple(ctx.stokes),
                                     workers=ctx.workers, fresh=bool(getattr(ctx.args, 'fresh', False)),
                                     supersede=getattr(ctx.args, 'supersede', None),
                                     resume=bool(getattr(ctx.args, 'resume', False)),
                                     stack_planned='stack' in ctx.stages)
        except (Refusal, SystemExit) as e:
            unit_refused(ctx.run, f'{lst}_{n}_nonmember', f'masters sub-LST wrapper: {e}', ctx.dry, ctx, tier=TIER)
            continue
        if ctx.dry:
            ctx.planned += 1
            ctx.would_refuse += int(res.get('would_refuse') or 0)
        else:
            done = [u for u in ctx.run.units if u.name in res.get('units', []) and u.status == 'done']
            say(f'frames: nonmember {lst}_{n}: {len(res.get("units", []))} unit(s), {len(done)} done')


# ---------------------------------------------------------------------------------------------------------------------
# mirrors (non-production root): production inputs this root lacks
# ---------------------------------------------------------------------------------------------------------------------

def _is_band_dir(rel, lst, bands):
    parts = rel.split(os.sep)
    return len(parts) == 4 and parts[0].startswith(lst + '_') and parts[1] in bands and parts[2] in ('I', 'V')


def mirror_inputs(ctx, lst, subbands, with_groups=False):
    """Link into this root the production inputs of wideband / tree that it lacks: member window stacks, _pb, _nframes and
    arm-A residuals (<LST>/), arm-B residuals (<LST>_x2/), the refine logs and arm A's loo logs; with_groups also the
    wideband products of production for groups this root has no real product of (tree / verify without a wideband here)."""
    if not ctx.mirror:
        return 0
    bands = set(subbands) | (set(_GROUPS_ALL) if with_groups else set())
    pat = re.compile(r'^sublst_%s_\d{4}-\d\d-\d\d_(\d+MHz|Red|Green|Blue|Full)_[IV]_w\d(_pb|_nframes|_resid_loo)?\.fits$' % lst)
    own_groups = set()
    for p in glob.glob(os.path.join(ctx.lst_dir(lst), f'{lst}_*', '*', '[IV]', '*.fits')):
        g = os.path.basename(os.path.dirname(os.path.dirname(p)))
        if g in _GROUPS_ALL and not os.path.islink(p):
            own_groups.add(g)

    def inc_stack(rel):
        if rel.startswith('logs' + os.sep):
            return bool(re.match(r'^loo_%s_\d+MHz_[IV]\.log$' % lst, os.path.basename(rel)))
        m = pat.match(os.path.basename(rel))
        return bool(m and _is_band_dir(rel, lst, bands) and m.group(1) not in own_groups)

    def inc_x2(rel):
        if rel.startswith('logs' + os.sep):
            return bool(re.match(r'^%s_\d+MHz\.log$' % lst, os.path.basename(rel)))
        m = pat.match(os.path.basename(rel))
        return bool(m and m.group(2) == '_resid_loo' and _is_band_dir(rel, lst, bands) and m.group(1) not in own_groups)
    n = link_tree(ctx.lst_dir(lst, ctx.prod), ctx.lst_dir(lst), inc_stack)
    n += link_tree(ctx.x2_dir(lst, ctx.prod), ctx.x2_dir(lst), inc_x2)
    say(f'frames: mirror {lst}: {n} new link(s) into {ctx.root} from {ctx.prod}')
    return n


# ---------------------------------------------------------------------------------------------------------------------
# stage: wideband
# ---------------------------------------------------------------------------------------------------------------------

def _stacks(root, lst, sb, St):
    return sorted(glob.glob(os.path.join(root, f'{lst}_*', sb, St, f'sublst_{lst}_*_{sb}_{St}_w[0-9].fits')))


def _nights(root, lst):
    out = []
    for p in glob.glob(os.path.join(root, f'{lst}_*')):
        n = os.path.basename(p).split('_', 1)[1]
        if os.path.isdir(p) and re.match(r'^\d{4}-\d\d-\d\d$', n):
            out.append(n)
    return sorted(out)


def arm_a_complete(ctx, lst, sb, base=None):
    """(ok, why, evidence path): arm A of (LST, sb) is complete - the masters loo record (default) or production's log lines."""
    from exopipe.phase3.core import record
    base = base or ctx.root
    if ctx.loo_evidence() == 'record':
        p = record.record_path(base, 'masters', f'loo_{lst}_{sb}')
        rec = record.load_record(base, 'masters', f'loo_{lst}_{sb}')
        st = rec.get('status') if rec else 'missing'
        return st == 'done', f'masters loo record {p}: {st}', p
    root = ctx.lst_dir(lst, base)
    bad = []
    for St in ('I', 'V'):
        log = os.path.join(root, 'logs', f'loo_{lst}_{sb}_{St}.log')
        try:
            ok = any(l.startswith(f'SUBLST-LOO {lst} {sb} {St} windows=') for l in open(log, errors='replace'))
        except OSError:
            ok = False
        if not ok:
            bad.append(log)
    return not bad, (f'no SUBLST-LOO line in {bad}' if bad else 'SUBLST-LOO lines present (production logs)'), \
        os.path.join(root, 'logs', f'loo_{lst}_{sb}_I.log')


def night_arm_a_complete(ctx, lst, sb, night):
    """Arm A of ONE night of the per-hour path at one subband, from the record of the night's own unit
    nonmember_<LST>_<night>_<sb> (stage nonmember) under this root: (ok, why, record path), ok only for status done. None
    when this root holds no record of that unit: the night's arm-A residuals at that subband are then not the per-hour
    unit's (the night is in the window Masters there, or the batch stage wrote them) and the LST-wide evidence decides."""
    from exopipe.phase3.core import record
    from exopipe.phase3.lib.masters import sublst_nonmember as NM
    name = NM.unit_name(lst, night, sb)
    rec = record.load_record(ctx.root, 'frames', name)
    if not rec:
        return None
    p = record.record_path(ctx.root, 'frames', name)
    st = rec.get('status')
    return st == 'done', f'record of the night\'s own arm-A unit {name} ({p}): {st}', p


def wideband_arm_a(ctx, lst, sb, night=None, base=None):
    """The arm-A evidence the wideband stage reads for one member subband: (ok, why, evidence path, how to complete it).
    With `night` (per hour, ONE night) the night's own record decides where this root holds one (night_arm_a_complete);
    otherwise, and for every LST-wide run, arm_a_complete."""
    own = night_arm_a_complete(ctx, lst, sb, night) if night else None
    if own is not None:
        return own + (f'run `frames --tier sublst --stages nonmember --epochs {lst}_{night}` for {sb} first',)
    return arm_a_complete(ctx, lst, sb, base) + ('run `masters --tier sublst --stages loo` first (nocat:232-234; '
                                                 'crosscheck decision 6)',)


def refine_complete(ctx, lst, sb, base=None):
    """S-G8 (nocat:236-247): (ok, why, log). THIS run's refine = its log newer than the subband's newest window stack (the
    production driver compared with its RUN-START, before the stacks); complete = final line refused=0, or
    exopipe_ops/sublst_refine_approved.py (in place) accepts the refusals as exactly the approved windows."""
    from exopipe.phase3.core import legacy
    log = os.path.join(ctx.x2_dir(lst, base), 'logs', f'{lst}_{sb}.log')
    if not os.path.exists(log):
        return False, f'no refine log {log}', log
    newest = newest_mtime([p for St in ('I', 'V') for p in _stacks(ctx.lst_dir(lst, base), lst, sb, St)])
    if mtime(log) < newest:
        return False, f'refine log {log} is older than the newest {sb} window stack (a refine of an earlier run)', log
    fin = [l.strip() for l in open(log, errors='replace') if l.startswith(f'SUBLST-REFINE {lst} {sb} ')]
    if not fin:
        return False, f'refine log {log} has no final SUBLST-REFINE line', log
    if re.search(r'refused=0\b', fin[-1]):
        return True, fin[-1], log
    try:
        script = legacy_script('exopipe_ops', ctx.S.REFINE_APPROVED_SCRIPT)
        res = legacy.run(script, [lst, sb, log], python=interpreter_key('frames_sublst', 'refine_approved'),
                         refusal_codes=())
    except Exception as e:
        return False, f'{fin[-1]} - approvals not judged: {e}', log
    if res.rc == 0:
        return True, f'{fin[-1]} (refusals are exactly the approved windows)', log
    last = res.tail.strip().splitlines()[-1][:200] if res.tail.strip() else f'exit {res.rc}'
    return False, f'{fin[-1]} - refusals not approved ({last})', log


def group_members(G):
    from exopipe.phase3.core import cells
    return list(cells.band_members(G))


# The wideband engine's lines (clv15/phase3_sublst_wideband.py since clv15 582c92e, 2026-10-23). Per (night, window k, group)
# and Stokes, a window is built only when every member but 18 MHz has every product of that window for that Stokes and the V
# stack its weight comes from; else one REFUSED-INCOMPLETE line per Stokes. A window built in one Stokes says
# '(Stokes X only)'. The call ends with one count line, SUBLST-WIDEBAND ... refused_incomplete=<n>.
_WB_WINDOW = re.compile(r'^WIDEBAND (?P<lst>\S+) (?P<night>\S+) (?P<g>\S+) w(?P<k>\d+) members=(?P<members>[^\s;]+) '
                        r'weights=(?P<weights>[^\s;]+)(?: \(Stokes (?P<only>[IV]) only\))?(?:; left out: (?P<left>.*))?$')
_WB_INCOMPLETE = re.compile(r'^REFUSED-INCOMPLETE (?P<lst>\S+) (?P<night>\S+) (?P<g>\S+) (?P<S>[IV]) w(?P<k>\d+): '
                            r'required member\(s\) (?P<missing>\S*) lack a product\b')
_WB_SKIP = re.compile(r'^SKIP (?P<lst>\S+) (?P<night>\S+) (?P<g>\S+) w(?P<k>\d+): ')
_WB_BAND_RULE = re.compile(r'^REFUSED-BAND-RULE (?P<lst>\S+) (?P<night>\S+) (?P<g>\S+): ')
_WB_COUNTS = re.compile(r'^SUBLST-WIDEBAND (?P<lst>\S+) groups=(?P<groups>\S+) windows=(?P<windows>\d+) '
                        r'skipped=(?P<skipped>\d+) refused_band_rule=(?P<band_rule>\d+) '
                        r'refused_incomplete=(?P<incomplete>\d+)$')
_WB_ITEM_LINES = ('REFUSED-INCOMPLETE ', 'REFUSED-BAND-RULE ')    # item outcomes: never the reason an engine call refused
# what marks a failed call as the engine's refusal: its sys.exit('REFUSED: ...'), and production modules it calls refuse with
# RuntimeError('... REFUSING ...') (exopipe flux_correction.pixel_factor: FLUXBAKE without FLUXCORR; coadd_full_pipeline
# .load_exclusions: no exclusion list, C-09) - legacy.run's texts with 'REFUSED' narrowed to 'REFUSED:', so the item lines
# above never make a crash a refusal
_WB_REFUSAL_TEXT = ('REFUSED:', 'REFUSING', 'REFUSE:')


def wideband_outcome(lines, lst, night, G):
    """What one wideband engine call (--lst <lst> --groups <G> --nights <night>) reported, reconciled with its count line.
    Returns {'windows': [(k, built Stokes tuple, members, line)], 'incomplete': [(S, k, missing members, line)],
    'skipped': [line], 'band_rule': [line], 'counts': {windows, skipped, band_rule, incomplete}, 'count_line': line}.
    ValueError when a line names another LST / night / group, a WIDEBAND, REFUSED-INCOMPLETE or count line does not parse (an
    engine older than clv15 582c92e prints no refused_incomplete=), a window or a (Stokes, window) is reported twice, a Stokes
    is both built and refused, a window built in one Stokes has no REFUSED-INCOMPLETE line for the other, or the count line is
    missing (the call did not finish) or disagrees with the lines."""
    out = {'windows': [], 'incomplete': [], 'skipped': [], 'band_rule': [], 'counts': None, 'count_line': None}
    built_at, refused_at = {}, set()
    for raw in lines:
        line = raw.strip()
        q = _WB_COUNTS.match(line)
        if q:
            if out['counts'] is not None:
                raise ValueError(f'two count lines: {line[:160]}')
            if (q['lst'], q['groups']) != (lst, G):
                raise ValueError(f'count line of another LST / group: {line[:160]}')
            out['counts'] = {k: int(q[k]) for k in ('windows', 'skipped', 'band_rule', 'incomplete')}
            out['count_line'] = line
            continue
        for kind, rx in (('windows', _WB_WINDOW), ('incomplete', _WB_INCOMPLETE), ('skipped', _WB_SKIP),
                         ('band_rule', _WB_BAND_RULE)):
            q = rx.match(line)
            if q:
                break
        else:
            if line.startswith('SUBLST-WIDEBAND '):
                raise ValueError(f'count line not understood (an engine before clv15 582c92e prints no refused_incomplete=): '
                                 f'{line[:160]}')
            if line.startswith(('WIDEBAND ', 'REFUSED-INCOMPLETE ')):
                raise ValueError(f'engine line not understood: {line[:160]}')
            continue
        if (q['lst'], q['night'], q['g']) != (lst, night, G):
            raise ValueError(f'a line of another LST / night / group: {line[:160]}')
        if kind == 'windows':
            k = int(q['k'])
            if k in built_at:
                raise ValueError(f'window w{k} reported twice')
            built_at[k] = (q['only'],) if q['only'] else ('I', 'V')
            out['windows'].append((k, built_at[k], q['members'].split(','), line))
        elif kind == 'incomplete':
            key = (q['S'], int(q['k']))
            if key in refused_at:
                raise ValueError(f'Stokes {key[0]} w{key[1]} refused twice')
            refused_at.add(key)
            out['incomplete'].append((key[0], key[1], [x for x in q['missing'].split(',') if x], line))
        else:
            out[kind].append(line)
    if out['counts'] is None:
        raise ValueError('no count line "SUBLST-WIDEBAND ... refused_incomplete=<n>" (the call did not finish)')
    seen = {'windows': len(out['windows']), 'skipped': len(out['skipped']), 'band_rule': len(out['band_rule']),
            'incomplete': len(out['incomplete'])}
    if seen != out['counts']:
        raise ValueError(f'the count line "{out["count_line"][:200]}" disagrees with the lines: {seen}')
    for k, built in built_at.items():
        for S in ('I', 'V'):
            if S in built and (S, k) in refused_at:
                raise ValueError(f'w{k} Stokes {S} both built and refused as incomplete')
            if S not in built and (S, k) not in refused_at:
                raise ValueError(f'w{k} built in Stokes {built[0]} only, with no REFUSED-INCOMPLETE line for Stokes {S}')
    return out


def wideband_products(root, x2, lst, night, G, S, k):
    """The five products of one wideband (night, window, Stokes): stack, _pb, _nframes, set-A residual (<root>), set-B
    residual (<x2>) - the engine's names (wideband:181-185, clv15 582c92e)."""
    b = f'sublst_{lst}_{night}_{G}_{S}_w{k}'
    d, dx = os.path.join(root, f'{lst}_{night}', G, S), os.path.join(x2, f'{lst}_{night}', G, S)
    return [os.path.join(d, b + '.fits'), os.path.join(d, b + '_pb.fits'), os.path.join(d, b + '_nframes.fits'),
            os.path.join(d, b + '_resid_loo.fits'), os.path.join(dx, b + '_resid_loo.fits')]


def stage_wideband(ctx, lst, nights, per_hour):
    from exopipe.phase3.core.record import NoInput
    S = ctx.S
    groups = ctx.groups()
    members = sorted({sb for G in groups for sb in group_members(G)}, key=lambda s: int(s[:-3]))
    if ctx.dry:
        base = ctx.input_base(lst)
        alln = nights or _nights(ctx.lst_dir(lst, base), lst)
        plan(f'  STAGE wideband {lst}: groups {",".join(groups)}; nights '
             f'{",".join(alln) if len(alln) <= 6 else "all %d" % len(alln)}; inputs {ctx.lst_dir(lst, base)} + '
             f'{ctx.x2_dir(lst, base)}; outputs {ctx.lst_dir(lst)} + {ctx.x2_dir(lst)}')
        one_night = alln[0] if per_hour and len(alln) == 1 else None
        for sb in members:
            ok, why, _, _ = wideband_arm_a(ctx, lst, sb, one_night, base if ctx.loo_evidence() == 'log' else None)
            ctx.gate(f'arm A complete {sb}', ok, why)
            ok, why, _ = refine_complete(ctx, lst, sb, base)
            ctx.gate(f'S-G8 x2 complete {sb}', ok, why)
        ok, py = check_interpreter(interpreter_key('frames_sublst', 'wideband'))
        ctx.gate('interpreter', ok, py)
        for G in groups:
            plan(f'    UNIT {lst}_wideband_{G}' + (f'_{alln[0]}' if per_hour and len(alln) == 1 else '')
                 + f': phase3_sublst_wideband.py --root {ctx.lst_dir(lst)} --x2-root {ctx.x2_dir(lst)} --lst {lst} '
                   f'--groups {G} --nights <each of {len(alln)}>')
            ctx.planned += 1
        return
    script = legacy_script('clv15', 'phase3_sublst_wideband.py')
    py = interpreter_key('frames_sublst', 'wideband')
    if ctx.mirror:
        mirror_inputs(ctx, lst, members)
    root, x2 = ctx.lst_dir(lst), ctx.x2_dir(lst)
    alln = list(nights) if nights else _nights(root, lst)
    ex = exclusions()
    one_night = alln[0] if per_hour and len(alln) == 1 else None

    def one(G):
        name = f'{lst}_wideband_{G}' + (f'_{one_night}' if one_night else '')
        mem = group_members(G)
        ev = {sb: (wideband_arm_a(ctx, lst, sb, one_night), refine_complete(ctx, lst, sb)) for sb in mem}
        inputs = [p for sb in mem for p in (ev[sb][0][2], ev[sb][1][2]) if os.path.exists(p)] + [script]
        admitted = ex.admitted(lst, G, alln)
        if try_resume(ctx.run, ctx.args, name, inputs, admitted):
            return
        with ctx.run.unit(name, tier=TIER, stage='wideband', lst=lst, group=G, nights=alln) as u:
            for p in inputs:
                u.add_input(p)
            u.set_admitted(admitted)
            for sb in mem:
                (aok, awhy, _, ahow), (rok, rwhy, _) = ev[sb]
                if not aok:
                    u.refuse(f'arm A of {lst} {sb} not complete ({awhy}): {ahow}')
                if not rok:
                    u.refuse(f'S-G8 x2 refine of {lst} {sb} not complete: {rwhy}')
            run_n = []
            for n in alln:
                if n in admitted:
                    run_n.append(n)
                else:
                    u.refuse_item(os.path.join(root, f'{lst}_{n}', G), f'({lst}, {G}, {n}) is excluded under '
                                  f'core.exclusions (band rule) - not passed to the engine', gate='exclusions')
            if not run_n:
                raise NoInput(f'no admitted night with window stacks under {root}'
                              + (f' ({len(u.refused_items)} excluded)' if u.refused_items else ''))
            outdirs = [os.path.join(r, f'{lst}_{n}', G, St) for n in run_n for r in (root, x2) for St in ('I', 'V')]
            if ctx.mirror:
                for d in outdirs:
                    unlink_symlinks(d)
            clean_start(ctx.run, ctx.args, name, [p for d in outdirs for p in glob.glob(os.path.join(d, '*.fits'))])
            t0 = time.time()
            results = {}

            def night(n):
                log = os.path.join(root, 'logs', f'wideband_{lst}_{G}_{n}.log')
                # a call refuses by the engine's REFUSED: / REFUSING exits only; its REFUSED-INCOMPLETE / REFUSED-BAND-RULE
                # lines are item outcomes, so a crash after them stays an error (_WB_REFUSAL_TEXT)
                results[n] = (log,) + engine(script, ['--root', root, '--x2-root', x2, '--lst', lst, '--groups', G,
                                                      '--nights', n], py, log, unit=u, refusal_text=_WB_REFUSAL_TEXT)
            parallel(night, run_n, ctx.workers or S.WIDEBAND_PARALLEL)
            nwin = n_one = 0
            tally = {'windows': 0, 'skipped': 0, 'band_rule': 0, 'incomplete': 0}
            u.meta['wideband_counts'] = {}
            for n in run_n:
                if n not in results:
                    raise RuntimeError(f'wideband {G} {n}: the engine call did not complete (see the run log)')
                log, res, lines = results[n]
                if not res.ok:
                    why = refusal_line([l for l in lines if not l.strip().startswith(_WB_ITEM_LINES)], res)
                    if res.refused:
                        u.refuse(f'wideband {G} {n}: {why} (log {log})')
                    raise RuntimeError(f'wideband {G} {n} failed: {why} (log {log})')
                try:
                    o = wideband_outcome(lines, lst, n, G)
                except ValueError as e:
                    raise RuntimeError(f'wideband {G} {n}: the engine log is not understood - {e} (log {log})')
                u.meta['wideband_counts'][n] = o['counts']
                for kk in tally:
                    tally[kk] += o['counts'][kk]
                for k, built, _members, l in o['windows']:
                    nwin += 1
                    n_one += len(built) == 1
                    for St in built:                 # a window built in one Stokes only has that Stokes' products only
                        for p in wideband_products(root, x2, lst, n, G, St, k):
                            if not (os.path.exists(p) and mtime(p) >= t0 - 2):
                                raise RuntimeError(f'wideband reported "{l[:120]}" but did not write {p}')
                            u.add_output(p, 'sublst_wideband')
                for St, k, _missing, l in o['incomplete']:
                    # S-G9 required members (clv15 582c92e): no wideband from a partial set, 18 MHz the one exception
                    u.refuse_item(wideband_products(root, x2, lst, n, G, St, k)[0], l[:300], gate='S-G9')
                for l in o['skipped']:
                    u.note(l[:300])
                for l in o['band_rule']:                        # the engine's own band-rule refusal (clv15 6f5fb31)
                    u.refuse_item(os.path.join(root, f'{lst}_{n}', G), l[:300], gate='exclusions')
            # S-G12, production's RUN-STARTS-CLEAN rule (nocat:76-82: an earlier run's products refuse the whole LST at its
            # start). Here --fresh (or this unit's own earlier record) lets the engine overwrite them, so what it did NOT write is
            # caught after it: any file or link in this unit's write directories that is not a product of this run's WIDEBAND
            # lines - a window the engine no longer builds, a (window, Stokes) it now refuses as incomplete, a .part leftover -
            # would be paired by the tree with this run's products
            mine = {os.path.abspath(o['path']) for o in u.outputs}
            stale = sorted(p for d in outdirs for p in glob.glob(os.path.join(d, '*')) if os.path.abspath(p) not in mine)
            if stale:
                u.refuse(f'S-G12 {len(stale)} file(s) in this unit\'s write directories that this run did not write, e.g. '
                         f'{stale[0]} - an earlier run\'s, which the tree would pair with this run\'s products (production '
                         f'refuses such an LST at its start, nocat:76-82): move them aside and re-run with --fresh')
            u.note(f'{nwin} wideband {G} window(s) over {len(run_n)} night(s)'
                   + (f', {n_one} built in one Stokes only' if n_one else '')
                   + (f'; {tally["incomplete"]} (night, window, Stokes) refused as incomplete (S-G9: a required member lacks '
                      f'a product; 18 MHz the one exception)' if tally['incomplete'] else ''))
            why0 = (f'{tally["incomplete"]} (night, window, Stokes) refused as incomplete (a required member lacks a product), '
                    f'{tally["skipped"]} window(s) skipped (fewer than {S.WIDEBAND_MIN_MEMBERS} members), '
                    f'{tally["band_rule"]} night(s) refused under the band rule; REFUSED-INCOMPLETE / SKIP lines in the logs')
            if nwin == 0 and not per_hour:
                u.refuse(f'S-G10 no wideband {G} windows over {lst} (nocat:261): {why0}')
            if nwin == 0:
                raise NoInput(f'no wideband {G} window for {",".join(run_n)}: {why0}')
        report(u)

    for G in groups:              # one group at a time; its nights run WIDEBAND_PARALLEL at a time (nocat:258)
        parallel(one, [G], 1)


# ---------------------------------------------------------------------------------------------------------------------
# stage: tree
# ---------------------------------------------------------------------------------------------------------------------

def tree_refine_gate(ctx, lst, b, base=None):
    """S-G8 for tree B: [(subband, ok, why, refine log)] for every subband whose arm-B residuals the band's tree reads (the
    band itself, or a group's members - its wideband set-B residuals are built from theirs). Production builds the tree only
    after the x2 refine of every subband is complete (nocat:236-247, then :264)."""
    sbs = group_members(b) if b in _GROUPS_ALL else [b]
    return [(sb,) + tuple(refine_complete(ctx, lst, sb, base)) for sb in sbs]


def stage_tree(ctx, lst):
    S = ctx.S
    bands = [b for b in S.TREE_BANDS if not ctx.scopes or b in ctx.scopes]
    tree = ctx.tree_dir()
    if ctx.dry:
        base = ctx.input_base(f'{lst}_x2')
        plan(f'  STAGE tree {lst}: {len(bands)} band(s) ({S.TREE_PARALLEL} at a time); residuals {ctx.x2_dir(lst, base)}; '
             f'window PBs {ctx.lst_dir(lst, base)}; tree -> {tree}')
        ok, py = check_interpreter(interpreter_key('frames_sublst', 'tree'))
        ctx.gate('interpreter', ok, py)
        for b in bands:
            n = len(glob.glob(os.path.join(ctx.x2_dir(lst, base), f'{lst}_*', b, '?', f'sublst_{lst}_*_{b}_?_w?_resid_loo.fits')))
            plan(f'    UNIT {lst}_tree_{b}: {n} arm-B residual file(s)')
            for sb, ok, why, _ in tree_refine_gate(ctx, lst, b, base):
                ctx.gate(f'S-G8 x2 complete {sb}', ok, why)
            ctx.planned += 1
        return
    script = legacy_script('clv15', 'phase3_sublst_tenmin_tree.py')
    py = interpreter_key('frames_sublst', 'tree')
    if ctx.mirror:
        mirror_inputs(ctx, lst, list(ctx.C.instrument.ALL_SUBBANDS), with_groups=True)
    os.makedirs(tree, exist_ok=True)
    root, x2 = ctx.lst_dir(lst), ctx.x2_dir(lst)
    from exopipe.phase3.lib.masters.tier_sublst import frozen_weights

    def one(b):
        name = f'{lst}_tree_{b}'
        res_ = sorted(glob.glob(os.path.join(x2, f'{lst}_*', b, '?', f'sublst_{lst}_*_{b}_?_w?_resid_loo.fits')))
        pbs = sorted(glob.glob(os.path.join(root, f'{lst}_*', b, 'I', f'sublst_{lst}_*_{b}_I_w?_pb.fits')))
        wp = frozen_weights(lst, b)
        rlog = os.path.join(x2, 'logs', f'{lst}_{b}.log')
        gate = tree_refine_gate(ctx, lst, b)
        inputs = res_ + pbs + [p for p in (wp, rlog, script) if os.path.exists(p)] \
            + [g[3] for g in gate if g[3] != rlog and os.path.exists(g[3])]
        nights = sorted({re.search(r'_(\d{4}-\d\d-\d\d)_', os.path.basename(p)).group(1) for p in res_})
        admitted = exclusions().admitted(lst, b, nights)
        if try_resume(ctx.run, ctx.args, name, inputs, admitted):
            return
        with ctx.run.unit(name, tier=TIER, stage='tree', lst=lst, band=b) as u:
            for p in inputs:
                u.add_input(p)
            u.set_admitted(admitted)
            for sb, ok, why, _ in gate:
                if not ok:
                    u.refuse(f'S-G8 x2 refine of {lst} {sb} not complete ({why}): tree B is built only after the refine of '
                             f'every subband it reads (nocat:236-247 before :264)')
            cdir = os.path.join(tree, f'{lst}_{b}')
            # one writer per band's tree: the per-night entry (stage tree-nonmember) takes the same lease
            from exopipe.phase3.lib.frames import sublst_tree_nonmember as TN
            lease = TN.tree_lease(tree, lst, b, name)
            lease.acquire(say=lambda m: say(f'frames: [{name}] {m}'))
            try:
                clean_start(ctx.run, ctx.args, name, [os.path.join(cdir, 'frames.csv')])
                log = os.path.join(root, 'logs', f'tree_{lst}_{b}.log')
                res, lines = engine(script, ['--resid-root', x2, '--stack-root', root, '--tree', tree, '--lst', lst,
                                             '--subbands', b], py, log, unit=u)
            finally:
                lease.release()
            fin = [l for l in lines if l.startswith(f'TREE {lst} {b} ')]
            if not res.ok or not fin:
                why = refusal_line(lines, res)
                if res.refused:
                    u.refuse(f'S-G11 tree: {why} (log {log})')
                raise RuntimeError(f'tree failed: {why} (log {log})')
            u.note(fin[-1][:300])
            u.add_output(os.path.join(cdir, 'frames.csv'), 'tenmin_tree', lst=lst, band=b)
            if 'NO-WINDOWS' in fin[-1]:
                u.add_output(os.path.join(cdir, 'NO_WINDOWS.txt'), 'tenmin_tree', lst=lst, band=b)
            for p in sorted(glob.glob(os.path.join(cdir, 'bin_masters', f'PB_bin*_{b}.fits'))):
                u.add_output(p, 'tenmin_tree', lst=lst, band=b)
        report(u)

    parallel(one, bands, ctx.workers or S.TREE_PARALLEL)


# ---------------------------------------------------------------------------------------------------------------------
# stage: verify (content test (b))
# ---------------------------------------------------------------------------------------------------------------------

def stage_verify(ctx, lst):
    base = ctx.input_base(lst)
    root, x2 = ctx.lst_dir(lst, base), ctx.x2_dir(lst, base)
    tree = ctx.tree_dir()
    if ctx.mirror and not glob.glob(os.path.join(tree, f'{lst}_*')):
        tree = ctx.tree_dir(ctx.prod)
    bands = [b for b in ctx.S.TREE_BANDS if os.path.isdir(os.path.join(tree, f'{lst}_{b}'))]
    name = f'{lst}_verify'
    if ctx.dry:
        plan(f'  STAGE verify {lst}: content test (b) R1-R3 + residual cards on {root} + {x2}; trees of {len(bands)} band(s) '
             f'under {tree}')
        plan(f'    UNIT {name}')
        ctx.planned += 1
        return
    with ctx.run.unit(name, tier=TIER, stage='verify', lst=lst) as u:
        u.set_admitted(exclusions().admitted(lst, 'Full', _nights(root, lst)))
        u.add_output(root, 'sublst_residual_set', x2_root=x2, lst=lst)
        for b in bands:
            u.add_output(os.path.join(tree, f'{lst}_{b}', 'frames.csv'), 'tenmin_tree', lst=lst, band=b)
            nw = os.path.join(tree, f'{lst}_{b}', 'NO_WINDOWS.txt')
            if os.path.exists(nw):
                u.add_output(nw, 'tenmin_tree', lst=lst, band=b)
    report(u)
