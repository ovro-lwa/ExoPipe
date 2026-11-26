"""exopipe3 - the ONE entry point of the OVRO-LWA ExoPipe Phase 3 pipeline.

    exopipe3 --help                 the steps, in pipeline order
    exopipe3 <step> --help          what the step reads and writes, its failure modes, every flag
    exopipe3 config [--section S]   every tunable (exopipe/phase3/config/: shared.py + steps/<step>.py)

Every step is independently runnable by a person or by Claude, documents its inputs, outputs and failure modes in
--help, refuses (exit 2) instead of warning, has --dry-run, and writes a machine-readable record of what it did under
<out>/_records/<step>/. Exit codes: 0 done (every content test passed), 1 error, 2 refused by a gate, 3 nothing to do.
Every writer step that is not a dry run passes the space gate first (core/space.py, config section space): on Lustre the run
user's quota headroom (bytes and files) and the file system's free space, elsewhere statvfs, for --out and --work; it reads
again before a unit every space.RECHECK_S seconds and, once low or unreadable, refuses that unit (its refused record is
written, cell meta included) and stops the step. Exempt (SPACE_GATE = False): retention (its writes free room), notify,
status and serve (small writes, needed most when room is low); a step's space_gate(args, out) picks roots and floors per
invocation (archive: push and sync only, on the archive root too; photometry: a campaign with its own floors). See
space_gate_of.
While a step runs, SIGTERM interrupts it as Ctrl-C does (Terminated, a KeyboardInterrupt: every rollback runs, the unit is
recorded 'interrupted'); a second SIGTERM terminates at once. A process forked from the step (a pool worker) takes SIGTERM's
default action, and the step's descendants are ended before its rollbacks run (_sigterm_interrupts).
"""
import contextlib
import os
import sys

from exopipe.phase3.core import env as _env

_env.apply()            # before anything imports numpy (core/env.py)

import argparse        # noqa: E402
import importlib       # noqa: E402
import threading      # noqa: E402

# (name, one-line summary, group) in pipeline order. The module is exopipe.phase3.steps.<name>.
STEPS = [
    # per hour: one LST hour of one night, in near real time, against the installed masters (DESIGN 12.1)
    ('frames', 'per-epoch / per-window / per-frame residuals and search frames, pbcorr siblings, noise maps', 'per hour'),
    ('flow', 'alignment flows: snapshot frames and sub-LST windows (per hour); per-night co-add flows (masters builds)', 'per hour'),
    ('search', 'detection + gates with fitted positions (deep, sub-LST, snapshot)', 'per hour'),
    ('catalog', 'merge, conformance, association, identity, selection, names (--incremental adds one hour); snapshot chain', 'per hour'),
    ('forced', 'forced photometry (Gaussian fit >= 7 sigma, empirical-beam template below); event panels; the wide-area burst '
               'test and the night-activity table', 'per hour'),
    ('curves', 'light-curve products', 'per hour'),
    ('pages', 'review pages (served queues, inspection twins, multiples), incremental or full', 'per hour'),
    ('serve', 'review servers and the hub; served-queue and exclusion content tests; exact-pid restart', 'per hour'),
    ('archive', 'manifest, checksum transfer, content audit, quarantine; retire excluded nights; rollback', 'per hour'),
    ('notify', "the hour's one email: the Phase 2 report plus the Phase 3 highlights (target detections, top transients)", 'per hour'),
    # survey (intermittent): many hours on disk (DESIGN 12.2)
    ('masters', 'co-add masters + LOO + QA gates; V clean; flux scale; PB maps; source catalogue (all tiers)', 'survey'),
    ('dewarp', 'master dewarp per LST: RBF screen on the rebuilt masters, cubic apply, content test D1-D6', 'survey'),
    ('install', 'gated install of rebuilt products into the working tree; quarantine round; re-run list', 'survey'),
    ('calibrate', 'versioned reference tables: derive, diff, install, verify', 'survey'),
    ('hybrid', 'hybrid NoTaper route: stacks, large-scale reconstruction, wideband, hybrid masters, Full, per-tile zero-spacing '
               'fill', 'survey'),
    ('mosaic', 'mosaicked products', 'survey'),
    ('photometry', 'photometry of input samples (Hot/Warm Jupiters, flare/UCD, 10/25/50 pc)', 'survey'),
    ('followup', 'follow-up of high-interest candidates (honing, dynamic spectra, periodicity, folding, dedispersion)', 'survey'),
    ('jupiter', 'the Jupiter reflection program', 'survey'),
    # tools
    ('status', 'completion, exclusion conformance, stale products, coverage, rebuild due; emits item lists', 'tools'),
    ('retention', 'raw night data kept 2 weeks: retention table, extraction holds, expiry plan (dry run), the ruled '
                  'delete', 'tools'),
    ('verify', 'run the content test of any product class alone; compare two product roots (parity)', 'tools'),
    ('run', 'multi-node runner: jobs, chains (e.g. the per-hour chain), --after, --code, hold/release/drain/retry', 'tools'),
    ('drive', 'per-hour driver: each hour\'s chain as Slurm jobs over the stage records; catch-up windows; periodic jobs', 'tools'),
]

# Selectors a step may declare in SELECTORS; cli adds the flags.
_SELECTOR_FLAGS = {
    'tier': (['--tier'], dict(help='tier: deep, sublst or snapshot (the step lists the tiers it supports)')),
    'cells': (['--cells'], dict(help='cells: 04h_Full, 04h_*, @file, record:STATUS:STEP@OUT, JOB:STATUS')),
    'lsts': (['--lsts'], dict(help='LSTs: 04h, 04h,05h, @file')),
    'nights': (['--nights'], dict(help='nights YYYY-MM-DD: list, @file')),
    'scopes': (['--scopes'], dict(help='subbands/bands: 18MHz, Red, Full, ...')),
    'stokes': (['--stokes'], dict(help='Stokes: I, V or I,V')),
    'exclude_cells': (['--exclude-cells'], dict(help='cells to leave out (same grammar as --cells)')),
    'run_label': (['--run'], dict(dest='run_label', help='run label of the products to read/write (e.g. P3S5)')),
    'epochs': (['--epochs'], dict(help='per-hour items <LST>_<date> (all bands and tiers of that hour): 04h_2025-10-14, @file')),
}


def _common(p, step_mod):
    g = p.add_argument_group('common to every step')
    g.add_argument('--out', help='product root for this run (default: the step\'s production root from config; '
                                 'production roots refuse writes while paths.PRODUCTION_WRITES is False)')
    g.add_argument('--work', help='node-local scratch root (default from the site file)')
    g.add_argument('--set', action='append', default=[], metavar='SECTION.KEY=VALUE',
                   help='override one tunable for this run (recorded); repeatable. One dict entry: '
                        'SECTION.KEY[entry]=VALUE (tuple keys comma-separated: fitting.K_TEMPLATE[deep,I]=(0.5,0.5)). '
                        'Protected keys refuse; a ruled key needs --ruling-for or a non-production --out')
    g.add_argument('--config', metavar='FILE', help='python file of section.KEY = value overrides (recorded)')
    g.add_argument('--ruling-for', action='append', default=[], metavar='SECTION.KEY=LEDGER_ID',
                   help="authorise overriding a RULED key with Gregg's ruling: the ledger ID must be a row of "
                        "paths.FIX_LEDGER (recorded with the override); repeatable")
    g.add_argument('--dry-run', action='store_true', help='plan only: inputs found, outputs, gates; writes nothing')
    g.add_argument('--parity', action='store_true',
                   help='parity run: refuse (PermissionError) every write outside --out, --work, /tmp and /dev/null, in '
                        'this process and in every Python subprocess it starts (core/guard.py); needs --out')
    g.add_argument('--injected', action='store_true',
                   help='an INJECTED run: its data carry synthetic sources (a completeness battery, or a step run on an '
                        'injected run\'s products). It writes no production product (core/injected.py): --out is required '
                        'and must be scratch, paths.PRODUCTION_WRITES is False, every write outside --out, --work and scratch '
                        'refuses in this process and its children, and its records say so. Implied by search --inject; '
                        'inherited from an injected parent (EXOPIPE3_INJECTED); `run submit --injected` forwards it')
    g.add_argument('--resume', action='store_true',
                   help='skip units whose record is done and whose inputs, admitted nights, code, config are '
                        'unchanged and whose outputs pass their content tests now')
    g.add_argument('--fresh', action='store_true', help='ignore earlier records and outputs of the selected units')
    g.add_argument('--workers', type=int, default=None, help='parallel units in this process (default from config)')
    g.add_argument('--cores', default='auto', help="CPU cores: 'auto' = recorder-free cores of this node "
                                                   "(node_free_cores.sh), 'none' = no pinning, or a list 0-23,30-39")
    g.add_argument('--threads', type=int, default=None, help='BLAS/OpenMP threads per process')
    g.add_argument('--waive', action='append', default=[], metavar='TEST',
                   help='waive one named content test or gate; requires --ruling')
    g.add_argument('--ruling', action='append', default=[], metavar='TEXT',
                   help="Gregg's words, UTC and ledger ID for each --waive (same order)")
    for sel in getattr(step_mod, 'SELECTORS', ()):
        flags, kw = _SELECTOR_FLAGS[sel]
        kw = dict(kw)
        if sel == 'tier' and getattr(step_mod, 'TIERS', None):
            kw['choices'] = list(step_mod.TIERS)
            kw['default'] = step_mod.TIERS[0]
            kw['help'] = f"tier ({', '.join(step_mod.TIERS)}; default {step_mod.TIERS[0]})"
        g.add_argument(*flags, **kw)


def _pin(cores):
    if cores in (None, 'none'):
        return None
    if cores == 'auto':
        import subprocess
        from exopipe.phase3.config import CFG
        try:
            cores = subprocess.run(CFG.runner.FREE_CORES_CMD, shell=True, capture_output=True, text=True,
                                   timeout=30).stdout.strip()
        except Exception:
            cores = ''
        if not cores:
            raise SystemExit('refused: could not read the recorder-free cores (runner.FREE_CORES_CMD); '
                             'pass --cores LIST or --cores none')
    from exopipe.phase3.core.env import parse_cores
    os.sched_setaffinity(0, parse_cores(cores))
    return cores


def _guard_out(out, key=None):
    """The refusal text when --out is a production root during the refactor window, else None (exit 2, not 1). ``key``: the
    step's name in the narrow cutover allowance (paths.PRODUCTION_WRITES_ALLOWED: 'hybrid', 'mosaic:v5', ...)."""
    from exopipe.phase3.config import CFG
    if CFG.writes_allowed(key, out):
        return None
    root = CFG.production_root_of(out)
    if root is not None:
        return (f'--out {out} is under the production root {root}, paths.PRODUCTION_WRITES is False (refactor window) and '
                f'paths.PRODUCTION_WRITES_ALLOWED does not open it to {key!r}. Write parity/scratch output to your own root.')
    return None


def _allow_key(name, args):
    """The step's name in paths.PRODUCTION_WRITES_ALLOWED: 'mosaic:<generation>' for the mosaic step, 'archive:<class>' for
    `archive push --class <class>` (its records beside the archive, as the push's own G2 key), else the step name."""
    if name == 'mosaic':
        return f'mosaic:{getattr(args, "generation", None)}'
    if name == 'archive' and getattr(args, 'action', None) == 'push' and getattr(args, 'push_class', None):
        return f'archive:{args.push_class}'
    return name


def _parity(args, out):
    """Install the parity write guard for this process and its Python subprocesses (DESIGN 11.1)."""
    import importlib as _il
    guard = _il.import_module('exopipe.phase3.core.guard')
    roots = [out] + ([args.work] if args.work else [])
    if not args.work:            # node-local scratch (core/fitsio.scratch_dir) is temporary, never a product
        from exopipe.phase3.config import CFG
        from exopipe.phase3.core import fitsio
        try:
            scratch = fitsio.scratch_root(None)
        except Exception:
            scratch = None
        if scratch and CFG.production_root_of(scratch) is None:
            roots.append(scratch)
    log = os.path.join(out, '_records', '_parity_guard_refusals.log')
    guard.install(roots, log=log)
    guard.export_env(roots, log=log)
    print(f'parity guard: writes allowed only under {", ".join(guard.allowed_roots())}', file=sys.stderr)


def _msg(e):
    return e.args[0] if isinstance(e, KeyError) and e.args else e


def _print_overrides(CFG):
    for o in CFG.overrides:
        print(f'override {o[0]}: {o[1]!r} -> {o[2]!r}' + (f'   [{o[4]}]' if o[4] else ''))


def _config_cmd(argv):
    p = argparse.ArgumentParser(prog='exopipe3 config', description='Show the Phase 3 tunables (exopipe/phase3/config/: shared.py, steps/<step>.py).')
    p.add_argument('--section', help='print one section with its comments (units, rulings), its file:line, L = locked and R = ruled keys')
    p.add_argument('--step', help='print every section the step declares in CONFIG (and name the shared sections)')
    p.add_argument('--set', action='append', default=[], help='show the effect of overrides (SECTION.KEY[entry]=V)')
    p.add_argument('--config', metavar='FILE', help='show the effect of a file of overrides')
    p.add_argument('--ruling-for', action='append', default=[], metavar='SECTION.KEY=LEDGER_ID',
                   help='check a ruling ID for a ruled key (as a step would)')
    p.add_argument('--hash', action='store_true', help='print the config hash')
    a = p.parse_args(argv)
    from exopipe.phase3.config import CFG, SITE_FILE, ConfigRefusal
    try:
        CFG.authorize(a.ruling_for, mode='display')
        if a.config:
            CFG.apply_override_file(a.config)
        CFG.apply_override_strings(a.set)
    except ConfigRefusal as e:
        print(f'REFUSED: {e}', file=sys.stderr)
        return 2
    except (KeyError, ValueError) as e:
        print(f'error: {_msg(e)}', file=sys.stderr)
        return 1
    if a.step:
        return _describe_step(a.step, CFG, a.hash)
    if a.hash:
        print(CFG.hash())
        return 0
    print(CFG.describe(a.section))
    if not a.section:
        print(f'\nsite file: {SITE_FILE or "(none: EXOPIPE3_SITE unset)"}')
    _print_overrides(CFG)
    return 0


def _describe_step(name, CFG, hash_only=False):
    from exopipe.phase3 import config
    _env.legacy()
    try:
        mod = importlib.import_module(f'exopipe.phase3.steps.{name}')
    except ImportError as e:
        print(f'error: no step module exopipe.phase3.steps.{name} ({e})', file=sys.stderr)
        return 1
    try:
        decl = config.declared_sections(mod)
    except KeyError as e:
        print(f'error: {_msg(e)}', file=sys.stderr)
        return 1
    if hash_only:
        print(CFG.hash(config.step_sections(mod)))
        return 0
    if decl is None:
        print(f'step {name}: CONFIG not declared - add CONFIG = (sections it reads) to exopipe/phase3/steps/{name}.py. '
              f'Until then its resume hash covers every section (exopipe3 config lists them).')
        return 0
    shared = [s for s in config.SHARED_SECTIONS if s not in decl]
    print(f'# step {name} declares CONFIG = {decl!r}')
    print(f'# shared sections (always in its resume hash): {", ".join(config.SHARED_SECTIONS)}')
    for sec in decl:
        print(f'\n# ---- section {sec} ' + '-' * 60)
        print(CFG.describe(sec))
    if shared:
        print(f'# not declared but shared (read with exopipe3 config --section S): {", ".join(shared)}')
    print(f'# step config hash: {CFG.hash(config.step_sections(mod))}')
    _print_overrides(CFG)
    return 0


class Terminated(KeyboardInterrupt):
    """The step process received SIGTERM (an operator's kill, a node shutdown, a scheduler's stop). Raised in its main
    thread while the step runs (_sigterm_interrupts), so everything written for Ctrl-C runs for it too: a placement in
    progress is undone (lib/photometry/steps_cutouts.placement), the unit is recorded 'interrupted' (core/record), the run
    summary is written - and the process then ends as after Ctrl-C (fix verification N2, 2026-11-03)."""


_FORK_RESET = set()           # handlers of _sigterm_interrupts now installed: the at-fork hooks and wrappers act only while one is
_FORK_HOOKS = []              # [True] once the at-fork hooks are registered (once per process)
_FORK_TLS = threading.local()  # per thread: .mask, the mask the before hook saved for the after hooks (None: it did nothing);
#                                .in_fork, inside a wrapped fork call (_forked); .deferred, a SIGTERM the handler deferred there
_REAL_FORKS = {}              # os.fork, os.forkpty and _posixsubprocess.fork_exec as they were before the wrappers (_forked)
_CHILD_TERM_S = 5.0           # s: an interrupted step's descendants get SIGTERM for this long (_end_descendants)
_CHILD_KILL_S = 5.0           # s: then SIGKILL, and this long to end, before the ones still there are named
_CHILD_POLL_S = 0.1           # s: between two looks at them (only the step's descendants are read)
_PR_SET_CHILD_SUBREAPER = 36  # prctl(2) options (linux/prctl.h)
_PR_GET_CHILD_SUBREAPER = 37


def _block_sigterm_for_fork():
    """os.register_at_fork(before=...), in the forking thread. While a handler of _sigterm_interrupts is installed: save the
    thread's signal mask (thread-local, for the after hooks) and block SIGTERM in the thread through the fork, so the child
    starts with SIGTERM blocked (_reset_inherited_sigterm). Otherwise nothing."""
    if not _FORK_RESET:
        return
    import signal
    _FORK_TLS.mask = signal.pthread_sigmask(signal.SIG_BLOCK, ())      # the mask as it is (no change)
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})


def _restore_mask_after_fork():
    """os.register_at_fork(after_in_parent=...), in the forking thread: the mask the before hook saved is set again
    (SIG_SETMASK: a thread that had SIGTERM blocked before the fork still has it blocked). Nothing when that hook did nothing."""
    mask = getattr(_FORK_TLS, 'mask', None)
    if mask is None:
        return
    _FORK_TLS.mask = None
    import signal
    signal.pthread_sigmask(signal.SIG_SETMASK, mask)


def _sigterm_reset_in_child(mask):
    """In a child forked while a handler of _sigterm_interrupts is installed: the CLI's handler gives way to SIGTERM's default
    disposition, then ``mask`` (the forking thread's, saved before the fork) is set again (SIG_SETMASK). A foreign Python
    handler (one the step's code installed over the CLI's) gives way to the default disposition while the mask is set and is
    set again after it (_reset_inherited_sigterm)."""
    import signal
    handler = signal.getsignal(signal.SIGTERM)
    if any(handler is h for h in _FORK_RESET):
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
    elif callable(handler):
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        signal.signal(signal.SIGTERM, handler)
    else:
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)


def _reset_inherited_sigterm():
    """os.register_at_fork(after_in_child=...), in a child forked while _sigterm_interrupts' handler is installed (nothing
    otherwise). The CLI's handler gives way to SIGTERM's default disposition, so the kernel ends the child at once even inside
    a long C call (numpy, I/O), where a Python handler would only run when the call returns. Then the forking thread's mask is
    set again (SIG_SETMASK): a SIGTERM sent to the child since the fork - blocked until here (_block_sigterm_for_fork) - ends
    it here by the default action, unless that thread had SIGTERM blocked before the fork (then it stays blocked and pending).
    A foreign Python handler (one the step's code installed over the CLI's) is set again after the mask, for the SIGTERMs that
    come later; a SIGTERM pending here still ends the child by the default action. Run here, the foreign handler would raise
    inside an at-fork hook, where Python drops the exception (sys.exit, Terminated), and the child would run on. (A SIGTERM
    that arrives after the foreign handler is set again and before the at-fork hooks have returned still runs it there.)"""
    mask = getattr(_FORK_TLS, 'mask', None)
    if mask is None:
        return
    _FORK_TLS.mask = None
    _sigterm_reset_in_child(mask)


def _register_fork_hooks():
    """Once per process, the at-fork hooks of _sigterm_interrupts. logging is imported first, so its hooks run inside the
    CLI's: Python runs before hooks in the reverse order of registration and after hooks in the order."""
    import logging  # noqa: F401 - its at-fork hooks registered before the CLI's (docstring)
    os.register_at_fork(before=_block_sigterm_for_fork, after_in_parent=_restore_mask_after_fork,
                        after_in_child=_reset_inherited_sigterm)
    _FORK_HOOKS.append(True)


def _forked(real, args, block):
    """``real(*args)``, a fork of the step's process (os.fork, os.forkpty, subprocess's _posixsubprocess.fork_exec), made
    while a handler of _sigterm_interrupts is installed (_sigterm_interrupts, "The fork window"). The calling thread is
    marked as forking (thread-local) through the call: a SIGTERM the handler takes in it meanwhile (inside an at-fork hook,
    whoever registered it) is deferred, and raised here once ``real`` has returned, in this ordinary code, so Terminated
    reaches the caller. With ``block``, SIGTERM is also blocked in the thread through the call: in the child the CLI's handler
    then gives way to the default disposition and the thread's mask is set again; in the parent the mask is set again, and a
    SIGTERM that came meanwhile is taken there and raised (Terminated). fork_exec is not blocked: without preexec_fn it runs no
    Python code, and the program it execs would start with SIGTERM blocked; with preexec_fn the CLI's at-fork hooks block it
    (_block_sigterm_for_fork)."""
    import signal
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, ()) if block else None     # the mask as it is (no change)
    _FORK_TLS.in_fork, _FORK_TLS.deferred = True, False
    child = False
    try:
        if block:
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        r = real(*args)
        child = block and (r if isinstance(r, int) else r[0]) == 0
    finally:
        _FORK_TLS.in_fork = False
        if child:
            _FORK_TLS.deferred = False                  # the parent's
            _sigterm_reset_in_child(mask)
        else:
            if mask is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, mask)    # a SIGTERM blocked through the fork: raised in here
            _raise_deferred()
    return r


def _raise_deferred():
    """A SIGTERM the handler deferred in this thread's wrapped fork call (_forked): the handler now takes it (Terminated)."""
    if not getattr(_FORK_TLS, 'deferred', False):
        return
    _FORK_TLS.deferred = False
    import signal
    handler = signal.getsignal(signal.SIGTERM)
    if any(handler is h for h in _FORK_RESET):
        handler(signal.SIGTERM, None)


def _fork_in_step():
    """os.fork while a handler of _sigterm_interrupts is installed (_forked)."""
    return _forked(_REAL_FORKS['fork'], (), True)


def _forkpty_in_step():
    """os.forkpty while a handler of _sigterm_interrupts is installed (_forked)."""
    return _forked(_REAL_FORKS['forkpty'], (), True)


def _fork_exec_in_step(*args):
    """_posixsubprocess.fork_exec (subprocess's start of a child) while a handler of _sigterm_interrupts is installed
    (_forked; SIGTERM is not blocked)."""
    return _forked(_REAL_FORKS['fork_exec'], args, False)


def _swap_forks(new):
    """Sets os.fork, os.forkpty and _posixsubprocess.fork_exec to ``new`` (a triple; None: the wrappers) and returns the
    triple they were. The wrappers are installed over the real functions only (_REAL_FORKS keeps those)."""
    import _posixsubprocess
    was = (os.fork, os.forkpty, _posixsubprocess.fork_exec)
    if new is None:
        if was[0] is not _fork_in_step:
            _REAL_FORKS.update(fork=was[0], forkpty=was[1], fork_exec=was[2])
        new = (_fork_in_step, _forkpty_in_step, _fork_exec_in_step)
    os.fork, os.forkpty, _posixsubprocess.fork_exec = new
    return was


def _child_subreaper(on):
    """prctl(PR_SET_CHILD_SUBREAPER, on) for this process (Linux >= 3.4): a process orphaned below it is re-parented to it,
    not to init. Returns the setting it had; OSError when the kernel refuses."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong)
    prctl.restype = ctypes.c_int
    was = ctypes.c_int(0)
    if prctl(_PR_GET_CHILD_SUBREAPER, ctypes.addressof(was), 0, 0, 0) != 0 \
            or prctl(_PR_SET_CHILD_SUBREAPER, 1 if on else 0, 0, 0, 0) != 0:
        e = ctypes.get_errno()
        raise OSError(e, 'prctl(PR_SET_CHILD_SUBREAPER): ' + os.strerror(e))
    return was.value


def _proc_stat(pid):
    """(state, parent pid, session, start time in clock ticks after boot - stat field 22 -, command name) of process ``pid``
    from /proc/<pid>/stat, or None when it is gone."""
    try:
        with open('/proc/%d/stat' % pid, 'rb') as f:
            st = f.read()
    except OSError:
        return None
    j = st.rindex(b')')
    x = st[j + 2:].split()                      # state ppid pgrp session ... (the name before may hold spaces)
    return x[0].decode(), int(x[1]), int(x[3]), int(x[19]), st[st.index(b'(') + 1:j].decode('utf-8', 'replace')


def _exe(pid):
    """The executable process ``pid`` runs (/proc/<pid>/exe; it changes when the process execs), or None."""
    try:
        return os.readlink('/proc/%d/exe' % pid)
    except OSError:
        return None


def _descendants(root, sid):
    """{pid: _proc_stat(pid)} of the descendants of process ``root`` in session ``sid``, found by descent only: the children
    lists of root's threads (/proc/<root>/task/<tid>/children) and, in turn, of each descendant found; no other process is
    read. A descendant in another session (it called setsid: start_new_session, a daemon) is left out with its whole tree. A
    descendant in a process group of its own in session ``sid`` is a descendant like any other."""
    out, todo = {}, [root]
    while todo:
        p = todo.pop()
        try:
            tids = os.listdir('/proc/%d/task' % p)
        except OSError:
            continue
        for t in tids:
            try:
                with open('/proc/%d/task/%s/children' % (p, t), 'rb') as f:
                    kids = f.read().split()
            except OSError:
                continue
            for k in map(int, kids):
                if k in out or k == root:
                    continue
                st = _proc_stat(k)
                if st is not None and st[2] == sid:
                    out[k] = st
                    todo.append(k)
    return out


def _pool_workers():
    """The pids of this process's multiprocessing Pool workers: the processes multiprocessing records as started here
    (multiprocessing.process._children) that multiprocessing.pool named as its workers (ForkPoolWorker-N and the like; a
    started Process no longer holds its target, bpo-30775)."""
    proc = sys.modules.get('multiprocessing.process')
    if proc is None or 'multiprocessing.pool' not in sys.modules:
        return set()
    return {p.pid for p in list(getattr(proc, '_children', ())) if 'PoolWorker-' in (p.name or '') and p.pid}


def _how_ended(pid):
    """How the step's own child ``pid`` (ended, not yet reaped) ended, read without reaping it (WNOWAIT)."""
    try:
        i = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        return 'ended'
    if i is None:
        return 'ended'
    return ('exit %d' if i.si_code == os.CLD_EXITED else 'signal %d') % i.si_status


def _signal_checked(name, pid, sig, rec, sid, termed, killed):
    """Sends ``sig`` to ``pid`` only if it is still the process the look recorded (``rec``: [start time, name, executable]):
    its start time (stat field 22) is read again and must be the recorded one (a pid freed and taken again has another), and
    it must still be in the step's session ``sid``. A changed name or executable with the same start time is an exec of that
    process since the look: accepted, recorded in ``rec`` and said. SIGTERM goes once per program (``termed``: pid -> the
    (name, executable) it went to; again after an exec, which a SIGTERM caught before it may not survive), SIGKILL once
    (``killed``). Each signal sent and each one withheld is a line on stderr: pid, name, and the check."""
    import signal
    st = _proc_stat(pid)
    if st is None or st[0] in 'ZX':
        return                                          # ended since the look: nothing to send
    sname = signal.Signals(sig).name
    if st[3] != rec[0] or st[2] != sid:
        print(f'SIGTERM: exopipe3 {name}: {sname} not sent to {pid} ({st[4]}): start time {st[3]} ticks after boot in session '
              f'{st[2]}, at the look {rec[0]} in session {sid}: not the process found', file=sys.stderr, flush=True)
        return
    exe = _exe(pid)
    prog = (st[4], exe)
    if (termed.get(pid) == prog) if sig == signal.SIGTERM else (pid in killed):
        return
    note = '' if prog == (rec[1], rec[2]) else f'; exec since the look (was {rec[1]}, {rec[2]})'
    rec[1], rec[2] = prog
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return
    if sig == signal.SIGTERM:
        termed[pid] = prog
    else:
        killed.add(pid)
    print(f'SIGTERM: exopipe3 {name}: {sname} to {pid} ({st[4]}, {exe}): start time {st[3]} ticks after boot, as at the '
          f'look{note}', file=sys.stderr, flush=True)


def _end_descendants(name, when):
    """Ends the descendants of the interrupted step's process (_sigterm_interrupts, "Descendants of an interrupted step"):
    they are found by descent (_descendants; every _CHILD_POLL_S), each gets SIGTERM (again after an exec) for _CHILD_TERM_S,
    then SIGKILL, and has _CHILD_KILL_S more to end; every signal is checked first (_signal_checked). The step's own Pool
    workers are left to their pool; their descendants are not. The step's own children are not reaped here (how they ended is
    read with WNOWAIT): their pids cannot be taken by another process before their last signal, and they stay for the code
    that started them (a Popen) to reap. One stderr line names every descendant found alive and how it ended (``when``:
    before or after the rollbacks; one already ended when first found, such as an orphan's zombie the step has not reaped,
    is not named), the pool workers left, and any process still there after SIGKILL. Returns the pids of those."""
    import signal
    import time
    me, sid = os.getpid(), os.getsid(0)
    seen, how, termed, killed, pool_left, alive = {}, {}, {}, set(), set(), set()
    t_kill = time.monotonic() + _CHILD_TERM_S
    t_end = t_kill + _CHILD_KILL_S
    while True:
        pool = _pool_workers()
        cur = _descendants(me, sid)
        live = []
        for pid, st in cur.items():
            if pid in pool:
                pool_left.add(pid)
                continue
            if pid not in seen or seen[pid][0] != st[3]:        # first found, or its pid now another descendant's
                seen[pid] = [st[3], st[4], _exe(pid)]
                how.pop(pid, None)
                termed.pop(pid, None)
                killed.discard(pid)
            if st[0] in 'ZX':
                if pid not in how:
                    how[pid] = _how_ended(pid) if st[1] == me else 'ended'
            else:
                live.append(pid)
                alive.add(pid)
        for pid in seen:
            if pid not in cur and pid not in how:
                st = _proc_stat(pid)
                how[pid] = 'left the session' if st is not None and st[3] == seen[pid][0] else 'ended'
        now = time.monotonic()
        if not live or now >= t_end:
            break
        for pid in live:
            _signal_checked(name, pid, signal.SIGTERM if now < t_kill else signal.SIGKILL, seen[pid], sid, termed, killed)
        time.sleep(_CHILD_POLL_S)
    if alive or pool_left:
        print(f"SIGTERM: exopipe3 {name}: the step's descendants, {when}: "
              + (', '.join(f'{pid} ({seen[pid][1]}: {how.get(pid, "still there")})' for pid in sorted(alive)) or 'none')
              + (f'; Pool workers left to their pool: {", ".join(map(str, sorted(pool_left)))}' if pool_left else '')
              + (f'; still there {_CHILD_KILL_S:g} s after SIGKILL: {", ".join(map(str, sorted(live)))}' if live else ''),
              file=sys.stderr, flush=True)
    return live


@contextlib.contextmanager
def _sigterm_interrupts(name):
    """While the step ``name`` runs: SIGTERM raises Terminated in the main thread, once - the first SIGTERM restores the
    default action, so a second one terminates at once (as before). Installed only over the default disposition, in the main
    thread, and never for `exopipe3 run` (its supervisor and worker loops keep the default; each step they launch is a
    process of its own and gets it); the previous disposition is restored when the step returns. In this order, a SIGTERM:
    the handler ends the step's descendants, then raises Terminated, so every rollback runs after they have ended, and the
    unit is then recorded 'interrupted'.

    Only the step's own process (the one that installed the handler) is interrupted. A process forked from it - every pool
    worker of the step and of the in-place modules it calls, which v2 cannot give an initializer; Pool.terminate(), i.e. every
    `with Pool(...)` exit, ends its workers with SIGTERM - gets SIGTERM's default disposition at the fork
    (_reset_inherited_sigterm). So a forked worker ends at once, without Python clean-up, as every pool worker did before
    2026-11-03: clean-up and rollback belong to the step's own process (a placement is undone there); a child that needs its
    own (deep_search's extractor fork) installs its handler after the fork. Before 2026-11-17 the inherited handler raised
    Terminated in the child: the photometry parity run P3 wrote a second run record from a forked worker that had unwound into
    the step's own code, and a reproduction (fork pools with maxtasksperchild=1) raised Terminated in a child before
    multiprocessing's own try block, whose pool then signaled the whole process group (os.kill(0, ...)), interrupted the step
    process itself and left it hanging at exit (tests/test_cli_sigterm_children.py).

    The fork window. While the handler is installed, os.fork and os.forkpty are wrapped (_fork_in_step, _forked): SIGTERM is
    blocked in the forking thread from before the at-fork hooks to after them. In the child the CLI's handler then gives way
    to the default disposition (_reset_inherited_sigterm, at-fork, and again in the wrapper) before the thread's mask is set
    again. Without the block the child holds the handler's C-level signal catcher from the fork until that reset, and CPython
    drops a SIGTERM caught there: one caught before the child clears its pending-signal flags (PyOS_AfterFork_Child,
    issue9535) is lost silently, and one caught inside the reset's own signal.signal() call, after that call's check for
    pending signals, is lost with "Signal 15 ignored due to race condition" (bpo-43406). Pool.terminate() signals workers that
    may have been forked a moment before (a maxtasksperchild replacement, a pool exited right after it was made); such a
    worker lived on, waiting for a task, and the pool's join of it never returned (tests/test_cli_sigterm_children.py hung on
    lwacalim07, 2026-11-26). In the parent the wrapper sets the mask again once the fork has returned; a SIGTERM that came
    meanwhile is taken there and raises Terminated in the wrapper, which is ordinary code, so it reaches the caller (the
    step's own os.fork(), multiprocessing's start of a worker). Python runs the at-fork hooks in the forking thread and drops
    an exception raised in one, so the handler must not raise there, whichever hooks there are and in whatever order they
    were registered. Blocking alone does not ensure that: it is per thread, and in a step with other threads a SIGTERM to the
    process is caught by one of them, after which Python runs the handler in the main thread at its next instruction, which
    may be inside an at-fork hook. So the wrapper also marks the forking thread (thread-local), and the handler, taken while
    the main thread is inside a wrapped fork call, only defers the SIGTERM; the wrapper raises it once the fork has returned
    (_raise_deferred). subprocess's fork (_posixsubprocess.fork_exec) is wrapped the same way, marked but not blocked: without
    preexec_fn it runs no Python code and the program it execs would start with SIGTERM blocked; with preexec_fn its at-fork
    hooks block SIGTERM (_block_sigterm_for_fork; the child is reset by _reset_inherited_sigterm). A fork made otherwise (by a
    C extension, or through a reference to os.fork taken before the step started) still blocks SIGTERM through the CLI's
    at-fork hooks, but a SIGTERM the handler takes inside an at-fork hook there raises in the hook, and Python drops it: the
    step's descendants have been ended, and the step runs on with SIGTERM's default action (a second SIGTERM ends it).
    (tests/test_cli_sigterm_fork_window.py)

    Descendants of an interrupted step. While the handler is installed the step's process is a child subreaper
    (prctl PR_SET_CHILD_SUBREAPER; set back when the step returns), so a process orphaned below it - a script's child left
    running when subprocess.run, interrupted, killed the script with SIGKILL; a double-forked helper - is re-parented to the
    step, not to init, however the step was started (the driver, an operator's shell). (Such an orphan that ends while the
    step runs stays a zombie until the step exits.) When the handler takes SIGTERM it ends the step's descendants before it
    raises Terminated (_end_descendants): SIGTERM to each (again after an exec: a subprocess child forked in C holds the
    handler's catcher until it execs, and a SIGTERM caught there is dropped), SIGKILL after _CHILD_TERM_S. They are found by
    descent only, every _CHILD_POLL_S (the children lists of the step's threads and, in turn, of each descendant; no other
    process is read), so a process the step did not start (a wrapper's monitor, the shell's log tee) is never signaled. A
    descendant that called setsid (start_new_session: a detached server) is left alone with its tree; a descendant in a
    process group of its own in the step's session is ended. Before each signal the target's start time (stat field 22) is
    read again and must match the look's; a changed name or executable is accepted as an exec of the same process; each
    signal is logged with pid, name and that check (_signal_checked). pidfd_open, which would close the moment between that
    check and kill(2), is not in this kernel (syscall 434: ENOSYS on 4.18.0-477, lwacalim07, 2026-11-26). The step's own
    children are not reaped until their last signal is sent (here not at all), so their pids cannot be taken by another
    process meanwhile. The step's multiprocessing Pool workers are left to their pool: Pool.terminate() (every
    `with Pool(...)` exit; multiprocessing's exit handler for a pool never terminated) takes the pool's task-queue lock before
    it signals them, and a worker ended from outside while it waited for a task holds that lock for good, so terminate()
    never returns (an idle fork pool, lwacalim07, 2026-11-26). Their own descendants are ended. After the rollbacks a second
    pass ends what is left (a descendant a rollback started; a pool worker's child orphaned when the pool ended the worker).
    A child the handler itself runs in (one forked by C code that then runs Python without the at-fork hooks) ends by the
    default action."""
    import signal
    if name == 'run' or threading.current_thread() is not threading.main_thread() \
            or signal.getsignal(signal.SIGTERM) != signal.SIG_DFL:
        yield
        return
    owner = os.getpid()
    fired = []                  # Terminated raised: the step was interrupted

    def _on_sigterm(signum, frame):
        if os.getpid() != owner:                    # a child that runs Python without the at-fork hooks (docstring)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGTERM)
            os._exit(128 + signal.SIGTERM)          # should the signal be blocked in this thread: end all the same
        if getattr(_FORK_TLS, 'in_fork', False):    # inside a wrapped fork call: raised once it has returned (_forked)
            _FORK_TLS.deferred = True
            return
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        if not fired:
            print(f'SIGTERM: exopipe3 {name} interrupted as for Ctrl-C (a placement in progress is undone); a second SIGTERM '
                  f'terminates at once', file=sys.stderr, flush=True)
        fired.append(True)
        try:
            _end_descendants(name, 'ended before the rollbacks')
        except Exception as e:                      # never in the way of the interruption itself
            print(f"SIGTERM: exopipe3 {name}: ending the step's descendants failed: {e!r}", file=sys.stderr, flush=True)
        raise Terminated('SIGTERM')

    if not _FORK_HOOKS:
        _register_fork_hooks()
    subreaper_was = _child_subreaper(True)
    _FORK_RESET.add(_on_sigterm)
    forks_were = _swap_forks(None)
    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        _swap_forks(forks_were)
        _FORK_RESET.discard(_on_sigterm)
        try:
            if fired:
                _end_descendants(name, 'still there after the rollbacks')
        finally:
            _child_subreaper(subreaper_was)


def build_parser():
    p = argparse.ArgumentParser(prog='exopipe3', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='step', metavar='<step>')
    group = None
    for name, summary, grp in STEPS:
        sub.add_parser(name, help=summary + ('' if grp == group else f'   [{grp}]'), add_help=False)
        group = grp
    sub.add_parser('config', help='show every tunable (exopipe/phase3/config/)', add_help=False)
    return p


def space_gate_of(mod, args, out):
    """(roots, floors section name) the space gate checks for this run of step ``mod``, or None (no gate): a step that
    writes nothing has none; SPACE_GATE = False exempts a step (its writes free room - retention - or are small and needed
    most when room is low - notify, status, serve); a module function space_gate(args, out) decides per invocation (archive:
    push and sync, with the archive root; photometry: a campaign's cutout file system with its own floors). Otherwise --out
    and --work with the floors of section space."""
    if not getattr(mod, 'WRITES', True) or getattr(mod, 'SPACE_GATE', True) is False:
        return None
    work = [args.work] if getattr(args, 'work', None) else []
    hook = getattr(mod, 'space_gate', None)
    if hook is not None:
        got = hook(args, out)
        if not got:
            return None
        roots, floors = got
        return [r for r in dict.fromkeys(list(roots) + work) if r], floors or 'space'
    return [out] + work, 'space'


def main(argv=None):
    # a chain step of `exopipe3 run` hands its outcome back on this descriptor (core/runner.py step_outcome); taken out of
    # the environment first, so no process this step starts inherits it
    outcome_fd = os.environ.pop('EXOPIPE3_STEP_OUTCOME_FD', None)
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help'):
        build_parser().print_help()
        return 0
    name, rest = argv[0], argv[1:]
    if name == 'config':
        return _config_cmd(rest)
    if name not in [s[0] for s in STEPS]:
        build_parser().print_help()
        print(f'\nunknown step {name!r}', file=sys.stderr)
        return 1
    _env.legacy()       # refactor window: legacy roots + old-name finder (core/env.legacy)
    # The step module is imported only to build its parser; heavy imports belong inside its run().
    mod = importlib.import_module(f'exopipe.phase3.steps.{name}')
    sp = argparse.ArgumentParser(prog=f'exopipe3 {name}', description=mod.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mod.add_arguments(sp)
    _common(sp, mod)
    tail = []
    if name == 'run' and '--' in rest:           # `exopipe3 run submit ... -- <step command>`
        i = rest.index('--')
        rest, tail = rest[:i], rest[i + 1:]
    args = sp.parse_args(rest)
    args.step_argv = tail
    if len(args.waive) != len(args.ruling):
        sp.error('every --waive needs its own --ruling "<Gregg\'s words, UTC, ledger ID>"')
    _env.apply(args.threads)

    from exopipe.phase3.config import CFG, ConfigRefusal, step_sections
    from exopipe.phase3.core import record
    try:
        sections = step_sections(mod)          # CONFIG + shared sections: the step's resume hash (DESIGN 11.4)
        # Ruled keys may be overridden with --ruling-for or into a non-production --out; `run submit` and `drive` forward
        # their overrides to every step process, which checks them against its own --out.
        CFG.authorize(args.ruling_for, out=args.out or mod.default_out(args),
                      mode='forward' if name in ('run', 'drive') else 'run')
        # parity replay of protected inputs (config.set): only under --parity into a non-production --out
        CFG.parity_replay = bool(getattr(args, 'parity', False) and name != 'run' and args.out
                                 and CFG.production_root_of(args.out) is None)
        if args.config:
            CFG.apply_override_file(args.config)
        CFG.apply_override_strings(args.set)
        # a step's own flags that set config keys (e.g. hybrid --clean-region; the step's config_overrides(args)): applied
        # here with --set's checks (protected and ruled keys), so a refusal exits 2 before any record is made and the
        # run's configuration carries the value from the start
        _step_set = getattr(mod, 'config_overrides', None)
        for _k, _v, _src in (_step_set(args) if _step_set else ()):
            CFG.set(_k, _v, source=_src)
        out = args.out or mod.default_out(args)
        # a VERSIONED RUN (lib/serve/versions.py) is written through its work directory, never through the version its pointer
        # names: resolved here, before every guard below (check_out, the injected-run guard, the flag checks, _guard_out), so
        # each checks the directory the step writes; its writer lock is taken after them (below)
        versioned_out = getattr(mod, 'WRITES', True) and name != 'run'
        if versioned_out:
            from exopipe.phase3.lib.serve import versions as _versions
            out_given = out
            try:
                out = _versions.writer_out(out, lock=False)
            except _versions.Refusal as e:
                print(f'REFUSED: {e}', file=sys.stderr)
                return record.EXIT_REFUSED
        CFG.check_out(out)
        from exopipe.phase3.core import fallback as _fb, wrap as _wr
        from exopipe.phase3.lib.catalog import deep_omission as _do
        # B-30 / B-32: a misspelt policy or wrap value refuses here; the deep omission's pending values too (Q fix
        # verification N3: an invalid --set catalog_omission.* value refuses before the run, never as a late error)
        _bad = _fb.check_values() + _wr.check_values() + _do.check_values()
        if _bad:
            raise ConfigRefusal('invalid value(s): ' + '; '.join(_bad))
    except ConfigRefusal as e:
        print(f'REFUSED: {e}', file=sys.stderr)
        return record.EXIT_REFUSED
    except (KeyError, ValueError) as e:
        print(f'error: {_msg(e)}', file=sys.stderr)
        return record.EXIT_ERROR
    pinned = _pin(args.cores)
    from exopipe.phase3.core import injected as _inj
    # `exopipe3 run` is never an injected run itself: `run submit --injected` forwards the flag to every step command it
    # launches, on every node (steps/run.py; an inherited flag alone is refused there)
    inj = '' if name == 'run' else _inj.requested(args)
    try:
        if inj:                                # an INJECTED run writes no production product (core/injected.py)
            try:
                default = mod.default_out(args)
            except Exception:                  # noqa: BLE001 - the --out given is still checked against every refused root
                default = None
            lockf = _versions.writer_lock_of(out) if versioned_out and not args.dry_run else None
            roots = _inj.begin(inj, out, explicit_out=bool(args.out), work=args.work, default_out=default,
                               writes=getattr(mod, 'WRITES', True), locks=[lockf] if lockf else [])
            print(f'{_inj.LABEL} ({inj}): writes allowed only under {", ".join(roots)}', file=sys.stderr)
        if name != 'run':                      # destination flags (injected) / an injected --out and sources (ordinary)
            _inj.check_flags(name, args, out, writes=getattr(mod, 'WRITES', True))
    except record.Refusal as e:
        print(f'REFUSED: {e}', file=sys.stderr)
        return record.EXIT_REFUSED
    if getattr(mod, 'WRITES', True) and not args.dry_run:
        why = _guard_out(out, _allow_key(name, args))
        if why:
            print(f'REFUSED: {why}', file=sys.stderr)
            return record.EXIT_REFUSED
    if versioned_out and not args.dry_run:
        # the writer lock (shared), kept until this process ends: a serve's snapshot or a conversion never reads a half-written
        # run; the output resolved again under it must be the one the guards checked (a conversion in between: refused)
        try:
            locked = _versions.writer_out(out_given, lock=True)
        except _versions.Refusal as e:
            print(f'REFUSED: {e}', file=sys.stderr)
            return record.EXIT_REFUSED
        if locked != out:
            print(f'REFUSED: --out {out_given} resolved to {out} for the guards and to {locked} under its writer lock (the run '
                  f'was converted meanwhile) - nothing written; run the step again', file=sys.stderr)
            return record.EXIT_REFUSED
    if args.parity and name != 'run' and not inj:   # `run submit --parity` forwards it to every step command instead;
        if not args.out:                            # an injected run's guard (above) already confines every write
            print('REFUSED: --parity needs --out (the parity root)', file=sys.stderr)
            return record.EXIT_REFUSED
        _parity(args, out)
    from exopipe.phase3.core import fallback, wrap
    fallback.export_env()                      # B-30: descendants use this step's policies; non-legacy ones spool here
    wrap.export_env()                          # B-32: descendants wrap the same in-place modules (core/wrap.py)
    run = record.Run(name, [f'exopipe3 {name}'] + rest, out, CFG, dry_run=args.dry_run,
                     waivers=list(zip(args.waive, args.ruling)), resume=args.resume, fresh=args.fresh,
                     config_sections=sections)
    run.cores = pinned
    run.no_records = not getattr(mod, 'WRITES', True)
    record.mark_injected(run)                  # an injected run's start record, before the step writes anything
    with _sigterm_interrupts(name):            # SIGTERM unwinds the step as Ctrl-C does (Terminated)
        try:
            gate_on = space_gate_of(mod, args, out) if not args.dry_run else None
            if gate_on:
                # the space gate of the writer steps (core/space.py; config section space): the run user's Lustre quota
                # headroom and the free space of each root it writes, now and again before its units (record.Run.unit)
                from exopipe.phase3.core import space as _space
                run.space_gate = _space.Gate(gate_on[0], floors=getattr(CFG, gate_on[1]))
                run.space_gate.require()
            if getattr(mod, 'WRITES', True) and getattr(mod, 'PRODUCTS', True):
                record.check_code_identity()   # B-30 GITTAG_FALLBACK: before the step's run() (core/record.py)
            rc = mod.run(args, run)
        except record.Refusal as e:
            print(f'REFUSED: {e}', file=sys.stderr)
            run.run_reason = f'refused: {e}'           # a run-level refusal before any unit: kept in the run summary
            rc = record.EXIT_REFUSED
        except record.NoInput as e:
            print(f'NO-INPUT: {e}', file=sys.stderr)
            run.run_reason = f'no-input: {e}'
            rc = record.EXIT_NO_INPUT
        except KeyboardInterrupt as e:             # Ctrl-C or SIGTERM: the run summary says so, then it propagates
            run.run_reason = f'interrupted: {type(e).__name__} {e}'.strip()
            raise
        finally:
            summ = run.write_summary()
    if rc is None:
        rc = run.exit_code()
    by = run.summary()
    for u in run.units:                  # why: one line per unit that did not end done (the record holds the rest)
        if u.status in ('refused', 'error') and u.reason:
            print(f'  {u.status} {u.name}: {u.reason[:600]}', file=sys.stderr)
    print(f'exopipe3 {name}: {", ".join(f"{k} {v}" for k, v in sorted(by.items())) or "no units"}; exit {rc}'
          + (f'; record {summ}' if summ else ''), file=sys.stderr)
    if outcome_fd:                       # a chain step: the runner reads which units (cells) refused (core/runner.py)
        from exopipe.phase3.core import runner as _runner
        try:
            _runner.hand_outcome(outcome_fd, _runner.step_outcome(name, rc, run, summ))
        except Exception as e:           # noqa: BLE001 - no outcome: the runner treats an exit 2 as the whole step's refusal
            print(f'the step outcome was not handed to the runner ({type(e).__name__}: {e}); a refusal stops the chain '
                  f'item', file=sys.stderr)
    return rc


if __name__ == '__main__':
    sys.exit(main())
