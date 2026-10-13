"""exopipe3 - the ONE entry point of the OVRO-LWA ExoPipe Phase 3 pipeline.

    exopipe3 --help                 the steps, in pipeline order
    exopipe3 <step> --help          what the step reads and writes, its failure modes, every flag
    exopipe3 config [--section S]   every tunable (one file: exopipe/phase3/config.py)

Every step is independently runnable by a person or by Claude, documents its inputs, outputs and failure modes in
--help, refuses (exit 2) instead of warning, has --dry-run, and writes a machine-readable record of what it did under
<out>/_records/<step>/. Exit codes: 0 done (every content test passed), 1 error, 2 refused by a gate, 3 nothing to do.
"""
import os
import sys

from exopipe.phase3.core import env as _env

_env.apply()            # before anything imports numpy (core/env.py)

import argparse        # noqa: E402
import importlib       # noqa: E402

# (name, one-line summary, group) in pipeline order. The module is exopipe.phase3.steps.<name>.
STEPS = [
    # per hour: one LST hour of one night, in near real time, against the installed masters (DESIGN 12.1)
    ('frames', 'per-epoch / per-window / per-frame residuals and search frames, pbcorr siblings, noise maps', 'per hour'),
    ('flow', 'alignment flows: snapshot frames and sub-LST windows (per hour); per-night co-add flows (masters builds)', 'per hour'),
    ('search', 'detection + gates with fitted positions (deep, sub-LST, snapshot)', 'per hour'),
    ('catalog', 'merge, conformance, association, identity, selection, names (--incremental adds one hour); snapshot chain', 'per hour'),
    ('forced', 'forced photometry (Gaussian fit >= 7 sigma, empirical-beam template below); event panels', 'per hour'),
    ('curves', 'light-curve products', 'per hour'),
    ('pages', 'review pages (served queues, inspection twins, multiples), incremental or full', 'per hour'),
    ('serve', 'review servers and the hub; served-queue and exclusion content tests; exact-pid restart', 'per hour'),
    ('archive', 'manifest, checksum transfer, content audit, quarantine; retire excluded nights; rollback', 'per hour'),
    # survey (intermittent): many hours on disk (DESIGN 12.2)
    ('masters', 'co-add masters + LOO + QA gates; V clean; flux scale; PB maps; source catalogue (all tiers)', 'survey'),
    ('dewarp', 'master dewarp per LST: RBF screen on the rebuilt masters, cubic apply, content test D1-D6', 'survey'),
    ('install', 'gated install of rebuilt products into the working tree; quarantine round; re-run list', 'survey'),
    ('calibrate', 'versioned reference tables: derive, diff, install, verify', 'survey'),
    ('mosaic', 'mosaicked products', 'survey'),
    ('photometry', 'photometry of input samples (Hot/Warm Jupiters, flare/UCD, 10/25/50 pc)', 'survey'),
    ('followup', 'follow-up of high-interest candidates (honing, dynamic spectra, periodicity, folding, dedispersion)', 'survey'),
    ('jupiter', 'the Jupiter reflection program', 'survey'),
    # tools
    ('status', 'completion, exclusion conformance, stale products, coverage, rebuild due; emits item lists', 'tools'),
    ('verify', 'run the content test of any product class alone; compare two product roots (parity)', 'tools'),
    ('run', 'multi-node runner: jobs, chains (e.g. the per-hour chain), --after, --code, hold/release/drain/retry', 'tools'),
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
    cpus = set()
    for part in cores.split(','):
        a, _, b = part.partition('-')
        cpus.update(range(int(a), int(b or a) + 1))
    os.sched_setaffinity(0, cpus)
    return cores


def _guard_out(out):
    from exopipe.phase3.config import CFG
    if CFG.paths.PRODUCTION_WRITES:
        return
    root = CFG.production_root_of(out)
    if root is not None:
        raise SystemExit(f'refused: --out {out} is under the production root {root} and paths.PRODUCTION_WRITES '
                         f'is False (refactor window). Write parity/scratch output to your own root.')


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
    p = argparse.ArgumentParser(prog='exopipe3 config', description='Show the Phase 3 tunables (config.py).')
    p.add_argument('--section', help='print one section with its comments (units, rulings)')
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


def build_parser():
    p = argparse.ArgumentParser(prog='exopipe3', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='step', metavar='<step>')
    group = None
    for name, summary, grp in STEPS:
        sub.add_parser(name, help=summary + ('' if grp == group else f'   [{grp}]'), add_help=False)
        group = grp
    sub.add_parser('config', help='show every tunable (config.py)', add_help=False)
    return p


def main(argv=None):
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
        # Ruled keys may be overridden with --ruling-for or into a non-production --out; `run submit` forwards its
        # overrides to every step process, which checks them against its own --out.
        CFG.authorize(args.ruling_for, out=args.out or mod.default_out(args),
                      mode='forward' if name == 'run' else 'run')
        # parity replay of protected inputs (config.set): only under --parity into a non-production --out
        CFG.parity_replay = bool(getattr(args, 'parity', False) and name != 'run' and args.out
                                 and CFG.production_root_of(args.out) is None)
        if args.config:
            CFG.apply_override_file(args.config)
        CFG.apply_override_strings(args.set)
        out = args.out or mod.default_out(args)
        CFG.check_out(out)
    except ConfigRefusal as e:
        print(f'REFUSED: {e}', file=sys.stderr)
        return record.EXIT_REFUSED
    except (KeyError, ValueError) as e:
        print(f'error: {_msg(e)}', file=sys.stderr)
        return record.EXIT_ERROR
    pinned = _pin(args.cores)
    if getattr(mod, 'WRITES', True) and not args.dry_run:
        _guard_out(out)
    if args.parity and name != 'run':          # `run submit --parity` forwards it to every step command instead
        if not args.out:
            print('REFUSED: --parity needs --out (the parity root)', file=sys.stderr)
            return record.EXIT_REFUSED
        _parity(args, out)
    run = record.Run(name, [f'exopipe3 {name}'] + rest, out, CFG, dry_run=args.dry_run,
                     waivers=list(zip(args.waive, args.ruling)), resume=args.resume, fresh=args.fresh,
                     config_sections=sections)
    run.cores = pinned
    run.no_records = not getattr(mod, 'WRITES', True)
    try:
        rc = mod.run(args, run)
    except record.Refusal as e:
        print(f'REFUSED: {e}', file=sys.stderr)
        rc = record.EXIT_REFUSED
    except record.NoInput as e:
        print(f'NO-INPUT: {e}', file=sys.stderr)
        rc = record.EXIT_NO_INPUT
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
    return rc


if __name__ == '__main__':
    sys.exit(main())
