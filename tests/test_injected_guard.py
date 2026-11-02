#!/usr/bin/env python3
"""STANDING TEST - the INJECTED-RUN GUARD (core/injected.py; plan/CUTOVER_READINESS.md 'INJECTED RUNS WRITE NO PRODUCTION
PRODUCT, STRUCTURALLY'; ledger INCIDENT-INJECTED-SEARCH-WROTE-PRODUCTION-PHOTOMETRY-CUTOUTS-HS-PSC-EVENT-UNREPRODUCIBLE-20261024).

  flag         EXOPIPE3_INJECTED: from --inject / --injected, inherited (sticky), forwarded by `run submit`, given to every
               core.legacy.run child with the engines' P3_SNAP_INJECTED=1 after the caller's env (never dropped); run and unit
               records carry 'injected'
  out gate     an injected run needs an explicit scratch --out and refuses a --out / --work under any refused root (CLI exit 2,
               before anything is written); paths.PRODUCTION_WRITES is False in it (begin; a child's config load)
  write guard  INJECTED-RUN GUARD: a write outside --out / --work / scratch refuses in the process and its Python children
  WRITERS      one refusal test per production writer of the notes ("archive tree, photometry root, served run,
               data_decisions"), with paths.PRODUCTION_WRITES True - the post-cutover state:
                 archive tree    archive G2 (Ctx.guard_writes, every apply) refuses the archive root; the copy primitive's
                                 write is refused by the guard
                 photometry root the snapshot search's cutout root and engine switches; a legacy engine writing its own
                                 default root (the incident's path) refused in the child
                 served run      serve refuses (in process and CLI); pages / catalog into a served-run root refused at --out
                 data_decisions  refused by name wherever it lives; a write to it refused in the process and in a child
Production paths are only ever CHECKED (every check raises before any I/O); each write attempt targets a synthetic path - outside
/tmp where the guard must refuse it (a path under / that no user can create, so a missing refusal still writes nothing).

Run:  python tests/test_injected_guard.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')
for _k in ('EXOPIPE3_INJECTED', 'P3_SNAP_INJECTED', 'EXOPIPE3_PARITY_ROOT', 'EXOPIPE3_PARITY_LOG'):
    os.environ.pop(_k, None)            # the test process itself starts as an ordinary run

from exopipe.phase3.config import CFG                                  # noqa: E402
from exopipe.phase3.core import env as ENV                             # noqa: E402
ENV.legacy()
from exopipe.phase3.core import guard, injected, legacy, record, runner   # noqa: E402
from exopipe.phase3.core.record import Refusal                         # noqa: E402

T = '/tmp/exopipe3_tests_injected_guard'
OUTSIDE = '/exopipe3_injected_guard_test_nonexistent'      # outside every allowed root; no user can create it under /
PY = sys.executable
FAILS, N = [], [0]
SAVED_ENV = ('EXOPIPE3_INJECTED', 'P3_SNAP_INJECTED', 'EXOPIPE3_PARITY_ROOT', 'EXOPIPE3_PARITY_LOG', 'PYTHONPATH', 'TMPDIR')


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({str(detail)[:700]})'), flush=True)
    if not ok:
        FAILS.append(name)


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    return None


class Clean:
    """A block run as the post-cutover state (paths.PRODUCTION_WRITES True, optional synthetic paths) and undone after: the
    guard uninstalled, the environment, PRODUCTION_WRITES and the patched paths restored."""

    def __init__(self, **paths):
        self.paths = paths

    def __enter__(self):
        self.env = {k: os.environ.get(k) for k in SAVED_ENV}
        self.pw = CFG.paths.PRODUCTION_WRITES
        self.saved = {k: getattr(CFG.paths, k) for k in self.paths}
        for k, v in self.paths.items():
            setattr(CFG.paths, k, v)
        CFG.paths.PRODUCTION_WRITES = True
        return self

    def __exit__(self, *exc):
        guard.uninstall()
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        CFG.paths.PRODUCTION_WRITES = self.pw
        for k, v in self.saved.items():
            setattr(CFG.paths, k, v)
        tempfile.tempdir = None
        return False


def cli(*args, env=None, timeout=300):
    e = {k: v for k, v in os.environ.items() if k not in SAVED_ENV[:4]}
    e['PYTHONPATH'] = REPO + (os.pathsep + e['PYTHONPATH'] if e.get('PYTHONPATH') else '')
    e.update(env or {})
    return subprocess.run([PY, '-m', 'exopipe.phase3'] + [str(a) for a in args], cwd=REPO, env=e, capture_output=True,
                          text=True, timeout=timeout)


def files_under(d):
    return sorted(os.path.join(dp, f) for dp, _dn, fn in os.walk(d) for f in fn) if os.path.isdir(d) else []


def child_script(name, body):
    """A small 'legacy engine' in T: runs ``body`` and prints a JSON line of its results (exception class names)."""
    p = os.path.join(T, name)
    with open(p, 'w') as f:
        f.write('import json, os, sys\nres = {}\n'
                'def attempt(key, fn):\n'
                '    try:\n'
                '        fn()\n'
                '        res[key] = "ok"\n'
                '    except Exception as e:\n'
                '        res[key] = type(e).__name__\n' + body +
                '\nprint("RES" + json.dumps(res))\n')
    return p


def child_result(log):
    lines = [x for x in open(log).read().splitlines() if x.startswith('RES')]
    return json.loads(lines[-1][3:]) if lines else {}


# =====================================================================================================================

def test_flag_runner_records():
    NS = SimpleNamespace
    check('requested: --inject MANIFEST', injected.requested(NS(inject='m.csv', injected=False), environ={}) == '--inject m.csv')
    check('requested: --injected', injected.requested(NS(inject=None, injected=True), environ={}) == '--injected')
    check('requested: inherited from the parent (sticky, wins)', injected.requested(
        NS(inject='m.csv', injected=True), environ={injected.ENV: 'search --inject x (parent)'}) == 'search --inject x (parent)')
    check('requested: an ordinary run', injected.requested(NS(), environ={}) == '')
    env = {}
    injected.mark('first', env)
    injected.mark('second', env)
    check('mark is sticky and sets the engines\' P3_SNAP_INJECTED=1', env == {injected.ENV: 'first', 'P3_SNAP_INJECTED': '1'},
          env)
    check('no flag: child_env / engine_env empty', injected.child_env() == {} and injected.engine_env() == {})
    # runner: `run submit --injected` forwards the flag into every step command and records it
    from exopipe.phase3.steps import run as srun
    fwd = srun.forward_argv(NS(set=[], config=None, ruling_for=[], parity=False, injected=True))
    check('run submit forwards --injected; the job record notes it', fwd == ['--injected']
          and runner._overrides_of(fwd)['injected'] is True and runner._overrides_of([])['injected'] is False, fwd)
    # legacy children: EXOPIPE3_INJECTED and P3_SNAP_INJECTED=1 after the caller's env; nothing without the flag
    os.makedirs(T, exist_ok=True)
    sc = child_script('env_child.py', 'res["EXOPIPE3_INJECTED"] = os.environ.get("EXOPIPE3_INJECTED")\n'
                                     'res["P3_SNAP_INJECTED"] = os.environ.get("P3_SNAP_INJECTED")\n')
    log0 = os.path.join(T, 'env_child0.log')
    r0 = legacy.run(sc, env={'P3_SNAP_INJECTED': '0'}, log=log0)
    with Clean():
        injected.mark('test: legacy child')
        log1 = os.path.join(T, 'env_child1.log')
        r1 = legacy.run(sc, env={'P3_SNAP_INJECTED': '0', 'EXOPIPE3_INJECTED': ''}, log=log1)
    a, b = child_result(log0), child_result(log1)
    check('an ordinary run\'s legacy child gets no injected flag (the caller\'s P3_SNAP_INJECTED=0 kept)',
          r0.ok and a == {'EXOPIPE3_INJECTED': None, 'P3_SNAP_INJECTED': '0'}, (r0.rc, a))
    check('an injected run\'s legacy child: EXOPIPE3_INJECTED and P3_SNAP_INJECTED=1 even when the caller passes 0 / empty',
          r1.ok and b == {'EXOPIPE3_INJECTED': 'test: legacy child', 'P3_SNAP_INJECTED': '1'}, (r1.rc, b))
    # records: the run header (hence every unit record) carries the reason; record.injected reads it
    out = os.path.join(T, 'rec')
    with Clean():
        injected.mark('search --inject m.csv')
        rr = record.Run('test', ['x'], out, CFG)
        with rr.unit('u') as u:
            p = os.path.join(out, 'x.csv')
            os.makedirs(out, exist_ok=True)
            open(p, 'w').write('a\n1\n')
            u.add_output(p, 'csv_table')
    rec = record.load_record(out, 'test', 'u')
    plain = record.Run('test', ['x'], out, CFG)
    check('an injected run\'s records carry injected = the reason (record.injected); an ordinary run\'s none',
          rec.get('injected') == 'search --inject m.csv' and record.injected(rec) == 'search --inject m.csv'
          and 'injected' not in plain.header() and record.injected(plain.header()) == '', rec.get('injected'))


def test_refused_roots():
    roots = set(injected.refused_roots())
    rp = os.path.realpath
    want = {'paths.PRODUCTION_ROOTS': [rp(x) for x in CFG.paths.PRODUCTION_ROOTS],
            'archive tree': [rp(CFG.paths.PHASE3_ARCHIVE_ROOT)],
            'photometry roots': [rp(CFG.forced_snapshot.PHOT_ROOT), rp(CFG.archive_snapshot.PHOT_ROOT)]
            + [rp(x) for x in CFG.photometry.OUT_ROOT.values()],
            'served-run roots': [rp(CFG.paths.TRANSIENTS_ROOT), rp(CFG.paths.SNAPSHOT_REVIEW_ROOT)],
            'Phase-1 / Phase-2 data': [rp(CFG.paths.PHASE1_ROOT), rp(CFG.paths.PHASE2_ROOT)],
            'ops, clv15, reference tables': [rp(CFG.paths.OPS_ROOT), rp(CFG.paths.CLV15_ROOT), rp(CFG.paths.REFERENCE_DIR)],
            'other sections\' production roots': [rp(x) for x in CFG.followup.PRODUCTION_ROOTS]
            + [rp(x) for x in CFG.masters.EXTRA_PRODUCTION_ROOTS]}
    miss = {k: [x for x in v if x not in roots] for k, v in want.items()}
    check('refused_roots holds every production root class', not any(miss.values()), miss)
    check('refused_roots never holds /tmp, /dev/shm or /', not roots & {'/tmp', '/dev/shm', '/', '/dev/null'}, sorted(roots)[:5])
    check('data_decisions, the fix ledger and the run-pin file are refused by name',
          injected.refused_root_of(CFG.paths.DATA_DECISIONS) == CFG.paths.DATA_DECISIONS
          and injected.refused_root_of(CFG.paths.FIX_LEDGER) == CFG.paths.FIX_LEDGER
          and injected.refused_root_of(CFG.paths.PHASE1_RUN_PINS) == CFG.paths.PHASE1_RUN_PINS)
    check('the photometry generations and served runs are under refused roots; /tmp is not',
          injected.refused_root_of(os.path.join(CFG.paths.PHASE3_ARCHIVE_ROOT, 'photometry_gen5', 'x.fits'))
          and injected.refused_root_of(os.path.join(CFG.paths.TRANSIENTS_ROOT, 'Run_P3S5', 'queue.txt'))
          and injected.refused_root_of(os.path.join(CFG.serve.HUB_DIR, 'registry.json'))
          and injected.refused_root_of('/tmp/x') is None)


def test_begin_out_gate_and_guard():
    out = os.path.join(T, 'out')
    with Clean():
        e = raises(Refusal, injected.begin, 'test', out, explicit_out=False, default_out=CFG.paths.TRANSIENTS_ROOT)
        check('no explicit --out: refused (the default is a production root)', e is not None and 'explicit --out' in str(e), e)
    with Clean():
        e = raises(Refusal, injected.begin, 'test', os.path.join(CFG.paths.TRANSIENTS_ROOT, 'Run_INJ'))
        check('--out under a served-run root: refused, naming that root', e is not None
              and f'is under {os.path.realpath(CFG.paths.TRANSIENTS_ROOT)},' in str(e), e)
    with Clean():
        e = raises(Refusal, injected.begin, 'test', out, work=os.path.join(CFG.paths.COADD_SB_ROOT, 'w'))
        check('--work under a production tree: refused', e is not None and '--work' in str(e), e)
    with Clean():
        e = raises(Refusal, injected.begin, 'test', os.path.join(OUTSIDE, 'step_root', 'x'),
                   default_out=os.path.join(OUTSIDE, 'step_root'))
        check('the step\'s own default --out is refused as an --out even where no other root lists it', e is not None
              and 'step_root' in str(e), e)
        check('... a /tmp default (verify, run) never refuses /tmp', injected.refused_root_of('/tmp/x', extra=('/tmp',)) is None)
    with Clean():
        roots = injected.begin('test: begin', out, default_out=CFG.paths.TRANSIENTS_ROOT)
        ok_env = (os.environ.get(injected.ENV) == 'test: begin' and os.environ.get('P3_SNAP_INJECTED') == '1'
                  and out in os.environ.get('EXOPIPE3_PARITY_ROOT', '').split(os.pathsep)
                  and os.environ.get('PYTHONPATH', '').split(os.pathsep)[0] == guard.SITECUSTOMIZE_DIR)
        check('begin: the flag, P3_SNAP_INJECTED=1, the guard roots and the sitecustomize exported to children', ok_env,
              {k: os.environ.get(k) for k in SAVED_ENV})
        check('begin: paths.PRODUCTION_WRITES False (it was True)', CFG.paths.PRODUCTION_WRITES is False)
        check('begin: the INJECTED-RUN GUARD is installed over --out (+ scratch, /tmp)', guard.active()
              and guard._STATE.get('label') == injected.LABEL and os.path.realpath(out) in roots, roots)
        check('the guard refuses the real production targets (checked, never written): data_decisions, the photometry root, '
              'the archive tree, a served run', not any(guard.is_allowed(p) for p in (
                  CFG.paths.DATA_DECISIONS, os.path.join(CFG.paths.PHASE3_ARCHIVE_ROOT, 'photometry', 'x.fits'),
                  os.path.join(CFG.paths.PHASE3_ARCHIVE_ROOT, CFG.paths.ARCHIVE_RUN, 'x.fits'),
                  os.path.join(CFG.paths.TRANSIENTS_ROOT, 'Run_P3S5', 'x.csv'))))
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, 'ok.txt'), 'w') as f:
            f.write('scratch')
        e = raises(PermissionError, lambda: open(os.path.join(OUTSIDE, 'x.txt'), 'w'))
        check('a write outside the scratch roots raises InjectedWriteRefused (a ParityWriteRefused, PermissionError)',
              isinstance(e, guard.InjectedWriteRefused) and isinstance(e, guard.ParityWriteRefused), repr(e))
        log = os.path.join(out, '_records', injected.LOG_NAME)
        check('the refusal is logged as INJECTED-RUN GUARD in <out>/_records', os.path.isfile(log)
              and 'INJECTED-RUN GUARD' in open(log).read() and OUTSIDE in open(log).read())
    check('after the block: no guard, no flag, PRODUCTION_WRITES restored', not guard.active()
          and not injected.active() and CFG.paths.PRODUCTION_WRITES is False)
    # a child's config load: PRODUCTION_WRITES False under an inherited flag (the subprocess reads the package's default,
    # False in the refactor window, so the forcing is shown on a True default through force_no_production_writes)
    with Clean():
        os.environ[injected.ENV] = 'test: child'
        check('force_no_production_writes: True default -> False under the flag', injected.force_no_production_writes()
              and CFG.paths.PRODUCTION_WRITES is False)
    code = ('import os, sys; sys.path.insert(0, %r); os.environ["EXOPIPE3_INJECTED"] = "child"\n'
            'from exopipe.phase3.config import CFG; print("PW", CFG.paths.PRODUCTION_WRITES)' % REPO)
    r = subprocess.run([PY, '-c', code], capture_output=True, text=True, timeout=120)
    check('a child that loads the config under the flag has PRODUCTION_WRITES False', 'PW False' in r.stdout, r.stderr[-500:])


def test_writer_archive_tree():
    from exopipe.phase3.lib.archive import common as K
    out = os.path.join(T, 'arch_out')
    arch = CFG.paths.PHASE3_ARCHIVE_ROOT                 # the real archive tree: only CHECKED below
    args = SimpleNamespace(tier='deep', action='push', archive_root=None, generation_run=None, apply=True)
    with Clean():
        ctx = K.Ctx(args, SimpleNamespace(out_root=out, dry_run=False))
        check('ordinary run, PRODUCTION_WRITES True: archive G2 lets an apply reach the archive tree (no write made here)',
              raises(Refusal, ctx.guard_writes, [arch], 'push --class epoch') is None)
    with Clean():
        injected.begin('test: archive', out)
        CFG.paths.PRODUCTION_WRITES = True               # whatever flips it back: the writer still refuses by the flag
        ctx = K.Ctx(args, SimpleNamespace(out_root=out, dry_run=False))
        e = raises(Refusal, ctx.guard_writes, [arch], 'push --class epoch')
        check('ARCHIVE TREE: under the flag archive G2 refuses the archive root by name (INJECTED-RUN GUARD)',
              e is not None and injected.LABEL in str(e) and 'archive push' in str(e), e)
        mirror = os.path.join(out, 'archive_mirror')
        check('... and accepts a mirror archive root under the run\'s scratch --out',
              raises(Refusal, ctx.guard_writes, [mirror], 'push --class epoch') is None)
        src = os.path.join(out, 'product.fits')
        os.makedirs(out, exist_ok=True)
        open(src, 'wb').write(b'x' * 64)
        e = raises(Exception, K.verified_copy, src, os.path.join(OUTSIDE, 'Run_20260909_000000', 'product.fits'))
        check('... and the copy primitive\'s write outside scratch is refused by the guard itself (InjectedWriteRefused)',
              isinstance(e, guard.InjectedWriteRefused), repr(e))
        e = raises(Refusal, injected.check, os.path.join(arch, CFG.paths.ARCHIVE_RUN, '04h', 'x.fits'), 'archive push')
        check('injected.check refuses the archive generation Run', e is not None)


def test_writer_photometry_root():
    from exopipe.phase3.lib.search import tier_snapshot as TS
    out = os.path.join(T, 'phot_out')
    run = SimpleNamespace(out_root=out)
    sargs = SimpleNamespace(snap_pause_1min=False)
    prod_root = os.path.join(CFG.paths.PHASE3_ARCHIVE_ROOT, CFG.search_snapshot.CUTOUT_SUBDIR)
    with Clean():
        env0, _ = TS.engine_env(run, sargs)
        check('ordinary run, PRODUCTION_WRITES True: the cutout root is the production photometry root, cutouts on, no '
              'injected switch', TS.cutout_root(run) == prod_root and env0['P3_SNAP_CUTOUTS'] == '1'
              and 'P3_SNAP_INJECTED' not in env0, env0)
    with Clean():
        injected.begin('test: photometry', out)
        env1, _ = TS.engine_env(run, sargs)
        check('PHOTOMETRY ROOT: under the flag the cutout root is <out>/photometry, no cutout is cut (P3_SNAP_CUTOUTS=0) and '
              'the engine gets P3_SNAP_INJECTED=1', TS.cutout_root(run) == os.path.join(out, 'photometry')
              and env1['P3_SNAP_CUTOUT_ROOT'] == os.path.join(out, 'photometry') and env1['P3_SNAP_CUTOUTS'] == '0'
              and env1.get('P3_SNAP_INJECTED') == '1', env1)
        CFG.paths.PRODUCTION_WRITES = True
        e = raises(Refusal, TS.cutout_root, run)
        check('... and refuses the production photometry root by name if PRODUCTION_WRITES were True again',
              e is not None and injected.LABEL in str(e) and 'P3_SNAP_CUTOUT_ROOT' in str(e), e)
        CFG.paths.PRODUCTION_WRITES = False
        for p in (os.path.join(prod_root, 'HWJ', '41MHz', 'HS_Psc', 'x.fits'),
                  os.path.join(CFG.paths.PHASE3_ARCHIVE_ROOT, 'photometry_gen5', 'x.fits')):
            check(f'injected.check refuses {p.replace(CFG.paths.PHASE3_ARCHIVE_ROOT, "<archive>")}',
                  raises(Refusal, injected.check, p, 'cutouts') is not None)
        # the incident's path: an in-place engine writes its own DEFAULT root (not the configured one) - the guard reaches it
        sc = child_script('engine_child.py',
                          f'attempt("default_root", lambda: (os.makedirs({OUTSIDE + "/phase3/photometry/HWJ"!r}, exist_ok=True),'
                          f' open({OUTSIDE + "/phase3/photometry/HWJ/HS_Psc_04h_cut.fits"!r}, "wb").write(b"x")))\n'
                          'attempt("configured_root", lambda: (os.makedirs(os.environ["P3_SNAP_CUTOUT_ROOT"], exist_ok=True), '
                          'open(os.path.join(os.environ["P3_SNAP_CUTOUT_ROOT"], "ok.fits"), "wb").write(b"x")))\n'
                          'res["P3_SNAP_INJECTED"] = os.environ.get("P3_SNAP_INJECTED")\n')
        log = os.path.join(T, 'engine_child.log')
        r = legacy.run(sc, env={k: v for k, v in env1.items() if v is not None}, log=log)
        res = child_result(log)
        check('PHOTOMETRY ROOT, legacy engine: its default-root write is refused in the child (InjectedWriteRefused), the '
              'configured scratch root is written, the child had P3_SNAP_INJECTED=1',
              r.rc == 0 and res.get('default_root') == 'InjectedWriteRefused' and res.get('configured_root') == 'ok'
              and res.get('P3_SNAP_INJECTED') == '1' and not os.path.exists(OUTSIDE), (r.rc, res, r.tail[-400:]))


def test_writer_served_run():
    from exopipe.phase3.steps import serve as SV
    with Clean():
        injected.begin('test: serve', os.path.join(T, 'serve_out'))
        # arguments that can do nothing (no --restart / --stop, a dry run): were the refusal missing, the step would stop at
        # its first missing argument - this test must never reach a server
        e = raises(Refusal, SV.run, SimpleNamespace(restart=None, stop=None), SimpleNamespace(out_root=T, dry_run=True))
        check('SERVED RUN: serve refuses an injected run before anything (its first statement: restarts and stops too)',
              e is not None and 'never served' in str(e), e)
        import inspect
        body = inspect.getsource(SV.run).split('injected.refuse(', 1)[0]
        check('... the refusal precedes --restart / --stop and the plan', 'restart_or_stop' not in body
              and 'planmod.make' not in body)
        for p in (os.path.join(CFG.paths.TRANSIENTS_ROOT, 'Run_P3S5', 'queue_p3s3_V.txt'),
                  os.path.join(CFG.paths.SNAPSHOT_REVIEW_ROOT, 'Run_SNAPCOMB_20261026', 'fp_filters.csv'),
                  os.path.join(CFG.serve.HUB_DIR, 'registry.json')):
            check(f'injected.check refuses the served-run / hub file {os.path.basename(os.path.dirname(p))}/'
                  f'{os.path.basename(p)}', raises(Refusal, injected.check, p, 'pages') is not None)
    sout = os.path.join(T, 'serve_cli')
    r = cli('serve', '--tier', 'deep', '--run', 'P3S5', '--injected', '--out', sout, '--dry-run', '--cores', 'none')
    check('SERVED RUN (CLI): `serve --injected` exits 2 (INJECTED-RUN GUARD: never served), nothing written',
          r.returncode == 2 and 'never served' in r.stderr and not files_under(sout), (r.returncode, r.stderr[-600:]))
    r = cli('pages', '--tier', 'deep', '--injected', '--dry-run', '--cores', 'none')
    check('SERVED RUN (CLI): `pages --injected` without --out (default: the served-run root) exits 2', r.returncode == 2
          and 'explicit --out' in r.stderr, (r.returncode, r.stderr[-400:]))
    r = cli('pages', '--tier', 'deep', '--injected', '--out', os.path.join(CFG.paths.TRANSIENTS_ROOT, 'Run_INJTEST'),
            '--dry-run', '--cores', 'none')
    check('SERVED RUN (CLI): `pages --injected --out <served-run root>/...` exits 2 before anything is written',
          r.returncode == 2 and injected.LABEL in r.stderr
          and not os.path.exists(os.path.join(CFG.paths.TRANSIENTS_ROOT, 'Run_INJTEST')), (r.returncode, r.stderr[-400:]))
    r = cli('catalog', '--tier', 'deep', '--injected', '--dry-run', '--cores', 'none', '--cells', '04h_Red')
    check('SERVED RUN (CLI): `catalog --injected` without --out (the publish root) exits 2', r.returncode == 2
          and 'explicit --out' in r.stderr, (r.returncode, r.stderr[-400:]))


def test_writer_data_decisions():
    real = CFG.paths.DATA_DECISIONS
    e = raises(Refusal, lambda: (injected.mark('test: dd', os.environ), injected.check(real, 'exclusions')))
    os.environ.pop(injected.ENV, None)
    os.environ.pop('P3_SNAP_INJECTED', None)
    check('DATA_DECISIONS: injected.check refuses the real file by name (checked, never opened)', e is not None
          and real in str(e), e)
    synth_dir = os.path.join(T, 'clv15')
    os.makedirs(synth_dir, exist_ok=True)
    synth = os.path.join(synth_dir, 'data_decisions.csv')
    with open(synth, 'w') as f:
        f.write('lst,scope,date,action,reason\n')
    with Clean(DATA_DECISIONS=synth, CLV15_ROOT=synth_dir):
        out = os.path.join(T, 'dd_out')
        injected.begin('test: data_decisions', out)
        e = raises(Refusal, injected.check, synth, 'a writer of exclusions')
        check('DATA_DECISIONS: refused by name wherever it lives (here under /tmp, otherwise scratch)', e is not None
              and injected.LABEL in str(e), e)
        # the write itself: the guard without its always-allowed /tmp (as test_core_v2 item 5), so the /tmp copy is outside
        ptmp = os.path.join(out, 'ptmp')
        os.makedirs(ptmp, exist_ok=True)
        os.environ['TMPDIR'] = ptmp
        tempfile.tempdir = ptmp
        guard.install([out], defaults=('/dev/null',), log=os.path.join(out, '_records', injected.LOG_NAME),
                      label=injected.LABEL)
        e = raises(PermissionError, lambda: open(synth, 'a'))
        e2 = raises(PermissionError, os.replace, os.path.join(ptmp, 'x'), synth)
        check('DATA_DECISIONS: appending to it or replacing it raises InjectedWriteRefused in the process',
              isinstance(e, guard.InjectedWriteRefused) and isinstance(e2, guard.InjectedWriteRefused), (repr(e), repr(e2)))
        guard.uninstall()
        injected.begin('test: data_decisions', out)              # the run's guard again (exported to children)
        sc = child_script('dd_child.py', f'attempt("append", lambda: open({OUTSIDE + "/clv15/data_decisions.csv"!r}, "a")'
                                         f'.write("04h,Full,2025-01-01,exclude,synthetic\\n"))\n')
        log = os.path.join(T, 'dd_child.log')
        r = legacy.run(sc, log=log)
        res = child_result(log)
        check('DATA_DECISIONS: a child\'s append to an exclusion list outside scratch is refused (InjectedWriteRefused)',
              r.rc == 0 and res.get('append') == 'InjectedWriteRefused' and not os.path.exists(OUTSIDE), (r.rc, res))
    check('the synthetic data_decisions.csv is unchanged', open(synth).read() == 'lst,scope,date,action,reason\n')


def test_cli_inject_and_inherit():
    man = os.path.join(T, 'manifest.csv')
    with open(man, 'w') as f:
        f.write('inj_id,ra,dec,flux_jy\n')
    r = cli('search', '--tier', 'snapshot', '--cells', '18h_2030-01-01', '--inject', man, '--dry-run', '--cores', 'none')
    check('`search --inject` without --out exits 2 (the default is the production ship tree)', r.returncode == 2
          and 'explicit --out' in r.stderr and '--inject' in r.stderr, (r.returncode, r.stderr[-400:]))
    r = cli('search', '--tier', 'snapshot', '--cells', '18h_2030-01-01', '--inject', man, '--dry-run', '--cores', 'none',
            '--out', os.path.join(CFG.paths.SNAPSHOT_FLEET_OUT, 'inject_test'))
    check('`search --inject --out <snapshot ship tree>` exits 2', r.returncode == 2 and injected.LABEL in r.stderr
          and not os.path.exists(os.path.join(CFG.paths.SNAPSHOT_FLEET_OUT, 'inject_test')), (r.returncode, r.stderr[-400:]))
    good = os.path.join(T, 'good.csv')
    with open(good, 'w') as f:
        f.write('a\n1\n')
    vout = os.path.join(T, 'vout')
    r = cli('verify', 'csv_table', good, '--cores', 'none', '--out', vout, env={injected.ENV: 'battery 2026-11-02 (parent)'})
    check('an inherited flag makes any step an injected run (verify: guard on, done)', r.returncode == 0
          and 'INJECTED-RUN GUARD (battery 2026-11-02 (parent))' in r.stderr, (r.returncode, r.stderr[-400:]))
    r = cli('verify', 'csv_table', good, '--cores', 'none', '--out', vout, '--emit-failing',
            os.path.join(OUTSIDE, 'failing.txt'), env={injected.ENV: 'battery (parent)'})
    check('... whose write outside scratch is refused by the guard (InjectedWriteRefused)', r.returncode == 1
          and 'InjectedWriteRefused' in r.stderr and not os.path.exists(OUTSIDE), (r.returncode, r.stderr[-400:]))


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    for t in (test_flag_runner_records, test_refused_roots, test_begin_out_gate_and_guard, test_writer_archive_tree,
              test_writer_photometry_root, test_writer_served_run, test_writer_data_decisions, test_cli_inject_and_inherit):
        try:
            t()
        except Exception as e:                     # noqa: BLE001 - a crashing test is a failure, the rest still run
            import traceback
            traceback.print_exc()
            check(f'{t.__name__} raised {type(e).__name__}: {e}', False)
        finally:
            guard.uninstall()
    shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed' + (f'; FAILED: {FAILS}' if FAILS else ''))
    return 1 if FAILS else 0


if __name__ == '__main__':
    raise SystemExit(main())
