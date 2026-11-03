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
    check('... whose destination flag outside scratch is refused before the step runs (exit 2, verify --emit-failing)',
          r.returncode == 2 and 'verify --emit-failing' in r.stderr and injected.LABEL in r.stderr
          and not os.path.exists(OUTSIDE), (r.returncode, r.stderr[-400:]))
    sc = child_script('emit_child.py', f'attempt("emit", lambda: open({os.path.join(OUTSIDE, "failing.txt")!r}, "w"))\n')
    r = subprocess.run([PY, sc], capture_output=True, text=True, timeout=120,
                       env=dict({k: v for k, v in os.environ.items() if k not in SAVED_ENV[:4]},
                                **guard.export_env([vout], env={}), **{injected.ENV: 'battery (parent)'}))
    res = json.loads(([x for x in r.stdout.splitlines() if x.startswith('RES')] or ['RES{}'])[-1][3:])
    check('... and a child\'s write there is still refused by the guard itself (InjectedWriteRefused)',
          res.get('emit') == 'InjectedWriteRefused' and not os.path.exists(OUTSIDE), (res, r.stderr[-300:]))


# ---- review fixes 2026-11-02: jupiter serve, consumers of injected products, destination flags, run submit ---------------

def make_injected_out(name):
    """An injected run's --out with its start record (core/record.mark_injected), written the way cli writes it."""
    out = os.path.join(T, name)
    with Clean():
        injected.mark(f'test: {name}')
        rr = record.Run('search', ['exopipe3', 'search', '--inject', 'm.csv'], out, CFG)
        rec = record.mark_injected(rr)
    return out, rec


def test_jupiter_serve_refused():
    from exopipe.phase3.steps import jupiter as JS
    with Clean():
        injected.begin('test: jupiter serve', os.path.join(T, 'jserve'))
        # arguments that can do nothing: were the refusal missing, the step would stop at its first missing argument
        e = raises(Refusal, JS.run, SimpleNamespace(action='serve'), SimpleNamespace(out_root=T, dry_run=True))
        check('JUPITER SERVE: refused in an injected run before anything (8800 / 8801, parity ports)', e is not None
              and 'never served' in str(e) and 'jupiter serve' in str(e), e)
    jout = os.path.join(T, 'jserve_cli')
    r = cli('jupiter', 'serve', '--injected', '--out', jout, '--port-offset', '1000', '--dry-run', '--cores', 'none')
    check('JUPITER SERVE (CLI): `jupiter serve --injected --out <scratch> --port-offset 1000` exits 2, nothing written',
          r.returncode == 2 and 'never served' in r.stderr and not files_under(jout), (r.returncode, r.stderr[-500:]))


def test_injected_records_and_resume():
    out, rec = make_injected_out('inj_records')
    check('mark_injected writes <out>/_records/_injected/<run_id>.json naming the reason',
          rec and os.path.dirname(rec) == os.path.join(out, '_records', record.INJECTED_DIR)
          and json.load(open(rec)).get('injected') == 'test: inj_records', rec)
    hits = record.injected_runs(os.path.join(out, 'attempts', '04h_Red_V_x', 'f.fits'))
    check('injected_runs finds the start record from any path below the --out (the path need not exist)',
          len(hits) == 1 and hits[0][0] == out and hits[0][2] == 'test: inj_records', hits)
    check('... and nothing for an ordinary tree', record.injected_runs(os.path.join(T, 'plain', 'x')) == [])
    plain = record.Run('search', ['x'], out, CFG)
    plain.dry_run = True
    check('an ordinary run, a dry run and a step without records write no start record',
          record.mark_injected(plain) is None)
    # resume never crosses the injected / ordinary line
    rdir = os.path.join(T, 'resume')
    good = os.path.join(rdir, 'x.csv')
    os.makedirs(rdir, exist_ok=True)
    open(good, 'w').write('a\n1\n')

    def unit(name, inj):
        with Clean():
            if inj:
                injected.mark('test: resume')
            rr = record.Run('t', ['x'], rdir, CFG)
            rr.commit = 'abc1234'
            with rr.unit(name) as u:
                u.add_output(good, 'csv_table')
    unit('u_inj', True)
    unit('u_plain', False)
    ok1, why1 = record.resume_check(rdir, 't', 'u_inj')
    ok2, why2 = record.resume_check(rdir, 't', 'u_plain')
    with Clean():
        injected.mark('test: resume')
        ok3, why3 = record.resume_check(rdir, 't', 'u_inj')
        ok4, why4 = record.resume_check(rdir, 't', 'u_plain')
    check('resume_check: an ordinary run never adopts an injected run\'s unit, an injected run never an ordinary one; the same '
          'kind resumes', not ok1 and 'injected run' in why1 and ok2 and ok3 and not ok4 and 'ordinary run' in why4,
          (why1, why2, why3, why4))
    # an ordinary run into an injected --out is refused before the step runs (no resume, no reading, no writing)
    e = raises(Refusal, injected.check_flags, 'catalog', SimpleNamespace(), out, writes=True)
    check('check_flags: an ordinary run whose --out holds an injected run\'s start record is refused', e is not None
          and 'holds an injected run' in str(e), e)
    check('... a step that writes nothing (verify) is not', raises(Refusal, injected.check_flags, 'verify', SimpleNamespace(),
                                                                    out, writes=False) is None)
    r = cli('pages', '--tier', 'deep', '--out', out, '--dry-run', '--cores', 'none')
    check('CLI: an ordinary `pages --out <injected run\'s --out>` exits 2', r.returncode == 2
          and 'holds an injected run' in r.stderr, (r.returncode, r.stderr[-400:]))
    r = cli('pages', '--tier', 'deep', '--out', os.path.join(out, 'sub'), '--resume', '--dry-run', '--cores', 'none')
    check('CLI: ... and an ordinary --resume into a directory below it', r.returncode == 2 and 'holds an injected run' in
          r.stderr, (r.returncode, r.stderr[-400:]))
    with Clean():
        injected.mark('test: again')
        e = raises(Refusal, injected.check_flags, 'catalog', SimpleNamespace(), out, writes=True)
    check('an injected run may use an injected run\'s --out again', e is None, e)


def test_source_flags():
    out, _rec = make_injected_out('inj_source')
    src = os.path.join(out, 'Run_P3S9')
    plain = os.path.join(T, 'plain_src', 'Run_P3S9')
    os.makedirs(plain, exist_ok=True)
    tmpl = os.path.join(out, 'gen', '{cell}')
    import socket
    me = socket.gethostname().split('.')[0]
    for step, flags in sorted(injected.SOURCE_FLAGS.items()):
        for flag in flags:
            val = [src] if flag == 'attempt' else (f'run:{src}' if flag == 'inc_source' else src)
            e = raises(Refusal, injected.check_flags, step, SimpleNamespace(**{flag: val}), None, writes=False)
            ok_plain = raises(Refusal, injected.check_flags, step, SimpleNamespace(
                **{flag: [plain] if flag == 'attempt' else plain}), None, writes=False) is None
            check(f'SOURCE {step} --{flag.replace("_", "-")}: an injected run\'s products refused, an ordinary run\'s read',
                  e is not None and 'holds an injected run' in str(e) and ok_plain, (e, ok_plain))
    kw = [('search', 'replay_attempt', 'published'), ('serve', 'carry_from', 'served'), ('serve', 'carry_from', 'none'),
          ('archive', 'staging_root', 'none'), ('catalog', 'inc_source', 'search')]
    check('keyword values (published, served, none, search) name no path', all(
        raises(Refusal, injected.check_flags, st, SimpleNamespace(**{f: v}), None, writes=False) is None for st, f, v in kw))
    # the deep-omission stage's sources (Q review F7): every run of --omit-runs as the stage reads it - a comma list, an @FILE
    # list, a bare name under paths.SNAPSHOT_REVIEW_ROOT - and the refusal record --omit-refusals
    omit = lambda **a: raises(Refusal, injected.check_flags, 'catalog', SimpleNamespace(**a), None, writes=False)
    e = omit(omit_runs=f'{plain},{src}')
    check('SOURCE catalog --omit-runs RUN,RUN: an injected run anywhere in the comma list refused', e is not None
          and 'holds an injected run' in str(e) and 'catalog --omit-runs' in str(e), e)
    lst_plain, lst_mixed = os.path.join(T, 'omit_runs_plain.txt'), os.path.join(T, 'omit_runs_mixed.txt')
    open(lst_plain, 'w').write(f'# the view\n{plain}\n\n')
    open(lst_mixed, 'w').write(f'{plain}   # first\n{src}\n')
    e, ok = omit(omit_runs='@' + lst_mixed), omit(omit_runs='@' + lst_plain) is None
    check('SOURCE catalog --omit-runs @FILE: each run the file lists is checked (an injected one refused, ordinary ones read)',
          e is not None and 'holds an injected run' in str(e) and ok, (e, ok))
    e = omit(omit_runs='@' + os.path.join(T, 'no_such_list.txt'))
    check('SOURCE catalog --omit-runs @FILE: an unreadable list refuses (its runs cannot be shown ordinary)', e is not None
          and 'cannot be read' in str(e), e)
    saved_rr = CFG.paths.SNAPSHOT_REVIEW_ROOT
    try:
        CFG.paths.SNAPSHOT_REVIEW_ROOT = out
        e = omit(omit_runs='Run_P3S9')
        CFG.paths.SNAPSHOT_REVIEW_ROOT = os.path.dirname(plain)
        ok = omit(omit_runs='Run_P3S9') is None
    finally:
        CFG.paths.SNAPSHOT_REVIEW_ROOT = saved_rr
    check('SOURCE catalog --omit-runs NAME: a bare name resolves under paths.SNAPSHOT_REVIEW_ROOT (as the stage reads it)',
          e is not None and 'holds an injected run' in str(e) and ok, (e, ok))
    e = omit(omit_refusals=os.path.join(out, 'refusals'))
    check('SOURCE catalog --omit-refusals: a refusal record under an injected run refused', e is not None
          and 'catalog --omit-refusals' in str(e), e)
    e = raises(Refusal, injected.check_flags, 'install', SimpleNamespace(gen=tmpl), None, writes=False)
    check('a {cell} template is checked at its fixed part (install --gen <injected out>/gen/{cell})', e is not None, e)
    e = raises(Refusal, injected.check_flags, 'serve', SimpleNamespace(source=f'{me}:{src}'), None, writes=False)
    check('serve --source HOST:PATH on this host is checked locally', e is not None and 'holds an injected run' in str(e), e)
    e = raises(Refusal, injected.check_flags, 'serve', SimpleNamespace(source=f'nohost-exopipe3-test.invalid:{src}'), None,
               writes=False)
    check('serve --source on a host that cannot be reached is refused (the source cannot be shown ordinary)', e is not None
          and 'cannot tell' in str(e), e)
    with Clean():
        injected.mark('test: injected consumer')
        check('an injected run may read an injected run\'s products', raises(
            Refusal, injected.check_flags, 'pages', SimpleNamespace(from_run=src), None, writes=False) is None)
    # the four consumers of the review, end to end (each refuses before the step runs; dry runs)
    for argv, what in ((['serve', '--tier', 'snapshot', '--run', 'Run_SNAPINJ', '--source', src, '--port-offset', '1000'],
                        'serve --source'),
                       (['archive', 'sync', '--attempt', os.path.join(out, 'attempts', '04h_Red_V_20261102T000000Z')],
                        'archive --attempt'),
                       (['install', '--gen', tmpl], 'install --gen'),
                       (['pages', '--tier', 'deep', '--from-run', src], 'pages --from-run'),
                       (['catalog', '--tier', 'snapshot', '--stages', 'omit', '--omit-runs', f'{plain},{src}'],
                        'catalog --omit-runs')):
        r = cli(*argv, '--out', os.path.join(T, 'consumer_out'), '--dry-run', '--cores', 'none')
        check(f'CLI: an ordinary `{what} <injected run\'s products>` exits 2', r.returncode == 2 and what in r.stderr
              and 'holds an injected run' in r.stderr, (r.returncode, r.stderr[-400:]))


def test_dest_flags():
    out = os.path.join(T, 'dest_out')
    inside = os.path.join(out, 'inside')
    outside_existing = os.path.dirname(REPO)          # an existing directory outside every scratch root: never written here
    with Clean():
        injected.begin('test: destinations', out)
        for step, flags in sorted(injected.DEST_FLAGS.items()):
            for flag in flags:
                bad = raises(Refusal, injected.check_flags, step, SimpleNamespace(**{flag: outside_existing}), out)
                good = raises(Refusal, injected.check_flags, step, SimpleNamespace(**{flag: inside}), out)
                remote = raises(Refusal, injected.check_flags, step, SimpleNamespace(**{flag: 'lwacalim07:/data07/x'}), out)
                in_cell = (step, flag) in injected._IN_CELL          # a relative value names a directory inside the cell
                check(f'DEST {step} --{flag.replace("_", "-")}: an existing directory outside scratch refused, '
                      + ('a relative one stays inside the cell' if in_cell else 'a remote one refused')
                      + ', one under --out accepted', bad is not None and (remote is None if in_cell else remote is not None)
                      and good is None and f'{step} --{flag.replace("_", "-")}' in str(bad), (bad, remote, good))
        check('jupiter --qdir: a name inside the site cell (relative) is no free destination',
              raises(Refusal, injected.check_flags, 'jupiter', SimpleNamespace(qdir='quarantine_superseded_x'), out) is None)
        e = raises(Refusal, injected.check_flags, 'forced', SimpleNamespace(deep_panels_run='Run_P3S5_forced9'), out)
        check('forced --deep-panels-run NAME resolves under paths.TRANSIENTS_ROOT and is refused', e is not None
              and CFG.paths.TRANSIENTS_ROOT.rstrip('/') in str(e), e)
        e = raises(Refusal, injected.check_flags, 'archive', SimpleNamespace(stores=f'{inside}/a.csv,{OUTSIDE}/b.csv'), out)
        check('archive --stores: every file of the comma list is checked', e is not None and OUTSIDE in str(e), e)
        e = raises(Refusal, injected.check_flags, 'photometry', SimpleNamespace(stage_root=os.path.join(outside_existing,
                                                                                                       '{cell}')), out)
        check('a {cell} destination template is checked at its fixed part', e is not None, e)
    check('an ordinary run\'s destination flags are not checked here (production writes are gated by PRODUCTION_WRITES)',
          raises(Refusal, injected.check_flags, 'photometry', SimpleNamespace(stage_root=outside_existing), None) is None)
    # the three flags the review named, end to end: refused before the step runs, nothing written there
    for argv, what in ((['photometry', '--tier', 'deep', '--stage-root', outside_existing], 'photometry --stage-root'),
                       (['forced', '--tier', 'snapshot', '--snap-review-root', outside_existing], 'forced --snap-review-root'),
                       (['pages', '--tier', 'sublst', '--sublst-place-target', outside_existing], 'pages --sublst-place-target')):
        before = sorted(os.listdir(outside_existing))
        r = cli(*argv, '--injected', '--out', os.path.join(T, 'dest_cli'), '--dry-run', '--cores', 'none')
        check(f'CLI: `{what} <an existing directory outside scratch> --injected` exits 2, nothing written there',
              r.returncode == 2 and what in r.stderr and injected.LABEL in r.stderr
              and sorted(os.listdir(outside_existing)) == before, (r.returncode, r.stderr[-400:]))
    with Clean():
        e = raises(Refusal, injected.begin, 'test', '/tmp')
        e2 = raises(Refusal, injected.begin, 'test', '/dev/shm')
        check('an injected run\'s --out is never a shared scratch root itself (/tmp, /dev/shm): its start record would mark '
              'every run below as injected', e is not None and e2 is not None and 'shared scratch root' in str(e), (e, e2))


def test_flag_tables_name_real_flags():
    """Every flag of DEST_FLAGS / SOURCE_FLAGS is a destination of that step's real parser (as cli builds it): a renamed flag
    must never drop out of its check silently."""
    import argparse
    import importlib
    from exopipe.phase3 import cli as CLI
    bad = []
    for table in (injected.DEST_FLAGS, injected.SOURCE_FLAGS):
        for step, flags in table.items():
            mod = importlib.import_module(f'exopipe.phase3.steps.{step}')
            sp = argparse.ArgumentParser(prog=f'exopipe3 {step}')
            mod.add_arguments(sp)
            CLI._common(sp, mod)
            dests = {a.dest for a in sp._actions}
            bad += [f'{step} {f}' for f in flags if f not in dests]
    check('every flag the checks name is a real flag of its step (no silent drop-out after a rename)', not bad, bad)

def test_run_submit_gate():
    from exopipe.phase3.steps import run as srun
    with Clean():
        injected.mark('test: inherited')
        e = raises(Refusal, srun.submit_injected_gate, SimpleNamespace(injected=False))
        ok = raises(Refusal, srun.submit_injected_gate, SimpleNamespace(injected=True)) is None
        check('run submit: an inherited EXOPIPE3_INJECTED without --injected is refused; with --injected it is forwarded',
              e is not None and 'pass --injected' in str(e) and ok, (e, ok))
        guard.install([os.path.join(T, 'nested')], log=None, label=injected.LABEL)
        e = raises(Refusal, srun.submit_injected_gate, SimpleNamespace(injected=True))
        check('run submit inside an injected run\'s processes (the guard inherited) is refused: its job records live under '
              'runner.JOBS_ROOT', e is not None and 'JOBS_ROOT' in str(e), e)
    check('an ordinary shell: the gate passes', raises(Refusal, srun.submit_injected_gate, SimpleNamespace(injected=False))
          is None)


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    for t in (test_flag_runner_records, test_refused_roots, test_begin_out_gate_and_guard, test_writer_archive_tree,
              test_writer_photometry_root, test_writer_served_run, test_writer_data_decisions, test_cli_inject_and_inherit,
              test_jupiter_serve_refused, test_injected_records_and_resume, test_source_flags, test_dest_flags,
              test_flag_tables_name_real_flags, test_run_submit_gate):
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
