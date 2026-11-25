#!/usr/bin/env python3
"""STANDING TEST - the per-hour chain runs a new hour end to end without refusing on chain bugs (data-free).

runner.CHAINS['epoch'] (config/shared.py), cutover row 4 and the chain defects found on 2026-11-24:
  1. the parent catalogues of the deep run's latest FINISHED table pass (lib/catalog/deep_records.pass_parent_dir, gate
     P-PASS; keyword `pass` = catalog.PARENT_FROM_PASS): production's pass log (ops/fork_serve_p3s5_20261008.sh 'pass start
     (parents DIR)' ... 'FORK SERVE DONE') or the v2 deep chain's unit records, the attempt that started last: a failed,
     refused or unfinished latest attempt refuses (a rerun from a later stage not done, a refused or failed preflight,
     a stage an earlier attempt left refused), never falling back to an older finished pass; a missing snapshot refuses;
     any other value passes through unchanged; catalog --incremental resolves it lazily, under the SOURCE run's own label
  2. old module names in a core.legacy.run child (core/parity_sitecustomize/sitecustomize.py + core/_shim.py fallback
     finder): with this tree first on PYTHONPATH an old name maps to its new module; a file at the old path still wins;
     an unknown name still fails (PR 161's Q1); the fallback finder never stands in for the primary hook
  3. gate R-ROOTS (core/runner.py, runner.CHAIN_REQUIRED_ROOTS): roots resolved from config names at submit, with the job's
     forwarded --config/--set; a submit whose nodes cannot read them refuses; a worker on a host that cannot (not a
     directory here, or the site file gives it to another host) claims nothing and leaves a worker record (an exception in
     the check: an error record); when every lane refused, the supervisor ends the items and runs their closing steps; the
     per-hour driver books only a node with the roots and its stage refuses where a root is not mounted
  4. gate SI-G8 (lib/catalog/incremental_snapshot.gate_forced) accepts the v2 stage-1 evidence: done measure records of
     the cell covering every snapshot band of both kinds; a failed production marker newer than them refuses, an older
     one does not; partial records refuse; production's done marker settles it without reading v2 records; a forced
     fleet's claim refuses before any v2 record is read; each stage-1 root's records are read once
  admission and the parents: the admit stage takes catalog.INSTALL_LOCK on a production tree (waits while a deep table
     pass holds it); after a cell's Master changes, production's parent gate (gate_inputs.parents, called in place by
     every parent-reading step) refuses the old parent catalogue (needs the site's exopipe_ops: skipped elsewhere)
  5. the chain: the three admission steps first, the snapshot photometry between the light curves and the snapshot
     catalog, `pass` on the three parent-reading steps, every step template parses with its step's own arguments

Run standalone:  python tests/test_chain_cutover_fixes.py [test_name ...]
"""
import calendar
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
PY = sys.executable


class Scratch:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix=f'test_chain_cutover_fixes_{self.name}_')
        return self.path

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)


def _wjson(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        json.dump(obj, f)


# ---------------------------------------------------------------------------------------------------------------------
# 1. the parent snapshot of the deep run's latest table pass
# ---------------------------------------------------------------------------------------------------------------------

def _log(run_dir, lines):
    from exopipe.phase3.config import CFG
    with open(os.path.join(run_dir, CFG.catalog.DEEP_PASS_LOG), 'w') as f:
        f.write(''.join(ln + '\n' for ln in lines))


def test_pass_parent_from_production_log():
    from exopipe.phase3.lib.catalog import deep_records as D
    with Scratch('log') as d:
        run = os.path.join(d, 'Run_P3S5')
        os.makedirs(run)
        p1, p2 = os.path.join(d, 'parents_1'), os.path.join(d, 'parents_2')
        os.makedirs(p1)
        try:
            D.pass_parent_dir(run, 'P3S5')
            raise AssertionError('no pass recorded: refused')
        except ValueError as e:
            assert str(e).startswith('P-PASS: no table pass'), e
        # a finished pass, then a failed one: the latest refuses (its tables may be mid-pass)
        _log(run, ['[2026-11-13T04:49:00Z] install lock held',
                   f'[2026-11-13T04:52:02Z] pass start (parents {p1})',
                   '14:53 merge gate REFUSED - pass aborted',                  # not a pass-log line of the launcher
                   '[2026-11-13T10:41:03Z] FORK SERVE DONE'])
        par, how = D.pass_parent_dir(run, 'P3S5')
        assert par == p1 and 'production pass log' in how and '2026-11-13T04:52:02Z' in how, (par, how)
        _log(run, [f'[2026-11-13T04:52:02Z] pass start (parents {p1})', '[2026-11-13T10:41:03Z] FORK SERVE DONE',
                   f'[2026-11-13T17:54:14Z] pass start (parents {p2})',
                   '[2026-11-13T21:49:38Z] PASS FAILED (rc 1; /x/pass_fork.log)', '[2026-11-13T22:00:00Z] FORK SERVE DONE'])
        try:
            D.pass_parent_dir(run, 'P3S5')
            raise AssertionError('a failed latest pass: refused')
        except ValueError as e:
            assert 'did not finish: PASS FAILED (rc 1' in str(e) and p2 in str(e), e
        # the filters refused after the pass: no FORK SERVE DONE
        _log(run, [f'[2026-11-13T17:54:14Z] pass start (parents {p2})',
                   '[2026-11-14T02:40:00Z] REFUSED: scintillation_index failed - no serve'])
        try:
            D.pass_parent_dir(run, 'P3S5')
            raise AssertionError('a refused latest pass: refused')
        except ValueError as e:
            assert 'did not finish: REFUSED: scintillation_index failed' in str(e), e
        # running (no end line yet)
        _log(run, [f'[2026-11-13T17:54:14Z] pass start (parents {p2})'])
        try:
            D.pass_parent_dir(run, 'P3S5')
            raise AssertionError('a running pass: refused')
        except ValueError as e:
            assert 'no end line (running or killed)' in str(e), e
        # finished, but its snapshot is not on this host
        _log(run, [f'[2026-11-13T17:54:14Z] pass start (parents {p2})', '[2026-11-14T02:51:47Z] FORK SERVE DONE'])
        try:
            D.pass_parent_dir(run, 'P3S5')
            raise AssertionError('a missing snapshot: refused')
        except ValueError as e:
            assert 'is not a directory on this host' in str(e) and p2 in str(e), e
        os.makedirs(p2)
        assert D.pass_parent_dir(run, 'P3S5')[0] == p2


def _att(run, label, rid, day, stages, status='done', meta=None, dry=False):
    """Unit records of one attempt ``rid`` of the v2 deep chain: each stage started on 2026-11-<day> at its index hour."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import record
    for st in stages:
        k = CFG.catalog.DEEP_STAGES.index(st.split('.')[0])
        _wjson(record.record_path(run, 'catalog', f'{label}.{st}'),
               {'unit': f'{label}.{st}', 'run_id': rid, 'dry_run': dry,
                'status': status if not isinstance(status, dict) else status.get(st, 'done'),
                'started': f'2026-11-{day}T{k:02d}:00:00Z', 'finished': f'2026-11-{day}T{k:02d}:30:00Z',
                'meta': (meta or {}) if st == 'preflight' else {}})


def _refused(run, label, *want):
    from exopipe.phase3.lib.catalog import deep_records as D
    try:
        got = D.pass_parent_dir(run, label)
    except ValueError as e:
        assert all(w in str(e) for w in want), (want, str(e))
        return str(e)
    raise AssertionError(f'expected P-PASS to refuse ({want}); got {got}')


def test_pass_parent_from_v2_record_and_the_newer_pass():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.catalog import deep_records as D
    S = CFG.catalog.DEEP_STAGES
    last = S[-1]
    with Scratch('v2') as d:
        run = os.path.join(d, 'Run_P3S9')
        os.makedirs(run)
        pv, pp = os.path.join(d, 'v2_parents'), os.path.join(d, 'prod_parents')
        os.makedirs(pv)
        os.makedirs(pp)
        # a dry-run attempt is no attempt
        _att(run, 'P3S9', 'catalog_dry', '19', S, meta={'parent_dir': pv}, dry=True)
        _refused(run, 'P3S9', 'no table pass')
        # attempt A: preflight .. fork (the default stop before forced): the pass has not reached its last stage
        _att(run, 'P3S9', 'catalog_A', '20', S[:S.index('fork') + 1], meta={'parent_dir': pv})
        _refused(run, 'P3S9', 'attempt catalog_A', 'ran from stage preflight', f'no record of the last stage {last}')
        # attempt B: --from union .. list, done: the pass finished; the parents are the preflight's
        _att(run, 'P3S9', 'catalog_B', '21', S[S.index('union'):])
        par, how = D.pass_parent_dir(run, 'P3S9')
        assert par == pv and 'attempt catalog_B' in how and 'pass started 2026-11-21T' in how, (par, how)
        # an older finished production pass loses; a newer one wins
        _log(run, [f'[2026-11-13T17:54:14Z] pass start (parents {pp})', '[2026-11-14T02:51:47Z] FORK SERVE DONE'])
        assert D.pass_parent_dir(run, 'P3S9')[0] == pv
        _log(run, [f'[2026-11-22T00:00:00Z] pass start (parents {pp})', '[2026-11-22T02:51:47Z] FORK SERVE DONE'])
        assert D.pass_parent_dir(run, 'P3S9')[0] == pp
        # ... and a newer production pass that failed refuses: never the older finished v2 pass
        _log(run, [f'[2026-11-22T00:00:00Z] pass start (parents {pp})', '[2026-11-22T02:51:47Z] PASS FAILED (rc 1)'])
        _refused(run, 'P3S9', 'production pass log', 'PASS FAILED (rc 1)')
        _log(run, [f'[2026-11-13T17:54:14Z] pass start (parents {pp})', '[2026-11-14T02:51:47Z] FORK SERVE DONE'])
        # (a) a NEWER rerun from a later stage that is not done refuses: still running (no record of the last stage) ...
        _att(run, 'P3S9', 'catalog_C', '23', ['identity'])
        _refused(run, 'P3S9', 'attempt catalog_C', 'ran from stage identity', f'no record of the last stage {last}')
        # ... or one of its stages refused or failed (even with the last stage done)
        _att(run, 'P3S9', 'catalog_C', '23', S[S.index('identity'):], status={'localize': 'error'})
        _refused(run, 'P3S9', 'attempt catalog_C', 'its records not done: P3S9.localize error')
        # the rerun done through the last stage: finished
        _att(run, 'P3S9', 'catalog_C', '23', S[S.index('identity'):])
        par, how = D.pass_parent_dir(run, 'P3S9')
        assert par == pv and 'attempt catalog_C' in how and 'pass started 2026-11-23T' in how, (par, how)
        # a stage an earlier attempt left refused, under a later rerun that skipped it: refused
        _att(run, 'P3S9', 'catalog_D', '24', ['union'], status='refused')
        _refused(run, 'P3S9', 'attempt catalog_D', 'its records not done: P3S9.union refused')
        _att(run, 'P3S9', 'catalog_E', '25', S[S.index('identity'):])
        _refused(run, 'P3S9', 'attempt catalog_E', 'an earlier attempt left not done: P3S9.union refused (attempt catalog_D)')
        _att(run, 'P3S9', 'catalog_F', '26', S[S.index('union'):])
        assert D.pass_parent_dir(run, 'P3S9')[0] == pv
        # (b) a NEWER preflight that refused (or failed) refuses, though production's log holds an older finished pass
        for st in ('refused', 'error'):
            _att(run, 'P3S9', 'catalog_G', '27', ['preflight'], status=st, meta={'parent_dir': pv})
            _refused(run, 'P3S9', 'attempt catalog_G', 'started 2026-11-27T00:00:00Z',
                     f'its records not done: P3S9.preflight {st}')
        # a newer preflight done without a parent snapshot named: refused
        _att(run, 'P3S9', 'catalog_H', '28', S, meta={})
        _refused(run, 'P3S9', 'attempt catalog_H', 'P3S9.preflight record', 'names no parent snapshot')


def test_resolve_parent_arg():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.catalog import deep_records as D
    kw = CFG.catalog.PARENT_FROM_PASS
    assert kw == 'pass'
    assert D.resolve_parent_arg('/some/dir', '/no/run', 'P3S5') == ('/some/dir', None, None)
    assert D.resolve_parent_arg(None, None, None) == (None, None, None)
    par, how, why = D.resolve_parent_arg(kw, None, 'P3S5')
    assert par is None and why.startswith('P-PASS: --parent-dir pass needs the deep run'), why
    with Scratch('arg') as d:
        run = os.path.join(d, 'Run_P3S7')
        os.makedirs(run)
        p = os.path.join(d, 'parents')
        os.makedirs(p)
        _log(run, [f'[2026-11-13T17:54:14Z] pass start (parents {p})', '[2026-11-14T02:51:47Z] FORK SERVE DONE'])
        par, how, why = D.resolve_parent_arg(kw, run, None)            # the label from Run_<label>
        assert (par, why) == (p, None) and 'pass started 2026-11-13T17:54:14Z' in how, (par, how, why)
        par, how, why = D.resolve_parent_arg(kw, os.path.join(d, 'Run_none'), 'P3S7')
        assert par is None and why.startswith('P-PASS: no table pass'), why


def test_incremental_parents_lazy_by_source_label():
    """catalog --tier deep --incremental --inc-parents pass: the source stays lazy (a phase that reads no parents never
    opens it) and P-PASS reads the SOURCE run's records under its own label (Run_<label>), not --run's."""
    from exopipe.phase3.lib.catalog import incremental as INC
    from exopipe.phase3.config import CFG
    S = CFG.catalog.DEEP_STAGES
    with Scratch('inc') as d:
        src = os.path.join(d, 'Run_OTHER')
        os.makedirs(src)
        pv = os.path.join(d, 'parents_other')
        os.makedirs(pv)
        _att(src, 'OTHER', 'catalog_A', '20', S, meta={'parent_dir': pv})
        run = types.SimpleNamespace(out_root=os.path.join(d, 'out'), dry_run=True, fresh=False)

        def ctx(source):
            return INC._Ctx(types.SimpleNamespace(run_label='P3S5', inc_phase=None, inc_parents='pass', parents=None,
                                                  inc_source=source, inc_rebuild=False), run)
        c = ctx('search')                     # a source spec that refuses when opened: the context is still made
        assert c._source is None
        try:
            c.parents
            raise AssertionError('opening the source refuses')
        except Exception as e:
            assert 'search' in str(e), e
        c = ctx(f'run:{src}')
        assert c._source is None and c._parents is None
        assert c.parents == pv and 'attempt catalog_A' in c.parents_how and c.parents_why is None, (c.parents, c.parents_how)
        # --run's label (P3S5) holds no record there: resolving by it would have refused
        from exopipe.phase3.lib.catalog import deep_records as D
        assert D.resolve_parent_arg('pass', src, 'P3S5')[2].startswith('P-PASS: no table pass')


# ---------------------------------------------------------------------------------------------------------------------
# 2. old module names in a core.legacy.run child
# ---------------------------------------------------------------------------------------------------------------------

def _child(pythonpath, code, cwd):
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    env['PYTHONPATH'] = os.pathsep.join(pythonpath)
    return subprocess.run([PY, '-c', code], env=env, cwd=cwd, capture_output=True, text=True, timeout=300)


def test_old_module_names_in_a_child():
    boot = os.path.join(REPO, 'exopipe', 'phase3', 'core', 'parity_sitecustomize')
    from exopipe.phase3.core import _shim
    old = 'exopipe.phase3.phase3_catalog'
    new = _shim.LEGACY[old]
    with Scratch('child') as d:
        # this tree first (core/runner._child_env), a production module importing the old name (deep_forced_radio.py:156)
        r = _child([boot, REPO], f'import {old} as m, sys; print(m.__name__, m.__file__); '
                                 f'from {old} import gaussian_photometry; '
                                 'print(",".join(getattr(f, "__name__", type(f).__name__) for f in sys.meta_path))', d)
        assert r.returncode == 0, r.stderr[-2000:]
        name, path, finders = r.stdout.split()[:3]
        assert name == new and path.startswith(os.path.join(REPO, 'exopipe')), r.stdout
        finders = finders.split(',')                                     # after the path finder (appended at start-up)
        assert finders.index('_LegacyFallbackFinder') > finders.index('PathFinder'), finders
        assert 'LEGACY-NAMES' not in r.stderr, r.stderr
        # an unknown name still fails; a name outside LEGACY is never mapped
        r = _child([boot, REPO], 'import exopipe.phase3.not_a_module_x', d)
        assert r.returncode != 0 and "No module named 'exopipe.phase3.not_a_module_x'" in r.stderr, r.stderr[-800:]
        # a file at the old path wins (the production package first: the old file is found before the fallback is asked)
        prod = os.path.join(d, 'prod')
        os.makedirs(os.path.join(prod, 'exopipe', 'phase3'))
        for p in (os.path.join(prod, 'exopipe', '__init__.py'), os.path.join(prod, 'exopipe', 'phase3', '__init__.py')):
            open(p, 'w').close()
        with open(os.path.join(prod, 'exopipe', 'phase3', 'phase3_catalog.py'), 'w') as f:
            f.write('WHO = "old file"\n')
        r = _child([boot, prod, REPO], f'import {old} as m; print(m.__name__, m.WHO)', d)
        assert r.returncode == 0 and r.stdout.strip() == f'{old} old file', (r.stdout, r.stderr[-800:])
        # idempotent: a second install (another copy loaded by path) adds nothing
        r = _child([boot, REPO], 'import sys, importlib.util as u; '
                                 f's = u.spec_from_file_location("x_shim", {os.path.join(REPO, "exopipe/phase3/core/_shim.py")!r}); '
                                 'm = u.module_from_spec(s); s.loader.exec_module(m); m.install_legacy_fallback_finder(); '
                                 'print(sum(type(f).__name__ == "_LegacyFallbackFinder" for f in sys.meta_path))', d)
        assert r.returncode == 0 and r.stdout.strip() == '1', (r.stdout, r.stderr[-800:])
        # the fallback (a subclass of the primary hook) never masks the primary: with the fallback installed first, the
        # primary is still put first on sys.meta_path, and installing it again adds nothing
        r = _child([boot, REPO], 'import sys; from exopipe.phase3.core import _shim as S; '
                                 'assert any(type(f).__name__ == "_LegacyFallbackFinder" for f in sys.meta_path); '
                                 'f = S.install_legacy_finder(); g = S.install_legacy_finder(); '
                                 'names = [type(x).__name__ for x in sys.meta_path]; '
                                 'print(type(f).__name__, sys.meta_path[0] is f, g is f, names.count("_LegacyFinder"))', d)
        assert r.returncode == 0 and r.stdout.split() == ['_LegacyFinder', 'True', 'True', '1'], (r.stdout, r.stderr[-800:])


# ---------------------------------------------------------------------------------------------------------------------
# 3. gate R-ROOTS
# ---------------------------------------------------------------------------------------------------------------------

class runner_cfg:
    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        from exopipe.phase3.config import CFG
        self.old = {k: getattr(CFG.runner, k) for k in self.kw}
        for k, v in self.kw.items():
            setattr(CFG.runner, k, v)
        return CFG

    def __exit__(self, *exc):
        from exopipe.phase3.config import CFG
        for k, v in self.old.items():
            setattr(CFG.runner, k, v)


def test_required_roots_gate():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import env as E, runner
    roots = runner.chain_roots('epoch')
    assert [r for r, _ in roots] == list(CFG.runner.CHAIN_REQUIRED_ROOTS['epoch']), roots
    assert dict(roots)['paths.COADD_SB_ROOT'] == CFG.paths.COADD_SB_ROOT
    assert runner.chain_roots('deep-build-cell') == []
    me = E.host()
    with Scratch('roots') as d:
        here = os.path.join(d, 'here')
        os.makedirs(here)
        gone = os.path.join(d, 'not_mounted')
        other = os.path.join(d, 'elsewhere')
        os.makedirs(other)
        old_nl = CFG.site.NODE_LOCAL_ROOTS
        try:
            CFG.site.NODE_LOCAL_ROOTS = dict(old_nl or {}, **{'some-other-host-x': [other]})
            miss = runner.missing_roots([['paths.A', here], ['paths.B', gone], ['paths.C', other]])
            assert [(r, p) for r, p, _ in miss] == [('paths.B', gone), ('paths.C', other)], miss
            assert miss[0][2].startswith('not a directory on this host') and 'some-other-host-x' in miss[1][2], miss
            # another host: only the site file decides (its mounts are not seen from here)
            assert runner.missing_roots([['paths.B', gone], ['paths.C', other]], 'some-other-host-x') == []
            assert [r for r, _, _ in runner.missing_roots([['paths.C', other]], 'a-third-host')] == ['paths.C']
            # the site file gives this host the root: readable
            CFG.site.NODE_LOCAL_ROOTS = dict(old_nl or {}, **{me: [other]})
            assert runner.missing_roots([['paths.C', other]]) == []
            assert runner.missing_roots([['paths.C', other]], 'localhost') == []
        finally:
            CFG.site.NODE_LOCAL_ROOTS = old_nl
        # an unknown config name or a relative value refuses the submit
        for bad, want in ((('paths.NO_SUCH_ROOT',), 'not a config value'), (('runner.POLL_S',), 'not an absolute path')):
            with runner_cfg(CHAIN_REQUIRED_ROOTS={'tr': bad}):
                try:
                    runner.chain_roots('tr')
                    raise AssertionError(bad)
                except SystemExit as e:
                    assert want in str(e), e
        # the job's forwarded --config / --set overrides apply (the steps read the overridden value), the last one wins;
        # this process's config is never changed
        CFG.paths.R_ROOTS_TEST_A = here
        CFG.paths.R_ROOTS_TEST_D = {'deep': here, 'snap': here}
        cfgf = os.path.join(d, 'over.cfg')
        with open(cfgf, 'w') as f:
            f.write(f"paths.R_ROOTS_TEST_A = {other!r}   # moved\npaths.R_ROOTS_TEST_D[deep] = {other!r}\n")
        try:
            with runner_cfg(CHAIN_REQUIRED_ROOTS={'tr': ('paths.R_ROOTS_TEST_A', 'paths.R_ROOTS_TEST_D.deep',
                                                         'paths.R_ROOTS_TEST_D.snap')}):
                assert runner.chain_roots('tr') == [['paths.R_ROOTS_TEST_A', here], ['paths.R_ROOTS_TEST_D.deep', here],
                                                    ['paths.R_ROOTS_TEST_D.snap', here]]
                got = runner.chain_roots('tr', ['--config', cfgf])
                assert got == [['paths.R_ROOTS_TEST_A', other], ['paths.R_ROOTS_TEST_D.deep', other],
                               ['paths.R_ROOTS_TEST_D.snap', here]], got
                got = runner.chain_roots('tr', ['--config', cfgf, '--set', f'paths.R_ROOTS_TEST_A={gone}',
                                                '--set', 'paths.R_ROOTS_TEST_D={"deep": %r, "snap": %r}' % (gone, other),
                                                '--ruling-for', 'X', '--parity'])
                assert got == [['paths.R_ROOTS_TEST_A', gone], ['paths.R_ROOTS_TEST_D.deep', gone],
                               ['paths.R_ROOTS_TEST_D.snap', other]], got
                assert CFG.paths.R_ROOTS_TEST_A == here and CFG.paths.R_ROOTS_TEST_D['deep'] == here
                try:
                    runner.chain_roots('tr', ['--set', 'paths.R_ROOTS_TEST_A=relative/dir'])
                    raise AssertionError('a relative override: refused')
                except SystemExit as e:
                    assert "'relative/dir' is not an absolute path" in str(e), e
        finally:
            del CFG.paths.R_ROOTS_TEST_A, CFG.paths.R_ROOTS_TEST_D
        jobs = os.path.join(d, 'jobs')
        CFG.paths.R_ROOTS_TEST_MISSING = gone
        CFG.paths.R_ROOTS_TEST_HERE = here
        try:
            with runner_cfg(JOBS_ROOT=jobs, POLL_S=1, CHAINS={'tr': ['notify --epochs {item}']}, CHAIN_FINALLY={},
                            CHAIN_REQUIRED_ROOTS={'tr': ('paths.R_ROOTS_TEST_MISSING',)}, CELL_REFUSAL_CHAINS=('tr',)):
                hour = '04h_2025-10-14'
                # a submit whose node list has no node that can read the roots refuses before any job record
                try:
                    runner.submit(None, [], [hour], [(socket.gethostname(), 1)], start=False, chain='tr')
                    raise AssertionError('no node with the roots: refused at submit')
                except SystemExit as e:
                    assert str(e).startswith('refused (R-ROOTS): no node of --nodes') and gone in str(e), e
                assert not os.path.isdir(jobs) or os.listdir(jobs) == [], os.listdir(jobs)
                # ... nor with the forwarded override that moves the root away from a readable one
                with runner_cfg(CHAIN_REQUIRED_ROOTS={'tr': ('paths.R_ROOTS_TEST_HERE',)}):
                    try:
                        runner.submit(None, [], [hour], [(socket.gethostname(), 1)], start=False, chain='tr',
                                      forward=['--set', f'paths.R_ROOTS_TEST_HERE={gone}'])
                        raise AssertionError('the forwarded override moves the root: refused at submit')
                    except SystemExit as e:
                        assert gone in str(e), e
                # a node list with one node that can read the roots (per the site file): the job is made; a worker on
                # this host (the disk not mounted here) claims NOTHING and exits with a worker record
                CFG.site.NODE_LOCAL_ROOTS = dict(old_nl or {}, **{'roots-host-x': [gone]})
                try:
                    job = runner.submit(None, [], [hour], [(me, 1), ('roots-host-x', 1)], start=False, chain='tr')
                finally:
                    CFG.site.NODE_LOCAL_ROOTS = old_nl
                assert runner.load_job(job)['chain_roots'] == [['paths.R_ROOTS_TEST_MISSING', gone]]
                runner.worker(job)
                st = runner.item_state(job, hour)
                assert st == {'status': 'queued', 'attempts': 0}, st            # not claimed, not consumed
                assert os.listdir(os.path.join(runner.job_dir(job), 'claims')) == []
                w = runner.worker_records(job)
                assert len(w) == 1 and w[0]['status'] == 'refused' and w[0]['code'] == 'R-ROOTS', w
                assert gone in w[0]['reason'] and w[0]['host'] == socket.gethostname() and w[0]['lane'] == 0, w
                # one lane of two refused (twice: a lane counts once): the supervisor waits for the other
                wd = os.path.join(runner.job_dir(job), runner.WORKERS_DIR)
                shutil.copy(os.path.join(wd, os.listdir(wd)[0]), os.path.join(wd, 'restarted_worker_0_1.json'))
                assert len(runner.worker_records(job)) == 2
                assert runner.end_unrunnable(job) == 0 and runner.item_state(job, hour)['status'] == 'queued'
                # an exception in the check: an error record, still nothing claimed
                real = runner.missing_roots
                runner.missing_roots = lambda roots, host=None: 1 / 0
                try:
                    runner.worker(job, lane=1)          # another lane (one record per worker process and lane)
                finally:
                    runner.missing_roots = real
                w = runner.worker_records(job)
                assert sorted(r['status'] for r in w) == ['error', 'refused', 'refused'], w
                assert any('ZeroDivisionError' in r['reason'] for r in w), w
                assert runner.item_state(job, hour)['status'] == 'queued'
                assert os.listdir(os.path.join(runner.job_dir(job), 'claims')) == []
                # every lane refused: the supervisor ends the item, naming every lane, and runs its closing steps
                fake = os.path.join(d, 'fakepy')                 # stands in for `python -m exopipe.phase3 fin ...`
                with open(fake, 'w') as f:
                    f.write(f'#!{sys.executable}\nimport sys, json\na = sys.argv[1:]\n'
                            "r = json.load(open(a[a.index('--record') + 1]))\n"
                            f"open({os.path.join(d, 'fin.txt')!r}, 'w').write(r['chain_status'] + ' ' + r['summary'])\n")
                os.chmod(fake, 0o755)
                j = runner.load_job(job)
                j['chain_finally'] = [['fin', '--record', '{record}']]
                j['python'] = fake
                runner._wjson(os.path.join(runner.job_dir(job), 'job.json'), j)
                assert runner.end_unrunnable(job) == 1
                st = runner.item_state(job, hour)
                state = runner.job_state(job)[0]
        finally:
            del CFG.paths.R_ROOTS_TEST_MISSING, CFG.paths.R_ROOTS_TEST_HERE
        assert st['status'] == 'refused' and st['rc'] == 2 and 'steps' not in st, st
        assert st['refusal'].startswith('R-ROOTS: no worker lane of the job can run it') and gone in st['refusal'], st
        assert 'ZeroDivisionError' in st['refusal'], st['refusal']
        assert st['summary'].startswith('REFUSED before any step of this attempt (R-ROOTS'), st['summary']
        assert [f['status'] for f in st['finally']] == ['done'], st['finally']
        fin = open(os.path.join(d, 'fin.txt')).read()
        assert fin.startswith('refused REFUSED before any step'), fin
        log = open(st['log']).read()
        assert 'REFUSED before any step of this attempt - R-ROOTS' in log and 'notify --epochs' not in log, log
        assert state == 'finished', state


def test_driver_books_a_node_with_the_roots():
    """The per-hour driver (core/driver.py) books an hour only on a node that can read the chain's roots (gate R-ROOTS:
    book), and its stage refuses on a node where a root is not mounted (run_stage), before the step runs."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import driver, runner
    with Scratch('drive') as d:
        gone = os.path.join(d, 'not_mounted')
        ps = types.SimpleNamespace(nodes=lambda: ['nodeA', 'nodeB'], node_state=lambda h: 'idle',
                                   free=lambda h, p: 1e15, step_forward=[])
        old_nl, old_sr = CFG.site.NODE_LOCAL_ROOTS, CFG.site.SCRATCH_ROOT
        CFG.paths.R_ROOTS_TEST_MISSING = gone
        try:
            CFG.site.NODE_LOCAL_ROOTS = dict(old_nl or {}, nodeB=[gone])
            CFG.site.SCRATCH_ROOT = dict(old_sr or {}, nodeA='/scratchA', nodeB='/scratchB')
            with runner_cfg(CHAIN_REQUIRED_ROOTS={CFG.driver.CHAIN: ('paths.R_ROOTS_TEST_MISSING',)}):
                h = {'epoch': '04h_2025-10-14', 'job': None}
                node, booking, why = driver.book(ps, {}, h)
                assert node == 'nodeB' and why is None, (node, why)
                CFG.site.NODE_LOCAL_ROOTS = dict(old_nl or {}, nodeC=[gone])
                node, booking, why = driver.book(ps, {}, h)
                assert node is None and 'nodeA: cannot read paths.R_ROOTS_TEST_MISSING' in why and 'nodeB: cannot' in why, why
                # the stage on a node where the root is not mounted (any more): refused before the step, the hour refused
                CFG.site.NODE_LOCAL_ROOTS = old_nl
                jobs = os.path.join(d, 'jobs')
                with runner_cfg(JOBS_ROOT=jobs, CHAINS={CFG.driver.CHAIN: ['notify --epochs {item}']}, CHAIN_FINALLY={}):
                    os.makedirs(gone)                          # mounted at submit
                    job = runner.submit(None, [], [h['epoch']], [(socket.gethostname(), 1)], start=False,
                                        chain=CFG.driver.CHAIN)
                    os.rmdir(gone)                             # gone when the stage runs
                    rc = driver.run_stage(job, h['epoch'], 0)
                    st = runner.item_state(job, h['epoch'])
        finally:
            CFG.site.NODE_LOCAL_ROOTS, CFG.site.SCRATCH_ROOT = old_nl, old_sr
            del CFG.paths.R_ROOTS_TEST_MISSING
        assert rc == 2 and st['status'] == 'refused' and st['steps'][0]['status'] == 'refused', st
        assert st['refusal'].startswith('R-ROOTS: this host') and gone in st['refusal'], st
        assert 'pid' not in st['steps'][0] and 'argv' not in st['steps'][0], st['steps'][0]


# ---------------------------------------------------------------------------------------------------------------------
# admission and the parents: the install lock; a Master that changed refuses its stale parent
# ---------------------------------------------------------------------------------------------------------------------

def test_admit_takes_the_install_lock():
    import fcntl
    import inspect
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.masters import admit_stages as AS, deep_inplace as IP
    src = inspect.getsource(AS.admit_stage)
    assert src.index('install_lease(cd, u.note)') < src.index('admit_inputs(cell, args, co, out)'), 'taken before the nights'
    with Scratch('lease') as d:
        root = os.path.join(d, 'coadd_sb')
        cd = os.path.join(root, '04h_50MHz')
        os.makedirs(cd)
        notes = []
        assert AS.install_lease(cd, notes.append) is None and notes == []        # a copy of the cell: not taken
        lock = os.path.join(d, '.install.lock')
        old = IP.is_production_path, CFG.catalog.INSTALL_LOCK, CFG.paths.OPS_ROOT
        IP.is_production_path = lambda p: root if os.path.abspath(p).startswith(root + '/') else None
        CFG.catalog.INSTALL_LOCK, CFG.paths.OPS_ROOT = '{ops_root}/.install.lock', d
        holder = None
        try:
            # a deep table pass holds the lock: the admission waits for it
            holder = subprocess.Popen([PY, '-c', 'import fcntl, os, sys, time; fd = os.open(sys.argv[1], os.O_RDWR | '
                                       'os.O_CREAT); fcntl.flock(fd, fcntl.LOCK_EX); print("held", flush=True); '
                                       'time.sleep(2)', lock], stdout=subprocess.PIPE, text=True)
            assert holder.stdout.readline().strip() == 'held'
            t0 = time.time()
            fd = AS.install_lease(cd, notes.append)
            waited = time.time() - t0
            assert fd is not None and waited > 1.0, waited
            assert notes[0].startswith(f'waiting for the install lock {lock}') and notes[1] == f'install lock {lock} held'
            probe = os.open(lock, os.O_RDWR)
            try:
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    raise AssertionError('the admission holds the lock')
                except BlockingIOError:
                    pass
                os.close(fd)                                               # released
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(probe)
        finally:
            IP.is_production_path, CFG.catalog.INSTALL_LOCK, CFG.paths.OPS_ROOT = old
            if holder is not None:
                holder.wait(30)


def test_stale_parent_refused_after_a_master_change():
    """The parent read of every parent-reading step of the chain is production's gate_inputs.parents, called in place
    (deep_forced_frames.py:31 for photometry deep, per cell; p3s2_conform.py:295/308 and multiples_fork.py:339/573 for the
    catalog's in-place stages; lib/calibrate/bright_parent._gate_inputs): after a cell's Master changes, its parent
    catalogue (the .gen.json digest of the old Master) is REFUSED, and the catalogue rebuilt on the new Master is
    accepted. Needs the site's production modules (the cluster): returns the reason it cannot run elsewhere."""
    import hashlib
    import numpy as np
    from astropy.io import fits
    from exopipe.phase3.lib.calibrate import common as K
    with Scratch('stale') as d:
        lst, band = '04h', '50MHz'
        coadd, par = os.path.join(d, 'coadd_sb'), os.path.join(d, 'parents')
        cf = os.path.join(coadd, f'{lst}_{band}', 'coadd_final')
        os.makedirs(cf)
        os.makedirs(par)
        m = os.path.join(cf, f'I_deep_Taper_R075_{band}.fits')
        cat = os.path.join(par, f'master_R075_{lst}_{band}.gaul.csv')

        def install(data):                                              # a Master generation and its parent catalogue
            fits.writeto(m, data, overwrite=True)
            with open(cat, 'w') as f:
                f.write('# Gaul_id,...\n0,0,0,0,60.0,0,45.0,0,0,0,%.1f,0\n' % float(np.nanmax(data)))
            return hashlib.md5(np.ascontiguousarray(np.nan_to_num(data)).tobytes()).hexdigest()

        def gate():                                                     # a new step process each time (per-process cache)
            G = K.import_inplace('gate_inputs', env={'P3_PARENT_DIR': par, 'P3_COADD_ROOT': coadd}, reload=True)
            try:
                return len(G.parents(lst, band))
            except RuntimeError as e:
                return str(e)
        old = np.arange(64, dtype=np.float32).reshape(8, 8)
        old[0, 0] = np.nan
        md5_old = install(old)
        with open(cat + '.gen.json', 'w') as f:
            json.dump({'master': os.path.basename(m), 'master_md5_data': md5_old}, f)
        try:
            assert gate() == 1
        except ImportError as e:
            return f'production gate_inputs is not importable here ({e})'
        new = old.copy()
        new[4, 4] += 1.0                                                # the Master changed (a new generation)
        fits.writeto(m, new, overwrite=True)
        why = gate()
        assert isinstance(why, str) and 'stale generation' in why and 'REFUSING' in why, why
        with open(cat + '.gen.json', 'w') as f:                        # the parent rebuilt on the new Master: accepted
            json.dump({'master': os.path.basename(m), 'master_md5_data': install(new)}, f)
        assert gate() == 1
        with open(cat + '.gen.json', 'w') as f:                        # a catalogue of another master: refused
            json.dump({'master': 'I_deep_Taper_R0_50MHz.fits', 'master_md5_data': install(new)}, f)
        assert 'not I_deep_Taper_R075_50MHz.fits' in str(gate())


# ---------------------------------------------------------------------------------------------------------------------
# 4. gate SI-G8 with the v2 stage-1 records
# ---------------------------------------------------------------------------------------------------------------------

def _si_ctx(d):
    from exopipe.phase3.config import CFG
    phot = types.SimpleNamespace(OUT_ROOT={'snapshot': os.path.join(d, 'prod_forced')}, STAGE1_DIR='stage1_gen5')
    return types.SimpleNamespace(c=types.SimpleNamespace(INC_FORCED_STATE_DIR=os.path.join(d, 'state')),
                                 cfg=types.SimpleNamespace(photometry=phot, catalog_snapshot=CFG.catalog_snapshot),
                                 prod=False, OUT=os.path.join(d, 'out'), dry=False)


def _measure_record(root, cell, bands, status='done', finished='2026-11-24T10:00:00Z', suffix=''):
    _wjson(os.path.join(root, '_records', 'photometry', f'measure.{cell}{suffix}.json'),
           {'unit': f'measure.{cell}{suffix}', 'status': status, 'finished': finished, 'meta': {'bands_done': list(bands)}})


def test_si_g8_accepts_v2_stage1_records():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.catalog import incremental_snapshot as IS
    cell = '07h_2025-10-14'
    bands = list(CFG.photometry.SNAPSHOT_BANDS)
    sb = [b for b in bands if 'MHz' in b]
    assert sb and len(sb) < len(bands)

    def gate(ctx):
        lines = []
        try:
            IS.gate_forced(ctx, cell, lines)
            return 'pass', lines
        except Refusal as e:
            return str(e), lines

    with Scratch('sig8') as d:
        ctx = _si_ctx(d)
        os.makedirs(os.path.join(ctx.c.INC_FORCED_STATE_DIR, 'done'))
        os.makedirs(os.path.join(ctx.c.INC_FORCED_STATE_DIR, 'failed'))
        res, lines = gate(ctx)
        assert res.startswith(f'SI-G8: {cell}: forced photometry stage 1 not done (v2 records:'), res
        # the subbands only: still not done (both kinds needed)
        _measure_record(ctx.OUT, cell, sb, suffix='.' + '+'.join(sb))
        res, _ = gate(ctx)
        assert res.startswith('SI-G8') and 'wb bands' in res, res
        # every band under the run's --out (the chain gives photometry and catalog one --out): done
        _measure_record(ctx.OUT, cell, bands)
        res, lines = gate(ctx)
        assert res == 'pass' and 'v2 measure records done for every band' in lines[-1], (res, lines)
        # a failed production marker newer than the records refuses; an older one does not
        fm = os.path.join(ctx.c.INC_FORCED_STATE_DIR, 'failed', cell)
        open(fm, 'w').close()
        os.utime(fm, (time.time(), time.time()))
        res, lines = gate(ctx)
        assert res.startswith(f'SI-G8: {cell}: forced photometry stage 1 marked failed') and 'after the v2 measure' in res, res
        old = calendar.timegm(time.strptime('2026-11-01T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ'))
        os.utime(fm, (old, old))
        res, lines = gate(ctx)
        assert res == 'pass' and 'newer than the failed marker' in lines[-1], (res, lines)
        os.remove(fm)
        # a not-done record of the full set: the cell is not complete
        _measure_record(ctx.OUT, cell, bands, status='refused')
        res, _ = gate(ctx)
        assert res.startswith('SI-G8') and 'not done' in res, res
        # the production root's records count too; on the production root ctx.OUT is not searched
        _measure_record(ctx.cfg.photometry.OUT_ROOT['snapshot'], cell, bands)
        assert gate(ctx)[0] == 'pass'
        shutil.rmtree(os.path.join(ctx.cfg.photometry.OUT_ROOT['snapshot'], '_records'))
        _measure_record(ctx.OUT, cell, bands)
        ctx.prod = True
        assert gate(ctx)[0].startswith('SI-G8')
        ctx.prod = False
        # each stage-1 root's records are read once (both kinds and the end time from one read)
        from exopipe.phase3.lib.photometry import steps_snapshot as SS
        real_mr, reads = SS.measure_records, []
        SS.measure_records = lambda st1, c: reads.append(st1) or real_mr(st1, c)
        try:
            assert gate(ctx)[0] == 'pass'
        finally:
            SS.measure_records = real_mr
        assert len(reads) == len(set(reads)) == 2, reads              # the production root (none there), then --out
        # a forced fleet's claim on the cell refuses BEFORE any v2 evidence is read (the fleet is measuring the cell now)
        real_ev = IS.forced_v2_evidence
        IS.forced_v2_evidence = lambda *a: (_ for _ in ()).throw(AssertionError('v2 evidence read'))
        claim = os.path.join(ctx.c.INC_FORCED_STATE_DIR, 'claims', cell)
        try:
            os.makedirs(claim)
            res, lines = gate(ctx)
            assert res.startswith(f'SI-G8: {cell}: forced photometry stage 1 not done (a forced fleet holds it: {claim}; '
                                  'v2 evidence not read while it runs)'), res
            open(fm, 'w').close()                                        # failed and held: refused as marked failed
            res, _ = gate(ctx)
            assert res.startswith(f'SI-G8: {cell}: forced photometry stage 1 marked failed (a forced fleet holds it'), res
            os.remove(fm)
            # production's done marker settles the gate: no v2 evidence read (a claim beside it included)
            open(os.path.join(ctx.c.INC_FORCED_STATE_DIR, 'done', cell), 'w').close()
            assert gate(ctx)[0] == 'pass'
            os.rmdir(claim)
            shutil.rmtree(os.path.join(ctx.OUT, '_records'))
            # production's done marker alone still passes (no v2 record)
            assert gate(ctx)[0] == 'pass'
        finally:
            IS.forced_v2_evidence = real_ev
        # dry run: planned, never refused
        os.remove(os.path.join(ctx.c.INC_FORCED_STATE_DIR, 'done', cell))
        ctx.dry = True
        res, lines = gate(ctx)
        assert res == 'pass' and 'not done' in lines[-1], (res, lines)


# ---------------------------------------------------------------------------------------------------------------------
# 5. the chain
# ---------------------------------------------------------------------------------------------------------------------

def test_the_per_hour_chain():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import runner
    steps = runner.chain_steps('epoch')
    label = [' '.join(a[:a.index('--epochs')]) for a in steps]
    assert label[:4] == ['masters --tier deep --stages admit-prep', 'flow --tier deep --stages admit',
                         'masters --tier deep --stages admit', 'search --tier deep'], label[:4]
    i = label.index('curves --tier snapshot')
    assert label[i + 1:i + 3] == ['photometry --tier snapshot', 'catalog --tier snapshot --incremental'], label[i:i + 3]
    kw = CFG.catalog.PARENT_FROM_PASS
    want = {'photometry --tier deep': '--parent-dir', 'catalog --tier deep': '--inc-parents', 'pages --run': '--deep-parent-dir'}
    for head, flag in want.items():
        hit = [a for a in steps if ' '.join(a).startswith(head)]
        assert len(hit) == 1 and hit[0][hit[0].index(flag) + 1] == kw, (head, hit)
    assert 'CHAIN_REQUIRED_ROOTS' in dir(CFG.runner) and runner.chain_roots('epoch')


def test_every_chain_step_parses():
    """Each step template of the per-hour chain (and its closing step), with {item} and the chain variables substituted,
    is accepted by its step's own argument parser (as cli.main builds it): no unknown flag, no usage error (exit 1)."""
    import argparse
    import importlib
    from exopipe.phase3 import cli
    from exopipe.phase3.core import runner
    item = '07h_2025-10-14'
    for argv in runner.chain_steps('epoch') + runner.chain_finally('epoch'):
        a = [x.replace('{item}', item).replace('{record}', '/dev/null') for x in runner._subst_vars(argv)]
        name, rest = a[0], a[1:]
        mod = importlib.import_module(f'exopipe.phase3.steps.{name}')
        sp = argparse.ArgumentParser(prog=f'exopipe3 {name}')
        mod.add_arguments(sp)
        cli._common(sp, mod)
        try:
            sp.parse_args(rest + ['--dry-run'])
        except SystemExit as e:
            raise AssertionError(f'{shlex.join(a)}: usage error (exit {e.code})')


TESTS = [test_pass_parent_from_production_log, test_pass_parent_from_v2_record_and_the_newer_pass, test_resolve_parent_arg,
         test_incremental_parents_lazy_by_source_label, test_old_module_names_in_a_child, test_required_roots_gate,
         test_driver_books_a_node_with_the_roots, test_admit_takes_the_install_lock,
         test_stale_parent_refused_after_a_master_change, test_si_g8_accepts_v2_stage1_records, test_the_per_hour_chain,
         test_every_chain_step_parses]

if __name__ == '__main__':
    want = set(sys.argv[1:])
    n, skipped = 0, []
    for t in TESTS:
        if want and t.__name__ not in want:
            continue
        why = t()
        n += 1
        if why:                               # a test that needs the cluster says why it could not run here
            skipped.append(t.__name__)
            print(f'SKIP {t.__name__}: {why}', flush=True)
        else:
            print(f'PASS {t.__name__}', flush=True)
    print(f'ALL {n - len(skipped)} PASSED' + (f', {len(skipped)} SKIPPED (cluster only): {", ".join(skipped)}'
                                               if skipped else ''))
