#!/usr/bin/env python3
"""STANDING TEST - the per-hour chain runs a new hour end to end without refusing on chain bugs (data-free).

runner.CHAINS['epoch'] (config/shared.py), cutover row 4 and the chain defects found on 2026-11-24:
  1. the parent catalogues of the deep run's latest FINISHED table pass (lib/catalog/deep_records.pass_parent_dir, gate
     P-PASS; keyword `pass` = catalog.PARENT_FROM_PASS): production's pass log (ops/fork_serve_p3s5_20261008.sh 'pass start
     (parents DIR)' ... 'FORK SERVE DONE') or the v2 deep chain's preflight record, the newer pass; a failed, refused or
     unfinished latest pass refuses, a missing snapshot refuses, any other value passes through unchanged
  2. old module names in a core.legacy.run child (core/parity_sitecustomize/sitecustomize.py + core/_shim.py fallback
     finder): with this tree first on PYTHONPATH an old name maps to its new module; a file at the old path still wins;
     an unknown name still fails (PR 161's Q1)
  3. gate R-ROOTS (core/runner.py, runner.CHAIN_REQUIRED_ROOTS): roots resolved from config names at submit; a root that is
     not a directory here or that the site file gives to another host refuses the item before its first step
  4. gate SI-G8 (lib/catalog/incremental_snapshot.gate_forced) accepts the v2 stage-1 evidence: done measure records of
     the cell covering every snapshot band of both kinds; a failed production marker newer than them refuses, an older
     one does not; partial records refuse
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


def test_pass_parent_from_v2_record_and_the_newer_pass():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import record
    from exopipe.phase3.lib.catalog import deep_records as D
    last = CFG.catalog.DEEP_STAGES[-1]
    with Scratch('v2') as d:
        run = os.path.join(d, 'Run_P3S9')
        os.makedirs(run)
        pv, pp = os.path.join(d, 'v2_parents'), os.path.join(d, 'prod_parents')
        os.makedirs(pv)
        os.makedirs(pp)
        pre = record.record_path(run, 'catalog', 'P3S9.preflight')
        _wjson(pre, {'unit': 'P3S9.preflight', 'status': 'done', 'dry_run': False, 'started': '2026-11-20T01:00:00Z',
                     'finished': '2026-11-20T01:05:00Z', 'meta': {'parent_dir': pv}})
        try:
            D.pass_parent_dir(run, 'P3S9')
            raise AssertionError('a v2 pass without its last stage: refused')
        except ValueError as e:
            assert f'P3S9.{last} record missing' in str(e), e
        lst = record.record_path(run, 'catalog', f'P3S9.{last}')
        _wjson(lst, {'unit': f'P3S9.{last}', 'status': 'refused', 'finished': '2026-11-20T03:00:00Z'})
        try:
            D.pass_parent_dir(run, 'P3S9')
            raise AssertionError('a v2 pass whose last stage refused: refused')
        except ValueError as e:
            assert f'P3S9.{last} record refused at 2026-11-20T03:00:00Z' in str(e), e
        _wjson(lst, {'unit': f'P3S9.{last}', 'status': 'done', 'finished': '2026-11-19T03:00:00Z'})   # an older pass's
        try:
            D.pass_parent_dir(run, 'P3S9')
            raise AssertionError('the last stage record of an older pass: refused')
        except ValueError as e:
            assert 'did not finish' in str(e), e
        _wjson(lst, {'unit': f'P3S9.{last}', 'status': 'done', 'finished': '2026-11-20T04:00:00Z'})
        par, how = D.pass_parent_dir(run, 'P3S9')
        assert par == pv and 'v2 preflight record' in how, (par, how)
        # an older finished production pass loses; a newer one wins
        _log(run, [f'[2026-11-13T17:54:14Z] pass start (parents {pp})', '[2026-11-14T02:51:47Z] FORK SERVE DONE'])
        assert D.pass_parent_dir(run, 'P3S9')[0] == pv
        _log(run, [f'[2026-11-21T17:54:14Z] pass start (parents {pp})', '[2026-11-22T02:51:47Z] FORK SERVE DONE'])
        assert D.pass_parent_dir(run, 'P3S9')[0] == pp
        # a dry-run or not-done preflight is no pass
        _log(run, [])
        _wjson(pre, {'unit': 'P3S9.preflight', 'status': 'done', 'dry_run': True, 'finished': '2026-11-20T01:05:00Z',
                     'meta': {'parent_dir': pv}})
        try:
            D.pass_parent_dir(run, 'P3S9')
            raise AssertionError('a dry-run preflight record: no pass')
        except ValueError as e:
            assert 'no table pass' in str(e), e


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
            # the site file gives this host the root: readable
            CFG.site.NODE_LOCAL_ROOTS = dict(old_nl or {}, **{E.host(): [other]})
            assert runner.missing_roots([['paths.C', other]]) == []
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
        # a chain item on a host without a root: refused before its first step, the step never starts
        jobs = os.path.join(d, 'jobs')
        CFG.paths.R_ROOTS_TEST_MISSING = gone
        try:
            with runner_cfg(JOBS_ROOT=jobs, POLL_S=1, CHAINS={'tr': ['notify --epochs {item}']}, CHAIN_FINALLY={},
                            CHAIN_REQUIRED_ROOTS={'tr': ('paths.R_ROOTS_TEST_MISSING',)}, CELL_REFUSAL_CHAINS=('tr',)):
                hour = '04h_2025-10-14'
                job = runner.submit(None, [], [hour], [(socket.gethostname(), 1)], start=False, chain='tr')
                assert runner.load_job(job)['chain_roots'] == [['paths.R_ROOTS_TEST_MISSING', gone]]
                runner.worker(job)
                st = runner.item_state(job, hour)
        finally:
            del CFG.paths.R_ROOTS_TEST_MISSING
        assert st['status'] == 'refused' and st['rc'] == 2 and st['steps'] == [], st
        assert st['refusal'].startswith('R-ROOTS: this host') and gone in st['refusal'], st['refusal']
        assert st['summary'].startswith('REFUSED before step 1 (R-ROOTS'), st['summary']
        log = open(st['log']).read()
        assert 'REFUSED before the first step - R-ROOTS' in log and 'step 1/1' not in log, log


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
        # production's done marker alone still passes (no v2 record)
        shutil.rmtree(os.path.join(ctx.OUT, '_records'))
        open(os.path.join(ctx.c.INC_FORCED_STATE_DIR, 'done', cell), 'w').close()
        assert gate(ctx)[0] == 'pass'
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
         test_old_module_names_in_a_child, test_required_roots_gate, test_si_g8_accepts_v2_stage1_records,
         test_the_per_hour_chain, test_every_chain_step_parses]

if __name__ == '__main__':
    want = set(sys.argv[1:])
    n = 0
    for t in TESTS:
        if want and t.__name__ not in want:
            continue
        t()
        n += 1
        print(f'PASS {t.__name__}', flush=True)
    print(f'ALL {n} PASSED')
