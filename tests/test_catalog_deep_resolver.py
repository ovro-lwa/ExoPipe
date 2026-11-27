#!/usr/bin/env python3
"""Data-free tests: the deep catalog chain takes each (cell, Stokes)'s published attempt from THE shared resolver of the deep
omission (deep_omission.published_attempts; cutover row 64 step 2, register gap option (b)).

The v2 deep search registers an attempt in its own out root (<collect root>/published_<S>.csv, deep_step.register_row) and
never in production's register (paths.OPS_ROOT / search_run.REGISTER, p3s5_state/published_<S>.csv), which no
production-writes allowance covers. The chain's harvest therefore writes the resolver's attempts into the chain's OWN state
(deep_ported.resolved_state under <run>/_state/), which production's p3s3_sources.py / p3s3_merge.py read (P3S3_STATE_DIR).
Asserts:
  (a) a v2 attempt registered ONLY in a collect root's register is the published attempt, and the written state names it
      under production's last-row rule (p3s3_sources.published_attempts: the live script when the cluster has it, else
      its rule restated here);
  (b) production's attempt, registered only in production's state, is still found when no register names a newer one (an
      OLDER v2 attempt does not displace it), and a cell of production's state alone is unchanged;
  (c) two registers naming different attempts with the same newest UTC (a tie) and a published attempt on no disk are not
      settled (the chain's harvest refuses them);
  (d) production's state is never written (bytes before = after), and the chain's state is under its run directory in
      production too (DeepRun.state_dir);
  (e) the chain and the omission take the same attempt (deep_omission.published_attempt per cell);
  (f) an attempt outside every catalog.COLLECT_ROOTS root (production's p3s3_sources.py could not name it) refuses.

Scratch: a fresh directory under $EXOPIPE3_TEST_SCRATCH, else the system temp directory, removed at exit.
Run: python tests/test_catalog_deep_resolver.py [test names]   (exit 0 = all passed, 1 = a failure, 2 = imports missing)
"""
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

_IMPORT_ERROR = None
try:
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.catalog import deep_chain as DC
    from exopipe.phase3.lib.catalog import deep_omission as DO
    from exopipe.phase3.lib.catalog import deep_ported as DP
except Exception as e:  # noqa: BLE001
    _IMPORT_ERROR = e
    if __name__ != '__main__':
        raise

LABEL = 'P3S5'
LOW = LABEL.lower()
PROD_SOURCES = '/lustre/gh/main/exopipe_ops/p3s3_sources.py'     # production's module, run (never edited) when present
TMP = None
OLD, NEW = '20261107T012101', '20261127T153810'                  # an 11-07 fleet attempt; a v2 re-search attempt


def _tmp():
    global TMP
    if TMP is None:
        TMP = tempfile.mkdtemp(prefix='catalog_deep_resolver_', dir=os.environ.get('EXOPIPE3_TEST_SCRATCH') or None)
    return tempfile.mkdtemp(dir=TMP)


class Patch:
    """Set config attributes for a block, restore after."""

    def __init__(self, *items):
        self.items, self.saved = items, []

    def __enter__(self):
        for obj, k, v in self.items:
            self.saved.append((obj, k, getattr(obj, k)))
            setattr(obj, k, v)
        return self

    def __exit__(self, *a):
        for obj, k, v in reversed(self.saved):
            setattr(obj, k, v)


class Run:
    """The DeepRun methods under test on a minimal run (no chain, no records)."""
    prod_state_dir = DC.DeepRun.prod_state_dir if _IMPORT_ERROR is None else None
    state_dir = DC.DeepRun.state_dir if _IMPORT_ERROR is None else None
    published_by_resolver = DC.DeepRun.published_by_resolver if _IMPORT_ERROR is None else None

    def __init__(self, U, cells=None):
        self.cfg, self.label, self.low, self.U = CFG, LABEL, LOW, U
        self.filtered, self._cells = cells is not None, cells

    def keep_cell(self, cell):
        return cell in self._cells


def world():
    """A collect root with two node trees and production's state: (root dir, collect root, production state dir, patches)."""
    d = _tmp()
    coll = os.path.join(d, 'collect')
    state = os.path.join(d, 'ops', f'{LOW}_state')
    os.makedirs(state)
    patches = ((CFG.paths, 'OPS_ROOT', os.path.join(d, 'ops')),
               (CFG.search_run, 'COLLECT_ROOT', os.path.join(coll, f'{LOW}_collect_{{host}}')),
               (CFG.search_run, 'COLLECT_ROOTS', (coll,)), (CFG.search_run, 'COLLECT_LABEL', LOW),
               (CFG.search_run, 'REGISTER', f'{LOW}_state/published_{{stokes}}.csv'),
               (CFG.catalog, 'COLLECT_ROOTS', (coll,)))
    for node in ('nodeA', 'nodeB'):
        os.makedirs(os.path.join(coll, f'{LOW}_collect_{node}', 'attempts'))
    return d, coll, state, patches


def attempt(coll, node, cell, st, att):
    p = os.path.join(coll, f'{LOW}_collect_{node}', 'attempts', f'{cell}_{st}_{att}', 'transients', f'Run_{LABEL}')
    os.makedirs(p)
    open(os.path.join(p, 'v_deep_ledger.csv'), 'w').write('lst\n')
    return os.path.dirname(os.path.dirname(p))


def register(path, *rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.writelines(f'{c},{h},{t}\n' for c, h, t in rows)


def snapshot(d):
    return {os.path.relpath(os.path.join(dp, fn), d): open(os.path.join(dp, fn), 'rb').read()
            for dp, _dn, fs in os.walk(d) for fn in fs}


def sources(state, coll):
    """{(cell, S): dir} as production's p3s3_sources.py names them from ``state``: the live module when the cluster has it,
    else its rule (p3s3_sources.py published_attempts + attempt_dir: the LAST row per (cell, Stokes); the first collect root
    holding <label>_collect_<node>/attempts/<cell>_<S>_<att>)."""
    if os.path.exists(PROD_SOURCES):
        env = dict(os.environ, P3S_LABEL=LABEL, P3S3_STATE_DIR=state, P3S3_COLLECT_ROOTS=coll)
        r = subprocess.run([sys.executable, PROD_SOURCES, 'V', 'I'], env=env, capture_output=True, text=True, check=True)
        return {(a['cell'], a['stokes']): a['dir'] for a in DP.parse_sources_output(r.stdout, LABEL, [coll])}
    seen = {}
    for st in ('V', 'I'):
        p = os.path.join(state, f'published_{st}.csv')
        if os.path.exists(p):
            for line in open(p):
                if line.strip():
                    cell, node, att = line.strip().split(',')[:3]
                    seen[(cell, st)] = os.path.join(coll, f'{LOW}_collect_{node}', 'attempts', f'{cell}_{st}_{att}')
    return seen


def test_newest_attempt_by_the_shared_resolver_through_the_chains_state():
    d, coll, prod, patches = world()
    with Patch(*patches):
        old_a = attempt(coll, 'nodeA', '00h_18MHz', 'V', OLD)            # the 11-07 fleet attempt (production's register)
        new_a = attempt(coll, 'nodeA', '00h_18MHz', 'V', NEW)            # its v2 re-search (the collect root's register only)
        only_prod = attempt(coll, 'nodeB', '00h_18MHz', 'I', OLD)
        older_v2 = attempt(coll, 'nodeA', '00h_18MHz', 'I', '20261101T000000')
        register(os.path.join(prod, 'published_V.csv'), ('00h_18MHz', 'nodeA', OLD))
        register(os.path.join(prod, 'published_I.csv'), ('00h_18MHz', 'nodeB', OLD))
        register(os.path.join(coll, f'{LOW}_collect_nodeA', 'published_V.csv'), ('00h_18MHz', 'nodeA', NEW))
        register(os.path.join(coll, f'{LOW}_collect_nodeA', 'published_I.csv'), ('00h_18MHz', 'nodeA', '20261101T000000'))
        before = snapshot(prod)
        run = Run(os.path.join(d, 'out', f'Run_{LABEL}'))
        pub, bad = run.published_by_resolver()
        assert not bad, bad
        assert pub['V']['00h_18MHz'] == ('nodeA', NEW, new_a), pub['V']                        # (a)
        assert pub['I']['00h_18MHz'] == ('nodeB', OLD, only_prod), pub['I']                    # (b)
        assert older_v2 != only_prod and os.path.isdir(old_a)
        roots = DO.search_roots(None)
        for st, m in pub.items():                                                              # (e)
            for cell, v in m.items():
                assert DO.published_attempt(cell, st, roots) == (v[2], ''), (cell, st)
        state = run.state_dir()
        assert state == os.path.join(run.U, '_state', f'{LOW}_state') and state != run.prod_state_dir()
        n, added = DP.resolved_state(run.prod_state_dir(), state, {st: {c: v[:2] for c, v in m.items()}
                                                                   for st, m in pub.items()})
        assert added == {'V': 1, 'I': 0}, added
        got = sources(state, coll)
        assert got == {(c, st): v[2] for st, m in pub.items() for c, v in m.items()}, got     # (a), (b) as production reads it
        assert open(os.path.join(state, 'published_V.csv')).read().splitlines() == [
            f'00h_18MHz,nodeA,{OLD}', f'00h_18MHz,nodeA,{NEW}'], 'the history and the appended row'
        assert snapshot(prod) == before, "production's state was written"                    # (d)


def test_production_run_never_takes_productions_state_as_its_own():
    d, coll, prod, patches = world()
    with Patch(*patches):
        run = Run('/fast/gh/main/phase3/Run_P3S5')                       # a production run directory (nothing written here)
        assert run.state_dir() == f'/fast/gh/main/phase3/Run_P3S5/_state/{LOW}_state'
        assert not run.state_dir().startswith(CFG.paths.OPS_ROOT), run.state_dir()


def test_a_tie_and_a_missing_attempt_are_not_settled():
    d, coll, prod, patches = world()
    with Patch(*patches):
        attempt(coll, 'nodeA', '01h_18MHz', 'V', OLD)
        attempt(coll, 'nodeB', '01h_18MHz', 'V', OLD)                    # another attempt, the same UTC
        register(os.path.join(prod, 'published_V.csv'), ('01h_18MHz', 'nodeA', OLD), ('02h_18MHz', 'nodeA', OLD))
        register(os.path.join(coll, f'{LOW}_collect_nodeB', 'published_V.csv'), ('01h_18MHz', 'nodeB', OLD))
        pub, bad = Run(os.path.join(d, 'out')).published_by_resolver()
        assert '01h_18MHz' not in pub['V'] and '02h_18MHz' not in pub['V'], pub
        assert len(bad) == 2 and all('is not settled' in b for b in bad), bad
        tie = next(b for b in bad if '01h_18MHz' in b)
        assert f'{LOW}_collect_nodeA' in tie and f'{LOW}_collect_nodeB' in tie, tie
        assert 'not on disk' in next(b for b in bad if '02h_18MHz' in b)
        # the cell filter: a filtered run resolves only its cells (here none of the unsettled ones)
        pub, bad = Run(os.path.join(d, 'out'), cells={'03h_18MHz'}).published_by_resolver()
        assert not bad and not pub['V'], (pub, bad)


def test_an_attempt_production_cannot_name_refuses():
    d, coll, prod, patches = world()
    with Patch(*patches):
        other = os.path.join(d, 'elsewhere', f'{LOW}_collect_nodeC')     # a search root outside catalog.COLLECT_ROOTS
        os.makedirs(os.path.join(other, 'attempts', f'00h_18MHz_V_{NEW}'))
        register(os.path.join(other, 'published_V.csv'), ('00h_18MHz', 'nodeC', NEW))
        with Patch((CFG.search_run, 'COLLECT_ROOT', os.path.join(d, '*', f'{LOW}_collect_{{host}}'))):
            try:
                Run(os.path.join(d, 'out')).published_by_resolver()
            except Refusal as e:
                assert 'cannot name it' in str(e), e
            else:
                raise AssertionError('an attempt outside catalog.COLLECT_ROOTS was accepted')


def test_the_resolvers_production_register_must_be_the_chains_state():
    d, coll, prod, patches = world()
    with Patch(*patches), Patch((CFG.search_run, 'REGISTER', 'p3s9_state/published_{stokes}.csv')):
        try:
            Run(os.path.join(d, 'out')).published_by_resolver()
        except Refusal as e:
            assert 'would not take the same attempts' in str(e), e
        else:
            raise AssertionError('a resolver reading another production register was accepted')


ORDER = ['test_newest_attempt_by_the_shared_resolver_through_the_chains_state',
         'test_production_run_never_takes_productions_state_as_its_own', 'test_a_tie_and_a_missing_attempt_are_not_settled',
         'test_an_attempt_production_cannot_name_refuses', 'test_the_resolvers_production_register_must_be_the_chains_state']


def main(names):
    if _IMPORT_ERROR is not None:
        print(f'imports missing: {_IMPORT_ERROR}')
        return 2
    fails = 0
    try:
        for n in names or ORDER:
            try:
                globals()[n]()
                print(f'PASS {n}')
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print(f'FAIL {n}: {e}')
    finally:
        if TMP:
            shutil.rmtree(TMP, ignore_errors=True)
    print(f'{len(names or ORDER) - fails} of {len(names or ORDER)} passed'
          + (f' (production\'s p3s3_sources.py run live: {PROD_SOURCES})' if os.path.exists(PROD_SOURCES) else
             ' (production\'s p3s3_sources.py rule restated: the module is not on this machine)'))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
