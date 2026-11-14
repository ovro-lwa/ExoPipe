"""The 18 MHz carve-out in the deep search's master verification (search_detect.MASTER_HELD_18MHZ; lib/search/admitted.verify_master).

Gregg: 2026-10-17 R-GREGG-20261017-18MHZ-CARVEOUT-MEANS-RED-FROM-4 "Red can be built without 18 MHz, i.e. using just the 4 other
subbands. This is the only allowed case where wideband data can be built without a full 5-subband set" (Full likewise);
IMPL-R-GREGG-20261017-18MHZ-ENFORCE-GOING-FORWARD "enforce going forward, no rebuild now"; 2026-10-23 "The one exception to the
exclusion rule is 18 MHz". An EXISTING Red / Full master night holding its excluded 18 MHz is never a violation, never refused, never
a reason to rebuild. Until this change only the search step's preflight (gate E3) kept such a night; the engine's own membership gate
verified strictly, so 12h Red passed the preflight and was refused inside the engine.

Checks: the key (ruled 'keep', the ledger IDs; production's strict 'refuse' in the parity file); master_held_18 (a value checked,
the step's environment before the config, an unknown value refused); the search step and the frames engine hand it to the engine
child; the preflight and the engine's gate call the one helper (no second copy of the rule). Real data (read only; skipped without
the co-add root): the installed 12h Red master, holding its excluded 18 MHz on 2025-01-08 (and on 2025-04-05, whose 18 MHz
data_decisions.csv excludes since 2026-11-13; the checks take the held nights from the 'keep' report, not a fixed list) - the engine's
own membership-gate block,
cut from deep_search.run_deep_cell and executed, passes under the default and REFUSES under the parity value; the step's dry run
plans the cell under the default and refuses it at gate E3 under the parity value.
The variable EXOPIPE3_MASTER_HELD_18MHZ carries only the step's own config value to an engine child: an inherited value that differs
refuses in the step (step_held_18: the preflight, engine_env, the frames child_env), in both directions; the preflight judges with
the config value, never the variable; on 12h Red a dry run with a differing inherited value refuses naming it, one that agrees plans.
Run: python tests/test_search_18mhz.py
"""
import csv
import os
import shutil
import subprocess
import sys
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from exopipe.phase3.core import env as _env                          # noqa: E402
_env.legacy()
from exopipe.phase3.config import CFG                                # noqa: E402

T = '/tmp/exopipe3_tests_search_18mhz'
COADD = '/data07/gh/phase3/coadd_sb'
CELL = os.path.join(COADD, '12h_Red')
LEDGER = '/lustre/gh/main/clv15/fix_ledger.csv'
IDS = ('R-GREGG-20261017-18MHZ-CARVEOUT-MEANS-RED-FROM-4', 'IMPL-R-GREGG-20261017-18MHZ-ENFORCE-GOING-FORWARD',
       'CORRECTION-12H-RED-18MHZ-IS-THE-CARVE-OUT-NOT-A-DEEP-ISSUE-20261023')
FAILS, N = [], [0]


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({str(detail)[:500]})'))
    if not ok:
        FAILS.append(name)


class _Env:
    """EXOPIPE3_MASTER_HELD_18MHZ set (or removed) for a block, as an engine child sees it."""
    def __init__(self, value):
        from exopipe.phase3.lib.search import admitted as ADS
        self.k, self.v = ADS.MASTER_18_ENV, value

    def __enter__(self):
        self.old = os.environ.pop(self.k, None)
        if self.v is not None:
            os.environ[self.k] = self.v

    def __exit__(self, *a):
        os.environ.pop(self.k, None)
        if self.old is not None:
            os.environ[self.k] = self.old


def _nights(msgs):
    """The nights ('YYYY-MM-DD') that verify_master's messages name after '12h Red '."""
    return sorted({m.split('12h Red ', 1)[1][:10] for m in msgs if '12h Red ' in m})


def gate_block():
    """The engine's master-membership gate, dedented: from its `if cadence == 'deep' ...` to the 'verified' print."""
    ln = open(os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'search', 'deep_search.py')).read().split('\n')
    i = [k for k, x in enumerate(ln) if x.strip() == "if cadence == 'deep' and os.environ.get('P3_ALLOW_OLD_MASTERS') != '1':"]
    assert len(i) == 1, len(i)
    j = next(k for k in range(i[0], len(ln)) if 'master membership verified against the rulings' in ln[k])
    return textwrap.dedent('\n'.join(ln[i[0]:j + 1]))


def test_config_and_helper():
    from exopipe.phase3.lib.search import admitted as ADS
    sd = CFG.search_detect
    ruled = sd.RULED.get('MASTER_HELD_18MHZ')
    ids = {r['fix_id'] for r in csv.DictReader(open(LEDGER))} if os.path.exists(LEDGER) else set(IDS)
    check('config: search_detect.MASTER_HELD_18MHZ = \'keep\' by default, ruled by the 2026-10-17 / 10-23 ledger rows',
          sd.MASTER_HELD_18MHZ == 'keep' and tuple(ruled) == IDS and set(IDS) <= ids, (sd.MASTER_HELD_18MHZ, ruled))
    par = open(os.path.join(REPO, 'exopipe', 'phase3', 'parity', 'production_20261013.cfg')).read()
    check('parity: production_20261013.cfg sets production\'s strict value (\'refuse\')',
          "search_detect.MASTER_HELD_18MHZ = 'refuse'" in par)
    with _Env(None):
        a = ADS.master_held_18()
    with _Env('refuse'):
        b = ADS.master_held_18()
    try:
        ADS.master_held_18('Keep')
        bad = None
    except RuntimeError as e:
        bad = str(e)
    with _Env('nope'):
        try:
            ADS.master_held_18()
            bad2 = None
        except RuntimeError as e:
            bad2 = str(e)
    CFG.reset()
    CFG.authorize([], out=T)
    CFG.set('search_detect.MASTER_HELD_18MHZ', 'refuse')
    try:
        with _Env(None):
            c = ADS.master_held_18()
    finally:
        CFG.reset()
    check('master_held_18: the config without a step (\'keep\'), the step\'s environment before the config, the config\'s value '
          'when set; an unknown value REFUSES', (a, b, c) == ('keep', 'refuse', 'refuse') and bad and 'REFUSING' in bad
          and bad2 and 'REFUSING' in bad2, (a, b, c, bad, bad2))


def test_wiring():
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.search import deep_step as DST, admitted as ADS
    from exopipe.phase3.lib.frames import deep_engine as DE
    os.makedirs(T, exist_ok=True)
    CFG.reset()
    e0 = DST.engine_env(f'{T}/c', f'{T}/a', f'{T}/f', 'X', f'{T}/ar', f'{T}/tmp')

    def spec():
        return DE.UnitSpec(tier='deep', lst='12h', scope='Red', stokes='I', label='X', attempt=os.path.join(T, 'att'),
                           coadd_root=COADD, archive_root=os.path.join(T, 'archive'), archive_run='Run_TEST', archive_copy=True,
                           work_root=os.path.join(T, 'work'))
    c0 = DE.child_env(spec())
    CFG.authorize([], out=T)
    CFG.set('search_detect.MASTER_HELD_18MHZ', 'refuse')
    try:
        e1 = DST.engine_env(f'{T}/c', f'{T}/a', f'{T}/f', 'X', f'{T}/ar', f'{T}/tmp')
        c1 = DE.child_env(spec())
    finally:
        CFG.reset()
    k = ADS.MASTER_18_ENV
    check('wiring: the search step and the frames engine hand the value to the engine child (default keep; refuse when set)',
          (e0.get(k), c0.get(k), e1.get(k), c1.get(k)) == ('keep', 'keep', 'refuse', 'refuse'),
          (e0.get(k), c0.get(k), e1.get(k), c1.get(k)))
    ds = open(os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'search', 'deep_search.py')).read()
    st = open(os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'search', 'deep_step.py')).read()
    check('one helper: the engine\'s membership gate and the step\'s gate E3 call admitted.verify_master (the step with its own '
          'config value), no keep_18 copy', ds.count('_ADS.verify_master(base, policy=_ADS.policy())') == 1
          and st.count('ADS.verify_master(base, held_18=h18)') == 1 and 'h18 = ADS.step_held_18(c.search_detect.MASTER_HELD_18MHZ)' in st
          and 'keep_18' not in ds and 'keep_18' not in st and '# WAVE2 D-ADMIT, WAVE2 18-MHZ' in ds)


def test_inherited_variable():
    """The variable carries only the step's own config value to an engine child: an inherited one that differs refuses in the
    step (preflight, engine_env, frames child_env), in both directions; the preflight judges with the config value, never the
    variable (the reviewer's case: exported 'refuse' under the default config refused 12h Red at gate E3)."""
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.search import admitted as ADS, deep_step as DST
    from exopipe.phase3.lib.frames import deep_engine as DE
    out = {}
    for inh, cfgv in ((None, 'keep'), ('keep', 'keep'), ('refuse', 'refuse'), ('refuse', 'keep'), ('keep', 'refuse')):
        with _Env(inh):
            try:
                out[(inh, cfgv)] = ADS.step_held_18(cfgv)
            except RuntimeError as e:
                out[(inh, cfgv)] = 'REFUSED' if 'inherited' in str(e) and 'REFUSING' in str(e) else str(e)
    check('step_held_18: the config value when no variable is inherited or it agrees; an inherited value that differs REFUSES, in '
          'both directions (refuse over keep, keep over refuse)',
          out == {(None, 'keep'): 'keep', ('keep', 'keep'): 'keep', ('refuse', 'refuse'): 'refuse', ('refuse', 'keep'): 'REFUSED',
                  ('keep', 'refuse'): 'REFUSED'}, out)
    os.makedirs(T, exist_ok=True)

    def spec():
        return DE.UnitSpec(tier='deep', lst='12h', scope='Red', stokes='I', label='X', attempt=os.path.join(T, 'att'),
                           coadd_root=COADD, archive_root=os.path.join(T, 'archive'), archive_run='Run_TEST', archive_copy=True,
                           work_root=os.path.join(T, 'work'))
    got = {}
    for inh, cfgv in (('refuse', 'keep'), ('keep', 'refuse'), ('keep', 'keep')):
        CFG.reset()
        if cfgv != 'keep':
            CFG.authorize([], out=T)
            CFG.set('search_detect.MASTER_HELD_18MHZ', cfgv)
        try:
            with _Env(inh):
                r = []
                for fn in (lambda: DST.engine_env(f'{T}/c', f'{T}/a', f'{T}/f', 'X', f'{T}/ar', f'{T}/tmp'), lambda: DE.child_env(spec())):
                    try:
                        r.append(fn()[ADS.MASTER_18_ENV])
                    except Refusal as e:
                        r.append('REFUSED' if 'inherited' in str(e) else str(e)[:80])
                got[(inh, cfgv)] = tuple(r)
        finally:
            CFG.reset()
    check('engine_env and the frames child_env: refused for an inherited value that differs (both directions); the config value '
          'handed on when it agrees', got == {('refuse', 'keep'): ('REFUSED', 'REFUSED'), ('keep', 'refuse'): ('REFUSED', 'REFUSED'),
                                              ('keep', 'keep'): ('keep', 'keep')}, got)
    if not os.path.isdir(CELL):
        print(f'SKIP the real 12h Red checks of the inherited variable (no {CELL})')
        return
    with _Env('refuse'):
        vk = ADS.verify_master(CELL, held_18='keep')
        ve = ADS.verify_master(CELL)
    held = _nights(m for m in vk[1] if 'holds its excluded 18 MHz' in m)
    check(f'real 12h Red: verify_master with the step\'s value (\'keep\') ignores a variable, the engine path (no value) reads it '
          f'(its step\'s): its violations are the held nights {held}', not vk[0] and ve[0] and '2025-01-08' in held
          and all('18 MHz excluded but' in v for v in ve[0]) and _nights(ve[0]) == held, (vk[0][:1], held, ve[0][:2]))
    runs = {}
    for name, inh, extra in (('refuse inherited, default config', 'refuse', []),
                             ('keep inherited, config refuse', 'keep', ['--set', 'search_detect.MASTER_HELD_18MHZ=refuse']),
                             ('keep inherited, default config', 'keep', [])):
        o = os.path.join(T, 'dry_inh_' + str(len(runs)))
        env = {k: v for k, v in os.environ.items() if k != ADS.MASTER_18_ENV}
        env.update(PYTHONPATH=REPO, **{ADS.MASTER_18_ENV: inh})
        p = subprocess.run([sys.executable, '-m', 'exopipe.phase3', 'search', '--tier', 'deep', '--cells', '12h_Red', '--stokes', 'I',
                            '--dry-run', '--out', o, '--work', os.path.join(T, 'w'), '--cores', 'none'] + extra,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, cwd=REPO, timeout=900, env=env)
        runs[name] = (p.returncode, p.stdout, os.path.exists(o))
    a, b, c = (runs[k] for k in ('refuse inherited, default config', 'keep inherited, config refuse', 'keep inherited, default config'))
    check('real 12h Red, the step\'s dry run: an inherited variable that differs from the config refuses at the preflight naming it '
          '(both directions), never judged by it at gate E3; one that agrees plans the cell; nothing written',
          a[0] == 2 and 'is inherited' in a[1] and 'membership rulings (gate E3)' not in a[1] and not a[2]
          and b[0] == 2 and 'is inherited' in b[1] and 'membership rulings (gate E3)' not in b[1] and not b[2]
          and c[0] == 0 and 'gate E3 membership rulings: ok' in c[1] and not c[2],
          {k: (v[0], v[1].strip()[-160:]) for k, v in runs.items()})


def test_real_12h_red():
    from exopipe.phase3.lib.search import admitted as ADS
    if not os.path.isdir(CELL):
        print(f'SKIP the real 12h Red master (no {CELL})')
        return
    with _Env('keep'):
        vk, rk = ADS.verify_master(CELL)
    with _Env('refuse'):
        vr, rr = ADS.verify_master(CELL)
    held = _nights(r for r in rk if 'holds its excluded 18 MHz' in r)
    check(f'real 12h Red: verify_master keeps the nights holding their excluded 18 MHz {held} (reported) under \'keep\', and '
          f'they are exactly the violations under \'refuse\'', not vk and '2025-01-08' in held and vr
          and all('18 MHz excluded but' in v for v in vr) and _nights(vr) == held, (vk[:2], rk[:2], vr[:2]))
    code = gate_block()
    res = {}
    for val in ('keep', 'refuse'):
        said = []
        ns = dict(os=os, cadence='deep', base=CELL, lst='12h', scope='Red', print=lambda *a, **k: said.append(' '.join(map(str, a))))
        with _Env(val):
            try:
                exec(code, ns)
                res[val] = ('searched', said)
            except RuntimeError as e:
                res[val] = ('refused', str(e))
    check('real 12h Red, the engine\'s own membership gate (cut from run_deep_cell): passes under the default \'keep\' (the cell is '
          'searched; the held night reported) and REFUSES under the parity value \'refuse\'',
          res['keep'][0] == 'searched' and any('membership verified' in s for s in res['keep'][1])
          and any('2025-01-08' in s and '18 MHz' in s for s in res['keep'][1])
          and res['refuse'][0] == 'refused' and 'REFUSING to search' in res['refuse'][1] and '18 MHz excluded but' in res['refuse'][1],
          {k: (v[0], str(v[1])[-200:]) for k, v in res.items()})
    outs = {}
    for name, extra in (('default', []), ('parity', ['--set', 'search_detect.MASTER_HELD_18MHZ=refuse'])):
        o = os.path.join(T, f'dry_{name}')
        p = subprocess.run([sys.executable, '-m', 'exopipe.phase3', 'search', '--tier', 'deep', '--cells', '12h_Red', '--stokes', 'I',
                            '--dry-run', '--out', o, '--work', os.path.join(T, 'w'), '--cores', 'none'] + extra,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, cwd=REPO, timeout=900,
                           env=dict(os.environ, PYTHONPATH=REPO))
        outs[name] = (p.returncode, p.stdout, os.path.exists(o))
    d, q = outs['default'], outs['parity']
    check('real 12h Red, the step\'s dry run: planned under the default (gate E3 ok), refused at gate E3 under the parity value; '
          'nothing written either way', d[0] == 0 and 'gate E3 membership rulings: ok' in d[1] and not d[2]
          and q[0] == 2 and 'membership rulings (gate E3)' in q[1] and '18 MHz' in q[1] and not q[2],
          (d[0], d[1].strip()[-200:], q[0], q[1].strip()[-300:]))


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        for fn in (test_config_and_helper, test_wiring, test_inherited_variable, test_real_12h_red):
            try:
                fn()
            except Exception as e:
                import traceback
                traceback.print_exc()
                check(f'{fn.__name__} ran to the end', False, f'{type(e).__name__}: {e}')
    finally:
        CFG.reset()
        if not os.environ.get('EXOPIPE3_TEST_KEEP'):
            shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
