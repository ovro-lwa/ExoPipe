"""B-01 (b), Gregg 2026-10-13: "no patch path in the new code; patched Run_P3S4 results stay in Run_P3S5 until the next full
re-search". The branch has no patch path (no code runs ops/p3s5_patch_*), the deep harvest records which published
attempts are patched (harvest_manifest patched / patched_from, from the attempt's own PATCH_RECORD.json), and
`exopipe3 status --patched` lists their cells - the ones a full re-search must redo. Each check fails if a patch script
were wired in, a patched attempt went unrecorded or an unpatched one were flagged, or a listed row disagreed with its
attempt's record.
Run: python tests/test_b01_patched.py
"""
import ast
import csv
import glob
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from exopipe.phase3.core import env as ENV                          # noqa: E402
ENV.legacy()                                                        # the legacy roots and the patch import guard
from exopipe.phase3.core import legacy as L                         # noqa: E402
from exopipe.phase3.lib.catalog import deep_ported as DP           # noqa: E402

T = '/tmp/exopipe3_tests_b01_patched'
FAILS, N = [], [0]
PATCH_SCRIPTS = ('p3s5_patch_cell.py', 'p3s5_patch_runner.sh', 'p3s5_patch_untouched.sh', 'p3s5_repatch_epochgate.py')


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({detail})'))
    if not ok:
        FAILS.append(name)


def test_no_patch_path():
    """No string constant in the branch's code is a patch script's name or path on its own - the form a script to run
    takes (a docstring or a message may mention the scripts inside a sentence) - and no import names a patch module; over
    every Python file of the exopipe package and the parity values file."""
    hits = []
    for dp, _, fs in os.walk(os.path.join(REPO, 'exopipe')):
        for f in fs:
            p = os.path.join(dp, f)
            if f.endswith('.cfg'):
                for i, line in enumerate(open(p), 1):
                    if any(sc.split('.')[0] in line for sc in PATCH_SCRIPTS):
                        hits.append(f'{os.path.relpath(p, REPO)}:{i}')
                continue
            if not f.endswith('.py'):
                continue
            try:
                tree = ast.parse(open(p).read())
            except SyntaxError:                  # a py2 file of an old package: not branch code
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    v = node.value.strip()
                    if v in PATCH_SCRIPTS or any(v.endswith('/' + s) for s in PATCH_SCRIPTS):
                        hits.append(f'{os.path.relpath(p, REPO)}:{node.lineno} {v}')
                names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                         [node.module or ''] + [a.name for a in node.names] if isinstance(node, ast.ImportFrom) else [])
                if any(L.PATCH_NAME.match(n.rsplit('.', 1)[-1]) for n in names if n):
                    hits.append(f'{os.path.relpath(p, REPO)}:{node.lineno} import {names}')
    check('no code names a patch script as a path to run, or imports a patch module (exopipe/**, parity values)', not hits,
          hits[:4])


def test_runtime_refusals():
    """Every in-place entry point refuses a patch path, however it was spelled - and lets an ordinary script through."""
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.status import steps_inplace as IPL
    from exopipe.phase3.lib.calibrate import common as CC
    from exopipe.phase3.lib.catalog import incremental as INC
    from exopipe.phase3.lib.jupiter import common as JC
    ops = os.path.join(T, 'no_such_ops')     # the refusal is by name: never hand the real patch scripts to a runner here
    calls = [('core.legacy.run(script)', lambda: L.run(os.path.join(ops, 'p3s5_patch_cell.py'), ['05h_23MHz'])),
             ('core.legacy.run(bash -c ...)', lambda: L.run(os.path.join(T, 'no_such_bash'), ['-c', 'p3s5_patch_runner.sh cells.txt 4'])),
             ('core.legacy.run(a runner)', lambda: L.run(os.path.join(T, 'no_such_bash'), [os.path.join(ops, 'membership_rebuild_20261007/install_runner.sh')])),
             ('core.legacy.run(concatenated name)', lambda: L.run(os.path.join(ops, 'p3s5_patch_' + 'cell.py'))),
             ('steps_inplace.legacy_script', lambda: IPL.legacy_script('p3s5_patch_untouched.sh')),
             ('steps_inplace.import_quiet', lambda: IPL.import_quiet('p3s5_patch_cell')),
             ('steps_inplace.call_attr', lambda: IPL.call_attr('p3s5_repatch_epochgate', {})),
             ('calibrate.common.import_inplace', lambda: CC.import_inplace('p3s5_patch_cell')),
             ('catalog.incremental._load_module_from', lambda: INC._load_module_from(os.path.join(ops, 'p3s5_patch_cell.py'), 'x')),
             ('steps_inplace.run_path (runpy by path)', lambda: IPL.run_path(os.path.join(ops, 'p3s5_patch_cell.py'))),
             ('jupiter in_place --module (its child imports no exopipe: refused by core.legacy.run)',
              lambda: JC.in_place(None, os.path.join(ops, 'p3s5_patch_cell.py'))),
             ('a plain import (the import guard)', lambda: __import__('p3s5_patch_cell')),
             ('importlib.import_module (the import guard)', lambda: __import__('importlib').import_module('p3s5_repatch_epochgate'))]
    bad = []
    for name, f in calls:
        try:
            f()
            bad.append(f'{name}: not refused')
        except Refusal as e:
            if 'patch path' not in str(e):
                bad.append(f'{name}: {e}')
        except Exception as e:                   # noqa: BLE001
            bad.append(f'{name}: {type(e).__name__}: {e}')
    check(f'every in-place entry point refuses a patch path ({len(calls)} spellings)', not bad, bad)
    from exopipe.phase3.lib.mosaic import steps_worker as MW
    try:
        MW._runpy(os.path.join(ops, 'p3s5_patch_cell.py'), [])
        why = 'not refused'
    except MW._Refused as e:                     # the worker's refusal: its task result says 'refused', not 'error'
        why = str(e)
    except Exception as e:                       # noqa: BLE001
        why = f'{type(e).__name__}: {e}'
    check('the mosaic worker refuses a patch script it would run by path, as its own refusal', 'patch path' in why, why)
    os.makedirs(T, exist_ok=True)
    child = os.path.join(T, 'imports_a_patch.py')
    with open(child, 'w') as f:
        f.write('import p3s5_patch_cell\n')
    res = L.run(child, [], log=os.path.join(T, 'imports_a_patch.log'))
    check('an in-place child that imports a patch module fails there (the children\'s import guard)',
          res.rc != 0 and 'patch path' in (res.tail or ''), (res.rc, (res.tail or '')[-300:]))
    ok = not L.patch_path(os.path.join(ops, 'p3s3_merge.py')) and not L.patch_path('p3s3_sources') \
        and not L.patch_path('/x/patch_lists/dewarped_05h_18MHz.txt') and not L.patch_path(os.path.join(ops, 'p3s5_parents.py'))
    check('an ordinary script, module or file is not a patch path', ok)


def test_patched_from():
    shutil.rmtree(T, ignore_errors=True)
    for name, rec in (('plain', None), ('patched', {'source_attempt': '/x/Run_P3S4/att_00h_23MHz_V'}),
                      ('broken', 'not json'), ('nosource', {'cell': '00h_23MHz'})):
        d = os.path.join(T, name)
        os.makedirs(d)
        if rec is not None:
            with open(os.path.join(d, DP.PATCH_RECORD), 'w') as f:
                f.write(rec if isinstance(rec, str) else json.dumps(rec))
    got = {n: DP.patched_from(os.path.join(T, n)) for n in ('plain', 'patched', 'broken', 'nosource')}
    check('patched_from: no record -> None; a record -> its source attempt; unreadable or without one -> unknown',
          got == {'plain': None, 'patched': '/x/Run_P3S4/att_00h_23MHz_V', 'broken': 'unknown', 'nosource': 'unknown'}, got)
    # the harvest manifest's content test re-derives the flag from the attempt's record
    from exopipe.phase3.lib.catalog import tests as CT
    man = os.path.join(T, 'harvest_manifest.csv')
    for name in ('plain', 'patched'):
        os.makedirs(os.path.join(T, name, 'transients', 'Run_P3S5'))
        open(os.path.join(T, name, 'transients', 'Run_P3S5', 'v_deep_ledger.csv'), 'w').write('x\n')

    def manifest(flag_plain, flag_patched):
        with open(man, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['cell', 'stokes', 'node', 'att', 'dir', 'harvested', 'label', 'patched', 'patched_from'])
            w.writerow(['00h_23MHz', 'V', 'n', 'a1', os.path.join(T, 'plain'), '1', 'P3S5', flag_plain[0], flag_plain[1]])
            w.writerow(['00h_27MHz', 'V', 'n', 'a2', os.path.join(T, 'patched'), '1', 'P3S5', flag_patched[0], flag_patched[1]])
        return CT.harvest_every_published_attempt(man)
    ok = manifest(('0', ''), ('1', '/x/Run_P3S4/att_00h_23MHz_V'))
    bad = manifest(('0', ''), ('0', ''))
    check('harvest manifest test: passes with the flags the records give, fails when a patched attempt is not flagged',
          ok[0] and 'patched Run_P3S4 results' in ok[2] and not bad[0] and 'PATCH_RECORD' in str(bad[2]), (ok, bad))
    stray = manifest(('0', '/x/Run_P3S4/somewhere'), ('1', '/x/Run_P3S4/att_00h_23MHz_V'))
    with open(man, 'w', newline='') as f:                 # a manifest written without the patch columns
        w = csv.writer(f)
        w.writerow(['cell', 'stokes', 'node', 'att', 'dir', 'harvested', 'label'])
        w.writerow(['00h_23MHz', 'V', 'n', 'a1', os.path.join(T, 'plain'), '1', 'P3S5'])
    nocols = CT.harvest_every_published_attempt(man)
    check('harvest manifest test: fails on an unpatched row naming a source, and on a manifest without the patch columns',
          not stray[0] and not nocols[0] and 'missing columns' in str(nocols[2]), (stray, nocols))
    # the status table's content test
    from exopipe.phase3.lib.status import tests as STT
    tab = os.path.join(T, 'patched_attempts_deep.csv')

    def table(rows):
        with open(tab, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['cell', 'lst', 'scope', 'stokes', 'attempt', 'dir', 'patched', 'patched_from'])
            w.writerows(rows)
        return STT.patched_from_own_record(tab)
    good = table([['00h_23MHz', '00h', '23MHz', 'V', 'a1', os.path.join(T, 'plain'), '0', ''],
                  ['00h_27MHz', '00h', '27MHz', 'V', 'a2', os.path.join(T, 'patched'), '1', '/x/Run_P3S4/att_00h_23MHz_V']])
    wrong = table([['00h_23MHz', '00h', '23MHz', 'V', 'a1', os.path.join(T, 'plain'), '1', 'x']])
    check('status table test: each row as its attempt\'s record says; a row that claims a patch its record lacks fails',
          good[0] and good[1] == 1 and not wrong[0], (good, wrong))


def test_real_count():
    """Read only: production's published attempts (p3s5_state) - patched ones counted from their own records."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.status import steps_inplace as IPL
    S = IPL.search_sources(CFG.status.SEARCH_LABEL['deep'])
    atts = S.published_attempts(['V', 'I'])
    pat = [a for a in atts if DP.patched_from(a['dir']) is not None]
    by = {st: sum(1 for a in pat if a['stokes'] == st) for st in ('V', 'I')}
    check(f'real: {len(atts)} published attempts, {by["V"]} V + {by["I"]} I patched (each with its source attempt)',
          len(atts) > 800 and by['V'] > 0 and by['I'] > 0 and all(DP.patched_from(a['dir']) != 'unknown' for a in pat), by)


def test_status_cli():
    """`exopipe3 status --patched` on production's register (read only; the table goes to a scratch --out): --lsts
    selects that LST's cells (it listed nothing before 2026-10-15), --nights / --epochs refuse, the register files are the
    unit's inputs, and the table's content test fails when a row is missing."""
    import subprocess
    out = os.path.join(T, 'status_out')

    def run(*a):
        p = subprocess.run([sys.executable, '-m', 'exopipe.phase3', 'status', '--patched', '--out', out] + list(a), cwd=REPO,
                           env=dict(os.environ), capture_output=True, text=True, timeout=900)
        recs = sorted(glob.glob(os.path.join(out, '_records', 'status', 'report_patched*.json')), key=os.path.getmtime)
        return p.returncode, (json.load(open(recs[-1])) if recs else {}), p.stdout + p.stderr
    rc, rec, txt = run('--lsts', '05h')
    tabs = [o['path'] for o in rec.get('outputs', []) if o.get('class') == 'patched_attempts']
    rows = list(csv.DictReader(open(tabs[0]))) if tabs else []
    check('--lsts 05h: the 05h cells\' published attempts only (done, inputs = the register files)',
          rc == 0 and rows and all(r['lst'] == '05h' for r in rows) and len({r['cell'] for r in rows}) > 1
          and any(p.endswith('published_V.csv') for p in rec.get('inputs', {})), (rc, len(rows), txt[-300:]))
    if tabs:
        from exopipe.phase3.core import validate
        ctx = [o.get('ctx') or {} for o in rec['outputs'] if o['path'] == tabs[0]][0]
        check('the unit records the expanded selection its table was made from (cells, not the spec)',
              ctx.get('cells') and all(c.startswith('05h_') for c in ctx['cells']) and ctx['cells'] == sorted(ctx['cells'])
              and {r['cell'] for r in rows} <= set(ctx['cells']) and 'cells_spec' not in ctx, ctx)
        good = validate.run('patched_attempts', tabs[0], **ctx)
        with open(tabs[0]) as f:
            lines = f.readlines()
        with open(tabs[0], 'w') as f:
            f.writelines(lines[:-1])                                 # one published attempt left out
        bad = validate.run('patched_attempts', tabs[0], **ctx)
        check('the table\'s content test passes as written and fails with a row missing',
              all(t['passed'] for t in good) and not all(t['passed'] for t in bad) and 'missing' in bad[0]['detail'],
              (good, bad))
    rc, rec, txt = run('--nights', '2025-01-01')
    check('--nights refuses (a published attempt holds every night of its cell)',
          rc == 2 and rec.get('status') == 'refused' and '--nights' in rec.get('reason', ''), (rc, rec.get('reason'), txt[-200:]))


def main():
    try:
        test_no_patch_path()
        test_runtime_refusals()
        test_patched_from()
        test_real_count()
        test_status_cli()
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
