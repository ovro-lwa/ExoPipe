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
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

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
    takes (a docstring or a message may mention the scripts inside a sentence)."""
    hits = []
    for dp, _, fs in os.walk(os.path.join(REPO, 'exopipe', 'phase3')):
        for f in fs:
            if not f.endswith('.py'):
                continue
            p = os.path.join(dp, f)
            tree = ast.parse(open(p).read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    v = node.value.strip()
                    if v in PATCH_SCRIPTS or any(v.endswith('/' + s) for s in PATCH_SCRIPTS):
                        hits.append(f'{os.path.relpath(p, REPO)}:{node.lineno} {v}')
    check('no code names a patch script as a path to run (ops/p3s5_patch_*)', not hits, hits[:4])


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
    from exopipe.phase3.core import env as ENV
    from exopipe.phase3.lib.status import steps_inplace as IPL
    ENV.legacy()                                          # the in-place scripts' root on sys.path, as the status step has it
    S = IPL.import_quiet('p3s3_sources', env={'P3S_LABEL': CFG.status.SEARCH_LABEL['deep']})
    atts = S.published_attempts(['V', 'I'])
    pat = [a for a in atts if DP.patched_from(a['dir']) is not None]
    by = {st: sum(1 for a in pat if a['stokes'] == st) for st in ('V', 'I')}
    check(f'real: {len(atts)} published attempts, {by["V"]} V + {by["I"]} I patched (each with its source attempt)',
          len(atts) > 800 and by['V'] > 0 and by['I'] > 0 and all(DP.patched_from(a['dir']) != 'unknown' for a in pat), by)


def main():
    try:
        test_no_patch_path()
        test_patched_from()
        test_real_count()
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
