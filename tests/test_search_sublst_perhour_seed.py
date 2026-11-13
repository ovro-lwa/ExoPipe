#!/usr/bin/env python3
"""STANDING TEST - the seed of a per-hour (one night) 10-minute search: `exopipe3 search --tier sublst --epochs <LST>_<date>`.

The per-hour mode copies the published cell's cumulative records into the new attempt, the engine appends the night's rows,
and the publish copies the attempt's files over the published ones. Two records were missing from the seed (found
2026-11-11):

  size_rule.json            the size / axis rule the published ledger was searched under. Without it the engine reads the
                            seeded ledger as production's rule and refuses every cell searched under the envelope rule
                            ('one rule per run', lib/search/size_rule.write_run_record).
  search_table_record.csv   every verdict of the search's bright-source table. Without it the publish left the cell with
                            the one night's rows.

  seed      the seed copies both, byte for byte, with the five items it copied before; items the published cell lacks are
            skipped; a second seed changes nothing
  rule      with the seed the engine's own check (write_run_record) accepts the attempt and without the record it refuses:
            the defect and its repair, on the engine's function
  append    the night's rows appended to the seeded table record and published leave BOTH nights' rows in the cell
  header    the preflight gate refuses a published table record with another column set, exactly where the engine's own
            append (deep_search._append_locked, run here from its source) would refuse at the END of the search
  wiring    detect() seeds through seed_attempt, and the per-hour plan holds the gate after the size-rule gate

Synthetic files only, each test in its own temporary directory. Run standalone:
    python tests/test_search_sublst_perhour_seed.py [test_name ...]
"""
import ast
import csv
import fcntl
import filecmp
import os
import shutil
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from exopipe.phase3.lib.search import size_rule as SZ          # noqa: E402
from exopipe.phase3.lib.search import tests as ST              # noqa: E402
from exopipe.phase3.lib.search import tier_sublst as TS        # noqa: E402

STEP = os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'search', 'tier_sublst.py')
ENGINE = os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'search', 'deep_search.py')
OLD_SEED = ('master_catalog.csv', 'v_deep_ledger.csv', 'v_deep_images.csv', 'sigstack', 'refusals')
NEW_SEED = (SZ.RUN_RECORD, 'search_table_record.csv')


def _env(md5='e184ac13f6eb0123456789abcdef0123'):
    """Stand-in for the size rule's envelope product: what size_rule.run_record reads of it."""
    return types.SimpleNamespace(source='/nowhere/size_envelope_10min.json', md5=md5, full_band='off')


def _table_rows(date, n=3):
    return [['03h', '73MHz', 'V', date, f'{45.0 + i:.4f}', '30.0000', str(100 + i), '200', '7.5', 'kept', '']
            + [''] * (len(ST.STR_COLS) - 11) for i in range(n)]


def _published(root, table=True, record=True, header=None):
    """A published run directory of one cell searched under the envelope rule, holding night 2024-12-19."""
    rd = os.path.join(root, 'pub', 'transients', 'Run_SLSTB_03h_73MHz_V')
    os.makedirs(os.path.join(rd, 'sigstack'))
    os.makedirs(os.path.join(rd, 'refusals'))
    for name, text in (('master_catalog.csv', 'source_id,date\ns1,2024-12-19\n'),
                       ('v_deep_ledger.csv', 'lst,scope,stokes,date,final\n03h,73MHz,V,2024-12-19,1\n'),
                       ('v_deep_images.csv', 'lst,scope,stokes,date,n_final\n03h,73MHz,V,2024-12-19,1\n'),
                       (os.path.join('sigstack', 'sigstack_03h_73MHz_V_w0.fits'), 'stand-in for a sigma-stack map\n'),
                       (os.path.join('refusals', '03h_73MHz_V_night_excluded.csv'), 'date,why\n')):
        with open(os.path.join(rd, name), 'w') as f:
            f.write(text)
    if record:
        SZ.write_run_record(os.path.join(root, 'rec_src'), _env())      # the engine's own writer, in an empty directory
        shutil.copy2(os.path.join(root, 'rec_src', SZ.RUN_RECORD), os.path.join(rd, SZ.RUN_RECORD))
    if table:
        with open(os.path.join(rd, 'search_table_record.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header or ST.STR_COLS)
            w.writerows(_table_rows('2024-12-19') if header is None else [['x'] * len(header)])
    return rd


def _engine_function(name):
    """A function of the engine, compiled from its source alone (importing the engine reads the site configuration)."""
    tree = ast.parse(open(ENGINE).read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(fn) == 1, f'{name}: {len(fn)} definitions in {ENGINE}'
    ns = dict(csv=csv, fcntl=fcntl, os=os)
    exec(compile(ast.Module(body=fn, type_ignores=[]), ENGINE, 'exec'), ns)
    return ns[name]


def _engine_literal(name):
    tree = ast.parse(open(ENGINE).read())
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return ast.literal_eval(n.value)
    raise AssertionError(f'{name} not assigned at module level in {ENGINE}')


def test_seed_names_hold_the_two_records():
    names = TS.per_hour_seed_names()
    assert names[:len(OLD_SEED)] == OLD_SEED, names               # what was seeded before, unchanged and first
    assert set(names) == set(OLD_SEED) | set(NEW_SEED), names
    assert SZ.RUN_RECORD == 'size_rule.json'
    assert len(set(names)) == len(names)


def test_seed_copies_every_item_byte_for_byte():
    root = tempfile.mkdtemp(prefix='perhour_seed_')
    try:
        src = _published(root)
        dst = os.path.join(root, 'attempt', 'transients', 'Run_SLSTB_03h_73MHz_V')
        done = TS.seed_attempt(src, dst)
        assert done == list(TS.per_hour_seed_names()), done
        for name in OLD_SEED[:3] + NEW_SEED:
            assert filecmp.cmp(os.path.join(src, name), os.path.join(dst, name), shallow=False), name
        for sub in ('sigstack', 'refusals'):
            cmp_ = filecmp.dircmp(os.path.join(src, sub), os.path.join(dst, sub))
            assert not cmp_.left_only and not cmp_.right_only and not cmp_.diff_files, sub
        assert sorted(os.listdir(dst)) == sorted(os.listdir(src))       # nothing else, no temporary file left
        again = TS.seed_attempt(src, dst)                               # a retried unit: same result
        assert again == done and sorted(os.listdir(dst)) == sorted(os.listdir(src))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_seed_skips_what_the_published_cell_lacks():
    root = tempfile.mkdtemp(prefix='perhour_seed_')
    try:
        src = _published(root, table=False, record=False)               # a cell searched under production's rule, before the
        dst = os.path.join(root, 'attempt', 'run')                      # table record existed
        done = TS.seed_attempt(src, dst)
        assert done == list(OLD_SEED), done
        assert not os.path.exists(os.path.join(dst, SZ.RUN_RECORD))
        assert not os.path.exists(os.path.join(dst, 'search_table_record.csv'))
        empty = os.path.join(root, 'none')
        os.makedirs(empty)
        assert TS.seed_attempt(empty, os.path.join(root, 'attempt2')) == []
        assert os.path.isdir(os.path.join(root, 'attempt2'))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_engine_accepts_the_seeded_attempt_and_refused_without_the_record():
    root = tempfile.mkdtemp(prefix='perhour_seed_')
    try:
        src = _published(root)
        # the defect: the five old items only - a ledger without its record reads as production's rule
        old = os.path.join(root, 'attempt_old')
        os.makedirs(old)
        for name in OLD_SEED:
            TS.copy_tree(os.path.join(src, name), os.path.join(old, name))
        assert SZ.read_run_record(old)['rule'] == 'production'
        try:
            SZ.write_run_record(old, _env())
            raise AssertionError('an envelope search of a ledger without its size-rule record was not refused')
        except SZ.RuleRefusal as e:
            assert 'one rule per run' in str(e), e
        # the repair: the seeded attempt carries the published cell's rule and the engine accepts it
        new = os.path.join(root, 'attempt_new')
        TS.seed_attempt(src, new)
        assert SZ.same_rule(SZ.read_run_record(new), SZ.read_run_record(src))
        before = open(os.path.join(new, SZ.RUN_RECORD), 'rb').read()
        rec = SZ.write_run_record(new, _env())
        assert rec['rule'] == 'envelope' and open(os.path.join(new, SZ.RUN_RECORD), 'rb').read() == before
        # another envelope product than the published cell's still refuses: the seed admits no second rule
        try:
            SZ.write_run_record(new, _env(md5='ffffffffffffffffffffffffffffffff'))
            raise AssertionError('a second envelope product was accepted in one run directory')
        except SZ.RuleRefusal:
            pass
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_publish_keeps_both_nights_in_the_table_record():
    root = tempfile.mkdtemp(prefix='perhour_seed_')
    try:
        append = _engine_function('_append_locked')
        src = _published(root)
        dst = os.path.join(root, 'attempt', 'run')
        TS.seed_attempt(src, dst)
        append(os.path.join(dst, 'search_table_record.csv'), ST.STR_COLS, _table_rows('2024-12-27', n=2))   # the engine's append
        TS.copy_tree(os.path.join(dst, 'search_table_record.csv'), os.path.join(src, 'search_table_record.csv'))   # the publish
        rows = list(csv.reader(open(os.path.join(src, 'search_table_record.csv'), newline='')))
        assert rows[0] == list(ST.STR_COLS)
        dates = [r[3] for r in rows[1:]]
        assert dates == ['2024-12-19'] * 3 + ['2024-12-27'] * 2, dates
        # without the seed the same two steps leave the one night only (the defect)
        bare = os.path.join(root, 'attempt_bare')
        os.makedirs(bare)
        append(os.path.join(bare, 'search_table_record.csv'), ST.STR_COLS, _table_rows('2024-12-27', n=2))
        rows = list(csv.reader(open(os.path.join(bare, 'search_table_record.csv'), newline='')))
        assert [r[3] for r in rows[1:]] == ['2024-12-27'] * 2
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_header_gate_refuses_where_the_engine_append_would():
    assert _engine_literal('STR_COLS') == list(ST.STR_COLS)            # the gate's columns are the engine's
    append = _engine_function('_append_locked')
    cases = (('the engine\'s columns', None, False),
             ('one column fewer (a record of older code)', list(ST.STR_COLS[:-1]), True),
             ('one column renamed', list(ST.STR_COLS[:-1]) + ['other'], True),
             ('two columns swapped', [ST.STR_COLS[1], ST.STR_COLS[0]] + list(ST.STR_COLS[2:]), True))
    for what, header, refused in cases:
        root = tempfile.mkdtemp(prefix='perhour_seed_')
        try:
            rd = _published(root, header=header)
            why = TS.table_record_refusal(rd)
            assert (why is not None) == refused, (what, why)
            if refused:
                assert 'search_table_record.csv' in why and str(len(header)) in why and '--fresh' in why, why
            try:
                append(os.path.join(rd, 'search_table_record.csv'), ST.STR_COLS, _table_rows('2024-12-27', n=1))
                engine_refused = False
            except RuntimeError:
                engine_refused = True
            assert engine_refused == refused, what                     # gate and engine agree
        finally:
            shutil.rmtree(root, ignore_errors=True)
    root = tempfile.mkdtemp(prefix='perhour_seed_')
    try:
        rd = _published(root, table=False)
        assert TS.table_record_refusal(rd) is None                     # no record yet: the engine starts one
        open(os.path.join(rd, 'search_table_record.csv'), 'w').close()
        assert TS.table_record_refusal(rd) is None                     # an empty file: the engine appends without refusing
        append(os.path.join(rd, 'search_table_record.csv'), ST.STR_COLS, _table_rows('2024-12-27', n=1))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_detect_seeds_through_seed_attempt_and_the_plan_holds_the_gate():
    tree = ast.parse(open(STEP).read())
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    calls = [n for n in ast.walk(funcs['detect']) if isinstance(n, ast.Call) and getattr(n.func, 'id', '') == 'seed_attempt']
    assert len(calls) == 1, 'detect() must seed the attempt through seed_attempt, once'
    for n in ast.walk(funcs['detect']):                                # no second, private list of seeded names in detect()
        assert not (isinstance(n, ast.Constant) and n.value == 'v_deep_ledger.csv'), 'detect() names seeded files itself'
    holders = [name for name, f in funcs.items()
               if any(isinstance(n, ast.Constant) and n.value == 'epoch.table_record_header' for n in ast.walk(f))]
    assert len(holders) == 1 and holders[0] != 'detect', holders       # in the plan (before any work), not in detect()
    lines = {}
    for n in ast.walk(funcs[holders[0]]):
        if isinstance(n, ast.Constant) and n.value in ('epoch.size_rule_of_the_base', 'epoch.table_record_header'):
            lines[n.value] = n.lineno
    assert lines['epoch.size_rule_of_the_base'] < lines['epoch.table_record_header'], lines
    gate_calls = [n for n in ast.walk(funcs[holders[0]])
                  if isinstance(n, ast.Call) and getattr(n.func, 'id', '') == 'table_record_refusal']
    assert len(gate_calls) == 1


def _all_tests():
    return sorted((n, f) for n, f in globals().items() if n.startswith('test_') and callable(f))


def main(argv):
    names = argv or [n for n, _ in _all_tests()]
    fails = 0
    for n, f in _all_tests():
        if n not in names:
            continue
        try:
            f()
            print(f'PASS  {n}')
        except Exception as e:                     # noqa: BLE001 - report every failure
            fails += 1
            import traceback
            traceback.print_exc()
            print(f'FAIL  {n}: {e}')
    total = len([n for n, _ in _all_tests() if n in names])
    print(f'{total - fails}/{total} passed')
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
