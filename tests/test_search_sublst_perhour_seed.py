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
            skipped; a second seed changes nothing; a short copy is tried once more and then refused
  rule      with the seed the engine's own check (write_run_record) accepts the attempt and without the record it refuses:
            the defect and its repair, on the engine's function
  append    the night's rows appended to the seeded table record and published leave BOTH nights' rows in the cell
  header    the preflight gate refuses a published gate ledger, image ledger or table record with another column set,
            exactly where the engine's own append (deep_search._append_locked, run here from its source) would refuse at
            the END of the search, and an empty one; its column lists are the engine's
  publish   the publish's own file list carries the two records to the published cell
  wiring    detect() seeds through seed_attempt, and the per-hour plan holds the gate after the size-rule gate
  guard     two per-hour searches of one cell: the cell's lease admits one holder at a time (another Stokes, or another
            published root, is another lease); at the publish a published record that changed, appeared or vanished since
            the seed is named and refuses; run_unit takes the lease before the unit and checks before publish()

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

from exopipe.phase3.lib.search import gate_population as GP    # noqa: E402
from exopipe.phase3.lib.search import size_rule as SZ          # noqa: E402
from exopipe.phase3.lib.search import tests as ST              # noqa: E402
from exopipe.phase3.lib.search import tier_sublst as TS        # noqa: E402

STEP = os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'search', 'tier_sublst.py')
ENGINE = os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'search', 'deep_search.py')
OLD_SEED = ('master_catalog.csv', 'v_deep_ledger.csv', 'v_deep_images.csv', 'sigstack', 'refusals')
NEW_SEED = (SZ.RUN_RECORD, 'search_table_record.csv', GP.FILE)
LABEL = 'SLSTB_03h_73MHz_V'
LABEL_DIR = f'Run_{LABEL}'


def _env(md5='e184ac13f6eb0123456789abcdef0123'):
    """Stand-in for the size rule's envelope product: what size_rule.run_record reads of it."""
    return types.SimpleNamespace(source='/nowhere/size_envelope_10min.json', md5=md5, full_band='off')


def _table_rows(date, n=3):
    return [['03h', '73MHz', 'V', date, f'{45.0 + i:.4f}', '30.0000', str(100 + i), '200', '7.5', 'kept', '']
            + [''] * (len(ST.STR_COLS) - 11) for i in range(n)]


def _csv_text(cols, date='2024-12-19'):
    return ','.join(cols) + '\n' + ','.join(['03h', '73MHz', 'V', date] + ['1'] * (len(cols) - 4)) + '\n'


def _published(root, table=True, record=True, header=None):
    """A published run directory of one cell holding night 2024-12-19, searched under the envelope rule (record=True: its
    gate ledger carries the size rule's columns) or under production's. header: another header for the table's record."""
    rd = os.path.join(root, 'pub', 'transients', LABEL_DIR)
    os.makedirs(os.path.join(rd, 'sigstack'))
    os.makedirs(os.path.join(rd, 'refusals'))
    gled = list(ST.GLED_COLS) + (list(ST.GLED_COLS_SIZE) if record else [])
    for name, text in (('master_catalog.csv', 'source_id,date\ns1,2024-12-19\n'),
                       ('v_deep_ledger.csv', _csv_text(gled)),
                       ('v_deep_images.csv', _csv_text(ST.GIMG_COLS)),
                       (os.path.join('sigstack', 'sigstack_03h_73MHz_V_w0.fits'), 'stand-in for a sigma-stack map\n'),
                       (os.path.join('refusals', '03h_73MHz_V_night_excluded.csv'), 'date,why\n')):
        with open(os.path.join(rd, name), 'w') as f:
            f.write(text)
    frames = [f'2024-12-19T{w:02d}' for w in range(6)]                  # the populations its whole-cell search recorded
    GP.append(rd, [GP.row('03h', '73MHz', g, frames, 0.25, 2.0 if g != 'beam' else 1.5, 5, 'applied', 'search', '20261101T000000')
                   for g in (GP.TIER_GATES['I'], GP.TIER_GATES['V'], 'beam')])
    os.remove(os.path.join(rd, GP.FILE + '.lock'))
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
        dst = os.path.join(root, 'attempt', 'transients', LABEL_DIR)
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


def test_seed_refuses_a_short_copy():
    real = TS.copy_tree
    for bad_tries, refused in ((1, False), (2, True)):               # short once: tried again; short twice: refused
        root = tempfile.mkdtemp(prefix='perhour_seed_')
        calls = []

        def short(a, b, _n=bad_tries, _calls=calls):
            real(a, b)
            if os.path.basename(a) == 'v_deep_ledger.csv':
                _calls.append(a)
                if len(_calls) <= _n:
                    with open(b, 'r+b') as f:                         # what Lustre did: the copy ends early, no error
                        f.truncate(os.path.getsize(a) // 2)
        try:
            src = _published(root)
            dst = os.path.join(root, 'attempt', 'run')
            TS.copy_tree = short
            try:
                done = TS.seed_attempt(src, dst)
                assert not refused, 'a ledger copied short twice was seeded'
                assert done == list(TS.per_hour_seed_names()) and len(calls) == 2
                assert filecmp.cmp(os.path.join(src, 'v_deep_ledger.csv'), os.path.join(dst, 'v_deep_ledger.csv'), shallow=False)
            except Exception as e:                                    # noqa: BLE001 - the step's Refusal
                assert refused and type(e).__name__ == 'Refusal' and 'copied short twice' in str(e), repr(e)
        finally:
            TS.copy_tree = real
            shutil.rmtree(root, ignore_errors=True)
    root = tempfile.mkdtemp(prefix='perhour_seed_')
    try:                                                             # a file missing from a copied directory is short too
        src = _published(root)
        dst = os.path.join(root, 'attempt', 'run')
        TS.seed_attempt(src, dst)
        os.unlink(os.path.join(dst, 'sigstack', 'sigstack_03h_73MHz_V_w0.fits'))
        assert TS._short_copies(os.path.join(src, 'sigstack'), os.path.join(dst, 'sigstack'))
        assert not TS._short_copies(os.path.join(src, 'refusals'), os.path.join(dst, 'refusals'))
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


def test_gate_columns_are_the_engines():
    assert _engine_literal('STR_COLS') == list(ST.STR_COLS)
    assert _engine_literal('GIMG_COLS') == list(ST.GIMG_COLS)
    # the engine: GLED_COLS = [the 28 columns] (its first assignment), GLED_COLS_V1 = list(GLED_COLS), then
    # GLED_COLS = GLED_COLS_V1 + GLED_COLS_TABLE
    assert _engine_literal('GLED_COLS') + _engine_literal('GLED_COLS_TABLE') == list(ST.GLED_COLS)
    assert list(SZ.GLED_COLS_SIZE) == list(ST.GLED_COLS_SIZE)         # the engine imports the size rule's columns from size_rule
    for record in (True, False):
        root = tempfile.mkdtemp(prefix='perhour_seed_')
        try:
            want = TS.appended_headers(_published(root, record=record))
            assert want == {'v_deep_ledger.csv': list(ST.GLED_COLS) + (list(ST.GLED_COLS_SIZE) if record else []),
                            'v_deep_images.csv': list(ST.GIMG_COLS), 'search_table_record.csv': list(ST.STR_COLS)}
        finally:
            shutil.rmtree(root, ignore_errors=True)


def test_header_gate_refuses_where_the_engine_append_would():
    append = _engine_function('_append_locked')
    changes = (('the engine\'s columns', lambda c: list(c), False),
               ('one column fewer (a record of older code)', lambda c: list(c[:-1]), True),
               ('one column renamed', lambda c: list(c[:-1]) + ['other'], True),
               ('two columns swapped', lambda c: [c[1], c[0]] + list(c[2:]), True))
    for record in (True, False):                                      # a cell under the envelope rule, one under production's
        for name in ('v_deep_ledger.csv', 'v_deep_images.csv', 'search_table_record.csv'):
            for what, change, refused in changes:
                root = tempfile.mkdtemp(prefix='perhour_seed_')
                try:
                    rd = _published(root, record=record)
                    cols = TS.appended_headers(rd)[name]
                    header = change(cols)
                    with open(os.path.join(rd, name), 'w') as f:
                        f.write(_csv_text(header))
                    why = TS.append_header_refusal(rd)
                    assert (why is not None) == refused, (name, what, why)
                    if refused:
                        assert name in why and str(len(header)) in why and '--fresh' in why, why
                    try:
                        append(os.path.join(rd, name), cols, [['03h', '73MHz', 'V', '2024-12-27'] + ['1'] * (len(cols) - 4)])
                        engine_refused = False
                    except RuntimeError:
                        engine_refused = True
                    assert engine_refused == refused, (name, what)    # gate and engine agree
                finally:
                    shutil.rmtree(root, ignore_errors=True)
    root = tempfile.mkdtemp(prefix='perhour_seed_')
    try:                                                             # the gate ledger's columns follow the cell's own rule
        rd = _published(root, record=True)
        with open(os.path.join(rd, 'v_deep_ledger.csv'), 'w') as f:
            f.write(_csv_text(ST.GLED_COLS))                         # an envelope cell whose ledger lacks the rule's columns
        assert 'v_deep_ledger.csv' in (TS.append_header_refusal(rd) or '')
    finally:
        shutil.rmtree(root, ignore_errors=True)
    root = tempfile.mkdtemp(prefix='perhour_seed_')
    try:
        rd = _published(root, table=False)
        assert TS.append_header_refusal(rd) is None                    # no table record yet: the engine starts one
        p = os.path.join(rd, 'search_table_record.csv')
        open(p, 'w').close()                                           # an EMPTY record: the engine would append rows with no
        why = TS.append_header_refusal(rd)                             # header (shown below), so the gate refuses it
        assert why and 'is empty' in why and 'search_table_record.csv' in why, why
        append(p, ST.STR_COLS, _table_rows('2024-12-27', n=1))
        assert next(csv.reader(open(p, newline=''))) != list(ST.STR_COLS)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_publish_file_list_carries_the_two_records():
    root = tempfile.mkdtemp(prefix='perhour_seed_')
    try:
        src = _published(root)
        attempt = os.path.join(root, 'attempt')
        TS.seed_attempt(src, os.path.join(attempt, 'transients', LABEL_DIR))
        plan = dict(label=LABEL, cell='03h_73MHz', arm='A', date='2024-12-27', publish=os.path.join(root, 'pub'))
        pairs = dict(TS._publish_files(plan, attempt))                 # the publish's own list (search_sublst.PUBLISH_ITEMS)
        for name in NEW_SEED + ('v_deep_ledger.csv', 'v_deep_images.csv', 'master_catalog.csv'):
            a = os.path.join(attempt, 'transients', LABEL_DIR, name)
            assert pairs.get(a) == os.path.join(root, 'pub', 'transients', LABEL_DIR, name), (name, pairs.get(a))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_detect_seeds_through_seed_attempt_and_the_plan_holds_the_gate():
    tree = ast.parse(open(STEP).read())
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    calls = [n for n in ast.walk(funcs['detect']) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
             and getattr(n.value.func, 'id', '') == 'seed_attempt']
    assert len(calls) == 1, 'detect() must seed the attempt through seed_attempt, once, and keep what it returns (the note)'
    for n in ast.walk(funcs['detect']):                                # no second, private list of seeded names in detect()
        assert not (isinstance(n, ast.Constant) and n.value == 'v_deep_ledger.csv'), 'detect() names seeded files itself'
    holders = [name for name, f in funcs.items()
               if any(isinstance(n, ast.Constant) and n.value == 'epoch.appended_headers' for n in ast.walk(f))]
    assert len(holders) == 1 and holders[0] != 'detect', holders       # in the plan (before any work), not in detect()
    lines = {}
    for n in ast.walk(funcs[holders[0]]):
        if isinstance(n, ast.Constant) and n.value in ('epoch.size_rule_of_the_base', 'epoch.appended_headers'):
            lines[n.value] = n.lineno
    assert lines['epoch.size_rule_of_the_base'] < lines['epoch.appended_headers'], lines
    gate_calls = [n for n in ast.walk(funcs[holders[0]])
                  if isinstance(n, ast.Call) and getattr(n.func, 'id', '') == 'append_header_refusal']
    assert len(gate_calls) == 1


def test_seed_state_names_a_record_changed_since_the_seed():
    root = tempfile.mkdtemp(prefix='perhour_seed_')
    try:
        src = _published(root, table=False)                            # the published cell has no table record yet
        dst = os.path.join(root, 'attempt', 'transients', LABEL_DIR)
        TS.seed_attempt(src, dst)
        state = TS.seed_state(dst)
        assert set(state) == set(TS.per_hour_seed_names()) - {'sigstack', 'refusals'}, state
        assert state['search_table_record.csv'] is None and all(v for k, v in state.items() if k != 'search_table_record.csv')
        assert TS.seed_changed(src, state) is None                     # nothing published meanwhile
        with open(os.path.join(dst, 'v_deep_ledger.csv'), 'a') as f:   # the engine appends to the ATTEMPT: not a change
            f.write('03h,73MHz,V,2024-12-27\n')
        assert TS.seed_changed(src, state) is None
        with open(os.path.join(src, 'v_deep_images.csv'), 'a') as f:   # another search published a night
            f.write('03h,73MHz,V,2024-12-23\n')
        why = TS.seed_changed(src, state)
        assert why and 'v_deep_images.csv' in why and 'changed after this attempt was seeded' in why, why
        src2 = _published(os.path.join(root, 'two'), table=False)
        dst2 = os.path.join(root, 'two', 'attempt', 'transients', LABEL_DIR)
        TS.seed_attempt(src2, dst2)
        state2 = TS.seed_state(dst2)
        with open(os.path.join(src2, 'search_table_record.csv'), 'w') as f:
            f.write(','.join(ST.STR_COLS) + '\n')
        why = TS.seed_changed(src2, state2)
        assert why and 'search_table_record.csv' in why and 'did not exist' in why, why
        os.remove(os.path.join(src2, 'search_table_record.csv'))
        os.remove(os.path.join(src2, 'master_catalog.csv'))
        why = TS.seed_changed(src2, state2)
        assert why and 'master_catalog.csv' in why and 'is gone' in why, why
        assert 'no record' in TS.seed_changed(src, None)               # an attempt without a seed record never publishes
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_cell_lease_admits_one_search_of_a_cell_at_a_time():
    root = tempfile.mkdtemp(prefix='perhour_seed_')
    try:
        lr = os.path.join(root, 'leases')
        plan = dict(cell='03h_73MHz', stokes='V', publish=os.path.join(root, 'pub', 'search_B', '03h_73MHz_V'))
        a = TS.cell_lease(lr, plan, 'sublst-B_03h_73MHz_V_2024-12-19')
        b = TS.cell_lease(lr, plan, 'sublst-B_03h_73MHz_V_2024-12-27')
        assert a.capacity == 1 and a.dir == b.dir
        ok, _ = a.try_acquire()
        assert ok
        ok, live = b.try_acquire()                                     # the same cell and Stokes: refused while a holds it
        assert not ok and [h.get('unit') for h in live] == ['sublst-B_03h_73MHz_V_2024-12-19'], live
        other_stokes = TS.cell_lease(lr, dict(plan, stokes='I'), 'u')
        other_root = TS.cell_lease(lr, dict(plan, publish=os.path.join(root, 'parity', 'search_B', '03h_73MHz_V')), 'u')
        assert len({a.dir, other_stokes.dir, other_root.dir}) == 3
        for other in (other_stokes, other_root):
            ok, _ = other.try_acquire()
            assert ok
            other.release()
        a.release()
        ok, _ = b.try_acquire()                                        # released: the second search goes on
        assert ok
        b.release()
        link = os.path.join(root, 'pub_link')                          # another spelling of the same published directory
        os.symlink(os.path.join(root, 'pub'), link)
        assert TS.cell_lease(lr, dict(plan, publish=os.path.join(link, 'search_B', '03h_73MHz_V')), 'u').dir == a.dir
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_held_cell_lease_is_taken_and_released():
    root = tempfile.mkdtemp(prefix='perhour_seed_')
    saved = TS.lease_root
    TS.lease_root = lambda out_root: os.path.join(out_root, '_leases')       # never the configured (production) root
    try:
        run = types.SimpleNamespace(dry_run=False, out_root=root)
        full = dict(cell='03h_73MHz', stokes='V', publish=os.path.join(root, 'pub', 'search_B', '03h_73MHz_V'),
                    date='2024-12-27', refuse='', noinput='')
        for case in (dict(full, refuse='x'), dict(full, noinput='x')):
            with TS.held_cell_lease(run, case, 'u', ['detect', 'publish']) as lease:
                assert lease is None, case                             # a unit that will not seed
        with TS.held_cell_lease(types.SimpleNamespace(dry_run=True, out_root=root), full, 'u', ['detect']) as lease:
            assert lease is None
        with TS.held_cell_lease(run, full, 'u', ['fit', 'verify']) as lease:
            assert lease is None                                       # no detect: nothing is seeded or published
        for case in (full, dict(full, date='')):                       # a per-hour search and a full pass both hold it
            other = TS.cell_lease(TS.lease_root(root), case, 'other')
            with TS.held_cell_lease(run, case, 'u', ['detect', 'publish']) as lease:
                assert lease is not None and os.path.isfile(lease.path), case
                ok, live = other.try_acquire()
                assert not ok and [h.get('unit') for h in live] == ['u'], live
            assert not os.path.exists(lease.path)                      # released at the end of the block
            ok, _ = other.try_acquire()
            assert ok
            other.release()
            try:
                with TS.held_cell_lease(run, case, 'u', ['detect']) as lease:
                    raise KeyError('the unit failed')
            except KeyError:
                pass
            assert not os.path.exists(lease.path)                      # and when the unit raises
    finally:
        TS.lease_root = saved
        shutil.rmtree(root, ignore_errors=True)


def _stubbed_unit(root, changed, resume_ok=False, lose_lease=False):
    """run_unit of one per-hour unit with the plan, the engine and the publish replaced by stand-ins that record their
    calls. Returns (calls, the run's unit record status, the lease holders seen by the engine)."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import record
    calls, seen = [], {}
    pub = os.path.join(root, 'pub', 'search_B', '03h_73MHz_V')

    def preflight(args, run, cell, lst, band, stokes, date):
        calls.append('plan')
        return dict(cell=cell, stokes=stokes, publish=pub, label=LABEL, date=date, refuse='', noinput='', gates=[], tree='t',
                    coadd='c', windows=[], excluded_nights=[], included_dates=[date], admitted=[date], inputs=[])

    def detect(run, u, plan, attempt):
        calls.append('detect')
        ldir = TS.cell_lease(TS.lease_root(run.out_root), plan, 'x').hdir
        seen['holders'] = sorted(os.listdir(ldir))
        seen['attempt'] = attempt
        plan['seed_state'] = {'v_deep_images.csv': 'md5-at-the-seed'}
        if lose_lease:
            for f in seen['holders']:
                os.remove(os.path.join(ldir, f))
        return dict(summary='stand-in', rc=0, image_rows=1)

    def publish(u, run, plan, attempt):
        calls.append('publish')
        return pub, (lambda: calls.append('rollback'))

    def seed_changed(run_dir, state):
        calls.append('seed_changed')
        assert state == {'v_deep_images.csv': 'md5-at-the-seed'}
        return changed

    names = dict(preflight=preflight, detect=detect, publish=publish, seed_changed=seed_changed,
                 lease_root=lambda out_root: os.path.join(out_root, '_leases'), make_lease=lambda *a, **k: None,
                 wait_free=lambda *a, **k: None, products=lambda *a, **k: [])
    saved = {k: getattr(TS, k) for k in names}
    run = record.Run('search', ['test'], os.path.join(root, 'out'), CFG, config_sections=None)
    if resume_ok:
        run.resume, run.resume_check = True, (lambda unit, inputs, admitted: (calls.append('resume') or len(calls) > 2, 'r'))
    for k, v in names.items():
        setattr(TS, k, v)
    try:
        args = types.SimpleNamespace(sublst_arm='B', replay_attempt=None, sublst_keep_attempt=False)
        try:
            TS.run_unit(args, run, '03h_73MHz', '03h', '73MHz', 'V', '2024-12-27', ['detect', 'publish'],
                        os.path.join(root, 'work'))
        except record.Refusal as e:
            seen['refusal'] = str(e)
    finally:
        for k, v in saved.items():
            setattr(TS, k, v)
    u = run.units[-1] if getattr(run, 'units', None) else None
    seen['status'], seen['reason'] = getattr(u, 'status', None), getattr(u, 'reason', '') or seen.get('refusal', '')
    seen['left'] = sorted(f for _d, _s, fs in os.walk(os.path.join(root, 'out', '_leases')) for f in fs if f.endswith('.json'))
    return calls, seen


def test_run_unit_plans_under_the_lease_and_refuses_a_changed_record():
    root = tempfile.mkdtemp(prefix='perhour_seed_')
    try:
        calls, seen = _stubbed_unit(os.path.join(root, 'a'), changed=None)
        assert calls == ['plan', 'plan', 'detect', 'seed_changed', 'publish'], calls     # planned again under the lease
        assert len(seen['holders']) == 1 and not seen['left'], seen                       # held at the engine, released after
        assert os.path.basename(seen['attempt']).startswith('03h_73MHz_V_2024-12-27_'), seen['attempt']
        calls, seen = _stubbed_unit(os.path.join(root, 'b'), changed='the image ledger changed after this attempt')
        assert calls == ['plan', 'plan', 'detect', 'seed_changed'], calls                 # never published
        assert 'NOT published' in seen['reason'] and not seen['left'], seen
        calls, seen = _stubbed_unit(os.path.join(root, 'c'), changed=None, lose_lease=True)
        assert calls == ['plan', 'plan', 'detect'], calls                                 # the lease was lost: no publish
        assert 'no longer holds the lease' in seen['reason'], seen
        calls, seen = _stubbed_unit(os.path.join(root, 'd'), changed=None, resume_ok=True)
        assert calls == ['plan', 'resume', 'resume'], calls       # done by the other run meanwhile: skipped under the lease
        assert seen['status'] == 'skipped-valid' and not seen['left'], seen
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_run_unit_holds_the_lease_and_checks_before_the_publish():
    tree = ast.parse(open(STEP).read())
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def calls(fn, name):
        return [n for n in ast.walk(funcs[fn]) if isinstance(n, ast.Call) and getattr(n.func, 'id', '') == name]
    lease, unit_with = calls('run_unit', 'held_cell_lease'), None
    assert len(lease) == 1
    for n in ast.walk(funcs['run_unit']):                              # the with statement that opens the unit's record
        if isinstance(n, ast.With) and any(isinstance(i.context_expr, ast.Call) and getattr(i.context_expr.func, 'attr', '')
                                           == 'unit' for i in n.items):
            unit_with = n
    assert unit_with is not None and lease[0].lineno < unit_with.lineno          # taken before the unit starts
    assert getattr(unit_with.items[0].context_expr, 'id', '') == 'held'            # and released when the unit ends
    check, pub = calls('run_unit', 'seed_changed'), calls('run_unit', 'publish')
    assert len(check) == 1 and len(pub) == 1 and check[0].lineno < pub[0].lineno
    assert len(calls('detect', 'seed_state')) == 1
    state_line = calls('detect', 'seed_state')[0].lineno
    assert calls('detect', 'seed_attempt')[0].lineno < state_line < calls('detect', 'legacy_call')[0].lineno   # before the engine


def _all_tests():
    return sorted((n, f) for n, f in globals().items() if n.startswith('test_') and callable(f))


def main(argv):
    names = argv or [n for n, _ in _all_tests()]
    unknown = sorted(set(names) - {n for n, _ in _all_tests()})
    if unknown:
        print(f'no such test: {", ".join(unknown)}')
        return 2
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
