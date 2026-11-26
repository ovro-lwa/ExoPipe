#!/usr/bin/env python3
"""Data-free tests of the snapshot chain's per-tier serving stage (cutover row 71; lib/catalog/tier_snapshot.st_tierserve, gate
P-G27) and of the content test of its table (lib/catalog/tests_snapshot.served_tiers_rows, class served_tiers).

The stage runs production's clv15 phase3_snapshot_tierserve.py in place (here a stub in a temporary code tree, its behavior set by
the environment variable TSTUB_MODE) with --samples-dir = the run's recorded copies (run_samples.copy_dir), verified against
their record before the call and again after it. The table is written to a temporary name beside RUN/served_tiers.csv and placed
only after the call's self-check line, the row count it reported and the content test; any refusal leaves no table and no
temporary file, and an existing table is kept. No record, a copy with another md5, the chain's outputs missing: refused before
the call (the stub is never run). Nothing falls back to the live catalogs.

Run: python tests/test_served_tiers.py [test names]   (exit 0 = all passed, 1 = a failure, 2 = imports missing)
"""
import csv
import glob
import gzip
import hashlib
import inspect
import json
import os
import shutil
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

_IMPORT_ERROR = None
try:
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import record
    from exopipe.phase3.lib.catalog import run_samples as RS
    from exopipe.phase3.lib.catalog import tests_snapshot as TP
    from exopipe.phase3.lib.catalog import tier_snapshot as P
    from exopipe.phase3.lib.catalog import tier_sublst
except Exception as e:  # noqa: BLE001
    _IMPORT_ERROR = e
    if __name__ != '__main__':
        raise

NAMES = ['Hot_Warm_Jupiters.csv', '10pc.csv', '25pc.csv', 'LPTs.csv', 'flare_ucd.csv', 'sydney_stars.csv', 'pulsars_atnf.csv']
COLS = ['src_index', 'identity_jname', 'chain_outcome', 'tier', 'n_members_tier', 'n_stripped_tier', 'best_det_uid_tier',
        'best_snr_tier', 'best_stokes_tier', 'best_verdict_tier', 'singleton_floor_tier', 'n_inst_tier', 'multi_integration_tier',
        'routes_tier', 'qualifies', 'fail_reason', 'tier_holds_source_best', 'snr_at_floor_rounding', 'add_without_products',
        'identity_qualifies_tiers', 'n_members_10s', 'best_snr_10s', 'n_members_1min', 'best_snr_1min', 'identity_note']


def good_rows():
    """Three rows of a consistent table: source 0 served (10s qualifies, 1min does not), source 1 not designated (10s, does not
    qualify) - one served source, as the chain summary of the World says."""
    def row(src, jn, outcome, tier, nm, routes, q, reason, qt, n10, n1):
        return dict(zip(COLS, [src, jn, outcome, tier, nm, '0', 'c%s%s' % (src, tier), '9.5', 'I', 'ok', '7.0', nm, '1',
                               routes, q, reason, '1', '0', str(int(q == '1' and outcome != 'served')), qt, n10, '9.5', n1,
                               '9.5' if n1 != '0' else '', '']))
    return [row('0', 'J0101+0101', 'served', '10s', '3', 'Red;Blue', '1', '', '10s', '3', '2'),
            row('0', 'J0101+0101', 'served', '1min', '2', '', '0', 'singleton: snr 6.1 < 7.0', '10s', '3', '2'),
            row('1', 'J0202+0202', 'not-designated', '10s', '1', '', '0', 'singleton: snr 5.0 < 7.0', '', '1', '0')]


def write_table(path, rows, cols=COLS):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


STUB = r'''
import csv, json, os, sys
mode = os.environ.get("TSTUB_MODE", "ok")
a = sys.argv[1:]
def opt(k):
    return a[a.index(k) + 1]
run, sd, out = opt("--run"), opt("--samples-dir"), opt("--out")
open(os.environ["TSTUB_CALLS"], "a").write(" ".join(a) + "\n")
# the stage hands the run's recorded copies, never the live catalogs, and the chain's code tree for the re-applied functions
assert sd == os.path.join(run, "sample_catalogs"), sd
assert os.environ.get("P3_TIERSERVE_CODE") == os.path.dirname(os.path.abspath(__file__)), os.environ.get("P3_TIERSERVE_CODE")
assert os.path.dirname(out) == run and out != os.path.join(run, "served_tiers.csv"), out
rows = json.loads(os.environ["TSTUB_ROWS"])
cols = json.loads(os.environ["TSTUB_COLS"])
if mode == "zero":
    rows = []
if mode == "bad":
    rows[0]["routes_tier"] = ""               # qualifies 1 without a route
if mode == "mutate":                          # a copy the record lists changes while the evaluation runs
    open(os.path.join(sd, "LPTs.csv"), "a").write("x,1,2\n")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow(r)
if mode == "refuse":                          # production: sys.exit("REFUSED: ...") after its self-check line
    print("self-check: 0 of 1 served sources reproduced (mi_tiers, catalog, routes, best member)")
    sys.exit("REFUSED: the reconstruction does not reproduce the chain on 1 served sources")
if mode != "noselfcheck":
    print("self-check: 1 of 1 served sources reproduced (mi_tiers, catalog, routes, best member)")
print("served identities 1: qualify in 10s only 1, 1min only 0, both 0, NEITHER 0")
print("wrote %s (%d rows)" % (out, len(rows) + (2 if mode == "rowcount" else 0)))
'''


class World(object):
    """A live samples directory, a code tree holding the stub tierserve, and a chain run (chain_summary.json with n_served 1,
    detection_record.csv.gz, the sample-catalog record of the chain). The work directory and the records lie outside the run,
    so the run's listing shows exactly what the stage placed there."""

    def __init__(self, record_samples=True):
        self.root = tempfile.mkdtemp(prefix='served_tiers_')
        self.src = os.path.join(self.root, 'live_catalogs')
        self.clv = os.path.join(self.root, 'clv15')
        self.run = os.path.join(self.root, 'Run_SNAPTEST_20261126')
        self.work = os.path.join(self.root, 'work')
        self.recs = os.path.join(self.root, 'records')
        self.calls = os.path.join(self.root, 'calls.txt')
        for d in (self.src, self.clv, self.run):
            os.makedirs(d)
        for i, n in enumerate(NAMES):
            with open(os.path.join(self.src, n), 'w') as f:
                f.write('name,ra,dec\n%s_a,%d.0,10.0\n' % (n[:-4], i))
        with open(os.path.join(self.clv, P.TIERSERVE), 'w') as f:
            f.write(STUB)
        with open(os.path.join(self.run, 'chain_summary.json'), 'w') as f:
            json.dump({'n_served': 1, 'cells': ['14h_2025-03-09']}, f)
        with gzip.open(os.path.join(self.run, 'detection_record.csv.gz'), 'wt') as f:
            f.write('cand_id,served,src_index\nc1,1,0\n')
        if record_samples:
            RS.write(self.run, self.src, NAMES, 'chain')
        self._py = dict(CFG.site.PYTHON)
        CFG.site.PYTHON['py36'] = sys.executable
        self._env = {k: os.environ.get(k) for k in ('TSTUB_MODE', 'TSTUB_ROWS', 'TSTUB_COLS', 'TSTUB_CALLS')}
        self.mode('ok')

    def mode(self, m, rows=None):
        os.environ.update(TSTUB_MODE=m, TSTUB_ROWS=json.dumps(rows if rows is not None else good_rows()),
                          TSTUB_COLS=json.dumps(COLS), TSTUB_CALLS=self.calls)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        CFG.site.PYTHON.clear()
        CFG.site.PYTHON.update(self._py)
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.root, ignore_errors=True)

    def listing(self):
        out = []
        for d, _ds, fs in os.walk(self.run):
            for f in fs:
                p = os.path.join(d, f)
                out.append((os.path.relpath(p, self.run), md5(p)))
        return sorted(out)

    def n_calls(self):
        return len(open(self.calls).read().splitlines()) if os.path.exists(self.calls) else 0

    def stage(self, dry=False):
        """Run the stage as tier_snapshot.run does: one unit of a record.Run, a tier_sublst.Stage. Returns (unit, plan)."""
        said = []
        ctx = types.SimpleNamespace(T=tier_sublst, clv=self.clv, RUN=self.run, W=self.work, c=CFG.catalog_snapshot, dry=dry,
                                    py36=sys.executable, log=types.SimpleNamespace(say=said.append, path=None))
        plan = []
        if dry:                       # tier_snapshot.run plans a --dry-run on a DryUnit (records nothing)
            u = tier_sublst.DryUnit()
            P.st_tierserve(ctx, tier_sublst.Stage(ctx, 'tierserve', u, tier='snapshot', in_place=()), plan)
            return u, plan
        rr = record.Run('catalog', ['test_served_tiers'], self.recs, CFG)
        with rr.unit('Run_SNAPTEST_20261126.tierserve', tier='snapshot', stage='tierserve') as u:
            st = tier_sublst.Stage(ctx, 'tierserve', u, tier='snapshot', in_place=())
            try:
                P.st_tierserve(ctx, st, plan)
            finally:
                st.close()
        u.said = said
        return u, plan

    @property
    def table(self):
        return os.path.join(self.run, P.SERVED_TIERS)


def md5(p):
    with open(p, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def no_tmp(w):
    left = [p for p in glob.glob(os.path.join(w.run, '*')) if p.endswith('.tmp') or '.tmp' in os.path.basename(p)]
    assert not left, left


def assert_refused_untouched(w, u, before, text, called):
    assert u.status == 'refused', (u.status, u.reason)
    assert 'P-G27' in u.reason and text in u.reason, u.reason
    assert w.listing() == before, (before, w.listing())
    no_tmp(w)
    assert w.n_calls() == called, (w.n_calls(), called)


# ---------------------------------------------------------------------------------------------------------------------
def test_stage_order_and_registration():
    """tierserve is the last stage, after the chain and its run content test (production's order: pertier_prepare after the
    content test, before the pages); the stage function and the content test are wired."""
    assert P.STAGES[-3:] == ('chain', 'test', 'tierserve'), P.STAGES
    assert P.FUNCS['tierserve'] is P.st_tierserve
    from exopipe.phase3.core import validate
    assert [n for n, _f in validate.tests_for('served_tiers')] == ['served_tiers_rows']
    assert tuple(COLS) == TP.SERVED_TIERS_COLUMNS


def test_dry_run_plans_the_recorded_copies():
    """--dry-run: the plan names --samples-dir <run>/sample_catalogs and the record; nothing runs, nothing is written."""
    with World() as w:
        before = w.listing()
        _u, plan = w.stage(dry=True)
        assert len(plan) == 1, plan
        assert '--samples-dir %s ' % RS.copy_dir(w.run) in plan[0] and RS.record_path(w.run) in plan[0], plan
        assert w.src not in plan[0], plan
        assert w.listing() == before and w.n_calls() == 0


def test_success_places_the_table():
    """The table is the stub's rows, placed at RUN/served_tiers.csv after the content test; no temporary file; the unit is done
    with the table as its output (class served_tiers, passed), the samples it read and the report lines recorded."""
    with World() as w:
        u, _plan = w.stage()
        assert u.status == 'done', (u.status, u.reason)
        with open(w.table, newline='') as f:
            assert list(csv.DictReader(f)) == good_rows()
        no_tmp(w)
        assert [o['class'] for o in u.outputs] == ['served_tiers'] and u.outputs[0]['path'] == w.table
        t = [x for x in u.tests if x['cls'] == 'served_tiers']
        assert t and all(x['passed'] for x in t), t
        m = u.meta['tierserve']
        assert m['rows'] == 3 and m['md5'] == md5(w.table) and m['replaced_md5'] is None and m['samples_how'] == 'chain', m
        assert m['samples'] == {n: md5(os.path.join(w.src, n)) for n in NAMES}, m
        assert any(x.startswith('self-check: 1 of 1') for x in m['report']), m['report']
        assert RS.record_path(w.run) in u.inputs and os.path.join(RS.copy_dir(w.run), 'LPTs.csv') in u.inputs
        argv = u.meta['legacy_calls'][0]['argv']
        assert argv[argv.index('--samples-dir') + 1] == RS.copy_dir(w.run), argv
        assert w.n_calls() == 1


def test_existing_table_replaced_on_success():
    with World() as w:
        write_table(w.table, good_rows()[:1])
        old = md5(w.table)
        u, _plan = w.stage()
        assert u.status == 'done', u.reason
        assert md5(w.table) != old and u.meta['tierserve']['replaced_md5'] == old
        no_tmp(w)


def test_refuses_without_a_record():
    """No sample_catalogs.json: refused before the call; nothing in the run changes; the live catalogs are never read."""
    with World(record_samples=False) as w:
        before = w.listing()
        u, _plan = w.stage()
        assert_refused_untouched(w, u, before, 'sample catalogs', 0)
        assert not os.path.exists(w.table)


def test_refuses_a_copy_with_another_md5():
    """A recorded copy (LPTs.csv) whose bytes changed after the record: refused before the call, nothing placed."""
    with World() as w:
        with open(os.path.join(RS.copy_dir(w.run), 'LPTs.csv'), 'a') as f:
            f.write('J1234+5678,1.0,2.0\n')
        before = w.listing()
        u, _plan = w.stage()
        assert_refused_untouched(w, u, before, 'LPTs.csv', 0)
        assert not os.path.exists(w.table)


def test_refuses_without_the_chain_outputs():
    with World() as w:
        os.remove(os.path.join(w.run, 'detection_record.csv.gz'))
        before = w.listing()
        u, _plan = w.stage()
        assert_refused_untouched(w, u, before, 'detection_record.csv.gz', 0)


def test_refusal_of_tierserve_keeps_the_existing_table():
    """tierserve's own refusal (its self-check), after it wrote its output: refused with its REFUSED line; the temporary file
    removed; the existing table kept byte for byte."""
    with World() as w:
        write_table(w.table, good_rows())
        before = w.listing()
        w.mode('refuse')
        u, _plan = w.stage()
        assert_refused_untouched(w, u, before, 'REFUSED: the reconstruction does not reproduce the chain', 1)


def test_refuses_a_record_changed_during_the_call():
    with World() as w:
        w.mode('mutate')
        u, _plan = w.stage()
        assert u.status == 'refused' and 'P-G27' in u.reason and 'LPTs.csv' in u.reason, u.reason
        assert not os.path.exists(w.table)
        no_tmp(w)


def test_refuses_without_the_self_check_line():
    with World() as w:
        before = w.listing()
        w.mode('noselfcheck')
        u, _plan = w.stage()
        assert_refused_untouched(w, u, before, 'self-check', 1)


def test_refuses_a_row_count_other_than_reported():
    with World() as w:
        before = w.listing()
        w.mode('rowcount')
        u, _plan = w.stage()
        assert_refused_untouched(w, u, before, '(5 rows)', 1)


def test_refuses_zero_rows():
    """A table with no rows is no output: refused, nothing placed (no done record for zero output)."""
    with World() as w:
        before = w.listing()
        w.mode('zero')
        u, _plan = w.stage()
        assert_refused_untouched(w, u, before, 'no rows', 1)


def test_refuses_a_table_failing_the_content_test():
    with World() as w:
        write_table(w.table, good_rows())
        before = w.listing()
        w.mode('bad')
        u, _plan = w.stage()
        assert_refused_untouched(w, u, before, 'content test', 1)


def _content(rows, n_served=1, cols=COLS, summary=True):
    d = tempfile.mkdtemp(prefix='served_tiers_ct_')
    try:
        p = os.path.join(d, 'served_tiers.csv')
        write_table(p, rows, cols)
        if summary:
            with open(os.path.join(d, 'chain_summary.json'), 'w') as f:
                json.dump({'n_served': n_served}, f)
        return TP.served_tiers_rows(p)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_content_test_passes_a_consistent_table():
    ok, n, detail = _content(good_rows())
    assert ok and n == 3, detail
    assert '10s 1 of 1, 1min 0 of 1' in detail, detail


def test_content_test_failures():
    """Each rule of the content test refuses a table that breaks it (one break per case)."""
    def broken(fn):
        rows = good_rows()
        fn(rows)
        return rows

    def set_(i, **kv):
        return lambda rows: rows[i].update(kv)

    cases = {
        'columns': (good_rows(), dict(cols=COLS[:-1])),
        'tier': (broken(set_(0, tier='5min')), {}),
        'outcome': (broken(set_(0, chain_outcome='kept')), {}),
        'qualifies without a route': (broken(set_(0, routes_tier='')), {}),
        'a failure reason on a qualifying tier': (broken(set_(0, fail_reason='x')), {}),
        'add_without_products': (broken(set_(2, add_without_products='1')), {}),
        'a (source, tier) twice': (broken(lambda rows: rows.append(dict(rows[1]))), {}),
        'two outcomes in a source': (broken(set_(1, chain_outcome='no-route', add_without_products='0')), {}),
        'identity_qualifies_tiers': (broken(set_(1, identity_qualifies_tiers='10s;1min')), {}),
        'n_members per tier': (broken(set_(0, n_members_1min='5')), {}),
        'no identity on a served source': (broken(lambda rows: [r.update(identity_jname='') for r in rows[:2]]), {}),
        'one identity on two served sources': (
            broken(lambda rows: rows[2].update(chain_outcome='served', identity_jname='J0101+0101')), dict(n_served=2)),
        'n_served': (good_rows(), dict(n_served=2)),
        'no chain summary': (good_rows(), dict(summary=False)),
        'no rows': ([], {}),
    }
    for name, (rows, kw) in cases.items():
        ok, _n, detail = _content(rows, **kw)
        assert not ok, (name, detail)


def test_production_wrapper_of_the_last_call():
    """_last_call_lines reads only the last call's section of a log that holds several calls."""
    d = tempfile.mkdtemp(prefix='served_tiers_log_')
    try:
        p = os.path.join(d, 'tierserve.log')
        with open(p, 'w') as f:
            f.write('\n===== t0 py a\nself-check: 1 of 1\nwrote x (3 rows)\n\n===== t1 py b\nREFUSED: no\n')
        assert P._last_call_lines(p) == ['REFUSED: no']
        assert P._last_call_lines(os.path.join(d, 'none.log')) == []
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main(argv):
    if _IMPORT_ERROR is not None:
        print(f'imports missing: {type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}')
        return 2
    tests = [(n, f) for n, f in sorted(globals().items(), key=lambda kv: inspect.getsourcelines(kv[1])[1]
                                       if callable(kv[1]) and kv[0].startswith('test_') else 0)
             if callable(f) and n.startswith('test_') and (not argv or n in argv)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f'PASS {n}')
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f'FAIL {n}: {type(e).__name__}: {str(e)[:600]}')
    print(f'{len(tests) - failed} of {len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
