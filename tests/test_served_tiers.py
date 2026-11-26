#!/usr/bin/env python3
"""Data-free tests of the snapshot chain's per-tier serving stage (cutover row 71; lib/catalog/tier_snapshot.st_tierserve, gate
P-G27) and of the content test of its table (lib/catalog/tests_snapshot.served_tiers_rows, class served_tiers).

The stage runs production's clv15 phase3_snapshot_tierserve.py in place (here a stub in a temporary code tree, its behavior set by
the environment variable TSTUB_MODE) with --samples-dir = the run's recorded copies (run_samples.copy_dir), verified against
their record before the call and again after it. The table is written to a temporary name beside RUN/served_tiers.csv and placed
only after the call's self-check line, the row count it reported and the content test; any refusal leaves no table and no
temporary file, and an existing table is kept. No record, a copy with another md5, the chain's outputs missing: refused before
the call (the stub is never run). Nothing falls back to the live catalogs. The content test compares the table with the chain's
record beside it by tierserve's own definition (a served source with no kept member has no row); its tiers and outcomes are
bound to the module's TIERS / OUTCOMES ([lit]). The call's log is replaced by each call (production's `>`), and the recorded
summary is exactly the lines production's grep keeps.

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
    """Three rows of a consistent table, the rows tierserve gives for RECORD: source 0 served (10s qualifies, 1min does not;
    one 1min member stripped by the reflection-band floor), source 1 not designated (10s, does not qualify); source 2, served
    by the chain with no kept member, has no row (production deletes it)."""
    def row(src, jn, outcome, tier, nm, ns, routes, q, reason, qt, n10, n1):
        return dict(zip(COLS, [src, jn, outcome, tier, nm, ns, 'c%s%s' % (src, tier), '9.5', 'I', 'ok', '7.0', nm, '1',
                               routes, q, reason, '1', '0', str(int(q == '1' and outcome != 'served')), qt, n10, '9.5', n1,
                               '9.5' if n1 != '0' else '', '']))
    return [row('0', 'J0101+0101', 'served', '10s', '3', '0', 'Red;Blue', '1', '', '10s', '3', '2'),
            row('0', 'J0101+0101', 'served', '1min', '2', '1', '', '0', 'singleton: snr 6.1 < 7.0', '10s', '3', '2'),
            row('1', 'J0202+0202', 'not-designated', '10s', '1', '0', '', '0', 'singleton: snr 5.0 < 7.0', '', '1', '0')]


# the chain's record (detection_record.csv.gz) whose rows by tierserve's definition are good_rows(): (cand_id, src_index,
# src_outcome, served, first_removed_by, tier)
RECORD_COLS = ['cand_id', 'src_index', 'src_outcome', 'served', 'first_removed_by', 'tier']
RECORD = [['c1', '0', 'served', '1', '', '10s'], ['c2', '0', 'served', '1', '', '10s'], ['c3', '0', 'served', '1', '', '10s'],
          ['c4', '0', 'served', '1', '', '1min'], ['c5', '0', 'served', '1', '', '1min'],
          ['c6', '0', 'served', '0', 'refl-floor', '1min'],          # stripped by the floor: n_stripped_tier
          ['c7', '0', 'served', '0', 'ghost', '10s'],               # neither kept nor stripped
          ['c8', '1', 'not-designated', '0', 'not-designated', '10s'],     # kept: removed by its own outcome
          ['c9', '2', 'served', '0', 'refl-floor', '10s'],          # a served source with no kept member: no row
          ['c10', '2', 'served', '0', 'vhf', '1min'],
          ['c11', '', '', '0', 'refl-floor', '10s'],                # no source
          ['c12', '3', 'merged', '1', '', '10s']]                   # an outcome tierserve does not read


def write_record(run, rows=None):
    with gzip.open(os.path.join(run, 'detection_record.csv.gz'), 'wt', newline='') as f:
        w = csv.writer(f)
        w.writerow(RECORD_COLS)
        w.writerows(RECORD if rows is None else rows)


def write_table(path, rows, cols=COLS):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


STUB = r'''
import csv, json, os, sys
TIERS = ("10s", "1min")
OUTCOMES = ("served", "no-route", "not-designated", "fit-pending")
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
for ln in REPORT:                              # production's report(): the summary keeps 4 of these lines
    print(ln)
print("wrote %s (%d rows)" % (out, len(rows) + (2 if mode == "rowcount" else 0)))
'''
REPORT = ["served identities 1: qualify in 10s only 1, 1min only 0, both 0, NEITHER 0",
          "  tier 10s: 1 served identities hold members, 1 qualify, 0 do not",
          "  tier 1min: 1 served identities hold members, 0 qualify, 1 do not",
          "served identities qualifying in NO tier (dropped from every tier page; identity_note): 0 {}",
          "CANNOT ADD (qualify in a tier, not served by the chain, no review products): 0 sources ()",
          "(source, tier) rows whose decisive comparison is at a floor within the record's 3-decimal SNR rounding and not the "
          "chain's own: 0",
          "  not-designated  10s   qualifies=0 singleton        1"]
STUB = STUB.replace('for ln in REPORT:', 'REPORT = %r\nfor ln in REPORT:' % REPORT)
SELF_CHECK = 'self-check: 1 of 1 served sources reproduced (mi_tiers, catalog, routes, best member)'
SUMMARY = [SELF_CHECK, REPORT[0], REPORT[3], REPORT[4]]      # snapshot_rechain_batch.sh:562 grep -hE keeps these


class World(object):
    """A live samples directory, a code tree holding the stub tierserve (the site's clv15 root while the World lasts), and a
    chain run (chain_summary.json with n_served 2 - source 2 served without a kept member - , detection_record.csv.gz =
    RECORD, the sample-catalog record of the chain). The work directory and the records lie outside the run, so the run's
    listing shows exactly what the stage placed there."""

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
            json.dump({'n_served': 2, 'cells': ['14h_2025-03-09']}, f)
        write_record(self.run)
        if record_samples:
            RS.write(self.run, self.src, NAMES, 'chain')
        self._py = dict(CFG.site.PYTHON)
        CFG.site.PYTHON['py36'] = sys.executable
        self._roots = CFG.site.LEGACY_CODE_ROOTS        # the [lit] binding reads TIERS / OUTCOMES from the stub's tree
        CFG.site.LEGACY_CODE_ROOTS = [self.clv] + [r for r in self._roots if not r.rstrip('/').endswith('/clv15')]
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
        CFG.site.LEGACY_CODE_ROOTS = self._roots
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
        return os.path.join(self.run, RS.SERVED_TIERS)

    @property
    def log(self):
        return os.path.join(self.work, P.TIERSERVE_LOG)


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
        assert m['report'] == SUMMARY and m['log'] == w.log, m      # exactly production's grep's lines
        assert u.meta['legacy_calls'][0]['log'] == w.log, u.meta['legacy_calls']
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


def _content(rows, cols=COLS, record=None, record_cols=None):
    """The content test on ``rows`` with the chain's record beside the table (RECORD by default; False: none)."""
    d = tempfile.mkdtemp(prefix='served_tiers_ct_')
    try:
        p = os.path.join(d, RS.SERVED_TIERS)
        write_table(p, rows, cols)
        if record is not False:
            write_record(d, record)
            if record_cols:
                with gzip.open(os.path.join(d, 'detection_record.csv.gz'), 'wt', newline='') as f:
                    csv.writer(f).writerow(record_cols)
        return TP.served_tiers_rows(p)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_content_test_passes_a_consistent_table():
    """The table equals the record's rows by tierserve's definition - source 2, served by the chain with no kept member, has
    no row and is accepted (the chain summary's n_served is not compared: production deletes such a source)."""
    ok, n, detail = _content(good_rows())
    assert ok and n == 3, detail
    assert '10s 1 of 1, 1min 0 of 1' in detail and '3 (source, tier) rows of 2 sources with a kept member (sources with ' \
        'no kept member, no row: 1), equal to the table' in detail, detail


def test_content_test_rows_by_the_record():
    """served_tiers_expected is production's load + evaluate definition; a served source with no kept member is accepted
    without a row, while a (source, tier) of the record without a row, a row the record does not give, and a row whose
    outcome or member counts differ are refused, each named."""
    d = tempfile.mkdtemp(prefix='served_tiers_exp_')
    try:
        write_record(d)
        exp, n_with, n_none = TP.served_tiers_expected(os.path.join(d, 'detection_record.csv.gz'), ('10s', '1min'),
                                                       ('served', 'no-route', 'not-designated', 'fit-pending'))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    assert exp == {('0', '10s'): ('served', '3', '0'), ('0', '1min'): ('served', '2', '1'),
                   ('1', '10s'): ('not-designated', '1', '0')} and (n_with, n_none) == (2, 1), (exp, n_with, n_none)
    # a served source with only stripped members, and one whose kept members lie in no written tier: no row, accepted
    more = RECORD + [['c13', '4', 'served', '0', 'refl-floor', '1min'], ['c14', '5', 'served', '1', '', '10min']]
    ok, _n, detail = _content(good_rows(), record=more)
    assert ok, detail
    # a missing row: the record gives (6, 10s), the table lacks it
    ok, _n, detail = _content(good_rows(), record=RECORD + [['c15', '6', 'no-route', '0', 'no-route', '10s']])
    assert not ok and "of the record without a row (e.g. [('6', '10s')])" in detail, detail
    # an extra row: the record no longer gives source 1
    ok, _n, detail = _content(good_rows(), record=[r for r in RECORD if r[1] != '1'])
    assert not ok and "rows the record does not give (e.g. [('1', '10s')])" in detail, detail
    # member counts: a fourth kept 10s member of source 0; a stripped one that is not
    ok, _n, detail = _content(good_rows(), record=RECORD + [['c16', '0', 'served', '1', '', '10s']])
    assert not ok and "0 10s: table ('served', '3', '0'), record ('served', '4', '0')" in detail, detail
    ok, _n, detail = _content(good_rows(), record=[r if r[0] != 'c6' else r[:4] + ['ghost', '1min'] for r in RECORD])
    assert not ok and "0 1min: table ('served', '2', '1'), record ('served', '2', '0')" in detail, detail
    # the chain outcome of a source
    ok, _n, detail = _content(good_rows(), record=[r if r[1] != '1' else [r[0], '1', 'no-route', '0', 'no-route', '10s']
                                                   for r in RECORD])
    assert not ok and "1 10s: table ('not-designated', '1', '0'), record ('no-route', '1', '0')" in detail, detail


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
            broken(lambda rows: rows[2].update(chain_outcome='served', identity_jname='J0101+0101')), {}),
        'no record beside the table': (good_rows(), dict(record=False)),
        'a record without src_outcome': (good_rows(), dict(record_cols=['cand_id', 'src_index', 'served',
                                                                        'first_removed_by', 'tier'])),
        'no rows': ([], {}),
    }
    for name, (rows, kw) in cases.items():
        ok, _n, detail = _content(rows, **kw)
        assert not ok, (name, detail)


def test_log_replaced_per_call():
    """WORK/tierserve.log holds the last call's output alone (production's `>`, placed through core/placement: no temporary
    file left), also when that call refused; the summary is read from the call's own output, exactly the lines production's
    grep keeps."""
    with World() as w:
        u, _plan = w.stage()
        assert u.status == 'done', u.reason
        first = open(w.log).read()
        u, _plan = w.stage()
        assert u.status == 'done', u.reason
        text = open(w.log).read()
        assert text.count('===== ') == 1 and text.count(SELF_CHECK) == 1 and text != first, text
        assert u.meta['tierserve']['report'] == SUMMARY, u.meta['tierserve']['report']
        assert [ln for ln in text.splitlines() if P.TIERSERVE_SUMMARY.search(ln)] == SUMMARY
        w.mode('refuse')
        u, _plan = w.stage()
        assert u.status == 'refused', u.reason
        text = open(w.log).read()
        assert text.count('===== ') == 1 and 'REFUSED: the reconstruction' in text and SELF_CHECK not in text, text
        assert sorted(os.listdir(w.work)) == [P.TIERSERVE_LOG], os.listdir(w.work)


def test_binding_of_tiers_and_outcomes():
    """catalog_snapshot.TIERSERVE_TIERS / _OUTCOMES (the content test's) are bound [lit] to tierserve's TIERS / OUTCOMES:
    in tier_snapshot.LIT and PROTECTED_KEYS, equal on the stub; a module whose TIERS differ, or that lacks OUTCOMES, is
    refused by the stage before the call (nothing written, the stub never run)."""
    for k in P.TIERSERVE_LIT:
        assert P.LIT[k] == ('clv15', P.TIERSERVE, {'catalog_snapshot.TIERSERVE_TIERS': 'TIERS',
                                                   'catalog_snapshot.TIERSERVE_OUTCOMES': 'OUTCOMES'}[k]), P.LIT[k]
        assert k.split('.')[1] in CFG.catalog_snapshot.PROTECTED_KEYS, k
    for old, new, key, status in (('TIERS = ("10s", "1min")', 'TIERS = ("10s", "1min", "10min")', 'TIERSERVE_TIERS', 'MISMATCH'),
                                  ('OUTCOMES = (', 'OUTCOMES_RENAMED = (', 'TIERSERVE_OUTCOMES', 'unbound')):
        with World() as w:
            rows = tier_sublst.binding_check({k: P.LIT[k] for k in P.TIERSERVE_LIT})
            assert [r[1] for r in rows] == ['ok', 'ok'], rows
            stub = os.path.join(w.clv, P.TIERSERVE)
            txt = open(stub).read()
            assert txt.count(old) == 1
            open(stub, 'w').write(txt.replace(old, new))
            before = w.listing()
            u, _plan = w.stage()
            assert_refused_untouched(w, u, before, 'catalog_snapshot.%s %s' % (key, status), 0)
            assert not os.path.exists(w.table)


def test_active_list_kept_until_tierserve_ran():
    """The production active list (BATCH_DIR/rechain_active/<NAME>.cells) is removed when the chain's LAST stage, tierserve,
    ran in the pass (or a unit did not finish): `--until test` keeps it, `--from tierserve` removes it. The stage functions
    are stand-ins that write one output each."""
    root = tempfile.mkdtemp(prefix='served_tiers_active_')
    prod, batch, name = os.path.join(root, 'prodroot'), os.path.join(root, 'pbatch'), 'Run_SNAPZ'
    os.makedirs(prod)
    cf = os.path.join(root, 'cells.txt')
    with open(cf, 'w') as f:
        f.write('14h_2025-03-09\n')

    def stand_in(stage):
        def f(ctx, s, plan):
            p = os.path.join(ctx.RUN, 'stand_in_%s.txt' % stage)
            os.makedirs(ctx.RUN, exist_ok=True)
            with open(p, 'w') as g:
                g.write(stage + '\n')
            s.u.add_output(p, 'stand_in', test=False)
        return f
    c, paths = CFG.catalog_snapshot, CFG.paths
    saved = (paths.PRODUCTION_ROOTS, c.BATCH_DIR, paths.OPS_ROOT, dict(CFG.site.PYTHON), dict(P.FUNCS),
             tier_sublst.check_from, CFG.site.LEGACY_CODE_ROOTS)
    try:
        CFG.site.LEGACY_CODE_ROOTS = [os.path.join(root, 'code', 'clv15'), os.path.join(root, 'code', 'exopipe_ops')]
        paths.PRODUCTION_ROOTS = tuple(saved[0]) + (prod,)
        c.BATCH_DIR, paths.OPS_ROOT = batch, os.path.join(root, 'ops')       # the driver log of a production run: here
        CFG.site.PYTHON['py36'] = sys.executable
        P.FUNCS.update({st: stand_in(st) for st in P.STAGES})
        tier_sublst.check_from = lambda *a, **k: []          # --from: the earlier stages' records are not the point here
        active = os.path.join(batch, 'rechain_active', '%s.cells' % name)
        got = {}
        for frm, until in ((None, 'test'), ('tierserve', None), (None, 'chain')):
            os.makedirs(os.path.dirname(active), exist_ok=True)
            shutil.copy(cf, active)
            args = types.SimpleNamespace(run_label=name, cells='@' + cf, snap_analysis=None, stages=None, from_stage=frm,
                                         until_stage=until, snap_item=None)
            rr = record.Run('catalog', ['test_served_tiers'], os.path.join(prod, name), CFG)
            P.run(args, rr)
            assert [u.status for u in rr.units] == ['done'] * len(rr.units) and rr.units, [(u.name, u.status, u.reason)
                                                                                           for u in rr.units]
            got[(frm, until)] = (os.path.exists(active), rr.units[-1].meta['stage'])
        assert got == {(None, 'test'): (True, 'test'), ('tierserve', None): (False, 'tierserve'),
                       (None, 'chain'): (True, 'chain')}, got
    finally:
        (paths.PRODUCTION_ROOTS, c.BATCH_DIR, paths.OPS_ROOT) = saved[:3]
        CFG.site.PYTHON.clear()
        CFG.site.PYTHON.update(saved[3])
        P.FUNCS.clear()
        P.FUNCS.update(saved[4])
        tier_sublst.check_from = saved[5]
        CFG.site.LEGACY_CODE_ROOTS = saved[6]
        shutil.rmtree(root, ignore_errors=True)


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
