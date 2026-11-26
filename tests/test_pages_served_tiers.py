#!/usr/bin/env python3
"""Data-free tests of the snapshot pages' per-tier serving (cutover row 72; lib/pages/tier_snapshot.stage_build): the pages
step passes the run's served_tiers.csv to the page builder exactly as production does (ops snapshot_rechain_batch.sh:607-612,
then `pertier_hook full` :570-587, :639), and refuses before anything is written when the table is missing or fails its
content test (class served_tiers, ExoPipe PR 185) - gate P23, which honors --waive like every gate; P24 on a linked backup
directory; the build's inputs (the record and the Card 2 back-fill's roots); the dry run's P23 prediction with --from-run; the
pages pass's default process count (production's 12). Also the content test of the Card 2 back-fill's files (class
snapshot_card2, lib/pages/tests.card2_backfill): G8's own row rule (deep_omission.card2_row_check), each file read once.

The in-place scripts are stubs in a temporary code tree (ops and clv15 legacy roots), each appending its argv to a calls file:
the page builder (a pass writes two identities' views; the views-only rewrite keeps the identities the table qualifies in the
10-s tier), apply_served_check.py (backup / restore of the views), the views content test, the Card 2 back-fill and the run
content test. Their failure modes are set by the environment variable PSTUB_MODE. The synthetic run's served_tiers.csv and
detection_record.csv.gz are those of tests/test_served_tiers.py (the record with the columns the Card 2 back-fill reads).

Run: python tests/test_pages_served_tiers.py [test names]   (exit 0 = all passed, 1 = a failure, 2 = imports missing)
"""
import collections
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import types
from contextlib import redirect_stdout

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tests'))
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

_IMPORT_ERROR = None
try:
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import record, validate
    from exopipe.phase3.lib.catalog import deep_omission as DO, run_samples as RS
    from exopipe.phase3.lib.pages import mirror, tests as PT, tier_snapshot as TSN
    import test_served_tiers as TS
except Exception as e:  # noqa: BLE001
    _IMPORT_ERROR = e
    if __name__ != '__main__':
        raise

NAME = 'Run_SNAPTEST_20261126'
# the record: test_served_tiers.RECORD (what tierserve evaluates) + the columns the Card 2 back-fill copies; c1-c3 are Stokes V
# association-pool rows of J0101+0101 (served), c8 of J0202+0202 (not served), c7 Stokes V outside the pool, the rest Stokes I
EXTRA_COLS = ['det_uid', 'stokes', 'in_assoc_pool', 'identity_jname', 'peak_jyb']
EXTRA = {'c1': ('u1', 'V', '1', 'J0101+0101', '0.9'), 'c2': ('u2', 'V', '1', 'J0101+0101', '-0.2'),
         'c3': ('u3', 'V', '1', 'J0101+0101', '0.5'), 'c7': ('u7', 'V', '0', 'J0101+0101', '0.3'),
         'c8': ('u8', 'V', '1', 'J0202+0202', '0.4')}
I_HERE = {'u1': '30', 'u2': '2', 'u3': ''}    # Stokes I under the detection (Jy): u1 0.03 (flagged), u2 0.1; u3 no Master
CARD2_COLS = ['det_uid', 'cand_id', 'ra_peak', 'dec_peak', 'cell', 'band', 'tier', 'ts', 'instant_key', 'snr', 'peak_jyb', 'ra',
              'dec', 'served', 'identity_jname', 'first_removed_by', 'i_master_pix_jyb', 'scale', 'scale_source', 'i_here_jy',
              'v_over_i_here', 'card2_would_remove', 'card2_ratio']      # clv15 phase3_snapshot_card2.py COLUMNS

STUB_PAGES = r'''
import csv, os, sys
a = sys.argv[1:]
def opt(k):
    return a[a.index(k) + 1]
open(os.environ["PSTUB_CALLS"], "a").write("pages " + " ".join(a) + "\n")
mode = os.environ.get("PSTUB_MODE", "")
run = opt("--run")
ids = [l.strip() for l in open(os.path.join(run, "queue.txt")) if l.strip() and not l.startswith("#")]
if "--views-only" in a:
    keep = [r["identity_jname"] for r in csv.DictReader(open(opt("--served-tiers")))
            if r["tier"] == "10s" and r["qualifies"] == "1"]
    ids = [j for j in ids if j in keep]
    if "views" in mode.split(","):      # a half-written rewrite, then the builder's refusal (snapshot_pages.py:892)
        open(os.path.join(run, "queue_V_10s.txt"), "w").write("# queue_name: half\n")
        print("WARNING: a warning line")
        sys.exit("REFUSED: views-only pass, 1 of 2 kept panels not on disk (list: %s/missing_panels.csv)" % run)
elif "pages" in mode.split(","):
    sys.exit("REFUSED: 2 panels missing")
else:
    open(os.path.join(run, "missing_panels.csv"), "w").close()
    for j in ids:
        d = os.path.join(run, "candidates", "V", j + "_23MHz")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "detections.csv"), "w") as f:
            f.write("lst,date,cand_id,ra,dec,pos_err_stat_arcsec,pos_err_arcsec\n13h,2025-04-21,c%s,10.0,1.0,30,60\n" % j)
for view in ("source_catalogs_V", "source_catalogs_V_10s"):
    os.makedirs(os.path.join(run, view), exist_ok=True)
    with open(os.path.join(run, view, "23MHz_sources_survivors.csv"), "w") as f:
        f.write("identity_id,review_dir,ra,n_det,n_det_tier\n")
        for j in ids:
            f.write("%s,candidates/V/%s_23MHz,10.0,1,1\n" % (j, j))
with open(os.path.join(run, "queue_V.txt"), "w") as f:
    f.write("# run: %s (snapshot tier, Stokes V)\n# selection: stub\n" % os.path.basename(run) + "".join(j + "\n" for j in ids))
with open(os.path.join(run, "queue_V_10s.txt"), "w") as f:
    f.write("# queue_name: snapshot Stokes V, 10-s frame tier\n# selection: stub\n" + "".join(j + "\n" for j in ids))
print("done: %d identities" % len(ids))
'''

STUB_CHECK = r'''
import glob, os, shutil, sys
open(os.environ["PSTUB_CALLS"], "a").write("check " + " ".join(sys.argv[1:]) + "\n")
cmd, src, dst = sys.argv[1:4]
if cmd == "restore" and "restore" in os.environ.get("PSTUB_MODE", "").split(","):
    sys.exit("REFUSED: the backup is not complete")
for p in glob.glob(os.path.join(src, "queue_[IV]*.txt")) + glob.glob(os.path.join(src, "source_catalogs_[IV]*")):
    q = os.path.join(dst, os.path.basename(p))
    if os.path.isdir(p):
        shutil.rmtree(q, ignore_errors=True)
        shutil.copytree(p, q)
    else:
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(p, q)
print("%s: %s -> %s" % (cmd, src, dst))
'''

STUB_VTEST = r'''
import os, sys
open(os.environ["PSTUB_CALLS"], "a").write("vtest " + " ".join(sys.argv[1:]) + "\n")
mode = os.environ.get("PSTUB_MODE", "").split(",")
print("V 10s : queue 1 identities, catalog rows 1; T1 non-qualifying 0; T2 qualifying missing 0; T3 bad rows 0; T5 2 - 1 = 1")
print("V union: queue_V.txt 1 identities, union of tier queues 1")
if "vtest" in mode:
    print("PERTIER-VIEWS-FAIL: T1 V 10s e.g. J0202+0202")
    sys.exit(1)
if "noverdict" not in mode:
    print("PERTIER-VIEWS-OK")
'''

STUB_CARD2 = r'''
import csv, gzip, json, os, sys
a = sys.argv[1:]
open(os.environ["PSTUB_CALLS"], "a").write("card2 " + " ".join(a) + "\n")
if "card2" in os.environ.get("PSTUB_MODE", "").split(","):
    print("warning: a warning line")
    sys.exit("REFUSED: record run_status is not 'complete': {'partial': 1}")
run, ratio = a[a.index("--run") + 1], float(a[a.index("--ratio") + 1])
cols, ihere = json.loads(os.environ["PSTUB_CARD2_COLS"]), json.loads(os.environ["PSTUB_IHERE"])
out = []
for r in csv.DictReader(gzip.open(os.path.join(run, "detection_record.csv.gz"), "rt")):
    if r["stokes"] == "V" and r["in_assoc_pool"] == "1":
        ih = ihere.get(r["det_uid"], "")
        vi = abs(float(r["peak_jyb"])) / float(ih) if ih else None
        o = dict.fromkeys(cols, "")
        o.update(det_uid=r["det_uid"], cand_id=r["cand_id"], tier=r["tier"], served=r["served"], identity_jname=r["identity_jname"],
                 peak_jyb=r["peak_jyb"], ra="10.0", dec="1.0", i_here_jy=ih, card2_ratio="%g" % ratio,
                 v_over_i_here=("%.5f" % vi) if vi is not None else "",
                 card2_would_remove="" if vi is None else ("1" if vi <= ratio else "0"))
        out.append(o)
with open(os.path.join(run, "card2_vleak_columns.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "unit", "meaning"])
    w.writerows([c, "", "stub"] for c in cols)
with open(os.path.join(run, "card2_vleak.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(out)
srv = [o for o in out if o["served"] == "1"]
with open(os.path.join(run, "card2_filter.csv"), "w", newline="") as f:
    f.write("# Card 2 (stub): ratio %g\n" % ratio)
    w = csv.DictWriter(f, fieldnames=["cand_id", "ra", "dec", "v_over_i_here", "card2_would_remove"], extrasaction="ignore")
    w.writeheader()
    w.writerows(srv)
print("CARD2 %s: %d V pool rows (%d served); would remove %d" % (os.path.basename(run), len(out), len(srv),
      sum(1 for o in srv if o["card2_would_remove"] == "1")))
'''

STUB_CT = r'''
import os, sys
open(os.environ["PSTUB_CALLS"], "a").write("ct " + " ".join(sys.argv[1:]) + "\n")
print("R1-R7 passed")
'''


def md5(p):
    with open(p, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def write_record(run, rows=None):
    with gzip.open(os.path.join(run, 'detection_record.csv.gz'), 'wt', newline='') as f:
        w = csv.writer(f)
        w.writerow(TS.RECORD_COLS + EXTRA_COLS)
        for r in (TS.RECORD if rows is None else rows):
            w.writerow(r + list(EXTRA.get(r[0], ('u' + r[0][1:], 'I', '1', '', '1.0'))))


class World(object):
    """Stub legacy roots, a finished chain run (chain_summary.json, queue.txt of two identities, the record, served_tiers.csv
    = test_served_tiers.good_rows(): J0101+0101 qualifies in the 10-s tier, J0202+0202 does not) and a fit store. The unit is
    run in-process with a record.Run under <root>/out, the pages step's W (RUN = W/<name>)."""

    def __init__(self, table=True):
        self.root = tempfile.mkdtemp(prefix='pages_served_tiers_')
        self.ops, self.clv = os.path.join(self.root, 'ops'), os.path.join(self.root, 'clv15')
        self.W = os.path.join(self.root, 'out')
        self.run = os.path.join(self.W, NAME)
        self.calls = os.path.join(self.root, 'calls.txt')
        for d in (self.ops, self.clv, self.run):
            os.makedirs(d)
        for root, fn, src in ((self.clv, 'phase3_snapshot_pages.py', STUB_PAGES), (self.ops, 'apply_served_check.py', STUB_CHECK),
                              (self.ops, 'pertier_views_content_test.py', STUB_VTEST),
                              (self.clv, 'phase3_snapshot_card2.py', STUB_CARD2),
                              (self.ops, 'snapshot_rechain_content_test.py', STUB_CT),
                              (self.ops, 'classify_transients.py', "PNG_RE = r'^rank\\d+_.+\\.png$'\n")):
            with open(os.path.join(root, fn), 'w') as f:
                f.write(src)
        with open(os.path.join(self.run, 'chain_summary.json'), 'w') as f:
            json.dump({'n_served': 2, 'cells': ['13h_2025-04-21']}, f)
        with open(os.path.join(self.run, 'queue.txt'), 'w') as f:
            f.write('J0101+0101\nJ0202+0202\n')
        write_record(self.run)
        if table:
            TS.write_table(self.table, TS.good_rows())
        self.store = os.path.join(self.W, 'fits_store.csv')
        with open(self.store, 'w') as f:
            f.write('cand_id\n')
        # the Card 2 back-fill's roots (pages_snapshot.MASTERS_ROOT, catalog_snapshot.LIGHTCURVES_ROOT, MASTER_CATALOGS):
        # empty directories of the world, inputs of the build unit
        self.card2_roots = [os.path.join(self.root, d) for d in ('masters', 'lightcurves', 'master_catalogs')]
        for d in self.card2_roots:
            os.makedirs(d)
        self._roots = (CFG.pages_snapshot.MASTERS_ROOT, CFG.catalog_snapshot.LIGHTCURVES_ROOT,
                       CFG.catalog_snapshot.MASTER_CATALOGS)
        (CFG.pages_snapshot.MASTERS_ROOT, CFG.catalog_snapshot.LIGHTCURVES_ROOT,
         CFG.catalog_snapshot.MASTER_CATALOGS) = self.card2_roots
        self._py = dict(CFG.site.PYTHON)
        CFG.site.PYTHON['py36'] = sys.executable
        self._env = {k: os.environ.get(k) for k in ('PSTUB_MODE', 'PSTUB_CALLS', 'PSTUB_CARD2_COLS', 'PSTUB_IHERE')}
        os.environ.update(PSTUB_CALLS=self.calls, PSTUB_CARD2_COLS=json.dumps(CARD2_COLS), PSTUB_IHERE=json.dumps(I_HERE))
        self.mode('')
        self._gate, self._plan = mirror.gate, mirror.plan_lines
        mirror.gate = lambda *a, **kw: None          # the stubs carry none of the mirrored constants
        mirror.plan_lines = lambda *a, **kw: []

    def mode(self, m):
        os.environ['PSTUB_MODE'] = m

    def __enter__(self):
        return self

    def __exit__(self, *a):
        mirror.gate, mirror.plan_lines = self._gate, self._plan
        CFG.pages_snapshot.MASTERS_ROOT, CFG.catalog_snapshot.LIGHTCURVES_ROOT, CFG.catalog_snapshot.MASTER_CATALOGS = self._roots
        CFG.site.PYTHON.clear()
        CFG.site.PYTHON.update(self._py)
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.root, ignore_errors=True)

    @property
    def table(self):
        return os.path.join(self.run, RS.SERVED_TIERS)

    def ctx(self, keep=False, from_run=None):
        return types.SimpleNamespace(name=NAME, out=self.W, RUN=self.run, W=self.W, roots={'ops': self.ops, 'clv15': self.clv},
                                     stages=['build'], store=self.store, keep=keep, nproc=3, from_run=from_run, epochs=None)

    def build(self, keep=False, waivers=None):
        rr = record.Run('pages', ['test_pages_served_tiers'], self.W, CFG, waivers=waivers)
        done = {}
        TSN.stage_build(self.ctx(keep), rr, done)
        u = rr.units[-1]
        assert (u.status == 'done') == (f'{NAME}:build' in done), (u.status, done)
        return u

    def calls_made(self):
        if not os.path.exists(self.calls):
            return []
        return [ln.split(' ', 1) for ln in open(self.calls).read().splitlines()]

    def dry(self, **kw):
        f = io.StringIO()
        with redirect_stdout(f):
            rc = TSN.dry_run(self.ctx(**kw))
        return rc, f.getvalue()

    def pages_left(self):
        """Every page product of the run: page directories, the Stokes queues, the views, the Card 2 files."""
        left = []
        for d, _ds, fs in os.walk(self.run):
            for f in fs:
                rel = os.path.relpath(os.path.join(d, f), self.run)
                if (rel.startswith(('candidates', 'source_catalogs_', 'queue_')) or f.startswith('card2_')
                        or f == 'missing_panels.csv'):
                    left.append(rel)
        return sorted(left)


def queue_ids(p):
    return [ln.strip() for ln in open(p) if ln.strip() and not ln.startswith('#')]


def expected_argv(w, st, views_only=False, keep=False):
    """Production's argument lists (snapshot_rechain_batch.sh:609-610 pages(), :575-576 the hook's views-only call)."""
    S = CFG.pages_snapshot
    head = ['--run', w.run, '--proto-root', CFG.paths.SNAPSHOT_FLEET_OUT, '--masters-root', S.MASTERS_ROOT, '--fit-store', w.store]
    if views_only:
        return head + ['--nproc', '2', '--served-tiers', st, '--views-only']
    return head + ['--nproc', '3'] + (['--resume'] if keep else []) + ['--served-tiers', st]


def assert_refused_nothing_built(w, u, gate, text, calls):
    assert u.status == 'refused', (u.status, u.reason)
    assert u.reason.startswith(gate) and text in u.reason, u.reason
    assert [c[0] for c in w.calls_made()] == calls, w.calls_made()


# ---------------------------------------------------------------------------------------------------------------------
def test_registration_and_config():
    """The Card 2 class is registered; the views-only rewrite's process count is production's 2; the Card 2 back-fill's
    arguments are production's defaults (clv15 phase3_snapshot_card2.py:43-48) taken from config."""
    assert [n for n, _f in validate.tests_for('snapshot_card2')] == ['card2_backfill']
    assert CFG.pages_snapshot.VIEWS_NPROC == 2
    assert PT._card2_names() == TSN.CARD2_FILES, (PT._card2_names(), TSN.CARD2_FILES)
    assert TSN.CARD2_FILES[0] == CFG.catalog_omission.SNAPSHOT_CARD2, CFG.catalog_omission.SNAPSHOT_CARD2
    assert set(PT.CARD2_READ) <= set(CARD2_COLS)
    c = types.SimpleNamespace(RUN='/x/' + NAME)
    a = TSN.card2_argv(c)
    assert a == ['--run', c.RUN, '--ratio', '0.05', '--masters-root', '/lustre/pipeline/exopipe/phase3/snapshot_masters_gen4',
                 '--lightcurves-root', '/lustre/gh/main/exopipe_ops/snapshot_lightcurves',
                 '--master-catalogs', '/lustre/gh/main/exopipe_ops/snapshot_master_catalogs',
                 '--proto-root', '/lustre/gh/main/exopipe_ops/snapshot_fleet_out'], a


def test_page_workers_default_is_productions():
    """The pages pass's process count defaults to production's 12 (snapshot_rechain_batch.sh:610 --nproc
    ${RECHAIN_PAGES_NPROC:-12}; :635-637: 24 workers reached the 400 GiB /user.slice limit and were OOM-killed); --shards
    overrides it."""
    assert CFG.pages.SNAP_PAGES_NPROC == 12, CFG.pages.SNAP_PAGES_NPROC
    with World() as w:
        for shards, want in ((None, 12), (5, 5)):
            args = types.SimpleNamespace(run_label=NAME, legacy_root=None, snap_fit_store=w.store, snap_keep_panels=False,
                                         shards=shards, from_run=None, epochs=None, stages=None, from_stage=None,
                                         until_stage=None)
            c = TSN.context(args, types.SimpleNamespace(out_root=w.W))
            a = TSN.build_argv(c, w.table)
            assert c.nproc == want and a[a.index('--nproc') + 1] == want, (c.nproc, a)


def test_missing_table_refuses_before_anything():
    """P23: no RUN/served_tiers.csv - refused before the builder (no script called), no page product in the run."""
    with World(table=False) as w:
        u = w.build()
        assert_refused_nothing_built(w, u, 'P23', 'has no served_tiers.csv', [])
        assert w.pages_left() == [], w.pages_left()
        assert not os.path.exists(os.path.join(w.W, TSN.PERTIER))


def test_failing_table_refuses_before_anything():
    """P23: a table failing its content test (a row 'qualifies 1' without a route; another with a member count the record does
    not give) - refused before the builder, the failed tests recorded on the unit (class served_tiers), nothing built."""
    for damage in ('route', 'count'):
        with World() as w:
            rows = TS.good_rows()
            if damage == 'route':
                rows[0]['routes_tier'] = ''
            else:
                rows[0]['n_members_tier'] = '4'
            TS.write_table(w.table, rows)
            u = w.build()
            assert_refused_nothing_built(w, u, 'P23', 'fails its content test', [])
            t = [x for x in u.tests if x['cls'] == 'served_tiers']
            assert t and not all(x['passed'] for x in t), t
            assert w.pages_left() == [], w.pages_left()


def test_waived_table_failure_builds():
    """P23 honors --waive like every gate: a table failing its content test builds when --waive covers each failed test (the
    failures recorded with the ruling, a note naming them); a waiver of another test does not cover it."""
    rows = TS.good_rows()
    rows[0]['routes_tier'] = ''
    with World() as w:
        TS.write_table(w.table, rows)
        failed = [t['name'] for t in validate.run('served_tiers', w.table) if not t['passed']]
        assert failed, 'the damaged table passes its content test'
        u = w.build(waivers=[(n, 'R-TEST-WAIVER') for n in failed])
        assert u.status == 'done', (u.status, u.reason)
        t = [x for x in u.tests if x['cls'] == 'served_tiers' and not x['passed']]
        assert sorted(x['name'] for x in t) == sorted(failed) and all(x.get('waived') == 'R-TEST-WAIVER' for x in t), t
        assert [c[0] for c in w.calls_made()] == ['pages', 'check', 'pages', 'vtest', 'card2', 'ct'], w.calls_made()
        assert any('waived' in n for n in u.notes), u.notes
    with World() as w:
        TS.write_table(w.table, rows)
        u = w.build(waivers=[('served_tiers.another_test', 'R-TEST-WAIVER')])
        assert_refused_nothing_built(w, u, 'P23', 'fails its content test', [])


def test_success_passes_the_table_as_production():
    """A passing table: the builder gets --served-tiers RUN/served_tiers.csv after --nproc N [--resume] (production's order),
    then production's hook in its order - backup of the views to W/pertier/pre_full, the views-only rewrite (--nproc 2
    --served-tiers ... --views-only), the views test on (RUN, table, backup), the Card 2 back-fill - then the run content test.
    The 10-s view holds only the identity qualifying there; the Card 2 files are outputs, content-tested; the hook's lines are
    recorded."""
    for keep in (False, True):
        with World() as w:
            u = w.build(keep=keep)
            assert u.status == 'done', (u.status, u.reason, [t for t in u.tests if not t['passed']])
            pre = os.path.join(w.W, TSN.PERTIER, 'pre_full')
            calls = w.calls_made()
            assert [c[0] for c in calls] == ['pages', 'check', 'pages', 'vtest', 'card2', 'ct'], calls
            assert calls[0][1].split() == expected_argv(w, w.table, keep=keep), calls[0]
            assert calls[1][1].split() == ['backup', w.run, pre], calls[1]
            assert calls[2][1].split() == expected_argv(w, w.table, views_only=True), calls[2]
            assert calls[3][1].split() == [w.run, w.table, pre], calls[3]
            assert calls[4][1].split() == TSN.card2_argv(w.ctx()), calls[4]
            assert calls[5][1].split() == [w.run, '--pages'], calls[5]
            assert queue_ids(os.path.join(w.run, 'queue_V_10s.txt')) == ['J0101+0101']
            assert queue_ids(os.path.join(pre, 'queue_V_10s.txt')) == ['J0101+0101', 'J0202+0202']      # the pass's own views
            c2 = [o for o in u.outputs if o['class'] == 'snapshot_card2']
            assert [os.path.basename(o['path']) for o in c2] == list(TSN.CARD2_FILES), c2
            t = [x for x in u.tests if x['cls'] == 'snapshot_card2']
            assert len(t) == 3 and all(x['passed'] for x in t), t
            assert u.inputs and w.table in u.inputs, u.inputs
            m = u.meta['pertier']
            assert m['verdict'] == 'PERTIER-VIEWS-OK' and m['views'][0].startswith('V 10s') and m['card2'][0].startswith('CARD2'), m
            assert m['backup'] == pre
            DO.card2_removed(w.run, None, CFG.catalog_omission.CARD2_RATIO, CFG.catalog_omission.SNAPSHOT_CARD2)   # G8 passes


def test_inputs_hold_the_record_and_card2_roots():
    """The build unit's recorded inputs include detection_record.csv.gz and the Card 2 back-fill's roots (the Masters, the
    light curves, the Master catalogs), so the record's input check (core.record.resume_check: every recorded input's
    fingerprint) sees a change of any of them."""
    with World() as w:
        u = w.build()
        assert u.status == 'done', (u.status, u.reason)
        rec = record.load_record(w.W, 'pages', f'{NAME}:build')
        want = [os.path.join(w.run, 'detection_record.csv.gz')] + w.card2_roots
        for p in want:
            assert p in rec['inputs'] and record.same_fingerprint(rec['inputs'][p], record.fingerprint(p)), p
        write_record(w.run, TS.RECORD[:-1])
        for d in w.card2_roots:
            open(os.path.join(d, 'added'), 'w').close()
        for p in want:
            assert not record.same_fingerprint(rec['inputs'][p], record.fingerprint(p)), p


def test_linked_backup_dir_refuses():
    """P24: W/pertier/pre_full a symbolic link, W/pertier a symbolic link, or pre_full not a directory - refused before the
    pages pass (nothing called, nothing built), the link and what it points at left as they are; the dry run predicts it; the
    hook's own check (before its removal) refuses the same way."""
    for kind in ('pre', 'pertier', 'file'):
        with World() as w:
            target = os.path.join(w.root, 'elsewhere')
            os.makedirs(os.path.join(target, 'pre_full'))
            keep = os.path.join(target, 'pre_full', 'keep.txt')
            open(keep, 'w').write('x')
            ptd = os.path.join(w.W, TSN.PERTIER)
            pre = os.path.join(ptd, 'pre_full')
            if kind == 'pertier':
                os.symlink(target, ptd)
            else:
                os.makedirs(ptd)
                if kind == 'pre':
                    os.symlink(os.path.join(target, 'pre_full'), pre)
                else:
                    open(pre, 'w').write('x')
            text = 'not a directory' if kind == 'file' else 'is a symbolic link'
            u = w.build()
            assert_refused_nothing_built(w, u, 'P24', text, [])
            assert w.pages_left() == [] and os.path.isfile(keep)
            assert os.path.islink(ptd if kind == 'pertier' else pre) or kind == 'file'
            rc, out = w.dry()
            assert rc == 2 and 'WOULD REFUSE: build: P24' in out and text in out, out
            try:
                TSN.pertier_backup_dir(w.ctx())
                raise AssertionError('pertier_backup_dir accepted ' + kind)
            except record.Refusal as e:
                assert str(e).startswith('P24') and text in str(e), e


def test_views_rewrite_failure_restores_the_views():
    """P24: the views-only rewrite fails after writing part of a view - the views are restored from the backup (the pages pass's
    own), the stage refuses with the builder's last non-WARNING line; no Card 2, no run content test."""
    with World() as w:
        w.mode('views')
        u = w.build()
        assert_refused_nothing_built(w, u, 'P24', 'views restored from', ['pages', 'check', 'pages', 'check'])
        assert 'REFUSED: views-only pass' in u.reason, u.reason
        assert w.calls_made()[3][1].split()[0] == 'restore'
        assert queue_ids(os.path.join(w.run, 'queue_V_10s.txt')) == ['J0101+0101', 'J0202+0202']
        assert not any(f.startswith('card2_') for f in os.listdir(w.run))


def test_views_test_failure_restores_the_views():
    """P24: the views test fails (PERTIER-VIEWS-FAIL, exit 1) or exits 0 without the PERTIER-VIEWS-OK line (production's grep) -
    the rewritten views are replaced by the backup and the stage refuses with the test's last line."""
    for mode, text in (('vtest', 'PERTIER-VIEWS-FAIL: T1'), ('noverdict', 'V union: queue_V.txt')):
        with World() as w:
            w.mode(mode)
            u = w.build()
            assert_refused_nothing_built(w, u, 'P24', 'content test (' + text, ['pages', 'check', 'pages', 'vtest', 'check'])
            assert queue_ids(os.path.join(w.run, 'queue_V_10s.txt')) == ['J0101+0101', 'J0202+0202']


def test_failed_restore_is_named():
    with World() as w:
        w.mode('vtest,restore')
        u = w.build()
        assert_refused_nothing_built(w, u, 'P24', 'AND the restore from', ['pages', 'check', 'pages', 'vtest', 'check'])


def test_card2_failure_refuses():
    """P25: the Card 2 back-fill fails - refused with its last line that is not a warning; no run content test, no done."""
    with World() as w:
        w.mode('card2')
        u = w.build()
        assert_refused_nothing_built(w, u, 'P25', "record run_status is not 'complete'", ['pages', 'check', 'pages', 'vtest',
                                                                                          'card2'])


def test_pages_pass_failure_is_not_hooked():
    with World() as w:
        w.mode('pages')
        u = w.build()
        assert_refused_nothing_built(w, u, 'P18-P20', 'REFUSED: 2 panels missing', ['pages'])


def test_dry_run():
    """The dry run reports the table: missing - WOULD REFUSE P23; present - the builder's argv with --served-tiers and the
    hook's commands. Nothing is called or written."""
    for table in (False, True):
        with World(table=table) as w:
            rc, out = w.dry()
            assert w.calls_made() == [] and w.pages_left() == []
            if table:
                assert rc == 0, out
                assert ' '.join(expected_argv(w, w.table)) in out and '--views-only' in out and 'PERTIER-VIEWS-OK' in out, out
            else:
                assert rc == 2 and 'WOULD REFUSE: P23' in out, out
    # --from-run: the table the build will read is the source run's (stagein copies it) - P23 predicted from it, nothing copied
    for run_table, src_table in ((False, True), (True, False), (False, False)):
        with World(table=run_table) as w:
            src = os.path.join(w.root, 'source', NAME)
            os.makedirs(src)
            shutil.copy(os.path.join(w.run, 'chain_summary.json'), src)
            if src_table:
                TS.write_table(os.path.join(src, RS.SERVED_TIERS), TS.good_rows())
            before = sorted(os.listdir(w.run))
            rc, out = w.dry(from_run=src)
            assert sorted(os.listdir(w.run)) == before and w.calls_made() == [], out
            if src_table:
                assert rc == 0 and 'WOULD REFUSE' not in out, out
            else:
                assert rc == 2 and f'WOULD REFUSE: P23 no {RS.SERVED_TIERS} in {src}' in out and 'nothing is copied' in out, out


def _card2_world(w):
    """The Card 2 files of the synthetic run (the stub back-fill) - checked to pass, then damaged copy by copy."""
    w.build()
    return {f: os.path.join(w.run, f) for f in TSN.CARD2_FILES}


_TICK = [0]


def _stamp(p):
    """A distinct modification time after each write: the content test holds card2_vleak.csv's parse by the file's stat, and
    two same-size writes in one tick of the filesystem clock would share one."""
    _TICK[0] += 1
    t = 1700000000 * 10 ** 9 + _TICK[0] * 10 ** 6
    os.utime(p, ns=(t, t))


def _rewrite(p, fn):
    with open(p, newline='') as f:
        rows = list(csv.reader(f))
    rows = fn(rows)
    with open(p, 'w', newline='') as f:
        csv.writer(f).writerows(rows)
    _stamp(p)


def _g8(w):
    """G8 (deep_omission.card2_removed) on the run: None when it accepts the file, else its refusal."""
    co = CFG.catalog_omission
    try:
        return None, DO.card2_removed(w.run, None, co.CARD2_RATIO, co.SNAPSHOT_CARD2)[2]
    except Exception as e:  # noqa: BLE001
        return str(e), None


def test_card2_content_test():
    """snapshot_card2 on the stub back-fill's files: all three pass, and G8 accepts card2_vleak.csv. Each damage fails its
    file; a damage of G8's row rule (deep_omission.card2_row_check, shared) is refused by G8 too - a flag that disagrees with
    the rounded ratio, a ratio that is not |peak_jyb| / i_here_jy, a flag without a ratio, a number that is not one, a ratio
    other than catalog_omission.CARD2_RATIO on every row (one consistent ratio is not enough). The test's own checks: a row the
    record does not give (or one dropped), card2_filter.csv not the served rows (an empty one counts 0 rows), a column
    dictionary that differs, a file missing beside it. The threshold's half unit: a rounded ratio at 0.05000 passes with either
    flag, counted; 0.04999 is judged."""
    with World() as w:
        p = _card2_world(w)
        for f in TSN.CARD2_FILES:
            r = validate.run('snapshot_card2', p[f])
            assert r and all(x['passed'] for x in r), (f, r)
        assert _g8(w)[0] is None, _g8(w)
        good = {f: open(p[f]).read() for f in TSN.CARD2_FILES}
        H = CARD2_COLS
        vl = TSN.CARD2_FILES[0]

        def restore():
            for f in TSN.CARD2_FILES:
                with open(p[f], 'w') as fh:
                    fh.write(good[f])
                _stamp(p[f])

        def damaged(fname, fn, want, g8=None):
            _rewrite(p[fname], fn)
            r = validate.run('snapshot_card2', p[fname])[0]
            g8r = _g8(w)[0]
            restore()
            assert not r['passed'] and want in r['detail'], (fname, want, r)
            if g8 is not None:
                assert g8r is not None and g8 in g8r, (want, g8r)
            return r, g8r

        def setcol(i, col, val):
            def fn(rows):
                for j in ([i] if isinstance(i, int) else i):
                    rows[j][H.index(col)] = val
                return rows
            return fn
        # rows 1-4: u1 (0.9/30 = 0.03, flagged), u2 (0.2/2 = 0.1), u3 and u8 (no ratio)
        damaged(vl, setcol(1, 'card2_would_remove', '0'), 'disagrees with its ratio', 'disagrees with its ratio')
        damaged(vl, setcol(2, 'card2_would_remove', '1'), 'disagrees with its ratio', 'disagrees with its ratio')
        damaged(vl, setcol(2, 'v_over_i_here', '0.03000'), 'but |peak|/i_here', 'but |peak|/i_here')
        damaged(vl, setcol(3, 'card2_would_remove', '1'), 'flagged without a ratio', 'flagged without a ratio')
        damaged(vl, setcol(1, 'i_here_jy', 'nan'), 'not finite numbers', 'not finite numbers')
        damaged(vl, setcol([1, 2, 3, 4], 'card2_ratio', '0.1'), 'not catalog_omission.CARD2_RATIO 0.05', 'ratio 0.1, not 0.05')
        r, g8r = damaged(vl, setcol(3, 'card2_ratio', '0.1'), 'not catalog_omission.CARD2_RATIO 0.05')
        assert g8r is None, g8r                  # G8 judges the ratio only where there is one; the test, on every row
        damaged(vl, setcol(1, 'served', '0'), "record's 4 Stokes V association-pool rows")
        damaged(vl, lambda rows: rows[:-1], "record's 4 Stokes V association-pool rows")
        damaged(vl, lambda rows: rows[:2] + [rows[2] + ['extra']] + rows[3:], 'without 23 fields')
        damaged(TSN.CARD2_FILES[2], lambda rows: rows[:-1], 'served rows of card2_vleak.csv')
        damaged(TSN.CARD2_FILES[1], lambda rows: rows[:-1], 'names')
        # the columns branch reads card2_vleak.csv's header only: a damaged body does not fail card2_vleak_columns.csv
        _rewrite(p[vl], lambda rows: rows[:2] + [rows[2] + ['extra']] + rows[3:])
        r = validate.run('snapshot_card2', p[TSN.CARD2_FILES[1]])[0]
        restore()
        assert r['passed'], r
        # an empty card2_filter.csv: 0 rows
        with open(p[TSN.CARD2_FILES[2]], 'w'):
            pass
        _stamp(p[TSN.CARD2_FILES[2]])
        r = validate.run('snapshot_card2', p[TSN.CARD2_FILES[2]])[0]
        restore()
        assert not r['passed'] and r['value'] == 0, r
        # the threshold's half unit (G8's rule): u1 at 0.9/18 = 0.05000 stands with either flag, counted; 0.04999 is judged
        for v, flag, ok in (('0.05000', '0', True), ('0.05000', '1', True), ('0.04999', '0', False)):
            _rewrite(p[vl], lambda rows: setcol(1, 'card2_would_remove', flag)(setcol(1, 'v_over_i_here', v)(
                setcol(1, 'i_here_jy', '18')(rows))))
            r = validate.run('snapshot_card2', p[vl])[0]
            g8r, n = _g8(w)
            restore()
            assert r['passed'] == ok and (g8r is None) == ok, (v, flag, r, g8r)
            if ok:
                assert '1 within half a unit' in r['detail'] and n['at_threshold_within_rounding'] == 1, (r, n)
        os.remove(p[TSN.CARD2_FILES[2]])
        r = validate.run('snapshot_card2', p[vl])[0]
        assert not r['passed'] and 'no card2_filter.csv beside it' in r['detail'], r


def test_card2_files_read_once_per_build():
    """Each Card 2 file is opened once by the build's content tests (validate.run per file): card2_vleak.csv's parse is shared
    by its own test and card2_filter.csv's, the columns test takes its header from it."""
    with World() as w:
        names = {os.path.join(w.run, f): f for f in TSN.CARD2_FILES}
        seen = collections.Counter()
        real = open

        def counting(path, *a, **kw):
            if isinstance(path, str) and os.path.abspath(path) in names:
                seen[names[os.path.abspath(path)]] += 1
            return real(path, *a, **kw)
        PT.open = counting
        try:
            u = w.build()
        finally:
            del PT.open
        assert u.status == 'done', (u.status, u.reason)
        assert seen == {f: 1 for f in TSN.CARD2_FILES}, seen


ALL = [test_registration_and_config, test_missing_table_refuses_before_anything, test_failing_table_refuses_before_anything,
       test_success_passes_the_table_as_production, test_views_rewrite_failure_restores_the_views,
       test_views_test_failure_restores_the_views, test_failed_restore_is_named, test_card2_failure_refuses,
       test_pages_pass_failure_is_not_hooked, test_dry_run, test_card2_content_test, test_page_workers_default_is_productions,
       test_waived_table_failure_builds, test_inputs_hold_the_record_and_card2_roots, test_linked_backup_dir_refuses,
       test_card2_files_read_once_per_build]


if __name__ == '__main__':
    if _IMPORT_ERROR is not None:
        print(f'IMPORT FAILED: {_IMPORT_ERROR!r}')
        sys.exit(2)
    want = set(sys.argv[1:])
    failed = 0
    for t in ALL:
        if want and t.__name__ not in want:
            continue
        try:
            t()
            print(f'PASS {t.__name__}')
        except Exception:  # noqa: BLE001
            import traceback
            failed += 1
            print(f'FAIL {t.__name__}')
            traceback.print_exc()
    print(f'{failed} failed')
    sys.exit(1 if failed else 0)
