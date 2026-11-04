"""serve --tier snapshot --family comb: THE COMBINED SNAPSHOT VIEW - the snapshot tier's CUMULATIVE step, the counterpart of the
10-minute cumulative (lib/serve/sublst_build.py). Spec: docs/specs/snapshot_combined_view.md.

Port of exopipe_ops snapshot_combined_view.py (1af2dab 2026-10-21 .. 5b8b8ff 2026-10-31; dde1760, dbc40a6 2026-11-04) and of its
dated drivers snapshot_batch/comb_build_20261102.sh / comb_build_20261104.sh (steps 1-4, the same in both). Production serves the 10-s and 1-minute snapshot pages (p8986-p8989) from ONE
combined view of every served batch run (36 since 2026-10-26). Why a view, ported as the cumulative step (refactor chat
2026-11-03): the real-time pipeline appends hours, so a full all-cell re-chain per night does not fit. The rulings: the view
(R-REFACTOR-20261020-BUILD-COMBINED-SNAPSHOT-VIEW, citing Gregg; Gregg 2026-10-21 "Start the combined build",
IMPLEMENTED-COMBINED-SNAPSHOT-VIEW-BUILDER-FIRST-BUILD-20261021), one page per tier and Stokes (Gregg 2026-10-26: "No! I just
mean combine the batches for each tier/polarization into one page.", R-GREGG-20261026-5-LST-REBUILD-AFTER-EVERYTHING-DONE-
COMBINED-VIEW-ONE-PAGE-PER-TIER-POLARIZATION), one tier per page directory (Gregg 2026-10-03, SNAP-PAGES-ONE-TIER-PER-PAGE;
DEFECT-SNAPCOMB-PAGE-DIRS-HOLD-BOTH-TIERS-20261031), the deep omission (R-GREGG-20261024-DEEP-OMISSION-DESIGN-APPROVED-AND-
10MIN-HOLD-RELEASED). VIEW LIMITS, written into every queue header as production writes them: (1) routing is not re-run on
merged identities - a source served by no batch alone but servable from two batches' detections combined is absent; (2) the
merge is two-level (batch identities, then across batches) and can differ from one all-cell chain.

    exopipe3 serve --tier snapshot --family comb --run Run_SNAPCOMB_20261103 --view-runs @RUNS --dry-run
    exopipe3 serve --tier snapshot --family comb --run Run_SNAPCOMB_PARITY --view-runs R1,R2 --out SCRATCH \
        --set catalog_omission.REFUSALS=DIR --stages omit,build,filtercat,localize,ready

Stages (--stages / --from / --until; default all, in this order); each runs only after the previous one ended done:
  omit       (unit snapshot_<view>_omit) the deep-omission stage on the view's batch runs (lib/catalog/deep_omission.run_snapshot,
             the products of `catalog --tier snapshot --stages omit`) into the view's omission directory serve.SNAPVIEW_OMIT_DIR
             (production location <SNAPSHOT_REVIEW_ROOT>/deep_omission/<view>; parity <out>/snapshot/deep_omission/<view>):
             decisions, record, the served counts they were made with (deep_omission_served.csv) and the KEPT PRODUCT
             deep_omission_kept.csv (classes omission_decisions, omission_record, omission_served, omission_kept). With
             --view-omit DIR the stage is not run: DIR is an omit product made for exactly these runs.
  build      (unit snapshot_<view>_build) the view <view>.part, renamed to <view> only after every test passed, built by this
             module's CHILD entry point under the production interpreter (serve.SNAPVIEW_PYTHON - production's driver runs
             the builder under it; Python 3.6 + clv15 + deep_omission.py, nothing else of the branch):
               1  clusters = the served batch identities (each source_catalogs/*_sources_survivors.csv source_id; its detections
                  the rows of its review_dir detections.csv, one per detection key dkey), anchored at the identity's served
                  common-frame position (ra_common, dec_common, pos_err_common_arcsec);
               0  THE KEPT PRODUCT APPLIED (production's step 0 recomputed the decisions; here the deep-omission stage's product
                  is read): an identity keeps exactly its 'detection kept' rows, one with none is dropped, every other served
                  detection must sit on an OMIT key of the decisions beside it; per tier a batch source is dropped / re-pointed,
                  a member row dropped and a panel left unlinked exactly where the kept product says - each decision is also
                  re-derived with production's rule from the same OMIT keys, and any disagreement REFUSES (G-K);
               2  the chain's merge_identities_or_refuse (clv15 phase3_snapshot_chain.py, in place) and its content test (T1);
               3  names: the best-|SNR| member identity's J-name, a clash '<J>~<batch>';
               4  the general catalogue source_catalogs/ (review_dir = candidates/_identity/<name>), the per-tier views
                  source_catalogs_<S>_<T>/ (one survivors row per combined source, every member row), one page directory per
                  tier candidates/<S>/<name>_<band>[_1min] holding HARD LINKS of the member batch sources' panels and a merged
                  detections.csv, the queues queue_<S>_<T>.txt (production's header: view limits, deep-omission line, the first
                  batch's selection lines), the record deep_omission.csv / deep_omission_dropped.csv / deep_omission_summary.txt
                  in production's formats, detection_ts.csv, combined_view_summary.json.
             Production's content tests T1-T6 run in the child before the rename (any failure: nothing is moved into place).
             Then class snapshot_view on the view (lib/serve/tests_snapshot_view.py: the same tests read back from disk, the
             identities against the kept product, the record against the omit stage's, every panel's own key against the OMIT
             keys and every page's files against its member batch sources' less the panel drops - F1).
  filtercat  (unit snapshot_<view>_filtercat) the driver's step 2: exopipe_ops snapshot_filter_catalog.py VIEW <the member batches'
             fit stores serve.SNAPVIEW_FIT_STORE> in place (one row per detection: size, axis ratio, A-team distance, n_v_epoch;
             coadd_root.txt), the union of the member batches' card2_filter.csv (first member's comment lines + a view line,
             one header, each body line once in batch order) and FILTER-INPUTS (class filter_catalog: every view detection has
             its row, none twice). Classes filter_catalog (per_detection), snapshot_card2_union. The catalogue's detections.csv
             pass reads with SNAPCAT_PROCS = serve.SNAPVIEW_FILTERCAT_PROCS processes (exopipe_ops 1b9475c; 6, the Lustre rule;
             production's default 16; always passed, never inherited; F-G4 a count not >= 1).
  localize   (unit snapshot_<view>_localize) the driver's step 3: snapshot_localize_tiers.sh VIEW in place with
             serve.SNAPVIEW_LOC_ENV and PYTHONHASHSEED = serve.SNAPSHOT_FIGURE_HASHSEED (F10: the figures reproducible;
             lib/serve/snapshot_copy.localize_stage, the batch serve's own stage; G21 LOCALIZE-TIERS-OK).
  ready      (unit snapshot_<view>_ready) the driver's step 4 'VIEW-READY-FOR-HIDES': the serve-time exclusion gate on the view
             (G17, the gate snapshot_serve.sh runs on it before registering), the snapshot_view tests again on the final files
             (filter catalogue and localization in place), and the note VIEW-READY-FOR-HIDES <view>.
Gates (refuse, exit 2): G-V1 view name (--run, the snapshot prefix) and runs (--view-runs: none, a run twice, a run without
source_catalogs); G-V2 the view or its .part exists (never overwritten); G-V3 the omit product: its files, its content tests now,
its runs = the view's runs in order; G-V4 the served set changed since the omit stage (the served digest of the runs as read now
!= the omit summary's, Q fix verification N2); G-V5 a batch run on another file system than a production view, or a panel
that cannot be hard-linked there (a scratch or parity view copies such a panel, recorded: the parity guard refuses a hard link of
a file outside the parity roots); G-V6 a run or the omit product under an injected run's start record (an ordinary run); G-K the
kept product does not apply exactly (a kept row of no served detection, a served detection neither kept nor on an OMIT key, a
per-tier row the rule does not
drop or a drop it does not list, a row left unapplied); T1-T6 (production's); F-G1 a member batch without its fit store, F-G2 a
card2_filter.csv missing or of another header, F-G3 the filter catalogue script failing, F-G4 a reader count not >= 1; G21
localization; G17 at ready.
V_combview review fixes (2026-11-04): G-V2 runs before ANY stage when omit or build is asked (F2: an existing view's omit product
is never rewritten); G-V7 a batch run whose per-tier survivors are not exactly the identities its served_tiers.csv serves, or a
kept per-tier source whose page holds no detection (F4: an incomplete input refuses, never booked as a deep omission); G-V8 the
omit product's decisions re-made now from the refusal record and fleet records the build reads, with its served counts, must
equal its table byte for byte (F5: never a stale product); a refused build moves its .part, a view failing its read-back tests
moves the view, aside to <view>.refused_<UTC> (F7).
V2 follow-up (2026-11-04; exopipe_ops dde1760, dbc40a6 at the pin fa90670): the omission per IMAGE (a wideband detection omitted
only when its own frame / block holds the omitted subband; fail closed without a member table), CARD 2 per frame in the 10-s tier
(re-derived here from each batch's card2_vleak.csv; record kinds 'card2 detection', 'card2 member row', 'card2 panel'), the
RE-JUDGING of every batch source on the detections it keeps (the omit stage's, lib/catalog/rejudge.py - the chain's route
functions cannot run in this build; applied as 'source re-judged' / 'member row re-judged', rejudged.csv), the RE-POINT of a
partial source by its own (ts, subband) detection (the review's defect RP fixed), T7 (no Card 2 detection left, every re-judged
(batch source, tier) met, every combined row ranked by a detection it keeps), the queue's CARD 2 AND RE-JUDGING line and the
summary's card2 / rejudge keys - production's view byte for byte (tests/test_snapshot_view.py test_parity). G-V8 also re-checks
the cells (the rate population) and the re-judging / Card 2 inputs (size, mtime) against the omit product; G-V9 a partial source
whose own detection is absent from its page; G-V10 an image absent from a member table.
Production details carried for parity, not ruled (spec 'Pending Gregg'): the name-clash rule, the queue order, the survivors
row of a combined source (its member row of largest |best S/N|), the duplicate batch_members column of production's survivors
files.
NOT PORTED (spec 'Not ported'): production's REPAIR MODE (SNAPCOMB_REPAIR_OF), the test-build switch SNAPCOMB_DEEP_OMISSION=0,
the registration of the view's pages on p8986-p8989 with its carry, page context and hub (snapshot_serve.sh SNAP_REGISTER=1;
serve.PAGES holds no combined-view page), the two snapshot hides the refactor chat writes before it (no FP-filter step yet), an
incremental append of one new batch to an existing view (production rebuilds the view whole).
"""
import collections
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time

STAGES = ('omit', 'build', 'filtercat', 'localize', 'ready')
FAMILY = 'comb'

# ---- production's texts (snapshot_combined_view.py; drift-tested in tests/test_snapshot_view.py) -------------------------------
QUEUE_NAME = '# queue_name: snapshot Stokes %s, %s tier - COMBINED VIEW of %d served batches\n'
QUEUE_RUN = '# run: %s (combined view; batches %s)\n'
VIEW_LIMITS = ("# VIEW LIMITS: (1) routing is not re-run on merged identities - a source served by no batch alone but servable "
               "from two batches' detections combined is absent; (2) the merge is two-level (batch identities, then across "
               "batches) and can differ from one all-cell chain\n")
DEEP_OMISSION_LINE = ("# DEEP OMISSION: %d of %d served detections omitted by the deep-omission rule (Gregg 2026-10-24), %d (LST, band, "
                      "night, Stokes, tier) omitted%s; record deep_omission_dropped.csv, decisions deep_omission.csv\n")
DEEP_OMISSION_WB = ' + %d wideband ones in the images that hold the omitted subband only'      # dde1760
CARD2_REJUDGE_LINE = ("# CARD 2 AND RE-JUDGING (Gregg 2026-11-04): %d Stokes V 10-s detections removed by Card 2 (Stokes I Master flux "
                      "under them > 20 x |V|); %d batch (source, tier) re-judged out on the detections they keep; record "
                      "deep_omission_dropped.csv, rejudged.csv\n")                            # dbc40a6
SUMMARY_LINE1 = '%d detections omitted by the deep-omission rule (Gregg 2026-10-24)'
SUMMARY_LINE1_NEW = ('; %d Stokes V 10-s detections removed by Card 2 and %d batch sources re-judged out on the detections they keep '
                     '(Gregg 2026-11-04)')                                                     # dbc40a6
SUMMARY_LINE2 = 'served detections %d, kept %d; decisions %d (%d OMIT keys%s); %s\n'
SUMMARY_WB = ' + %d wideband per image'                                                       # dde1760
TESTS_PASSED = 'T1 T2 T3 T4 T5 T6 T7 PASS'
PANEL_TOKEN = re.compile(r"^rank\d+_\d\dh_\d{4}-\d\d-\d\d_([A-Za-z0-9]+)\.(?:png|page\.json)$")
MEMBER_ROW = re.compile(r"^(.*)_(\d{2}h_\d{4}-\d{2}-\d{2})_([^_]+)$")

csv.field_size_limit(10 ** 9)


class ViewRefusal(Exception):
    """A gate of the build refused (the child exits 2 with REFUSING and the reason; the step records the refusal)."""


# =====================================================================================================================
# helpers shared by the child and the tests (Python 3.6; stdlib only)
# =====================================================================================================================

def rows(p):
    """The rows of a CSV file, [] when it is absent (production's rows(): one metadata call less per file)."""
    try:
        with open(p, newline='') as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []


def write_csv(p, hdr, rr):
    """production's write(): DictWriter, extrasaction 'ignore' (a duplicate name in hdr writes its value twice, as production)."""
    with open(p, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=hdr, extrasaction='ignore')
        w.writeheader()
        w.writerows(rr)


def hdr_union(rr):
    """The union of the rows' keys in first-appearance order, + 'batch_run' when absent (production's hdr_union)."""
    h = []
    for x in rr:
        for k in x:
            if k not in h:
                h.append(k)
    return h + ([] if 'batch_run' in h else ['batch_run'])


def page_dir(S, T, sid, suffix):
    """A combined source's page directory, relative to the view: candidates/<S>/<sid><suffix[T]> with '+' -> 'p', '~' -> '_'."""
    return os.path.join('candidates', S, (sid + suffix[T]).replace('+', 'p').replace('~', '_'))


def identity_dir(name):
    """A combined identity's directory (the general catalogue's review_dir), relative to the view."""
    return os.path.join('candidates', '_identity', name.replace('+', 'p').replace('~', '_'))


def panel_tier(fn):
    """The tier of a linked panel by its file token (the page code's rule, exopipe_ops 2b56954): '...1min' = a 1-min block, 14
    digits at the end = a 10-s frame; None = no tier token (astrometry figures, other files)."""
    m = PANEL_TOKEN.match(fn)
    return None if not m else '1min' if m.group(1).endswith('1min') else '10s' if re.search(r'\d{14}$', m.group(1)) else None


def dir_tier(pd, suffix):
    return '1min' if pd.endswith(suffix['1min']) else '10s'


def load_standalone(path, name='exopipe3_deep_omission_standalone'):
    """deep_omission.py loaded from its file without the exopipe package (the child: its stdlib-only helpers - dkey, okey,
    batch_tag, the key patterns, the multiset digest)."""
    import importlib.util
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _h(k):
    """A detection key's 16-byte hash (the kept product's index: millions of keys)."""
    return hashlib.blake2b('\x1f'.join(k).encode('utf-8'), digest_size=16).digest()


class KeptProduct:
    """The deep-omission stage's products for the view's runs (deep_omission.run_snapshot), read for application:
    OM / n_dec from deep_omission.csv, the kept product deep_omission_kept.csv indexed ((run, identity) -> kept detection keys;
    per-tier rows counted per row key), the summary json. ``take`` consumes one per-tier row; ``unapplied`` lists the rest."""

    def __init__(self, d, DO):
        self.dir = d
        self.decisions = os.path.join(d, 'deep_omission.csv')
        self.path = os.path.join(d, 'deep_omission_kept.csv')
        with open(os.path.join(d, 'deep_omission_summary.json')) as f:
            self.summary = json.load(f)
        dec = rows(self.decisions)
        self.n_dec = len(dec)
        self.OM = DO.omitted_keys(dec)
        self.kept = collections.defaultdict(set)
        self.n_kept = 0
        self.per = {k: collections.Counter() for k in ('source dropped', 'source re-judged', 'member row dropped',
                                                       'member row re-judged', 'card2 member row dropped', 'panel dropped',
                                                       'card2 panel dropped')}
        self.repoint = {}
        with open(self.path, newline='') as f:
            rd = csv.DictReader(f)
            if list(rd.fieldnames or []) != list(DO.KEPT_COLS):
                raise ViewRefusal('G-V3 %s: columns %s are not the kept product\'s' % (self.path, rd.fieldnames))
            for i, r in enumerate(rd):
                kd = r['kind']
                if kd == 'detection kept':
                    s = self.kept[(r['run'], r['identity'])]
                    h = _h((r['cand_id'], r['ra_peak'], r['dec_peak']))
                    if h in s:
                        raise ViewRefusal('G-K kept product row %d: %s %s kept twice' % (i + 2, r['identity'], r['cand_id']))
                    s.add(h)
                    self.n_kept += 1
                elif kd in ('source dropped', 'source re-judged'):
                    self.per[kd][(r['run'], r['catalog'], r['source_id'], r['note'])] += 1
                elif kd in ('member row dropped', 'member row re-judged', 'card2 member row dropped'):
                    self.per[kd][(r['run'], r['catalog'], r['source_id'], r['cand_id'])] += 1
                elif kd in ('panel dropped', 'card2 panel dropped'):
                    self.per[kd][(r['run'], r['catalog'], r['source_id'], r['stokes'], r['tier'], r['note'])] += 1
                elif kd in ('source re-pointed', 'source routes'):
                    k = (r['run'], r['catalog'], r['source_id'])
                    if k in self.repoint:
                        raise ViewRefusal('G-K kept product row %d: source %s re-pointed twice' % (i + 2, r['source_id']))
                    self.repoint[k] = (kd, json.loads(r['fields'], object_pairs_hook=collections.OrderedDict))
                else:
                    raise ViewRefusal('G-K kept product row %d: unknown kind %r' % (i + 2, kd))

    def take(self, kind, key):
        """True when the kept product drops this row: one of its rows of ``kind`` with ``key`` is consumed."""
        c = self.per[kind]
        if c.get(key, 0) > 0:
            c[key] -= 1
            return True
        return False

    def unapplied(self):
        """{kind: n} of the kept product's rows no served row consumed."""
        out = {k: sum(v for v in c.values() if v > 0) for k, c in self.per.items()}
        out['source re-pointed or routes'] = len(self.repoint)
        out['detection kept'] = sum(len(s) for s in self.kept.values())
        return {k: v for k, v in out.items() if v}


# =====================================================================================================================
# THE BUILD (child entry point, Python 3.6): production's steps 1-4 and T1-T6, step 0 = the kept product applied
# =====================================================================================================================

def build(job):
    """Build the view job['out'] from the served batch runs job['runs'] (in order) and the omit product job['kept_dir']; see the
    module docstring. Writes OUT.part, renames it to OUT when every test passed. Returns the stats dict. ViewRefusal on a gate."""
    T0 = time.time()
    OUT = job['out'].rstrip('/')
    RUNS = [os.path.abspath(r.rstrip('/')) for r in job['runs']]
    PART = OUT + '.part'
    RUNNAME = os.path.basename(OUT)
    TIERS, STOKES, TSFX = tuple(job['tiers']), tuple(job['stokes']), dict(job['suffix'])
    # production's search path at the chain's import: its script directory (exopipe_ops) with clv15 inserted before it
    # (snapshot_combined_view.py:42) - so every module the chain imports resolves as it does in production
    for p in (job['ops'], job['clv15']):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    import phase3_snapshot_chain as CH           # noqa: E402 - the chain's merge, in place (production imports it at its top)
    csv.field_size_limit(10 ** 9)                # after the chain's import, as production (the chain sets 10**8)
    DO = load_standalone(job['deep_omission_py'])
    dkey, okey = DO.dkey, DO.okey
    CIDK, PANEL = DO.SNAP_CID, DO.SNAP_PANEL
    f = lambda v: float(v) if v not in (None, '') else None     # noqa: E731

    def log(m):
        print('%6.0f s  %s' % (time.time() - T0, m), flush=True)
    if not RUNS:
        raise ViewRefusal('G-V1 no batch run given')
    if len(set(RUNS)) != len(RUNS):
        raise ViewRefusal('G-V1 a batch run is given twice')
    if os.path.exists(OUT) or os.path.exists(PART):
        raise ViewRefusal('G-V2 %s or %s exists - never overwritten' % (OUT, PART))
    TAG = {r: DO.batch_tag(r) for r in RUNS}
    for r in RUNS:
        if not os.path.isdir(os.path.join(r, 'source_catalogs')):
            raise ViewRefusal('G-V1 %s has no source_catalogs' % r)
    K = KeptProduct(job['kept_dir'], DO)
    if [os.path.abspath(x) for x in K.summary.get('runs') or []] != RUNS:
        raise ViewRefusal('G-V3 the omit product %s was made for the runs %s, not the view\'s %s (same runs, same order)'
                          % (K.dir, K.summary.get('runs'), RUNS))
    os.environ['EXOPIPE3_CLV15'] = job['clv15']         # the standalone deep_omission's minute key (clv15 in place)
    try:
        DEC = rows(K.decisions)
        OMIS = DO.Omission(DEC, DO.SnapshotRecords(job['proto'], DO.Reader(workers=job.get('read_workers', 6)),
                                                   names=job['names']))
        # CARD 2 (snapshot_combined_view.py dbc40a6 section 0b), re-derived here from each batch's card2_vleak.csv
        C2 = DO.Card2(RUNS, None, ratio=job['card2_ratio'], name=job['card2_file'])
        # RE-JUDGING: the omit stage's (the chain's route functions cannot run in this build - V_combview review T5)
        REJ = DO.Rejudged.from_csv(os.path.join(K.dir, DO.REJUDGED_FILE), RUNS)
    except DO.OmissionRefusal as e:
        raise ViewRefusal(str(e))
    OM, OMWB = OMIS.OM, OMIS.OMWB
    d_omitted, d_card2 = OMIS.det, C2.det
    c2want = K.summary.get('card2_counts') or {}
    for k_, v_ in (('removed_10s_v_records', C2.n['removed_10s']), ('removed_10s_v_served', C2.n['removed_10s_served']),
                   ('not_evaluated_10s_v', C2.n['not_evaluated_10s'])):
        if c2want.get(k_) != v_:
            raise ViewRefusal('G-V4 Card 2: the batches\' card2_vleak.csv give %s %s, the omit product %s - a file changed since '
                              'the omit stage' % (k_, v_, c2want.get(k_)))
    REC = []

    # ---- 1. clusters = served batch identities (production's step 1)
    clusters, anchors, cinfo = [], [], []
    pending = []
    for r in RUNS:
        ids = collections.OrderedDict()
        for p in sorted(glob.glob(os.path.join(r, 'source_catalogs', '*_sources_survivors.csv'))):
            for row in rows(p):
                ids.setdefault(row['source_id'], []).append((os.path.basename(p), row))
        n0 = len(pending)
        for J, grs in ids.items():
            dets = collections.OrderedDict()
            for _, row in grs:
                for d in rows(os.path.join(r, row['review_dir'], 'detections.csv')):
                    dets.setdefault(dkey(d), d)
            if not dets:
                raise ViewRefusal('G-V1 %s identity %s has no detections' % (TAG[r], J))
            mem = []
            for d in dets.values():
                m = dict(ts=d['ts'], sb=d['subband'], st=d['stokes'], tier=d['tier'], cell='%s_%s' % (d['lst'], d['date']),
                         ra=float(d['ra']), dec=float(d['dec']), snr=float(d['snr']), _row=d)
                if f(d.get('ra_peak')) is not None:
                    m['ra_peak'], m['dec_peak'] = float(d['ra_peak']), float(d['dec_peak'])
                mem.append(m)
            g = grs[0][1]
            rc, dc, ec = f(g.get('ra_common')), f(g.get('dec_common')), f(g.get('pos_err_common_arcsec'))
            anc = None
            if rc is not None and dc is not None and ec:
                anc = dict(ra=float(g['ra']), dec=float(g['dec']), err_arcmin=ec / 60.0,
                           common=dict(ra=rc, dec=dc, err_arcmin=ec / 60.0, key=None, frame_arcmin=0.0))
            pending.append((r, J, grs, mem, anc))
        log('%s: %d identities, %d detections' % (TAG[r], len(pending) - n0, sum(len(p[3]) for p in pending[n0:])))
    log('card 2: %d 10-s Stokes V detections of the %d batches\' records rejected (%d of them served); %d 10-s rows not evaluated '
        '(no Stokes I under them)' % (C2.n['removed_10s'], len(RUNS), C2.n['removed_10s_served'], C2.n['not_evaluated_10s']))
    n_all = sum(len(p[3]) for p in pending)
    # G-V4: the served set as read now is the one the omit stage decided on (Q fix verification N2)
    D = DO.MultisetDigest()
    for r, J, _g, mem, _a in pending:
        for m in mem:
            D.add(DO.detection_item(TAG[r], J, dkey(m['_row'])))
    want = (K.summary.get('digests') or {}).get('served')
    if not want or D.text() != want:
        raise ViewRefusal('G-V4 the served batch runs changed since the omit stage (served digest now %s, the omit summary\'s %s)'
                          ' - run the omit stage again' % (D.text()[:28], (want or 'none')[:28]))
    # G-V4, the counts (V_combview review F5): the served counts the decisions were made with (deep_omission_served.csv, re-decided
    # by the step side's G-V8) are the served set's less its Card 2 detections (dbc40a6: not counted as served), read now
    cnt = collections.Counter(okey(m['_row']) for p in pending for m in p[3] if not d_card2(m['_row']))
    now = {(k[0], k[2], k[1], k[3], k[4]): n for k, n in cnt.items()}
    rec_counts = {(r['lst'], r['night'], r['band'], r['stokes'], r['tier']): int(r['n'])
                  for r in rows(os.path.join(K.dir, 'deep_omission_served.csv'))}
    if now != rec_counts:
        bad = sorted(k for k in set(now) | set(rec_counts) if now.get(k) != rec_counts.get(k))
        raise ViewRefusal('G-V4 the omit product\'s served counts (deep_omission_served.csv) are not the served set\'s: %d keys '
                          'differ (e.g. %s)' % (len(bad), bad[:2]))
    complete_runs(RUNS, TAG, TIERS)       # G-V7 (V_combview review F4)

    # ---- 0. THE KEPT PRODUCT APPLIED (production's step 0 and section 0b decided here; the omit stage's product is read)
    DROPPED = set()
    for r, J, grs, mem, anc in pending:
        want_k = K.kept.pop((r, J), set())
        keep = []
        for m in mem:
            d = m['_row']
            on_om, on_c2 = d_omitted(d), d_card2(d)
            if _h(dkey(d)) in want_k:
                if on_om or on_c2:
                    raise ViewRefusal('G-K %s %s: the kept product keeps %s, which the rule removes (%s)'
                                      % (TAG[r], J, d['cand_id'], 'an omitted key' if on_om else 'Card 2'))
                keep.append(m)
            elif on_om:
                REC.append(dict(kind='detection', batch=TAG[r], identity=J, cand_id=d['cand_id'], lst=d['lst'], band=okey(d)[1],
                                night=d['date'], stokes=d['stokes'], tier=d['tier'], snr=d['snr']))
            elif on_c2:
                REC.append(dict(kind='card2 detection', batch=TAG[r], identity=J, cand_id=d['cand_id'], lst=d['lst'],
                                band=okey(d)[1], night=d['date'], stokes=d['stokes'], tier=d['tier'], snr=d['snr']))
                C2.JC_C2[(r, J, d['cand_id'])] += 1
            else:
                raise ViewRefusal('G-K %s %s: %s is neither kept by the kept product nor removed by the rule (an omitted key, '
                                  'Card 2)' % (TAG[r], J, d['cand_id']))
            C2.JC_ALL[(r, J, d['cand_id'])] += 1
        if len(keep) != len(want_k):
            raise ViewRefusal('G-K %s %s: the kept product keeps %d detections the served identity does not hold'
                              % (TAG[r], J, len(want_k) - len(keep)))
        if not keep:
            DROPPED.add((r, J))
            REC.append(dict(kind='identity dropped', batch=TAG[r], identity=J))
            continue
        clusters.append(keep)
        anchors.append(anc)
        cinfo.append((r, J, grs))
    if K.kept:
        raise ViewRefusal('G-K the kept product keeps detections of %d identities no served run holds (e.g. %s)'
                          % (len(K.kept), sorted(K.kept)[:2]))
    del pending
    n_om = sum(1 for x in REC if x['kind'] == 'detection')
    n_c2 = sum(1 for x in REC if x['kind'] == 'card2 detection')
    n_mixed_c2 = C2.n_mixed()
    log('deep omission: the kept product of %s applied - %d decisions (%d OMIT, %d of them wideband per image), %d of %d served '
        'detections omitted, %d batch identities dropped' % (K.dir, K.n_dec, len(OM) + len(OMWB), len(OMWB), n_om, n_all,
                                                              len(DROPPED)))
    log('re-judging (the omit stage\'s, %s): %d (batch source, tier) re-judged out, %d still qualify on the members they keep'
        % (DO.REJUDGED_FILE, len(REJ.OUT), len(REJ.ROUTES)))
    n_in = sum(len(c) for c in clusters)
    if n_in + n_om + n_c2 != n_all or n_in != K.n_kept:
        raise ViewRefusal('G-K omission record %d + Card 2 record %d + kept %d != served %d (kept product %d)'
                          % (n_om, n_c2, n_in, n_all, K.n_kept))
    keys_in = collections.Counter(dkey(m['_row']) for c in clusters for m in c)
    dup_in = [k for k, n in keys_in.items() if n > 1]
    if dup_in:
        raise ViewRefusal('G-V1 %d detections in two batch identities (e.g. %s) - batches are not cell-disjoint'
                          % (len(dup_in), dup_in[:2]))
    log('clusters: %d batch identities, %d detections, %d without a common-frame anchor'
        % (len(clusters), n_in, sum(a is None for a in anchors)))

    # ---- 2. the chain's merge + its content test (T1; merge_identities_or_refuse exits REFUSING on a failure)
    cid = {id(m): i for i, c in enumerate(clusters) for m in c}
    merged, _merged_anchors, mstats = CH.merge_identities_or_refuse(clusters, anchors)
    log('merge: %s' % json.dumps({k: v for k, v in mstats.items() if k != 'rule'}))
    keys_out = collections.Counter(dkey(m['_row']) for c in merged for m in c)
    if keys_out != keys_in:
        raise ViewRefusal('T2 detections lost %d, added %d, duplicated %d' % (len(set(keys_in) - set(keys_out)),
                          len(set(keys_out) - set(keys_in)), sum(1 for n in keys_out.values() if n > 1)))
    log('T2 PASS: %d detections, each in exactly one of %d combined identities' % (len(keys_out), len(merged)))

    # ---- 3. names; batch identity -> combined identity
    members_of = []
    for c in merged:
        cs = sorted({cid[id(m)] for m in c}, key=lambda i: (-max(abs(m['snr']) for m in clusters[i]), i))
        members_of.append(cs)
    name, used, b2c = [], set(), {}
    for k, cs in enumerate(members_of):
        r0, J0, _ = cinfo[cs[0]]
        nm = J0 if J0 not in used else '%s~%s' % (J0, TAG[r0])
        used.add(nm)
        name.append(nm)
        for i in cs:
            b2c[(cinfo[i][0], cinfo[i][1])] = k
    n_multi = sum(len(cs) > 1 for cs in members_of)
    log('names: %d combined identities, %d from two or more batches, %d renamed for a name clash'
        % (len(name), n_multi, sum('~' in n for n in name)))

    # ---- 4. the view under OUT.part
    os.makedirs(os.path.join(PART, 'source_catalogs'))
    job['_made_part'] = True             # F7: this build made the .part (child_main moves only its own aside)
    os.makedirs(os.path.join(PART, 'judgements'))
    # 4a the general catalogue (the localizer's input)
    gen = collections.defaultdict(list)
    for k, cs in enumerate(members_of):
        r0, J0, grs = cinfo[cs[0]]
        d = identity_dir(name[k])
        os.makedirs(os.path.join(PART, d))
        dets = [dict(m['_row'], batch_run=TAG[cinfo[i][0]]) for i in cs for m in clusters[i]]
        write_csv(os.path.join(PART, d, 'detections.csv'), hdr_union(dets), dets)
        for fn, row in grs:
            gen[fn].append(dict(row, source_id=name[k], review_dir=d, n_det=len(dets),
                                best_snr='%.2f' % max(abs(float(x['snr'])) for x in dets), batch_run=TAG[r0],
                                batch_members='+'.join('%s:%s' % (TAG[cinfo[i][0]], cinfo[i][1]) for i in cs)))
    for fn, rr in gen.items():
        write_csv(os.path.join(PART, 'source_catalogs', fn), hdr_union(rr) + ['batch_members'], rr)
    # 4b the per-tier views (production's; every omission decision taken from the kept product, checked against the rule)
    n_links = n_link_expected = n_src_rows = n_mem_rows = 0
    PDETS = {}
    MEMBERS_PD = collections.defaultdict(set)
    PANELS_PD = collections.defaultdict(list)
    qrows = collections.defaultdict(dict)
    src_seen = collections.Counter()
    REJ_SEEN = collections.Counter()         # (run, J, tier) re-judged out -> its per-tier source rows met (T7)
    T7_BEST = []                              # combined per-tier rows whose best detection is not among the page's detections
    xstat = collections.Counter()           # the per-tier page detections that are no kept detection of their identity
    copy_runs = set()                       # batch runs whose panels a scratch / parity view copies (a hard link failed)

    def agree(kind, prod, kept, what):
        if prod != kept:
            raise ViewRefusal('G-K %s: production\'s rule %s it, the kept product %s it (%s)'
                              % (kind, 'drops' if prod else 'keeps', 'drops' if kept else 'keeps', what))
    for S in STOKES:
        for T in TIERS:
            cdir = 'source_catalogs_%s_%s' % (S, T)
            os.makedirs(os.path.join(PART, cdir))
            surv = collections.defaultdict(list)
            memr = collections.defaultdict(list)
            for r in RUNS:
                # PRODUCTION ORDER: glob order (the directory's listing, unsorted) - the record rows and ties of equal |S/N|
                # follow it, kept for byte parity
                for p in glob.glob(os.path.join(r, cdir, '*_sources_survivors.csv')):
                    band = os.path.basename(p)[: -len('_sources_survivors.csv')]
                    catalog = '%s/%s_sources_survivors.csv' % (cdir, band)
                    for row in rows(p):
                        k = b2c.get((r, row['identity_id']))
                        what = '%s %s %s' % (TAG[r], cdir, row['source_id'])
                        if k is None and (r, row['identity_id']) in DROPPED:
                            agree('source dropped', True, K.take('source dropped', (r, catalog, row['source_id'],
                                                                                    DO.IDENTITY_EMPTY_NOTE)), what)
                            REC.append(dict(kind='source dropped', batch=TAG[r], identity=row['identity_id'],
                                            source=row['source_id'], stokes=S, tier=T, band=band,
                                            note=DO.IDENTITY_EMPTY_NOTE))
                            continue
                        rj = (r, row['identity_id'], T) in REJ.OUT
                        note = REJ.NOTE.get((r, row['identity_id'], T), '')
                        agree('source re-judged', rj, K.take('source re-judged', (r, catalog, row['source_id'], note)), what)
                        if rj:
                            REC.append(dict(kind='source re-judged', batch=TAG[r], identity=row['identity_id'],
                                            source=row['source_id'], stokes=S, tier=T, band=band, note=note))
                            REJ_SEEN[(r, row['identity_id'], T)] += 1
                            continue
                        if k is None:
                            raise ViewRefusal('T3 %s %s source %s of identity %s is in no combined identity'
                                              % (TAG[r], cdir, row['source_id'], row['identity_id']))
                        surv[(band, k)].append((r, row))
                        n_src_rows += 1
                        src_seen[(r, cdir, row['source_id'])] += 1
                for p in glob.glob(os.path.join(r, cdir, '*_source_members.csv')):
                    band = os.path.basename(p)[: -len('_source_members.csv')]
                    catalog = '%s/%s_source_members.csv' % (cdir, band)
                    for row in rows(p):
                        mm = MEMBER_ROW.match(row['source_id'])
                        k = b2c.get((r, mm.group(1))) if mm else None
                        ck = CIDK.match(row.get('cand_id', ''))
                        mkey = (r, catalog, row['source_id'], row.get('cand_id', ''))
                        what = '%s %s %s %s' % (TAG[r], cdir, row['source_id'], row.get('cand_id', ''))
                        drop = bool((mm and k is None and (r, mm.group(1)) in DROPPED)
                                    or (ck and OMIS.key((ck.group(1), ck.group(3), ck.group(2), ck.group(4), ck.group(5)),
                                                        row['cand_id'][ck.end():])))
                        agree('member row dropped', drop, K.take('member row dropped', mkey), what)
                        if drop:
                            REC.append(dict(kind='member row', batch=TAG[r], source=row['source_id'],
                                            cand_id=row.get('cand_id', ''), stokes=S, tier=T, band=band))
                            continue
                        rj = bool(mm and (r, mm.group(1), T) in REJ.OUT)
                        agree('member row re-judged', rj, K.take('member row re-judged', mkey), what)
                        if rj:
                            REC.append(dict(kind='member row re-judged', batch=TAG[r], source=row['source_id'],
                                            cand_id=row.get('cand_id', ''), stokes=S, tier=T, band=band))
                            continue
                        c2 = bool(mm and C2.jc_card2(r, mm.group(1), row.get('cand_id', '')))
                        agree('card2 member row dropped', c2, K.take('card2 member row dropped', mkey), what)
                        if c2:
                            REC.append(dict(kind='card2 member row', batch=TAG[r], source=row['source_id'],
                                            cand_id=row.get('cand_id', ''), stokes=S, tier=T, band=band))
                            continue
                        if k is None:
                            raise ViewRefusal('T3 %s %s member source %s is in no combined identity' % (TAG[r], cdir, row['source_id']))
                        memr[band].append(dict(row, source_id='%s_%s_%s' % (name[k], mm.group(2), mm.group(3)), batch_run=TAG[r]))
                        MEMBERS_PD[page_dir(S, T, '%s_%s' % (name[k], band), TSFX)].add(row['cand_id'])
                        n_mem_rows += 1
            out_s = collections.defaultdict(list)
            for (band, k), rr0 in surv.items():
                dcache = {}
                rr = []
                for r, row in rr0:
                    src = os.path.join(r, row['review_dir'])
                    catalog = '%s/%s_sources_survivors.csv' % (cdir, band)
                    what = '%s %s %s' % (TAG[r], cdir, row['source_id'])
                    if src not in dcache:
                        dcache[src] = rows(os.path.join(src, 'detections.csv'))
                    ad = dcache[src]
                    if not ad:        # G-V7 (F4): an incomplete input is never booked as a deep omission
                        raise ViewRefusal('G-V7 %s %s source %s: its page %s holds no detection (detections.csv missing or empty) - '
                                          'an incomplete batch run' % (TAG[r], cdir, row['source_id'], row['review_dir']))
                    kd = [d for d in ad if not d_omitted(d) and not d_card2(d)]
                    note = DO.SOURCE_EMPTY_CARD2_NOTE if any(d_card2(d) for d in ad) else DO.SOURCE_EMPTY_NOTE
                    agree('source dropped', not kd, K.take('source dropped', (r, catalog, row['source_id'], note)), what)
                    if not kd:
                        REC.append(dict(kind='source dropped', batch=TAG[r], identity=row['identity_id'], source=row['source_id'],
                                        stokes=S, tier=T, band=band, note=note))
                        continue
                    # THE RE-POINT (dbc40a6; the refactor port review's defect RP fixed): the row's best detection is the one
                    # of its own (ts, subband) - best_cand_id names the member SOURCE, never a detection
                    mine = collections.OrderedDict()
                    top = None
                    if len(kd) < len(ad):
                        bd = [d for d in ad if d['ts'] == row.get('ts') and d['subband'] == row.get('subband')]
                        if not bd:
                            raise ViewRefusal('G-V9 %s source %s names a best detection (ts %s, subband %s) absent from its '
                                              'detections.csv' % (what, row['source_id'], row.get('ts'), row.get('subband')))
                        kdk = {dkey(d) for d in kd}
                        if all(dkey(d) not in kdk for d in bd):
                            top = max(kd, key=lambda d: abs(float(d['snr'])))
                            mine.update([('best_cand_id', '%s_%s_%s_%s' % (row['identity_id'], top['lst'], top['date'], band)),
                                         ('best_snr', top['snr']), ('ts', top['ts']), ('subband', top['subband']),
                                         ('fit_verdict', top.get('fit_verdict', '')), ('peak_jy_best', ''),
                                         ('deep_omission', DO.REPOINT_NOTE)])
                        if (r, row['identity_id'], T) in REJ.ROUTES:
                            mine['rejudged_routes'] = REJ.ROUTES[(r, row['identity_id'], T)]
                    got = K.repoint.pop((r, catalog, row['source_id']), None)
                    want_kind = None if not mine else ('source re-pointed' if top is not None else 'source routes')
                    if (got[0] if got else None) != want_kind or (got and list(got[1].items()) != list(mine.items())):
                        raise ViewRefusal('G-K %s: the rule %s, the kept product %s' % (
                            what, '%s %s' % (want_kind, json.dumps(mine)) if want_kind else 'keeps it as it is',
                            '%s %s' % (got[0], json.dumps(got[1])) if got else 'keeps it as it is'))
                    if top is not None:
                        REC.append(dict(kind='source partial', batch=TAG[r], identity=row['identity_id'], source=row['source_id'],
                                        stokes=S, tier=T, band=band, note=DO.repoint_record_note(row, top)))
                    if mine:
                        row = dict(row)
                        for x, v in mine.items():           # production's key order (a new column's place in the header)
                            row[x] = v
                    rr.append((r, row))
                if not rr:
                    continue
                sid = '%s_%s' % (name[k], band)
                pd = page_dir(S, T, sid, TSFX)
                os.makedirs(os.path.join(PART, pd), exist_ok=True)
                kept_of = None
                dets, seen = [], set()
                for r, row in rr:
                    src = os.path.join(r, row['review_dir'])
                    for fn in sorted(os.listdir(src)):
                        if fn in ('detections.csv', 'astrometry.png'):
                            continue
                        pm = PANEL.match(fn)
                        skip = bool(pm and OMIS.key((pm.group(1), pm.group(3), pm.group(2), S, T), DO.panel_image(pm.group(4))))
                        agree('panel dropped', skip, K.take('panel dropped', (r, row['review_dir'], row['source_id'], S, T, fn)),
                              '%s %s' % (row['review_dir'], fn))
                        if skip:
                            REC.append(dict(kind='panel', batch=TAG[r], source=row['source_id'], stokes=S, tier=T,
                                            band=pm.group(3), note=fn))
                            continue
                        c2 = bool(pm and S == 'V' and T == '10s' and C2.JC_C2 and C2.jc_card2(
                            r, row['identity_id'], '%s_%s_%s_%s_%s_%s_%s' % (pm.group(1), pm.group(2), pm.group(3), S, T,
                                                                             pm.group(4)[:8], pm.group(4)[8:])))
                        agree('card2 panel dropped', c2, K.take('card2 panel dropped', (r, row['review_dir'], row['source_id'], S,
                                                                                        T, fn)), '%s %s' % (row['review_dir'], fn))
                        if c2:
                            REC.append(dict(kind='card2 panel', batch=TAG[r], source=row['source_id'], stokes=S, tier=T,
                                            band=pm.group(3), note=fn))
                            continue
                        n_link_expected += 1
                        dst = os.path.join(PART, pd, fn)
                        if os.path.exists(dst):
                            dst = os.path.join(PART, pd, '%s_%s' % (TAG[r], fn))
                        # production hard-links every panel (one file system: its view and batches on /data07). A view
                        # under a production root refuses a link that fails (G-V5: a copy of ~3 million panels would fill
                        # the review disk); a scratch / parity view COPIES the panel - same bytes - where the link fails
                        # (another file system, or the parity guard, which refuses a hard link of a file outside the parity
                        # roots: core/guard.py), and the rest of that batch run's panels without trying again; counted in
                        # the stats as panels_copied
                        if r in copy_runs:
                            shutil.copy2(os.path.join(src, fn), dst)
                            xstat['panels_copied'] += 1
                        else:
                            try:
                                os.link(os.path.join(src, fn), dst)
                            except OSError as e:
                                if not job.get('copy_fallback'):
                                    raise ViewRefusal('G-V5 a panel could not be hard-linked (%s: %s) - the view and its batch '
                                                      'runs must share one file system' % (os.path.join(src, fn), e))
                                log('%s: a panel cannot be hard-linked here (%s) - this scratch view copies its panels' % (TAG[r], e))
                                copy_runs.add(r)
                                shutil.copy2(os.path.join(src, fn), dst)
                                xstat['panels_copied'] += 1
                        n_links += 1
                        PANELS_PD[pd].append(fn)
                    for d in (dcache[src] if src in dcache else rows(os.path.join(src, 'detections.csv'))):
                        if d_omitted(d) or d_card2(d):
                            continue
                        if dkey(d) not in seen:
                            seen.add(dkey(d))
                            dets.append(dict(d, batch_run=TAG[r]))
                    if kept_of is None:
                        kept_of = {dkey(m['_row']) for i in members_of[k] for m in clusters[i]}
                    kept_here = [d for d in dcache.get(src, ()) if not d_omitted(d) and not d_card2(d)]
                    xstat['page_detections'] += len(kept_here)
                    xstat['page_detections_not_of_identity'] += sum(1 for d in kept_here if dkey(d) not in kept_of)
                pdets = PDETS.setdefault(pd, collections.OrderedDict())
                for d in dets:
                    pdets.setdefault(dkey(d), d)
                write_csv(os.path.join(PART, pd, 'detections.csv'), hdr_union(list(pdets.values())), list(pdets.values()))
                best = max(rr, key=lambda x: abs(float(x[1].get('best_snr') or 0)))[1]
                out_s[band].append(dict(best, source_id=sid, identity_id=name[k], review_dir=pd, n_det=len(dets),
                                        n_det_identity=sum(len(clusters[i]) for i in members_of[k]),
                                        batch_members='+'.join(sorted({TAG[r] for r, _ in rr}))))
                q = qrows[(S, T)]
                q[name[k]] = max(q.get(name[k], 0.0), abs(float(best.get('best_snr') or 0)))
                if not any(d['ts'] == best.get('ts') and d['subband'] == best.get('subband') for d in pdets.values()):
                    T7_BEST.append('%s %s' % (pd, best.get('ts')))      # T7: the combined row's best detection is one it keeps
            for band in sorted(set(out_s) | set(memr)):
                rr = sorted(out_s.get(band, []), key=lambda x: -abs(float(x.get('best_snr') or 0)))
                for i, x in enumerate(rr):
                    x['rank'] = 'rank%04d' % (i + 1)
                # PRODUCTION FORMAT: hdr_union already ends with batch_members (+ batch_run), so the header names
                # batch_members twice and every row writes it twice (snapshot_combined_view.py:299) - kept for byte parity
                write_csv(os.path.join(PART, cdir, '%s_sources_survivors.csv' % band), hdr_union(rr) + ['batch_members'], rr)
                write_csv(os.path.join(PART, cdir, '%s_source_members.csv' % band), hdr_union(memr.get(band, [])),
                          memr.get(band, []))
            hdr0 = []
            for r in RUNS:
                p = os.path.join(r, 'queue_%s_%s.txt' % (S, T))
                if os.path.exists(p):
                    with open(p) as fh:
                        hdr0 = [ln.rstrip('\n') for ln in fh if ln.startswith('#')]
                    break
            with open(os.path.join(PART, 'queue_%s_%s.txt' % (S, T)), 'w') as fh:
                fh.write(QUEUE_NAME % (S, T, len(RUNS)))
                fh.write(QUEUE_RUN % (RUNNAME, ' '.join(TAG[r] for r in RUNS)))
                fh.write(VIEW_LIMITS)
                fh.write(DEEP_OMISSION_LINE % (n_om, n_all, len(OM), DEEP_OMISSION_WB % len(OMWB) if OMWB else ''))
                fh.write(CARD2_REJUDGE_LINE % (n_c2, len(REJ.OUT)))
                for ln in hdr0:
                    if not ln.startswith(('# queue_name', '# run')):
                        fh.write(ln + '\n')
                for nm in sorted(qrows[(S, T)], key=lambda n: (-qrows[(S, T)][n], n)):
                    fh.write(nm + '\n')
    left = K.unapplied()
    if left:
        raise ViewRefusal('G-K the kept product holds rows no served row matched: %s - it is not the drop of these runs'
                          % json.dumps(left, sort_keys=True))
    # T3 conservation
    if n_links != n_link_expected or any(v != 1 for v in src_seen.values()):
        raise ViewRefusal('T3 panels linked %d of %d; batch source rows seen more than once: %d'
                          % (n_links, n_link_expected, sum(v != 1 for v in src_seen.values())))
    log('T3 PASS: %d batch per-tier source rows -> combined sources, %d member rows, %d panels hard-linked (= the member '
        'directories\')' % (n_src_rows, n_mem_rows, n_links))
    # T4 queues
    for (S, T), q in qrows.items():
        with open(os.path.join(PART, 'queue_%s_%s.txt' % (S, T))) as fh:
            names = [ln.strip() for ln in fh if ln.strip() and not ln.startswith('#')]
        if len(names) != len(set(names)) or set(names) != set(q):
            raise ViewRefusal('T4 queue_%s_%s lists %d names, %d distinct, %d identities expected'
                              % (S, T, len(names), len(set(names)), len(q)))
    log('T4 PASS: queues %s' % ' '.join('%s_%s=%d' % (S, T, len(q)) for (S, T), q in sorted(qrows.items())))
    # T5 every member row's detection in its page directory's detections.csv
    miss, ex5 = 0, []
    for pd_, cs in MEMBERS_PD.items():
        m5 = cs - {d['cand_id'] for d in PDETS.get(pd_, {}).values()}
        if m5:
            miss += len(m5)
            if len(ex5) < 3:
                ex5.append('%s: %s' % (pd_, sorted(m5)[0]))
    if miss:
        raise ViewRefusal('T5 %d member detections in no page directory\'s detections.csv (e.g. %s)' % (miss, ex5))
    log('T5 PASS: %d member detections, each in its page directory\'s detections.csv (%d page directories)'
        % (sum(len(v) for v in MEMBERS_PD.values()), len(MEMBERS_PD)))
    # T6 one tier per page directory
    n6r = n6p = 0
    e6 = []
    n6u = collections.Counter()
    for pd_, dd in PDETS.items():
        for d in dd.values():
            if d.get('tier') != dir_tier(pd_, TSFX):
                n6r += 1
                if len(e6) < 3:
                    e6.append('%s: row %s tier %s' % (pd_, d['cand_id'], d.get('tier')))
    for pd_, fns in PANELS_PD.items():
        for fn in fns:
            t = panel_tier(fn)
            if t is None:
                n6u[os.path.splitext(fn)[1] or fn] += 1
            elif t != dir_tier(pd_, TSFX):
                n6p += 1
                if len(e6) < 3:
                    e6.append('%s: panel %s' % (pd_, fn))
    if n6r or n6p:
        raise ViewRefusal('T6 %d detections.csv rows and %d panels of the other tier in a page directory (e.g. %s)'
                          % (n6r, n6p, e6))
    log('T6 PASS: %d page directories, each one tier (%d detections.csv rows, %d panels with a tier token; %d linked files '
        'without one: %s)' % (len(PDETS), sum(len(v) for v in PDETS.values()), sum(len(v) for v in PANELS_PD.values())
                              - sum(n6u.values()), sum(n6u.values()), dict(n6u) or 'none'))
    # T7 (dbc40a6): no Card 2 detection in any identity or page directory, every (batch source, tier) re-judged out met in its
    # batch's per-tier catalogs (it qualified, so it had rows) and none of its rows used, every combined per-tier row ranked by a
    # detection it keeps
    n7c = sum(1 for c in clusters for m in c if d_card2(m['_row'])) + sum(1 for v in PDETS.values() for d in v.values()
                                                                           if d_card2(d))
    n7r = [k_ for k_ in REJ.OUT if not REJ_SEEN[k_]]
    if n7c or n7r or T7_BEST:
        raise ViewRefusal('T7 %d Card 2 detections left in the view; %d re-judged (batch source, tier) with no per-tier row (e.g. '
                          '%s); %d combined rows ranked by a detection they do not keep (e.g. %s)'
                          % (n7c, len(n7r), n7r[:3], len(T7_BEST), T7_BEST[:3]))
    log('T7 PASS: 0 of %d Card 2 detections left (identities and %d page directories); %d re-judged (batch source, tier), every '
        'one met in its batch\'s per-tier catalogs (%d rows not used); every combined per-tier row ranked by a detection it '
        'keeps; %d (identity, frame) with Card 2 and other detections (kept)' % (n_c2, len(PDETS), len(REJ.OUT),
                                                                                  sum(REJ_SEEN.values()), n_mixed_c2))
    if OMIS.absent:     # V_combview review T2: the omit stage refuses these (G7) - here the inputs changed since
        raise ViewRefusal('G-V10 %d lookups of a per-image key found no image in the cell\'s member table (e.g. %s) - the image '
                          'ids of the detections and the member tables differ' % (sum(OMIS.absent.values()),
                                                                                    sorted(OMIS.absent)[:2]))
    # run-level files
    dts = []
    for r in RUNS:
        dts += rows(os.path.join(r, 'detection_ts.csv'))
    if dts:
        write_csv(os.path.join(PART, 'detection_ts.csv'), list(dts[0].keys()), dts)
    shutil.copyfile(K.decisions, os.path.join(PART, 'deep_omission.csv'))       # the stage's table = production's, byte for byte
    write_csv(os.path.join(PART, 'deep_omission_dropped.csv'), hdr_union(REC)[:-1] if REC else ['kind'], REC)
    shutil.copyfile(os.path.join(K.dir, DO.REJUDGED_FILE), os.path.join(PART, 'rejudged.csv'))   # production's rejudged.csv
    omc = collections.Counter(x['kind'] for x in REC)
    with open(os.path.join(PART, 'deep_omission_summary.txt'), 'w') as fh:
        fh.write(SUMMARY_LINE1 % n_om + SUMMARY_LINE1_NEW % (n_c2, len(REJ.OUT)) + '\n')
        fh.write(SUMMARY_LINE2 % (n_all, n_in, K.n_dec, len(OM), SUMMARY_WB % len(OMWB) if OMWB else '',
                                  ', '.join('%s %d' % kv for kv in sorted(omc.items()))))
    rjs = K.summary.get('rejudge') or {}
    if rjs.get('per_tier_rows_dropped') != sum(REJ_SEEN.values()) or rjs.get('rejudged_out') != len(REJ.OUT):
        raise ViewRefusal('G-K re-judging: the omit product\'s summary drops %s per-tier rows of %s re-judged (source, tier), this '
                          'build %d of %d' % (rjs.get('per_tier_rows_dropped'), rjs.get('rejudged_out'), sum(REJ_SEEN.values()),
                                              len(REJ.OUT)))
    summ = collections.OrderedDict([
        ('out', RUNNAME), ('batches', [TAG[r] for r in RUNS]), ('batch_identities', len(clusters)), ('detections', n_in),
        ('deep_omission', collections.OrderedDict([('on', True), ('served', n_all), ('omitted', n_om), ('omit_keys', len(OM)),
                                                   ('omit_wb_image_keys', len(OMWB)), ('record', dict(omc))])),
        ('combined_identities', len(merged)), ('multi_batch_identities', n_multi), ('renamed', sum('~' in n for n in name)),
        ('merge', mstats), ('panels_linked', n_links), ('per_tier_source_rows', n_src_rows), ('member_rows', n_mem_rows),
        ('queues', {'%s_%s' % k: len(v) for k, v in qrows.items()}), ('tests', TESTS_PASSED),
        ('seconds', round(time.time() - T0)),
        ('card2', collections.OrderedDict([('ratio', job['card2_ratio']), ('removed_10s_v_records', C2.n['removed_10s']),
                                           ('removed_10s_v_served', C2.n['removed_10s_served']),
                                           ('not_evaluated_10s_v', C2.n['not_evaluated_10s']), ('removed_from_view', n_c2),
                                           ('mixed_identity_frames_kept', n_mixed_c2)])),
        ('rejudge', collections.OrderedDict([(k_, rjs.get(k_)) for k_ in (
            'served_sources_reproduced', 'touched', 'still_qualify', 'rejudged_out', 'every_member_removed', 'per_tier_rows_dropped',
            'removed_detections_in_records', 'g3d_removals_sharing_a_group_with_card2')]))])
    with open(os.path.join(PART, 'combined_view_summary.json'), 'w') as fh:
        json.dump(summ, fh, indent=1, default=str)
    os.rename(PART, OUT)
    stats = dict(summary=summ, kept_dir=K.dir, kept_applied=dict(kept=n_in, omitted=n_om, card2=n_c2, decisions=K.n_dec,
                                                                 omit_keys=len(OM), omit_wb_image_keys=len(OMWB)),
                 record=dict(omc), page_directories=len(PDETS), stale_best=len(T7_BEST),
                 rejudged_out=len(REJ.OUT), page_detections=xstat['page_detections'],
                 page_detections_not_of_identity=xstat['page_detections_not_of_identity'], panels_copied=xstat['panels_copied'],
                 absent_images=sum(OMIS.absent.values()), seconds=round(time.time() - T0, 1))
    log('COMBINED-VIEW-OK %s' % json.dumps({k: v for k, v in summ.items() if k != 'merge'}, default=str))
    return stats


def complete_runs(runs, tag, tiers):
    """G-V7 (V_combview review F4), the build's own re-check of the omit stage's G4b: each batch run's per-tier survivors hold
    exactly the identities its served_tiers.csv serves in each tier (chain_outcome 'served', qualifies 1)."""
    for r in runs:
        st = os.path.join(r, 'served_tiers.csv')
        if not os.path.isfile(st):
            raise ViewRefusal('G-V7 %s has no served_tiers.csv: its per-tier catalogues cannot be shown complete' % tag[r])
        want = collections.defaultdict(set)
        for x in rows(st):
            if x.get('chain_outcome') == 'served' and str(x.get('qualifies')) == '1':
                want[x['tier']].add(x['identity_jname'])
        for T in tiers:
            have = {x['identity_id'] for p in glob.glob(os.path.join(r, 'source_catalogs_[IV]_%s' % T, '*_sources_survivors.csv'))
                    for x in rows(p)}
            if have != want.get(T, set()):
                raise ViewRefusal('G-V7 %s tier %s: its per-tier survivors hold %d identities served_tiers.csv does not serve and '
                                  'lack %d it serves - an incomplete or inconsistent batch run'
                                  % (tag[r], T, len(have - want.get(T, set())), len(want.get(T, set()) - have)))


def child_main(job_path):
    """`python3.6 snapshot_view.py build JOB.json` (core/legacy.run): the build; the stats into job['result']; REFUSING + exit 2 on a
    gate (ViewRefusal), exit 1 on an error (the traceback in the log). A refused or failed build moves its <view>.part aside to
    <view>.refused_<UTC> (F7): the view name stays free, nothing is deleted."""
    with open(job_path) as f:
        job = json.load(f)

    def part_aside():                    # only the .part THIS build made (a G-V2 refusal never touches another's)
        part = job['out'].rstrip('/') + '.part'
        return move_aside(part) if job.get('_made_part') and os.path.isdir(part) else None
    try:
        stats = build(job)
    except ViewRefusal as e:
        aside = part_aside()
        print('REFUSING: %s%s' % (e, ('; the partial view moved aside to %s' % aside) if aside else ''), flush=True)
        return 2
    except Exception:
        aside = part_aside()
        if aside:
            print('the build failed; the partial view moved aside to %s' % aside, flush=True)
        raise
    tmp = job['result'] + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(stats, f, indent=1, default=str)
    os.replace(tmp, job['result'])
    return 0


# =====================================================================================================================
# the step side (Python 3.8, the serve step's process)
# =====================================================================================================================

def _CFG():
    from exopipe.phase3.config import CFG
    return CFG


def _refuse(msg):
    from exopipe.phase3.core.record import Refusal
    raise Refusal(msg)


def runs_arg(v):
    """The view's batch runs, in order, from --view-runs: a comma list, '@FILE' (one per line, '#' comments) or bare names under
    paths.SNAPSHOT_REVIEW_ROOT (deep_omission._runs_arg's reading); absolute paths."""
    if not v:
        return []
    if v.startswith('@'):
        try:
            with open(v[1:]) as fh:
                items = [ln.split('#', 1)[0].strip() for ln in fh]
        except OSError as e:
            _refuse(f'G-V1 --view-runs {v}: the list cannot be read ({e})')
    else:
        items = [x.strip() for x in v.split(',')]
    root = _CFG().paths.SNAPSHOT_REVIEW_ROOT
    return [os.path.abspath(x if os.path.isabs(x) else os.path.join(root, x)) for x in items if x]


class ViewPlan:
    """Where the view goes: production <SNAPSHOT_REVIEW_ROOT>/<view> (refused while paths.PRODUCTION_WRITES is False), a parity
    --out <out>/snapshot/<view> (plan.make's layout); the omit product beside it (serve.SNAPVIEW_OMIT_DIR) or --view-omit."""

    def __init__(self, args, out):
        CFG = _CFG()
        S = CFG.serve
        self.tier, self.family = 'snapshot', FAMILY
        self.out = os.path.abspath(out)
        self.parity = CFG.production_root_of(self.out) is None
        label = getattr(args, 'run_label', None)
        if not label:
            _refuse(f'G-V1 view name required: --run {S.SNAPSHOT_RUN_PREFIX}COMB_<YYYYMMDD> (production: Run_SNAPCOMB_<date>)')
        if not label.startswith(S.SNAPSHOT_RUN_PREFIX) or '/' in label:
            _refuse(f'G-V1 --run {label!r}: a view name starts with {S.SNAPSHOT_RUN_PREFIX!r} and names no directory')
        self.run_name = label
        self.U_src = os.path.join(CFG.paths.SNAPSHOT_REVIEW_ROOT, label)
        self.U = os.path.join(self.out, 'snapshot', label) if self.parity else self.U_src
        self.runs = runs_arg(getattr(args, 'view_runs', None))
        given = getattr(args, 'view_omit', None)
        self.omit_given = bool(given)
        self.omit_dir = os.path.abspath(given) if given else S.SNAPVIEW_OMIT_DIR.format(root=os.path.dirname(self.U), name=label)
        self.notes = []

    def unit(self, stage):
        return f'snapshot_{self.run_name}_{stage}'


def select_stages(args):
    from exopipe.phase3.steps.serve import select_stages as sel
    st = sel(STAGES, args)
    if getattr(args, 'view_omit', None) and 'omit' in st:
        if getattr(args, 'stages', None) and 'omit' in [s.strip() for s in args.stages.split(',')]:
            _refuse('--view-omit DIR names an omit product made earlier: do not also run the omit stage (--stages omit)')
        st = [s for s in st if s != 'omit']
    return st


def view_exists(plan):
    """[the view and / or its .part that exist now] (G-V2)."""
    return [p for p in (plan.U, plan.U + '.part') if os.path.exists(p)]


def gate_runs(plan, stages=STAGES):
    """G-V1 (runs), G-V2 (the view exists: a run with the omit or build stage refuses BEFORE any stage - the omit product of an
    existing view is its product of record and is never rewritten; V_combview review F2), G-V6 (injected) - before any stage."""
    if set(stages) & {'omit', 'build'} and view_exists(plan):
        _refuse(f'G-V2 {" and ".join(view_exists(plan))} exists - a view is never overwritten, nor its omit product '
                f'{plan.omit_dir} rewritten (give another --run; run filtercat / localize / ready alone on an existing view)')
    if not plan.runs:
        _refuse('G-V1 no batch run: --view-runs RUN,RUN|@FILE (the served batch runs of the view, in order)')
    if len(set(plan.runs)) != len(plan.runs):
        _refuse(f'G-V1 a batch run is listed twice: {sorted(r for r in plan.runs if plan.runs.count(r) > 1)[:2]}')
    bad = [r for r in plan.runs if not os.path.isdir(os.path.join(r, 'source_catalogs'))]
    if bad:
        _refuse(f'G-V1 {len(bad)} run(s) without source_catalogs (not a served batch run), e.g. {bad[:2]}')
    from exopipe.phase3.core import injected, record
    if not injected.active():
        for p in plan.runs + ([plan.omit_dir] if os.path.exists(plan.omit_dir) else []):
            hit = record.injected_runs(p)
            if hit:
                _refuse(f'G-V6 {injected.LABEL}: {p}: {hit[0][0]} holds an injected run\'s products ({hit[0][1]}) - an ordinary '
                        f'view never reads them')


def omit_stage(run, plan):
    """The deep-omission stage on the view's runs into plan.omit_dir (deep_omission.run_snapshot; its classes)."""
    from exopipe.phase3.lib.catalog import deep_omission as DO
    with run.unit(plan.unit('omit'), stage='omit', tier='snapshot', runs=plan.runs, out=plan.omit_dir) as u:
        rdir = _CFG().catalog_omission.REFUSALS
        DO.injected_record_gate(rdir)
        summ = DO.run_snapshot(plan.runs, plan.omit_dir, rdir=rdir, u=u, label=plan.run_name)
        u.meta['summary'] = summ
        u.note('deep omission (snapshot): %d decisions, %d OMIT keys, %d of %d served detections omitted'
               % (summ['decisions'], summ['omit_keys'], summ['omitted'], summ['served']))
        dec = os.path.join(plan.omit_dir, 'deep_omission.csv')
        u.add_output(dec, 'omission_decisions', tier='snapshot')
        u.add_output(os.path.join(plan.omit_dir, 'deep_omission_dropped.csv'), 'omission_record', tier='snapshot', decisions=dec)
        u.add_output(os.path.join(plan.omit_dir, 'deep_omission_served.csv'), 'omission_served', tier='snapshot')
        u.add_output(os.path.join(plan.omit_dir, 'deep_omission_rejudged.csv'), 'omission_rejudged', tier='snapshot')
        u.add_output(os.path.join(plan.omit_dir, 'deep_omission_kept.csv'), 'omission_kept', tier='snapshot', decisions=dec)
    return u


def check_omit_product(u, plan):
    """G-V3 on the step side: the omit product's files, its content tests now (N2's digests included), its runs."""
    from exopipe.phase3.core import validate
    d = plan.omit_dir
    need = ('deep_omission.csv', 'deep_omission_dropped.csv', 'deep_omission_served.csv', 'deep_omission_kept.csv',
            'deep_omission_rejudged.csv', 'deep_omission_summary.json')
    miss = [x for x in need if not os.path.isfile(os.path.join(d, x))]
    if miss:
        u.refuse(f'G-V3 the omit product {d} lacks {", ".join(miss)} (run the omit stage, or give --view-omit DIR)')
    with open(os.path.join(d, 'deep_omission_summary.json')) as f:
        summ = json.load(f)
    if [os.path.abspath(x) for x in summ.get('runs') or []] != plan.runs:
        u.refuse(f'G-V3 the omit product {d} was made for the runs {summ.get("runs")}, not the view\'s (same runs, same order)')
    dec = os.path.join(d, 'deep_omission.csv')
    bad = []
    for cls, name, ctx in (('omission_decisions', 'deep_omission.csv', {}), ('omission_record', 'deep_omission_dropped.csv',
                                                                            {'decisions': dec}),
                           ('omission_served', 'deep_omission_served.csv', {}),
                           ('omission_rejudged', 'deep_omission_rejudged.csv', {}),
                           ('omission_kept', 'deep_omission_kept.csv', {'decisions': dec})):
        p = os.path.join(d, name)
        u.add_input(p)
        for t in validate.run(cls, p, tier='snapshot', **ctx):
            u.check(f'G-V3 {t["name"]}', t['passed'], value=t['value'], detail=t['detail'], cls=cls, path=p)
            if not t['passed']:
                bad.append(f'{t["name"]}: {str(t["detail"])[:200]}')
    if bad:
        u.refuse('G-V3 the omit product fails its content tests: ' + '; '.join(bad[:3]))
    return summ


def check_decisions_current(u, plan, summ, recs):
    """G-V8 (V_combview review F5): the omit product's decisions are CURRENT - re-deciding now, from the refusal record and the
    fleet records the build reads (catalog_omission.REFUSALS, paths.SNAPSHOT_FLEET_OUT; the ones the omit product names) with the
    served counts the decisions were made with (deep_omission_served.csv; the child checks them against the served runs, G-V4),
    gives the omit product's deep_omission.csv byte for byte. A product made before the deep search refused more nights, or
    before the snapshot search rewrote a noise or member table, refuses - production decides inside its build from the current
    inputs. Also N1's injected-run guard on the record."""
    import filecmp
    from exopipe.phase3.lib.catalog import deep_omission as DO
    CFG = _CFG()
    rdir = os.path.abspath(CFG.catalog_omission.REFUSALS)
    proto = os.path.abspath(CFG.paths.SNAPSHOT_FLEET_OUT)
    made = (os.path.abspath(summ.get('refusals') or ''), os.path.abspath(summ.get('proto') or ''))
    if made != (rdir, proto):
        u.refuse(f'G-V8 the omit product {plan.omit_dir} was decided from the refusal record {made[0]} and the fleet {made[1]}; '
                 f'the build reads {rdir} and {proto}')
    DO._record_gate(rdir)
    served = DO.read_served(os.path.join(plan.omit_dir, 'deep_omission_served.csv'))
    now = os.path.join(recs, f'{plan.run_name}_decisions_now.csv')
    if 'cells' not in summ:
        u.refuse(f'G-V8 the omit product {plan.omit_dir} records no cells (an omit product made before the rate population of '
                 f'exopipe_ops dbc40a6) - run the omit stage again')
    cells_now = DO.view_cells(plan.runs, DO.Reader())
    if sorted(cells_now) != list(summ['cells']):
        u.refuse(f'G-V8 the batch runs\' searched cells changed since the omit stage ({len(cells_now)} now, {len(summ["cells"])} '
                 f'recorded) - run the omit stage again')
    inp_now = DO.rejudge_inputs(plan.runs)
    if json.loads(json.dumps(inp_now)) != summ.get('rejudge_inputs'):
        bad = [t for t in inp_now if json.loads(json.dumps(inp_now[t])) != (summ.get('rejudge_inputs') or {}).get(t)]
        u.refuse(f'G-V8 the re-judging / Card 2 inputs (detection record, chain summary, served_tiers.csv, card2_vleak.csv) of '
                 f'{", ".join(bad) or "the runs"} changed since the omit stage - its re-judging is stale; run the omit stage again')
    rows = DO.decide_snapshot(summ.get('lsts') or [], served, rdir=rdir, proto=proto, cells=set(summ['cells']))
    DO.write_decisions(now, rows, DO.TIER_SNAPSHOT)
    dec = os.path.join(plan.omit_dir, 'deep_omission.csv')
    same = filecmp.cmp(now, dec, shallow=False)
    if not same:
        a = set(open(dec).read().splitlines())
        b = set(open(now).read().splitlines())
        u.refuse(f'G-V8 the omit product is stale: re-deciding now from {rdir} and {proto} gives another table ({len(a - b)} '
                 f'rows of the product are gone, {len(b - a)} new; re-decided table {now}) - run the omit stage again')
    u.check('G-V8_decisions_current', same, value=len(rows), detail=f'{len(rows)} decisions re-made now = the omit product\'s')
    os.remove(now)


def move_aside(path):
    """Move a refused view (or its .part) aside to <view>.refused_<UTC> (V_combview review F7): the view name is free for a retry,
    nothing is deleted. Returns the new path, or None when ``path`` does not exist."""
    if not os.path.exists(path):
        return None
    base = path[:-len('.part')] if path.endswith('.part') else path
    aside = '%s.refused_%s' % (base, time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    n = 1
    while os.path.exists(aside):
        n += 1
        aside = '%s.refused_%s_%d' % (base, time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()), n)
    os.rename(path, aside)
    return aside


def build_stage(run, plan):
    from exopipe.phase3.core import legacy
    from exopipe.phase3.lib.serve import flow
    CFG = _CFG()
    S = CFG.serve
    with run.unit(plan.unit('build'), stage='build', tier='snapshot', view=plan.U, runs=plan.runs, omit=plan.omit_dir) as u:
        flow.excl_note(u)
        summ = check_omit_product(u, plan)
        if os.path.exists(plan.U) or os.path.exists(plan.U + '.part'):
            u.refuse(f'G-V2 {plan.U} or its .part exists - a view is never overwritten (give another --run)')
        os.makedirs(os.path.dirname(plan.U), exist_ok=True)
        dev = _dev(os.path.dirname(plan.U))
        other = [r for r in plan.runs if _dev(r) != dev]
        if other and not plan.parity:
            u.refuse(f'G-V5 the panels are HARD LINKS: {len(other)} batch run(s) on another file system than the view '
                     f'{os.path.dirname(plan.U)}, e.g. {other[:2]}')
        if other:
            u.note(f'{len(other)} batch run(s) on another file system than this scratch view: their panels are copied')
        for r in plan.runs:
            u.add_input(os.path.join(r, 'source_catalogs'))
        recs = os.path.join(plan.out, '_records', 'serve', 'snapshot_view')
        os.makedirs(recs, exist_ok=True)
        check_decisions_current(u, plan, summ, recs)        # G-V8 (F5)
        co = CFG.catalog_omission
        job = dict(out=plan.U, runs=plan.runs, kept_dir=plan.omit_dir, tiers=list(S.SNAPVIEW_TIERS),
                   stokes=list(S.SNAPVIEW_STOKES), suffix=dict(S.SNAPVIEW_PAGE_SUFFIX),
                   clv15=_legacy_root('clv15'), ops=_legacy_root('exopipe_ops'), deep_omission_py=_deep_omission_py(),
                   proto=summ['proto'], names=dict(noise=dict(co.SNAPSHOT_NOISE), members=co.SNAPSHOT_MEMBERS,
                                                   admission=co.SNAPSHOT_ADMISSION),
                   card2_ratio=co.CARD2_RATIO, card2_file=co.SNAPSHOT_CARD2, read_workers=co.READ_WORKERS,
                   result=os.path.join(recs, f'{plan.run_name}_build_result.json'),
                   copy_fallback=plan.parity)        # a scratch / parity view copies a panel whose hard link fails (G-V5)
        jp = os.path.join(recs, f'{plan.run_name}_build_job.json')
        with open(jp, 'w') as f:
            json.dump(job, f, indent=1)
        if os.path.exists(job['result']):
            os.remove(job['result'])
        u.note(f'omit product {plan.omit_dir}: {summ.get("served")} served, {summ.get("omitted")} omitted, '
               f'{summ.get("omit_keys")} OMIT keys')
        res = legacy.run(os.path.abspath(__file__), ['build', jp], python=S.SNAPVIEW_PYTHON, cwd=os.path.dirname(plan.U),
                         log=os.path.join(recs, f'{plan.run_name}_build.log'), timeout=S.SNAPVIEW_BUILD_TIMEOUT_S, unit=u,
                         refusal_codes=(2,))
        if not res.ok:
            aside = move_aside(plan.U + '.part')     # F7: a .part left by the child (it moves its own aside on a gate)
            u.refuse(f'the view build {"refused" if res.refused else "failed"}: {res.reason} - nothing moved into place'
                     + (f'; the partial view moved aside to {aside}' if aside else '') + f' (log {res.log})')
        with open(job['result']) as f:
            st = json.load(f)
        u.meta['build'] = {k: v for k, v in st.items() if k != 'summary'}
        u.meta['summary'] = {k: v for k, v in st['summary'].items() if k != 'merge'}
        u.note('view %s: %d combined identities (%d from two or more batches, %d renamed) of %d batch identities, %d detections, '
               '%d omitted; queues %s; T1-T7 PASS' % (plan.U, st['summary']['combined_identities'],
                                                     st['summary']['multi_batch_identities'], st['summary']['renamed'],
                                                     st['summary']['batch_identities'], st['summary']['detections'],
                                                     st['summary']['deep_omission']['omitted'], st['summary']['queues']))
        if st.get('panels_copied'):
            u.note(f'{st["panels_copied"]} panels COPIED, not hard-linked (a scratch / parity view: the link failed - another '
                   f'file system or the parity guard, which refuses a hard link of a file outside the parity roots; production '
                   f'links them; same bytes)')
        u.note(f'Card 2: {st["summary"]["card2"]["removed_from_view"]} detections removed; re-judging: '
               f'{st["rejudged_out"]} (batch source, tier) re-judged out; combined rows ranked by a detection they do not keep '
               f'(T7): {st["stale_best"]}; per-tier page detections not among their identity\'s kept detections: '
               f'{st["page_detections_not_of_identity"]} of {st["page_detections"]}')
        u.add_output(plan.U, 'snapshot_view', kept=os.path.join(plan.omit_dir, 'deep_omission_kept.csv'), runs=plan.runs)
        bad = u.failed_tests()
        if bad:              # F7: the built view fails its read-back tests - moved aside, the view name left free
            aside = move_aside(plan.U)
            u.refuse(f'the built view fails its read-back tests ({", ".join(sorted({t["name"] for t in bad}))}): moved aside to '
                     f'{aside}; nothing is left under {plan.U} - {bad[0]["name"]}: {str(bad[0].get("detail"))[:400]}')
    return u


def filtercat_stage(run, plan):
    """The driver's step 2 (comb_build_20261102.sh:50-80): the filter catalogue over the member batches' fit stores, the card2
    union, FILTER-INPUTS (class filter_catalog, per detection)."""
    from exopipe.phase3.lib.serve import scripts
    CFG = _CFG()
    S = CFG.serve
    with run.unit(plan.unit('filtercat'), stage='filtercat', tier='snapshot', view=plan.U) as u:
        if not os.path.isdir(plan.U):
            u.refuse(f'no view {plan.U} (run the build stage)')
        stores = [S.SNAPVIEW_FIT_STORE.format(batch_dir=CFG.catalog_snapshot.BATCH_DIR, run=os.path.basename(r)) for r in plan.runs]
        bad = [s for s in stores if not (os.path.isfile(s) and os.path.getsize(s) > 0)]
        if bad:
            u.refuse(f'F-G1 no fit store {bad[0]} ({len(bad)} of {len(stores)} member batches) - comb_build_20261102.sh:51')
        for s in stores:
            u.add_input(s)
        recs = os.path.join(plan.out, '_records', 'serve', 'snapshot_view')
        os.makedirs(recs, exist_ok=True)
        try:
            procs = filtercat_procs(S.SNAPVIEW_FILTERCAT_PROCS)
        except ValueError as e:
            u.refuse(str(e))
        u.meta['filtercat_procs'] = procs
        # SNAPCAT_PROCS always passed (exopipe_ops 1b9475c; no inherited value reaches the child): the Lustre rule's reader count
        res = scripts.run('snapshot_filter_catalog', [plan.U] + stores, python=S.SNAPVIEW_PYTHON, env={'SNAPCAT_PROCS': str(procs)},
                          log=os.path.join(recs, f'{plan.run_name}_filtercat.log'), unit=u, timeout=S.SNAPVIEW_BUILD_TIMEOUT_S)
        if not res.ok:
            u.refuse(f'F-G3 snapshot_filter_catalog.py {"refused" if res.refused else "failed"}: {res.reason}')
        u.note(scripts.last_line(res.tail))
        n, nb = card2_union(plan.U, plan.runs)
        u.note(f'card2_filter.csv: the union of {nb} member batches, {n} rows')
        u.add_output(os.path.join(plan.U, 'master_catalog.csv'), 'filter_catalog', per_detection=True)
        u.add_output(os.path.join(plan.U, 'card2_filter.csv'), 'snapshot_card2_union', runs=plan.runs)
    return u


def filtercat_procs(n):
    """The filter catalogue's reader count (serve.SNAPVIEW_FILTERCAT_PROCS, pages_snapshot.FILTERCAT_PROCS): an integer >= 1, else
    F-G4 refuses (production's int(os.environ['SNAPCAT_PROCS']) would raise on a non-integer and run serially below 2)."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f'F-G4 the filter catalogue reader count must be an integer >= 1, not {n!r}')
    return n


def card2_union(V, runs):
    """comb_build_20261102.sh:53-69 (the PYC heredoc): the first member's comment lines + '# combined view <name>: the union of
    its <n> member batches' card2_filter.csv' + its header + every member's body lines in batch order, each once; written
    aside and renamed. F-G2 a member without the file, or with another header. Returns (rows, members)."""
    first, hdr, seen, out = None, None, set(), []
    for r in runs:
        p = os.path.join(r, 'card2_filter.csv')
        if not os.path.isfile(p):
            _refuse(f'F-G2 card2 union: {p} is missing')
        with open(p) as fh:
            lines = fh.read().splitlines()
        com = [ln for ln in lines if ln.startswith('#')]
        body = [ln for ln in lines if not ln.startswith('#')]
        if not body:
            _refuse(f'F-G2 card2 union: {p} has no header')
        if first is None:
            first, hdr = com, body[0]
        elif body[0] != hdr:
            _refuse(f'F-G2 card2 union: {r} header differs')
        for ln in body[1:]:
            if ln not in seen:
                seen.add(ln)
                out.append(ln)
    with open(V + '/card2_filter.csv.part', 'w') as fh:
        fh.write('\n'.join(first + ["# combined view %s: the union of its %d member batches' card2_filter.csv"
                                    % (os.path.basename(V), len(runs)), hdr] + out) + '\n')
    os.replace(V + '/card2_filter.csv.part', V + '/card2_filter.csv')
    return len(out), len(runs)


def ready_stage(run, plan):
    """The driver's step 4: the serve-time exclusion gate on the view (G17, as snapshot_serve.sh:26-27 runs it before the
    registration), every snapshot_view test on the final files, VIEW-READY-FOR-HIDES."""
    from exopipe.phase3.lib.serve import exclusion_gate, flow
    with run.unit(plan.unit('ready'), stage='ready', tier='snapshot', view=plan.U) as u:
        if not os.path.isdir(plan.U):
            u.refuse(f'no view {plan.U} (run the build stage)')
        flow.excl_note(u)
        scan = exclusion_gate.scan_snapshot_run(plan.U)
        u.check('G17_serve_time_exclusion_gate', not scan.hits, value=len(scan.hits) if scan.hits else 0,
                detail=scan.summary())
        if scan.hits:
            u.refuse(f'G17 serve-time exclusion gate REFUSED the view {plan.run_name}: {scan.summary()} - not ready to register')
        for T in _CFG().serve.SNAPVIEW_TIERS:
            u.add_input(os.path.join(plan.U, f'source_catalogs_loc_{T}', 'localizations.csv'))
        u.add_output(plan.U, 'snapshot_view', kept=os.path.join(plan.omit_dir, 'deep_omission_kept.csv'), runs=plan.runs,
                     final=True)
        u.note(f'VIEW-READY-FOR-HIDES {plan.U}: the two snapshot hides and the cross-tier pair files are written next (the '
               f'refactor chat; no FP-filter step yet), then the registration on p8986-p8989 (not ported: snapshot_serve.sh '
               f'SNAP_REGISTER=1 with CARRY_FROM the served view)')
    return u


def _dev(p):
    """The file system (st_dev) of a path: the view and every batch run must share one (G-V5: the panels are hard links)."""
    return os.stat(p).st_dev


def _legacy_root(name):
    from exopipe.phase3.core.env import legacy_root
    return os.path.normpath(legacy_root(name))


def _deep_omission_py():
    from exopipe.phase3.lib.catalog import deep_omission
    return os.path.abspath(deep_omission.__file__)


def run(args, run_):
    """`exopipe3 serve --tier snapshot --family comb` (steps/serve.py dispatches here before the page plan)."""
    from exopipe.phase3.lib.serve import flow, snapshot_copy
    plan = ViewPlan(args, run_.out_root)
    stages = select_stages(args)
    gate_runs(plan, stages)
    if run_.dry_run:
        return describe(plan, stages)
    if 'omit' in stages and flow.halts(omit_stage(run_, plan), 'omit', need_done=True):
        return None
    if 'build' in stages:
        if flow.halts(build_stage(run_, plan), 'build', need_done=True):
            return None
    elif set(stages) & {'filtercat', 'localize', 'ready'}:
        flow.require_done(run_, plan.unit('build'), 'the view (its build stage)')
    if 'filtercat' in stages and flow.halts(filtercat_stage(run_, plan), 'filtercat', need_done=True):
        return None
    if 'localize' in stages:
        if flow.halts(snapshot_copy.localize_stage(run_, plan, env=_CFG().serve.SNAPVIEW_LOC_ENV), 'localize', need_done=True):
            return None
    if 'ready' in stages:
        for st in ('filtercat', 'localize'):
            if st not in stages:
                flow.require_done(run_, plan.unit(st), f'the view\'s {st} stage')
        ready_stage(run_, plan)
    return None


def describe(plan, stages):
    """--dry-run: the plan and the gates read now."""
    S = _CFG().serve
    print(f'serve --tier snapshot --family {FAMILY}: the combined snapshot view (lib/serve/snapshot_view.py)')
    print(f'  stages: {", ".join(stages)}')
    print(f'  view: {plan.U}' + ('   [EXISTS - the omit and build stages refuse (G-V2); filtercat, localize and ready run on it]'
                                  if os.path.exists(plan.U) else '')
          + ('   (parity: under --out)' if plan.parity else ''))
    print(f'  batch runs ({len(plan.runs)}, in order): ' + ' '.join(os.path.basename(r) for r in plan.runs))
    print(f'  omit product: {plan.omit_dir}' + (' (given)' if plan.omit_given else ' (the omit stage writes it)')
          + ('' if os.path.isdir(plan.omit_dir) or 'omit' in stages else '   [ABSENT - the build would refuse, G-V3]'))
    print(f'  build: child under site PYTHON[{S.SNAPVIEW_PYTHON!r}], the chain merge in place ({_legacy_root("clv15")})')
    print(f'  filtercat: snapshot_filter_catalog.py over ' + ', '.join(
        S.SNAPVIEW_FIT_STORE.format(batch_dir=_CFG().catalog_snapshot.BATCH_DIR, run=os.path.basename(r)) for r in plan.runs[:2])
          + (' ...' if len(plan.runs) > 2 else '') + f' (SNAPCAT_PROCS={S.SNAPVIEW_FILTERCAT_PROCS})')
    print(f'  localize: snapshot_localize_tiers.sh with {S.SNAPVIEW_LOC_ENV}')
    return 0


if __name__ == '__main__':          # the child entry point through core/legacy.run (Python 3.6 + clv15 + deep_omission.py)
    # the script's own directory (lib/serve: common.py, plan.py, queues.py, tests.py ...) never shadows a module the chain imports
    if sys.path and os.path.abspath(sys.path[0] or '.') == os.path.dirname(os.path.abspath(__file__)):
        sys.path.pop(0)
    if len(sys.argv) == 3 and sys.argv[1] == 'build':
        sys.exit(child_main(sys.argv[2]))
    sys.exit('usage: snapshot_view.py build JOB.json')
