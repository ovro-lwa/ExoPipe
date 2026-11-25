"""THE RESTORE-ONLY WAVE'S PER-BATCH TEST: every detection a replaced batch run served, classified in the run that replaces it
(port of exopipe_ops snapshot_restore_wave_diff.py, 8ff3fc8 2026-11-04, with its 2026-11-15/16 changes 9cfcca4 3ca5efc 3000328
236ce9e 0904351 for the Release 1 view; pinned in tests/production_pin.py).

Gregg 2026-11-04 ~15:30 UTC, verbatim "Everything as you recommend." (R-GREGG-20261104-EVERYTHING-AS-RECOMMENDED, item 1: the
re-chain wave of all 36 snapshot batches with the restore-only 3-D gate, clv15 b8d1989); the refactor chat 2026-11-04: "every
detection the previous run served that the new run does not serve, with the stage that removed it and the restored detection that
changed its group ... a per-batch count line. Gregg gets the wave totals of restored and secondary removals before the new view is
served." The new view is served only when every restore-only member run's test PASSED (G27, lib/serve/snapshot_copy.py; production
snapshot_serve.sh 8ff3fc8).

Read from the two runs' own files. A detection is keyed by det_uid (cand_id + the search's peak pixel, the same key in every
record version). Every detection the PREVIOUS run's detection_record.csv.gz serves (served = 1) is classified in the NEW run's
record (CAUSES, in this order):
  served          served again;
  card2           removed by Card 2 per frame (10-s Stokes V; clv15 1279bd4, in the chain since 2026-11-04);
  secondary       removed by a stage after the 3-D gate (POST3D) while one or more restored detections (restored_by) share a group
                  with it in the new run - first the group the removing stage works on (GROUP_COL; the source-level stages:
                  src_index), then every other grouping down to the pre-gate cluster (FALLBACK), then the new groups of its previous
                  identity's other members;
  secondary-pool  removed after the gate with no restored detection in any of its groups, its previous source served by a catalog
                  route (E-promotion) the new run lacks while the same catalog match's chance expectation E is higher now (the
                  restored detections enlarge the pool catalog_route counts);
  3d-gate         removed by the 3-D gate under both tables (a difference of the previous run's code or inputs; listed for review);
  other-later     removed after the gate with no restored detection in its groups (listed for review);
  other-earlier   removed by a stage before the gate other than Card 2 (a difference of code or inputs; listed for review);
  not-served      in the record, not served, with no removing stage;
  excluded        not in the new record, on a cell-band excluded NOW (data_decisions.csv read at the test by the snapshot serve
                  exclusion gate's own read, lib/serve/exclusion_gate.fresh_exclusions('snapshot') = core.exclusions 'snapshot-gpu',
                  as production's coadd_full_pipeline.load_exclusions(scope='snapshot-gpu') + is_excluded: a subband by its exact
                  row, a wideband by the constituent rule) - the re-chain applied an exclusion written after the old run;
  reprocessed     not in the new record, on a cell-band not excluded, from a candidate table (<cell>/candidates_<subband>_<Stokes>
                  _median[_1min].csv) that a recorded re-run replaced after the old run's record was written and before the new
                  run's: a quarantine MANIFEST.csv lists the original - one a run record of `exopipe3 archive supersede` names
                  (recorded_quarantines: <records root>/_records/archive/supersede_*.json, catalog_snapshot.RESTORE_WAVE_RECORDS_
                  ROOTS) or, for the re-runs production made before cutover, one under catalog_snapshot.RESTORE_WAVE_REMADE_ROOTS
                  (<root>/<cell>/<UTC stamp>/); the quarantine directory is in removed_detail, and whether the new record holds a detection
                  of the same cell, subband, Stokes and tier within 2 px of the old peak pixel, at the same time stamp or within one
                  frame interval of it (10 s; 1 min): re-found and served, re-found and not served (its removing stage named), or
                  not re-found, the re-found at a shifted time stamp counted too;
  absent          not in the new record at all, on a cell-band not excluded, from a table not replaced since.
The new run's detections served now and not before are counted, split by the restored_by tag.
FIRST VERSION: OLD 'none' = a batch with no earlier run (production 9cfcca4, the Release 1 view's B7A1-B7L1): nothing was served
before, every check of the new run applies and its served detections count as gained; refused when the batch has any other run in
the new run's review directory.
FAIL when: a previously served detection is absent; the new run serves a detection on a cell-band excluded now; the exclusion list
cannot be read; a re-run quarantine directory has no UTC stamp; a supersede record cannot be read, or names a quarantine
directory without its MANIFEST.csv; the new record's restored rows (restored_by set) differ in number
from its chain summary's gate3d_restore_only.restored (or the summary has no such key); a restored row lacks gate_3d_installed =
remove with
gate_3d_restore not remove, or an untagged row has that pair; a tag other than RESTORED_BY; a served detections.csv of the new run
(candidates/<band>/ of the chain, candidates/<Stokes>/ of the pages) lacks the restored_by column; the tagged rows of the chain's
detections.csv files differ from the record's served tagged rows.

  diff(old, new, out=None, remade_roots=None, records_roots=None)
                             writes <out or new>/restore_wave_served_diff.csv (one row per previously served detection not served
                             now) and .json (counts, later stages, status) - production's bytes - and returns the json dict; the
                             count line is LINE % ...; OLD may be 'none' (first version); remade_roots default
                             catalog_snapshot.RESTORE_WAVE_REMADE_ROOTS, records_roots catalog_snapshot.RESTORE_WAVE_RECORDS_ROOTS
                             (None = archive.RECORDS_ROOT)
  totals(runs)               (exit status, the totals line) over the runs' json files (the wave totals Gregg is shown)
Production's script runs as `snapshot_restore_wave_diff.py OLD NEW` (exit 1 on FAIL), `none NEW` and `--totals NEW ...`; this
module is the same with its output directory a parameter (a scratch or parity run never writes into a member run it reads), the
exclusion read through v2's single reader, and the re-runs read from v2's own supersede records as well as from the roots of
production's (production hardcodes REMADE_ROOTS). With no supersede record the bytes are production's.
"""
import calendar
import collections
import csv
import glob
import gzip
import json
import os
import re
import time

from exopipe.phase3.core import placement

csv.field_size_limit(10 ** 9)
RESTORED_BY = "restore-only 3-D gate 20261104"            # clv15 phase3_snapshot_chain.RESTORED_BY (catalog_snapshot.RESTORED_BY)
POST3D = ["seed-post3d", "assoc", "seed-assoc", "anchor-assoc", "seed-anchor", "identity-merge", "seed-identity", "refl-floor",
          "vhf-event", "neg-lobe", "asymmetry", "seed-veto", "no-route", "not-designated", "fit-pending"]
GROUP_COL = {"seed-post3d": "assoc0_cluster", "assoc": "assoc_cluster", "seed-assoc": "assoc_cluster",
             "anchor-assoc": "anchor_cluster", "seed-anchor": "anchor_cluster",
             "identity-merge": "identity_index", "seed-identity": "identity_index"}   # the source-level stages: src_index
FALLBACK = ["src_index", "identity_index", "anchor_cluster", "assoc_cluster", "assoc0_cluster"]
CAUSES = ["served", "card2", "secondary", "secondary-pool", "3d-gate", "other-later", "other-earlier", "not-served", "excluded",
          "reprocessed", "absent"]
EXM = re.compile(r"^(.*?)\(.*?E=([0-9.]+)")
CSV_NAME, JSON_NAME = "restore_wave_served_diff.csv", "restore_wave_served_diff.json"
COLS = ["det_uid", "cand_id", "cell", "subband", "band", "stokes", "tier", "ts", "snr", "previous_identity", "cause",
        "removed_by_stage", "removed_detail", "group_column", "group_index", "n_restored_in_group", "restored_in_group",
        "restored_in_group_snr"]
WANT_OLD = ["det_uid", "cand_id", "cell", "subband", "band", "stokes", "tier", "ts", "snr", "identity_jname", "served",
            "catalog_promoted", "xm"]
WANT_NEW = ["det_uid", "cand_id", "cell", "subband", "ra_peak", "dec_peak", "snr", "served", "first_removed_by", "removed_detail",
            "identity_jname",
            "gate_3d", "gate_3d_installed", "gate_3d_restore", "restored_by", "catalog_promoted", "xm"] + FALLBACK
CHAIN_DIRS = (("Red", 1), ("Green", 1), ("Blue", 1), ("Full", 1), ("I", 0), ("V", 0))
LINE = ("RESTORE-WAVE-DIFF %s %s -> %s: previously served %d; served again %d; not served now %d = Card 2 %d + secondary %d (%s) + "
        "secondary through the catalog route's pool E %d + 3-D gate %d + other after the gate %d + other before the gate %d + not "
        "served without a stage %d; on cell-bands excluded since %d; from tables re-made since %d (re-found within 2 px and "
        "served %d, re-found and not served %d, of these re-found at a shifted time stamp %d; not re-found %d); absent %d; "
        "restored %d (served %d); served now and not before %d (restored %d, other %d); %s")


def tag_of(run):
    b = os.path.basename(run.rstrip("/"))
    return b[len("Run_SNAP"):].split("_")[0] if b.startswith("Run_SNAP") else b


def records(path, want):
    """dicts of the wanted columns of a gzipped detection record ('' for a column it lacks)"""
    with gzip.open(path, "rt", newline="") as f:
        rd = csv.reader(f)
        h = next(rd)
        miss = [c for c in want if c not in h]
        ix = [(c, h.index(c)) for c in want if c in h]
        for r in rd:
            d = {c: r[i] for c, i in ix}
            for c in miss:
                d[c] = ""
            yield d


def cts(cand):
    """cand_id <cell>_<subband>_<Stokes>_<tier>_<YYYYMMDD>_<HHMMSS> -> (the part before the time stamp, UTC seconds), else None"""
    a = cand.rsplit("_", 2)
    try:
        return a[0], calendar.timegm(time.strptime(a[1] + a[2], "%Y%m%d%H%M%S"))
    except (IndexError, ValueError):
        return None


def k5(cand, ra, dec):
    try:
        return (cand, "%.5f" % (float(ra) % 360.0), "%.5f" % float(dec))
    except ValueError:
        return (cand, ra, dec)


def totals(runs):
    """(status: 0 every run PASS / 1 not, the totals line) over the runs' JSON files (snapshot_restore_wave_diff.py --totals)."""
    tot, rows, bad = collections.Counter(), [], []
    for r in runs:
        p = os.path.join(r, JSON_NAME)
        if not os.path.exists(p):
            bad.append("%s: no %s" % (tag_of(r), JSON_NAME))
            continue
        j = json.load(open(p))
        rows.append(j)
        if j.get("status") != "PASS":
            bad.append("%s: %s" % (tag_of(r), j.get("status")))
        for k, v in j["counts"].items():
            if isinstance(v, int):
                tot[k] += v
        for k, v in j["later_stages"].items():
            tot["secondary@" + k] += v
    line = ("RESTORE-WAVE-TOTALS over %d runs: previously served %d; served again %d; not served now %d = Card 2 %d + secondary %d "
            "(%s) + secondary through the catalog route's pool E %d + 3-D gate %d + other after the gate %d + other before the gate "
            "%d + not served without a stage %d; on cell-bands excluded since %d; from tables re-made since %d (re-found within 2 "
            "px and served %d, re-found and not served %d, of these re-found at a shifted time stamp %d; not re-found %d); absent %d; "
            "restored %d (served %d); served now and not before %d (restored %d, other %d)%s"
            % (len(rows), tot["prev_served"], tot["served"],
               tot["prev_served"] - tot["served"] - tot["excluded"] - tot["reprocessed"] - tot["absent"], tot["card2"],
               tot["secondary"], ", ".join("%s %d" % (k[10:], v) for k, v in sorted(tot.items()) if k.startswith("secondary@")),
               tot["secondary-pool"], tot["3d-gate"], tot["other-later"], tot["other-earlier"], tot["not-served"], tot["excluded"],
               tot["reprocessed"], tot["reprocessed_refound_served"], tot["reprocessed_refound_not_served"],
               tot["reprocessed_refound_shifted"], tot["reprocessed_not_refound"], tot["absent"],
               tot["restored"], tot["restored_served"], tot["gained"], tot["gained_restored"], tot["gained_other"],
               ("; NOT PASSED: " + "; ".join(bad)) if bad else "; every run PASS"))
    return (1 if bad else 0), line


def _stamp_time(qd):
    """the UTC time of a quarantine directory named <YYYYmmddTHHMMSSZ>, else None"""
    try:
        return calendar.timegm(time.strptime(os.path.basename(qd.rstrip("/")), "%Y%m%dT%H%M%SZ"))
    except ValueError:
        return None


def recorded_quarantines(records_roots, fails, t_old, t_new):
    """MANIFEST.csv files of the re-runs v2 recorded: every run record of `exopipe3 archive supersede`
    (<root>/_records/archive/supersede_*.json; lib/archive/snapshot.run_supersede) names its quarantine root and cell (meta
    quar_root, cell) - every <quar_root>/<cell>/<UTC stamp>/MANIFEST.csv counts, whatever the unit's status (a refused audit, or
    a ship script stopped before its SHIPPED line, may already have moved tables) - and, when the script ran to its SHIPPED line,
    its quarantine directory (meta shipped.quarantine), which must hold a MANIFEST.csv. A record is its unit's LATEST run (a later
    plan or apply of the cell rewrites it): a re-run under a --quar-root that a later run of the same cell did not use is found
    only through RESTORE_WAVE_REMADE_ROOTS (else its rows FAIL as absent - never a silent pass). FAILs (appended to ``fails``)
    are limited to what can bear on a re-run between the two records: a record that cannot be read (or is not the record's shape)
    written after OLD's (mtime > t_old), and a named quarantine directory without its MANIFEST.csv whose stamp lies between the
    two records or cannot be read. Each MANIFEST.csv once (by real path)."""
    out = {}
    for root in records_roots:
        for p in sorted(glob.glob(os.path.join(root, "_records", "archive", "supersede_*.json"))):
            try:
                with open(p) as f:
                    meta = json.load(f).get("meta") or {}
                shipped = meta.get("shipped") or {}
                if not (isinstance(shipped, dict)
                        and all(isinstance(v, str) for v in (meta.get("quar_root") or "", meta.get("cell") or "",
                                                             shipped.get("quarantine") or ""))):
                    raise ValueError("meta.shipped not an object, or quar_root / cell / shipped.quarantine not text")
            except (OSError, ValueError, AttributeError) as e:
                try:
                    later = os.path.getmtime(p) > t_old
                except OSError:
                    later = True
                if later:
                    fails.append("a supersede run record could not be read: %s (%s)" % (p, e))
                continue
            found = []
            if meta.get("quar_root") and meta.get("cell"):
                found += glob.glob(os.path.join(meta["quar_root"], meta["cell"], "*", "MANIFEST.csv"))
            q = shipped.get("quarantine")
            if q:
                if os.path.isfile(os.path.join(q, "MANIFEST.csv")):
                    found.append(os.path.join(q, "MANIFEST.csv"))
                else:
                    t_q = _stamp_time(q)
                    if t_q is None or t_old < t_q < t_new:
                        fails.append("the supersede run record %s names the quarantine %s, which has no MANIFEST.csv" % (p, q))
            for m in found:
                out.setdefault(os.path.realpath(m), m)
    return sorted(out.values())


def remade_tables(old, new, roots, fails, records_roots=()):
    """(cell, candidate table name) -> the quarantine directory of the recorded re-run that replaced it after OLD's record was
    written and before NEW's (a stamp strictly between the two records' mtimes), and the number of such MANIFEST.csv files - those
    under ``roots`` (<root>/<cell>/<UTC stamp>/; production's list as it is, a file under two roots counted twice) and those of
    the supersede run records under ``records_roots`` (recorded_quarantines) not already among them; a quarantine directory
    without a UTC stamp is a FAIL (appended to ``fails``)."""
    remade, n_man = {}, 0
    t_old, t_new = (os.path.getmtime(os.path.join(r, "detection_record.csv.gz")) for r in (old, new))
    ms = [p for root in roots for p in glob.glob(os.path.join(root, "*", "*", "MANIFEST.csv"))]
    seen = {os.path.realpath(p) for p in ms}
    ms += [p for p in recorded_quarantines(records_roots, fails, t_old, t_new) if os.path.realpath(p) not in seen]
    for m in sorted(ms):
        qd = os.path.dirname(m)
        t_q = _stamp_time(qd)
        if t_q is None:
            fails.append("a re-run quarantine directory without a UTC stamp: %s" % qd)
            continue
        if not t_old < t_q < t_new:
            continue
        n_man += 1
        with open(m, newline="") as f:
            for r in csv.DictReader(f):
                b = os.path.basename(r["original_path"])
                if b.startswith("candidates_"):
                    remade[(os.path.basename(os.path.dirname(r["original_path"])), b)] = qd
    return remade, n_man


def table_of(p):
    """the candidate table a detection came from: candidates_<subband>_<Stokes>_median[_1min].csv"""
    return "candidates_%s_%s_median%s.csv" % (p["subband"], p["stokes"], "_1min" if p["tier"] == "1min" else "")


def pixel(u, cand):
    """the search's peak pixel (x, y) from det_uid = <cand_id>_<x>_<y>, else None"""
    xy = u[len(cand) + 1:].split("_")
    return (int(xy[0]), int(xy[1])) if len(xy) == 2 and all(v.lstrip("-").isdigit() for v in xy) else None


def diff(old, new, out=None, remade_roots=None, records_roots=None):
    """The test of NEW against the run it replaces, OLD, or 'none' for a first version (module docstring): writes CSV_NAME /
    JSON_NAME under ``out`` (default NEW, as production) and returns the JSON dict (its 'status' PASS or 'FAIL: ...'; 'line' the
    count line, not written). A first version whose batch has another run in NEW's directory REFUSES (core.record.Refusal),
    nothing written - production's exit 'REFUSING: ...'."""
    old, new = (a.rstrip("/") for a in (old, new))
    out = (out or new).rstrip("/")
    if remade_roots is None or records_roots is None:
        from exopipe.phase3.config import CFG
        if remade_roots is None:
            remade_roots = CFG.catalog_snapshot.RESTORE_WAVE_REMADE_ROOTS
        if records_roots is None:
            records_roots = CFG.catalog_snapshot.RESTORE_WAVE_RECORDS_ROOTS
            if records_roots is None:                         # None = the archive step's records root; () = none read
                records_roots = (CFG.archive.RECORDS_ROOT,)
    if isinstance(records_roots, str):
        records_roots = (records_roots,)
    fails = []
    # ---- the previous run's served detections; a FIRST VERSION has none
    prev = {}
    if old == "none":
        b = tag_of(new)
        from exopipe.phase3.config import CFG
        # beside NEW (production) and, for a NEW outside it (a scratch or parity tree), the review root too (v2: a first version
        # run elsewhere must not hide an earlier run of its batch in review)
        an = os.path.abspath(new)
        dirs = sorted({os.path.dirname(an)} | ({CFG.paths.SNAPSHOT_REVIEW_ROOT} if os.path.isdir(CFG.paths.SNAPSHOT_REVIEW_ROOT)
                                               else set()))
        pre = [x for d in dirs for x in os.listdir(d) if re.match(r"^Run_SNAP%s\d+_\d{8}$" % b[:3], x) and tag_of(x) != b
               and os.path.join(d, x) != an]
        if pre:
            from exopipe.phase3.core.record import Refusal
            raise Refusal("REFUSING: OLD 'none' but batch %s has other runs in review: %s" % (b[:3], sorted(pre)))
    else:
        for d in records(os.path.join(old, "detection_record.csv.gz"), WANT_OLD):
            if d["served"] == "1":
                prev[d["det_uid"]] = d
    # ---- the exclusion list, read now by the snapshot serve gate's own read: a detection missing from the new record on a
    # cell-band excluded since is 'excluded', not 'absent'; one the NEW run serves on a cell-band excluded now is a FAIL; any
    # failure to read the list is a FAIL
    excl, excl_read, n_excl = None, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), 0
    try:
        from exopipe.phase3.lib.serve.exclusion_gate import fresh_exclusions
        _ex, _cache = fresh_exclusions("snapshot"), {}
        n_excl = len(_ex.rows)

        def excl(cell, sb):
            if (cell, sb) not in _cache:
                if "_" not in cell:          # v2: a row without an <LST>_<date> cell is a FAIL, not a crash (production raises)
                    fails.append("a detection record row with no <LST>_<date> cell (%r, subband %r): its exclusion cannot be "
                                 "read" % (cell, sb))
                    _cache[(cell, sb)] = False
                else:
                    l_, d_ = cell.split("_", 1)
                    _cache[(cell, sb)] = bool(_ex.is_excluded(l_, sb, d_))
            return _cache[(cell, sb)]
    except Exception as e:                          # noqa: BLE001 - production's: any failure to read the list is a FAIL
        fails.append("the exclusion list could not be read (%s)" % e)
    # ---- candidate tables replaced by a recorded re-run between the two records
    remade, n_man = (remade_tables(old, new, remade_roots, fails, records_roots)
                     if old != "none" else ({}, 0))
    # ---- the new run: the rows of the previously served, every restored row, every served row's key
    now, restored, gained = {}, {}, collections.Counter()
    gmap = {c: collections.defaultdict(list) for c in FALLBACK}
    n_rows, pair_bad, rb_bad, served_excl = 0, [], [], []
    # re-made cells' rows by cand_id less its time stamp: <cell>_<subband>_<Stokes>_<tier> -> (t, x, y, det_uid, served, stage)
    remade_cells, nidx = {c for c, _ in remade}, collections.defaultdict(list)
    for d in records(os.path.join(new, "detection_record.csv.gz"), WANT_NEW):
        n_rows += 1
        u = d["det_uid"]
        if d["cell"] in remade_cells:
            xy, ct = pixel(u, d["cand_id"]), cts(d["cand_id"])
            if ct and xy:
                nidx[ct[0]].append((ct[1], xy[0], xy[1], u, d["served"], d["first_removed_by"]))
        pair = d["gate_3d_installed"] == "remove" and d["gate_3d_restore"] != "remove"
        if d["restored_by"]:
            if d["restored_by"] != RESTORED_BY:
                rb_bad.append(u)
            if not pair:
                pair_bad.append(u)
            restored[u] = d
            for c in FALLBACK:
                if d[c] != "":
                    gmap[c][d[c]].append(u)
        elif pair:
            pair_bad.append(u)
        if u in prev:
            now[u] = d
        elif d["served"] == "1":
            gained["restored" if d["restored_by"] else "other"] += 1
        if d["served"] == "1" and excl is not None and excl(d["cell"], d["subband"]):
            served_excl.append(u)
    if served_excl:
        fails.append("%d detections served by the new run on cell-bands excluded now (e.g. %s)"
                     % (len(served_excl), served_excl[:3]))
    # ---- classify
    by_old_id = collections.defaultdict(list)
    for u, p in prev.items():
        by_old_id[p["identity_jname"]].append(u)
    rows, cnt, later, refound = [], collections.Counter(), collections.Counter(), collections.Counter()
    for u, p in prev.items():
        d = now.get(u)
        if d is None:
            st, det, gcol, gval, rs = "", "", "", "", []
            q = remade.get((p["cell"], table_of(p)))
            if excl is not None and excl(p["cell"], p["subband"]):
                cause = "excluded"
            elif q:
                cause, det = "reprocessed", "table replaced by the re-run quarantined in " + q
                xy, ct = pixel(u, p["cand_id"]), cts(p["cand_id"])
                near = []
                if ct and xy:
                    step = 60 if p["tier"] == "1min" else 10
                    # the same time stamp first, then the nearest stamp within one frame interval; within 2 px
                    near = sorted((abs(t - ct[1]), max(abs(x - xy[0]), abs(y - xy[1])), uu, sv, stg)
                                  for t, x, y, uu, sv, stg in nidx.get(ct[0], [])
                                  if abs(t - ct[1]) <= step and max(abs(x - xy[0]), abs(y - xy[1])) <= 2)
                if near:
                    dt, r, uu, sv, stg = near[0]
                    k = "served" if sv == "1" else "not-served"
                    refound["shifted"] += dt > 0
                    det += "; re-found %s within %d px%s as %s%s" % (
                        k.replace("-", " "), r, " at a time stamp %d s away" % dt if dt else "", uu,
                        "" if k == "served" else " (removed by %s)" % (stg or "no stage"))
                else:
                    k = "not-found"
                    det += "; not re-found within 2 px and one frame interval"
                refound[k] += 1
            else:
                cause = "absent"
        elif d["served"] == "1":
            cnt["served"] += 1
            continue
        else:
            st, det = d["first_removed_by"], d["removed_detail"]
            gcol, gval, rs = "", "", []
            if st == "card2":
                cause = "card2"
            elif st == "bright-parent-3d":
                cause = "3d-gate"
            elif st in POST3D:
                own = GROUP_COL.get(st, "src_index")
                for c in [own] + [c for c in FALLBACK if c != own]:
                    if d.get(c, "") != "":
                        r_ = [x for x in gmap[c].get(d[c], []) if x != u]
                        if r_:
                            gcol, gval, rs = c, d[c], r_
                            break
                if not rs:
                    co = [x for x in by_old_id.get(p["identity_jname"], []) if x != u and x in now]
                    for c in FALLBACK:
                        vals = sorted({now[x][c] for x in co if now[x].get(c, "") != ""})
                        r_ = sorted({y for v in vals for y in gmap[c].get(v, [])})
                        if r_:
                            gcol, gval, rs = "previous identity members' " + c, ";".join(vals)[:200], r_
                            break
                if not gcol:
                    gcol, gval = own, d.get(own, "")
                cause = "secondary" if rs else "other-later"
                if not rs and p["catalog_promoted"] and not d["catalog_promoted"]:
                    eo = {m.group(1).strip(): float(m.group(2)) for x in p["xm"].split(";") for m in [EXM.match(x.strip())] if m}
                    en = {m.group(1).strip(): float(m.group(2)) for x in d["xm"].split(";") for m in [EXM.match(x.strip())] if m}
                    up = ["%s E %.3f -> %.3f" % (k, eo[k], en[k]) for k in eo if k in en and en[k] > eo[k]]
                    if up:
                        cause, gcol, gval = "secondary-pool", "catalog route E (pool)", "; ".join(up)[:200]
                if rs:
                    later[st] += 1
            elif st:
                cause = "other-earlier"
            else:
                cause = "not-served"
        cnt[cause] += 1
        rows.append(dict(det_uid=u, cand_id=p["cand_id"], cell=p["cell"], subband=p["subband"], band=p["band"], stokes=p["stokes"],
                         tier=p["tier"], ts=p["ts"], snr=p["snr"], previous_identity=p["identity_jname"], cause=cause,
                         removed_by_stage=st, removed_detail=det, group_column=gcol, group_index=gval, n_restored_in_group=len(rs),
                         restored_in_group=";".join(rs[:50]),
                         restored_in_group_snr=";".join(restored[x]["snr"] for x in rs[:50])))
    if cnt["absent"]:
        fails.append("%d previously served detections absent from the new record (e.g. %s)"
                     % (cnt["absent"], [r["det_uid"] for r in rows if r["cause"] == "absent"][:3]))
    # ---- the chain summary's restored count
    S = json.load(open(os.path.join(new, "chain_summary.json")))
    ro = S.get("gate3d_restore_only")
    if not ro:
        fails.append("the new run's chain summary has no gate3d_restore_only (not a restore-only run)")
    elif int(ro["restored"]) != len(restored):
        fails.append("record restored rows %d, chain summary restored %d" % (len(restored), int(ro["restored"])))
    if pair_bad:
        fails.append("%d rows whose restored_by tag disagrees with gate_3d_installed / gate_3d_restore (e.g. %s)"
                     % (len(pair_bad), pair_bad[:3]))
    if rb_bad:
        fails.append("%d rows with a restored_by other than %r" % (len(rb_bad), RESTORED_BY))
    rs_served = {k5(d["cand_id"], d["ra_peak"], d["dec_peak"]) for d in restored.values() if d["served"] == "1"}
    # ---- the served detections.csv files: the column everywhere; the chain's tagged rows = the record's served tagged rows
    nocol, n_files, tagged_csv = [], 0, set()
    for sub, chain_dir in CHAIN_DIRS:
        for p in glob.glob(os.path.join(new, "candidates", sub, "*", "detections.csv")):
            n_files += 1
            with open(p, newline="") as f:
                rd = csv.DictReader(f)
                if "restored_by" not in (rd.fieldnames or []):
                    nocol.append(p)
                    continue
                if chain_dir:
                    for r in rd:
                        if r["restored_by"]:
                            tagged_csv.add(k5(r["cand_id"], r["ra_peak"], r["dec_peak"]))
    if nocol:
        fails.append("%d served detections.csv without the restored_by column (e.g. %s)" % (len(nocol), nocol[:2]))
    if tagged_csv != rs_served:
        fails.append("tagged detections.csv rows %d, record served tagged rows %d (%d only in the files, %d only in the record)"
                     % (len(tagged_csv), len(rs_served), len(tagged_csv - rs_served), len(rs_served - tagged_csv)))
    # ---- write
    status = "FAIL: " + "; ".join(fails) if fails else "PASS"
    order = {c: i for i, c in enumerate(CAUSES)}
    rows.sort(key=lambda r: (order[r["cause"]], r["cell"], r["ts"], r["det_uid"]))
    os.makedirs(out, exist_ok=True)
    with placement.atomic(os.path.join(out, CSV_NAME)) as tmp, open(tmp, "w", newline="") as f:   # group and ACL (placement)
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    counts = collections.OrderedDict(
        prev_served=len(prev), **{c: cnt[c] for c in CAUSES}, restored=len(restored), restored_served=len(rs_served),
        gained=sum(gained.values()), gained_restored=gained["restored"], gained_other=gained["other"], new_record_rows=n_rows,
        detections_csv_files=n_files, served_on_excluded=len(served_excl), reprocessed_refound_served=refound["served"],
        reprocessed_refound_not_served=refound["not-served"], reprocessed_refound_shifted=refound["shifted"],
        reprocessed_not_refound=refound["not-found"])
    j = collections.OrderedDict(old=old, new=new, tag_old=tag_of(old), tag_new=tag_of(new), counts=counts,
                                later_stages=dict(sorted(later.items())), exclusions_read=excl_read, exclusion_rows=n_excl,
                                remade_manifests=n_man, remade_tables=len(remade), status=status)
    with open(os.path.join(out, JSON_NAME), "w") as f:
        json.dump(j, f, indent=2)
    line = LINE % (tag_of(new), os.path.basename(old), os.path.basename(new), len(prev), cnt["served"],
                   len(prev) - cnt["served"] - cnt["excluded"] - cnt["reprocessed"] - cnt["absent"], cnt["card2"], cnt["secondary"],
                   ", ".join("%s %d" % kv for kv in sorted(later.items())) or "none", cnt["secondary-pool"], cnt["3d-gate"],
                   cnt["other-later"], cnt["other-earlier"], cnt["not-served"], cnt["excluded"], cnt["reprocessed"],
                   refound["served"], refound["not-served"], refound["shifted"], refound["not-found"], cnt["absent"], len(restored),
                   len(rs_served), sum(gained.values()), gained["restored"], gained["other"], status)
    return dict(j, line=line)
