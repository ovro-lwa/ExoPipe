"""THE RESTORE-ONLY WAVE'S PER-BATCH TEST: every detection a replaced batch run served, classified in the run that replaces it
(port of exopipe_ops snapshot_restore_wave_diff.py, 8ff3fc8 2026-11-04; pinned in tests/production_pin.py).

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
  absent          not in the new record at all.
The new run's detections served now and not before are counted, split by the restored_by tag.
FAIL when: a previously served detection is absent; the new record's restored rows (restored_by set) differ in number from its chain
summary's gate3d_restore_only.restored (or the summary has no such key); a restored row lacks gate_3d_installed = remove with
gate_3d_restore not remove, or an untagged row has that pair; a tag other than RESTORED_BY; a served detections.csv of the new run
(candidates/<band>/ of the chain, candidates/<Stokes>/ of the pages) lacks the restored_by column; the tagged rows of the chain's
detections.csv files differ from the record's served tagged rows.

  diff(old, new, out=None)   writes <out or new>/restore_wave_served_diff.csv (one row per previously served detection not served
                             now) and .json (counts, later stages, status) - production's bytes - and returns the json dict; the
                             count line is LINE % ...
  totals(runs)               (exit status, the totals line) over the runs' json files (the wave totals Gregg is shown)
Production's script runs as `snapshot_restore_wave_diff.py OLD NEW` (exit 1 on FAIL) and `--totals NEW ...`; this module is the
same with its output directory a parameter (a scratch or parity run never writes into a member run it reads).
"""
import collections
import csv
import glob
import gzip
import json
import os
import re

csv.field_size_limit(10 ** 9)
RESTORED_BY = "restore-only 3-D gate 20261104"            # clv15 phase3_snapshot_chain.RESTORED_BY (catalog_snapshot.RESTORED_BY)
POST3D = ["seed-post3d", "assoc", "seed-assoc", "anchor-assoc", "seed-anchor", "identity-merge", "seed-identity", "refl-floor",
          "vhf-event", "neg-lobe", "asymmetry", "seed-veto", "no-route", "not-designated", "fit-pending"]
GROUP_COL = {"seed-post3d": "assoc0_cluster", "assoc": "assoc_cluster", "seed-assoc": "assoc_cluster",
             "anchor-assoc": "anchor_cluster", "seed-anchor": "anchor_cluster",
             "identity-merge": "identity_index", "seed-identity": "identity_index"}   # the source-level stages: src_index
FALLBACK = ["src_index", "identity_index", "anchor_cluster", "assoc_cluster", "assoc0_cluster"]
CAUSES = ["served", "card2", "secondary", "secondary-pool", "3d-gate", "other-later", "other-earlier", "not-served", "absent"]
EXM = re.compile(r"^(.*?)\(.*?E=([0-9.]+)")
CSV_NAME, JSON_NAME = "restore_wave_served_diff.csv", "restore_wave_served_diff.json"
COLS = ["det_uid", "cand_id", "cell", "subband", "band", "stokes", "tier", "ts", "snr", "previous_identity", "cause",
        "removed_by_stage", "removed_detail", "group_column", "group_index", "n_restored_in_group", "restored_in_group",
        "restored_in_group_snr"]
WANT_OLD = ["det_uid", "cand_id", "cell", "subband", "band", "stokes", "tier", "ts", "snr", "identity_jname", "served",
            "catalog_promoted", "xm"]
WANT_NEW = ["det_uid", "cand_id", "ra_peak", "dec_peak", "snr", "served", "first_removed_by", "removed_detail", "identity_jname",
            "gate_3d", "gate_3d_installed", "gate_3d_restore", "restored_by", "catalog_promoted", "xm"] + FALLBACK
CHAIN_DIRS = (("Red", 1), ("Green", 1), ("Blue", 1), ("Full", 1), ("I", 0), ("V", 0))
LINE = ("RESTORE-WAVE-DIFF %s %s -> %s: previously served %d; served again %d; not served now %d = Card 2 %d + secondary %d (%s) + "
        "secondary through the catalog route's pool E %d + 3-D gate %d + other after the gate %d + other before the gate %d + not "
        "served without a stage %d; absent %d; restored %d (served %d); served now and not before %d (restored %d, other %d); %s")


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
            "%d + not served without a stage %d; absent %d; restored %d (served %d); served now and not before %d (restored %d, other "
            "%d)%s"
            % (len(rows), tot["prev_served"], tot["served"], tot["prev_served"] - tot["served"] - tot["absent"], tot["card2"],
               tot["secondary"], ", ".join("%s %d" % (k[10:], v) for k, v in sorted(tot.items()) if k.startswith("secondary@")),
               tot["secondary-pool"], tot["3d-gate"], tot["other-later"], tot["other-earlier"], tot["not-served"], tot["absent"],
               tot["restored"], tot["restored_served"], tot["gained"], tot["gained_restored"], tot["gained_other"],
               ("; NOT PASSED: " + "; ".join(bad)) if bad else "; every run PASS"))
    return (1 if bad else 0), line


def diff(old, new, out=None):
    """The test of NEW against the run it replaces, OLD (module docstring): writes CSV_NAME / JSON_NAME under ``out`` (default
    NEW, as production) and returns the JSON dict (its 'status' PASS or 'FAIL: ...'; 'line' the count line, not written)."""
    old, new = (a.rstrip("/") for a in (old, new))
    out = (out or new).rstrip("/")
    fails = []
    # ---- the previous run's served detections
    prev = {}
    for d in records(os.path.join(old, "detection_record.csv.gz"), WANT_OLD):
        if d["served"] == "1":
            prev[d["det_uid"]] = d
    # ---- the new run: the rows of the previously served, every restored row, every served row's key
    now, restored, gained = {}, {}, collections.Counter()
    gmap = {c: collections.defaultdict(list) for c in FALLBACK}
    n_rows, pair_bad, rb_bad = 0, [], []
    for d in records(os.path.join(new, "detection_record.csv.gz"), WANT_NEW):
        n_rows += 1
        u = d["det_uid"]
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
    # ---- classify
    by_old_id = collections.defaultdict(list)
    for u, p in prev.items():
        by_old_id[p["identity_jname"]].append(u)
    rows, cnt, later = [], collections.Counter(), collections.Counter()
    for u, p in prev.items():
        d = now.get(u)
        if d is None:
            cause, st, det, gcol, gval, rs = "absent", "", "", "", "", []
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
    with open(os.path.join(out, CSV_NAME + ".part"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    os.replace(os.path.join(out, CSV_NAME + ".part"), os.path.join(out, CSV_NAME))
    counts = collections.OrderedDict(
        prev_served=len(prev), **{c: cnt[c] for c in CAUSES}, restored=len(restored), restored_served=len(rs_served),
        gained=sum(gained.values()), gained_restored=gained["restored"], gained_other=gained["other"], new_record_rows=n_rows,
        detections_csv_files=n_files)
    j = collections.OrderedDict(old=old, new=new, tag_old=tag_of(old), tag_new=tag_of(new), counts=counts,
                                later_stages=dict(sorted(later.items())), status=status)
    with open(os.path.join(out, JSON_NAME), "w") as f:
        json.dump(j, f, indent=2)
    line = LINE % (tag_of(new), os.path.basename(old), os.path.basename(new), len(prev), cnt["served"],
                   len(prev) - cnt["served"] - cnt["absent"], cnt["card2"], cnt["secondary"],
                   ", ".join("%s %d" % kv for kv in sorted(later.items())) or "none", cnt["secondary-pool"], cnt["3d-gate"],
                   cnt["other-later"], cnt["other-earlier"], cnt["not-served"], cnt["absent"], len(restored), len(rs_served),
                   sum(gained.values()), gained["restored"], gained["other"], status)
    return dict(j, line=line)
