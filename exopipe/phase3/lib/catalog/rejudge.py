"""RE-JUDGING a served batch's sources on the members they keep after the deep omission and Card 2 - the port of exopipe_ops
snapshot_rejudge.py (dbc40a6; rejudge, run_inputs), for the snapshot tier's cumulative step (lib/serve/snapshot_view.py: its
omit stage runs this file once per batch run as a CHILD under the production interpreter, catalog_omission.REJUDGE_PYTHON).

RULING (Gregg 2026-11-04 ~09:4x UTC at the refactor chat's terminal; clv15 fix_ledger R-GREGG-20261104-DEEP-OMISSION-SHORT-TIERS-
ANSWERS): re-judging a source that lost detections to the omission on its remaining ones - "I see reflections coinciding with other
types of sources, including possibly real ones, so this makes sense, if I understand it correctly" - one change in production and
the refactor, the count of removed sources shown before serving (his go recorded 2026-11-04T15:30:05Z in
exopipe_ops/snapshot_batch/SNAPCOMB_CARD2_REJUDGE_SERVE_APPROVED).

METHOD (production's, line for line): clv15 phase3_snapshot_tierserve's reconstruction - for every served source of the batch, its
members from the per-detection record (detection_record.csv.gz) and the chain's OWN functions in the chain's order (seed rule ->
route_repeats -> reflection-band floor -> catalog route over the pool -> route pass) per tier, on the members left after
removed(row) (the deep omission per image and Card 2; the E pool without the removed rows). clv15 phase3_snapshot_chain /
phase3_snapshot_tierserve are run IN PLACE (the chain the batches were chained with). SELF-CHECKS (refuse on failure): the
full-member reconstruction reproduces the chain's mi_tiers, catalog promotion, routes and best member on every served source, and
the per-tier evaluation without removals reproduces the batch's served_tiers.csv on every (served source, tier).

Child entry (Python 3.6, the standard library + clv15 + deep_omission.py loaded from its file): python rejudge.py JOB.json
  job: run, decisions (deep_omission.csv), proto (the fleet root), names (the fleet file names), card2 (the run's Card 2 det_uids),
  clv15 (the chain's root), samples_dir, deep_omission_py, result (the JSON written: the touched (source, tier) rows in production's
  order, the counts, the removed detections by reason in first-seen order, the 3-D-gate neighbours). REFUSING + exit 2 on a gate.
"""
import csv
import gzip
import json
import os
import sys
from collections import Counter, OrderedDict

csv.field_size_limit(10 ** 9)


class RejudgeRefusal(Exception):
    """A self-check or input gate of the re-judging (production's RuntimeError('REFUSED: ...'))."""


def rejudge(run, removed, samples_dir, log=print):
    """removed(record_row) -> '' or the removal reason. Returns (touched {(identity_jname, tier): row}, counts, removed_uids
    {det_uid: reason}, g3d_neighbours) - snapshot_rejudge.rejudge (dbc40a6), verbatim in effect."""
    import phase3_snapshot_chain as C
    import phase3_snapshot_tierserve as TS
    summ = json.load(open(os.path.join(run, "chain_summary.json")))
    rows, pool, status, rem_uid = [], [], Counter(), {}
    g3d_c2_groups, g3d_rows = set(), []
    with gzip.open(os.path.join(run, "detection_record.csv.gz"), "rt") as f:
        rd = csv.DictReader(f)
        miss = [k for k in TS.NEED + ("assoc0_cluster", "gate_3d") if k not in rd.fieldnames]
        if miss:
            raise RejudgeRefusal("%s/detection_record.csv.gz lacks %s" % (run, ",".join(miss)))
        for r in rd:
            status[r["run_status"]] += 1
            why = removed(r)
            if why:
                rem_uid[r["det_uid"]] = why
                if "card2" in why and r["assoc0_cluster"] != "":
                    g3d_c2_groups.add((r["cell"], r["assoc0_cluster"]))
            if r["first_removed_by"] == "bright-parent-3d" and r["assoc0_cluster"] != "":
                g3d_rows.append((r["cell"], r["assoc0_cluster"], r["det_uid"]))
            if r["in_assoc_pool"] == "1" and r["first_removed_by"] not in TS.POOL_OUT:
                pool.append(dict(st=r["stokes"], tier=r["tier"], band=r["band"], snr=float(r["snr"]), det_uid=r["det_uid"]))
            if r["src_index"] != "" and r["src_outcome"] in TS.OUTCOMES:
                rows.append({k: r[k] for k in TS.NEED})
    if set(status) != {"complete"}:
        raise RejudgeRefusal("%s record run_status is not 'complete' on every row: %s" % (run, dict(status)))
    # the 3-D gate's removals in a first-association group that also holds a Card 2 detection: with Card 2 ahead of the gate (the
    # chain's order from clv15 1279bd4) the gate would have judged those groups without it - recorded, not re-admitted
    g3d_nb = sum(1 for c, g, u in g3d_rows if (c, g) in g3d_c2_groups and u not in rem_uid)
    sg = summ.get("seeded_growth", {})

    class A(object):
        singleton_floor = float(summ.get("singleton_floor", C.SINGLETON_FLOOR))
        singleton_floor_v = float(summ.get("singleton_floor_v", C.SINGLETON_FLOOR_V))
        repeat_floor = float(summ.get("repeat_floor", C.REPEAT_FLOOR))
        assoc_floor = float(summ.get("assoc_floor", C.ASSOC_FLOOR))
        store_floor = float(sg.get("store_floor", summ.get("store_floor", C.STORE_FLOOR)))
        seed_floor = float(sg.get("seed_floor", C.SEED_FLOOR))
    args = A()
    seed_on = args.seed_floor > args.store_floor
    cells = summ["cells"] if isinstance(summ["cells"], list) else list(summ["cells"])
    C.CELLS_GLOBAL.clear()
    C.CELLS_GLOBAL.update({c: C.cell_centre(c) for c in cells})
    cats = C.load_samples(samples_dir)
    area = C.footprint_area_deg2(C.CELLS_GLOBAL)
    if summ.get("footprint_deg2") is not None and abs(round(area, 1) - float(summ["footprint_deg2"])) > 0.05:
        raise RejudgeRefusal("%s footprint %.1f deg2 differs from the chain's %s" % (run, area, summ["footprint_deg2"]))
    if summ.get("n_pool_after_gates") is not None and len(pool) != int(summ["n_pool_after_gates"]):
        raise RejudgeRefusal("%s pool rebuilt from the record holds %d rows, the chain's n_pool_after_gates %s"
                             % (run, len(pool), summ["n_pool_after_gates"]))
    src = OrderedDict()
    for r in rows:
        o = r["src_outcome"]
        kept = (r["served"] == "1") if o == "served" else (r["first_removed_by"] == o)
        e = src.setdefault(r["src_index"], dict(outcome=o, kept=[], stripped=[], row0=r, jn=""))
        if kept:
            e["kept"].append(TS.member(r))
            if r["identity_jname"]:
                e["jn"] = r["identity_jname"]
        elif r["first_removed_by"] == "refl-floor":
            e["stripped"].append(TS.member(r))
    for k in list(src):
        if not src[k]["kept"]:
            del src[k]
            continue
        src[k]["chain_singleton"] = "singleton" in src[k]["row0"]["routes"].split(";")

    def fix_best(ns, e):
        cb = e["row0"]["src_best_det_uid"]
        if ns["best"]["det_uid"] != cb:
            for m in ns["mem"]:
                if m["det_uid"] == cb and abs(m["snr"]) == abs(ns["best"]["snr"]):
                    ns["best"] = m
                    break
        return ns

    def prepare(mem, full, e):      # phase3_snapshot_tierserve.evaluate's prepare, verbatim in effect
        best = max(mem, key=lambda m: abs(m["snr"]))
        if not full and seed_on and not (len(mem) == 1 and abs(mem[0]["snr"]) < args.assoc_floor) \
                and abs(best["snr"]) < args.seed_floor:
            return dict(qualifies=0, reason="seed", routes=[], best=best, ns=None)
        ns = fix_best(C.route_repeats([mem], args.repeat_floor)[0], e)
        kept, _ = C.reflection_band_source_gate([ns], args.repeat_floor)
        if not kept:
            return dict(qualifies=0, reason="refl-floor", routes=[], best=ns["best"], ns=None, ns0=ns)
        fix_best(kept[0], e)
        return dict(ns=kept[0], best=kept[0]["best"])

    def finish(res, e):             # phase3_snapshot_tierserve.evaluate's finish, verbatim in effect
        if res["ns"] is None:
            return res
        ns = res["ns"]
        b = ns["best"]
        routes = []
        if ns["multi_integration"]:
            routes.append("multi-integration")
        routes += list(ns.get("catalog", []))
        fl = C.singleton_floor_for(b, args)
        single_ok = abs(b["snr"]) >= fl
        if abs(abs(b["snr"]) - fl) < TS.SNR_ROUND:
            if b["det_uid"] == e["row0"]["src_best_det_uid"] and b["verdict"] == "unresolved-constrained":
                single_ok = e["chain_singleton"]
            else:
                res["rounding"] = 1
        if single_ok and b["verdict"] == "unresolved-constrained":
            routes.append("singleton")
        reason = "" if routes else ("fit-pending" if single_ok and b["verdict"] == "fit-pending" else
                                    ("not-designated:" + b["verdict"] if single_ok else "no-route"))
        res.update(qualifies=int(bool(routes)), reason=reason, routes=routes, best=b)
        return res

    touched = {k for k, e in src.items() if e["outcome"] == "served"
               and any(m["det_uid"] in rem_uid for m in e["kept"] + e["stripped"])}
    full_res = {k: prepare(list(e["kept"]), True, e) for k, e in src.items() if e["outcome"] == "served"}
    t0, t1 = {}, {}
    for k, e in src.items():
        if e["outcome"] != "served":
            continue
        allm = e["kept"] + e["stripped"]
        for t in TS.TIERS:
            mt = [m for m in allm if m["tier"] == t]
            if mt:
                t0[(k, t)] = prepare(mt, False, e)
            if k in touched:
                mr = [m for m in mt if m["det_uid"] not in rem_uid]
                t1[(k, t)] = prepare(mr, False, e) if mr else None
    C.catalog_route([r_["ns"] for r_ in full_res.values() if r_["ns"] is not None] +
                    [r_["ns"] for r_ in t0.values() if r_["ns"] is not None], pool, cats, area)
    C.catalog_route([r_["ns"] for r_ in t1.values() if r_ and r_["ns"] is not None],
                    [p for p in pool if p["det_uid"] not in rem_uid], cats, area)
    bad = []
    for k, e in src.items():
        if e["outcome"] != "served":
            continue
        full = finish(full_res[k], e)
        r0 = e["row0"]
        want = (r0["src_mi_tiers"], r0["catalog_promoted"], r0["routes"], r0["src_best_det_uid"])
        got = ((full["ns"] or full.get("ns0") or {}).get("mi_tiers"), ";".join((full["ns"] or {}).get("catalog", [])),
               ";".join(full["routes"]), full["best"]["det_uid"])
        if want != got:
            bad.append((k, e["jn"]))
    if bad:
        raise RejudgeRefusal("%s the reconstruction does not reproduce the chain on %d served sources (e.g. %s)"
                             % (run, len(bad), bad[:3]))
    st_old = {}
    for r in csv.DictReader(open(os.path.join(run, "served_tiers.csv"))):
        if r["chain_outcome"] == "served":
            st_old[(r["identity_jname"], r["tier"])] = r["qualifies"]
    mism = [(e["jn"], t) for (k, t), r_ in t0.items() for e in (src[k],) if str(finish(r_, e)["qualifies"]) != st_old.get((e["jn"], t))]
    if mism:
        raise RejudgeRefusal("%s the per-tier evaluation does not reproduce served_tiers.csv on %d (source, tier) (e.g. %s)"
                             % (run, len(mism), mism[:3]))
    out, n = OrderedDict(), Counter()
    for (k, t), r1 in t1.items():
        e = src[k]
        if (k, t) not in t0:
            continue
        q0 = t0[(k, t)]["qualifies"]
        mt = [m for m in e["kept"] + e["stripped"] if m["tier"] == t]
        nrem = Counter(rem_uid[m["det_uid"]] for m in mt if m["det_uid"] in rem_uid)
        if not nrem:
            continue
        if r1 is None:
            q1, why, rt = 0, "every member removed", ""
        else:
            r1 = finish(r1, e)
            q1, why, rt = r1["qualifies"], r1["reason"], ";".join(r1.get("routes", []))
        key = (e["jn"], t)
        if key in out:
            raise RejudgeRefusal("%s identity %s tier %s is two served sources" % (run, e["jn"], t))
        out[key] = OrderedDict(identity_jname=e["jn"], tier=t, src_index=k, qualifies_before=q0, qualifies_after=q1,
                               reason_after=why, routes_after=rt, n_members=len(mt), n_removed=sum(nrem.values()),
                               removed_by=";".join("%s %d" % kv for kv in sorted(nrem.items())))
        if q0 == 1:
            n["served (source, tier) touched"] += 1
            n["re-judged out" if (q1 == 0 and r1 is not None) else ("every member removed" if q1 == 0 else "still qualifies")] += 1
    n["served sources"] = sum(1 for e in src.values() if e["outcome"] == "served")
    n["3-D gate removals sharing a group with a Card 2 detection"] = g3d_nb
    log("re-judging %s: self-checks PASS (%d served sources reproduced; served_tiers.csv reproduced); %s"
        % (os.path.basename(run), n["served sources"], dict(n)))
    return out, n, rem_uid, g3d_nb


def removed_fn(DO, dec_rows, recs, c2_uids):
    """removed(record_row) of snapshot_rejudge.run_inputs: 'omit' (the deep omission per image), 'card2', 'omit;card2' or ''.
    The omission is deep_omission.Omission (the decisions per image, its member tables under the fleet root); its count of
    images absent from a member table is returned with it."""
    om = DO.Omission(dec_rows, recs)

    def removed(r):
        lst, date = r["cell"].split("_", 1)
        k = (lst, r["subband"], date, r["stokes"], r["tier"])
        o = om.key(k, r["ts"])
        c2 = r["det_uid"] in c2_uids
        return "omit;card2" if (o and c2) else ("omit" if o else ("card2" if c2 else ""))
    return removed, om


def child_main(job_path):
    """The child (module docstring): writes the result JSON; REFUSING + exit 2 on a gate."""
    import gc
    with open(job_path) as f:
        job = json.load(f)
    gc.disable()        # as production's worker: a full collection would walk every object of a large record (no fork here)
    for p in (job["clv15"],):
        if p not in sys.path:
            sys.path.insert(0, p)
    import importlib.util
    spec = importlib.util.spec_from_file_location("exopipe3_deep_omission_rejudge", job["deep_omission_py"])
    DO = importlib.util.module_from_spec(spec)
    sys.modules["exopipe3_deep_omission_rejudge"] = DO
    spec.loader.exec_module(DO)
    with open(job["decisions"], newline="") as f:
        dec = list(csv.DictReader(f))
    recs = DO.SnapshotRecords(job["proto"], DO.Reader(workers=1), names=job["names"])
    c2 = set(job["card2"])
    try:
        removed, om = removed_fn(DO, dec, recs, c2)
        out, n, rem_uid, g3d = rejudge(job["run"], removed, job["samples_dir"], log=lambda *a: None)
    except (RejudgeRefusal, DO.OmissionRefusal) as e:
        print("REFUSING: re-judging %s: %s" % (os.path.basename(job["run"]), e), flush=True)
        return 2
    res = dict(run=job["run"], rejudge=[[J, T, list(row.items())] for (J, T), row in out.items()],
               counts=dict(n), n_removed=list(Counter(rem_uid.values()).items()), g3d_neighbours=g3d,
               absent_images=[[list(k), v] for k, v in sorted(om.absent.items())])
    tmp = job["result"] + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, job["result"])
    print("REJUDGE-OK %s: %d served sources reproduced, %d (source, tier) touched" % (
        os.path.basename(job["run"]), n["served sources"], len(out)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(child_main(sys.argv[1]))
