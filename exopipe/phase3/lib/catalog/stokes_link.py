"""The cross-Stokes link: one name for a source seen in Stokes I and in Stokes V of the same tier.

RULINGS
  R-REFACTOR-20261104-CROSS-STOKES-LINK-COLUMN: a link layer in its own column; identity_id stays per Stokes; the source pages,
  the transient list, the cross-match lists and the localization merge on the link; the interim localization is per (Stokes,
  identity). It implements R-CROSS-TIER-SINGLE-IDENTITY ("a single name per source", named by the member with the smallest
  position error) for I and V.
  R-GREGG-20261124-CROSS-STOKES-LINK-TEST-APPROVED (Gregg 2026-11-24, verbatim: "This is for linking across tiers, e.g. at the
  top of each source page? Either way, approved."; the link is across Stokes within a tier): an I identity and a V identity of
  the same tier are linked when design B's identity-level test passes on the two identities' mean positions and covariances
  (association probability P >= 0.97, 4-sigma cap), with the other Stokes' identity density as the background; one-to-many
  links form one connected component, named by its best-localized member and marked ambiguous.

THE TEST (design B's arithmetic; the positions and the association in core/astrometry.py identity_means and
identity_association; every constant from config catalog_grouping, read through grouping.Params - one source: P_IDENTITY and
U_CAP for the decision, and a Params override (tests, diagnostics) changes both)
  identity moments  per identity: the inverse-covariance mean of its detections (grouping.mean_position's arithmetic, in the
                    brightest member's tangent plane) and the covariance of that mean, (sum C^-1)^-1 (arcsec^2; C = the
                    detection's covariance, grouping.covariance: elevation term and Master-frame term included)
  the pair          d = the V mean's (east, north) offset from the I mean; S = Cov_I + Cov_V + STAGE_B_FLOOR_ARCSEC^2 (the stage-B
                    error floor, once, as grouping evaluate adds it to a unit against a component); u = sqrt(d^T S^-1 d)
  likelihood        f = the window-source offset profile PROFILE_WS at u as an areal density (grouping._areal's arithmetic)
  background        rho = the other Stokes' identity means per arcsec^2 in design B's background probes (PROBES_PX along the
                    frame x axis of the identity's brightest detection, discs of PROBE_RADIUS_PX['other_lst'], median, floor
                    PROBE_FLOOR_COUNT per disc: grouping._probe_lists / _category_density's rho_OL); a link is symmetric, so
                    the pair takes the larger of rho(V identities around the I identity) and rho(I identities around the V
                    identity)
  decision          odds = f / rho (prior weight 1: one counterpart); P = odds / (1 + odds); linked when u <= U_CAP and
                    P >= P_IDENTITY (the identity level's operating point, grouping.operating_point)
  components        linked pairs form connected components (union-find); a component of two is a link, a component of three or
                    more is AMBIGUOUS (one-to-many); the component is named by its best-localized member (smallest
                    sqrt(det Cov), Stokes I first on a tie, then ID order): link_id = that member's stable identity ID,
                    link_name = its display name
  recorded          per pair: offset (arcsec), u, P, the offset percentile (core/astrometry.offset_pct_profile, PROFILE_WS)
                    and the chance-coincidence probability (core/astrometry.p_chance_ellipse with the pair's rho and S)

PRODUCTS (stokes_link/ beside grouping_I/ and grouping_V/: a symbolic link to the current version in stokes_link.versions/,
switched atomically - lib/catalog/stokes_link_tables.py, which holds the names, the columns and the one reader of each table)
  stokes_links.csv       LINK_COLS: one row per identity of a component of two or more (an identity absent from the table is
                         its own source: link_id = its group_identity_id)
  stokes_link_pairs.csv  PAIR_COLS: one row per linked (I, V) pair
  identity_merges.csv    MERGE_COLS: every identity ID of either Stokes that a grouping run merged into another, with the
                         ID its 'into' chain ends at (the state saved with each grouping table, grouping_<S>/state_next) and that
                         ID's status: 'live', or 'retired' (valid history: the survivor left the grouping later). A label made on
                         a page shown before the merge follows it to a live survivor's page (R-GREGG-20261124-CROSS-STOKES-LINK-
                         TEST-APPROVED, Q2: "carry labels along the grouping's merge events"; lib/serve/label_identity.py); one
                         whose survivor retired is refused (HTTP 409). A chain that cycles or ends at an ID the state never
                         issued REFUSES the link.
  summary.json           counts and the inputs (the two grouping tables and their sha1)
"""
import collections
import csv
import datetime
import hashlib
import json
import math
import os
import shutil

import numpy as np
from scipy.spatial import cKDTree

from exopipe.phase3.core import astrometry as A
from exopipe.phase3.lib.catalog import grouping as G
from exopipe.phase3.lib.catalog.stokes_link_tables import (  # noqa: F401  (the one set of names and readers, re-exported)
    DIRNAME, FORMAT, INTO_STATUS, KEEP_SUPERSEDED, LINK_COLS, LINKS, LIST_COLS, LOC_ENV, MERGE_COLS, MERGES, PAIR_COLS, PAIRS,
    SUMMARY, VERSIONS, TableRefused, dir_of, link_dir, link_key, link_test_values, members, partner_names, read_links,
    read_merge_rows, read_merges, read_pairs, read_rows, read_summary)


class LinkRefused(TableRefused):
    pass


def grouped_stokes(out):
    """the Stokes with a grouping table under out (grouping_<S>/detections.csv): a link exists only when both do - with one
    Stokes grouped there is nothing to link, and its names are its own (grouping.display_names reserves the other's)"""
    from exopipe.phase3.lib.catalog import grouping_stage as GS
    return {S for S in ('I', 'V') if os.path.exists(os.path.join(GS.group_dir(out, S), GS.DETECTIONS))}


# =====================================================================================================================
# identity moments
# =====================================================================================================================

def identity_moments(det, identity_id):
    """per identity of one Stokes: dict(ids, ra, dec, cov (n,2,2 arcsec^2), lst, band (of the brightest member), n_det, snr).
    det: grouping det (ra, dec in the grouping's frame, C, snr, lst, band); identity_id: per detection (same order)."""
    ident = np.asarray([str(x) for x in identity_id])
    if len(ident) != len(det['ra']):
        raise LinkRefused(f'stokes link: {len(ident)} identity labels for {len(det["ra"])} detections - REFUSING')
    m = A.identity_means(det['ra'], det['dec'], det['C'], det['snr'], ident)
    anchor = m.pop('anchor')
    return dict(m, lst=np.asarray(det['lst'])[anchor], band=np.asarray(det['band'])[anchor],
                snr=np.abs(np.asarray(det['snr'], float))[anchor])


def background(mom, other, P):
    """areal density (per arcsec^2) of the OTHER Stokes' identity means around each identity of mom: design B's probes
    (grouping._probe_lists: PROBES_PX along the frame x axis of the identity's brightest detection), discs of
    PROBE_RADIUS_PX['other_lst'], median over the probes, floor PROBE_FLOOR_COUNT per disc (grouping._category_density rho_OL)"""
    n = len(mom['ids'])
    t = cKDTree(G._unit(other['ra'], other['dec']))
    lists, rdeg = G._probe_lists(mom['ra'], mom['dec'], mom['lst'], mom['band'], P.probe_r['other_lst'], t, P)
    cnt = np.full((len(P.probes), n), np.nan)
    for k, (idx, ii, _jj) in enumerate(lists):
        cnt[k, idx] = np.bincount(ii, minlength=n)[idx]
    with np.errstate(all='ignore'):
        c = np.nanmedian(cnt, 0)
    c = np.maximum(np.where(np.isfinite(c), c, P.probe_floor), P.probe_floor)
    return c / (math.pi * rdeg ** 2) / G.DEG2


# =====================================================================================================================
# the test and the components
# =====================================================================================================================

def pair_test(mI, mV, tier, P=None):
    """every (I, V) identity pair within STAGE_B_RADIUS_DEG (candidates; performance only): arrays i, v, offset_arcsec, u, p,
    rho_I, rho_V (per arcsec^2), detS (arcsec^4) and the decision `linked` (core/astrometry.identity_association: u <= U_CAP
    and P >= P_IDENTITY, both from P - config catalog_grouping unless the caller's Params overrides them)"""
    P = P or G.Params()
    try:
        G.operating_point(tier, 'I')                      # a tier without an operating point REFUSES
    except G.GroupingRefused as e:
        raise LinkRefused(f'stokes link: {e}') from None
    p_thr, u_cap = float(P.p_thr), float(P.u_cap)
    rho_V_at_I = background(mI, mV, P)                    # V identities around each I identity
    rho_I_at_V = background(mV, mI, P)
    tV = cKDTree(G._unit(mV['ra'], mV['dec']))
    near = tV.query_ball_point(G._unit(mI['ra'], mI['dec']), G._chord(P.rnb))
    ln = np.fromiter((len(r) for r in near), int, len(near))
    i = np.repeat(np.arange(len(near)), ln); v = np.fromiter((j for r in near for j in r), int, ln.sum())
    rho = np.maximum(rho_V_at_I[i], rho_I_at_V[v])
    a = A.identity_association(mI['ra'][i], mI['dec'][i], mI['cov'][i], mV['ra'][v], mV['dec'][v], mV['cov'][v], rho,
                               P.prof_ws, P.floor, p_thr, u_cap)
    return dict(i=i, v=v, offset_arcsec=a['offset_arcsec'], u=a['u'], p=a['p'], rho_I=rho_I_at_V[v], rho_V=rho_V_at_I[i],
                rho=rho, detS=a['detS'], linked=a['associated'], p_thr=p_thr, u_cap=u_cap)


def components(nI, nV, i, v):
    """connected components of the bipartite graph I x V given its edges (i, v): an array of component labels over the
    nI + nV nodes (I first), -1 for a node without an edge"""
    parent = np.arange(nI + nV)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in zip(i.tolist(), (nI + v).tolist()):
        ra_, rb = find(a), find(b)
        if ra_ != rb:
            parent[max(ra_, rb)] = min(ra_, rb)
    deg = np.bincount(np.r_[i, nI + v], minlength=nI + nV)
    lab = np.array([find(x) for x in range(nI + nV)])
    lab[deg == 0] = -1
    return lab


def pos_err(cov):
    """the geometric-mean 1-sigma position error (arcsec) of covariances: det(cov)^(1/4)"""
    return np.sqrt(np.sqrt(np.maximum(cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] ** 2, 0.0)))


def link(mI, mV, namesI, namesV, tier, P=None):
    """(link rows, pair rows, counts) of the two Stokes' identity moments; names*: {identity_id: display name}"""
    P = P or G.Params()
    t = pair_test(mI, mV, tier, P)
    sel = np.nonzero(t['linked'])[0]
    nI, nV = len(mI['ids']), len(mV['ids'])
    lab = components(nI, nV, t['i'][sel], t['v'][sel])
    errI, errV = pos_err(mI['cov']), pos_err(mV['cov'])
    node_id = np.r_[mI['ids'], mV['ids']]
    node_stokes = np.r_[np.full(nI, 'I'), np.full(nV, 'V')]
    node_err = np.r_[errI, errV]
    node_name = [namesI[x] for x in mI['ids']] + [namesV[x] for x in mV['ids']]
    members = collections.defaultdict(list)
    for x in np.nonzero(lab >= 0)[0]:
        members[int(lab[x])].append(int(x))
    head = {}
    for c, xs in members.items():
        head[c] = min(xs, key=lambda x: (node_err[x], node_stokes[x] != 'I', G._seq_of(node_id[x]), str(node_id[x])))
    # per pair: the astrometry-module quantities
    pairs = []
    best = {}
    for k in sel.tolist():
        i, v = int(t['i'][k]), int(t['v'][k])
        c = int(lab[i]); h = head[c]
        rec = dict(link_id=str(node_id[h]), link_ambiguous=int(len(members[c]) > 2),
                   I_identity_id=str(mI['ids'][i]), I_identity_name=namesI[mI['ids'][i]],
                   V_identity_id=str(mV['ids'][v]), V_identity_name=namesV[mV['ids'][v]],
                   offset_arcsec=round(float(t['offset_arcsec'][k]), 2), u=round(float(t['u'][k]), 4), p=float(t['p'][k]),
                   offset_pct=round(A.offset_pct_profile(t['u'][k], P.prof_ws), 3),
                   p_chance=float(A.p_chance_ellipse(t['rho'][k], t['u'][k], t['detS'][k])),
                   rho_I_per_deg2=round(float(t['rho_I'][k]) * G.DEG2, 4), rho_V_per_deg2=round(float(t['rho_V'][k]) * G.DEG2, 4),
                   I_pos_err_arcsec=round(float(errI[i]), 2), V_pos_err_arcsec=round(float(errV[v]), 2))
        pairs.append(rec)
        for x, y in ((i, nI + v), (nI + v, i)):
            best.setdefault(x, []).append((rec['p'], -rec['u'], y, rec))
    rows = []
    for c in sorted(members, key=lambda c: (G._seq_of(node_id[head[c]]), str(node_id[head[c]]))):
        h = head[c]
        for x in sorted(members[c], key=lambda x: (node_stokes[x], G._seq_of(node_id[x]))):
            nb = sorted(best[x], key=lambda z: (-z[0], -z[1]))
            r0 = nb[0][3]
            rows.append(dict(link_id=str(node_id[h]), link_name=node_name[h], link_size=len(members[c]),
                             link_ambiguous=int(len(members[c]) > 2), stokes=str(node_stokes[x]), identity_id=str(node_id[x]),
                             identity_name=node_name[x], linked_ids=';'.join(str(node_id[z[2]]) for z in nb),
                             linked_names=';'.join(node_name[z[2]] for z in nb), best_p=r0['p'], best_u=r0['u'],
                             best_offset_arcsec=r0['offset_arcsec'], best_offset_pct=r0['offset_pct'],
                             best_p_chance=r0['p_chance'],
                             mean_ra=round(float((mI if x < nI else mV)['ra'][x if x < nI else x - nI]), 6),
                             mean_dec=round(float((mI if x < nI else mV)['dec'][x if x < nI else x - nI]), 6),
                             pos_err_arcsec=round(float(node_err[x]), 2),
                             n_det=int((mI if x < nI else mV)['n_det'][x if x < nI else x - nI])))
    sizes = collections.Counter(len(m) for m in members.values())
    counts = dict(identities_I=nI, identities_V=nV, candidate_pairs=int(len(t['i'])),
                  pairs_within_cap=int((t['u'] <= t['u_cap']).sum()), linked_pairs=len(pairs), components=len(members),
                  components_of_two=int(sizes.get(2, 0)), ambiguous_components=int(sum(v for k, v in sizes.items() if k > 2)),
                  component_sizes={str(k): int(v) for k, v in sorted(sizes.items())},
                  linked_identities_I=int(sum(1 for x in np.nonzero(lab >= 0)[0] if x < nI)),
                  linked_identities_V=int(sum(1 for x in np.nonzero(lab >= 0)[0] if x >= nI)),
                  p_threshold=t['p_thr'], u_cap=t['u_cap'], floor_arcsec=P.floor)
    return rows, pairs, counts


# =====================================================================================================================
# the stage: both Stokes' grouping tables -> stokes_link/
# =====================================================================================================================

def _sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def _write_csv(path, cols, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cols))
        w.writeheader()
        w.writerows(rows)
    return path


def labels_of_table(path, det):
    """(identity_id per detection of det, {identity_id: display name}) from a grouping detections.csv; a detection of det the
    table lacks, or a table row det lacks, REFUSES (the link must use the very table the builds read)"""
    by = {}
    names = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            by[r['cand_id']] = r['identity_id']
            names[r['identity_id']] = r['identity_name']
    cid = [str(x) for x in det['cand_id']]
    miss = [c for c in cid if c not in by]
    if miss or len(by) != len(cid):
        raise LinkRefused(f'stokes link: grouping table {path} holds {len(by)} detections, the input {len(cid)} '
                          f'({len(miss)} of the input absent, e.g. {miss[:3]}) - REFUSING')
    return [by[c] for c in cid], names


SIDE_INPUT = {'10min': 'pertie', 'deep': 'frame_beam'}       # the grouping summary's inputs key of load_det's second file


def dets_of_tables(out, tier, have=None):
    """{S: det} of both Stokes' present groupings under out: have[S] (the det a caller just grouped), else loaded from the
    inputs grouping_<S>/summary.json records (grouping_stage.load_det); a missing grouping, summary or input REFUSES (and
    run_link refuses a det that is not the table's: labels_of_table)"""
    from exopipe.phase3.lib.catalog import grouping_stage as GS
    dets = dict(have or {})
    for S in ('I', 'V'):
        if S in dets:
            continue
        sp = os.path.join(GS.group_dir(out, S), GS.SUMMARY)
        if not os.path.exists(sp):
            raise LinkRefused(f'stokes link: {sp} is missing (the link needs both Stokes\' groupings) - REFUSING')
        with open(sp) as f:
            inp = json.load(f).get('inputs') or {}
        mc, side = inp.get('master_catalog'), inp.get(SIDE_INPUT[tier])
        try:
            dets[S] = GS.load_det(tier, S, mc, side)
        except GS.StageRefusal as e:
            raise LinkRefused(f'stokes link: {S}: {e}') from None
    return dets


def check_names(out):
    """[] when no display name is held by an identity of each Stokes (grouping_stage.run_group reserves the other Stokes'
    names), else the shared names: a name names one (Stokes, identity), so the readers that key by name across both Stokes
    (the localization, the figures) never merge an I and a V identity by a name coincidence"""
    from exopipe.phase3.lib.catalog import grouping_stage as GS
    a, b = GS.present_names(out, 'I'), GS.present_names(out, 'V')
    return sorted(a & b)


def merges_of(out, tier):
    """[MERGE_COLS rows] of the grouping under out: every identity-level ID the state saved with grouping_<S>/ (state_next)
    holds as merged, with the ID its 'into' chain ends at (GroupingState.resolve) and that ID's status - 'live', or 'retired'
    (valid history: the survivor itself left the grouping in a later run). A missing state, a chain that cycles, or one that
    ends at an ID the state never issued (or at a non-identity ID) REFUSES - the merge history is broken, and a label carried
    along it could land on the wrong identity."""
    from exopipe.phase3.lib.catalog import grouping_stage as GS
    rows = []
    for S in ('I', 'V'):
        sp = GS.state_path(os.path.join(GS.group_dir(out, S), 'state_next'), tier, S)
        if not os.path.exists(sp):
            raise LinkRefused(f'stokes link: {sp} is missing (the state saved with the grouping table) - REFUSING')
        with open(sp) as f:
            doc = json.load(f)
        st = G.GroupingState(tier, S)
        st.ids = doc['ids']
        for i, r in sorted(doc['ids'].items()):
            if r.get('level') == 'identity' and r.get('status') == 'merged':
                try:
                    end = st.resolve(i)
                except G.GroupingRefused as e:
                    raise LinkRefused(f'stokes link: the merge history of {S} {i} in {sp} is broken ({e}) - REFUSING') from None
                e_ = st.ids.get(end)
                if e_ is None or e_.get('level') != 'identity' or e_.get('status') not in INTO_STATUS:
                    raise LinkRefused(f'stokes link: the merge chain of {S} {i} in {sp} ends at {end!r}, '
                                      + ('an ID the state never issued' if e_ is None else
                                         f'a {e_.get("level")} ID of status {e_.get("status")!r}')
                                      + ' - the merge history is broken - REFUSING')
                rows.append(dict(stokes=S, identity_id=i, into_id=end, into_status=e_['status'], run=r.get('last_run', '')))
    return rows


def run_link(det_by_stokes, out, tier, P=None):
    """Link the I and V identities of the grouping under `out` (grouping_I/, grouping_V/ written by run_group from these det)
    and write stokes_link/: a new version (new_version), then one atomic switch of the symbolic link (switch_to; the
    KEEP_SUPERSEDED versions before it kept, older ones deleted). Returns (paths, counts). A missing Stokes REFUSES: the link
    needs both. P: grouping.Params (default: config catalog_grouping - P_IDENTITY and U_CAP decide, and the summary records
    the values used)."""
    from exopipe.phase3.lib.catalog import grouping_stage as GS
    if set(det_by_stokes) != {'I', 'V'}:
        raise LinkRefused(f'stokes link: needs the I and V groupings, got {sorted(det_by_stokes)} - REFUSING')
    P = P or G.Params()
    shared = check_names(out)
    if shared:
        raise LinkRefused(f'stokes link: {len(shared)} display name(s) held in both Stokes, e.g. {shared[:3]} (the grouping '
                          f'reserves the other Stokes\' names; a name must name one (Stokes, identity)) - REFUSING')
    mom, names, tables = {}, {}, {}
    for S, det in det_by_stokes.items():
        tab = os.path.join(GS.group_dir(out, S), GS.DETECTIONS)
        if not os.path.exists(tab):
            raise LinkRefused(f'stokes link: {tab} is missing - REFUSING')
        ids, names[S] = labels_of_table(tab, det)
        mom[S] = identity_moments(det, ids)
        tables[S] = dict(path=tab, sha1=_sha1(tab))
    rows, pairs, counts = link(mom['I'], mom['V'], names['I'], names['V'], tier, P)
    merges = merges_of(out, tier)
    counts['merged_ids'] = {S: sum(1 for m in merges if m['stokes'] == S) for S in ('I', 'V')}
    counts['merged_ids_retired'] = {S: sum(1 for m in merges if m['stokes'] == S and m['into_status'] == 'retired')
                                    for S in ('I', 'V')}
    v = new_version(out)
    _write_csv(os.path.join(v, LINKS), LINK_COLS, rows)
    _write_csv(os.path.join(v, PAIRS), PAIR_COLS, pairs)
    _write_csv(os.path.join(v, MERGES), MERGE_COLS, merges)
    summ = dict(format=FORMAT, tier=tier, tables=tables, counts=counts, utc=_utc())
    with open(os.path.join(v, SUMMARY), 'w') as f:
        json.dump(summ, f, indent=1, sort_keys=True)
    final = switch_to(out, v)
    return {k: os.path.join(final, v_) for k, v_ in (('links', LINKS), ('pairs', PAIRS), ('summary', SUMMARY))}, counts


# =====================================================================================================================
# versions: stokes_link is a symbolic link to the current version, switched atomically
# =====================================================================================================================

def _utc():
    return datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def new_version(parent):
    """a new, empty version directory parent/stokes_link.versions/<UTC to the microsecond>_<pid>/ (unique: a name taken
    REFUSES rather than reuse it); readers never see it until switch_to points stokes_link at it"""
    vid = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S.%fZ') + f'_{os.getpid()}'
    v = os.path.join(parent, VERSIONS, vid)
    try:
        os.makedirs(os.path.join(parent, VERSIONS), exist_ok=True)
        os.mkdir(v)
    except OSError as e:
        raise LinkRefused(f'stokes link: cannot create the version directory {v} ({e}) - REFUSING') from None
    return v


def _versions(parent):
    d = os.path.join(parent, VERSIONS)
    return sorted(n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n))) if os.path.isdir(d) else []


def switch_to(parent, v):
    """Point parent/stokes_link at version v (a directory in parent/stokes_link.versions/) in ONE rename: a new relative
    symbolic link under a temporary name, os.replace over stokes_link - a reader that opens stokes_link/<file> gets the old
    version or the new one, never a moment without a link table. A stokes_link that is still a plain directory (written before
    this layout) is first moved into the versions directory (<its modification time>_adopted), the one moment it is
    absent. Then the versions older than v beyond the KEEP_SUPERSEDED newest are deleted (intermediates; a deletion that
    fails leaves the directory for the next switch). A failed rename, link or move REFUSES (LinkRefused). Returns
    parent/stokes_link."""
    final = link_dir(parent)
    rel = os.path.join(VERSIONS, os.path.basename(v))
    if os.path.realpath(os.path.dirname(v)) != os.path.realpath(os.path.join(parent, VERSIONS)) or not os.path.isdir(v):
        raise LinkRefused(f'stokes link: {v} is not a version directory of {parent} - REFUSING')
    tmp = f'{final}.switch.{os.getpid()}'
    try:
        if os.path.isdir(final) and not os.path.islink(final):
            t = datetime.datetime.utcfromtimestamp(os.stat(final).st_mtime).strftime('%Y%m%dT%H%M%S.%fZ')
            os.rename(final, os.path.join(parent, VERSIONS, f'{t}_adopted'))
        if os.path.lexists(tmp):
            os.remove(tmp)
        os.symlink(rel, tmp)
        os.replace(tmp, final)
    except OSError as e:
        raise LinkRefused(f'stokes link: switching {final} to {rel} failed ({e}) - REFUSING') from None
    cur = os.path.basename(v)
    older = [n for n in _versions(parent) if n < cur]
    for n in older[:max(0, len(older) - KEEP_SUPERSEDED)]:
        shutil.rmtree(os.path.join(parent, VERSIONS, n), ignore_errors=True)
    return final


# =====================================================================================================================
# readers: the link of a (Stokes, identity)
# =====================================================================================================================

def current_links(out, tables=None):
    """the path of out/stokes_link/stokes_links.csv after checking that the link was computed from the grouping tables the
    reader stands on: tables {S: sha1} (default: the sha1 of the present grouping_<S>/detections.csv under out). A missing
    link, or one computed from other tables (a grouping run whose link step refused), REFUSES: the readers merge on the
    link, and a stale link would merge identities of another grouping."""
    _check_version(out, link_dir(out), tables)
    return os.path.join(link_dir(out), LINKS)


def _check_version(out, d, tables=None):
    """current_links' check on one directory d (the current link, or the version a publish copies)"""
    from exopipe.phase3.lib.catalog import grouping_stage as GS
    sp, lp = os.path.join(d, SUMMARY), os.path.join(d, LINKS)
    if not (os.path.exists(sp) and os.path.exists(lp)):
        raise LinkRefused(f'stokes link: {d} holds no link (run the grouping stage: it links the two Stokes after grouping) - '
                          f'REFUSING')
    summ = read_summary(d)
    if summ.get('format') != FORMAT:
        raise LinkRefused(f'stokes link: {d} is of format {summ.get("format")!r}, not {FORMAT!r} (re-run the grouping stage) - '
                          f'REFUSING')
    for S in ('I', 'V'):
        want = (tables or {}).get(S) or _sha1(os.path.join(GS.group_dir(out, S), GS.DETECTIONS))
        got = ((summ.get('tables') or {}).get(S) or {}).get('sha1')
        if got != want:
            raise LinkRefused(f'stokes link: {d} was computed from another {S} grouping table (sha1 {str(got)[:12]}, the '
                              f'reader stands on {want[:12]}) - REFUSING (re-run the grouping stage)')
    return lp


def publish(out, dest, tables=None):
    """copy the current link of the grouping under out (checked as current_links checks it) into a new version of
    dest/stokes_link and switch to it atomically (switch_to: a live server reading dest never sees a moment without a link
    table) - a served run's own copy, which its localization and its pages read; returns dest/stokes_link/stokes_links.csv"""
    src = os.path.realpath(link_dir(out))          # one version, fixed: a switch under out while copying changes nothing here
    _check_version(out, src, tables)
    v = new_version(dest)
    try:
        for n in sorted(os.listdir(src)):
            shutil.copy2(os.path.join(src, n), os.path.join(v, n))
    except OSError as e:
        raise LinkRefused(f'stokes link: copying {src} into {v} failed ({e}) - REFUSING') from None
    return os.path.join(switch_to(dest, v), LINKS)



def add_link_columns(table, out, links_path):
    """The link columns (LIST_COLS) appended to a table with columns `stokes` and `identity_id` (the display name; else
    source_id) - the transient list: the link id and name of the row's identity (its component's, or its own stable ID and
    name when it is linked to nothing), the ambiguous flag and the other Stokes' names it is linked to. The table's leading
    '#' comment lines (the transient list's header: run, per-tier counts, limits, columns) are kept, one line naming the link
    columns added after them. Rewritten in place (temporary file, replaced). A row whose (Stokes, name) is no identity of the
    grouping under out REFUSES. Returns (rows, rows linked)."""
    from exopipe.phase3.lib.catalog import grouping_stage as GS
    lrows = read_rows(links_path, LINK_COLS)
    links = {(r['stokes'], r['identity_name']): r for r in lrows}
    comps = members(lrows)                          # the components, indexed by link_id once
    ids = {}
    for S in ('I', 'V'):
        with open(os.path.join(GS.group_dir(out, S), GS.IDENTITIES), newline='') as f:
            ids.update({(S, r['identity_name']): r['identity_id'] for r in csv.DictReader(f)})
    with open(table, newline='') as f:
        text = f.read().splitlines(True)
    k = 0
    while k < len(text) and text[k].startswith('#'):
        k += 1
    comments = text[:k]
    rd = csv.DictReader(text[k:])
    head = list(rd.fieldnames or [])
    rows = list(rd)
    if any(c in head for c in LIST_COLS):
        raise LinkRefused(f'{table} already holds link columns - REFUSING')
    n, bad = 0, []
    for r in rows:
        key = (r.get('stokes', ''), (r.get('identity_id') or '').strip() or r.get('source_id', ''))
        lk = links.get(key)
        if lk:
            n += 1
            others = [x for x in lk['linked_names'].split(';') if x]
            comp = [m for m in comps[lk['link_id']] if m['stokes'] != key[0]]
            r.update(stokes_link_id=lk['link_id'], stokes_link_name=lk['link_name'], stokes_link_ambiguous=lk['link_ambiguous'],
                     stokes_linked_names=';'.join(sorted({m['identity_name'] for m in comp} | set(others))))
        elif key in ids:
            r.update(stokes_link_id=ids[key], stokes_link_name=key[1], stokes_link_ambiguous='0', stokes_linked_names='')
        else:
            bad.append(key)
    if bad:
        raise LinkRefused(f'{table}: {len(bad)} row(s) whose (Stokes, identity) is no identity of the grouping under {out}, '
                          f'e.g. {bad[:3]} - REFUSING')
    tmp = f'{table}.tmp.{os.getpid()}'
    with open(tmp, 'w', newline='') as f:
        if comments:
            f.write(''.join(comments))
            f.write('# Cross-Stokes link (lib/catalog/stokes_link.py): stokes_link_id / stokes_link_name = the I/V component the '
                    'row\'s identity belongs to (its best-localized member; its own stable ID when linked to nothing), '
                    'stokes_link_ambiguous = 1 for a one-to-many component, stokes_linked_names = the other Stokes\' '
                    'identities it is linked to.\n')
        w = csv.DictWriter(f, fieldnames=head + list(LIST_COLS))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, table)
    return len(rows), n
