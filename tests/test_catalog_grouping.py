#!/usr/bin/env python3
"""Source grouping design B (refactor item F; lib/catalog/grouping.py, grouping_inputs.py) on FROZEN inputs: the port must give
the approved research code's results exactly, and the approved properties must hold.

FROZEN INPUTS (read-only; SHA256SUMS checked first): /lustre/gh/main/phase3/refactor_analysis/refactor_wt_20261102/F_grouping/
frozen_inputs - every detection of LST hours 04h, 05h and 21h (the LPT J0409+4640 and the four CLEAN5 pulsars) of the inputs the
approved numbers were computed on: the frozen Run_SLSTB of 2026-10-25 21:21 UTC (the preview's input; 10-minute I 5,834 and V 3,449
detections) and Run_P3S5's catalog of 2026-10-14 (deep I 1,003, V 14,275), with the frame beams and the CLEAN5 positions.
GOLDEN OUTPUTS (frozen_inputs/golden, made by F_grouping/scripts/make_golden.py with the RESEARCH code, group_B.py round 4
md5 c0b652978ed7b7427946062858e4e8ad, read-only copy): the research loaders' detection fields; the research partitions at the
ruled operating points; the research partitions of every injected dataset of the metrics below; the research incremental run.
CHECKS, per tier x Stokes (10-minute I and V, deep I and V):
  loader       the module's det (common frame through the in-place astrom_frame, error model, elevation term) = the research
               loader's, field by field, exactly
  partitions   identity and multiples labels = the research labels, element by element (the same component indices)
  nesting      group() = group_fork()[0]; the multiples level splits no identity; deep V: the multiples level is off
  determinism  a second run gives the same arrays; ORDER INDEPENDENCE: a permuted input gives the same partitions and IDs
  CLEAN5, LPT  the LPT whole at both levels; CLEAN5 in one piece as the research (10-minute I, deep I)
  chance_join  harness.chance_join (v2b: 10% sparse copies shifted 120 px toward the meridian, 3 draws) - every injected run's
               labels = the research labels; the rates (both levels) = research; identity <= multiples (nesting), both small
  absorption   harness.rich_absorption around clusters of >= 50 members: labels = research; the absorptions beyond 4 sigma are
               consistent with the known limit (<= 1%; binomial at 95%: the subsets hold ~60 such injections)
  faint        harness_extra.faint_repeat_recovery (n = 2/3/5; partner modes): labels = research; recovery at the multiples level
               >= identity level (nesting)
  incremental  a first run on the early 75% of the nights, then the rest added: every group (both levels) without a new
               detection is exactly a group of the first run (2026-11-14; the research frozen run's partitions are printed, no
               longer checked); a second run without a new detection changes nothing at either level; no early
               identity split; IDs carried; EVERY changed assignment has its event (merge / split / move); state save/load;
               refusals (mixed windows, batch over a state); regroup mode (every change recorded; a split event for exactly
               the IDs now in several groups)
  reused       a cand_id that names another detection (production re-numbers on a re-search; the review's
               V1_04h_2025-01-02T01_18MHz_0017 moved 20 deg): the incremental mode refuses its mixed window; regroup never
               carries it by cand_id ('reused' event); a window re-measured whole is matched within tolerance (a new window with
               exact fingerprints only) (10-minute V)
  tolerance    a re-measured detection (0.6 x tolerance, S/N x 1.2; also re-numbered) keeps its IDs - with exact fingerprints
               only it would retire; 1.6 x tolerance, or S/N x 1.4, is not the same detection; two candidates within tolerance
               match none ('ambiguous', refused by the incremental mode); the partitions never change (10-minute V)
  IDs, names   IDs fixed at creation (format, deterministic, order independent), the record logic on synthetic cases (the
               review's [P,P,P,Q,Q,Q,Q,Q] case), the state file (one file; no state refuses unless new=True), names unique, the
               frame generation in the run summary
The metric functions below are ports of harness.py v2b / harness_extra.py (same RNG draws); make_golden.py checked them against the
research harness's own functions on these inputs (golden/<tier>_<S>.json 'harness_crosscheck').
Run: NUMPY_MADVISE_HUGEPAGE=0 taskset -c $(bash /lustre/gh/main/exopipe_ops/node_free_cores.sh) python3 tests/test_catalog_grouping.py
     [--quick: skip the injected-metric re-runs (chance, absorption, faint)] [--only 10min_I,deep_V]
"""
import collections
import hashlib
import json
import math
import os
import shutil
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from exopipe.phase3.lib.catalog import grouping as G           # noqa: E402
from exopipe.phase3.lib.catalog import grouping_inputs as GI   # noqa: E402

FROZEN = '/lustre/gh/main/phase3/refactor_analysis/refactor_wt_20261102/F_grouping/frozen_inputs'
GOLDEN = os.path.join(FROZEN, 'golden')
T = '/tmp/exopipe3_tests_grouping'
DATASETS = (('10min', 'I'), ('10min', 'V'), ('deep', 'I'), ('deep', 'V'))
LV = ('identity', 'multiples')
BINS = ('<2', '2-4', '4-6', '>6')
# what each dataset exercises (the golden builder computes exactly these)
PLAN = {('10min', 'I'): dict(truth=True, chance=True, rich=True, faint=('same_night', 'other_night', 'mixed'), incremental=True),
        ('10min', 'V'): dict(truth=False, chance=True, rich=False, faint=('other_night',), incremental=True),
        ('deep', 'I'): dict(truth=True, chance=True, rich=True, faint=('other_night', 'mixed'), incremental=True),
        ('deep', 'V'): dict(truth=False, chance=True, rich=False, faint=(), incremental=False)}
FAINT_NSRC = 300
DET_FIELDS = ('cand_id', 'ra', 'dec', 'ra_raw', 'dec_raw', 'frame_shift', 'fsig', 'el', 'q', 'C', 'sig', 'stat', 'sys', 'beam',
              'snr', 'band', 'lst', 'key', 'night')
FAILS, N = [], [0]
_ENV0 = os.environ.get('P3_TENMIN_PERTIE')        # the 10-minute frame flag before any load (restored after each)


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({str(detail)[:600]})'), flush=True)
    if not ok:
        FAILS.append(name)
    return ok


# =====================================================================================================================
# inputs and the module's grouping per dataset
# =====================================================================================================================

def load(tier, S):
    if tier == '10min':
        return GI.load_tenmin(f'{FROZEN}/tenmin/master_catalog.csv', f'{FROZEN}/tenmin/pertie_detections.csv', S)
    return GI.load_deep(f'{FROZEN}/deep/master_catalog.csv', f'{FROZEN}/deep/frame_beam_bmaj.csv', S)


def module_grouper(tier, S):
    def fn(d):
        a, b = G.group_fork(d, p_fork='config', tier=tier, stokes=S)
        return np.asarray(a), np.asarray(b)
    return fn


def truth_rows():
    import csv
    return [(r['name'], float(r['ra']), float(r['dec'])) for r in csv.DictReader(open(f'{FROZEN}/truth_clean5.csv'))]


def _unit(ra, de):
    r, d = np.radians(ra), np.radians(de)
    return np.c_[np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)]


# =====================================================================================================================
# the harness metrics, ported (harness.py v2b chance_join / rich_absorption / truth; harness_extra.faint_repeat_recovery)
# =====================================================================================================================

def extend(det, tmpl, ra, dec, tier):
    """det + copies of the template rows tmpl at new positions (every n-long array field copied, el, q and C recomputed at the
    new positions with the tier's error model) - the harness's d2"""
    n = len(det['ra']); d2 = {}
    for key, v in det.items():
        d2[key] = np.concatenate([v, v[tmpl]]) if isinstance(v, np.ndarray) and len(v) == n else v
    d2['ra'] = np.r_[det['ra'], ra]; d2['dec'] = np.r_[det['dec'], dec]
    d2['el'], d2['q'] = G.elevation_pa(d2, tier); d2['C'] = G.covariance(d2, tier)
    return d2


def chance_join(det, tier, grouper, frac=0.10, reps=3, npix=120.0, seed=20261026):
    """harness.chance_join (v2b) at both levels from one grouping of each injected set: {level: (chance-join rate % per
    unrelated detection, projected % of real clusters with an unrelated member, injected)} and every rep's (ident, multiples)"""
    P = G.Params()
    rng = np.random.default_rng(seed); n = len(det['ra'])
    ra0 = G.frame_center(det['lst']); l, m = G.lm(det['ra'], det['dec'], ra0, P.frame_dec)
    acc = {lv: [0, 0, []] for lv in LV}; labs = []
    for r in range(reps):
        k = rng.choice(n, int(frac * n), replace=False)
        cd = np.radians(G.cdelt(det['band'][k]))
        a, d = G.radec(l[k] - np.sign(l[k]) * npix * cd, m[k], ra0[k], P.frame_dec)      # toward the meridian: stays on the image
        ok = np.isfinite(a) & np.isfinite(d); k, a, d = k[ok], a[ok], d[ok]
        d2 = extend(det, k, a, d, tier)
        two = grouper(d2); labs.append(two)
        for lv, lab in zip(LV, two):
            lab = np.asarray(lab); realc = set(lab[:n].tolist())
            acc[lv][0] += sum(1 for x in lab[n:] if x in realc); acc[lv][1] += len(k); acc[lv][2].append(len(realc))
    res = {lv: (round(100.0 * j / max(i, 1), 3), round(100.0 * (j / reps / frac) / max(np.mean(c), 1), 2), i)
           for lv, (j, i, c) in acc.items()}
    return res, labs


def rich_absorption(det, tier, labels, grouper_level, per=16, reps=2, seed=7, rich_n=50):
    """harness.rich_absorption (v2b): injections 5-30 arcmin around every cluster of >= rich_n members (half copies of its own
    members, half of other detections of its LSTs); {'own'|'other': {offset bin (sigma): (absorbed, injected)}}, the number of
    rich clusters, every rep's labels"""
    labels = np.asarray(labels); n = len(labels); rng = np.random.default_rng(seed)
    cnt = collections.Counter(labels.tolist()); rich = [c for c, v in cnt.items() if v >= rich_n]
    if not rich:
        return {kind: {b_: (0, 0) for b_ in BINS} for kind in ('own', 'other')}, 0, []
    V = _unit(det['ra'], det['dec']); res = {'own': collections.defaultdict(lambda: [0, 0]), 'other': collections.defaultdict(lambda: [0, 0])}
    by_lst = collections.defaultdict(list)
    for i, l in enumerate(det['lst']):
        by_lst[l].append(i)
    by_lst = {l: np.array(v) for l, v in by_lst.items()}
    labs = []
    for r in range(reps):
        src, pos = [], []
        for c in rich:
            idx = np.nonzero(labels == c)[0]; w = 1 / det['sig'][idx] ** 2
            mv = (V[idx] * w[:, None]).sum(0); mv /= np.linalg.norm(mv)
            ra0, de0 = math.degrees(math.atan2(mv[1], mv[0])) % 360, math.degrees(math.asin(mv[2]))
            lsts = sorted(set(det['lst'][idx].tolist()))
            pool_other = np.setdiff1d(np.concatenate([by_lst[l] for l in lsts]), idx)
            for t_ in range(per):
                kind = 'own' if t_ % 2 == 0 or not len(pool_other) else 'other'
                k = int(rng.choice(idx)) if kind == 'own' else int(rng.choice(pool_other))
                rr = math.sqrt(rng.uniform(5 ** 2, 30 ** 2)) / 60.0; th = rng.uniform(0, 2 * math.pi)
                dra = rr * math.sin(th) / math.cos(math.radians(de0)); dde = rr * math.cos(th)
                src.append((c, k, kind)); pos.append((ra0 + dra, de0 + dde, ra0, de0))
        k = np.array([x[1] for x in src])
        d2 = extend(det, k, np.array([p_[0] % 360 for p_ in pos]), np.array([p_[1] for p_ in pos]), tier)
        lab2 = np.asarray(grouper_level(d2)); labs.append(lab2)
        newc = {c: collections.Counter(lab2[:n][labels == c].tolist()).most_common(1)[0][0] for c in rich}
        for t_, ((c, _, kind), (ra_, de_, ra0, de0)) in enumerate(zip(src, pos)):
            dx, dy = G.offset_en(np.array([ra0]), np.array([de0]), np.array([ra_]), np.array([de_]))
            u = float(G.mahalanobis(dx, dy, d2['C'][n + t_:n + t_ + 1])[0])
            b = '<2' if u < 2 else ('2-4' if u < 4 else ('4-6' if u < 6 else '>6'))
            res[kind][b][1] += 1; res[kind][b][0] += int(lab2[n + t_] == newc[c])
    return {kind: {b: tuple(res[kind][b]) for b in BINS} for kind in ('own', 'other')}, len(rich), labs


TAIL_EDGES = (8.0, 10.0, 15.0)
TAIL_FRAC = (0.106, 0.071, 0.042, 0.028)
TAIL_SCALE = 3.0


def faint_recovery(det, tier, grouper, nsrc=FAINT_NSRC, ns=(2, 3, 5), seed=20261027, partners='mixed', tail=False):
    """harness_extra.faint_repeat_recovery at both levels from one grouping: synthetic sources of n detections (an anchor's
    template moved 40-150 px, partners from its LST hour - 'mixed': one in three from the adjacent hour; 'same_night'; 'other_night'),
    each placed at the true position + N(0, C); {level: {'n=k': {...}}} and the labels"""
    P = G.Params()
    rng = np.random.default_rng(seed); n = len(det['ra'])
    hours = np.array([int(x[:2]) for x in det['lst']])
    by_h = {h: np.nonzero(hours == h)[0] for h in range(24)}
    ra0 = G.frame_center(det['lst']); l, m = G.lm(det['ra'], det['dec'], ra0, P.frame_dec)
    tmpl, src_of, true_pos = [], [], []
    for s in range(nsrc):
        k = ns[s % len(ns)]
        a = int(rng.integers(n)); cd = math.radians(float(G.cdelt(det['band'][a:a + 1])[0]))
        npx = rng.uniform(40, 150)
        ta, td = G.radec(np.array([l[a] - np.sign(l[a]) * npx * cd]), np.array([m[a] + rng.uniform(-40, 40) * cd]), ra0[a:a + 1],
                         P.frame_dec)
        if not (np.isfinite(ta[0]) and np.isfinite(td[0])):
            continue
        h = hours[a]
        if partners == 'mixed':
            pool_h = (h + (1 if rng.random() < 0.5 else -1)) % 24 if s % 3 == 2 else h
            cand = by_h[pool_h] if len(by_h[pool_h]) else by_h[h]
        else:
            cand = by_h[h]
        if partners == 'same_night':
            cand = cand[det['night'][cand] == det['night'][a]]
        members, keys, nights = [a], {det['key'][a]}, {det['night'][a]}
        for _ in range(400):
            if len(members) == k or not len(cand):
                break
            b = int(rng.choice(cand))
            if det['key'][b] in keys or (partners == 'other_night' and det['night'][b] in nights):
                continue
            members.append(b); keys.add(det['key'][b]); nights.add(det['night'][b])
        if len(members) < k:
            continue
        for b in members:
            tmpl.append(b); src_of.append(s); true_pos.append((ta[0], td[0]))
    tmpl = np.array(tmpl); src_of = np.array(src_of); tp = np.array(true_pos)
    dt = {key: (v[tmpl] if isinstance(v, np.ndarray) and len(v) == n else v) for key, v in det.items()}
    dt['ra'], dt['dec'] = tp[:, 0], tp[:, 1]
    dt['el'], dt['q'] = G.elevation_pa(dt, tier)
    C = G.covariance(dt, tier)
    L = np.linalg.cholesky(C)
    z = rng.standard_normal((len(tmpl), 2))
    off = np.einsum('nij,nj->ni', L, z)                                  # (east, north) arcsec
    if tail:
        fr = np.array(TAIL_FRAC)[np.digitize(np.abs(det['snr'][tmpl]), TAIL_EDGES)]
        off = np.where((rng.random(len(tmpl)) < fr)[:, None], off * TAIL_SCALE, off)
    dec_i = tp[:, 1] + off[:, 1] / 3600.0
    ra_i = (tp[:, 0] + off[:, 0] / 3600.0 / np.cos(np.radians(tp[:, 1]))) % 360
    d2 = extend(det, tmpl, ra_i, dec_i, tier)
    two = grouper(d2)
    out = {}
    for lv, lab in zip(LV, two):
        lab = np.asarray(lab); li = lab[n:]; realc = set(lab[:n].tolist()); o = collections.OrderedDict()
        for k in ns:
            S_ = [s for s in sorted(set(src_of.tolist())) if (src_of == s).sum() == k]
            rec = mer = core = tot = 0
            for s in S_:
                ls = li[src_of == s]
                rec += int(len(set(ls.tolist())) == 1)
                mer += int(any(x in realc for x in ls.tolist()))
                core += int((ls == ls[0]).sum() - 1); tot += k - 1
            o['n=%d' % k] = dict(sources=len(S_), recovered_pct=round(100.0 * rec / max(len(S_), 1), 1),
                                 merged_real_pct=round(100.0 * mer / max(len(S_), 1), 1),
                                 partner_with_anchor_pct=round(100.0 * core / max(tot, 1), 1))
        out[lv] = o
    return out, two


def clean5(det, labels, rows=None):
    """harness truth / clean5_in_one_piece: per CLEAN5 source with >= 5 detections within 15 arcmin, (largest cluster's members,
    detections, pieces); in one piece when the largest holds >= 95%"""
    rows = truth_rows() if rows is None else rows
    V = _unit(det['ra'], det['dec']); labels = np.asarray(labels); out = {}
    for name, ra, dec in rows:
        d = np.degrees(np.arccos(np.clip(V @ _unit(np.array([ra]), np.array([dec]))[0], -1, 1))) * 60
        k = np.nonzero(d < 15.0)[0]
        if len(k) >= 5:
            c = collections.Counter(labels[k].tolist())
            out[name] = (c.most_common(1)[0][1], len(k), len(c))
    return out


def binom_tail(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p)"""
    return 1.0 - sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k))


def in_one_piece(t):
    return sum(1 for top, n, _ in t.values() if top >= 0.95 * n), len(t)


def early_mask(det, frac=0.75):
    """harness.stability's split: the detections of the first `frac` of the nights"""
    nights = np.array(sorted(set(det['night'].tolist()))); cut = nights[int(frac * (len(nights) - 1))]
    return det['night'] <= cut


def sub(det, idx):
    n = len(det['ra'])
    return {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in det.items()}


def multiples_counts(det, ident, fork, tier):
    mult = G.is_multiple(det, fork, tier)
    return dict(identities=len(set(np.asarray(ident).tolist())), multiples_groups=len(set(np.asarray(fork).tolist())),
                multiples=len(set(np.asarray(fork)[mult].tolist())), identities_in_multiples=len(set(np.asarray(ident)[mult].tolist())),
                detections_in_multiples=int(mult.sum()))


# =====================================================================================================================
# the checks
# =====================================================================================================================

def check_frozen():
    ok, bad = True, []
    for line in open(f'{FROZEN}/SHA256SUMS'):
        h, f = line.split()
        if hashlib.sha256(open(os.path.join(FROZEN, f), 'rb').read()).hexdigest() != h:
            bad.append(f)
    for line in open(f'{GOLDEN}/SHA256SUMS'):
        h, f = line.split()
        if hashlib.sha256(open(os.path.join(GOLDEN, f), 'rb').read()).hexdigest() != h:
            bad.append('golden/' + f)
    ok = not bad
    check('frozen inputs and golden outputs unchanged (SHA256SUMS)', ok, bad)
    return ok


def golden(tier, S):
    z = np.load(f'{GOLDEN}/{tier}_{S}.npz', allow_pickle=False)
    meta = json.load(open(f'{GOLDEN}/{tier}_{S}.json'))
    return z, meta


def eq_labels(a, b):
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def run_dataset(tier, S, quick=False):
    tag = f'{tier} {S}'
    z, meta = golden(tier, S)
    plan = PLAN[(tier, S)]
    # ---- loader
    det = load(tier, S)
    diffs = []
    for k in DET_FIELDS:
        a, b = np.asarray(det[k]), z['det_' + k]
        if a.dtype.kind in 'fc':
            same = a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])
        else:
            same = a.shape == b.shape and bool((a.astype(str) == b.astype(str)).all())
        if not same:
            diffs.append(k)
    check(f'{tag}: loader - every field of the module det equals the research loader exactly ({len(det["ra"])} detections)', not diffs, diffs)
    # ---- partitions
    grp = module_grouper(tier, S)
    ident, fork = grp(det)
    check(f'{tag}: identity labels = the research labels, element by element', eq_labels(ident, z['ident']),
          (len(set(ident.tolist())), len(set(z['ident'].tolist()))))
    check(f'{tag}: multiples labels = the research labels, element by element', eq_labels(fork, z['fork']),
          (len(set(fork.tolist())), len(set(z['fork'].tolist()))))
    mc = multiples_counts(det, ident, fork, tier)
    check(f'{tag}: identities / multiples-level groups / multiples = the research counts {meta["counts"]}', mc == meta['counts'], mc)
    # ---- nesting
    lab = np.asarray(G.group(det))
    check(f'{tag}: group() = group_fork()[0] exactly (one identity rule)', eq_labels(lab, ident))
    split = collections.defaultdict(set)
    for a, b in zip(ident.tolist(), fork.tolist()):
        split[a].add(b)
    check(f'{tag}: nesting - the multiples level splits no identity', all(len(v) == 1 for v in split.values()),
          sum(1 for v in split.values() if len(v) > 1))
    if G.operating_point(tier, S)[1] is None:
        check(f'{tag}: the multiples level is OFF (ruled: deep V identity only) - multiples = identities', eq_labels(fork, ident))
    else:
        check(f'{tag}: the multiples level merges (fewer groups than identities)', len(set(fork.tolist())) < len(set(ident.tolist())),
              mc)
    # ---- determinism and order independence
    i2, f2 = grp(det)
    check(f'{tag}: determinism - a second run gives identical label arrays', eq_labels(i2, ident) and eq_labels(f2, fork))
    perm = np.random.default_rng(515151).permutation(len(det['ra']))
    ip, fp = grp(sub(det, perm))
    iu = np.empty_like(ip); iu[perm] = ip; fu = np.empty_like(fp); fu[perm] = fp
    check(f'{tag}: order independence - a permuted input gives the same partitions (both levels)',
          G.same_partition(iu, ident) and G.same_partition(fu, fork))
    # ---- CLEAN5 and the LPT
    if plan['truth']:
        for lv, labs in zip(LV, (ident, fork)):
            t = clean5(det, labs)
            gold = {k: tuple(v) for k, v in meta['clean5'][lv].items()}
            check(f'{tag}: CLEAN5 {lv} = research {in_one_piece(gold)} ({t})', t == gold, (t, gold))
            lpt = t.get('LPT J0409+4640')
            check(f'{tag}: the LPT whole at the {lv} level ({lpt})', lpt is not None and lpt[0] == lpt[1] and lpt[2] == 1, lpt)
        one_i, n_i = in_one_piece(clean5(det, ident)); one_f, n_f = in_one_piece(clean5(det, fork))
        check(f'{tag}: CLEAN5 in one piece as approved (identity all but at most one: {one_i}/{n_i}; multiples all: {one_f}/{n_f})',
              n_i >= 3 and one_i >= n_i - 1 and one_f == n_f, (one_i, n_i, one_f, n_f))
    if quick:
        return det, ident, fork, z, meta
    # ---- chance joins
    if plan['chance']:
        res, labs = chance_join(det, tier, grp)
        same = all(eq_labels(a, z[f'chance{r}_ident']) and eq_labels(b, z[f'chance{r}_fork']) for r, (a, b) in enumerate(labs))
        check(f'{tag}: chance_join - every injected run\'s labels = research', same)
        gold = {lv: tuple(v) for lv, v in meta['chance'].items()}
        check(f'{tag}: chance_join rates = research ({res})', res == gold, (res, gold))
        # the multiples level only merges identities, so it joins at least as many injected detections; and both stay at the
        # scale of the approved numbers (full 10-minute I: 0.93% / 3.6%; these 3-LST subsets give up to ~2x)
        check(f'{tag}: chance_join identity {res["identity"][0]}% <= multiples {res["multiples"][0]}% (nesting), both small '
              f'(< 3% / < 8%)', res['identity'][0] <= res['multiples'][0] and res['identity'][0] < 3.0 and res['multiples'][0] < 8.0, res)
    # ---- absorption near rich clusters
    if plan['rich']:
        for lv, base in zip(LV, (ident, fork)):
            pick = 0 if lv == 'identity' else 1
            ra_, nrich, labs = rich_absorption(det, tier, base, lambda d, p=pick: grp(d)[p])
            same = all(eq_labels(a, z[f'rich_{lv}{r}']) for r, a in enumerate(labs))
            check(f'{tag}: absorption ({lv}) - every injected run\'s labels = research ({nrich} rich clusters)', same)
            gold = {k: {b: tuple(v) for b, v in d.items()} for k, d in meta['rich'][lv].items()}
            check(f'{tag}: absorption ({lv}) = research {ra_}', ra_ == gold, (ra_, gold))
            ab = sum(ra_[k][b][0] for k in ('own', 'other') for b in ('4-6', '>6'))
            inj = sum(ra_[k][b][1] for k in ('own', 'other') for b in ('4-6', '>6'))
            pv = binom_tail(ab, inj, 0.01)
            check(f'{tag}: absorption beyond 4 sigma ({lv}) {ab}/{inj} consistent with the known limit <= 1% (binomial '
                  f'P(>= {ab} | 1%) = {pv:.2f} >= 0.05)', inj > 0 and pv >= 0.05, (ab, inj, pv))
    # ---- faint repeaters
    for mode in plan['faint']:
        fr, two = faint_recovery(det, tier, grp, partners=mode)
        same = eq_labels(two[0], z[f'faint_{mode}_ident']) and eq_labels(two[1], z[f'faint_{mode}_fork'])
        check(f'{tag}: faint recovery ({mode}) - labels = research', same)
        check(f'{tag}: faint recovery ({mode}) = research', json.loads(json.dumps(fr)) == meta['faint'][mode],
              (fr, meta['faint'][mode]))
        rec = {lv: [fr[lv]['n=%d' % k]['recovered_pct'] for k in (2, 3, 5)] for lv in LV}
        check(f'{tag}: faint recovery ({mode}) n=2/3/5: identity {rec["identity"]} <= multiples {rec["multiples"]} (nesting)',
              all(a <= b for a, b in zip(rec['identity'], rec['multiples'])), rec)
    return det, ident, fork, z, meta


def unrecorded_changes(old_state, res):
    """the changes of res (a GroupingResult) against old_state that have no event, and events that match no change. Through
    res.matched (new cand_id -> (state cand_id, 'exact' | 'tolerance')): a matched detection whose identity or multiples ID
    changed needs exactly one 'move' event at that level (from the matched record's ID); a tolerance match needs its
    'remeasured' event; a cand_id of the state that is not matched to its own record needs a 'reused' event (beyond tolerance)
    or an 'ambiguous' one. Returns (missing, spurious, n_changed)."""
    moves = collections.Counter((e['level'], e['cand_id'], e['from_id'], e['to_id']) for e in res.events if e['event'] == 'move')
    reused = {e['cand_id'] for e in res.events if e['event'] == 'reused'}
    remeas = {e['cand_id']: e['cand_id_old'] for e in res.events if e['event'] == 'remeasured'}
    amb_new = {c for e in res.events if e['event'] == 'ambiguous' for c in e['new']}
    want = collections.Counter(); missing = []; n = 0
    for c, rec in res.state.assign.items():
        i, m = rec[0], rec[1]
        mm = res.matched.get(c)
        own = old_state.assign.get(c)
        if mm is not None:
            oc, kind = mm
            a = old_state.assign[oc]
            if kind == 'tolerance':
                n += 1
                if remeas.get(c) != oc:
                    missing.append(('remeasured', c, oc))
            for lv, x, y in (('identity', a[0], i), ('multiples', a[1], m)):
                if x != y:
                    n += 1; want[(lv, c, x, y)] += 1
                    if moves[(lv, c, x, y)] != 1:
                        missing.append((lv, c, x, y))
            if own is not None and oc != c:          # matched another record: its own cand_id moved beyond tolerance
                n += 1
                if c not in reused:
                    missing.append(('reused', c))
        elif own is not None:                        # a cand_id of the state matched by nothing: reused or ambiguous
            n += 1
            if c not in reused and c not in amb_new:
                missing.append(('reused/ambiguous', c))
    spurious = ([k for k in moves if k not in want] + [c for c in reused if c not in old_state.assign]
                + [c for c in reused if (res.matched.get(c) or ('',))[0] == c]
                + [c for c in remeas if (res.matched.get(c) or ('', ''))[1] != 'tolerance'])
    return missing, spurious, n


def run_incremental(tier, S, det, z, meta):
    """the incremental API on real data: a first run on the early nights, then every detection"""
    tag = f'{tier} {S}'
    em = early_mask(det)
    d_early = sub(det, np.nonzero(em)[0])
    r1 = G.group_catalog(d_early, tier, S, run_label='early')
    check(f'{tag}: incremental - the first run issues IDs 1..n per level ({r1.summary["identities"]} identities)',
          r1.summary['new_ids']['identity'] == r1.summary['identities'] and r1.state.next_seq['identity'] == r1.summary['identities'] + 1)
    r2 = G.group_catalog(det, tier, S, state=r1.state, run_label='late')
    # 2026-11-14 (AMEND-20261114-GROUPING-INCREMENTAL-CHANGES-ONLY-WHAT-NEW-DETECTIONS-TOUCH): the incremental mode changes only
    # what a new detection touches, so its partitions are no longer the research frozen run's (which also merged and seeded
    # among frozen components alone); the difference is printed, and the rule is checked instead
    late = set(np.asarray(det['cand_id'])[~em].astype(str).tolist())   # group_catalog's cand_ids are str
    for lv, now, gold in (('identity', r2.identity_id, z['incr_ident']), ('multiples', r2.multiples_id, z['incr_fork'])):
        print(f'  {tag}: incremental {lv} partition = the research frozen run: {G.same_partition(now, gold)}', flush=True)
        prev = r1.identity_id if lv == 'identity' else r1.multiples_id
        g1 = {}
        for c, x in zip(r1.cand_id.tolist(), prev.tolist()):
            g1.setdefault(x, set()).add(c)
        g2 = {}
        for c, x in zip(r2.cand_id.tolist(), now.tolist()):
            g2.setdefault(x, set()).add(c)
        early_only = [m for m in g2.values() if not (m & late)]
        same = {frozenset(m) for m in g1.values()}
        bad = [m for m in early_only if frozenset(m) not in same]
        check(f'{tag}: incremental - every {lv} group without a new detection is exactly a group of the first run '
              f'({len(early_only)} such groups)', not bad, [sorted(m)[:3] for m in bad[:3]])
    # a run without a new detection changes nothing at either level
    r0 = G.group_catalog(d_early, tier, S, state=r1.state, run_label='noop', mode='incremental')
    check(f'{tag}: incremental without a new detection - identity and multiples partitions unchanged, no merge '
          f'({r0.summary["merges"]} merges)', G.same_partition(r0.identity_id, r1.identity_id)
          and G.same_partition(r0.multiples_id, r1.multiples_id) and all(v == 0 for v in r0.summary['merges'].values())
          and r0.cand_id.tolist() == r1.cand_id.tolist(),
          (G.same_partition(r0.identity_id, r1.identity_id), G.same_partition(r0.multiples_id, r1.multiples_id),
           r0.cand_id.tolist() == r1.cand_id.tolist()))
    # no early identity split; every early ID present (live) or merged into a live one
    pos = {c: j for j, c in enumerate(r2.cand_id.tolist())}
    split = 0
    for i in set(r1.identity_id.tolist()):
        now = {r2.identity_id[pos[c]] for c, x in zip(r1.cand_id.tolist(), r1.identity_id.tolist()) if x == i}
        split += int(len(now) > 1)
    check(f'{tag}: incremental - no identity of the first run is split', split == 0, split)
    st = r2.state
    carried = all(st.resolve(i) in set(r2.identity_id.tolist()) for i in set(r1.identity_id.tolist()))
    check(f'{tag}: incremental - every first-run ID is live or merged into a live ID', carried)
    merged = [i for i, e in st.ids.items() if e['status'] == 'merged']
    ev = [e for e in r2.events if e['event'] == 'merge']
    check(f'{tag}: incremental - every merge is recorded ({len(merged)} absorbed IDs, {len(ev)} merge events)',
          sorted(merged) == sorted(x for e in ev for x in e['absorbed']) and all(st.ids[x]['into'] == e['id'] for e in ev for x in e['absorbed']))
    missing, spurious, nch = unrecorded_changes(r1.state, r2)
    check(f'{tag}: incremental - EVERY changed assignment has its event ({nch} changes; {r2.summary["moves"]} moves)',
          not missing and not spurious, (missing[:5], spurious[:5]))
    check(f'{tag}: incremental - identities only merge: every identity-level move is a merge',
          all(e['reason'] == 'merge' for e in r2.events if e['event'] == 'move' and e['level'] == 'identity'))
    new_seq = {lv: sorted(G._seq_of(i) for i in st.ids if i not in r1.state.ids and st.ids[i]['level'] == lv) for lv in LV}
    check(f'{tag}: incremental - new IDs continue each level\'s sequence without gaps or reuse '
          f'({len(new_seq["identity"])} identity, {len(new_seq["multiples"])} multiples)',
          all(new_seq[lv] == list(range(r1.state.next_seq[lv], st.next_seq[lv])) for lv in LV))
    check(f'{tag}: incremental - the summary counts the run ({r2.summary["merges"]} merges, {r2.summary["new_ids"]} new IDs)',
          r2.summary['mode'] == 'incremental' and r2.summary['n_det_previous'] == int(em.sum()))
    # save / load
    d = os.path.join(T, f'state_{tier}_{S}')
    pth = st.save(d)
    st2 = G.GroupingState.load(d, tier, S)
    check(f'{tag}: state save/load round trip (one file, fingerprints kept)', st2.ids == json.loads(json.dumps(st.ids))
          and st2.assign == st.assign and st2.next_seq == st.next_seq and os.listdir(d) == [os.path.basename(pth)])
    # refusals: a window with both kinds; a batch over a state
    keys = det['key']; w = keys[~em][0]
    half = np.nonzero(keys == w)[0]
    bad_state = G.GroupingState(tier, S, ids=st.ids, assign={c: st.assign[c] for c in det['cand_id'][half[:1]].tolist()},
                                next_seq=st.next_seq)
    try:
        G.group_catalog(det, tier, S, state=bad_state, run_label='x', mode='incremental'); refused = False
    except G.GroupingRefused as e:
        refused = 'regroup' in str(e)
    check(f'{tag}: incremental - a window holding both state and new detections REFUSES (names mode regroup)', refused)
    try:
        G.group_catalog(det, tier, S, state=r1.state, run_label='x', mode='batch'); refused = False
    except G.GroupingRefused:
        refused = True
    check(f'{tag}: a batch run over a state holding IDs REFUSES (IDs are never reused)', refused)
    # regroup: the whole partition recomputed (= the batch partition), IDs carried by member detections
    r3 = G.group_catalog(det, tier, S, state=r1.state, run_label='regroup', mode='regroup')
    check(f'{tag}: regroup - partitions = the batch partitions', G.same_partition(r3.identity_id, z['ident'])
          and G.same_partition(r3.multiples_id, z['fork']))
    live3 = set(r3.identity_id.tolist())
    ok = all(r3.state.resolve(i) in live3 or r3.state.ids[i]['status'] == 'retired'
             for i in r1.state.ids if r1.state.ids[i]['level'] == 'identity')
    check(f'{tag}: regroup - every first-run identity ID is live, merged into a live ID, or retired', ok)
    missing, spurious, nch = unrecorded_changes(r1.state, r3)
    check(f'{tag}: regroup - EVERY changed assignment has its event ({nch} changes; moves {r3.summary["moves"]}, splits '
          f'{r3.summary["splits"]}, merges {r3.summary["merges"]})', not missing and not spurious, (missing[:5], spurious[:5]))
    sp = [e for e in r3.events if e['event'] == 'split']
    pos3 = collections.defaultdict(collections.Counter)
    for c, (i, m, *_) in r3.state.assign.items():
        a = r1.state.assign.get(c)
        if a:
            pos3[('identity', a[0])][i] += 1; pos3[('multiples', a[1])][m] += 1
    truly = {k for k, v in pos3.items() if len(v) > 1}
    check(f'{tag}: regroup - a split event for exactly the IDs whose detections now lie in several groups ({len(truly)})',
          {(e['level'], e['id']) for e in sp} == truly and all(dict(pos3[(e['level'], e['id'])]) == e['pieces'] for e in sp))


def moved_copy(det, tier, rows, ddec=20.0, dsnr=0.0, snr_factor=1.0, dra_arcsec=0.0):
    """det with the detections `rows` changed under the SAME cand_ids: moved by ddec degrees in Dec and/or dra_arcsec east
    (catalog and common-frame positions), their S/N changed (+dsnr, x snr_factor) - what a re-search produces"""
    d = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in det.items()}
    for k in ('dec', 'dec_raw'):
        d[k][rows] = np.where(d[k][rows] > 0, d[k][rows] - ddec, d[k][rows] + ddec)
    for k, kd in (('ra', 'dec'), ('ra_raw', 'dec_raw')):
        d[k][rows] = (d[k][rows] + dra_arcsec / 3600.0 / np.cos(np.radians(d[kd][rows]))) % 360.0
    d['snr'][rows] = (d['snr'][rows] + dsnr) * snr_factor
    d['el'], d['q'] = G.elevation_pa(d, tier); d['C'] = G.covariance(d, tier)
    return d


def tolerance_arcsec(det, j):
    """the match tolerance of detection j against itself (config ID_MATCH_*): max(err fraction x its fitted error, FWHM
    fraction x its beam)"""
    g = G._cfg().catalog_grouping
    return max(g.ID_MATCH_POS_ERR_FRACTION * float(det['stat'][j]), g.ID_MATCH_POS_FWHM_FRACTION * float(det['beam'][j]) * 60.0)


def run_reused(tier, S, det):
    """a cand_id the state knows that names ANOTHER detection now (production re-numbers on a re-search) is never carried by
    cand_id alone. The review's case: V1_04h_2025-01-02T01_18MHz_0017 moved by 20 deg kept its identity with no refusal or event."""
    tag = f'{tier} {S}'
    r0 = G.group_catalog(det, tier, S, run_label='all')
    cid = det['cand_id'].tolist()
    keys = det['key']
    j = cid.index('V1_04h_2025-01-02T01_18MHz_0017') if 'V1_04h_2025-01-02T01_18MHz_0017' in cid else \
        next(k for k in range(len(cid)) if (keys == keys[k]).sum() >= 2)
    nwin = int((keys == keys[j]).sum())
    old_i, old_m = r0.identity_id[j], r0.multiples_id[j]
    d2 = moved_copy(det, tier, np.array([j]))
    try:
        G.group_catalog(d2, tier, S, state=r0.state, run_label='moved', mode='incremental'); refused, msg = False, ''
    except G.GroupingRefused as e:
        refused, msg = True, str(e)
    check(f'{tag}: reused cand_id {cid[j]} (moved 20 deg; its window holds {nwin}): the incremental mode REFUSES (mixed window, '
          f'names the reuse)', refused and 'reuse a cand_id' in msg and 'regroup' in msg, msg[:300])
    r = G.group_catalog(d2, tier, S, state=r0.state, run_label='moved', mode='regroup')
    ev = [e for e in r.events if e['event'] == 'reused']
    check(f'{tag}: regroup - the moved detection does not keep its old IDs ({old_i} -> {r.identity_id[j]}, {old_m} -> '
          f'{r.multiples_id[j]})', r.identity_id[j] != old_i and r.multiples_id[j] != old_m)
    check(f'{tag}: regroup - a reused event names the cand_id, both fingerprints and both assignments',
          len(ev) == 1 and ev[0]['cand_id'] == cid[j] and ev[0]['from_identity'] == old_i and ev[0]['to_identity'] == r.identity_id[j]
          and ev[0]['fingerprint_old'] != ev[0]['fingerprint_new'] and r.summary['n_cand_id_reused'] == 1
          and r.summary['n_det_state_absent'] == 1, (ev, r.summary.get('n_cand_id_reused')))
    missing, spurious, nch = unrecorded_changes(r0.state, r)
    check(f'{tag}: regroup with a reused cand_id - every changed assignment has its event ({nch} changes)', not missing and not spurious,
          (missing[:5], spurious[:5]))
    a, b = G.group_fork(d2, p_fork='config', tier=tier, stokes=S)
    check(f'{tag}: regroup with a reused cand_id - partitions = the batch partitions of the changed input',
          G.same_partition(r.identity_id, a) and G.same_partition(r.multiples_id, b))
    # a window re-measured WHOLE (every detection's S/N changed slightly, positions kept): with the tolerance (default) the same
    # detections - matched, IDs carried, the window frozen in the incremental mode; with exact fingerprints only, a new window
    rows = np.nonzero(keys == keys[j])[0]
    d3 = moved_copy(det, tier, rows, ddec=0.0, dsnr=0.01)
    r3 = G.group_catalog(d3, tier, S, state=r0.state, run_label='remeasured', mode='incremental')
    rem = {e['cand_id'] for e in r3.events if e['event'] == 'remeasured'}
    mchg = [(e['reason'], e['level']) for e in r3.events if e['event'] == 'move']
    check(f'{tag}: a window re-measured whole is matched within tolerance ({len(rows)} remeasured, every identity ID kept, nothing '
          f'absent; the multiples level changes only by recorded merges: {len(mchg)})',
          rem == set(det['cand_id'][rows].tolist()) and r3.summary['n_det_state_absent'] == 0
          and r3.summary['n_det_previous'] == len(cid) and list(r3.identity_id) == list(r0.identity_id)
          and all(x == ('merge', 'multiples') for x in mchg), r3.summary)
    r3x = G.group_catalog(d3, tier, S, state=r0.state, run_label='remeasured', mode='incremental', match_tolerance=False)
    ev3 = {e['cand_id'] for e in r3x.events if e['event'] == 'reused'}
    check(f'{tag}: ... with exact fingerprints only it is a new window ({len(rows)} reused cand_ids, the old detections absent)',
          ev3 == set(det['cand_id'][rows].tolist()) and r3x.summary['n_det_state_absent'] == len(rows)
          and r3x.summary['n_det_previous'] == len(cid) - len(rows) and r3x.summary['match_tolerance'] is None)
    for rr, what in ((r3, 'tolerance'), (r3x, 'exact')):
        missing, spurious, nch = unrecorded_changes(r0.state, rr)
        check(f'{tag}: incremental with a re-measured window ({what}) - every changed assignment has its event ({nch} changes)',
              not missing and not spurious, (missing[:5], spurious[:5]))


def run_tolerance(tier, S, det):
    """the tolerance match (coordinator 2026-11-02: a re-search of the same frames re-measures the same detection, and its stable
    IDs survive): a re-measured detection keeps its ID (same cand_id, or re-numbered); one beyond tolerance does not; an
    ambiguous match matches none and is recorded"""
    tag = f'{tier} {S}'
    r0 = G.group_catalog(det, tier, S, run_label='all')
    cid = det['cand_id'].tolist()
    cnt = collections.Counter(r0.identity_id.tolist())
    keys = det['key']
    # a detection alone in its identity (its ID lives or dies with it), in a window of >= 2 detections, positive S/N
    j = next(k for k in range(len(cid)) if cnt[r0.identity_id[k]] == 1 and (keys == keys[k]).sum() >= 2 and det['snr'][k] > 0)
    tol = tolerance_arcsec(det, j)
    old_i, old_m = r0.identity_id[j], r0.multiples_id[j]
    # (1) re-measured within tolerance (0.6 x tol east, S/N x 1.2), same cand_id: keeps its IDs
    d1 = moved_copy(det, tier, np.array([j]), ddec=0.0, dra_arcsec=0.6 * tol, snr_factor=1.2)
    r1 = G.group_catalog(d1, tier, S, state=r0.state, run_label='remeasure', mode='regroup')
    ev = [e for e in r1.events if e['event'] == 'remeasured']
    check(f'{tag}: tolerance - {cid[j]} re-measured 0.6 x tolerance ({0.6 * tol:.1f}") away with S/N x 1.2 keeps its IDs '
          f'({old_i} -> {r1.identity_id[j]})', r1.identity_id[j] == old_i and r1.multiples_id[j] == old_m and len(ev) == 1
          and ev[0]['cand_id'] == cid[j] and ev[0]['cand_id_old'] == cid[j] and not any(e['event'] == 'reused' for e in r1.events)
          and r1.matched[cid[j]] == (cid[j], 'tolerance'), (ev, r1.summary['n_matched_tolerance']))
    r1x = G.group_catalog(d1, tier, S, state=r0.state, run_label='remeasure', mode='regroup', match_tolerance=False)
    check(f'{tag}: ... with exact fingerprints only it would lose them ({old_i} retired, {r1x.identity_id[j]} new: the churn)',
          r1x.identity_id[j] != old_i and r1x.state.ids[old_i]['status'] == 'retired')
    try:
        ri = G.group_catalog(d1, tier, S, state=r0.state, run_label='remeasure', mode='incremental'); ok_inc = True
    except G.GroupingRefused as e:
        ok_inc, ri = str(e), None
    check(f'{tag}: ... and its window is not mixed for the incremental mode (no refusal; IDs kept)', ok_inc is True
          and ri.identity_id[j] == old_i, ok_inc)
    # (2) the same detection RE-NUMBERED (a new cand_id), re-measured within tolerance: keeps the IDs too
    d2 = moved_copy(det, tier, np.array([j]), ddec=0.0, dra_arcsec=0.6 * tol, snr_factor=1.2)
    d2['cand_id'] = d2['cand_id'].astype(object); d2['cand_id'][j] = cid[j] + '_renumbered'; d2['cand_id'] = d2['cand_id'].astype(str)
    r2 = G.group_catalog(d2, tier, S, state=r0.state, run_label='renumber', mode='regroup')
    ev = [e for e in r2.events if e['event'] == 'remeasured']
    check(f'{tag}: tolerance - the same detection re-numbered ({cid[j]}_renumbered) keeps the IDs ({r2.identity_id[j]})',
          r2.identity_id[j] == old_i and len(ev) == 1 and ev[0]['cand_id_old'] == cid[j]
          and r2.summary['n_det_state_absent'] == 0, (ev, r2.summary['n_det_state_absent']))
    # (3) beyond tolerance (1.6 x tol): not the same detection - 'reused', its old ID (it was alone) retires
    d3 = moved_copy(det, tier, np.array([j]), ddec=0.0, dra_arcsec=1.6 * tol)
    r3 = G.group_catalog(d3, tier, S, state=r0.state, run_label='beyond', mode='regroup')
    ev = [e for e in r3.events if e['event'] == 'reused']
    check(f'{tag}: tolerance - 1.6 x tolerance away it is NOT the same detection (reused event, not matched; {old_i} retired, '
          f'{r3.identity_id[j]})', len(ev) == 1 and ev[0]['cand_id'] == cid[j] and cid[j] not in r3.matched
          and r3.state.ids[old_i]['status'] == 'retired' and r3.identity_id[j] != old_i, (ev, r3.matched.get(cid[j])))
    d4 = moved_copy(det, tier, np.array([j]), ddec=0.0, snr_factor=1.4)
    r4 = G.group_catalog(d4, tier, S, state=r0.state, run_label='snr', mode='regroup')
    check(f'{tag}: tolerance - the same position with S/N x 1.4 (> 1.3) is NOT the same detection (reused)',
          any(e['event'] == 'reused' and e['cand_id'] == cid[j] for e in r4.events) and cid[j] not in r4.matched)
    # (4) AMBIGUOUS: the detection re-measured (0.3 x tol) AND a second new detection within tolerance of the same record
    #     (0.3 x tol the other way): none matched, recorded; the record is absent; neither is 'reused'
    dA = moved_copy(det, tier, np.array([j]), ddec=0.0, dra_arcsec=0.3 * tol)
    dB = moved_copy(det, tier, np.array([j]), ddec=0.0, dra_arcsec=-0.3 * tol)
    d5 = {k: (np.concatenate([v, v[[j]]]) if isinstance(v, np.ndarray) and len(v) == len(cid) else v) for k, v in dA.items()}
    for k in ('ra', 'dec', 'ra_raw', 'dec_raw', 'el', 'q', 'C', 'snr'):
        d5[k][-1] = dB[k][j]
    d5['cand_id'] = np.array(cid + [cid[j] + '_dup'])
    r5 = G.group_catalog(d5, tier, S, state=r0.state, run_label='ambiguous', mode='regroup')
    amb = [e for e in r5.events if e['event'] == 'ambiguous']
    check(f'{tag}: ambiguous - two new detections within tolerance of one record match NONE and are recorded ({amb[:1]})',
          len(amb) == 1 and amb[0]['new'] == sorted([cid[j], cid[j] + '_dup']) and amb[0]['old'] == [cid[j]]
          and cid[j] not in r5.matched and cid[j] + '_dup' not in r5.matched and r5.summary['n_ambiguous'] == dict(groups=1, new=2, old=1)
          and not any(e['event'] == 'reused' for e in r5.events) and r5.summary['n_det_state_absent'] == 1)
    try:
        G.group_catalog(d5, tier, S, state=r0.state, run_label='ambiguous', mode='incremental'); refused, msg = False, ''
    except G.GroupingRefused as e:
        refused, msg = True, str(e)
    check(f'{tag}: ambiguous - the incremental mode refuses the window and names the ambiguity', refused and 'ambiguous' in msg, msg[:300])
    for rr, dd, what in ((r1, d1, 're-measured'), (r2, d2, 're-numbered'), (r3, d3, 'beyond'), (r4, d4, 'S/N beyond'),
                         (r5, d5, 'ambiguous')):
        missing, spurious, nch = unrecorded_changes(r0.state, rr)
        a, b = G.group_fork(dd, p_fork='config', tier=tier, stokes=S)
        check(f'{tag}: {what} - every changed assignment has its event ({nch}); the partition = the batch partition (the match '
              f'decides only the IDs)', not missing and not spurious and G.same_partition(rr.identity_id, a)
              and G.same_partition(rr.multiples_id, b), (missing[:3], spurious[:3]))


def check_ids(tier, S, det, ident, fork):
    tag = f'{tier} {S}'
    r = G.group_catalog(det, tier, S, run_label='ids')
    check(f'{tag}: IDs - group_catalog partitions = group_fork', G.same_partition(r.identity_id, ident) and G.same_partition(r.multiples_id, fork))
    ids = sorted(set(r.identity_id.tolist()), key=G._seq_of)
    pre = G.format_id('identity', tier, S, 1).rsplit('-', 1)[0]
    check(f'{tag}: IDs - identity IDs are {pre}-0000001.. without gaps', [G._seq_of(i) for i in ids] == list(range(1, len(ids) + 1))
          and all(i.startswith(pre + '-') for i in ids))
    perm = np.random.default_rng(99).permutation(len(det['ra']))
    rp = G.group_catalog(sub(det, perm), tier, S, run_label='ids')
    m1 = dict(zip(r.cand_id.tolist(), r.identity_id.tolist())); m2 = dict(zip(rp.cand_id.tolist(), rp.identity_id.tolist()))
    n1 = dict(zip(r.cand_id.tolist(), r.multiples_id.tolist())); n2 = dict(zip(rp.cand_id.tolist(), rp.multiples_id.tolist()))
    check(f'{tag}: IDs - order independent (a permuted input gives every detection the same identity and multiples IDs)',
          m1 == m2 and n1 == n2)
    nm = [r.names[i] for i in ids]
    check(f'{tag}: names - unique within the catalog ({r.summary["name_suffixes"]} letter suffixes)', len(set(nm)) == len(nm))
    check(f'{tag}: the run summary records the astrom_frame generation ({r.summary["frame_generation"]})',
          r.summary['frame_generation'] == det['frame_generation'] and r.summary['frame_generation'])
    check(f'{tag}: names - J-names (JHHMM+DDMM or JHHMMSS+DDMMSS)', all(x[0] == 'J' and ('+' in x or '-' in x) for x in nm))
    return r


def check_record_logic():
    """the ID carry on synthetic partitions (no grouping run): merge keeps the oldest ID, a split-off piece gets a new ID, a vanished
    identity retires, refusals"""
    st = G.GroupingState('10min', 'I')
    for _ in range(3):
        st.issue('identity', 'r0')
    a, b, c = (G.format_id('identity', '10min', 'I', k) for k in (1, 2, 3))
    cid = np.array(['c1', 'c2', 'c3', 'c4', 'c5'])
    snr = np.array([9.0, 8.0, 7.5, 7.2, 10.0])
    prev = [b, a, a, '', '']            # c1 was in B, c2-c3 in A; C (seq 3) holds no detection now
    labels = np.array([0, 0, 1, 1, 2])  # {c1, c2} = B + part of A; {c3, c4} = rest of A + new; {c5} new
    try:
        G._carry('identity', labels, prev, cid, snr, st, 'r1', allow_split=False); refused = False
    except G.GroupingRefused:
        refused = True
    check('record: an incremental split REFUSES', refused)
    st = G.GroupingState('10min', 'I')
    for _ in range(3):
        st.issue('identity', 'r0')
    prev = [b, a, '', '', '']
    labels = np.array([0, 0, 1, 1, 2])   # {c1, c2}: B + A merge; {c3, c4}, {c5}: new
    out, ev = G._carry('identity', labels, prev, cid, snr, st, 'r1', allow_split=False)
    mv = [e for e in ev if e['event'] == 'move']
    check('record: a merge keeps the OLDEST ID and records the other as merged into it (+ a merge move for its detection)',
          out[0] == a and st.ids[b]['status'] == 'merged' and st.ids[b]['into'] == a and ev and ev[0]['event'] == 'merge'
          and ev[0]['id'] == a and ev[0]['absorbed'] == [b]
          and [(e['cand_id'], e['from_id'], e['to_id'], e['reason']) for e in mv] == [('c1', b, a, 'merge')], (out, ev))
    check('record: new identities get the next IDs, brightest first', out[2] == G.format_id('identity', '10min', 'I', 4)
          and out[1] == G.format_id('identity', '10min', 'I', 5), out)
    st = G.GroupingState('10min', 'I')
    for _ in range(2):
        st.issue('identity', 'r0')
    prev = [a, a, a, b, '']
    labels = np.array([0, 0, 1, 2, 2])   # A split 2 + 1 (regroup); B's detection joins a new one
    out, ev = G._carry('identity', labels, prev, cid, snr, st, 'r1', allow_split=True)
    sp = [e for e in ev if e['event'] == 'split']
    mv = [e for e in ev if e['event'] == 'move']
    check('record: a regroup split - the piece holding most detections keeps the ID, the other gets a new one, recorded (split + move)',
          out[0] == a and out[1] not in (a, b) and out[2] == b and len(sp) == 1 and sp[0]['id'] == a
          and sp[0]['pieces'] == {a: 2, out[1]: 1} and [(e['cand_id'], e['from_id'], e['to_id'], e['reason']) for e in mv]
          == [('c3', a, out[1], 'split')], (out, ev))
    # the review's case: previous [P,P,P,Q,Q,Q,Q,Q] regrouped [0,0,1,1,1,1,1,1] - P's third detection lands in Q's group
    st = G.GroupingState('10min', 'I')
    p_, q_ = st.issue('identity', 'r0'), st.issue('identity', 'r0')
    cid8 = np.array(['d%d' % k for k in range(8)]); snr8 = np.linspace(8.0, 9.0, 8)
    lab8 = np.array([0, 0, 1, 1, 1, 1, 1, 1]); prev8 = [p_] * 3 + [q_] * 5
    out, ev = G._carry('identity', lab8, prev8, cid8, snr8, st, 'r1', allow_split=True)
    mv = [(e['cand_id'], e['from_id'], e['to_id'], e['reason']) for e in ev if e['event'] == 'move']
    sp = [e for e in ev if e['event'] == 'split']
    check('record: the review case [P,P,P,Q,Q,Q,Q,Q] -> [0,0,1,1,1,1,1,1] records P\'s detection that took Q (move "moved" + P split)',
          out == {0: p_, 1: q_} and mv == [('d2', p_, q_, 'moved')] and len(sp) == 1 and sp[0]['id'] == p_
          and sp[0]['pieces'] == {p_: 2, q_: 1}, (out, ev))
    try:
        G._carry('identity', lab8, prev8, cid8, snr8, G.GroupingState('10min', 'I', ids=st.ids, next_seq=st.next_seq), 'r2',
                 allow_split=False); refused = False
    except G.GroupingRefused:
        refused = True
    check('record: ... and the incremental mode (identity level) REFUSES that split', refused)
    try:
        G._carry('identity', labels, ['GTI-0000099', '', '', '', ''], cid, snr, st, 'r2', allow_split=True); refused = False
    except G.GroupingRefused:
        refused = True
    check('record: a detection carrying an unknown ID REFUSES', refused)
    s2 = G.GroupingState('10min', 'I')
    i1 = s2.issue('identity', 'r0'); s2.next_seq['identity'] = 1
    try:
        s2.issue('identity', 'r1'); refused = False
    except G.GroupingRefused:
        refused = True
    check('record: an ID is never issued twice (a damaged sequence REFUSES)', refused and i1 == a)
    # the state file: one file, replaced atomically; no state REFUSES unless new=True; new=True over a state REFUSES
    d = os.path.join(T, 'state_files')
    try:
        G.GroupingState.load(d, '10min', 'I'); refused = False
    except G.GroupingRefused as e:
        refused = 'new=True' in str(e)
    empty = G.GroupingState.load(d, '10min', 'I', new=True)
    check('state: a directory without a state REFUSES unless new=True (IDs never restart at 1 silently)', refused and not empty.ids)
    s3 = G.GroupingState('10min', 'I'); i3 = s3.issue('identity', 'r0'); m3 = s3.issue('multiples', 'r0')
    s3.assign['c1'] = (i3, m3, ('04h|2025-01-02T01', '18MHz', '1.5', '2.5', '7.25'), '80.7', '30.0')
    pth = s3.save(d)
    try:
        G.GroupingState.load(d, '10min', 'I', new=True); refused = False
    except G.GroupingRefused:
        refused = True
    back = G.GroupingState.load(d, '10min', 'I')
    check('state: saved as ONE file; new=True over it REFUSES; load returns it (fingerprints included)', refused
          and os.listdir(d) == [os.path.basename(pth)] and back.assign == s3.assign and back.ids == s3.ids)
    doc = json.load(open(pth)); doc['assign']['c1'] = doc['assign']['c1'][:4]
    json.dump(doc, open(pth, 'w'))
    try:
        G.GroupingState.load(d, '10min', 'I'); refused = False
    except G.GroupingRefused:
        refused = True
    doc = json.load(open(pth)); doc['assign']['c1'] = [i3, m3, '04h|2025-01-02T01', '18MHz', '1.5', '2.5', '7.25', '80.7', '30.0']
    doc['next_seq']['identity'] = 1
    json.dump(doc, open(pth, 'w'))
    try:
        G.GroupingState.load(d, '10min', 'I'); refused2 = False
    except G.GroupingRefused:
        refused2 = True
    check('state: a malformed assignment or a next_seq that would re-issue an ID REFUSES', refused and refused2)


def check_config_and_refusals(det10):
    from exopipe.phase3.config import CFG
    check('config: operating points as ruled (identity 0.97, cap 4, multiples 0.85 / deep V off)',
          G.operating_point('10min', 'I') == (0.97, 0.85, 4.0) and G.operating_point('10min', 'V') == (0.97, 0.85, 4.0)
          and G.operating_point('deep', 'I') == (0.97, 0.85, 4.0) and G.operating_point('deep', 'V') == (0.97, None, 4.0))
    layout = ('SUBLST_DIR', 'STATE_SUBDIR', 'BUILDER_SHA1',   # where the stage writes and the builder pin (item F.3): not rulings
              'FROZEN_BUILDER_SHA1')                          # the builders whose population-wide quantities the frozen statistics name: a pin
    check('config: every key of catalog_grouping is ruled', all(CFG.ruled(f'catalog_grouping.{k}') for k, _ in CFG.catalog_grouping.items()
                                                             if not k.startswith('ID_') and k not in layout))
    try:
        G.cdelt(np.array(['91MHz'])); refused = False
    except G.GroupingRefused:
        refused = True
    check('an unknown band REFUSES (no silent pixel scale)', refused)
    d = dict(det10); d['C'] = det10['C'].copy(); d['C'][0, 0, 0] = np.nan
    try:
        G.group(d); refused = False
    except G.GroupingRefused:
        refused = True
    check('a non-finite covariance REFUSES', refused)
    d = dict(det10); d['band'] = det10['band'].copy(); d['band'][0] = 'Foo'
    try:
        G.group(d); refused = 'GroupingRefused'
    except G.GroupingRefused:
        refused = True
    except Exception as e:                       # noqa: BLE001 - the failure this check is about
        refused = type(e).__name__
    check('a band that is neither a wideband nor <nu>MHz REFUSES (GroupingRefused, not ValueError)', refused is True, refused)
    d = dict(det10); d['night'] = np.array(['n%05d' % k for k in range(len(det10['night']))])
    try:
        G.group(d); refused = False
    except G.GroupingRefused as e:
        refused = '1000' in str(e)
    check('1000 or more nights REFUSE (the (LST hour, night) code holds fewer)', refused)
    p = os.path.join(T, 'bad_master.csv')
    with open(f'{FROZEN}/tenmin/master_catalog.csv') as f, open(p, 'w') as g:
        lines = f.read().splitlines()
        hdr = lines[0].split(','); j = hdr.index('pos_stat_maj_arcsec')
        row = lines[1].split(','); row[j] = ''
        g.write('\n'.join([lines[0], ','.join(row)] + lines[2:]) + '\n')
    try:
        GI.load_tenmin(p, f'{FROZEN}/tenmin/pertie_detections.csv', 'I' if row[hdr.index('stokes')] == 'I' else 'V', common=False)
        refused = False
    except G.GroupingRefused as e:
        refused = row[0] in str(e)
    check('a detection without its fitted statistical error REFUSES, named (no median fill)', refused)
    check('the 10-minute frame flag P3_TENMIN_PERTIE is restored after every load', os.environ.get('P3_TENMIN_PERTIE') == _ENV0,
          (os.environ.get('P3_TENMIN_PERTIE'), _ENV0))


def main(argv):
    quick = '--quick' in argv
    only = None
    if '--only' in argv:
        only = {tuple(x.split('_')) for x in argv[argv.index('--only') + 1].split(',')}
    shutil.rmtree(T, ignore_errors=True); os.makedirs(T)
    try:
        if not check_frozen():
            print('frozen inputs changed - nothing else checked'); return 1
        check_record_logic()
        det10 = None
        for tier, S in DATASETS:
            if only and (tier, S) not in only:
                continue
            det, ident, fork, z, meta = run_dataset(tier, S, quick)
            if det10 is None and tier == '10min':
                det10 = det
            check_ids(tier, S, det, ident, fork)
            if PLAN[(tier, S)]['incremental']:
                run_incremental(tier, S, det, z, meta)
            if (tier, S) == ('10min', 'V'):
                run_reused(tier, S, det)
                run_tolerance(tier, S, det)
        if det10 is not None:
            check_config_and_refusals(det10)
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
