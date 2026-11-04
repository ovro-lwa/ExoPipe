"""THE v3 ANALYSIS of family W (stage wide_test) and its publication (stage wide_publish): WHICH CATALOGED SOURCE, IF ANY,
MAKES A FORCED TARGET'S EVENTS? Port of phase3/refactor_analysis/review_items_20261031/forced_wide/v3/forced_wide_v3.py
(2026-11-02/03), with the parts of v2 (forced_wide_analyze.py) its centering test calls, and v3/publish_v3_to_runs.py.
Gregg 2026-11-03: "v3 is not perfect, but it is approved. Proceed." The method (forced_wide_v3.py docstring; values in config
forced_wide_test), per (Stokes, tier, target), on the wide-area bursts (wide_search; the file / tier screen of stage 2 applied:
zone 1 out, -1 clean):
  BURSTS      Stokes I: positive peaks (what forced photometry counts); Stokes V: either sign.
  CANDIDATES  every physical cataloged source (>= S_MIN_JY at 74 MHz) within R_CAND FWHM of the target in a band with events;
              band flux F = S74 (REF_MHZ / nu)^SPECTRAL_INDEX.
  AMPLITUDE   a source can make an event of value v only if v <= MU x F. CAPABLE in a band = able to make at least
              CAPABLE_FRACTION of the target's events in that band; a candidate is tested in its capable bands only.
  TAPER TEST  (a capable candidate at >= NEAR FWHM): the bursts in a strip +-STRIP about the line candidate -> target, binned
              every BIN; the near and far windows (SIDE), (count + 0.5) / area, joined log-linearly and integrated over the
              target's disc (DISC) -> n_pred; n_obs = the bursts in the disc; summed over the bands. CALIBRATED on CONTROL LINES
              (cataloged sources of CTL_FLUX x the candidate's flux in every job of the same Stokes, imaginary targets at its
              distance in CTL_DIRS directions): kappa = sum n_obs / sum n_pred over the matched lines, lambda = kappa n_pred,
              ratio = (n_obs + 0.5) / (lambda + 0.5), p = (lines with a ratio >= the target's + 1) / (lines + 1). The taper
              ACCOUNTS for the count when p >= P_CONSISTENT and ratio <= RATIO_CONSISTENT; it FALLS SHORT when p <= P_EXCESS and
              ratio >= R_EXCESS.
  CENTERING   (a capable candidate within NEAR): v2's test - where the bursts in the pair's region center on the line C -> T
              against pooled flux-matched controls - with this candidate as the background.
  CENTROID    where the bursts within the target's disc center: mean offset, error sqrt(stat^2 + floor^2), z = offset / error;
              z_c = the centroid's distance from a candidate in the same error.
  VERDICT     fewer than MIN_NOBS bursts at the target -> UNDECIDED; evidence that a candidate makes them (the bursts center on
              it, v2's centering, night by night, its taper accounts for the count); z >= Z_OFF: BACKGROUND SOURCE (named) /
              NOT CENTERED ON THE TARGET (source not identified); z < Z_ON: UNDECIDED (centered on the target and on a cataloged
              source alike) / NOT EXPLAINED BY A CATALOGED SOURCE (centered on the target) / UNDECIDED; in between: BACKGROUND
              SOURCE / CONSISTENT WITH A BACKGROUND SOURCE / UNDECIDED; no capable candidate: NOT CENTERED ON / NOT EXPLAINED
              BY ... (no cataloged source can make the events). Neither NOT EXPLAINED nor NOT CENTERED names the target as the
              emitter: other causes stay open.
Every computation is the scripts' own, in their order: the results do not depend on the number of processes. The defaults
differ from the approved scripts where the refactor chat decided under Gregg's delegation (production's values in the parity
config, which reproduces the served files byte for byte):
  - v2's centering uses EVERY eligible control pair (forced_wide_test.V2_CTL_DRAW None) where the script drew 300 at random with
    a fixed seed - a draw over a list in pool order, which any change to the job pool re-draws (review F2, 2026-11-04);
  - the controls are ORDER-INDEPENDENT (CONTROL_ORDER 'canonical'; R-REFACTOR-20261104-FORCED-WIDE-CONTROLS-ORDER-INDEPENDENT):
    the control lines' pool, rotations and order keyed on stable identities (control_lines_canonical), v2's control pairs with
    canonical host ties, so adding or moving a job changes another target's controls only where that job adds controls to its
    pool;
  - PRECISION and its KNIFE-EDGE FLAG (the amendment ...-AMENDED-PRECISION-KNIFE-EDGE): every eligible control line
    (TAPER_CTL_MAX None, where the script kept 900); the centroid's bootstrap drawn per (target, tier) from a stable seed
    (BOOT_DRAW 'target') until the Monte Carlo error of its z is at most BOOT_Z_SE; and each row's knife_edge column naming
    every statistic that decides its verdict or named sources within KNIFE_EDGE_SE Monte Carlo errors of its threshold
    (knife_edges).
The figures and the review page of the analysis (figs_v3/, forced_wide_page_v3.py) are not ported: nothing of the pipeline
reads them."""
import collections
import csv
import hashlib
import math
import os

import numpy as np

TIERS = ('10s', '1min')
BANDS = ('18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz', '64MHz', '69MHz', '73MHz',
         '78MHz', '82MHz')
VERDICTS = ('BACKGROUND SOURCE', 'CONSISTENT WITH A BACKGROUND SOURCE',
            'NOT CENTERED ON THE TARGET (source not identified)',
            'NOT CENTERED ON THE TARGET (no cataloged source can make the events)',
            'NOT EXPLAINED BY A CATALOGED SOURCE (centered on the target)',
            'NOT EXPLAINED BY A CATALOGED SOURCE (no cataloged source can make the events)',
            'UNDECIDED', 'UNDECIDED (centered on the target and on a cataloged source alike)',
            'UNDECIDED (fewer than 3 bursts at the target)', 'UNDECIDED (fewer than 3 events)')
V2_VERDICTS = ('BACKGROUND SOURCE', 'NOT THE BACKGROUND SOURCE', 'UNDECIDED', 'UNDECIDED (cannot separate)', 'too few nights',
               'too few controls', 'no bursts in the region')
_FW = {}


def tc():
    from exopipe.phase3.config import CFG
    return CFG.forced_wide_test


def fw_table():
    """{band: snapshot frame FWHM arcmin} (phase3_snapshot_astro.beam_fwhm_arcmin, in place)."""
    if not _FW:
        from exopipe.phase3.lib.forced.wide_jobs import beam_fwhm_arcmin
        _FW.update({b: beam_fwhm_arcmin(b) for b in BANDS})
    return _FW


def band_flux(s74, band):
    c = tc()
    return s74 * (c.REF_MHZ / float(band[:-3])) ** c.SPECTRAL_INDEX


def _clean_sets(z):
    """the (cell, tier) of a burst file's CLEAN files (status ok, zone not 1, gated planes) and their plane counts per tier"""
    clean = set()
    nfr = collections.Counter()
    for c_, t, n, zn, s in zip(z['f_cell'], z['f_tier'], z['f_planes'], z['f_zone'], z['f_status']):
        if s == 'ok' and zn != 1 and n > 0:
            clean.add((str(c_), str(t)))
            nfr[str(t)] += int(n)
    return clean, nfr


def burst_path(bdir, stokes, identity, band):
    return os.path.join(bdir, f'{stokes}_{identity}_{band}.npz')


# ---------------------------------------------------------------------------------------------------------------------
# v3 (forced_wide_v3.py)
# ---------------------------------------------------------------------------------------------------------------------

class JobData:
    """one (Stokes, target) job's clean bursts per (band, tier), in the target's tangent plane (arcmin), with KD-trees"""

    def __init__(self, job, bdir):
        from scipy.spatial import cKDTree
        from exopipe.phase3.lib.forced.wide_jobs import tan_off, physical_sources
        c = tc()
        FW = fw_table()
        self.job = job
        self.st = job['stokes']
        self.T = tuple(job['T'])
        self.bands = list(job['bands'])
        self.b = {}
        for band in self.bands:
            p = burst_path(bdir, self.st, job['identity'], band)
            if not os.path.exists(p):
                continue
            z = np.load(p)
            clean, nfr = _clean_sets(z)
            keep = np.array([(c_, t) in clean for c_, t in zip(z['cell'], z['tier'])], bool)
            if self.st == 'I':
                keep &= z['snr'] > 0
            for tier in TIERS:
                m = keep & (z['tier'] == tier)
                e, n = tan_off(z['ra'][m], z['dec'][m], self.T[0], self.T[1])
                e = np.atleast_1d(np.asarray(e, float))
                n = np.atleast_1d(np.asarray(n, float))
                self.b[(band, tier)] = dict(e=e, n=n, cell=z['cell'][m], ts=z['ts'][m], snr=z['snr'][m], value=z['value'][m],
                                            tree=cKDTree(np.c_[e, n]) if m.sum() else None, frames=nfr[tier])
        self.srcs = physical_sources(self.T[0], self.T[1], c.CUT_HALF_ARCMIN + 2 * max(FW.values()), c.S_MIN_JY)
        self.sxy = np.array([tan_off(s[0], s[1], self.T[0], self.T[1]) for s in self.srcs]) if self.srcs else np.zeros((0, 2))

    def has(self, band, tier):
        d = self.b.get((band, tier))
        return d is not None and d['frames'] > 0


def line_counts(D, C, P, fw):
    """the taper test's counts for one band: C, P = (e, n) of the candidate and the (real or imaginary) target in this cutout's
    plane. Returns (n_obs, side bins [(a_center, count, area_FW2)], a_T) or None when the target's disc or both sides leave the
    cutout"""
    c = tc()
    CUT, EDGE, DISC, STRIP, BIN, SIDE = c.CUT_HALF_ARCMIN, c.CUT_EDGE_ARCMIN, c.DISC, c.STRIP, c.BIN, c.SIDE
    vx, vy = P[0] - C[0], P[1] - C[1]
    L = math.hypot(vx, vy)
    if L <= 0:
        return None
    ux, uy = vx / L, vy / L
    aT = L / fw
    # inside the cutout: the disc and the far side
    far = (C[0] + ux * (L + SIDE[1] * fw), C[1] + uy * (L + SIDE[1] * fw))
    if max(abs(P[0]), abs(P[1])) + DISC * fw > CUT - EDGE or max(abs(far[0]), abs(far[1])) > CUT - EDGE:
        return None
    n_obs = 0
    bins = []
    if D['tree'] is not None:
        n_obs = len(D['tree'].query_ball_point([P[0], P[1]], DISC * fw))
        # the bins' pre-selection: every bin lies within sqrt(SIDE[1]^2 + STRIP^2) < SIDE[1] + STRIP of P, so the 0.2 FWHM margin
        # of the script never changes a count
        idx = np.array(D['tree'].query_ball_point([P[0], P[1]], (SIDE[1] + 0.2) * fw + STRIP * fw), int)
    else:
        idx = np.zeros(0, int)
    x = D['e'][idx] - C[0]
    y = D['n'][idx] - C[1]
    al = (x * ux + y * uy) / fw
    ac = (-x * uy + y * ux) / fw
    dP = np.hypot(D['e'][idx] - P[0], D['n'][idx] - P[1]) / fw
    sel = (np.abs(ac) < STRIP) & (dP >= DISC)
    for lo in np.r_[np.arange(aT - SIDE[1], aT - SIDE[0] - 1e-9, BIN), np.arange(aT + SIDE[0], aT + SIDE[1] - 1e-9, BIN)]:
        hi = lo + BIN
        if lo < c.MIN_BIN:                           # never the candidate's own position
            continue
        mid = (C[0] + ux * (lo + BIN / 2) * fw, C[1] + uy * (lo + BIN / 2) * fw)
        if max(abs(mid[0]), abs(mid[1])) > CUT - EDGE:
            continue
        cnt = int(np.sum(sel & (al >= lo) & (al < hi)))
        bins.append((lo + BIN / 2, cnt, BIN * 2 * STRIP))
    near_ok = any(a < aT for a, _c, _ar in bins)
    far_ok = any(a > aT for a, _c, _ar in bins)
    if not (near_ok and far_ok):
        return None
    return n_obs, bins, aT


_DISC_A = {}


def disc_profile():
    """sample offsets along the line over the unit disc of radius DISC (FWHM units), equal-area weights"""
    c = tc()
    key = (c.DISC, c.DISC_GRID)
    if key not in _DISC_A:
        DISC = c.DISC
        g = np.linspace(-DISC, DISC, c.DISC_GRID)
        X, Y = np.meshgrid(g, g)
        m = X ** 2 + Y ** 2 <= DISC ** 2
        _DISC_A[key] = (X[m], math.pi * DISC ** 2 / m.sum())
    return _DISC_A[key]


def fit_predict(bins, aT):
    """LOCAL interpolation across the target's disc: the density of the near window and of the far window, each (count + 0.5)
    / area, joined log-linearly through their centers and integrated over the disc; the variance from the windows' Poisson
    errors"""
    a = np.array([b[0] for b in bins]) - aT
    c = np.array([b[1] for b in bins], float)
    ar = np.array([b[2] for b in bins])
    wn = a < 0
    wf = a > 0
    if not wn.any() or not wf.any():
        return float('nan'), float('nan'), 'one side missing'
    nn, nf = c[wn].sum() + 0.5, c[wf].sum() + 0.5
    An, Af = ar[wn].sum(), ar[wf].sum()
    xn = float(np.sum(a[wn] * ar[wn]) / An)
    xf = float(np.sum(a[wf] * ar[wf]) / Af)
    rn, rf = nn / An, nf / Af
    beta = math.log(rf / rn) / (xf - xn)
    alpha = math.log(rn) - beta * xn
    xs, w = disc_profile()
    f = np.exp(alpha + beta * xs) * w
    npred = float(f.sum())
    # d npred / d ln(rn), d ln(rf): the disc integral of exp(alpha + beta x) with alpha, beta linear in ln rn, ln rf
    g_n = float(np.sum(f * (xf - xs) / (xf - xn)))
    g_f = float(np.sum(f * (xs - xn) / (xf - xn)))
    var = g_n ** 2 / nn + g_f ** 2 / nf
    return npred, var, f'windows {int(c[wn].sum())} / {int(c[wf].sum())}'


def taper_test(J, C, P, bands, tier):
    """summed over bands: n_obs, n_pred, var, per-band rows; C, P in J's plane"""
    FW = fw_table()
    tot_o = 0
    tot_p = 0.0
    tot_v = 0.0
    rows = []
    for band in bands:
        if not J.has(band, tier):
            continue
        fw = FW[band]
        r = line_counts(J.b[(band, tier)], C, P, fw)
        if r is None:
            continue
        n_obs, bins, aT = r
        npred, var, how = fit_predict(bins, aT)
        tot_o += n_obs
        tot_p += npred
        tot_v += var
        rows.append(dict(band=band, aT=aT, n_obs=n_obs, n_pred=npred, var=var, how=how, bins=bins))
    if not rows:
        return None
    z = (tot_o - tot_p) / math.sqrt(max(tot_p + tot_v, 1e-9)) if (tot_p + tot_v) > 0 else float('nan')
    return dict(n_obs=tot_o, n_pred=tot_p, var=tot_v, z=z, bands=rows)


def candidates(J, ev, tier):
    """the target's candidate sources: [dict(src, xy, d, s74, bands_far, bands_near, reach)] for those within R_CAND of it in a
    band with events (reach: F, the fraction of events it can make, the median event / F)"""
    c = tc()
    FW = fw_table()
    vals = collections.defaultdict(list)
    for band, ts, v in ev:
        vals[band].append(abs(v))            # Stokes V events can be negative: the amplitude is |V|
    mu = c.MU[(J.st, tier)]
    out = []
    for k, s in enumerate(J.srcs):
        x, y = J.sxy[k]
        d = math.hypot(x, y)
        far, near, reach = [], [], {}
        for band, vv in vals.items():
            if band not in FW or d > c.R_CAND * FW[band] or not J.has(band, tier):
                continue
            F = band_flux(s[2], band)
            frac = float(np.mean(np.array(vv) <= mu * F))
            reach[band] = (round(F, 2), round(frac, 2), round(float(np.median(vv)) / F, 2))
            if frac >= c.CAPABLE_FRACTION:
                (far if d >= c.NEAR * FW[band] else near).append(band)
        if reach:
            out.append(dict(src=s, xy=(float(x), float(y)), d=d, s74=s[2], name=s[3], bands_far=far, bands_near=near, reach=reach))
    return out


def stable_uniform(*parts):
    """a number in [0, 1) from a stable hash (sha256) of the parts: the same in every process, on every node and for every job
    list (Python's hash() is salted per process)"""
    h = hashlib.sha256('|'.join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], 'big') / 18446744073709551616.0


def stable_int(*parts):
    """a 64-bit integer from the same stable hash (a generator's seed)"""
    h = hashlib.sha256('|'.join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], 'big')


def job_id(J):
    """a job's stable identity: Stokes:identity (JobData and V2Job alike)"""
    return '%s:%s' % (J.job['stokes'], J.job['identity'])


def source_id(s):
    """a cataloged physical source's stable identity: its name and position (physical_sources' tuple)"""
    return '%s|%.5f|%.5f' % (s[3], s[0], s[1])


def control_order():
    """forced_wide_test.CONTROL_ORDER: 'canonical' or 'pool'; anything else raises ValueError (the stage refuses it)."""
    o = tc().CONTROL_ORDER
    if o not in ('canonical', 'pool'):
        raise ValueError('forced_wide_test.CONTROL_ORDER %r: canonical or pool' % (o,))
    return o


def taper_cap():
    """forced_wide_test.TAPER_CTL_MAX: None (every eligible control line) or an int >= 1; anything else raises ValueError (the
    stage refuses it)."""
    v = tc().TAPER_CTL_MAX
    if v is None or (isinstance(v, int) and not isinstance(v, bool) and v >= 1):
        return v
    raise ValueError('forced_wide_test.TAPER_CTL_MAX %r: None (every eligible line) or an int >= 1' % (v,))


def mc_settings():
    """the Monte Carlo keys, checked: (BOOT_DRAW, BOOT_Z_SE, BOOT_MAX, KNIFE_EDGE_SE); a bad value raises ValueError (the stage
    refuses it before any job runs)."""
    c = tc()
    taper_cap()
    if c.BOOT_DRAW not in ('target', 'job'):
        raise ValueError('forced_wide_test.BOOT_DRAW %r: target or job' % (c.BOOT_DRAW,))
    for k in ('BOOT_Z_SE', 'KNIFE_EDGE_SE'):
        v = getattr(c, k)
        if v is not None and not (isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0):
            raise ValueError('forced_wide_test.%s %r: None or a number > 0' % (k, v))
    if not (isinstance(c.BOOTSTRAP, int) and isinstance(c.BOOT_MAX, int) and 2 <= c.BOOTSTRAP <= c.BOOT_MAX):
        raise ValueError('forced_wide_test.BOOTSTRAP %r / BOOT_MAX %r: ints, 2 <= BOOTSTRAP <= BOOT_MAX'
                         % (c.BOOTSTRAP, c.BOOT_MAX))
    return c.BOOT_DRAW, c.BOOT_Z_SE, c.BOOT_MAX, c.KNIFE_EDGE_SE


def _control_sources(pool, J0, cand, tier):
    """every eligible (host job, control source) of a candidate's control lines, in pool order: (J, nb, s, x, y, key, score) -
    a cataloged source of CTL_FLUX x the candidate's flux (>= CTL_MIN_JY) inside a job's cutout (CUT_EDGE_ARCMIN), the job
    having searched >= half the candidate's bands; key = its CTL_KEY_PER_DEG cell, score = (bands searched, -its distance
    from the cutout center)"""
    c = tc()
    CUT, EDGE = c.CUT_HALF_ARCMIN, c.CUT_EDGE_ARCMIN
    bands = cand['bands_far']
    for J in pool:
        if J is J0:
            continue
        nb = [b for b in bands if J.has(b, tier)]
        if len(nb) < math.ceil(len(bands) / 2):
            continue
        for k, s in enumerate(J.srcs):
            if not (c.CTL_FLUX[0] * cand['s74'] <= s[2] <= c.CTL_FLUX[1] * cand['s74']) or s[2] < c.CTL_MIN_JY:
                continue
            x, y = J.sxy[k]
            if max(abs(x), abs(y)) > CUT - EDGE:
                continue
            key = (round(s[0] * c.CTL_KEY_PER_DEG) / c.CTL_KEY_PER_DEG, round(s[1] * c.CTL_KEY_PER_DEG) / c.CTL_KEY_PER_DEG)
            yield J, nb, s, x, y, key, (len(nb), -max(abs(x), abs(y)))


def _bright1(J, cache):
    """the >= CTL_BRIGHT_JY sources of a job's cutout (offsets): a control line keeps NEAR x the smallest FWHM from them"""
    if J not in cache:
        c = tc()
        cache[J] = np.array([J.sxy[i] for i, s in enumerate(J.srcs) if s[2] >= c.CTL_BRIGHT_JY]) if J.srcs else np.zeros((0, 2))
    return cache[J]


def _line(J, C, P, nb, tier, b1, fwmin, fwmax):
    """one control line at the imaginary target P, or None where it is refused (too near the cutout's own target or a bright
    source; the taper test undefined there)"""
    c = tc()
    if math.hypot(*P) < c.CTL_TARGET_FWHM * fwmax:
        return None
    if len(b1) and np.min(np.hypot(b1[:, 0] - P[0], b1[:, 1] - P[1])) < c.NEAR * fwmin:
        return None
    r = taper_test(J, C, P, nb, tier)
    if r is None or not np.isfinite(r['z']):
        return None
    return r['z'], r['n_obs'], r['n_pred'], r['var']


def control_lines(pool, J0, cand, tier, rng):
    """z of the taper test on control lines (module notes) for one candidate of job J0: [(z, n_obs, n_pred, var)], in the
    order of forced_wide_test.CONTROL_ORDER ('canonical': order-independent, the job's generator untouched, each line with its
    control source's identity as a 5th member - the Monte Carlo errors' clusters; 'pool': production's, consuming ``rng``)"""
    if control_order() == 'canonical':
        zs, prov = control_lines_canonical(pool, J0, cand, tier)
        return [q + (p_[0],) for q, p_ in zip(zs, prov)]
    return control_lines_pool(pool, J0, cand, tier, rng)


def control_lines_pool(pool, J0, cand, tier, rng):
    """production's control lines (forced_wide_v3.py control_lines): one source per cell, the first job in pool order winning a
    tie, the list shuffled by the job's generator, each source's CTL_DIRS lines rotated by one draw of it, up to TAPER_CTL_MAX
    (production: 900)"""
    c = tc()
    cap = taper_cap()
    FW = fw_table()
    bands = cand['bands_far']
    d = cand['d']
    fwmin, fwmax = min(FW[b] for b in bands), max(FW[b] for b in bands)
    best = {}
    for J, nb, s, x, y, key, score in _control_sources(pool, J0, cand, tier):
        if key not in best or score > best[key][0]:
            best[key] = (score, J, (float(x), float(y)), nb)
    ents = [v[1:] for v in best.values()]
    rng.shuffle(ents)
    zs = []
    bright1 = {}
    for J, C, nb in ents:
        b1 = _bright1(J, bright1)
        for th in np.radians(np.arange(0, 360, 360 / c.CTL_DIRS)) + rng.uniform(0, 2 * math.pi / c.CTL_DIRS):
            r = _line(J, C, (C[0] + d * math.sin(th), C[1] + d * math.cos(th)), nb, tier, b1, fwmin, fwmax)
            if r is None:
                continue
            zs.append(r)
            if cap is not None and len(zs) >= cap:
                return zs
    return zs


def control_lines_canonical(pool, J0, cand, tier):
    """ORDER-INDEPENDENT control lines (CONTROL_ORDER 'canonical'; R-REFACTOR-20261104-FORCED-WIDE-CONTROLS-ORDER-INDEPENDENT):
    one source per cell, its host the job with the best score, a tie to the smallest (job identity, source identity); the
    sources in the order of a stable hash of (SEED, the target's identity, the source's identity) - the seeded selection a
    TAPER_CTL_MAX cap applies; line i of a source at (i + u_i) x 2 pi / CTL_DIRS, u_i a stable hash of (SEED, target, source, i).
    Returns ([(z, n_obs, n_pred, var)], [(source identity, host job identity, i)]) - the lines and where each came from."""
    c = tc()
    cap = taper_cap()
    FW = fw_table()
    bands = cand['bands_far']
    d = cand['d']
    fwmin, fwmax = min(FW[b] for b in bands), max(FW[b] for b in bands)
    best = {}
    for J, nb, s, x, y, key, score in _control_sources(pool, J0, cand, tier):
        tie = (job_id(J), source_id(s))
        cur = best.get(key)
        if cur is None or score > cur[0] or (score == cur[0] and tie < cur[4]):
            best[key] = (score, J, (float(x), float(y)), nb, tie)
    tid = job_id(J0)
    ents = sorted(((stable_uniform('rank', c.SEED, tid, v[4][1]), v[4][1], v[1], v[2], v[3]) for v in best.values()),
                  key=lambda e: (e[0], e[1]))
    zs, prov = [], []
    bright1 = {}
    step = 2 * math.pi / c.CTL_DIRS
    for _rank, sid, J, C, nb in ents:
        b1 = _bright1(J, bright1)
        for i in range(c.CTL_DIRS):
            th = (i + stable_uniform('rot', c.SEED, tid, sid, i)) * step
            r = _line(J, C, (C[0] + d * math.sin(th), C[1] + d * math.cos(th)), nb, tier, b1, fwmin, fwmax)
            if r is None:
                continue
            zs.append(r)
            prov.append((sid, job_id(J), i))
            if cap is not None and len(zs) >= cap:
                return zs, prov
    return zs, prov


def matched(zs, npred):
    """the control lines with a comparable prediction (module notes). Returns [(z, n_obs, n_pred, var)]"""
    c = tc()
    sel = []
    for k in c.MATCH_FACTORS:
        lo, hi = max(npred / k, c.MATCH_FLOORS[0]), max(npred * k, c.MATCH_FLOORS[1])
        sel = [q for q in zs if lo <= q[2] <= hi]
        if len(sel) >= c.MATCH_MIN:
            return sel
    return sel


def match_windows(zs, npred):
    """[(factor, the lines in its window)] for each of MATCH_FACTORS (matched() takes the first holding >= MATCH_MIN, else the
    last)"""
    c = tc()
    out = []
    for k in c.MATCH_FACTORS:
        lo, hi = max(npred / k, c.MATCH_FLOORS[0]), max(npred * k, c.MATCH_FLOORS[1])
        out.append((k, [q for q in zs if lo <= q[2] <= hi]))
    return out


def _within_var(vals, labels, per_line):
    """the variance of a sum over lines when only each line's position within its control source is re-drawn: the sum over
    sources of the within-source sum of squares x m / (m - 1); a source with one line contributes ``per_line``"""
    groups = collections.defaultdict(list)
    for v, l in zip(vals, labels):
        groups[l].append(v)
    tot = 0.0
    for g in groups.values():
        m = len(g)
        if m < 2:
            tot += per_line
        else:
            a = np.asarray(g, float)
            tot += float(np.sum((a - a.mean()) ** 2)) * m / (m - 1)
    return tot


def calibrate(tt, ctl, min_n=None):
    """the target's count against matched CONTROL LINES (module notes): kappa, lambda, ratio, the permutation p-value; also the
    negative-binomial p with the controls' extra-Poisson scatter, for the record. With the MONTE CARLO errors of p and the
    ratio (the knife-edge flag): p_se_line = the binomial sqrt(p (1 - p) / n) over the n matched lines; ratio_se_line = ratio x
    lambda / (lambda + 0.5) x SE(kappa) / kappa, SE(kappa) = sqrt(n / (n - 1) sum (n_obs - kappa n_pred)^2) / (sum n_pred +
    0.5) (the ratio estimator's spread over the matched lines); *_source = the same with only the within-source spread (when
    the lines carry their control source, CONTROL_ORDER canonical) - a zero within-source spread (no control source with lines
    on both sides, e.g. no line at or above the target's ratio) is degenerate, not a zero error: the *_line value then. min_n:
    the fewest lines calibrated (None: MATCH_MIN)."""
    import scipy.stats as SSTAT
    nan = float('nan')
    if len(ctl) < (tc().MATCH_MIN if min_n is None else max(int(min_n), 1)):
        return dict(ctl=(len(ctl), nan, nan), kappa=nan, phi=nan, lam=nan, lam_err=nan, ratio=nan, p_emp=nan, z_emp=nan, p_nb=nan)
    no = np.array([q[1] for q in ctl], float)
    npd = np.array([q[2] for q in ctl], float)
    vv = np.array([q[3] for q in ctl], float)
    kappa = (no.sum() + 0.5) / (npd.sum() + 0.5)
    lam_c = kappa * npd
    rc = (no + 0.5) / (lam_c + 0.5)
    lam = kappa * tt['n_pred']
    rt = (tt['n_obs'] + 0.5) / (lam + 0.5)
    p_emp = (int(np.sum(rc >= rt)) + 1) / (len(rc) + 1)
    phi = max(0.0, float(np.sum((no - lam_c) ** 2 - lam_c - kappa ** 2 * vv)) / max(float(np.sum(lam_c ** 2)), 1e-9))
    V = kappa ** 2 * tt['var'] + phi * lam ** 2
    n = int(tt['n_obs'])
    if lam <= 0:
        p_nb = 0.0 if n > 0 else 1.0
    elif V <= 1e-9:
        p_nb = float(SSTAT.poisson.sf(n - 1, lam))
    else:
        r = lam ** 2 / V
        q = r / (r + lam)
        p_nb = float(SSTAT.nbinom.sf(n - 1, r, q))
    out = dict(ctl=(len(ctl), round(kappa, 3), round(phi, 3)), kappa=kappa, phi=phi, lam=lam, lam_err=math.sqrt(max(V, 0.0)),
               ratio=rt, ctl_ratio_q=(float(np.percentile(rc, 50)), float(np.percentile(rc, 95)), float(rc.max())),
               p_emp=p_emp, z_emp=float(SSTAT.norm.isf(p_emp)), p_nb=p_nb)
    # the Monte Carlo errors (nothing above reads them)
    m = len(rc)
    ind = (rc >= rt).astype(float)
    d = no - kappa * npd
    den = npd.sum() + 0.5
    var_k = float(np.sum(d ** 2)) * m / max(m - 1, 1)
    gain = rt * lam / (lam + 0.5) / kappa if kappa > 0 else nan
    out.update(p_se_line=math.sqrt(p_emp * (1.0 - p_emp) / m), ratio_se_line=gain * math.sqrt(var_k) / den)
    if all(len(q) > 4 for q in ctl):
        lab = [q[4] for q in ctl]
        vp_, vk_ = _within_var(ind, lab, p_emp * (1.0 - p_emp)), _within_var(d, lab, var_k / m)
        out.update(p_se_source=math.sqrt(vp_) / m if vp_ > 0 else out['p_se_line'],
                   ratio_se_source=gain * math.sqrt(vk_) / den if vk_ > 0 else out['ratio_se_line'])
    return out


def per_night(J, C, tier, bands):
    """Spearman over nights: the target's bursts (disc) against C's spread (NIGHT_FAR / within NIGHT_CORE), nights with
    >= NIGHT_MIN_CORE at C"""
    import scipy.stats as SSTAT
    c = tc()
    FW = fw_table()
    tn, core, far = collections.Counter(), collections.Counter(), collections.Counter()
    for band in bands:
        if not J.has(band, tier):
            continue
        D = J.b[(band, tier)]
        fw = FW[band]
        if not len(D['e']):
            continue
        dT = np.hypot(D['e'], D['n']) / fw
        dC = np.hypot(D['e'] - C[0], D['n'] - C[1]) / fw
        night = np.array([c_[4:] for c_ in D['cell']])
        for nt in night[dT < c.DISC]:
            tn[nt] += 1
        for nt in night[dC < c.NIGHT_CORE]:
            core[nt] += 1
        for nt in night[(dC > c.NIGHT_FAR[0]) & (dC < c.NIGHT_FAR[1]) & (dT > c.NIGHT_CLEAR)]:
            far[nt] += 1
    nts = [n for n in core if core[n] >= c.NIGHT_MIN_CORE]
    if len(nts) < c.NIGHT_MIN_NIGHTS or sum(tn[n] for n in nts) < c.MIN_NOBS:
        return float('nan'), float('nan'), len(nts)
    x = [far[n] / core[n] for n in nts]
    y = [tn[n] for n in nts]
    rho, p = SSTAT.spearmanr(x, y)
    return float(rho), float(p), len(nts)


def v2_centering(J, cand, tier, v2pool, bdir):
    """v2's centering test with this candidate as the background, in its near capable bands"""
    T = J.T
    S = tuple(cand['src'][:2])
    jd = dict(J.job, S=list(S), s74=cand['s74'], bg_name=cand['name'], sep_arcmin=cand['d'], bands=cand['bands_near'])
    Jx = V2Job(jd, bdir)
    if not Jx.ok:
        return None
    ctl = v2_controls_for(Jx, v2pool)
    k = tc().V2_K
    r = Jx.region(S, T, tier, k, Jx.bands)
    if r is None:
        return dict(verdict='no bursts in the region', n_cells=0)
    cres = [Jc.region(C, Tp, tier, k, bands) for (Jc, C, Tp, bands, cs) in ctl]
    v, z, pw, mu, sc, ncv, fbg, ftg = v2_verdict(r['t'], r['err'], r['n_cells'], cres)
    return dict(verdict=v, z=z, power=pw, t=r['t'], t_err=r['err'], n_cells=r['n_cells'], n_bursts=r['n_bursts'], n_ctl=ncv)


def analyze(J, events, pool, v2pool, bdir, trace=None):
    """(result rows, candidate rows) of one job, both tiers (forced_wide_v3.analyze; one generator per job). KNIFE_EDGE_SE set:
    the Monte Carlo errors and the knife-edge flag added (knife_edges). trace: a list given -> (tier, res, cen) of each tier
    appended (tools)"""
    c = tc()
    ke = c.KNIFE_EDGE_SE
    rng = np.random.default_rng(c.SEED)
    rows, crows = [], []
    for tier in TIERS:
        ev = events.get((J.st, tier), {}).get(J.job['identity'], [])
        ev = [e for e in ev if e[0] in J.bands]
        if not ev:
            continue
        cands = candidates(J, ev, tier)
        capable = [c_ for c_ in cands if c_['bands_far'] or c_['bands_near']]
        res = []
        for cd in capable:
            out = dict(cd)
            if cd['bands_far']:
                tt = taper_test(J, cd['xy'], (0.0, 0.0), cd['bands_far'], tier)
                out['taper'] = tt
                if tt is not None:
                    zs = control_lines(pool, J, cd, tier, rng)
                    out.update(calibrate(tt, matched(zs, tt['n_pred'])))
                    if ke is not None:
                        out['_windows'] = ([q[4] for q in zs] if all(len(q) > 4 for q in zs) else None, len(zs),
                                           match_windows(zs, tt['n_pred']))
            if cd['bands_near']:
                out['center'] = v2_centering(J, cd, tier, v2pool, bdir)
            out['night'] = per_night(J, cd['xy'], tier, cd['bands_far'] + cd['bands_near'])
            res.append(out)
        evb = sorted({e[0] for e in ev}, key=lambda b: float(b[:-3]))
        cen = centroid(J, tier, evb, rng, [r['xy'] for r in res])
        for r in res:
            r['z_c'] = z_to(cen, r) if cen else float('nan')
        row = verdict_row(J, tier, ev, cands, res, cen)
        if ke is not None:
            for r in res:
                r['z_c_se'] = r['z_c'] * cen['rel'] if cen else float('nan')
                mc_errors(r)
            ks = knife_edges(J, tier, ev, cands, res, cen, row)
            row.update(bootstrap_n=cen['boot_n'] if cen else '', centroid_z_se=round(cen['z_se'], 4) if cen else '',
                       knife_edge=knife_text(ks))
        rows.append(row)
        for r in res:
            crows.append(cand_row(J, tier, r))
        if trace is not None:
            trace.append((tier, res, cen))
        for r in res:
            r.pop('_windows', None)
    return rows, crows


# ---------------------------------------------------------------------------------------------------------------------
# the Monte Carlo errors and the knife-edge flag (the amendment R-REFACTOR-20261104-FORCED-WIDE-CONTROLS-ORDER-INDEPENDENT-
# AMENDED-PRECISION-KNIFE-EDGE (3))
# ---------------------------------------------------------------------------------------------------------------------

# THE ERRORS THE FLAG USES (2026-11-04, validated on real rows against 30 independent global salts - every line re-rotated,
# every bootstrap re-drawn: se_validate.py, 26 calibrated candidates and 56 matched-line counts of 5 rows): 'source' = the
# binomial / spread WITHIN each control source (a salt re-draws a line's direction, never its control source: the
# between-source spread is no Monte Carlo noise) - salt sd / error median 0.89 (p), 1.00 (ratio), 0.95 (counts), 23, 25 and 53
# of 26 / 26 / 56 within a factor of 1.5; 'line' = over all lines alike - median 0.77 / 0.99 / 0.84 (conservative: the misses
# overestimate, up to 2x). Lines without their source (CONTROL_ORDER 'pool') fall back to 'line'.
SE_KIND = 'source'


def _count_se(labels, sel, N):
    """the Monte Carlo error of a matched-line window's count: within control sources (SE_KIND 'source', the lines' labels
    known) sqrt(sum over sources c (m - c) / (m - 1)) - m a source's lines, c those in the window, a source of one line
    q (1 - q), q = n / N; else, or when that is zero (degenerate: no source with lines on both sides of the window), the
    binomial sqrt(n (1 - n / N))"""
    n = len(sel)
    if not N:
        return 0.0
    binom = math.sqrt(max(n * (1.0 - n / N), 0.0))
    if SE_KIND != 'source' or labels is None:
        return binom
    m = collections.Counter(labels)
    cw = collections.Counter(q[4] for q in sel)
    q_ = n / N
    var = sum(q_ * (1.0 - q_) if mj < 2 else cw.get(j, 0) * (mj - cw.get(j, 0)) / (mj - 1.0) for j, mj in m.items())
    return math.sqrt(var) if var > 0 else binom


def mc_errors(r):
    """a candidate's Monte Carlo errors for the flag: p_se, ratio_se (calibrate's, SE_KIND) and per matched-line window its
    count's error (_count_se; r['_counts'] = [(factor, n, se)])"""
    nan = float('nan')
    for k in ('p', 'ratio'):
        r[k + '_se'] = r.get('%s_se_%s' % (k, SE_KIND), r.get(k + '_se_line', nan))
    w = r.get('_windows')
    r['_counts'] = [(f, len(sel), _count_se(w[0], sel, w[1])) for f, sel in w[2]] if w else []
    if w:
        ch = _window_choice([n for _f, n, _s in r['_counts']])
        r['n_ctl_se'] = r['_counts'][ch if ch is not None else -1][2]


def _window_choice(counts):
    """the matched-line window matched() takes: the first holding >= MATCH_MIN lines, else None (not calibrated)"""
    m = tc().MATCH_MIN
    return next((i for i, n in enumerate(counts) if n >= m), None)


def _outcome(row):
    return row['verdict'], frozenset(row['named'].split('; ')) if row.get('named') else frozenset()


_CAL_KEYS = ('ctl', 'kappa', 'phi', 'lam', 'lam_err', 'ratio', 'ctl_ratio_q', 'p_emp', 'z_emp', 'p_nb')


def knife_edges(J, tier, ev, cands, res, cen, row):
    """[(statistic, value, threshold, Monte Carlo error, format)] - every statistic of the row within KNIFE_EDGE_SE errors of
    a threshold that DECIDES the row: set alone just below and just above that threshold (the others as they are), it changes
    the verdict or the set of named sources (verdict_row re-run). The statistics: the centroid z (Z_ON, Z_OFF); per candidate
    its z_c (Z_C_ON, Z_C_NEAR), taper p (P_CONSISTENT, P_EXCESS), taper ratio (RATIO_CONSISTENT, R_EXCESS) and the count of
    each matched-line window (MATCH_MIN: flipped, matched() takes another window - calibrated on its lines - or none)."""
    c = tc()
    k = c.KNIFE_EDGE_SE
    nan = float('nan')
    base = _outcome(row)
    out = []
    if cen is None:
        return out

    def flips(res2, cen2):
        return _outcome(verdict_row(J, tier, ev, cands, res2, cen2)) != base

    def near(v, t, se):
        return v == v and se == se and se > 0 and abs(v - t) <= k * se
    eps = 1e-9
    for t in sorted({c.Z_ON, c.Z_OFF}):
        if near(cen['z'], t, cen['z_se']) and any(flips(res, dict(cen, z=t + e)) for e in (-eps, eps)):
            out.append(('centroid z', cen['z'], t, cen['z_se'], '%.2f'))
    for i, r in enumerate(res):
        nm = r['name'].split(' ', 1)[-1]

        def sub(**kw):
            return res[:i] + [dict(r, **kw)] + res[i + 1:]
        for t in sorted({c.Z_C_ON, c.Z_C_NEAR}):
            if near(r.get('z_c', nan), t, r.get('z_c_se', nan)) and any(flips(sub(z_c=t + e), cen) for e in (-eps, eps)):
                out.append(('centroid z to %s' % nm, r['z_c'], t, r['z_c_se'], '%.2f'))
        p = r.get('p_emp', nan)
        if p is not None and p == p:
            for t in sorted({c.P_CONSISTENT, c.P_EXCESS}):
                if near(p, t, r['p_se']) and any(flips(sub(p_emp=t + e), cen) for e in (-eps, eps)):
                    out.append(('taper p of %s' % nm, p, t, r['p_se'], '%.3g'))
            for t in sorted({c.RATIO_CONSISTENT, c.R_EXCESS}):
                if near(r['ratio'], t, r['ratio_se']) and any(flips(sub(ratio=t + e), cen) for e in (-eps, eps)):
                    out.append(('taper ratio of %s' % nm, r['ratio'], t, r['ratio_se'], '%.2f'))
        cnts = r.get('_counts') or []
        if cnts:
            now = _window_choice([n for _f, n, _s in cnts])
            for j, (f, n, se) in enumerate(cnts):
                if not near(float(n), float(c.MATCH_MIN), se):
                    continue
                alt = [x[1] for x in cnts]
                alt[j] = c.MATCH_MIN - 1 if n >= c.MATCH_MIN else c.MATCH_MIN
                ch = _window_choice(alt)
                if ch == now:
                    continue
                if ch is None:              # not calibrated
                    cal = dict.fromkeys(_CAL_KEYS, nan)
                    cal['ctl'] = (cnts[-1][1], nan, nan)
                else:                       # calibrated on that window's lines, however few
                    cal = calibrate(r['taper'], r['_windows'][2][ch][1], min_n=1)
                    if cal['p_emp'] != cal['p_emp']:
                        continue
                if flips(sub(**cal), cen):
                    out.append(('matched control lines (within %gx) of %s' % (f, nm), float(n), float(c.MATCH_MIN), se, '%d'))
    return out


def knife_text(ks):
    """the knife_edge column: '' or 'knife-edge: <statistic> <value> vs <threshold>[ and <threshold>], Monte Carlo error
    <error>; ...' (a statistic deciding at both of its thresholds is named once)"""
    if not ks:
        return ''
    merged = []
    for lab, v, t, se, fmt in ks:
        if merged and merged[-1][0] == lab and merged[-1][1] == v:
            merged[-1][2].append(t)
        else:
            merged.append((lab, v, [t], se, fmt))
    return 'knife-edge: ' + '; '.join('%s %s vs %s, Monte Carlo error %.2g' % (lab, fmt % v, ' and '.join('%g' % t for t in ts), se)
                                      for lab, v, ts, se, fmt in merged)


def taper_ok(r):
    tt = r.get('taper')
    return bool(tt and tt['n_obs'] >= tc().MIN_NOBS and r.get('p_emp') == r.get('p_emp') and r.get('p_emp') is not None)


def explains(r):
    """CONSISTENT: this candidate's taper accounts for the target's bursts in its tested bands (p_emp >= P_CONSISTENT and the
    target's count at most RATIO_CONSISTENT x the background), or v2's centering says BACKGROUND SOURCE"""
    c = tc()
    a = taper_ok(r) and r['p_emp'] >= c.P_CONSISTENT and r['ratio'] <= c.RATIO_CONSISTENT
    ce = r.get('center')
    return a or bool(ce and ce.get('verdict') == 'BACKGROUND SOURCE')


def shown(r):
    """CLEAR: consistent AND shown independently - the target's bursts follow this source's spread night by night (Spearman
    rho > 0, p < P_NIGHT) - or v2's centering test (its own power rule) says BACKGROUND SOURCE"""
    ce = r.get('center')
    if ce and ce.get('verdict') == 'BACKGROUND SOURCE':
        return True
    rho, p, nn = r['night']
    return explains(r) and rho == rho and rho > 0 and p < tc().P_NIGHT


def excess(r):
    """the target's bursts exceed this candidate's taper (at least R_EXCESS x its background AND as high at <= P_EXCESS of the
    control lines) and its centering in every test it had"""
    c = tc()
    parts = []
    if r['bands_far']:
        parts.append(taper_ok(r) and r['p_emp'] <= c.P_EXCESS and r['ratio'] >= c.R_EXCESS)
    if r['bands_near']:
        ce = r.get('center')
        parts.append(bool(ce and ce.get('verdict') == 'NOT THE BACKGROUND SOURCE'))
    return bool(parts) and all(parts)


_BOOT_CHUNK = 50000                 # resamples drawn at once (BOOT_DRAW 'target'): bounds a block's memory, never its values
# the stop is at BOOT_Z_SE / _BOOT_MARGIN: the error is estimated from the resamples themselves, and stopping on the first
# estimate under the target left the realized error up to 13% above it (zrow_boot.py, 150 salts per real row, 2026-11-04)
_BOOT_MARGIN = 1.1


def centroid(J, tier, bands, rng, zc_xy=()):
    """WHERE DO THE BURSTS AT THE TARGET CENTER? The bursts within its disc, offsets in FWHM of their band: the mean offset, its
    error = sqrt(stat^2 + floor^2) - stat from a bootstrap over nights (>= BOOT_MIN_NIGHTS nights; else CEN_DET / sqrt(n)),
    floor = max(CEN_FLOOR, CEN_NIGHT / sqrt(nights)); z_off = |offset| / error. The bootstrap's MONTE CARLO error: every z of
    this error (z, each candidate's z_c) has the standard error z x rel, rel = SE(stat^2) / (2 err^2) (err^2 = stat^2 + floor^2;
    stat^2 = the mean of q_b = ((x_b - x)^2 + (y_b - y)^2) / 2 over the resamples b, SE(stat^2) = sd(q_b) / sqrt(B)).
    BOOT_DRAW 'job' draws the nights from ``rng`` replicate by replicate (production's draws); 'target' from a generator of its
    own per (target, tier), vectorized. BOOT_Z_SE: more resamples, in blocks, until max(z, each candidate's z_c at most
    Z_C_NEAR) x rel <= BOOT_Z_SE / _BOOT_MARGIN or BOOT_MAX are drawn; zc_xy = the candidates' (e, n) offsets (arcmin)."""
    c = tc()
    FW = fw_table()
    xs, ys, cells, ifw = [], [], [], []
    for b in bands:
        if not J.has(b, tier):
            continue
        D = J.b[(b, tier)]
        fw = FW[b]
        m = np.hypot(D['e'], D['n']) / fw < c.DISC
        xs += list(D['e'][m] / fw)
        ys += list(D['n'][m] / fw)
        cells += list(D['cell'][m])
        ifw += [1.0 / fw] * int(m.sum())
    n = len(xs)
    if n < c.MIN_NOBS:
        return None
    xs, ys, cells = np.array(xs), np.array(ys), np.array(cells)
    cx, cy = float(xs.mean()), float(ys.mean())
    uc = sorted(set(cells))
    nn = len(uc)
    floor = max(c.CEN_FLOOR, c.CEN_NIGHT / math.sqrt(nn))
    ifw_m = float(np.mean(ifw))
    boot_n, rel = 0, 0.0
    if nn >= c.BOOT_MIN_NIGHTS:
        idx = [np.where(cells == c_)[0] for c_ in uc]
        if c.BOOT_DRAW == 'job':
            def block(m):             # production's draws (forced_wide_v3.py:477-479), from the job's generator
                return np.array([(xs[ii].mean(), ys[ii].mean()) for ii in (np.concatenate([idx[k] for k in rng.integers(0, nn, nn)])
                                                                         for _ in range(m))])
        else:
            g = np.random.default_rng(stable_int('boot', c.SEED, job_id(J), tier))
            sx = np.array([xs[i].sum() for i in idx])
            sy = np.array([ys[i].sum() for i in idx])
            cnt = np.array([len(i) for i in idx], float)

            def block(m):             # the same resampling (nn nights with replacement), each night by its sums and count
                out = []
                for a in range(0, m, _BOOT_CHUNK):
                    b = min(m - a, _BOOT_CHUNK)
                    k = g.integers(0, nn, size=(b, nn))
                    W = np.bincount((np.arange(b)[:, None] * nn + k).ravel(), minlength=b * nn).reshape(b, nn).astype(float)
                    tot = W @ cnt
                    out.append(np.c_[(W @ sx) / tot, (W @ sy) / tot])
                return out[0] if len(out) == 1 else np.concatenate(out)
        bs = block(c.BOOTSTRAP)
        while True:
            boot_n = len(bs)
            stat = math.hypot(bs[:, 0].std(), bs[:, 1].std()) / math.sqrt(2)
            err = math.hypot(stat, floor)
            q = ((bs[:, 0] - bs[:, 0].mean()) ** 2 + (bs[:, 1] - bs[:, 1].mean()) ** 2) / 2.0
            rel = float(q.std(ddof=1)) / math.sqrt(boot_n) / (2.0 * err ** 2)
            if c.BOOT_Z_SE is None or boot_n >= c.BOOT_MAX:
                break
            zz = [math.hypot(cx, cy) / err] + [min(math.hypot(cx - x * ifw_m, cy - y * ifw_m) / err, c.Z_C_NEAR)
                                               for x, y in zc_xy]
            crit = max(zz) * rel
            goal = c.BOOT_Z_SE / _BOOT_MARGIN
            if crit <= goal:
                break
            need = math.ceil(boot_n * (crit / goal) ** 2)                    # SE ~ 1 / sqrt(B)
            more = max(c.BOOTSTRAP, -(-(need - boot_n) // c.BOOTSTRAP) * c.BOOTSTRAP)
            bs = np.concatenate([bs, block(min(more, c.BOOT_MAX - boot_n))])
    else:
        stat = c.CEN_DET / math.sqrt(n)
        err = math.hypot(stat, floor)
    z = math.hypot(cx, cy) / err
    return dict(n=n, nights=nn, x=cx, y=cy, off=math.hypot(cx, cy), err=err, z=z,
                pa=math.degrees(math.atan2(cx, cy)) % 360.0, ifw=ifw_m, boot_n=boot_n, rel=rel, z_se=z * rel)


def z_to(cen, cand):
    """how far the centroid lies from a candidate, in the centroid's error (the candidate in FWHM units of the bursts' bands)"""
    cx, cy = cand['xy'][0] * cen['ifw'], cand['xy'][1] * cen['ifw']
    return math.hypot(cen['x'] - cx, cen['y'] - cy) / cen['err']


def tested_bands(r):
    tt = r.get('taper')
    return set(b['band'] for b in tt['bands']) if tt else set()


def verdict_row(J, tier, ev, cands, res, cen):
    """the target's verdict (module notes, VERDICT)"""
    c = tc()
    FW = fw_table()
    DISC, MU = c.DISC, c.MU
    n_ev = len(ev)
    vals = np.array([e[2] for e in ev])
    evb = sorted({e[0] for e in ev}, key=lambda b: float(b[:-3]))
    per_band = {b: (len(J.b[(b, tier)]['tree'].query_ball_point([0, 0], DISC * FW[b]))
                    if J.has(b, tier) and J.b[(b, tier)]['tree'] is not None else 0) for b in evb}
    disc = sum(per_band.values())
    f = lambda x, n=3: round(x, n) if x == x else ''        # noqa: E731 - the script's formatter
    base = dict(stokes=J.st, identity=J.job['identity'], target=J.job['target'], tier=tier, n_events=n_ev,
                event_jy_median=round(float(np.median(np.abs(vals))), 2), bursts_at_target=disc,
                bursts_by_band=' '.join(f'{b}:{n}' for b, n in per_band.items() if n), n_candidates=len(cands), n_capable=len(res),
                centroid_fwhm=f(cen['off']) if cen else '', centroid_arcmin=f(cen['off'] / cen['ifw'], 1) if cen else '',
                centroid_pa=f(cen['pa'], 0) if cen else '', centroid_err_fwhm=f(cen['err']) if cen else '',
                centroid_z=f(cen['z'], 1) if cen else '', centroid_nights=cen['nights'] if cen else '',
                v2_background=J.job.get('bg_name', '').split(' ', 1)[-1] if J.job.get('bg_name') else '')
    cen_txt = (f"the {cen['n']} bursts at the target center {cen['off'] / cen['ifw']:.1f}' ({cen['off']:.2f} FWHM) toward PA "
               f"{cen['pa']:.0f} deg, {cen['z']:.1f} sigma from it ({cen['nights']} nights)") if cen else ''

    def cover(r):
        bb = tested_bands(r) | set(r['bands_near'])
        return sum(n for b, n in per_band.items() if b in bb) / disc if disc else 0.0

    def nm(rr):
        return '; '.join(f"{r['name'].split(' ', 1)[-1]} {r['s74']:.1f} Jy at {r['d']:.1f}'" for r in rr)
    share = [r['lam'] if r.get('taper') and r.get('lam') == r.get('lam') else (r['taper']['n_pred'] if r.get('taper') else 0.0)
             for r in res]
    top = max(share) if share else 0.0
    order = sorted(range(len(res)), key=lambda i: (-(round(4 * share[i] / top) if top > 0 else 0), -res[i]['s74']))
    ordered = [res[i] for i in order]
    if disc < c.MIN_NOBS or cen is None:
        v = 'UNDECIDED (fewer than 3 bursts at the target)' if n_ev >= c.MIN_NOBS else 'UNDECIDED (fewer than 3 events)'
        return dict(base, verdict=v, named='', coverage='', why=' | '.join(describe(r) for r in ordered[:2]))
    if not res:
        bright = max(cands, key=lambda c_: c_['s74'], default=None)
        lim = (f"brightest within 2 FWHM: {bright['name'].split(' ', 1)[-1]} {bright['s74']:.1f} Jy at {bright['d']:.1f}' (the "
               f"events need {min(q[2] for q in bright['reach'].values()):.1f}x its band flux; the largest seen is "
               f"{MU[(J.st, tier)]:g}x)" if bright else 'no cataloged source >= 0.25 Jy within 2 FWHM')
        if cen['z'] >= c.Z_OFF:
            v = 'NOT CENTERED ON THE TARGET (no cataloged source can make the events)'
        elif cen['z'] < c.Z_ON:
            v = 'NOT EXPLAINED BY A CATALOGED SOURCE (no cataloged source can make the events)'
        else:
            v = 'UNDECIDED'
        return dict(base, verdict=v, named='', coverage='', why=f'{cen_txt}; {lim}')

    # evidence that a candidate makes the bursts at the target
    def evidence(r):
        e = []
        if r['z_c'] == r['z_c'] and r['z_c'] < c.Z_C_ON and r['d'] / np.mean([FW[b] for b in evb]) <= c.CENTER_WITHIN:
            e.append('the bursts center on it')
        ce = r.get('center')
        if ce and ce.get('verdict') == 'BACKGROUND SOURCE':
            e.append('centering test')
        rho, p, nn = r['night']
        if rho == rho and rho > 0 and p < c.P_NIGHT:
            e.append(f'night by night (rho {rho:+.2f}, p {p:.1g})')
        if explains(r) and cover(r) >= c.COVER and r.get('taper'):
            e.append('its taper accounts for the count')
        return e
    ev_by = [(r, evidence(r)) for r in ordered]
    strong = [(r, e) for r, e in ev_by if len(e) >= 2 or (e and e[0] in ('the bursts center on it', 'centering test'))
              or any(x.startswith('night by night') for x in e)]
    # named order: direct position evidence first, then the brighter
    strong.sort(key=lambda re_: (-int(any(x in ('the bursts center on it', 'centering test') for x in re_[1])), -re_[0]['s74']))
    near_unres = [r for r in ordered if r['bands_near'] and (r.get('center') or {}).get('verdict', '').startswith('UNDECIDED')
                  and r['z_c'] == r['z_c'] and r['z_c'] < c.Z_C_NEAR]
    if cen['z'] >= c.Z_OFF:
        # the bursts at the target center elsewhere
        if strong:
            v = 'BACKGROUND SOURCE'
            named = nm([r for r, e in strong])
            why = cen_txt + ' | ' + ' | '.join(f"{describe(r)} [evidence: {', '.join(e)}]" for r, e in strong[:2])
        else:
            v = 'NOT CENTERED ON THE TARGET (source not identified)'
            named = ''
            why = cen_txt + ' | ' + ' | '.join(describe(r) for r in ordered[:2])
        return dict(base, verdict=v, named=named, coverage='', why=why)
    if cen['z'] < c.Z_ON:
        # the bursts center on the target
        if near_unres or any(r['z_c'] == r['z_c'] and r['z_c'] < c.Z_C_ON for r in ordered):
            close = near_unres or [r for r in ordered if r['z_c'] < c.Z_C_ON]
            v = 'UNDECIDED (centered on the target and on a cataloged source alike)'
            named = ''
            why = cen_txt + ' | ' + ' | '.join(f"{describe(r)} (the centroid is {r['z_c']:.1f} sigma from it)" for r in close[:2])
        elif all(excess(r) for r in ordered if r['bands_far'] or r['bands_near']) and max(cover(r) for r in ordered) >= c.COVER:
            v = 'NOT EXPLAINED BY A CATALOGED SOURCE (centered on the target)'
            named = ''
            why = cen_txt + ' | ' + ' | '.join(describe(r) for r in ordered[:2])
        else:
            v = 'UNDECIDED'
            named = ''
            why = cen_txt + ' | ' + ' | '.join(describe(r) for r in ordered[:2])
        return dict(base, verdict=v, named=named, coverage='', why=why)
    # in between: the count and night tests decide a background; nothing is called unexplained
    if strong:
        v = 'BACKGROUND SOURCE'
        named = nm([r for r, e in strong])
        why = cen_txt + ' | ' + ' | '.join(f"{describe(r)} [evidence: {', '.join(e)}]" for r, e in strong[:2])
    elif any(explains(r) and cover(r) >= c.COVER for r in ordered):
        rr = [r for r in ordered if explains(r) and cover(r) >= c.COVER]
        v = 'CONSISTENT WITH A BACKGROUND SOURCE'
        named = nm(rr)
        why = cen_txt + ' | ' + ' | '.join(describe(r) for r in rr[:2])
    else:
        v = 'UNDECIDED'
        named = ''
        why = cen_txt + ' | ' + ' | '.join(describe(r) for r in ordered[:2])
    return dict(base, verdict=v, named=named, coverage='', why=why)


def describe(r):
    s = f"{r['name'].split(' ', 1)[-1]} {r['s74']:.1f} Jy at {r['d']:.1f}'"
    tt = r.get('taper')
    if tt:
        s += f": taper {','.join(b['band'] for b in tt['bands'])} - {tt['n_obs']} bursts at the target"
        if r.get('lam') == r.get('lam') and r.get('lam') is not None:
            q = r['ctl_ratio_q']
            s += (f", background {r['lam']:.1f} (interpolated {tt['n_pred']:.1f} x {r['kappa']:.2f}); ratio {r['ratio']:.2f} against "
                  f"{r['ctl'][0]} control lines (median {q[0]:.2f}, 95th {q[1]:.2f}, max {q[2]:.2f}): p {r['p_emp']:.3f}")
        else:
            s += f", interpolated {tt['n_pred']:.1f} (fewer than 15 matched control lines: not calibrated)"
    ce = r.get('center')
    if ce:
        s += f"; centering {','.join(r['bands_near'])}: {ce.get('verdict')}"
        if ce.get('t') is not None:
            s += f" (t {ce['t']:+.2f} +- {ce['t_err']:.2f}, {ce['n_cells']} nights)"
    rho, p, nn = r['night']
    if rho == rho:
        s += f"; per night rho {rho:+.2f} (p {p:.1g}, {nn} nights)"
    return s


def cand_row(J, tier, r):
    tt = r.get('taper') or {}
    ce = r.get('center') or {}
    f = lambda x, n=2: (round(x, n) if isinstance(x, float) and x == x else ('' if isinstance(x, float) else x))   # noqa: E731
    return dict(stokes=J.st, identity=J.job['identity'], tier=tier, candidate=r['name'], s74=round(r['s74'], 2),
                sep_arcmin=round(r['d'], 2), bands_far=' '.join(r['bands_far']), bands_near=' '.join(r['bands_near']),
                reach=' '.join(f"{b}:{q[2]}x" for b, q in sorted(r['reach'].items(), key=lambda kv: float(kv[0][:-3]))),
                n_obs=tt.get('n_obs', ''), n_pred=f(tt.get('n_pred', float('nan'))),
                pred_err=f(math.sqrt(tt['var']) if tt else float('nan')),
                n_ctl=(r.get('ctl') or ('',))[0], kappa=f(r.get('kappa', float('nan')), 3), phi=f(r.get('phi', float('nan')), 3),
                background=f(r.get('lam', float('nan'))), ratio=f(r.get('ratio', float('nan'))),
                ctl_ratio_median=f((r.get('ctl_ratio_q') or (float('nan'),))[0]),
                ctl_ratio_p95=f((r.get('ctl_ratio_q') or (0, float('nan')))[1]),
                p_emp=f(r.get('p_emp', float('nan')), 4), p_nb=f(r.get('p_nb', float('nan')), 6), shown=shown(r),
                centering=ce.get('verdict', ''),
                center_t=f(ce.get('t', float('nan')) if ce.get('t') is not None else float('nan')),
                centroid_to_candidate_sigma=f(r.get('z_c', float('nan')), 1), night_rho=f(r['night'][0]),
                night_p=f(r['night'][1], 4), night_n=r['night'][2], explains=explains(r), excess=excess(r),
                **({} if tc().KNIFE_EDGE_SE is None else dict(
                    p_emp_se=f(r.get('p_se', float('nan')), 4), ratio_se=f(r.get('ratio_se', float('nan')), 3),
                    n_ctl_se=f(r.get('n_ctl_se', float('nan')), 1),
                    centroid_to_candidate_se=f(r.get('z_c_se', float('nan')), 3))))


# ---------------------------------------------------------------------------------------------------------------------
# v2's centering test (forced_wide_analyze.py: Job, controls_for, verdict)
# ---------------------------------------------------------------------------------------------------------------------

def chord(arcmin):
    return 2 * math.sin(math.radians(arcmin / 60.0) / 2)


def sigma_band_arcmin(band):
    """per-detection position error for the weights (v2 module notes)"""
    c = tc()
    f = float(band[:-3])
    return math.hypot(c.V2_SYS_ARCSEC / 60.0 * (f / c.V2_SYS_REF_MHZ) ** c.V2_SYS_INDEX, c.V2_STAT_FWHM * fw_table()[band])


class V2Job:
    """one (Stokes, target) job's bursts for v2's centering test: the clean (cell, band, tier) set, the arrays, a spatial index,
    its cutout footprint"""

    def __init__(self, job, bdir):
        from scipy.spatial import cKDTree
        from exopipe.phase3.lib.forced.wide_jobs import FA, physical_sources
        c = tc()
        FW = fw_table()
        self.job = job
        self.ok = False
        parts = collections.defaultdict(list)
        self.clean = set()
        for b in job['bands']:
            p = burst_path(bdir, job['stokes'], job['identity'], b)
            if not os.path.exists(p):
                return
            z = np.load(p)
            for c_, t, n, zn, s in zip(z['f_cell'], z['f_tier'], z['f_planes'], z['f_zone'], z['f_status']):
                if s == 'ok' and zn != 1 and n > 0:
                    self.clean.add((str(c_), b, str(t)))
            n = len(z['snr'])
            for k in ('cell', 'tier', 'ts', 'ra', 'dec', 'snr'):
                parts[k].append(z[k])
            parts['band'].append(np.array([b] * n))
        d = {k: np.concatenate(v) for k, v in parts.items()}
        keep = np.array([(c_, b, t) in self.clean for c_, b, t in zip(d['cell'], d['band'], d['tier'])], bool)
        keep &= (d['snr'] > 0) if job['stokes'] == 'I' else np.ones(len(keep), bool)
        self.d = {k: v[keep] for k, v in d.items()}
        self.d['fw'] = np.array([FW[b] for b in self.d['band']])
        self.d['w'] = np.array([1.0 / sigma_band_arcmin(b) ** 2 for b in self.d['band']])
        fa = FA()
        self.tree = cKDTree(fa.uvec(self.d['ra'], self.d['dec'])) if len(self.d['ra']) else None
        self.bands = set(job['bands'])
        self.center = tuple(job['T'])
        self.srcs = physical_sources(self.center[0], self.center[1], c.CUT_HALF_ARCMIN + c.V2_SRC_PAD_ARCMIN, c.S_MIN_JY)
        self.very_bright = physical_sources(self.center[0], self.center[1], c.CUT_HALF_ARCMIN + c.V2_VERY_BRIGHT_PAD_ARCMIN,
                                            c.V2_VERY_BRIGHT_JY)
        self.ok = True

    def region(self, S, T, tier, k, bands, offsets=None):
        """the per-cell statistic of this job's bursts in the region of the pair (S, T) at k, in `bands` (None: no burst)"""
        from exopipe.phase3.lib.forced.wide_jobs import FA, tan_off, tan_pos
        FW = fw_table()
        de_, dn_ = tan_off(T[0], T[1], S[0], S[1])
        L2 = de_ ** 2 + dn_ ** 2
        sep = math.sqrt(L2)
        M = tan_pos(de_ / 2, dn_ / 2, S[0], S[1])
        rmax = sep / 2 + k * max(FW[b] for b in bands)
        if self.tree is None:
            return None
        idx = np.array(self.tree.query_ball_point(FA().uvec(*M), chord(rmax)), int)
        if not len(idx):
            return None
        d = self.d
        idx = idx[(d['tier'][idx] == tier) & np.isin(d['band'][idx], list(bands))]
        if not len(idx):
            return None
        e, n = tan_off(d['ra'][idx], d['dec'][idx], S[0], S[1])
        if offsets is not None:
            off = np.array([offsets.get((d['cell'][i], d['band'][i], d['tier'][i]), (0.0, 0.0, 0))[:2] for i in idx])
            e = e - off[:, 0]
            n = n - off[:, 1]
        rm = np.hypot(e - de_ / 2, n - dn_ / 2)
        m = rm < sep / 2 + k * d['fw'][idx]
        if not m.any():
            return None
        idx, e, n = idx[m], e[m], n[m]
        t = (e * de_ + n * dn_) / L2
        u = (-e * dn_ + n * de_) / L2
        by = collections.defaultdict(list)
        for c_, tt, ww in zip(d['cell'][idx], t, d['w'][idx]):
            by[c_].append((tt, ww))
        tn = np.array([np.average([x[0] for x in v], weights=[x[1] for x in v]) for v in by.values()])
        wn = np.array([sum(x[1] for x in v) for v in by.values()])
        mean = float(np.average(tn, weights=wn))
        err = float(np.sqrt(np.average((tn - mean) ** 2, weights=wn) / max(len(tn) - 1, 1))) if len(tn) > 1 else float('nan')
        return dict(t=mean, err=err, n_cells=len(tn), n_bursts=int(len(idx)), tb=t, ub=u, sep=sep,
                    fw=d['fw'][idx], band=d['band'][idx], cell=d['cell'][idx])


def v2_controls_for(Jx, jobs_loaded):
    """the pooled control pairs of target job Jx (v2 module notes): [(V2Job, C(ra, dec), T'(ra, dec), common bands, s74)]"""
    from exopipe.phase3.lib.forced.wide_jobs import FA, tan_off, tan_pos
    c = tc()
    FW = fw_table()
    fa = FA()
    CUT_HALF, MARGIN = c.CUT_HALF_ARCMIN, c.V2_CUT_MARGIN_ARCMIN
    jx = Jx.job
    S, T = tuple(jx['S']), tuple(jx['T'])
    s74 = jx['s74']
    de_, dn_ = tan_off(T[0], T[1], S[0], S[1])
    sep = math.hypot(de_, dn_)
    Mx = tan_pos(de_ / 2, dn_ / 2, S[0], S[1])
    rmax = sep / 2 + c.V2_K_MAX * max(FW[b] for b in jx['bands'])
    need = math.ceil(len(jx['bands']) / 2)
    canonical = control_order() == 'canonical'
    best = {}
    for J in jobs_loaded:
        bands = Jx.bands & J.bands
        if len(bands) < need:
            continue
        for cs in J.srcs:
            if not (max(c.V2_CTL_MIN_JY, c.V2_CTL_FLUX[0] * s74) <= cs[2] <= c.V2_CTL_FLUX[1] * s74):
                continue
            Mp = tan_pos(de_ / 2, dn_ / 2, cs[0], cs[1])
            Tp = tan_pos(de_, dn_, cs[0], cs[1])
            dcen = float(fa.sep_arcsec(Mp[0], Mp[1], J.center[0], J.center[1])) / 60.0
            if dcen + rmax > CUT_HALF - MARGIN:
                continue
            if float(fa.sep_arcsec(Mp[0], Mp[1], Mx[0], Mx[1])) / 60.0 < 2 * rmax:
                continue
            if any(o[2] > c.V2_CTL_BRIGHTER * cs[2] and float(fa.sep_arcsec(o[0], o[1], Mp[0], Mp[1])) / 60.0 < rmax for o in J.srcs):
                continue
            if any(float(fa.sep_arcsec(b[0], b[1], Mp[0], Mp[1])) / 3600.0 < c.V2_VERY_BRIGHT_DEG for b in J.very_bright
                   if float(fa.sep_arcsec(b[0], b[1], cs[0], cs[1])) / 60.0 > c.V2_VERY_BRIGHT_SELF_ARCMIN):
                continue
            key = (round(cs[0] * c.V2_KEY_PER_DEG) / c.V2_KEY_PER_DEG, round(cs[1] * c.V2_KEY_PER_DEG) / c.V2_KEY_PER_DEG)
            score = (len(bands), -dcen)
            if canonical:            # a tie to the smallest (job identity, source identity), never the first in pool order
                tie = (job_id(J), source_id(cs))
                cur = best.get(key)
                if cur is None or score > cur[0] or (score == cur[0] and tie < cur[2]):
                    best[key] = (score, (J, (cs[0], cs[1]), Tp, bands, cs[2]), tie)
            elif key not in best or score > best[key][0]:
                best[key] = (score, (J, (cs[0], cs[1]), Tp, bands, cs[2]))
    if canonical:                    # by the control source's identity: a capped draw (V2_CTL_DRAW) is then job-order free too
        out = [v[1] for v in sorted(best.values(), key=lambda v: v[2][1])]
    else:
        out = [v[1] for v in best.values()]
    draw = v2_draw()
    if draw is not None and len(out) > draw[0]:            # production's capped draw (forced_wide_analyze.py:188-190)
        rng = np.random.default_rng(draw[1])
        out = [out[i] for i in sorted(rng.choice(len(out), draw[0], replace=False))]
    return out


def v2_draw():
    """forced_wide_test.V2_CTL_DRAW: None (every eligible control pair) or (cap, seed); anything else raises ValueError (the
    stage refuses it before any job runs)."""
    d = tc().V2_CTL_DRAW
    if d is None:
        return None
    if (isinstance(d, (tuple, list)) and len(d) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in d)
            and d[0] >= 1 and d[1] >= 0):
        return int(d[0]), int(d[1])
    raise ValueError('forced_wide_test.V2_CTL_DRAW %r: None (every eligible control) or (cap >= 1, seed >= 0)' % (d,))


def v2_verdict(t, err, ncell, ctl):
    """(verdict, z, power, mu_c, sigma_c, n_ctl, f_bg, f_tg) - v2 module notes"""
    c = tc()
    cv = [x for x in ctl if x is not None and x['n_cells'] >= c.V2_CTL_MIN_CELLS and np.isfinite(x['err'])]
    nan = float('nan')
    if ncell < c.V2_MIN_NIGHTS or not np.isfinite(err):
        return 'too few nights', nan, nan, nan, nan, len(cv), nan, nan
    if len(cv) < c.V2_CTL_MIN_N:
        return 'too few controls', nan, nan, nan, nan, len(cv), nan, nan
    tc_ = np.array([x['t'] for x in cv])
    ec = np.array([x['err'] for x in cv])
    mu = float(np.median(tc_))
    mad = 1.4826 * float(np.median(np.abs(tc_ - mu)))
    sig = math.sqrt(max(0.0, mad ** 2 - float(np.median(ec ** 2))))
    s = math.sqrt(sig ** 2 + err ** 2)
    z = (t - mu) / s
    power = (1 - 2 * mu) / s
    f_bg = float(np.mean(tc_ >= t))
    f_tg = float(np.mean((1 - tc_) <= t))
    d_tg = power - z                    # how far the target-only expectation (1 - mu_c) lies above t, in the same s
    if power < c.V2_POWER_MIN:
        v = 'UNDECIDED (cannot separate)'
    elif z < c.V2_Z_BG and d_tg >= c.V2_D_TG:
        v = 'BACKGROUND SOURCE'
    elif z >= c.V2_Z_NOT:
        v = 'NOT THE BACKGROUND SOURCE'
    else:
        v = 'UNDECIDED'
    return v, z, power, mu, sig, len(cv), f_bg, f_tg


# ---------------------------------------------------------------------------------------------------------------------
# the run: every job of each Stokes (forced_wide_v3.main without the figures)
# ---------------------------------------------------------------------------------------------------------------------

_G = {}


def _one(i):
    return analyze(_G['todo'][i], _G['events'], _G['pool'], _G['v2pool'], _G['bdir'])


def analyze_all(jobs, events, bdir, workers=1, only=None, log=print):
    """(result rows, candidate rows) of the jobs (base then extension, the analysis' order): per Stokes I then V, the pool =
    every job of that Stokes, the v2 pool = those with every band searched; ``only`` = {(Stokes, identity)} analyzed (None:
    all). Fork workers share the loaded bursts copy-on-write; map keeps the order."""
    rows, crows = [], []
    for st in ('I', 'V'):
        pool = [JobData(j, bdir) for j in jobs if j['stokes'] == st]
        v2pool = [x for x in (V2Job(j, bdir) for j in jobs if j['stokes'] == st) if x.ok]
        todo = [J for J in pool if not only or (st, J.job['identity']) in only]
        log(f'Stokes {st}: {len(pool)} jobs loaded, {len(v2pool)} for the centering controls, {len(todo)} to analyze')
        _G.update(todo=todo, events=events, pool=pool, v2pool=v2pool, bdir=bdir)
        try:
            if workers > 1 and len(todo) > 1:
                from multiprocessing import get_context
                from exopipe.phase3.lib.forced.wide_search import pool_worker_init
                with get_context('fork').Pool(workers, initializer=pool_worker_init) as p:
                    outs = p.map(_one, range(len(todo)), chunksize=1)
            else:
                outs = [_one(i) for i in range(len(todo))]
        finally:
            _G.clear()
        for r, c_ in outs:
            rows += r
            crows += c_
    return rows, crows


def write_rows(path, rows):
    """CSV of rows with the union of their keys in first-seen order (the analysis' writer), tmp + rename."""
    write_rows_keys(path, rows, list(dict.fromkeys(k for r in rows for k in r)))


def write_rows_keys(path, rows, keys):
    """CSV of rows with these columns (csv.DictWriter, newline=''), tmp + rename."""
    tmp = '%s.tmp.%d' % (path, os.getpid())
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------------------------------
# the publication (publish_v3_to_runs.py)
# ---------------------------------------------------------------------------------------------------------------------

class NamedNotFound(ValueError):
    pass


def named_radec(rows, positions):
    """Each row with 'named_radec' added: the position of every named source ('ra,dec' with 5 decimals, ';'-joined), the
    physical source of that name (the catalogue prefix dropped) within NAMED_RADIUS_ARCMIN of the target. positions =
    {(Stokes, identity): (ra, dec)} of the jobs. Raises NamedNotFound (the stage refuses W-NAMED)."""
    from exopipe.phase3.lib.forced.wide_jobs import physical_sources
    c = tc()
    out = []
    for r in rows:
        r = dict(r)
        r['named_radec'] = ''
        if r.get('named'):
            T = positions[(r['stokes'], r['identity'])]
            srcs = {s[3].split(' ', 1)[-1]: s for s in physical_sources(T[0], T[1], c.NAMED_RADIUS_ARCMIN, c.S_MIN_JY)}
            pos = []
            for nm in r['named'].split('; '):
                key = ' '.join(nm.split(' ')[:2])
                s_ = srcs.get(key)
                if s_ is None:
                    raise NamedNotFound(f'{r["identity"]}: named {key} not found around the target')
                pos.append(f'{s_[0]:.5f},{s_[1]:.5f}')
            r['named_radec'] = ';'.join(pos)
        out.append(r)
    return out


def read_rows(path):
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        return list(rd), list(rd.fieldnames or [])
