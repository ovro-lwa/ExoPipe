"""The size / axis-ratio rule at every S/N (Gregg 2026-10-22, "All approved": ledger clv15 b8f73a1,
R-GREGG-20261022-ALL-APPROVED-SIZE-GATE-READMISSION-SUBAGENTS; spec exopipe_ops/refactor_20261012/plan/specs/fp_gates_v2.md
sections 1 and 3; DECISIONS_PENDING.md "Ruled by you 2026-10-22").

THE RULE. A detection is measured in its own search frame by ONE 2-D Gaussian - the run's own fitter (core.fitting.
gaussian_photometry) on the detection's stamp, in BOTH Stokes (Stokes V no longer on the PyBDSF component, which splits an
extended island into beam-sized pieces):
    size = fitted FWHM major / the epoch beam's BMAJ;   err = its 1-sigma fit error, same unit;   eff = size - err (no err: 0)
    axis = fitted axis ratio / the beam's (BMAJ / BMIN)
    S/N  = |fitted peak| / (1.4826 x MAD of the frame's non-zero pixels in an annulus OUTSIDE the fitted source, from
           r_in = min(6, max(2, 1.5 x size)) beams to r_in + 4 beams) - the scale the calibration injects on
and REMOVED, at every S/N, when eff > max(1.5, E_eff(S/N)) ('gaussfit-size') or axis > max(1.5, E_axis(S/N)) ('axis-ratio'),
where E(S/N) is the 99th percentile of beam-shaped point sources injected into the TIER'S OWN frames at that S/N (the
calibration product, ``Envelope``), linear in log S/N between the calibrated S/N and clamped to their range. 1.5 and 1.5 are
the ratified floors (R-SIZE-AXIS-GATE-1.5-RATIFIED). A failed fit, or no finite S/N, keeps the detection. The rule replaces
the fixed thresholds - the search's 1.5 / 1.5 above 8 sigma and the catalogue's 1.7 at 7-8 sigma for unpromoted singles.

THE CALIBRATION PRODUCT (one per tier; exopipe/common/size_envelope_<tier>.json, md5-pinned in config): per
'<Stokes>_<low|high>_<injected S/N>' the injections' n, n_fail, medians and 99th percentiles (eff_q99 and axis_q99 are the
rule's), built by ``envelope_from_injections`` from injections placed ON THE SKY inside the search's domain (``in_domain``:
elevation >= 25 deg at the image's mid-time, >= 2 deg from the A-team; >= 3 beams from any served detection), the stamp refused
only when non-finite or > 10% exact zeros (``stamp_ok``: the deep frames carry 2-5% isolated exact zeros on the sky), fitted
exactly as the detections are (``measure_detection``). Both band classes and both Stokes must be calibrated: a product without
one refuses (``Envelope.require``).

Reference implementation, reproduced by tools/size_rule_reproduce.py: phase3/refactor_analysis/
bright_source_contamination_20261020/{fp_size_common.py, size_filter2.py} (the 2026-10-22 page filters; first ported on the
never-merged branch fp-gates-v2, 8a52fa6 / 1851b13, lib/search/fp_gates_v2.py).

Engine wiring (lib/search/deep_search._size_test, both cadences): the step hands the rule and the product to the engine child
(RULE_ENV, ENVELOPE_ENV, ENVELOPE_MD5_ENV; search_detect / search_sublst SIZE_RULE, SIZE_ENVELOPE, SIZE_ENVELOPE_MD5);
'production' keeps the fixed rule above 8 sigma unchanged (the parity file). Under 'envelope' the engine writes RUN_RECORD into
its run directory (one rule per run) and the gate ledger carries GLED_COLS_SIZE (filled on every row the rule judged); the
catalogue reads the attempts' records (one rule per catalogue): its builder's 7-8 sigma gate is replaced (catalog_env), and a
conformance that removes by production's fixed size / axis thresholds contradicts the rule and is refused (fixed_removals_in,
final_findings). Otherwise pure functions; the record helpers write only the file they name.

Not wired here: the per-instant carry across bands (``carry_instant``; spec section 1 places it at publish, by position within
the association radius) and the snapshot tiers (their size gate is clv15 phase3_snapshot_fit.fit_verdict, in place, and no
calibration of the 10-s / 1-minute frames exists).
"""
import hashlib
import json
import math
import os

import numpy as np

from exopipe.phase3.core.fitting import gaussian_photometry

RULING = 'R-GREGG-20261022-ALL-APPROVED-SIZE-GATE-READMISSION-SUBAGENTS'
HIGH_BANDS = frozenset({'41MHz', '46MHz', '50MHz', '55MHz', '59MHz', '64MHz', '69MHz', '73MHz', '78MHz', '82MHz',
                        'Green', 'Blue', 'Full'})
STOKES = ('I', 'V')
CLASSES = ('low', 'high')
SIZE_FLOOR = 1.5          # (fitted major - 1 sigma) / epoch BMAJ; R-SIZE-AXIS-GATE-1.5-RATIFIED
AXIS_FLOOR = 1.5          # fitted axis ratio / the beam's; R-SIZE-AXIS-GATE-1.5-RATIFIED
SNR_GRID = (5, 6, 7, 8, 10, 12, 15, 20, 30)
QUANTILE_KEYS = ('eff_q99', 'axis_q99')   # the rule's percentiles in the product

ATEAM = ((350.850, 58.815), (299.868, 40.734), (83.633, 22.0145), (187.706, 12.391), (252.784, 4.993), (139.524, -12.096))
SITE_LAT_DEG = 37.2398                    # config catalog.SITE_LAT_DEG
ELEV_FLOOR_DEG = 25.0                     # the deep and 10-minute search floor (search_detect.ELEVATION_FLOOR_DEG)
ATEAM_VETO_DEG = 2.0                      # the calibration's A-team distance (search_detect.VETO_ATEAM_RADIUS_DEG)
MAX_ZERO_FRAC = 0.10                      # deep search frames carry 2-5% isolated exact zeros on the sky (the residual's mask)
R_OUT_MAX_BEAMS = 10.0                    # the annulus reaches at most 6 + 4 beams
EDGE_R_BEAMS = 6.0                        # near a frame edge: a stamp holding the annulus to 6 beams (the reference's fallback)

RULES = ('envelope', 'production')        # search_detect / search_sublst SIZE_RULE
RULE_ENV = 'EXOPIPE3_SIZE_RULE'           # how the step hands the rule to the engine child
ENVELOPE_ENV = 'EXOPIPE3_SIZE_ENVELOPE'   # ... the calibration product's path
ENVELOPE_MD5_ENV = 'EXOPIPE3_SIZE_ENVELOPE_MD5'   # ... and its pinned md5
RUN_RECORD = 'size_rule.json'             # in the engine's run directory (beside v_deep_ledger.csv): the rule it searched under
RECORD_VERSION = 'size-rule-1'
# the gate ledger's columns of the rule (appended to deep_search.GLED_COLS under 'envelope' only; production's ledger is unchanged)
GLED_COLS_SIZE = ['size_rule', 'size_snr', 'size_thr', 'axis_thr', 'size_excess', 'axis_excess', 'size_verdict']
VERDICTS = ('gaussfit-size', 'axis-ratio', 'point-like', 'fit-failed', 'no-snr')   # size_verdict; the first two remove
GATE_7_8_OFF = 'inf'                      # P3_SIZE_GATE_7_8 for the catalogue builder under 'envelope': its 7-8 sigma gate never acts


class RuleRefusal(RuntimeError):
    """The rule cannot be applied as configured (unknown rule, unreadable or unpinned product, a Stokes or band class it does not
    calibrate, a run directory searched under another rule). Its text holds REFUSING, so the engine child exits as refused."""


def _refuse(msg):
    raise RuleRefusal(f'{msg} - REFUSING')


def band_class(band):
    """'high' (41-82 MHz, Green, Blue, Full) or 'low' (18-36 MHz, Red)."""
    return 'high' if band in HIGH_BANDS else 'low'


def rule_value(value, what='SIZE_RULE'):
    """``value`` checked against RULES; an unknown or empty value REFUSES (a typo never selects either rule)."""
    if value not in RULES:
        _refuse(f'{what} = {value!r} is not one of {RULES}')
    return value


PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


def resolve(path):
    """A configured product path: absolute as given, else relative to the package root (exopipe/common/... in the repo)."""
    return path if (not path or os.path.isabs(path)) else os.path.join(PKG_ROOT, path)


def md5_file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------------
# measurement

def beam_image(bmaj_deg, bmin_deg, bpa_deg, cd_deg, R):
    """The beam (FWHM BMAJ x BMIN, astronomical PA from north through east; x grows to the west) on a (2R+1)^2 grid, peak 1."""
    y, x = np.mgrid[-R:R + 1, -R:R + 1].astype(float)
    sa, sb = bmaj_deg / cd_deg / 2.3548, bmin_deg / cd_deg / 2.3548
    t = math.radians(bpa_deg)
    u = -x * math.sin(t) + y * math.cos(t)
    v = -x * math.cos(t) - y * math.sin(t)
    return np.exp(-0.5 * ((u / sa) ** 2 + (v / sb) ** 2))


def elevation(ra, dec, lst_deg):
    """Elevation (deg) at OVRO of (RA, Dec) at local sidereal angle lst_deg (arrays allowed)."""
    hr = np.radians(lst_deg - np.asarray(ra)); d = np.radians(np.asarray(dec)); lat = math.radians(SITE_LAT_DEG)
    return np.degrees(np.arcsin(np.sin(d) * math.sin(lat) + np.cos(d) * math.cos(lat) * np.cos(hr)))


def ateam_sep(ra, dec):
    """Distance (deg) to the nearest A-team source."""
    ra, dec = np.radians(np.asarray(ra, float)), np.radians(np.asarray(dec, float))
    out = np.full(np.shape(ra), 180.0)
    for a, b in ATEAM:
        a, b = math.radians(a), math.radians(b)
        c = np.sin(dec) * math.sin(b) + np.cos(dec) * math.cos(b) * np.cos(ra - a)
        out = np.minimum(out, np.degrees(np.arccos(np.clip(c, -1, 1))))
    return out


def in_domain(ra, dec, lst_deg):
    """The search's domain, where injections belong: a finite position, elevation >= 25 deg at the image's mid-time, >= 2 deg
    from the A-team. (The first calibration placed every deep injection beyond the horizon.)"""
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    el = np.where(ok, elevation(np.nan_to_num(ra), np.nan_to_num(dec), lst_deg), -90.0)
    return ok & (el >= ELEV_FLOOR_DEG) & (ateam_sep(np.nan_to_num(ra), np.nan_to_num(dec)) >= ATEAM_VETO_DEG)


def stamp_ok(big):
    """An injection stamp: every pixel finite and at most 10% exact zeros (a stricter 'no zero' test refuses every on-sky stamp)."""
    return bool(np.isfinite(big).all() and np.mean(big == 0) <= MAX_ZERO_FRAC)


def stamp_radius(bpix):
    """The fitter's own stamp half-size (gaussian_photometry: max(8, round(2 x beam px)))."""
    return max(8, int(round(2 * bpix)))


def rms_radius(bpix):
    """Half-size of the stamp that holds the noise annulus out to R_OUT_MAX_BEAMS."""
    return int(math.ceil(R_OUT_MAX_BEAMS * bpix)) + 2


def edge_radius(bpix):
    """Half-size of the smaller stamp used near a frame edge (the annulus then reaches 6 beams)."""
    return int(math.ceil(EDGE_R_BEAMS * bpix)) + 2


def measure(stamp, R, bpix, cd_deg, bmaj_deg, bmin_deg):
    """One Gaussian on a (2R+1)^2 stamp centered on the detection: (size, axis, size_err, |peak|), size = fitted FWHM major /
    BMAJ, axis = fitted axis ratio / (BMAJ / BMIN), size_err the 1-sigma error of the major axis in BMAJ (nan if the fit gave
    none); None when the fit fails (the rule then keeps the detection)."""
    ph = gaussian_photometry(stamp, R, R, bpix, cd_deg)
    if not ph:
        return None
    size = ph['fwhm_maj_arcmin'] / 60.0 / bmaj_deg
    axis = ph['axis_ratio'] / (bmaj_deg / bmin_deg)
    e = ph.get('fwhm_maj_err_arcmin')
    err = e / 60.0 / bmaj_deg if e is not None else float('nan')
    return size, axis, err, abs(ph['peak_jy'])


def annulus_rms(big, Rn, bpix, size_beams=1.0):
    """1.4826 x MAD of the non-zero pixels in an annulus OUTSIDE the fitted source: from r_in = min(6, max(2, 1.5 x size)) to
    r_in + 4 beams; nan with 50 or fewer pixels. (A fixed 2-6 beam annulus sits in the wings of a source of >= 2 beams, lowers its
    S/N and so loosens the rule for its own targets.)"""
    r_in = min(6.0, max(2.0, 1.5 * size_beams)); r_out = r_in + 4.0
    gy, gx = np.mgrid[-Rn:Rn + 1, -Rn:Rn + 1]
    rr = np.hypot(gy, gx) / bpix
    a = big[(rr >= r_in) & (rr <= r_out)]
    a = a[a != 0]
    return 1.4826 * float(np.median(np.abs(a - np.median(a)))) if a.size > 50 else float('nan')


def measure_detection(big, Rn, bpix, cd_deg, bmaj_deg, bmin_deg):
    """The detection's record from its (2Rn+1)^2 stamp: size, axis, err, eff = size - err (err nan -> 0), S/N = |fitted peak| /
    the annulus rms outside the fitted size. None when the fit fails."""
    R = stamp_radius(bpix)
    r = measure(big[Rn - R:Rn + R + 1, Rn - R:Rn + R + 1], R, bpix, cd_deg, bmaj_deg, bmin_deg)
    if r is None:
        return None
    size, axis, err, peak = r
    rms = annulus_rms(big, Rn, bpix, size)
    return dict(size=size, axis=axis, err=err, eff=size - (err if np.isfinite(err) else 0.0),
                snr=(peak / rms) if (np.isfinite(rms) and rms > 0) else float('nan'))


def frame_stamp(img, x, y, bpix):
    """(stamp, its half-size) around integer pixel (x, y) of a frame (2-D array; non-finite pixels read as 0, the search's own
    reading): the full stamp of rms_radius, else - near an edge - the smaller one of edge_radius, else (None, None)."""
    ny, nx = img.shape
    for R_ in (rms_radius(bpix), edge_radius(bpix)):
        if R_ <= x < nx - R_ and R_ <= y < ny - R_:
            return np.nan_to_num(np.asarray(img[y - R_:y + R_ + 1, x - R_:x + R_ + 1], dtype=np.float64)), R_
    return None, None


def measure_at(img, x, y, bmaj_deg, bmin_deg, cd_deg):
    """measure_detection at integer pixel (x, y) of a frame whose beam is (BMAJ, BMIN) deg and pixel |CDELT1| deg, on the
    stamp ``frame_stamp`` takes (the beam in pixels is BMAJ / |CDELT1|, as the injections were measured); None when no stamp
    fits the frame or the fit fails."""
    bpix = float(bmaj_deg) / float(cd_deg)
    big, R_ = frame_stamp(img, int(x), int(y), bpix)
    if big is None:
        return None
    return measure_detection(big, R_, bpix, float(cd_deg), float(bmaj_deg), float(bmin_deg))


# ---------------------------------------------------------------------------------------------------------------------------
# the calibration product

class Envelope:
    """The calibration product: per '<Stokes>_<class>_<S/N>' the injections' n and percentiles. ``source`` and ``md5`` name the
    file it was read from (recorded with every verdict)."""

    def __init__(self, cal, source='', md5=''):
        if not isinstance(cal, dict):
            _refuse(f'size envelope {source or "(in memory)"}: not a mapping')
        self.cal = cal
        self.source = source
        self.md5 = md5
        self._pts = {}

    @classmethod
    def load(cls, path, md5=None):
        """The product at ``path``; with ``md5`` the file must have it. Unreadable, malformed or another md5: REFUSED."""
        if not path or not os.path.isfile(path):
            _refuse(f'size envelope {path!r} does not exist')
        got = md5_file(path)
        if md5 and got != md5:
            _refuse(f'size envelope {path}: md5 {got} is not the pinned {md5} (another calibration)')
        try:
            with open(path) as f:
                cal = json.load(f)
        except (OSError, ValueError) as e:
            _refuse(f'size envelope {path} unreadable ({type(e).__name__}: {e})')
        return cls(cal, source=path, md5=got)

    def points(self, stokes, cls, key):
        """[(S/N, value)] of one percentile, sorted by S/N."""
        out = []
        for n, v in self.cal.items():
            p = n.split('_')
            if len(p) == 3 and p[0] == stokes and p[1] == cls and isinstance(v, dict) and key in v:
                try:
                    out.append((float(p[2]), float(v[key])))
                except (TypeError, ValueError):
                    _refuse(f'size envelope {self.source}: {n}.{key} = {v[key]!r} is not a number')
        return sorted(out)

    def require(self, stokes=STOKES, classes=CLASSES, keys=QUANTILE_KEYS, min_points=2):
        """Refuse unless every (Stokes, class) is calibrated at >= min_points S/N with finite, positive percentiles."""
        for st in stokes:
            for cl in classes:
                for k in keys:
                    pts = self.points(st, cl, k)
                    if len(pts) < min_points:
                        _refuse(f'size envelope {self.source}: {st} {cl} {k} calibrated at {len(pts)} S/N (needs {min_points})')
                    if not all(np.isfinite(v) and v > 0 for _, v in pts) or not all(s > 0 for s, _ in pts):
                        _refuse(f'size envelope {self.source}: {st} {cl} {k} has a non-finite or non-positive entry')
        return self

    def value(self, stokes, cls, snr, key):
        """``key`` (e.g. 'eff_q99') at this S/N: linear in log S/N between grid points, clamped to the calibrated range."""
        k = (stokes, cls, key)
        if k not in self._pts:
            pts = self.points(stokes, cls, key)
            if not pts:
                raise KeyError('no calibration for %s %s' % (stokes, cls))
            self._pts[k] = (np.log([p[0] for p in pts]), np.array([p[1] for p in pts]), pts[0][0], pts[-1][0])
        xs, ys, lo, hi = self._pts[k]
        return float(np.interp(math.log(min(max(snr, lo), hi)), xs, ys))

    def thresholds(self, stokes, cls, snr):
        """(size threshold, axis threshold) = (max(1.5, eff_q99(S/N)), max(1.5, axis_q99(S/N)))."""
        return max(SIZE_FLOOR, self.value(stokes, cls, snr, 'eff_q99')), max(AXIS_FLOOR, self.value(stokes, cls, snr, 'axis_q99'))

    def label(self):
        """The product's identity on every judged ledger row: 'envelope:<file name>:<md5[:12]>'."""
        return f'envelope:{os.path.basename(self.source) or "memory"}:{(self.md5 or "")[:12]}'


def size_verdict(env, stokes, band, rec):
    """('gaussfit-size' | 'axis-ratio' | '', size excess, axis excess) for a measure_detection record; excess = value /
    threshold, > 1 removes. A record without a finite S/N is kept ('', nan, nan)."""
    if rec is None or not np.isfinite(rec['snr']):
        return '', float('nan'), float('nan')
    ts, ta = env.thresholds(stokes, band_class(band), abs(rec['snr']))
    sx, ax = rec['eff'] / ts, rec['axis'] / ta
    return ('gaussfit-size' if sx > 1 else ('axis-ratio' if ax > 1 else '')), sx, ax


def judge(env, stokes, band, rec):
    """Every quantity the verdict uses, for the record: dict(size_snr, size_thr, axis_thr, size_excess, axis_excess,
    size_verdict) - size_verdict one of VERDICTS ('fit-failed': no fit; 'no-snr': no finite S/N; both keep)."""
    if rec is None:
        return dict(size_snr='', size_thr='', axis_thr='', size_excess='', axis_excess='', size_verdict='fit-failed')
    if not np.isfinite(rec['snr']):
        return dict(size_snr='', size_thr='', axis_thr='', size_excess='', axis_excess='', size_verdict='no-snr')
    ts, ta = env.thresholds(stokes, band_class(band), abs(rec['snr']))
    v, sx, ax = size_verdict(env, stokes, band, rec)
    return dict(size_snr=round(float(rec['snr']), 2), size_thr=round(ts, 4), axis_thr=round(ta, 4),
                size_excess=round(float(sx), 6), axis_excess=round(float(ax), 6), size_verdict=v or 'point-like')


def envelope_from_injections(rows, min_n=30):
    """The calibration product from injection rows (st, cls, snr, fit, size, axis, eff, snr_meas): per (Stokes, class, S/N) n,
    n_fail, medians and 99th percentiles - the reference's aggregation (fp_size_common.envelope_table)."""
    cal = {}
    for st in STOKES:
        for cls in CLASSES:
            for snr in SNR_GRID:
                v0 = [o for o in rows if o['st'] == st and o['cls'] == cls and float(o['snr']) == snr]
                v = [o for o in v0 if int(o['fit'])]
                if len(v) < min_n:
                    continue
                s = np.array([float(o['size']) for o in v]); a = np.array([float(o['axis']) for o in v])
                e = np.array([float(o['eff']) for o in v])
                cal['%s_%s_%g' % (st, cls, snr)] = dict(
                    n=len(v), n_fail=len(v0) - len(v), size_q50=float(np.median(s)), size_q99=float(np.percentile(s, 99)),
                    eff_q99=float(np.percentile(e, 99)), axis_q50=float(np.median(a)), axis_q99=float(np.percentile(a, 99)),
                    gate_rate=float(((e > SIZE_FLOOR) | (a > AXIS_FLOOR)).mean()), size15_rate=float((s > SIZE_FLOOR).mean()),
                    snr_meas_q50=float(np.median([float(o['snr_meas']) for o in v])))
    return cal


def carry_instant(records):
    """Per-instant carry across frequency (R-GREGG-20261018-VETOES-PER-FRAME-REFINED-FREQUENCY-YES-TIME-NO): records =
    iterable of (identity, instant, removed_by_the_rule); returns the (identity, instant) pairs whose detections in EVERY band
    go, because one of them is removed. Never across instants."""
    out = set()
    for ident, inst, removed in records:
        if removed:
            out.add((ident, inst))
    return out


# ---------------------------------------------------------------------------------------------------------------------------
# the engine (lib/search/deep_search._size_test, both cadences)

def engine_test(c, stokes, band, img, bmaj_deg, bmin_deg, cd_deg, env, flag_snr=8.0, size_flag=1.3, axis_flag=1.3):
    """The rule on one candidate of the engine - a dict with its integer pix_x, pix_y and its catalogue S/N (local_snr or snr) -
    in its search frame ``img`` (2-D, the search's own pixels), against the epoch beam (BMAJ, BMIN deg) and pixel |CDELT1| deg.
    Records on ``c`` size_beams, size_err_beams, axis_norm (the rule's measurement; production keeps those names) and
    GLED_COLS_SIZE, and the record-only flags of production above ``flag_snr`` (the candidate's own S/N; size above
    ``size_flag`` beams, axis above ``axis_flag`` - kept by the rule at its S/N). Returns what deep_search._size_test returns:
    (size_beams or '', 'remove' | 'remove-axis' | 'flag' | ''). An epoch without a beam REFUSES (its frame cannot be judged)."""
    try:
        bmaj, bmin = float(bmaj_deg), float(bmin_deg)
    except (TypeError, ValueError):
        bmaj = bmin = float('nan')
    if not (np.isfinite(bmaj) and np.isfinite(bmin) and bmaj > 0 and bmin > 0):
        _refuse(f'size rule: the epoch of {c.get("pix_x")},{c.get("pix_y")} has no beam (BMAJ {bmaj_deg!r}, BMIN {bmin_deg!r})')
    rec = measure_at(img, int(c['pix_x']), int(c['pix_y']), bmaj, bmin, cd_deg)
    j = judge(env, stokes, band, rec)
    c.update(j)
    c['size_rule'] = env.label()
    if rec is None:
        return '', ''
    c['size_beams'] = round(rec['size'], 2)
    if np.isfinite(rec['err']):
        c['size_err_beams'] = round(rec['err'], 3)
    c['axis_norm'] = round(rec['axis'], 2)
    if j['size_verdict'] == 'gaussfit-size':
        return c['size_beams'], 'remove'
    if j['size_verdict'] == 'axis-ratio':
        return c['size_beams'], 'remove-axis'
    verdict = ''
    try:
        snr = abs(float(c.get('local_snr', c.get('snr', 0))))
    except (TypeError, ValueError):
        snr = 0.0
    if snr > flag_snr and j['size_verdict'] == 'point-like':
        if rec['size'] > size_flag:
            c['size_flag'] = f'size > {size_flag:g} beams (point-like at its S/N: limit {j["size_thr"]:.2f})'
            verdict = 'flag'
        if rec['axis'] > axis_flag:
            c['axis_flag'] = f'axis > {axis_flag:g} x beam (point-like at its S/N: limit {j["axis_thr"]:.2f})'
            verdict = 'flag'
    return c['size_beams'], verdict


def configured(cadence):
    """(rule, envelope path, md5) in the config section of a cadence: search_detect (deep), search_sublst (10min)."""
    from exopipe.phase3.config import CFG
    sec = CFG.search_detect if cadence == 'deep' else CFG.search_sublst
    return sec.SIZE_RULE, resolve(sec.SIZE_ENVELOPE), sec.SIZE_ENVELOPE_MD5


def step_env(rule, path, md5, what='SIZE_RULE'):
    """The engine child's environment for the rule, from the step's config values ({RULE_ENV[, ENVELOPE_ENV, ENVELOPE_MD5_ENV]});
    under 'envelope' the product is checked first: present, the pinned md5, every Stokes and band class - REFUSED otherwise."""
    rule_value(rule, what)
    if rule == 'production':
        return {RULE_ENV: 'production'}
    if not md5:
        _refuse(f'{what} = {rule!r}: no md5 pinned for the envelope {path!r}')
    path = resolve(path)
    Envelope.load(path, md5).require()
    return {RULE_ENV: rule, ENVELOPE_ENV: path, ENVELOPE_MD5_ENV: md5}


def engine_rule(cadence):
    """In the engine child: None under 'production' (the fixed rule above 8 sigma, unchanged), else the checked Envelope. From
    the step's environment (step_env); a process no step started reads its cadence's config. REFUSED: an unknown rule, a product
    missing, without its pinned md5 or with another, or without a Stokes or band class."""
    rule = os.environ.get(RULE_ENV)
    if rule is None:
        rule, path, md5 = configured(cadence)
    else:
        path, md5 = os.environ.get(ENVELOPE_ENV, ''), os.environ.get(ENVELOPE_MD5_ENV, '')
    rule_value(rule)
    if rule == 'production':
        return None
    if not md5:
        _refuse(f'size rule {rule!r}: no md5 pinned for the envelope {path!r}')
    return Envelope.load(path, md5).require()


# ---------------------------------------------------------------------------------------------------------------------------
# the run's record: which rule a run directory (an attempt's transients/Run_<label>) was searched under

def run_record(env):
    """The record of a run searched with ``env`` (None = production's fixed rule)."""
    if env is None:
        return dict(rule='production', envelope='', md5='', version=RECORD_VERSION, ruling='')
    return dict(rule='envelope', envelope=os.path.basename(env.source), md5=env.md5, version=RECORD_VERSION, ruling=RULING)


def read_run_record(run_dir):
    """The rule a run directory was searched under: its RUN_RECORD, else production's (every run before the rule). An
    unreadable or malformed record REFUSES."""
    p = os.path.join(run_dir, RUN_RECORD)
    if not os.path.exists(p):
        return run_record(None)
    try:
        with open(p) as f:
            rec = json.load(f)
    except (OSError, ValueError) as e:
        _refuse(f'{p} unreadable ({type(e).__name__}: {e})')
    if not isinstance(rec, dict) or set(rec) != set(run_record(None)) or rec.get('rule') not in RULES:
        _refuse(f'{p}: not a size-rule record ({str(rec)[:160]})')
    return rec


def same_rule(a, b):
    return (a['rule'], a['md5']) == (b['rule'], b['md5'])


def write_run_record(run_dir, env, ledger='v_deep_ledger.csv'):
    """Under 'envelope', write RUN_RECORD in the engine's run directory. A run directory searched before under another rule or
    product - its record, or a ledger without one (production) - REFUSES: one rule per run (an append would mix two rules in
    one ledger). Under production nothing is written (production's outputs unchanged); a run directory recorded under the
    envelope refuses it the same way. Returns the record."""
    want = run_record(env)
    p = os.path.join(run_dir, RUN_RECORD)
    have = read_run_record(run_dir)
    if (os.path.exists(p) or os.path.exists(os.path.join(run_dir, ledger))) and not same_rule(have, want):
        _refuse(f'{run_dir} was searched under the size rule {have["rule"]} {have["md5"][:12]}; this search applies '
                f'{want["rule"]} {want["md5"][:12]} - one rule per run: search the cell afresh')
    if env is not None and not os.path.exists(p):
        os.makedirs(run_dir, exist_ok=True)
        tmp = f'{p}.part{os.getpid()}'
        with open(tmp, 'w') as f:
            json.dump(want, f, indent=1, sort_keys=True)
        os.replace(tmp, p)
    return want


def one_rule(run_dirs, want=None, what='the merged attempts'):
    """The one record the run directories share (attempts merged into one catalogue). Two rules or two products among them
    REFUSE; with ``want`` (a record, e.g. run_record of the configured product) a run under another rule refuses too."""
    by = {}
    for d in run_dirs:
        r = read_run_record(d)
        by.setdefault((r['rule'], r['md5']), (r, []))[1].append(d)
    if len(by) > 1:
        parts = [f'{k[0]} {k[1][:12]} ({len(v[1])}, e.g. {v[1][0]})' for k, v in sorted(by.items())]
        _refuse(f'{what} were searched under {len(by)} size rules: ' + '; '.join(parts) + ' - one rule per catalogue: '
                f're-search the cells')
    rec = next(iter(by.values()))[0] if by else (want or run_record(None))
    if want is not None and by and not same_rule(rec, want):
        _refuse(f'{what} were searched under the size rule {rec["rule"]} {rec["md5"][:12]}, the configuration applies '
                f'{want["rule"]} {want["md5"][:12]} - re-search the cells (or set the rule the attempts were searched under)')
    return rec


def configured_record(cadence):
    """run_record of the configured rule and product of a cadence (the product checked: Envelope.require)."""
    rule, path, md5 = configured(cadence)
    rule_value(rule)
    if rule == 'production':
        return run_record(None)
    if not md5:
        _refuse(f'size rule {rule!r}: no md5 pinned for the envelope {path!r}')
    return run_record(Envelope.load(path, md5).require())


def ledger_rule(path):
    """The rule of a gate ledger from its header - 'envelope' when it ends with GLED_COLS_SIZE (a merged view's ledger: the
    merge refuses attempts of two rules), else production's; a missing file is production's. As run_record."""
    import csv
    if path and os.path.exists(path):
        with open(path, newline='') as f:
            head = next(csv.reader(f), None) or []
        if head[-len(GLED_COLS_SIZE):] == GLED_COLS_SIZE:
            return dict(run_record(None), rule='envelope', md5='', envelope='(from the ledger header)')
    return run_record(None)


# ---------------------------------------------------------------------------------------------------------------------------
# downstream: the catalogue's own size gates under the rule

def catalog_env(rec):
    """The catalogue builder's environment for a set of attempts searched under ``rec``: under 'envelope' its 7-8 sigma gate for
    unpromoted singles (build_source_catalogs_v3.py SIZE_GATE_7_8, 1.7 beams) is replaced by the rule - P3_SIZE_GATE_7_8 = inf,
    the gate never acts; under production nothing (the builder's own 1.7)."""
    return {'P3_SIZE_GATE_7_8': GATE_7_8_OFF} if rec['rule'] == 'envelope' else {}


FIXED_REASONS = ('gaussfit size ', 'axis ratio ')   # production's conformance (p3s2_conform.py / removal_rules.size_axis)


def fixed_rule_removals(removed_rows):
    """The rows a conformance removed by the FIXED size / axis thresholds above 8 sigma (production's p3s2_conform: 'gaussfit
    size ... beams > 2.5 above 8.0 sigma', 'axis ratio ... x beam > 2.0 above 8.0 sigma'). Under 'envelope' the search's rule is
    the one rule: any such removal contradicts it (the caller refuses)."""
    return [r for r in removed_rows if str(r.get('removal_reason') or '').startswith(FIXED_REASONS)]


def fixed_removals_in(paths, what='the conformance'):
    """The words of a refusal when any of the removal records ``paths`` (master_catalog_removed*.csv) holds a removal by the
    FIXED size / axis thresholds (fixed_rule_removals), else None - for a catalogue of attempts searched under 'envelope'."""
    import csv
    bad = []
    for p in paths:
        if os.path.exists(p):
            with open(p, newline='') as f:
                bad += [(os.path.basename(p), r) for r in fixed_rule_removals(csv.DictReader(f))]
    if not bad:
        return None
    f0, r0 = bad[0]
    return (f'{what}: {len(bad)} row(s) removed by the fixed size / axis thresholds of production\'s conformance while the '
            f'attempts were searched under the size / axis rule at every S/N (one rule: {RULING}) - e.g. {f0} '
            f'{r0.get("cand_id", "")}: {str(r0.get("removal_reason", ""))[:120]} - REFUSING')


def final_findings(rows):
    """[(row, 'size' | 'axis', words)] of gate-ledger FINAL rows (final == '1') a run under 'envelope' must not hold: a final
    the rule removes by its own recorded quantities (size_excess or axis_excess > 1, or a removing verdict), or a final the
    rule never judged (no size_verdict). The conformance check reads them from the ledger, never refits."""
    out = []
    for r in rows:
        if str(r.get('final')) != '1':
            continue
        v = r.get('size_verdict') or ''
        if v not in VERDICTS:
            out.append((r, 'size', f'size rule: a final row without a size verdict ({v!r}): the rule never judged it'))
            continue
        try:
            sx = float(r['size_excess']) if r.get('size_excess') not in (None, '') else float('nan')
            ax = float(r['axis_excess']) if r.get('axis_excess') not in (None, '') else float('nan')
        except (TypeError, ValueError):
            sx = ax = float('nan')
        big, long_ = v == 'gaussfit-size' or (np.isfinite(sx) and sx > 1), v == 'axis-ratio' or (np.isfinite(ax) and ax > 1)
        if big or long_:
            out.append((r, 'size' if big else 'axis',
                        f'size rule: a final row the rule removes (verdict {v}, size excess {r.get("size_excess")} of '
                        f'{r.get("size_thr")} beams, axis excess {r.get("axis_excess")} of {r.get("axis_thr")}, at S/N '
                        f'{r.get("size_snr")})'))
    return out
