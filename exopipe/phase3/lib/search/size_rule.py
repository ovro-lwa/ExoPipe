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
'production' keeps the fixed rule above 8 sigma unchanged (the parity file). Under 'envelope' the step first checks the
product (pinned md5, every Stokes and band class), the floors (SIZE_FLOOR_BEAMS / AXIS_REMOVE must be the rule's 1.5:
``check_floors``) and the imaging configuration the product was calibrated on (SIZE_ENVELOPE_FRAMES: ``check_frames``); the
engine writes RUN_RECORD into its run directory (one rule per run) and the gate ledger carries GLED_COLS_SIZE on EVERY
candidate row (spec section 4): size_role 'gate' where the rule decided the row (the detections reaching the size test, and
the Stokes I rows below the local-SNR floor - the multiples fork's 5-7 sigma member pool, ``engine_early``: one the rule
removes is 'reject_local_snr:<verdict>' and never a fork member), 'record' where an earlier gate had removed it (measured,
deciding nothing). The catalog reads the attempts' records (one rule per catalog): its builder's 7-8 sigma gate is replaced
(catalog_env); the conformance's own fixed size step contradicts the rule (refused: ``conform_removals``), and its fixed AXIS
step (axis ratio > 2.0 x beam above 8 sigma, Gregg 2026-09-05, R-AXIS-RATIO-REMOVAL-ABOVE-8SIGMA) follows
catalog_conform.CONFORM_AXIS_UNDER_RULE - PENDING GREGG'S RULING on whether the 2026-10-22 rule replaces it:
    'second_gate' (default until he rules): the step stands as a SECOND, fixed gate above 8 sigma on the rule's attempts, in
                  every conformance - production's p3s2_conform.py (deep 'remove', the multiples fork's validity runs, the
                  incremental path, 10-minute S-G13) and the deep check (lib/catalog/conform_check, which removes those rows
                  itself) - its removals recorded per unit (``conform_removals``), never refused;
    'replace':    the rule replaces the step: production's p3s2_conform.py gets P3_CONFORM_AXIS=0 (``conform_env``) - a switch
                  production does not have yet: the patch, default on and byte-identical, is CONFORM_AXIS_PATCH; until it is
                  applied every stage that would run the script REFUSES (``conform_axis_refusal``), and the deep check keeps
                  and flags the rows. The deep chain's multiples fork runs p3s2_conform.py in place for its validity step
                  (exopipe_ops/multiples_fork.py: conform, table off, on the chain's finals plus its table-cut rows), so the
                  deep catalog depends on the switch as much as the 10-minute S-G13 does.
Otherwise pure functions; the record helpers write only the file they name.

Not wired here: the per-instant carry across bands (``carry_instant``; spec section 1 places it at publish, by position within
the association radius - what it needs and the choices it leaves open are in its docstring; a known deviation until then), the
snapshot tiers (their size gate is clv15 phase3_snapshot_fit.fit_verdict, in place, and no calibration of the 10-s / 1-minute
frames exists) and the forced-photometry flag thresholds.
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
LOW_BANDS = frozenset({'18MHz', '23MHz', '27MHz', '32MHz', '36MHz', 'Red'})   # every other name refuses (band_class)
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
RECORD_VERSION = 'size-rule-2'            # 2 (2026-11-03): every candidate row judged or recorded (size_role); the Stokes I rows
                                          # below the local-SNR floor judged as the fork's member pool. One version per run.
# the gate ledger's columns of the rule (appended to deep_search.GLED_COLS under 'envelope' only; production's ledger is unchanged)
GLED_COLS_SIZE = ['size_rule', 'size_snr', 'size_thr', 'axis_thr', 'size_excess', 'axis_excess', 'size_verdict', 'size_role']
VERDICTS = ('gaussfit-size', 'axis-ratio', 'point-like', 'fit-failed', 'no-snr')   # size_verdict; the first two remove
REMOVING = ('gaussfit-size', 'axis-ratio')
ROLES = ('gate', 'record')                # size_role: the rule decided the row / measured it after an earlier gate removed it
MEMBER_REASON = 'reject_local_snr:'       # + a removing verdict: a Stokes I row below the local-SNR floor the rule removes - not
                                          # in the multiples fork's 5-7 sigma member pool (its veto_reason 'reject_local_snr')
GATE_7_8_OFF = 'inf'                      # P3_SIZE_GATE_7_8 for the catalogue builder under 'envelope': its 7-8 sigma gate never acts
# the conformance's fixed axis step under the rule (catalog_conform.CONFORM_AXIS_UNDER_RULE; module docstring) - PENDING GREGG
CONFORM_AXIS_MODES = ('second_gate', 'replace')
CONFORM_AXIS_ENV = 'P3_CONFORM_AXIS'      # production's switch of p3s2_conform.py's axis step ('0' = off), once the patch is applied
CONFORM_AXIS_PATCH = 'docs/pending_production_patch_p3s2_conform_axis.diff'   # in this repository: the patch, default on
AXIS_STEP_RULING = 'R-AXIS-RATIO-REMOVAL-ABOVE-8SIGMA'   # Gregg 2026-09-05: the conformance's axis-ratio step (2.0 x beam, > 8 sigma)


class RuleRefusal(RuntimeError):
    """The rule cannot be applied as configured (unknown rule, unreadable or unpinned product, a Stokes or band class it does not
    calibrate, a run directory searched under another rule). Its text holds REFUSING, so the engine child exits as refused."""


def _refuse(msg):
    raise RuleRefusal(f'{msg} - REFUSING')


def band_class(band):
    """'high' (41-82 MHz, Green, Blue, Full) or 'low' (18-36 MHz, Red); any other name REFUSES (a typo or a new band would
    otherwise take the low-band calibration silently)."""
    if band in HIGH_BANDS:
        return 'high'
    if band in LOW_BANDS:
        return 'low'
    _refuse(f'size rule: band {band!r} is in neither calibrated band class (low {sorted(LOW_BANDS)}, high {sorted(HIGH_BANDS)})')


def check_floors(size_floor, axis_floor, what='the size / axis rule'):
    """The rule's floors are the ratified 1.5 and 1.5 (R-SIZE-AXIS-GATE-1.5-RATIFIED; SIZE_FLOOR, AXIS_FLOOR), part of the rule
    and of no run record: a configured SIZE_FLOOR_BEAMS / AXIS_REMOVE of another value would be silently ignored under
    'envelope' - REFUSED (a new floor is a new rule: change SIZE_FLOOR / AXIS_FLOOR here and the record's version)."""
    try:
        sf, af = float(size_floor), float(axis_floor)
    except (TypeError, ValueError):
        sf = af = float('nan')
    if sf != SIZE_FLOOR or af != AXIS_FLOOR:
        _refuse(f'{what}: SIZE_FLOOR_BEAMS = {size_floor!r}, AXIS_REMOVE = {axis_floor!r}; under the rule its floors are the '
                f'ratified {SIZE_FLOOR:g} / {AXIS_FLOOR:g} (max(1.5, the envelope)) and another value would not be applied')


def check_frames(product_frames, run_frames, path=''):
    """The imaging configuration the product was calibrated on (config SIZE_ENVELOPE_FRAMES: {key: value, or a tuple of the
    values allowed}) against this search's (``run_frames``, the same keys): a key missing from either side, or another value,
    REFUSES - spec section 3: "a new imaging configuration needs new products before its gates are trusted"."""
    if not isinstance(product_frames, dict) or not product_frames:
        _refuse(f'size envelope {path}: no imaging configuration recorded for it (SIZE_ENVELOPE_FRAMES)')
    bad = []
    for k, want in sorted(product_frames.items()):
        if k not in run_frames:
            bad.append(f'{k}: this search names none')
            continue
        ok = run_frames[k] in want if isinstance(want, (tuple, list, frozenset, set)) else run_frames[k] == want
        if not ok:
            bad.append(f'{k} = {run_frames[k]!r}, calibrated on {want!r}')
    if bad:
        _refuse(f'size envelope {path} was calibrated on frames of another imaging configuration ({"; ".join(bad)}) - a new '
                f'imaging configuration needs a new calibration product before the rule is trusted on it (fp_gates_v2.md '
                f'section 3)')


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
    go, because one of them is removed. Never across instants.

    NOT WIRED (a known deviation from the spec's section 1 and 4 until it is; 2026-10-22 reference: 17 of 7,122 served deep
    detections, 2 I and 15 V, removed by the carry alone). What wiring it at publish needs, and the choices it leaves open:
      - the removals: the merged gate ledger's 'gaussfit-size' / 'axis-ratio' rows (the rule's own; whether the fork member
        pool's 'reject_local_snr:<verdict>' rows - not detections - carry too is open);
      - the match: the same instant (deep: LST and night; 10-minute: the window), another band, within "the tier's association
        radius" - the ledger rows carry no position error, so the radius needs a definition (the builder's K_ASSOC x the
        combined error from the fit's template error model, or a beam); the reference matched by identity (the served sources'
        identity_id), which a removed row has none of; the same Stokes only (the reference) or any (the per-instant ruling's
        "any band, subband, wideband and Stokes") is open;
      - where: every path that serves a detection - the deep check (or 'remove' conformance), the incremental path, 10-minute
        S-G13, AND the multiples fork, which re-reads the raw as-run rows and the ledger in production's code
        (exopipe_ops/multiples_fork.py): a carried detection would still be served by the fork without a production change
        (a list of removals its validity conformance reads) - a second production patch;
      - a reason of its own ('size-carry: ...') declared to the per-instant content test (lib/catalog/tests_per_instant)."""
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
    rec, j = _measure_row(c, stokes, band, img, bmaj_deg, bmin_deg, cd_deg, env, 'gate')
    if rec is None:
        return '', ''
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


def _measure_row(c, stokes, band, img, bmaj_deg, bmin_deg, cd_deg, env, role):
    """(measure_detection record or None, judge's dict) of one candidate, recorded on ``c``: GLED_COLS_SIZE (size_role
    ``role``) and production's size_beams / size_err_beams / axis_norm. An epoch without a beam REFUSES."""
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
    c['size_role'] = role
    if rec is not None:
        c['size_beams'] = round(rec['size'], 2)
        if np.isfinite(rec['err']):
            c['size_err_beams'] = round(rec['err'], 3)
        c['axis_norm'] = round(rec['axis'], 2)
    return rec, j


def engine_record(c, stokes, band, img, bmaj_deg, bmin_deg, cd_deg, env):
    """The rule's measurement on a candidate an EARLIER gate of the search removed (identification, Stokes I's master-structure
    gate): recorded on ``c`` with size_role 'record' - its verdict decides nothing (spec section 4: size, its error, axis,
    S/N, both thresholds and the verdict on every candidate row; Gregg's post-run filtering plan: every filter input kept per
    detection). Returns its size_verdict."""
    return _measure_row(c, stokes, band, img, bmaj_deg, bmin_deg, cd_deg, env, 'record')[1]['size_verdict']


def engine_early(c, reason, stokes, band, img, bmaj_deg, bmin_deg, cd_deg, env):
    """The rule on a candidate the identification removed (``reason``, its veto_reason): a Stokes I row below the local-SNR
    floor only ('reject_local_snr') is the multiples fork's 5-7 sigma member pool (exopipe_ops/multiples_fork.py reads exactly
    those rows), so the rule JUDGES it (size_role 'gate'): removed, it is MEMBER_REASON + the verdict ('reject_local_snr:
    gaussfit-size' / 'reject_local_snr:axis-ratio') and no fork member nor calibration row - "removed at EVERY S/N"; any other
    is recorded only (engine_record). Returns the row's veto_reason."""
    if stokes == 'I' and reason == 'reject_local_snr':
        _rec, j = _measure_row(c, stokes, band, img, bmaj_deg, bmin_deg, cd_deg, env, 'gate')
        return MEMBER_REASON + j['size_verdict'] if j['size_verdict'] in REMOVING else reason
    engine_record(c, stokes, band, img, bmaj_deg, bmin_deg, cd_deg, env)
    return reason


def configured(cadence):
    """(rule, envelope path, md5) in the config section of a cadence: search_detect (deep), search_sublst (10min)."""
    from exopipe.phase3.config import CFG
    sec = CFG.search_detect if cadence == 'deep' else CFG.search_sublst
    return sec.SIZE_RULE, resolve(sec.SIZE_ENVELOPE), sec.SIZE_ENVELOPE_MD5


def step_env(rule, path, md5, what='SIZE_RULE', floors=None, frames=None):
    """The engine child's environment for the rule, from the step's config values ({RULE_ENV[, ENVELOPE_ENV, ENVELOPE_MD5_ENV]});
    under 'envelope' it is checked first - the product present, the pinned md5, every Stokes and band class; ``floors`` (the
    section's SIZE_FLOOR_BEAMS, AXIS_REMOVE) the rule's 1.5 / 1.5 (check_floors); ``frames`` ((the product's recorded imaging
    configuration, this search's)) the same (check_frames) - REFUSED otherwise."""
    rule_value(rule, what)
    if rule == 'production':
        return {RULE_ENV: 'production'}
    if not md5:
        _refuse(f'{what} = {rule!r}: no md5 pinned for the envelope {path!r}')
    path = resolve(path)
    Envelope.load(path, md5).require()
    if floors is not None:
        check_floors(*floors, what=what)
    if frames is not None:
        check_frames(frames[0], frames[1], path)
    return {RULE_ENV: rule, ENVELOPE_ENV: path, ENVELOPE_MD5_ENV: md5}


def step_env_for(cadence, run_frames, what=None):
    """step_env with every check, from the cadence's config section: search_detect (deep) or search_sublst (10min);
    ``run_frames`` this search's imaging configuration (the keys of the section's SIZE_ENVELOPE_FRAMES)."""
    from exopipe.phase3.config import CFG
    sec = CFG.search_detect if cadence == 'deep' else CFG.search_sublst
    what = what or ('search_detect.SIZE_RULE' if cadence == 'deep' else 'search_sublst.SIZE_RULE')
    return step_env(sec.SIZE_RULE, sec.SIZE_ENVELOPE, sec.SIZE_ENVELOPE_MD5, what,
                    floors=(sec.SIZE_FLOOR_BEAMS, sec.AXIS_REMOVE), frames=(sec.SIZE_ENVELOPE_FRAMES, run_frames))


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


def _rule_key(r):
    """What makes two records one rule: the rule, the product's md5 and, under 'envelope', the record's version (what the
    engine wrote per row: RECORD_VERSION). A production record has no version of its own."""
    return (r['rule'], r['md5'], r['version'] if r['rule'] == 'envelope' else '')


def same_rule(a, b):
    return _rule_key(a) == _rule_key(b)


def write_run_record(run_dir, env, ledger='v_deep_ledger.csv'):
    """Under 'envelope', write RUN_RECORD in the engine's run directory. A run directory searched before under another rule or
    product - its record, or a ledger without one (production) - REFUSES: one rule per run (an append would mix two rules in
    one ledger). Under production nothing is written (production's outputs unchanged); a run directory recorded under the
    envelope refuses it the same way. Returns the record."""
    want = run_record(env)
    p = os.path.join(run_dir, RUN_RECORD)
    have = read_run_record(run_dir)
    if (os.path.exists(p) or os.path.exists(os.path.join(run_dir, ledger))) and not same_rule(have, want):
        _refuse(f'{run_dir} was searched under the size rule {_words(have)}; this search applies {_words(want)} - one rule per '
                f'run: search the cell afresh')
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
        by.setdefault(_rule_key(r), (r, []))[1].append(d)
    if len(by) > 1:
        parts = [f'{_words(v[0])} ({len(v[1])}, e.g. {v[1][0]})' for k, v in sorted(by.items())]
        _refuse(f'{what} were searched under {len(by)} size rules: ' + '; '.join(parts) + ' - one rule per catalogue: '
                f're-search the cells')
    rec = next(iter(by.values()))[0] if by else (want or run_record(None))
    if want is not None and by and not same_rule(rec, want):
        _refuse(f'{what} were searched under the size rule {_words(rec)}, the configuration applies {_words(want)} - '
                f're-search the cells (or set the rule the attempts were searched under)')
    return rec


def _words(r):
    """A record in a refusal's words: 'envelope e184ac13f6eb (size-rule-2)' / 'production'."""
    return f'{r["rule"]} {r["md5"][:12]} ({r["version"]})' if r['rule'] == 'envelope' else 'production'


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
        for cols, version in ((GLED_COLS_SIZE, RECORD_VERSION), (GLED_COLS_SIZE_V1, 'size-rule-1')):
            if head[-len(cols):] == cols:
                return dict(run_record(None), rule='envelope', md5='', envelope='(from the ledger header)', version=version)
    return run_record(None)


GLED_COLS_SIZE_V1 = GLED_COLS_SIZE[:7]    # the columns of a size-rule-1 search (8957c42: no size_role)


# ---------------------------------------------------------------------------------------------------------------------------
# downstream: the catalogue's own size gates under the rule

def catalog_env(rec):
    """The catalogue builder's environment for a set of attempts searched under ``rec``: under 'envelope' its 7-8 sigma gate for
    unpromoted singles (build_source_catalogs_v3.py SIZE_GATE_7_8, 1.7 beams) is replaced by the rule - P3_SIZE_GATE_7_8 = inf,
    the gate never acts; under production nothing (the builder's own 1.7)."""
    return {'P3_SIZE_GATE_7_8': GATE_7_8_OFF} if rec['rule'] == 'envelope' else {}


SIZE_STEP_REASON = 'gaussfit size '       # production's conformance (p3s2_conform.py / removal_rules.size_axis): its fixed size
AXIS_STEP_REASON = 'axis ratio '          # step (2.5 beams above 8 sigma) and its fixed axis step (2.0 x beam above 8 sigma)
FIXED_REASONS = (SIZE_STEP_REASON, AXIS_STEP_REASON)


def conform_axis_mode(value=None):
    """catalog_conform.CONFORM_AXIS_UNDER_RULE (``value`` checked): 'second_gate' or 'replace' (module docstring); an unknown or
    empty value REFUSES - a typo never selects either."""
    if value is None:
        from exopipe.phase3.config import CFG
        value = CFG.catalog_conform.CONFORM_AXIS_UNDER_RULE
    if value not in CONFORM_AXIS_MODES:
        _refuse(f'catalog_conform.CONFORM_AXIS_UNDER_RULE = {value!r} is not one of {CONFORM_AXIS_MODES}')
    return value


def conform_axis_step(rule):
    """True when the conformance's fixed axis step (axis ratio > 2.0 x beam above 8 sigma, R-AXIS-RATIO-REMOVAL-ABOVE-8SIGMA)
    acts on a catalog of attempts searched under ``rule`` (a record, or its 'rule'): always under 'production'; under
    'envelope' as catalog_conform.CONFORM_AXIS_UNDER_RULE says ('second_gate': it stands; 'replace': the rule replaced it)."""
    rule = rule_value(rule['rule'] if isinstance(rule, dict) else rule)
    return rule == 'production' or conform_axis_mode() == 'second_gate'


def conform_env(rule):
    """The environment production's p3s2_conform.py gets for a catalog of attempts searched under ``rule``: nothing, unless
    the axis step is off ('envelope' under 'replace'): {P3_CONFORM_AXIS: '0'} - the switch of the pending patch
    (CONFORM_AXIS_PATCH; check it first: conform_axis_refusal)."""
    return {} if conform_axis_step(rule) else {CONFORM_AXIS_ENV: '0'}


def conform_switch(script):
    """True when production's conformance at ``script`` reads the axis switch exactly as the pending patch writes it - the
    first os.environ.get('P3_CONFORM_AXIS', <default>) of its source (ast, never imported), default '1' (on: byte-identical
    production); False when it reads none (production today)."""
    from exopipe.phase3.lib.catalog import deep_binding as _DB
    try:
        return _DB.envdefault(CONFORM_AXIS_ENV)(_DB._tree(script)) == '1'
    except LookupError:
        return False


def conform_axis_refusal(rule, script, what='the conformance'):
    """The words of a refusal when attempts searched under ``rule`` need the conformance's axis step switched off ('envelope'
    under 'replace') and production's p3s2_conform.py at ``script`` has no such switch (or another default); else None. The
    caller refuses BEFORE it runs the script (the multiples fork would otherwise run to the end first)."""
    if conform_axis_step(rule):
        return None
    if os.path.exists(script) and conform_switch(script):
        return None
    return (f'{what}: catalog_conform.CONFORM_AXIS_UNDER_RULE = \'replace\' (the size / axis rule replaces the conformance\'s '
            f'axis-ratio step, {AXIS_STEP_RULING}) needs production\'s {script} to switch that step off ({CONFORM_AXIS_ENV}=0); it '
            f'has no {CONFORM_AXIS_ENV} switch (os.environ.get({CONFORM_AXIS_ENV!r}, \'1\')) - apply {CONFORM_AXIS_PATCH} of the '
            f'refactor repository in exopipe_ops (its owner\'s commit; default on, byte-identical), or set '
            f'catalog_conform.CONFORM_AXIS_UNDER_RULE=second_gate - REFUSING')


def removed_rows(paths):
    """[(file name, row)] of the removal records ``paths`` (master_catalog_removed*.csv), EACH ROW ONCE: the multiples fork
    keeps every conformance run's removals under its own tag (master_catalog_removed_<tag>.csv) beside the last run's
    master_catalog_removed.csv, the same rows twice - a row is its cand_id (else its whole content)."""
    import csv
    seen, out = set(), []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                k = r.get('cand_id') or tuple(sorted(r.items()))
                if k not in seen:
                    seen.add(k)
                    out.append((os.path.basename(p), r))
    return out


def fixed_rule_removals(removed, axis_step=False):
    """The rows a conformance removed by FIXED size / axis thresholds that contradict the rule: every removal by its size step
    ('gaussfit size ... beams > 2.5 above 8.0 sigma'), and by its axis step ('axis ratio ... x beam > 2.0 above 8.0 sigma')
    unless that step stands as the second gate (``axis_step``: conform_axis_step)."""
    kinds = (SIZE_STEP_REASON,) if axis_step else FIXED_REASONS
    return [r for r in removed if str(r.get('removal_reason') or '').startswith(kinds)]


def fixed_removals_in(paths, what='the conformance', axis_step=False):
    """The words of a refusal when the removal records ``paths`` hold a removal that contradicts the rule (fixed_rule_removals;
    each row counted once), else None - for a catalog of attempts searched under 'envelope'."""
    bad = [(f, r) for f, r in removed_rows(paths) if fixed_rule_removals([r], axis_step)]
    if not bad:
        return None
    f0, r0 = bad[0]
    return (f'{what}: {len(bad)} row(s) removed by the fixed size / axis thresholds of production\'s conformance while the '
            f'attempts were searched under the size / axis rule at every S/N (one rule: {RULING}) - e.g. {f0} '
            f'{r0.get("cand_id", "")}: {str(r0.get("removal_reason", ""))[:120]} - REFUSING')


def conform_removals(paths, rule, what='the conformance'):
    """(refusal words or None, record or None) for the removal records ``paths`` a conformance wrote for attempts searched under
    ``rule`` (a record or its 'rule'). Production's attempts: (None, None) - its removals stand. Under 'envelope': a removal by
    the fixed size step refuses (the step is off in every pass, catalog_conform.SIZE_STEP; acting, it contradicts the rule);
    the fixed axis step's removals are the SECOND GATE under 'second_gate' - recorded: {mode, ruling, rows, examples}, rows
    counted once each - and refuse under 'replace' (P3_CONFORM_AXIS=0 was passed: the script did not switch the step off)."""
    if rule_value(rule['rule'] if isinstance(rule, dict) else rule) == 'production':
        return None, None
    axis_on = conform_axis_step(rule)
    why = fixed_removals_in(paths, what, axis_step=axis_on)
    if why:
        if not axis_on:
            why = (f'{why[:-len(" - REFUSING")]} (catalog_conform.CONFORM_AXIS_UNDER_RULE = \'replace\' passed '
                   f'{CONFORM_AXIS_ENV}=0: the script did not switch its axis step off) - REFUSING')
        return why, None
    axis = [(f, r) for f, r in removed_rows(paths) if str(r.get('removal_reason') or '').startswith(AXIS_STEP_REASON)]
    return None, dict(mode=conform_axis_mode(), ruling=AXIS_STEP_RULING, rows=len(axis),
                      examples=[f'{f} {r.get("cand_id", "")}: {str(r.get("removal_reason", ""))[:100]}' for f, r in axis[:3]])


def final_findings(rows):
    """[(row, 'size' | 'axis', words)] of gate-ledger FINAL rows (final == '1') a run under 'envelope' must not hold: a final
    the rule removes by its own recorded quantities (size_excess or axis_excess > 1, or a removing verdict), or a final the
    rule never judged (no size_verdict). The conformance check reads them from the ledger, never refits."""
    out = []
    for r in rows:
        if str(r.get('final')) != '1':
            continue
        v = r.get('size_verdict') or ''
        if v not in VERDICTS or r.get('size_role') == 'record':
            out.append((r, 'size', f'size rule: a final row without a size verdict ({v!r}, size_role '
                                   f'{r.get("size_role")!r}): the rule never judged it'))
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
