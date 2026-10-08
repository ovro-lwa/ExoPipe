"""membership.py - THE co-add membership rulings in one place (Gregg 2026-10-07: "Yes, go but I want the list of every step that was
ratified lest we end up back here again in a week"; ledger APPROVED-FIX-STAGE2-STAGE3-VETOES-AND-RULINGS-REGISTER).

WHY THIS MODULE EXISTS: the two QA rulings of 2026-08-26 were carried out in August by side scripts that edited each cell's records
(exopipe_ops/reverdict_stage2.py, apply_blessed_drops*.py); the 2026-09-10 rebuild regenerated the records with the old metric and no
stage-3 test, and the vetoed nights went back into the masters (measured 2026-10-07: 132 stage-2 rejects and 30 stage-3 nights in 119
of 456 masters). Here the rulings are CODE, read by the co-add (coadd_full_pipeline), by the install (phase3_install) and by the deep
search (phase3_deep), and `verify` REFUSES a master whose member nights disagree with them.

RULINGS ENCODED (fix_ledger ids; Gregg's words in the register RATIFIED_STEPS_REGISTER.csv):
  stage 2 - R-STAGE2-PSFMATCH+QA3-RANGE-RULED (2026-08-26, "OPTION 2"): the Stokes I metric is the PSF-matched residual rms of the night
            against the master; Stokes V the unmatched night-minus-master rms; outliers by the production MAD rule (high side 3 sigma,
            low side R-GC5), union of the Stokes, plus the co-add beam gate. psfmatched_rms() is the recipe of the August re-verdict that
            Gregg's sign-off tables were built from (reverdict_stage2.py), verbatim: empirical beams, symmetric cross-convolution, second
            scale fit, dilution renormalisation, 8-beam destripe, robust inner rms.
  stage 3 - R-THIRD-STAGE-QA-FLUXALIGN (2026-08-25, "gate refuses, never warns") + R-QA3-BOUNDS-PROCESS-APPROVED (2026-08-26): the
            flux-alignment scale (measure_flux_scale default estimator, measured before flow alignment - the quantity the window was set
            on) inside 0.7-1.6 for >= 41 MHz and every wideband; lower edge 0.35 at 18 MHz, 0.55 at 23 and 27 MHz. 32 and 36 MHz have no
            separate ruling and take the general 0.7 (stated to Gregg 2026-10-07). A night outside leaves the co-add; it is not removed
            from the search by this module (search removal goes through Gregg's data_decisions sign-off, 2026-08-26 flow).
  sunlit  - Gregg 2026-09-26 intent + 2026-10-08 "Correct.": a band-night the solar contamination audit QUARANTINEs leaves the co-add.
  also checked by verify (implemented elsewhere, recorded per cell): exclusions (data_decisions.csv, absolute), the night-image check
            (2026-10-02), the epoch-image check (2026-10-04), grid refusals.

RECORDS: <cell>/coadd_final/stage2_verdict.csv - one row per night: the stage-2 values and verdicts, the metric version, and what the
metric was measured against ('provisional' = the build's provisional master; 'master:<md5 prefix>' = a verification of an installed master
by the dry run). verify() requires it, with metric == STAGE2_METRIC.

CLI (production; read-only unless --write):
  python -m exopipe.phase3.membership verdict <cell_dir> [--write]   stage-2 metric of every night against the cell's installed master
  python -m exopipe.phase3.membership verify  <cell_dir>             the content test; exit 1 on any violation
  python -m exopipe.phase3.membership dryrun  <cell_dir> [--out DIR] [--write-if-consistent]   required vs actual membership (JSON)
  python -m exopipe.phase3.membership adopt   <cell_dir> <verdict_csv>  install a reviewed dry-run verdict when membership agrees
"""
import csv
import hashlib
import json
import os
import sys

import numpy as np

STAGE2_METRIC = 'psfmatched-option2-v1'
QA3_HI = 1.6
QA3_LO = {'18MHz': 0.35, '23MHz': 0.55, '27MHz': 0.55}
QA3_LO_DEFAULT = 0.7
VERDICT_FILE = 'stage2_verdict.csv'
VERDICT_FIELDS = ['date', 'i_match_rms_mjy', 'v_rms_mjy', 'scale2_matched', 'i_outlier', 'v_outlier', 'beam_outlier', 'outlier',
                  'i_thresh_mjy', 'v_thresh_mjy', 'metric', 'measured_against']


def qa3_window(band):
    """(lower, upper) of the ratified stage-3 window for a band label ('18MHz' ... '82MHz', 'Red', 'Green', 'Blue', 'Full')."""
    return QA3_LO.get(band, QA3_LO_DEFAULT), QA3_HI


def qa3_inside(band, scale):
    try:
        s = float(scale)
    except (TypeError, ValueError):
        return False
    lo, hi = qa3_window(band)
    return bool(np.isfinite(s) and lo <= s <= hi)


# SUNLIT NIGHTS (Gregg 2026-09-26: "The intention was to refuse frames with the Sun above -12 deg Evening: refuse frames with the Sun above
# -18 deg. This has led to bad data in the Masters"; 2026-10-08, on keeping the solar-audit quarantined nights out of the masters: "Correct.")
# A band-night whose subband the solar contamination audit QUARANTINEs leaves the co-add; a wideband leaves when any constituent subband is
# quarantined (the is_excluded rule). The audit covers subbands below 46 MHz. Missing audit = REFUSE (a gate never admits what it cannot judge).
SOLAR_AUDIT = '/lustre/pipeline/exopipe/phase3/QA/solar_contamination_audit.csv'
_SOLAR = None


def solar_quarantined(lst, band, date):
    global _SOLAR
    if _SOLAR is None:
        if not os.path.exists(SOLAR_AUDIT):
            raise RuntimeError(f'solar contamination audit {SOLAR_AUDIT} missing - REFUSING (sunlit-night gate, Gregg 2026-10-08)')
        _SOLAR = {}
        for r in _rcsv(SOLAR_AUDIT):
            if r.get('verdict', '').startswith('QUARANTINE'):
                _SOLAR.setdefault((r['lst'], r['date']), set()).add(r['subband'].replace('MHz', ''))
    q = _SOLAR.get((lst, date), set())
    fs = band.replace('MHz', '')
    if fs in ('Red', 'Green', 'Blue', 'Full'):
        from exopipe.phase3.coadd_full_pipeline import _WB_SUBS
        return bool(_WB_SUBS[fs] & q)
    return fs in q


def inner_rms(d, margin):
    """Robust rms of the inner (1 - 2 x margin) of the frame: finite non-zero pixels, 4-MAD clip, standard deviation (reverdict recipe)."""
    h, w = d.shape
    m = int(min(h, w) * margin)
    v = d[m:h - m, m:w - m]
    v = v[np.isfinite(v) & (v != 0)]
    if v.size < 100:
        return float('nan')
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    v = v[np.abs(v - med) < 4 * mad]
    return float(np.std(v))


def _empirical_beam_px(img, hdr_bmaj_px):
    from exopipe.phase3.identify import fit_empirical_beam
    em = fit_empirical_beam(img)
    a = em[0] if np.isfinite(em[0]) and em[0] > 0 else hdr_bmaj_px
    b = em[1] if np.isfinite(em[1]) and em[1] > 0 else a
    return (float(a), float(b), 0.0)


def psfmatched_rms(epoch, master, master_hdr, obs_time, epoch_bmaj_deg=None):
    """Stokes I stage-2 metric (Option 2): (rms in Jy of the PSF-matched night-minus-master residual, second-fit scale).
    epoch and master on the same grid (Jy/beam); master_hdr gives the WCS, CDELT1 and BMAJ."""
    from astropy.convolution import convolve_fft
    from astropy.wcs import WCS
    from exopipe.phase3.phase3_i_psfmatch_ab import gk, dilution
    from exopipe.phase3.coadd_full_pipeline import measure_flux_scale, build_striation_model
    cd = abs(float(master_hdr['CDELT1']))
    w = WCS(master_hdr).celestial
    bm_px = _empirical_beam_px(master, float(master_hdr['BMAJ']) / cd)
    be_px = _empirical_beam_px(epoch, float(epoch_bmaj_deg or master_hdr['BMAJ']) / cd)
    invalid = (epoch == 0) | (master == 0) | ~np.isfinite(epoch)
    Ec = convolve_fft(np.nan_to_num(epoch), gk(*bm_px), allow_huge=True, normalize_kernel=True)
    Mc = convolve_fft(np.nan_to_num(master), gk(*be_px), allow_huge=True, normalize_kernel=True)
    bp_eff = float(np.hypot(be_px[0], bm_px[0]))
    try:
        mask = invalid | ~np.isfinite(Ec) | ~np.isfinite(Mc)
        _, co, ns = measure_flux_scale(Ec, Mc, w, bp_eff, mask, obs_time)
        dc2 = float(co[0])
        if ns >= 8 and 0.5 <= dc2 <= 2.0:
            h_, w_ = master.shape
            yg, xg = np.indices((h_, w_), dtype=float)
            sm = co[0] + co[1] * (2 * xg / w_ - 1) + co[2] * (2 * yg / h_ - 1)
            Ec = np.where(np.abs(sm) > 0.1, Ec / sm, Ec)
    except Exception:
        dc2 = float('nan')
    f = dilution(be_px, bm_px)
    diff = (Ec - Mc) / max(f, 1e-3)
    smod, _ = build_striation_model(np.nan_to_num(np.where(invalid, 0, diff)), w, invalid,
                                    ra_window_beams=8.0 * bp_eff, beam_pix=1.0)
    return inner_rms(np.where(invalid, np.nan, diff - smod), 0.15), dc2


def stage2_outliers(dates, i_vals, v_vals, beam_outliers=()):
    """The production outlier rule on the stage-2 values: coadd_full_pipeline.detect_outliers_mad per Stokes (high side OUTLIER_SIGMA,
    low side R-GC5), union with the beam gate. Returns (i_out, v_out, i_thr, v_thr, all_out)."""
    from exopipe.phase3.coadd_full_pipeline import detect_outliers_mad
    i_out, i_thr = detect_outliers_mad(list(dates), list(i_vals))
    v_out, v_thr = detect_outliers_mad(list(dates), list(v_vals))
    return set(i_out), set(v_out), i_thr, v_thr, set(i_out) | set(v_out) | set(beam_outliers)


def write_verdict(final_dir, rows):
    """Write <final_dir>/stage2_verdict.csv atomically (rows: dicts with VERDICT_FIELDS)."""
    p = os.path.join(final_dir, VERDICT_FILE)
    with open(p + '.new', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=VERDICT_FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    os.replace(p + '.new', p)
    return p


def _rcsv(p):
    if not os.path.exists(p):
        return None
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def _md5_prefix(path):
    from astropy.io import fits
    d = np.squeeze(fits.getdata(path)).astype(np.float32)
    return hashlib.md5(np.ascontiguousarray(np.nan_to_num(d)).tobytes()).hexdigest()[:12]


def cell_band(cell_dir):
    lst, band = os.path.basename(cell_dir.rstrip('/')).split('_', 1)
    return lst, band


def verdict_against_master(cell_dir, write=False, log=print):
    """Stage-2 metric of every night of a cell against its INSTALLED master (dry run / verification): night = master + archived residual
    (subtracted/<date>_I_sub_<band>.fits), exactly as the August re-verdict. Returns the rows; with write=True also writes the record.
    Stokes V: inner rms of the archived V residual (night minus master) at margin 0.25."""
    from astropy.io import fits
    from exopipe.phase3.coadd_full_pipeline import get_midpoint_time
    lst, band = cell_band(cell_dir)
    fd = os.path.join(cell_dir, 'coadd_final')
    mi = os.path.join(fd, f'I_deep_Taper_R075_{band}.fits')
    if not os.path.exists(mi):
        raise RuntimeError(f'{lst} {band}: no installed Stokes I master {mi}')
    with fits.open(mi) as h:
        mI = np.squeeze(h[0].data).astype(np.float32)
        hI = h[0].header.copy()
    W = _rcsv(os.path.join(fd, 'weights.csv')) or []
    dates = [r['date'] for r in W]
    tag = 'master:' + _md5_prefix(mi)
    rows = []
    for d in dates:
        sp = os.path.join(cell_dir, 'subtracted', f'{d}_I_sub_{band}.fits')
        vp = os.path.join(cell_dir, 'subtracted', f'{d}_V_sub_{band}.fits')
        i_rms = v_rms = dc2 = float('nan')
        if os.path.exists(sp):
            with fits.open(sp) as h:
                sub = np.squeeze(h[0].data).astype(np.float32)
                sh = h[0].header.copy()
            if sub.shape == mI.shape:
                try:
                    i_rms, dc2 = psfmatched_rms(mI + sub, mI, hI, get_midpoint_time(sh, sp), sh.get('BMAJ'))
                except Exception as ex:
                    log(f'  {lst} {band} {d}: stage-2 metric failed ({ex})')
        if os.path.exists(vp):
            with fits.open(vp) as h:
                v_rms = inner_rms(np.squeeze(h[0].data).astype(np.float32), 0.25)
        rows.append(dict(date=d, i_match_rms_mjy=1e3 * i_rms, v_rms_mjy=1e3 * v_rms, scale2_matched=dc2))
    from exopipe.phase3.coadd_full_pipeline import coadd_beam_gate
    try:
        beam_out, _ = coadd_beam_gate(lst, band, dates)
    except Exception:
        beam_out = []
    i_out, v_out, i_thr, v_thr, allout = stage2_outliers(dates, [r['i_match_rms_mjy'] for r in rows],
                                                        [r['v_rms_mjy'] for r in rows], beam_out)
    for r in rows:
        r.update(i_outlier=r['date'] in i_out, v_outlier=r['date'] in v_out, beam_outlier=r['date'] in set(beam_out),
                 outlier=r['date'] in allout, i_thresh_mjy=i_thr, v_thresh_mjy=v_thr, metric=STAGE2_METRIC, measured_against=tag)
        for k in ('i_match_rms_mjy', 'v_rms_mjy', 'scale2_matched', 'i_thresh_mjy', 'v_thresh_mjy'):
            r[k] = '' if not np.isfinite(r[k]) else round(float(r[k]), 4)
    if write:
        write_verdict(fd, rows)
    return rows


def verify(cell_dir, exclusions=None, verdict_rows=None):
    """THE CONTENT TEST of a cell's master membership against the rulings. Returns a list of violations (empty = pass).
    A master night (weights.csv in_master 1) must: not be excluded (data_decisions); pass stage 2 in a current-metric record; lie inside
    the stage-3 window (the reference night excepted); not FAIL the night-image or the epoch-image check. A non-member that passes
    every check is a violation too (a good night left out). verdict_rows: evaluate with these stage-2 rows instead of the record
    (the dry run's comparison before anything is written)."""
    lst, band = cell_band(cell_dir)
    fd = os.path.join(cell_dir, 'coadd_final')
    W = _rcsv(os.path.join(fd, 'weights.csv'))
    if W is None:
        return [f'{lst} {band}: no weights.csv - membership unknown']
    V = verdict_rows if verdict_rows is not None else _rcsv(os.path.join(fd, VERDICT_FILE))
    if V is None:
        return [f'{lst} {band}: no {VERDICT_FILE} - stage 2 never measured with the ruled metric']
    bad_metric = {r.get('metric') for r in V} - {STAGE2_METRIC}
    if bad_metric:
        return [f'{lst} {band}: {VERDICT_FILE} metric {sorted(bad_metric)} != {STAGE2_METRIC}']
    S2 = {r['date']: r for r in V}
    NG = {r['date']: r.get('verdict') for r in (_rcsv(os.path.join(fd, 'night_image_gate.csv')) or [])}
    EG = {r['date']: r.get('verdict') for r in (_rcsv(os.path.join(fd, 'epoch_gate.csv')) or [])}
    if exclusions is None:
        from exopipe.phase3.coadd_full_pipeline import load_exclusions
        exclusions = load_exclusions()
    from exopipe.phase3.coadd_full_pipeline import is_excluded
    out = []
    for r in W:
        d = r['date']
        member = r.get('in_master') == '1'
        ref = r.get('qa3_guard') == 'reference-identity'
        why = []
        if is_excluded(lst, band, d, exclusions):   # is_excluded strips 'MHz' itself
            why.append('excluded in data_decisions')
        s2 = S2.get(d)
        if s2 is None:
            why.append('no stage-2 verdict')
        elif s2.get('outlier') in ('True', '1', True):
            why.append('stage-2 outlier')
        if not ref and not qa3_inside(band, r.get('qa3_scale')):
            why.append(f'stage-3 scale {r.get("qa3_scale")} outside {qa3_window(band)}')
        if NG.get(d) == 'FAIL':
            why.append('night-image check FAIL')
        if solar_quarantined(lst, band, d):
            why.append('solar-audit QUARANTINE (sunlit night, Gregg 2026-10-08)')
        if EG.get(d) == 'FAIL':
            why.append('epoch-image check FAIL')
        if member and why:
            out.append(f'{lst} {band} {d}: IN the master but ' + '; '.join(why))
        elif not member and not why:
            out.append(f'{lst} {band} {d}: LEFT OUT of the master but passes every membership ruling')
    return out


# ── WHAT A MASTER MUST CARRY (2026-10-07 rulings audit: these steps run only in campaign scripts, so a plain rebuild would lose them) ──
REQUIRED_FLUXVER = 'fluxcorr-v2-20261001'    # the flux correction table in force (re-bake 2026-10-01); change only with a new ruled table
# the cells the RBF master dewarp was APPLIED to: the archived screen record (phase3_install writes it; applied=1). NOT the 10-01
# acceptance list (179): acceptance was re-measured on the flux-rebaked inputs on 2026-10-04 and 170 passed (Gregg 2026-10-01 "Go with
# (a) and (b')" applied to the re-baked masters) - corrected 2026-10-07 after the first version of this check used the 179
DEWARP_ACCEPTED = '/lustre/pipeline/exopipe/phase3/mdewarp_screens/Run_20260909_000000/mdewarp_screens.csv'
DEWARP_INTERP = 'cubic-spline'                # R-MDEWARP-CUBIC-REAPPLY, Gregg 2026-10-05 "Okay approved"


def verify_products(cell_dir, token='', cell=None):
    """THE PRODUCT TEST of a cell's masters against the rulings that a rebuild must reproduce. Returns violations (empty = pass):
      - the absolute flux scale and band correction are IN THE PIXELS (FLUXBAKE True) with the table in force (FLUXVER) - the
        2026-09-07/11 rulings and the 2026-10-01 re-bake; until 2026-10-07 only campaign scripts applied them;
      - both PB maps exist (one per robust weighting, 2026-10-01);
      - a cell accepted for the RBF master dewarp (2026-10-01) carries it, cubic (2026-10-05).
    cell_dir holds coadd_final/ (a cell, or a generation directory); token = the generation's file-name token ('_mdewarp' in a
    master-dewarp generation: I_deep_Taper_R075_mdewarp_46MHz.fits); cell = '<lst>_<band>' when cell_dir is not named after it."""
    from astropy.io import fits
    lst, band = (cell.split('_', 1) if cell else cell_band(cell_dir))
    fd = os.path.join(cell_dir, 'coadd_final')
    out = []
    for m in (f'I_deep_Taper_R075{token}_{band}.fits', f'V_deep_Taper_R0{token}_{band}.fits'):
        pth = os.path.join(fd, m)
        if not os.path.exists(pth):
            out.append(f'{lst} {band}: master {m} missing')
            continue
        h = fits.getheader(pth)
        if h.get('FLUXBAKE') is not True or not str(h.get('FLUXVER', '')).startswith(REQUIRED_FLUXVER) or h.get('SHSCALE') is None:
            out.append(f'{lst} {band}: {m} does not carry the ruled flux scale (FLUXBAKE {h.get("FLUXBAKE")}, FLUXVER '
                       f'{str(h.get("FLUXVER", ""))[:24]!r}, SHSCALE {h.get("SHSCALE")}; required FLUXBAKE True, {REQUIRED_FLUXVER})')
    for pb, plain in ((f'PB_deep{token}_{band}.fits', f'PB_deep_{band}.fits'), (f'PB_deep_R075{token}_{band}.fits', f'PB_deep_R075_{band}.fits')):
        # a flux-scaled generation keeps the PB maps' plain names (they are not scaled); a dewarp generation tokens them
        if not (os.path.exists(os.path.join(fd, pb)) or os.path.exists(os.path.join(fd, plain))):
            out.append(f'{lst} {band}: PB map {plain} missing')
    acc = {(r['lst'], r['scope']) for r in (_rcsv(DEWARP_ACCEPTED) or []) if r.get('applied', '1') == '1'}
    if not acc:
        out.append(f'{lst} {band}: master-dewarp acceptance list {DEWARP_ACCEPTED} unreadable')
    elif (lst, band) in acc:
        pth = os.path.join(fd, f'I_deep_Taper_R075{token}_{band}.fits')
        if os.path.exists(pth):
            h = fits.getheader(pth)
            if not h.get('MDWARP') or h.get('MDWINTP') != DEWARP_INTERP:
                out.append(f'{lst} {band}: accepted for the RBF master dewarp but the master is not dewarped with {DEWARP_INTERP} '
                           f'(MDWARP {h.get("MDWARP")}, MDWINTP {h.get("MDWINTP")})')
    return out


def unprocessed_nights(cell_dir, exclusions=None):
    """Nights with Phase-2 images for this cell that are not excluded but are absent from the master's record (weights.csv) and from
    its gate records (night_image_gate.csv, refusals_grid.csv): candidates for a night the co-add never saw - a re-admitted night,
    or Phase-2 data that arrived after the build (FINDING-495). A REPORT for the dry run, not a gate: a night the build refused as a
    blank field leaves no record file and shows here too."""
    from exopipe.phase3.coadd_full_pipeline import discover_dates, is_excluded, load_exclusions
    lst, band = cell_band(cell_dir)
    fs = band.replace('MHz', '')
    if exclusions is None:
        exclusions = load_exclusions()
    fd = os.path.join(cell_dir, 'coadd_final')
    seen = {r['date'] for r in (_rcsv(os.path.join(fd, 'weights.csv')) or [])}
    seen |= {r['date'] for r in (_rcsv(os.path.join(fd, 'night_image_gate.csv')) or [])}
    rg = os.path.join(cell_dir, 'refusals_grid.csv')
    if os.path.exists(rg):
        seen |= {ln.split(',')[2] for ln in open(rg) if ln.count(',') >= 3}
    avail = {d for d, _, _ in discover_dates(lst, fs) if not is_excluded(lst, fs, d, exclusions)}
    return sorted(avail - seen)


def main(argv):
    if len(argv) < 2 or argv[0] not in ('verdict', 'verify', 'dryrun', 'adopt'):
        sys.exit(__doc__)
    cmd, cell = argv[0], argv[1].rstrip('/')
    if cmd == 'dryrun':
        # STEP 3 of the 2026-10-07 plan: the stage-2 metric against the installed master, then the membership the rulings require vs
        # the membership the master has. --write-if-consistent writes the verdict record ONLY when they agree (the cell needs no rebuild)
        rows = verdict_against_master(cell, write=False)
        for r in rows:
            for k in ('i_outlier', 'v_outlier', 'beam_outlier', 'outlier'):
                r[k] = str(bool(r[k]))
        v = verify(cell, verdict_rows=rows)
        wrote = False
        if '--out' in argv:   # the computed verdict saved OUTSIDE the cell, for review and a later 'adopt'
            od = argv[argv.index('--out') + 1]; os.makedirs(od, exist_ok=True)
            with open(os.path.join(od, f'{os.path.basename(cell)}.stage2_verdict.csv'), 'w', newline='') as f:
                wcsv = csv.DictWriter(f, fieldnames=VERDICT_FIELDS, extrasaction='ignore'); wcsv.writeheader(); wcsv.writerows(rows)
        if not v and '--write-if-consistent' in argv:
            write_verdict(os.path.join(cell, 'coadd_final'), rows); wrote = True
        print(json.dumps(dict(cell=os.path.basename(cell), nights=len(rows), consistent=not v, wrote=wrote,
                              in_master_vetoed=[x for x in v if ': IN the master' in x],
                              left_out_good=[x for x in v if ': LEFT OUT' in x],
                              stage2_outliers=[r['date'] for r in rows if r['outlier'] == 'True'])))
        return 0
    if cmd == 'verdict':
        rows = verdict_against_master(cell, write='--write' in argv)
        n = sum(1 for r in rows if r['outlier'])
        print(json.dumps(dict(cell=os.path.basename(cell), nights=len(rows), stage2_outliers=n,
                              outliers=[r['date'] for r in rows if r['outlier']], written='--write' in argv)))
        return 0
    if cmd == 'adopt':
        # install a reviewed dry-run verdict as the cell's record - ONLY when the installed master already satisfies every ruling with it
        rows = _rcsv(argv[2])
        if not rows or {r.get('metric') for r in rows} != {STAGE2_METRIC}:
            sys.exit(f'REFUSED: {argv[2]} is not a {STAGE2_METRIC} verdict')
        v = verify(cell, verdict_rows=rows)
        if v:
            print(f'NOT ADOPTED {os.path.basename(cell)}: the master disagrees with the rulings ({len(v)}) - it needs a rebuild')
            return 1
        write_verdict(os.path.join(cell, 'coadd_final'), rows)
        print(f'ADOPTED {os.path.basename(cell)}: {len(rows)} nights; membership verified against the rulings')
        return 0
    v = verify(cell) + verify_products(cell)
    print(f'MEMBERSHIP+PRODUCTS {"PASS" if not v else "FAIL"} {os.path.basename(cell)}' + ''.join('\n  ' + x for x in v))
    return 0 if not v else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
