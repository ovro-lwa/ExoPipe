"""membership.py - THE co-add membership rulings in one place (Gregg 2026-10-07: "Yes, go but I want the list of every step that was
ratified lest we end up back here again in a week"; ledger APPROVED-FIX-STAGE2-STAGE3-VETOES-AND-RULINGS-REGISTER).

WHY THIS MODULE EXISTS: the stage-3 QA ruling of 2026-08-25/26 ran in August only in side scripts that edited each cell's records
(apply_blessed_drops*.py); the 2026-09-10 rebuild had no stage-3 test, and nights outside the window went back into the masters. Here
the rulings are CODE, read by the co-add (coadd_full_pipeline), by the install (phase3_install) and by the deep search (phase3_deep),
and `verify` REFUSES a master whose member nights disagree with them. (The 2026-10-07 count of "132 stage-2 rejects in the masters"
used the superseded PSF-matched metric and is withdrawn.)

RULINGS ENCODED (fix_ledger ids; Gregg's words in the register RATIFIED_STEPS_REGISTER.csv):
  stage 2 - the NON-MATCHED residual rms of the night against the provisional master, Stokes I (inner 70 %) and Stokes V (inner 50 %),
            coadd_full_pipeline.compute_residual_rms; outliers by the production MAD rule (high side OUTLIER_SIGMA, low side R-GC5),
            union of the Stokes, plus the co-add beam gate (2026-09-02/05); a night whose Stokes I value cannot be measured is refused.
            Canon: PHASE3_ARCHITECTURE.md section 7, amended 2026-09-05 - "there is NO PSF matching anywhere in the production chain"
            (Gregg 2026-09-01 "non-matched throughout is the target"; 2026-09-04 "Non-matched everywhere is correct including for the
            search frame"; R-ISEARCHFRAME-NOPSF-SUPERSEDES-PSFMATCH, R-NONMATCHED-EVERYWHERE-INCL-SEARCH-FRAME). The PSF-matched
            metric of the 2026-08-26 Option-2 ruling is SUPERSEDED; it was wrongly put back here on 2026-10-07 and removed 2026-10-08.
  stage 3 - R-THIRD-STAGE-QA-FLUXALIGN (2026-08-25, "gate refuses, never warns") + R-QA3-BOUNDS-PROCESS-APPROVED (2026-08-26): the
            flux-alignment scale (measure_flux_scale default estimator, measured before flow alignment - the quantity the window was set
            on) inside 0.7-1.6 for >= 41 MHz and every wideband; lower edge 0.35 at 18 MHz, 0.55 at 23 and 27 MHz. 32 and 36 MHz have no
            separate ruling and take the general 0.7 (stated to Gregg 2026-10-07). A night outside leaves the co-add; it is not removed
            from the search by this module (search removal goes through Gregg's data_decisions sign-off, 2026-08-26 flow).
  sunlit  - Gregg 2026-09-26 intent + 2026-10-08 "Correct.": a band-night the solar contamination audit QUARANTINEs leaves the co-add.
  also checked by verify (implemented elsewhere, recorded per cell): exclusions (data_decisions.csv, absolute), the night-image check
            (2026-10-02), the epoch-image check (2026-10-04), grid refusals.

RECORDS: the build's own stage-2 record - <cell>/coadd_final/stage2_verdict.csv (builds from 2026-10-08) or, for a master built
earlier, <cell>/residuals_provisional.csv (the provisional stage has written the same non-matched values there since 2026-08-12).
verify() re-applies the outlier rule to the RECORDED VALUES, so a flag edited by a script cannot decide membership, and reports every
night whose recorded flag disagrees with the rule (stage2_record). A record of another metric is refused.

CLI (production; read-only):
  python -m exopipe.phase3.membership verify <cell_dir>   the content test (membership + products); exit 1 on any violation
  python -m exopipe.phase3.membership stage2 <cell_dir>   the stage-2 record, the rule re-applied, and any disagreement (JSON)
"""
import csv
import json
import os
import sys

import numpy as np

STAGE2_METRIC = 'nonmatched-provisional-v1'
SUPERSEDED_METRICS = ('psfmatched-option2-v1',)   # 2026-10-07 PSF-matched records: refused (canon section 7, 2026-09-05)
QA3_HI = 1.6
QA3_LO = {'18MHz': 0.35, '23MHz': 0.55, '27MHz': 0.55}
QA3_LO_DEFAULT = 0.7
VERDICT_FILE = 'stage2_verdict.csv'
BUILD_RECORD = 'residuals_provisional.csv'
VERDICT_FIELDS = ['date', 'i_rms_mjy', 'v_rms_mjy', 'i_outlier', 'v_outlier', 'beam_outlier', 'outlier',
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


def cell_band(cell_dir):
    lst, band = os.path.basename(cell_dir.rstrip('/')).split('_', 1)
    return lst, band


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def _flag(x):
    return str(x).strip() in ('True', '1', 'true')


def stage2_record(cell_dir):
    """Stage 2 of the build that made the cell's master, with the outlier rule RE-APPLIED to the recorded values.
    Source: coadd_final/stage2_verdict.csv when its metric is STAGE2_METRIC, else the build's <cell>/residuals_provisional.csv when it
    names no other metric (builds before 2026-10-07 wrote no metric column; their values are compute_residual_rms, the ruled metric).
    The rule: stage2_outliers on the recorded Stokes I and V values with the build's beam-gate flag; a non-finite Stokes I value is
    refused. Returns dict(outlier={date: bool}, source, disagree=[...], problem): disagree lists each night whose recorded outlier flag
    differs from the rule re-applied (the record was edited after the build, or made under another rule); problem says why no usable
    record exists (None when one does)."""
    fd = os.path.join(cell_dir, 'coadd_final')
    V = _rcsv(os.path.join(fd, VERDICT_FILE))
    if V and {r.get('metric') for r in V} == {STAGE2_METRIC}:
        src, rows = os.path.join(fd, VERDICT_FILE), V
    else:
        src = os.path.join(cell_dir, BUILD_RECORD)
        rows = _rcsv(src)
        if not rows:
            return dict(outlier={}, source=None, disagree=[],
                        problem=f'no stage-2 record ({VERDICT_FILE} with metric {STAGE2_METRIC}, or {BUILD_RECORD})')
        other = {r.get('stage2_metric') for r in rows} - {None, '', STAGE2_METRIC}
        if other:
            return dict(outlier={}, source=src, disagree=[],
                        problem=f'{BUILD_RECORD} measured with {sorted(other)} - not the ruled non-matched metric')
    dates = [r['date'] for r in rows]
    i_vals = [_num(r.get('i_rms_mjy')) for r in rows]
    beam = {r['date'] for r in rows if _flag(r.get('beam_outlier'))}
    _, _, _, _, out = stage2_outliers(dates, i_vals, [_num(r.get('v_rms_mjy')) for r in rows], beam)
    out |= {d for d, x in zip(dates, i_vals) if not np.isfinite(x)}
    disagree = [f'{r["date"]}: recorded outlier={r.get("outlier")}, rule gives {r["date"] in out}'
                for r in rows if _flag(r.get('outlier')) != (r['date'] in out)]
    return dict(outlier={d: d in out for d in dates}, source=src, disagree=disagree, problem=None)


def verify(cell_dir, exclusions=None):
    """THE CONTENT TEST of a cell's master membership against the rulings. Returns a list of violations (empty = pass).
    A master night (weights.csv in_master 1) must: not be excluded (data_decisions); pass stage 2 (the rule re-applied to the build's
    own non-matched record, stage2_record); lie inside the stage-3 window (the reference night excepted); not be a sunlit night; not FAIL
    the night-image or the epoch-image check. A non-member that passes every check is a violation too (a good night left out)."""
    lst, band = cell_band(cell_dir)
    fd = os.path.join(cell_dir, 'coadd_final')
    W = _rcsv(os.path.join(fd, 'weights.csv'))
    if W is None:
        return [f'{lst} {band}: no weights.csv - membership unknown']
    st2 = stage2_record(cell_dir)
    if st2['problem']:
        return [f'{lst} {band}: stage 2 - {st2["problem"]}']
    S2 = st2['outlier']
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
        if d not in S2:
            why.append('no stage-2 value in the build record')
        elif S2[d]:
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
    if len(argv) < 2 or argv[0] not in ('verify', 'stage2'):
        sys.exit(__doc__)
    cmd, cell = argv[0], argv[1].rstrip('/')
    if cmd == 'stage2':
        r = stage2_record(cell)
        print(json.dumps(dict(cell=os.path.basename(cell), source=r['source'], problem=r['problem'],
                              outliers=sorted(d for d, o in r['outlier'].items() if o), disagree=r['disagree'])))
        return 0 if not r['problem'] else 1
    v = verify(cell) + verify_products(cell)
    print(f'MEMBERSHIP+PRODUCTS {"PASS" if not v else "FAIL"} {os.path.basename(cell)}' + ''.join('\n  ' + x for x in v))
    return 0 if not v else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
