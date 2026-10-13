#!/usr/bin/env python3
"""
phase3_deep.py — production DEEP transient search (wideband + subband).

Per (lst, scope): reuse B's deep co-add masters (coadd_final/) and B's
flux-scaled + flow-aligned deep epochs (subtracted/{date}_{I,V}_clean). Apply
the VALIDATED ×2-upflow refinement (the one step B's stored residuals skip —
it removes the sub-pixel-misalignment dipoles, the dominant FP class) against
the master, direct-subtract, striation-clean -> residual; then the production
I/V identification -> morphology veto -> survey catalog. No co-add, no
flux-scaling, no native subtraction is redone — only the ×2 refine + detect.

I = direct + sigma_stack + veto;  V = signed ±V + 20×I rule, SAME ×2 flow as I.
"""
from exopipe.phase3.config import coadd_root
import os, sys, glob, re, subprocess, time, csv, fcntl
_ARGV = list(sys.argv)        # captured before lab imports that reset sys.argv
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from exopipe.phase3.lib.masters.coadd import apply_flow_field, build_striation_model
from exopipe.phase3.lib.search import parent_curve_table as _PCT
from exopipe.phase3.lib.search.identification import (build_sigma_stack, run_identification_i,
                             run_identification_v, STRIATION_WINDOW_DEG)
from exopipe.phase3.core.residuals import fft_upsample2, fft_decimate2, write_up_fits
from exopipe.phase3.core.fitting import fit_empirical_beam
from exopipe.phase3.phase3_tenmin import _det_header
from exopipe.phase3.lib.search import veto as pv
from exopipe.phase3.lib.search import candidates as pc
from exopipe.phase3.lib.search import search_v as sv2
from astropy.coordinates import SkyCoord as _SC
_ATEAM_COORDS = _SC([sv2.CONFIG['A_TEAM'][k]['coord'].ra.deg
                     for k in sv2.CONFIG['A_TEAM']],
                    [sv2.CONFIG['A_TEAM'][k]['coord'].dec.deg
                     for k in sv2.CONFIG['A_TEAM']], unit='deg')
from scipy.ndimage import uniform_filter
from astropy.time import Time
from astropy.coordinates import get_body, SkyCoord
from exopipe.phase3.lib.masters.coadd import OVRO_LOC, load_exclusions, is_excluded

# ── FINAL V-deep search inventory (Gregg 2026-06-30; minimal veto, no silent gates) ──
# Applied IN-ORDER, all enforced by construction here (no hand-run side scripts):
#   masks: horizon / A-team / <20deg elevation (NOT searched -> excluded, not "missed")
#   BDSF extraction at CANDIDATE_SIGMA (=5)
#   WRITE FLOOR: local-SNR >= 5  (retain >=5sigma for repeaters/cross-match)
#   drop two-stage-QA outlier epochs   (ledger problem 9)
#   mstruct gate: |local_snr| >= 5*max(mstruct,1) on the destriped deep V master  -- the
#       SOLE structure veto; carries the FP rejection (veto-ROC 2026-06-30)
#   morphology veto (apply_veto: bilobe/ghost + axis-ratio/size) -- helps Green, ~free in Red
#   Jupiter ephemeris veto, Red + Full 5deg (mstruct can't catch a moving source)
# REMOVED 2026-06-30 (veto-ROC): the 20x I-leakage rule -- zero marginal FP value over
#   mstruct at high mult, severe injection loss at low mult (Red 20x: 103 inj lost / cell).
#   run_identification_v(i_leakage_mult=None) => OFF.
# NOT A GATE: corrupt-epoch count is RECORDED per row (n_v_epoch), never dropped (Gregg
#   2026-06-30 reverted the old drop). sstack is QUOTED (info-only), not gated.
SIGMA_MSTRUCT = 5.0
STRIATION_WINDOW_BEAMS = 8.0   # R-STRIATION-WINDOW-APPROVED-8BEAMS (2026-08-25)
JUPITER_VETO_DEG = {'Red': 5.0, 'Green': 0.5, 'Blue': 0.5, 'Full': 5.0}
# A-team veto stage: historical 2.0 deg / 5 sources kept for reference;
# the RATIFIED table (PHASE3_ARCHITECTURE.md §10.1 Card 6, 2026-08-11) is ONE
# six-source list (adds Her A — absent by omission, not ruling), V radius
# 3.0 deg / I radius 5.0 deg, strictly DOWNSTREAM of mstruct (G-03: no
# pre-detection blank). The V redo path (G-09, Gregg 2026-08-21) uses the
# ratified table below.
ATEAM_VETO_DEG = 2.0
ATEAM_VETO_DEG_V = 3.0
ATEAM_VETO_DEG_I = 5.0        # Card 6 ratified table; I downstream veto landed 2026-09-04
# Gaussian-fit SIZE removal above 8 sigma (Gregg 2026-09-04: 2.5 beams; Card 9
# allowance: fitted FWHM minus its 1-sigma uncertainty, against the EPOCH's own
# beam major axis from its Stage-1 product header). 2.0-2.5 beams flagged only.
# TIGHTENED 2026-09-08 (Gregg approved) from 2.5 after FIX-DEEP-V-SIZE-UNITS: the old floor
# was loose because Stokes V's fitted sizes were inflated by sec(zenith angle) (the SIN-projection
# units bug) and the constant is shared by both Stokes. With units correct, real unresolved field
# sources sit at p99 = 1.45 beams and every known real transient at 0.76-1.31.
SIZE_FLOOR_BEAMS = float(os.environ.get('P3_SIZE_FLOOR_BEAMS', '1.5'))
SIZE_FLAG_BEAMS = 1.3
SANITY_I_X = float(os.environ.get('P3_SANITY_I_X', '3.0'))   # I-only search sanity gate (Gregg 2026-09-05)
ROWLINE_MIN = 5                                              # I finals on one pixel row +/-1 -> all removed (Gregg 2026-09-05)
AXIS_REMOVE = float(os.environ.get('P3_AXIS_REMOVE', '1.5'))   # Gregg 2026-09-05, tightened 09-08 (same reasoning)
AXIS_FLAG = 1.3
SIZE_SNR_GATE = 8.0
# Flood keying for the I 2-D curve (R-IVETO-FLOOD-CURVE-RATIFIED 2026-08-31):
# a night is a flood night when its post-mstruct I candidate count exceeds the
# 90th percentile of its band x tier population. Reference population frozen
# 2026-09-04 from the P3S2 run's v_deep_images.csv (n_after_morph, I rows,
# duplicates removed; wb=wideband deep, sb=subband deep): re-derive from P3S3
# after its Stage 2 and record the change. Blue p90 = 1 is degenerate (any
# night with >= 2 post-mstruct candidates keys the flood curve) - reported.
# RETIRED 2026-09-05/06 with the flood-night curve it keyed, and unused since; the measured per-Stokes
# per-band table replaced it. Kept only as the record of the calibrated deep values. Gregg 2026-09-13 on
# the 10-minute tier: "there is no threshold until we review battery results."
FLOOD_P90_POSTMSTRUCT = {('Red', 'wb'): 8, ('Green', 'wb'): 9, ('Blue', 'wb'): 1,
                         ('Full', 'wb'): 8, ('Red', 'sb'): 3, ('Green', 'sb'): 2,
                         ('Blue', 'sb'): 1}
_BAND_CLASS = {'18': 'Red', '23': 'Red', '27': 'Red', '32': 'Red', '36': 'Red',
               '41': 'Green', '46': 'Green', '50': 'Green', '55': 'Green',
               '59': 'Green', '64': 'Blue', '69': 'Blue', '73': 'Blue',
               '78': 'Blue', '82': 'Blue'}
_ATEAM6_COORDS = _SC(
    ['23h23m24s', '19h59m28s', '05h34m31s', '12h30m49s', '09h18m05.7s',
     '16h51m08.15s'],
    ['+58d48m54s', '+40d44m02s', '+22d00m52s', '+12d23m28s', '-12d05m44s',
     '+04d59m33.3s'])   # Cas A, Cyg A, Tau A, Vir A, Hya A, Her A
# Moon ephemeris removal veto, 15 arcmin, per-epoch evaluation MANDATORY
# (Card 7c RATIFIED 2026-08-11; the Moon moves ~0.5 deg/hr). All bands, deep.
MOON_VETO_DEG = 0.25
MAX_V_PER_EPOCH = 10        # threshold for the n_v_epoch HIGH-COUNT flag (NOT a drop)

# ── V-PARITY GATE LEDGER (Gregg 2026-07-11: "record everything for posterity") ──
# Mirrors the Stokes-I i_deep_ledger/i_deep_images pattern: EVERY identified
# >=5sigma candidate gets a ledger row with each gate's verdict (vetoed
# candidates were previously discarded); plus one per-image row of counts at
# each gate. Appended with an fcntl lock (parallel cells), into the run root
# next to master_catalog.csv. Cells searched before 2026-07-11 ~18:30 UTC
# predate this and have no rows (see run-root README_gate_ledger.txt).
GLED_COLS = ['lst', 'scope', 'stokes', 'date', 'ra', 'dec', 'pix_x', 'pix_y',
             'snr', 'peak_jy', 'morph_pass', 'veto_reason',
             'mstruct', 'mstruct_raw', 'pass_mstruct', 'sstack',
             'jupiter_deg', 'jupiter_veto', 'final', 'n_v_epoch', 'moon_deg',
             # 2026-09-04 additions (audit R-14, R-09, size ruling, I A-team):
             'global_snr', 'size_beams', 'size_flag', 'nearest_ateam',
             'fp_prediction',
             # 2026-09-08: the axis gate's own quantity and the size fit's 1-sigma, so a
             # removal can be re-adjudicated from the ledger without re-searching the cell
             'axis_norm', 'size_err_beams']
GIMG_COLS = ['lst', 'scope', 'stokes', 'date', 'n_ident5', 'n_after_morph',
             'n_after_mstruct', 'n_after_jupiter', 'n_final',
             'n_raw_floor', 'elev_min_deg']   # 2026-09-04: raw >=floor count; elevation gate
# P3_GATE_LEDGER_ONLY=1: write ONLY the gate-ledger CSVs, skip master_catalog
# + candidate cutouts. Used by the posterity BACKFILL over already-searched
# cells (Gregg 2026-07-11) so re-runs never duplicate catalog rows.
GATE_LEDGER_ONLY = os.environ.get('P3_GATE_LEDGER_ONLY', '') == '1'


def _elev_floor(stokes):
    """The elevation floor (deg) the detection mask of this Stokes actually used - recorded per night in v_deep_images.csv
    elev_min_deg (2026-10-07: it was a literal 20.0 whatever the floor was)."""
    if stokes == 'I':
        from exopipe.phase3.lib.search import search_i as _sitm
        return float(_sitm.CONFIG['MIN_ELEVATION'])
    return float(sv2.CONFIG['MIN_ELEVATION'])


def _append_locked(path, cols, rows):
    """Lock-serialized CSV append (same pattern as phase3_i_deep)."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lk = open(path + '.lock', 'w')
    fcntl.flock(lk, fcntl.LOCK_EX)
    try:
        new = not os.path.exists(path)
        with open(path, 'a', newline='') as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(cols)
            w.writerows(rows)
    finally:
        fcntl.flock(lk, fcntl.LOCK_UN)
        lk.close()


def _gled_row(lst, scope, stokes, d, c, morph_pass, reason='', pass_ms='',
              jup_veto='', final=0):
    snr = c.get('local_snr', c.get('snr', ''))
    ra = dec = ''
    if c.get('ra') is not None:
        ra, dec = round(float(c['ra']), 5), round(float(c['dec']), 5)
    elif c.get('coord') is not None:
        ra = round(float(c['coord'].ra.deg), 5)
        dec = round(float(c['coord'].dec.deg), 5)
    return [lst, scope, stokes, d, ra, dec,
            int(c.get('pix_x', -1)), int(c.get('pix_y', -1)),
            round(float(snr), 2) if snr != '' else '',
            round(float(c.get('flux', c.get('peak_jy', 0))), 3),
            int(morph_pass), reason,
            c.get('mstruct', ''), c.get('mstruct_raw', ''), pass_ms,
            c.get('sstack', ''), c.get('jupiter_deg', ''), jup_veto,
            int(final), c.get('n_v_epoch', ''), c.get('moon_deg', ''),
            (round(float(c['global_snr']), 2) if c.get('global_snr') not in (None, '') else ''),
            c.get('size_beams', ''), c.get('size_flag', ''),
            c.get('nearest_ateam', ''), c.get('fp_prediction', ''),
            c.get('axis_norm', ''), c.get('size_err_beams', '')]


def _master_structure_map(vm, bpix, win_beams=4.0, valid=None, report=None):
    """Local rms of the V master normalized to clean-sky (>=1). Inlined from
    combined_transient_list (which imports phase3_deep, so can't import back).

    valid : imaged-pixel mask. MUST BE SUPPLIED whenever `vm` is a DIFFERENCED array
        (problem-ledger P12, Gregg-approved 2026-08-18). The running-median striation
        model reproduces its input EXACTLY on ~0.8% of pixels, so inferring validity
        from `vm != 0` on a difference reads those pixels as unimaged and hands them
        the floor value 1.0 — which sets the gate to its weakest possible setting at
        precisely the positions the gate exists to protect. The default reproduces the
        historical behaviour and is correct only when `vm` is a master.
    report : optional dict; receives 'n_invalid' so callers can verify that the map
        agrees with the master about which pixels are imaged.
    """
    if not np.isfinite(bpix) or bpix <= 0:   # empirical-beam fit can return NaN (13h/14h Blue/Full)
        bpix = 7.0
    if valid is None:
        valid = np.isfinite(vm) & (vm != 0)
    if report is not None:
        report['n_invalid'] = int((~valid).sum())
    w = max(int(win_beams * bpix), 5)
    vf = np.where(valid, vm, 0.0); m1 = uniform_filter(vf, w); m2 = uniform_filter(vf * vf, w)
    locrms = np.sqrt(np.maximum(m2 - m1 * m1, 0)); n = locrms.shape[0]; q = n // 4
    cen = locrms[q:3 * q, q:3 * q]; sky = np.median(cen[cen > 0]) if np.any(cen > 0) else 1.0
    m = locrms / max(sky, 1e-9); m[~valid] = 1.0; return np.maximum(m, 1.0)


def _mstruct_validity_guard(report, vmask, mp, tag):
    """REFUSE (CLAUDE.md §3) if a structure map disagrees with its master about which
    pixels are imaged. Trips the moment the `valid=` argument is dropped again, which
    is how problem-ledger P12 went unnoticed: 0.6-1.0% of imaged pixels were silently
    reclassified as blank sky and handed the gate floor."""
    n_master = int((~vmask).sum())
    n_map = int(report.get('n_invalid', -1))
    if n_map != n_master:
        raise RuntimeError(
            f'[mstruct-{tag}] VALIDITY GUARD REFUSED for {mp}: the structure map treats '
            f'{n_map:,} pixels as unimaged but the master has {n_master:,}. Validity must '
            f'come from the master, never be inferred from a differenced array (P12).')


def _mstruct_maps(master_V, mVp, beam):
    """(destriped, raw) mstruct from the deep V master — Gregg's validated gate.
    Falls back to raw if destriping fails (raw alone already separates sidelobes from
    real transients in validation).

    Validity comes from the MASTER for both maps (problem-ledger P12)."""
    if not np.isfinite(beam) or beam <= 0:
        beam = 7.0
    vmask = np.isfinite(master_V) & (master_V != 0)
    raw = _master_structure_map(master_V, beam, valid=vmask)
    try:
        vh = fits.getheader(mVp); mwcs = WCS(vh).celestial
        _hm, _am, iwm, _ot, _ac, _el = sv2.make_masks(vh, master_V.shape, fits_path=mVp)
        sm, _ = build_striation_model(master_V, mwcs, iwm,
                                      ra_window_beams=STRIATION_WINDOW_BEAMS, beam_pix=beam)
    except Exception as e:
        print(f'    [mstruct] destripe failed ({e}); using raw mstruct')
        return raw, raw
    dif = master_V - sm
    n_leak = int((vmask & (dif == 0)).sum())
    if n_leak:
        print(f'    [mstruct] P12: {n_leak:,} imaged pixels '
              f'({100.0 * n_leak / max(int(vmask.sum()), 1):.2f}%) are reproduced exactly '
              f'by the striation model; validity taken from the master')
    rep = {}
    destriped = _master_structure_map(dif, beam, valid=vmask, report=rep)
    _mstruct_validity_guard(rep, vmask, mVp, 'V')
    return destriped, raw


def _qa_outlier_dates(lst, scope):
    """Dates the two-stage QA flagged outlier==True — must NOT be searched (problem 9)."""
    p = f'{coadd_root()}/{lst}_{scope}/residuals_provisional.csv'
    if not os.path.exists(p):
        return set()
    return {r['date'] for r in csv.DictReader(open(p))
            if str(r.get('outlier')).strip().lower() == 'true'}


def _jupiter_skycoord(date):
    """Apparent ra/dec of Jupiter (strip distance -> avoids the separation(get_body)
    barycentric-parallax inflation)."""
    jb = get_body('jupiter', Time(date + ' 00:00:00', scale='utc'), OVRO_LOC)
    return SkyCoord(jb.ra, jb.dec)


def _body_skycoord(body, obs_time_str, date):
    """Apparent ra/dec of a solar-system body at the EPOCH's own observation
    time (Card 7a RATIFIED 2026-08-11: per-epoch obs_time for ALL ephemeris
    evaluations; the fixed-00:00 convention is retired on this path). Falls
    back to 00:00 UTC of the date only when the epoch header carries no
    DATE-OBS. Distance stripped -> avoids the separation(get_body)
    barycentric-parallax inflation."""
    try:
        t = Time(obs_time_str, scale='utc') if obs_time_str else \
            Time(date + ' 00:00:00', scale='utc')
    except Exception:
        t = Time(date + ' 00:00:00', scale='utc')
    b = get_body(body, t, OVRO_LOC)
    return SkyCoord(b.ra, b.dec)

# GC-PRE-3 (Gregg 2026-08-12): flow-align script is the repo's own copy
# (exopipe/phase2/align_images.py, byte-identical to the frozen clv15
# original) — never execute frozen-tree code from the production path.
# Shelled into py310_orca (JAX), never imported.
ALIGN = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                     'phase2', 'align_images.py')
WORKROOT = os.environ.get('P3_FLOW_ROOT', '/data07/gh/phase3/deep')   # moved off /fast (NVMe too small; was filling it)
COADD_ROOT = os.environ.get('P3_COADD_ROOT')   # NO default (Gregg 2026-09-04); use coadd_root() at point of use
MIN_EPOCHS = int(os.environ.get('P3_MIN_EPOCHS', '3'))   # Gregg 2026-09-11: 3 for this run (was 5); thin 00h cells searched
# DEEP (2026-10-07, rulings audit): Gregg's 2026-09-11 "Lower MIN_EPOCHS to 3 for this run" was scoped to that run (ledger: "a later
# run reverts to 5 unless re-ruled") and was never reverted. The deep cadence is back at 5 (on 2026-10-07 every searched deep cell had
# 5 or more nights, so nothing changes today); the 10-minute cadence keeps MIN_EPOCHS (the snapshot / sub-LST chat's).
# GREGG 2026-10-11, verbatim: "The deep search needs at least 5?! Thats nonsense and not something I would have ratified. If theres
# a Master, you only need 1 night to do a search." - the deep cadence searches any cell with one or more epochs (the 5 was a rider of
# the search design, never a ruled threshold: FINDING-THIN-CELLS-POLICY-SKIPPED-BY-MIN-EPOCHS-5-AFTER-FLOOR-WAIVER-FOR-GREGG).
DEEP_MIN_EPOCHS = int(os.environ.get('P3_DEEP_MIN_EPOCHS', '1'))
NPROC_RESID = int(os.environ.get('DEEP_NPROC_RESID', '6'))   # epoch-parallel residual build


def _load(p):
    a = fits.getdata(p).squeeze().astype(float)
    # zap non-finite + corrupt pixels (real sky is << 1e4 Jy/beam; some B
    # co-adds carry a single garbage pixel ~1e26 that breaks the beam fit)
    a[~np.isfinite(a)] = 0.0
    a[np.abs(a) > 1.0e6] = 0.0   # zap garbage pixels; well above any real source
    return a                     # (Cyg A ~1.7e4 Jy at low freq is kept)


def pick_i_master(base, scope):
    """Path of a SANE deep I master for flow-reference / beam / co-located-I use.
    Prefers Taper_R0; falls back to Taper_R075 when the R0 co-add is corrupt
    (13h Blue/Full + 14h Full: the Phase-2 wideband I Taper_R0 stacks themselves
    are garbage at ~1e20-1e30, so the R0 master cannot be rebuilt — 2026-07-02).
    Sanity = >=10% of finite pixels survive the 1e6 garbage zap."""
    for var in ('R0', 'R075'):
        p = f'{base}/coadd_final/I_deep_Taper_{var}_{scope}.fits'
        if not os.path.exists(p):
            continue
        a = _load(p)
        if (a != 0).sum() >= 0.1 * a.size:
            if var != 'R0':
                print(f'    [pick_i_master] Taper_R0 corrupt/absent -> using Taper_{var}')
            return p
    return None


def _epoch_time(base, date, scope):
    try:
        return fits.getheader(
            f'{base}/subtracted/{date}_I_sub_{scope}.fits').get('DATE-OBS')
    except Exception:
        return None


def _batch_flows(cell_work, mup, items):
    """Per-cell-serialised wrapper around _batch_flows_unlocked.

    RACE FIXED 2026-09-07 (INCIDENT-FLOW-CACHE-TRUNCATED-BY-CONCURRENT-STOKES): the search
    fleet claims cells PER STOKES (claims_I/<cell> and claims_V/<cell> are separate), so one
    node can search the same cell in Stokes I and Stokes V at the same time, and both share
    cell_work. Both could see a flow file missing, both launch the GPU batch, and both write
    the same ep_*_flow.npy - a reader then got a partially written file and np.load raised
    "cannot reshape array of size 0 into shape (6244,6244,2)" (07h 82MHz on lwacalim02).
    The lock covers compute AND load, so a reader never sees a half-written cache."""
    os.makedirs(cell_work, exist_ok=True)
    with open(f'{cell_work}/.flow.lock', 'w') as _lk:
        fcntl.flock(_lk, fcntl.LOCK_EX)
        try:
            return _batch_flows_unlocked(cell_work, mup, items)
        finally:
            fcntl.flock(_lk, fcntl.LOCK_UN)


def _batch_flows_unlocked(cell_work, mup, items):
    """items=[(date, ep_up_path)] -> {date: ×2 flow}, via align_images.py
    (py310_orca GPU). Cached on disk; only missing pairs are recomputed."""
    pairs = []
    for date, ep_up in items:
        fb = os.path.splitext(os.path.basename(ep_up))[0]
        if not os.path.exists(f'{cell_work}/{fb}_flow.npy'):
            pairs.append(f'{ep_up} {mup} {cell_work}')
    if pairs:
        pp = f'{cell_work}/flow_pairs_{os.getpid()}.txt'   # v2026-09-07: was a fixed name, overwritten by the other Stokes' worker
        with open(pp, 'w') as fh:
            fh.write('\n'.join(pairs) + '\n')
        env = {**os.environ,
               'XLA_FLAGS': '--xla_gpu_enable_command_buffer=',  # no CUDA graphs (OOM on ×2)
               'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
               # 0.5 (not 0.9) so 2 cells can share a GPU under batch parallelism;
               # the ×2 Brox flow fits in ~half a 16 GB A4000.
               'XLA_PYTHON_CLIENT_MEM_FRACTION': os.environ.get('DEEP_XLA_MEMFRAC', '0.5')}
        # ABSOLUTE env python, never ambient `conda run` (2026-08-31: in
        # non-interactive shells `conda` resolved to /opt/miniconda3, whose
        # STALE py310_orca lacks image_plane_correction — every t2b fleet
        # cell died at this call while interactive tests passed; two condas,
        # two same-named envs). Override via P3_FLOW_PYTHON if the install
        # moves.
        flow_py = os.environ.get(
            'P3_FLOW_PYTHON', '/opt/devel/pipeline/envs/py310_orca/bin/python')
        # 2026-09-13: ONE flow batch per node at a time (flock on a node-local lock file; the GPU aborted with
        # rc=-6 when several searches ran their flow batches together). Serialisation only - no numeric change.
        _lock = os.environ.get('P3_FLOW_LOCK', '/fast/gh/main/phase3/.flow_batch.lock')   # node-global (one GPU per node): every search family shares it
        r = subprocess.run(['flock', '-w', '86400', _lock, flow_py, ALIGN,
                            '--pairs', pp, '--no_qa', '--save_flow'],
                           capture_output=True, text=True, env=env)
        if r.returncode not in (0, 2):
            # REFUSAL (2026-08-22, after 17 wideband + 9 subband cell-bands
            # were marked done with ZERO searched epochs): a crashed flow
            # batch (GPU OOM etc.) must FAIL the cell, never degrade to
            # 'only 0 flows — skip' with exit 0. Gates refuse, they do not
            # silently no-op (CLAUDE.md §3).
            print(f'    flow batch rc={r.returncode}: {r.stderr[-400:]}\n'
                  f'    flow batch stdout tail: {r.stdout[-600:]}', flush=True)
            raise RuntimeError(f'flow batch failed rc={r.returncode} '
                               f'(see log above); cell must be rerun')
    flows = _Lazy() if _STREAM else {}
    for date, ep_up in items:
        fb = os.path.splitext(os.path.basename(ep_up))[0]
        fp = f'{cell_work}/{fb}_flow.npy'
        if os.path.exists(fp):
            if _STREAM:
                dict.__setitem__(flows, date, (lambda _fp=fp: np.load(_fp)))
            else:
                flows[date] = np.load(fp)
    return flows


# STREAMING CHAIN (Gregg 2026-09-12, order agreed; proposed in CARD_TENMIN_BATTERY_20260908 section 7): with
# P3_STREAM_RESID=1 the per-epoch x2 flow fields (~0.5 GB each), the epoch reconstructions and the residuals are
# NOT held in memory for every epoch of the cell; each is loaded (flows, epochs) or spilled to the node-local cell
# work dir and reloaded (residuals) on demand. Every number is computed by the same functions on the same arrays,
# so the output is bit-identical to the in-memory path; only the footprint changes (measured 75-110 GB per
# 10-minute search with ~150 frames held). Default OFF until validated on a cell; the deep fleet is untouched.
_STREAM = os.environ.get('P3_STREAM_RESID') == '1'
_SPILLED = []


class _Lazy(dict):
    """dict of key -> zero-argument loader; item access runs the loader (nothing cached)."""
    def __getitem__(self, k):
        return dict.__getitem__(self, k)()
    def get(self, k, default=None):
        return self[k] if k in self else default
    def items(self):
        return ((k, self[k]) for k in dict.keys(self))
    def values(self):
        return (self[k] for k in dict.keys(self))


def _spill(arr, path):
    np.save(path, arr); _SPILLED.append(path + ('' if path.endswith('.npy') else '.npy'))
    return path + ('' if path.endswith('.npy') else '.npy')


def _cleanup_spills():
    for p in _SPILLED:
        try:
            os.remove(p)
        except OSError:
            pass


import atexit as _atexit
_atexit.register(_cleanup_spills)

_STRIATION_OFF = os.environ.get('P3_STRIATION_OFF') == '1'   # battery variable (Gregg 2026-09-08); production default ON
_REFINE_OFF = os.environ.get('P3_REFINE_OFF') == '1'
SEARCH_TIER_SIGMA = float(os.environ.get('P3_SEARCH_TIER_SIGMA', '5.0'))   # R-GC10 Layer-2 search tier: 4.0 (Gregg 2026-09-09) -> 5.0 (Gregg 2026-09-13 "Change to MAD 5 for exclusion from transient search")         # battery variable: no per-frame x2 flow refinement (night flow only)


def _mid_hour_elev(lst, ra, dec):
    """Elevation (deg) at the deep frame's mid-hour (LST + 0.5 h) at OVRO: the elevation the deep bright-source band tables were
    derived at (p3s3_curve_derive.elev_at, key 'mid') and are applied at in the conformance (p3s2_conform, P3_TABLE_ELEV=1)."""
    lstdeg = 15.0 * (int(lst[:-1]) + 0.5); lat = np.radians(37.2398)
    hr = np.radians(lstdeg - np.asarray(ra, float)); dr = np.radians(np.asarray(dec, float))
    return np.degrees(np.arcsin(np.sin(dr) * np.sin(lat) + np.cos(dr) * np.cos(lat) * np.cos(hr)))


def _residual(epoch, master, flow, hdr, cd):
    """×2-upflow refine epoch -> direct subtract master -> striation clean."""
    up = epoch if _REFINE_OFF else fft_decimate2(apply_flow_field(fft_upsample2(epoch), flow))
    invalid = (up == 0) | (master == 0) | ~np.isfinite(up)
    if _STRIATION_OFF:
        return np.where(invalid, np.nan, up - master)
    w = WCS(hdr).celestial
    # window = 8 x beam (R-STRIATION-WINDOW-APPROVED-8BEAMS 2026-08-25;
    # supersedes STRIATION_WINDOW_DEG=4.375). BMAJ from the master header.
    _win_deg = 8.0 * float(hdr.get('BMAJ', STRIATION_WINDOW_DEG / 8.0))
    smodel, _ = build_striation_model(np.nan_to_num(up - master), w, invalid,
                                      ra_window_beams=_win_deg / cd,
                                      beam_pix=1.0)
    return np.where(invalid, np.nan, (up - master) - smodel)


_RESID_CTX = {}


def _resid_task(d):
    c = _RESID_CTX
    try:
        m = c['master'][d] if isinstance(c['master'], dict) else c['master']
        return d, _residual(c['epoch'][d], m, c['flows'][d], c['hdr'], c['cd'])
    except Exception:
        return d, None


def _resid_task_spill(d):
    """Streaming variant of _resid_task: the residual is written to the node-local spill dir and its
    path returned (the parent reloads on demand). Same _residual call, same arrays."""
    c = _RESID_CTX
    try:
        m = c['master'][d] if isinstance(c['master'], dict) else c['master']
        r = _residual(c['epoch'][d], m, c['flows'][d], c['hdr'], c['cd'])
        return d, _spill(r, f"{c['spill']}/resid_{c['tag']}_{d}_{os.getppid()}")
    except Exception:
        return d, None


def _build_residuals_mp(epoch, master, flows, hdr, cd, dates, nproc, spill=None, tag=''):
    """Build per-epoch residuals in parallel. Uses fork so the big master +
    flow arrays are shared copy-on-write (not pickled); only the date index is
    passed and the residual returned. Identical output to the serial path.
    spill (streaming): directory to write each residual to; the return is then a
    _Lazy mapping date -> loader instead of date -> array."""
    import multiprocessing as mp
    global _RESID_CTX
    _RESID_CTX = dict(epoch=epoch, master=master, flows=flows, hdr=hdr, cd=cd, spill=spill, tag=tag)
    if spill is not None:
        nproc = max(1, min(nproc, len(dates)))
        try:
            if nproc == 1:
                out = dict(_resid_task_spill(d) for d in dates)
            else:
                with mp.get_context('fork').Pool(nproc) as pool:
                    out = dict(pool.map(_resid_task_spill, dates))
        finally:
            _RESID_CTX = {}
        for pth in out.values():
            if pth is not None:
                _SPILLED.append(pth)
        lz = _Lazy()
        for d, pth in out.items():
            if pth is not None:
                dict.__setitem__(lz, d, (lambda _p=pth: np.load(_p)))
        return lz
    nproc = max(1, min(nproc, len(dates)))
    try:
        if nproc == 1:
            out = dict(_resid_task(d) for d in dates)
        else:
            with mp.get_context('fork').Pool(nproc) as pool:
                out = dict(pool.map(_resid_task, dates))
    finally:
        _RESID_CTX = {}
    return {d: r for d, r in out.items() if r is not None}


def _sigma_stack(resids, cell_work, hdr0, tag):
    tmp = []
    for d, r in resids.items():
        tp = f'{cell_work}/ss_{tag}_{d}_{os.getpid()}.fits'   # pid-unique: two searches may share a cell work dir
        fits.writeto(tp, np.nan_to_num(r).astype(np.float32), hdr0, overwrite=True)
        tmp.append(tp)
    ss = build_sigma_stack(tmp) if len(tmp) >= MIN_EPOCHS else None
    for tp in tmp:
        os.remove(tp)
    return ss


def _read_gaul_parents(gaulp):
    """PyBDSF gaul CSV -> [(ra, dec, peak_jy)]. Header is a '# ...' comment
    line (2026-08-30 parsing fix carried from the battery tooling)."""
    ghdr, out = None, []
    for l in (l for l in open(gaulp) if l.strip()):
        if l.startswith('#'):
            f_ = [x.strip() for x in l.lstrip('#').split(',')]
            if ghdr is None and 'RA' in f_:
                ghdr = f_
            continue
        if ghdr:
            rec = dict(zip(ghdr, [x.strip() for x in l.split(',')]))
            try:
                out.append((float(rec['RA']), float(rec['DEC']),
                            float(rec['Peak_flux'])))
            except Exception:
                pass
    return out


_V110_PY = '/opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3'
_V110_WRAP = ('/lustre/gh/main/exopipe_ops/design/FP_PREDICTION/'
              'fp_gate_redo_wrapper.py')


def _v110_cell_verdicts(lst, scope, base, cell_work, master_I, hdr0, em, cd,
                        v_pending):
    """One frozen-v110 invocation over a cell's post-mstruct V survivors.
    Returns {f'{date}_{ra5}_{dec5}': fp_prediction}. REFUSES (raises) on
    gate failure or unverdicted survivors — no candidate reaches the
    catalog unverdicted (R-ROUND2-RATIFIED)."""
    import csv as _csv
    cat = f'{cell_work}/v110_catalog.csv'
    samp = f'{cell_work}/v110_sample.csv'
    keys = {}
    with open(cat, 'w', newline='') as fc, open(samp, 'w', newline='') as fs:
        wc = _csv.writer(fc); ws = _csv.writer(fs)
        wc.writerow(['cand_id', 'lst', 'band', 'ra', 'dec', 'date'])
        ws.writerow(['cand_id', 'resid_path'])
        i = 0
        for d, keptv, vr_path, hdr, ot, _pm in v_pending:
            for c in keptv:
                cid = f'VD_P3_{lst}_{scope}_{d}_{i:04d}'; i += 1
                ra5 = round(c['coord'].ra.deg, 5)
                de5 = round(c['coord'].dec.deg, 5)
                keys[cid] = f'{d}_{ra5}_{de5}'
                wc.writerow([cid, lst, scope, ra5, de5, d])
                ws.writerow([cid, vr_path])
    outdir = f'{cell_work}/v110_out'
    os.makedirs(outdir, exist_ok=True)
    env = dict(os.environ, FPG_CATDIR=f'{cell_work}/catdir',
               FPG_COADD=os.path.dirname(base),
               FPG_RECON=f'{cell_work}/v110_recon')
    r = subprocess.run([_V110_PY, _V110_WRAP, cat, samp, outdir, 'p3']
                       + list(keys), cwd=os.path.dirname(_V110_WRAP),
                       env=env, capture_output=True, text=True,
                       # 2026-09-11: 4 h was exceeded by the V wideband cells with the most candidates (10h Blue/Full,
                       # 21h Blue: 300-700 peaks x 20-30 epochs), each attempt failing after hours of search; 12 h default
                       timeout=int(os.environ.get('P3_V110_TIMEOUT', '43200')))
    outp = f'{outdir}/V_top100_fp_gate_p3.csv'
    if r.returncode != 0 or not os.path.exists(outp):
        raise RuntimeError(
            f'{lst} {scope}: v110 in-chain stage FAILED rc={r.returncode} '
            f'— REFUSING to catalog unverdicted V survivors. '
            f'stderr tail: {r.stderr[-300:]}')
    verdicts = {}
    for row in _csv.DictReader(open(outp)):
        k = keys.get(row['cand_id'])
        if k:
            verdicts[k] = row.get('fp_prediction', '')
    missing = len(keys) - len(verdicts)
    if missing:
        raise RuntimeError(f'{lst} {scope}: {missing} V survivors left '
                           f'unverdicted by v110 — REFUSING')
    return verdicts


def _archive_frame_header(hdr0, base, d, scope, stokes, em, cd, thermnoi, imgrms,
                          master_file, loo_file, align_file, label, cell_tag, cadence='deep'):
    """Conformant-from-birth header for an ARCHIVED search frame
    (R-CONFORMANT-FROM-BIRTH-INCL-HEADERS, Gregg 2026-09-04; products audit
    rows B-01..B-05). Time block and beam come from the epoch's OWN Stage-1
    subtracted product (never the master's); the 7-item standard is stamped
    by phase3_products.write_standard_header, which declares the degenerate
    FREQ/STOKES axes with WCSAXES=4 (the convention DS9 accepts - the banned
    class is WCSAXES=2 alongside CTYPE3); parent-master provenance is
    RELABELLED (MSTR*), never inherited as this frame's own."""
    from exopipe.phase3.core import products as prod
    from astropy.time import TimeDelta as _TD
    sp = f'{base}/subtracted/{d}_{stokes}_sub_{scope}.fits'
    hs = fits.getheader(sp)
    start, end = hs.get('DATE-OBS'), hs.get('DATE-END')
    if end is None and start is not None and hs.get('DURATION'):
        end = (Time(start) + _TD(float(hs['DURATION']), format='sec')).isot
    if start is None or end is None:
        raise RuntimeError(f'{cell_tag} {d}: no epoch time block in {sp} - '
                           f'REFUSING to write an archive frame without its own time')
    if hs.get('BMAJ') is None:
        raise RuntimeError(f'{cell_tag} {d}: no beam in {sp} - REFUSING')
    h = WCS(hdr0).celestial.to_header()
    h['BUNIT'] = ('Jy/beam', 'Brightness unit')
    h['TELESCOP'] = ('OVRO-LWA', 'Telescope')
    mh = fits.getheader(master_file)
    loo_used = bool(loo_file and os.path.exists(loo_file))
    extra = {
        'LOOMSTR': (loo_used, 'Differenced against leave-one-out master'),
        'MSTRFILE': (os.path.basename(loo_file if loo_used else master_file)[:68],
                     'Master actually subtracted'),
        'MSTRNSTK': (int(mh.get('NSTACK', 0)), 'Parent master: epochs in stack'),
        'MSTRCOOP': (str(mh.get('COADDOP', ''))[:20], 'Parent master: co-add operator'),
        'MSTRMJD': (float(mh.get('MJD-MID', 0.0)), 'Parent master: midpoint MJD'),
        'MSTRDIG': (str(mh.get('CONSTDIG', 'legacy-no-digest'))[:32], 'Parent master constituent digest (CONSTDIG)'),
        'STRIWIN': (0.0 if _STRIATION_OFF else float(8.0 * em * cd), '[deg] Striation window (8 x beam; 0 = not applied)'),
        'CADENCE': (cadence, 'Frame cadence'),
    }
    if cadence == '10min':
        # contract step 66 (2026-09-12): the archived 10-minute search frame carries its bin cards from the
        # subtracted frame (bin master name and digest, leave-one-out flag and N, night flux factor)
        for _k, _c in (('BINMSTR', 'central bin master (Coadd/.../10min)'), ('BINLOO', 'subtracted its own leave-one-out bin master'),
                       ('BINNCO', 'constituents of the LOO bin master'), ('BINDIG', 'constituent digest of the bin master'),
                       ('FLUXDC', 'night DC flux factor (no gradient)'), ('PBRATIO', 'PB ratio step (none for per-bin masters)')):
            if hs.get(_k) is not None:
                extra[_k] = (hs.get(_k), _c)
    extra.update({
        'RUNLABEL': (str(label)[:20], 'Search run label'),
        'ORIGIN': ('exopipe phase3_deep search frame', 'Producer'),
    })
    if loo_used:
        extra['LOOEXCL'] = (d, 'Epoch excluded from the LOO master')
    basis = str(hs.get('TIMEFIX', 'v2-phase2')).replace('v2-', '', 1)
    prod.write_standard_header(
        h, scope=scope, stokes=stokes, start=start, end=end,
        inttime_s=hs.get('INTTIME'), time_basis=basis,
        beam=(float(hs['BMAJ']), float(hs['BMIN']), float(hs.get('BPA', 0.0))),
        ebeam=(float(em * cd), float(em * cd), 0.0),
        thermnoi=thermnoi, imgrms=imgrms, pbcorr=False, dewarped=True,
        aligned=True, alignref=os.path.basename(align_file),
        msub=True, msubfile=os.path.basename(loo_file if loo_used else master_file),
        ssub=not _STRIATION_OFF, p2run=hs.get('P2RUN'), calbp=hs.get('CALBP'),
        lstbin=(int(hs['LSTBIN']) if cadence == '10min' and hs.get('LSTBIN') is not None else None),
        p1run=hs.get('P1RUN'), extra=extra, validate=False)
    try:
        h['PRODNAME'] = (prod.build_name(
            scope, stokes, cadence, 'Taper',
            -0.75 if (stokes == 'I' and cadence == 'deep') else 0.0,
            proc=['dewarped', 'aligned', 'msub'] + ([] if _STRIATION_OFF else ['ssub']),
            lst=cell_tag.split()[0], epoch=start,
            lst_bin=(int(hs.get('LSTBIN')) if cadence == '10min' and hs.get('LSTBIN') is not None else None)),
            'Conformant product name')
    except Exception as _e:   # naming grammar gap -> held for review, never silent
        h['PRODNAME'] = ('unknown', f'build_name refused: {str(_e)[:40]}')
    h.add_history(f'{label} {stokes} search frame: {os.path.basename(sp)} '
                  f'(epoch) minus {extra["MSTRFILE"][0]} minus striation model')
    return h


def _size_test(c, snr, epoch_bmaj_deg, cd, stokes, resid, bpix, epoch_bmin_deg=None):
    """Gaussian-fit SIZE test (Gregg 2026-09-04): for |local SNR| > 8, remove
    when (fitted FWHM major - its 1-sigma uncertainty) exceeds SIZE_FLOOR_BEAMS x
    the epoch's own beam major axis; flag between SIZE_FLAG_BEAMS and the floor.
    V uses the BDSF-fitted shape recorded at extraction (bmaj_fit_deg, err);
    I uses the catalog photometry fit (Gaussian on the residual stamp) because
    the I extractor records no shape. Failed/missing fit => keep (never veto on
    the fitter's own failure). Returns (size_beams or '', 'remove'|'flag'|'')."""
    if not epoch_bmaj_deg or not np.isfinite(epoch_bmaj_deg) or epoch_bmaj_deg <= 0:
        return '', ''
    mj = err = None
    if stokes == 'V':
        mj, err = c.get('bmaj_fit_deg'), c.get('bmaj_fit_err_deg') or 0.0
    else:
        _ph = pc.gaussian_photometry(resid, int(c['pix_x']), int(c['pix_y']), bpix, cd)
        if _ph:
            mj = _ph['fwhm_maj_arcmin'] / 60.0
            err = (_ph.get('fwhm_maj_err_arcmin') or 0.0) / 60.0
    if mj is None or not np.isfinite(mj) or mj <= 0:
        return '', ''
    ratio = float(mj) / float(epoch_bmaj_deg)
    c['size_beams'] = round(ratio, 2)
    # 2026-09-08: keep the Card 9 allowance itself on the row (in beams), so the exact
    # quantity the gate tested - (Maj - 1 sigma)/BMAJ - is reconstructable from the ledger
    if err is not None and np.isfinite(err):
        c['size_err_beams'] = round(float(err) / float(epoch_bmaj_deg), 3)
    # AXIS-RATIO test (Gregg approved 2026-09-05): fitted axis ratio over the epoch
    # beam's own axis ratio; remove > AXIS_REMOVE above 8 sigma, flag AXIS_FLAG-AXIS_REMOVE,
    # nothing below 8 sigma. Measured: every known real (EQ Peg, FI Cnc, J0410 pulses)
    # within 0.70-1.16; gold real p95 1.25; the elongated I-Blue class sits at 2-15.
    _an = None
    try:
        if stokes == 'V':
            _mn = c.get('bmin_fit_deg')
            _ar = float(mj) / float(_mn) if _mn and float(_mn) > 0 else None
        else:
            _ar = float(_ph['axis_ratio']) if _ph and _ph.get('axis_ratio') else None
        if _ar and epoch_bmin_deg and float(epoch_bmin_deg) > 0:
            _an = _ar / (float(epoch_bmaj_deg) / float(epoch_bmin_deg))
            c['axis_norm'] = round(_an, 2)
    except Exception:
        _an = None
    if abs(snr) <= SIZE_SNR_GATE:
        return c['size_beams'], ''
    eff = (float(mj) - float(err if np.isfinite(err) else 0.0)) / float(epoch_bmaj_deg)
    if eff > SIZE_FLOOR_BEAMS:
        return c['size_beams'], 'remove'
    if _an is not None and _an > AXIS_REMOVE:
        return c['size_beams'], 'remove-axis'
    verdict = ''
    if ratio > SIZE_FLAG_BEAMS:
        c['size_flag'] = f'size {SIZE_FLAG_BEAMS}-{SIZE_FLOOR_BEAMS} beams'; verdict = 'flag'
    if _an is not None and _an > AXIS_FLAG:
        c['axis_flag'] = f'axis {AXIS_FLAG}-{AXIS_REMOVE} x beam'; verdict = verdict or 'flag'
    return c['size_beams'], verdict


def _epoch_bmaj(base, d, scope, stokes):
    try:
        return float(fits.getheader(f'{base}/subtracted/{d}_{stokes}_sub_{scope}.fits')['BMAJ'])
    except Exception:
        return None


def _epoch_bmin(base, d, scope, stokes):
    try:
        return float(fits.getheader(f'{base}/subtracted/{d}_{stokes}_sub_{scope}.fits')['BMIN'])
    except Exception:
        return None


_PB_CACHE = {}


def _write_pbcorr_sibling(frame_path, data, hdr, base, scope, cell_tag, d, pbp=None, robust=None):
    """S15.2 lowest-level file (4): the primary-beam-corrected sibling of the
    search frame, from the cell's time-integrated PB map (coadd_final/PB_deep).
    Conformant from birth: the frame's header + PBCORR=True + its own PRODNAME.
    Missing PB map = REFUSAL (a ruled product cannot be silently absent)."""
    # deep: the map of the frame's robust weighting (PB step 2, 2026-10-01: I Robust -0.75 -> PB_deep_R075, V Robust 0 ->
    # PB_deep); 10 minutes: the frame's bin PB map (contract step 61/66)
    if pbp is None:
        from exopipe.phase3.lib.masters.pb_apply import deep_pb_name as _dpn
        pbp = f'{base}/coadd_final/{_dpn(scope, robust)}'
    if pbp not in _PB_CACHE:
        if not os.path.exists(pbp):
            # 2026-09-04 23:10 UT: 42 of 96 wideband cells have NO PB map at their
            # canonical root (a MASTER-class gap, not a search defect). The search
            # frame is still written; the sibling is RECORDED as owed and generated
            # once the PB maps exist (phase3_pbcorr_masters).
            _rf = f"{os.environ.get('P3_STAGING_ROOT', '.')}/pbcorr_missing_{cell_tag.replace(' ', '_')}.csv"
            with open(_rf, 'a') as _fh:
                _fh.write(f'{cell_tag},{d},{os.path.basename(frame_path)},{pbp}\n')
            print(f'  {cell_tag} {d}: PB map {pbp} MISSING - pbcorr sibling NOT written '
                  f'(recorded in {os.path.basename(_rf)}; master-class gap)', flush=True)
            return None
        _pb = np.squeeze(fits.getdata(pbp)).astype(np.float32)
        _PB_CACHE[pbp] = np.where(_pb > 0.05, _pb, np.nan)
    pb = _PB_CACHE[pbp]
    if pb.shape != data.shape:
        raise RuntimeError(f'{cell_tag} {d}: PB map shape {pb.shape} != frame {data.shape}; REFUSING')
    h = hdr.copy()
    h['PBCORR'] = (True, 'Primary-beam corrected')
    h['PBFILE'] = (os.path.basename(pbp)[:68], 'PB map applied')
    pn = str(hdr.get('PRODNAME', ''))
    h['PRODNAME'] = (pn.replace('_dewarped_', '_pbcorr_dewarped_', 1) if '_dewarped_' in pn else 'unknown',
                     'Conformant product name')
    out = frame_path.replace('_redo_resid_', '_redo_resid_pbcorr_')
    _write_archive_frame(out, data / pb, h, cell_tag, d)
    return out


def _write_sigstack(sstack, hdr0, run_root, lst, scope, stokes):
    """Persist the per-pixel across-epoch noise map (grammar token 'sigstack';
    products audit B-11: built in memory only before 2026-09-04). Archive class
    to be ruled; written under the run root so the gate's input is on disk."""
    if sstack is None:
        return None
    d = f'{run_root}/sigstack'; os.makedirs(d, exist_ok=True)
    p = f'{d}/{lst}_{scope}_{stokes}_sigstack.fits'
    h = WCS(hdr0).celestial.to_header()
    for k in ('DATE-OBS', 'DATE-END', 'MJD-OBS', 'MJD-MID', 'DURATION', 'INTTIME', 'TIMEFIX',
              'BMAJ', 'BMIN', 'BPA', 'RESTFRQ', 'NSTACK', 'NCOADD', 'COADDOP'):
        if k in hdr0:
            h[k] = hdr0[k]
    h['BUNIT'] = ('ratio', 'across-epoch MAD noise / thermal, floored at 1')
    h['PRODUCT'] = ('sigstack', 'Ensemble per-pixel noise-structure map')
    h['STOKES'] = (stokes, 'Stokes parameter')
    fits.writeto(p + '.part', np.asarray(sstack, dtype=np.float32), h, overwrite=True)
    os.replace(p + '.part', p)
    return p


def _archive_root():
    return os.environ.get('P3_ARCHIVE_ROOT', '/lustre/pipeline/exopipe/phase3')   # read at CALL time


def _archive_frame_publish(path, lst, scope, d, stokes, cadence, cell_tag):
    """Write a deep search frame (or its pbcorr sibling) into the ratified archive tree
    {LST}/{date}/{RUN}/{band}/{Stokes}/deep/<PRODNAME>.fits (Gregg 2026-09-09, contract
    steps 0.4/51). Deep cadence only; P3_ARCHIVE_RUN names the generation; P3_ARCHIVE_WRITE=0
    is the only way to search without archiving (scratch, never production)."""
    if os.environ.get('P3_ARCHIVE_WRITE') == '0':
        return None
    # 2026-09-12 (contract step 66, Gregg "other recommendations approved"): the 10-minute cadence archives its
    # search frames too, under {LST}/{night}/{RUN}/{band}/{Stokes}/10min/ (the night = the frame key's date part).
    if cadence not in ('deep', '10min'):
        return None
    run = os.environ.get('P3_ARCHIVE_RUN')
    if not run:
        raise RuntimeError(f'{cell_tag} {d}: P3_ARCHIVE_RUN unset - the generation Run directory must be '
                           f'named; REFUSING to write a {cadence} search frame outside the archive (contract 51/66)')
    name = str(fits.getheader(path).get('PRODNAME', '') or '')
    if not name or name == 'unknown':
        raise RuntimeError(f'{cell_tag} {d}: {os.path.basename(path)} has no PRODNAME; REFUSING to archive')
    _night = d[:10]
    dst = f'{_archive_root()}/{lst}/{_night}/{run}/{scope}/{stokes}/{cadence}/{name if name.endswith(".fits") else name + ".fits"}'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    import shutil as _sh
    _sh.copy2(path, dst + '.part'); os.replace(dst + '.part', dst)
    return dst


def _write_archive_frame(path, data, hdr, cell_tag, d):
    """Atomic write + write-time validation gate (refuses, never warns)."""
    from exopipe.common.validate_fits_headers import validate_header
    _tmp = path + '.part'
    fits.writeto(_tmp, data.astype(np.float32), hdr, overwrite=True)
    rep = validate_header(_tmp, level='standard')
    if not rep.ok:
        os.remove(_tmp)
        raise RuntimeError(f'{cell_tag} {d}: archive frame header validation FAILED '
                           f'({"; ".join(rep.errors)[:300]}) - REFUSING to archive')
    os.replace(_tmp, path)


def run_deep_cell(lst, scope, label='PROD', included_dates=None, stokes='IV',
                  cadence='deep'):
    """cadence='deep' (hour epochs from the co-add's Stage-1 products) or '10min'
    (frames from tenmin_frames: PB-ratioed, DC-scaled, night-flow-aligned 10-minute
    images minus the Taper Robust 0 master; keys 'YYYY-MM-DDTkk'). Everything after
    the frame source is the same chain (Gregg 2026-09-08: 'identical to deep')."""
    t0 = time.time()
    if cadence not in ('deep', '10min'):
        raise ValueError(f'cadence {cadence!r}: deep or 10min')
    from exopipe.phase3.tenmin_frames import key_date as _kd, tenmin_root as _tr
    if cadence == 'deep':
        _kd = lambda d: d
    os.environ['P3_CADENCE'] = cadence
    global _STRIATION_OFF, _STREAM
    if os.environ.get('P3_STREAM_RESID') is None:
        # 2026-09-12: streaming is the DEFAULT for both cadences. Validated bit-identical at 10 minutes (10h Red,
        # 2 nights: catalog, 1,086 gate-ledger rows, 12 archived frames) and at deep (00h 50MHz, 6 epochs, both
        # Stokes: catalog 81 rows, 2,054 ledger rows, 16 archived frames). P3_STREAM_RESID=0 restores the
        # in-memory path for either cadence.
        _STREAM = True
    if _STREAM:
        print(f'  {lst} {scope}: STREAMING chain (flows/epochs on demand, residuals spilled to {WORKROOT})', flush=True)
    if cadence == '10min':
        # GREGG 2026-09-11 (battery card section 9 approved): 10-minute tier elevation floor 25 deg, both
        # Stokes (deep keeps 20.0 / 19.5). Measured: 25 deg removes 17 / 13 per cent of I / V finals at
        # 0.1 / 0.0 per cent injection cost; 30 deg would cost 9 per cent of the searched area.
        sv2.CONFIG['MIN_ELEVATION'] = float(os.environ.get('P3_TENMIN_MIN_ELEVATION', '25.0'))
        sv2.CONFIG['I_MIN_ELEVATION'] = sv2.CONFIG['MIN_ELEVATION']
        print(f'  {lst} {scope}: 10-minute tier elevation floor {sv2.CONFIG["MIN_ELEVATION"]:.1f} deg (Gregg 2026-09-11)')
    if cadence == 'deep':
        # GREGG 2026-10-06 ("3. (c)", after the 20-25 deg analysis: 32 per cent of Stokes I detections in 12 per cent of the sky
        # above 20 deg): the deep elevation floor moves to 25 deg for both Stokes, as the 10-minute tier (was 20.0 / 19.5)
        sv2.CONFIG['MIN_ELEVATION'] = float(os.environ.get('P3_DEEP_MIN_ELEVATION', '25.0'))
        sv2.CONFIG['I_MIN_ELEVATION'] = sv2.CONFIG['MIN_ELEVATION']
        # the Stokes I DETECTION mask is search_i_transients' own (phase3_pipeline.run_identification_i -> sit.make_masks); the
        # 2026-10-06 change set only the V module, so Stokes I still detected down to 20 deg and lost 20-25 deg only at publish
        # (found by the 2026-10-07 rulings audit) - the floor is now the same in both detection masks
        from exopipe.phase3.lib.search import search_i as _sit
        _sit.CONFIG['MIN_ELEVATION'] = sv2.CONFIG['MIN_ELEVATION']
        print(f'  {lst} {scope}: deep elevation floor {sv2.CONFIG["MIN_ELEVATION"]:.1f} deg, Stokes I and V detection masks (Gregg 2026-10-06)')
    if cadence == '10min' and os.environ.get('P3_STRIATION_10MIN') != '1':
        _STRIATION_OFF = True       # Gregg 2026-09-09: striation model OFF at 10 minutes (battery: FPs x3.6-x10 with it on)
        print(f'  {lst} {scope}: 10-minute cadence - striation model OFF (ruling 2026-09-09)', flush=True)
    # G-03 (ratified 2026-07-14/15; Gregg 2026-09-04: 'There is no pre-blank
    # authorized' - removed for the 5th time, now for BOTH Stokes): A-team is a
    # DOWNSTREAM veto stage (Card 6: six sources, V 3.0 deg / I 5.0 deg), never
    # a pre-detection blank. Both chains search the full sky above 20 deg.
    os.environ['P3_ATEAM_MASK_OFF'] = '1'
    base = f'{coadd_root()}/{lst}_{scope}'
    # BUG FIX (Gregg order 2026-08-29): wideband used the Taper R0 I master
    # here while subtracted/*_I_sub_* is built on the Taper R-0.75 chain
    # (coadd_full_pipeline.py:2681) — the epoch reconstruction I_sub+master
    # was therefore a variant hybrid, planting master-gradient artifacts at
    # source positions and mis-calibrating beam-based tests (it also fed the
    # V search's co-located-I context). R-VARIANT-DOC: I = Taper_Robust-0.75.
    _irob = 'R0' if cadence == '10min' else 'R075'   # 10-minute frames exist only as Robust 0
    mI = f'{base}/coadd_final/I_deep_Taper_{_irob}_{scope}.fits'
    if not os.path.exists(mI):
        # 2026-09-04 (silent-skip variant 4: offloaded flows without inputs):
        # a missing master is an input failure, never a skip - REFUSE.
        raise RuntimeError(f'{lst} {scope}: no deep I master at {mI} — '
                           f'inputs missing or unshipped; REFUSING the silent no-op')
    master_I = _load(mI); hdr0 = fits.getheader(mI); cd = abs(hdr0['CDELT1'])
    # ── MASTER-GENERATION GATE (2026-09-01) ────────────────────────────
    # REFUSES, does not warn. The master-creation policy of 2026-08-25
    # (plain inverse-variance mean, no per-pixel clipping along the epoch
    # axis; LOO masters for N_epochs<=20) was ratified and then NOT applied
    # to the subband masters. Because consumers hardcoded a superseded
    # co-add root, a 255-cell subband V fleet ran for hours against
    # sigma-clipped masters, and 70% of the delivered wideband Stokes I
    # list was built on a superseded or pre-policy basis. The clipped
    # masters carry high spatial structure (Gregg, at the 24 Boo position),
    # so searching them is not a degraded result, it is a wrong one.
    # COADDOP exists precisely so the generations can never be confused.
    # Set P3_ALLOW_OLD_MASTERS=1 only for a deliberate archaeological
    # comparison, never for production. MASTER_GEN: the current generation is
    # ivw-mean-v3 (flat-DC flux scale, R-TILT-REMOVED-NEXTGEN landed 2026-09-08).
    # A run launched on the superseded v2 generation must DECLARE it with
    # P3_MASTER_GEN=ivw-mean-v2 (the fleet launcher sets it; logged per cell).
    MASTER_GEN = os.environ.get('P3_MASTER_GEN', 'ivw-mean-v3')
    _op = hdr0.get('COADDOP')
    print(f'  {lst} {scope}: master generation COADDOP={_op!r} (required {MASTER_GEN!r})', flush=True)
    if _op != MASTER_GEN and os.environ.get('P3_ALLOW_OLD_MASTERS') != '1':
        raise RuntimeError(
            f'{lst} {scope}: REFUSING to search a superseded master. '
            f'{mI} has COADDOP={_op!r}, expected {MASTER_GEN!r} '
            f'(plain-mean + LOO policy, ratified 2026-08-25; flat-DC 2026-08-28). Rebuild the '
            f'cell, or point P3_COADD_ROOT at the current-generation root '
            f'(exopipe_ops/resolve_master_root.sh resolves it).')
    # ── MASTER-MEMBERSHIP GATE (2026-10-07, Gregg "Yes, go") ───────────
    # REFUSES, does not warn. The stage-3 (flux-alignment window) membership ruling of 2026-08-26 was applied in August by side
    # scripts and lost at the 2026-09-10 rebuild; stage 2 is the non-matched metric (canon section 7, 2026-09-05). A deep cell is
    # searched only when its master's member nights satisfy every membership ruling (exopipe.phase3.membership.verify: exclusions,
    # stage 2 with the ruled metric, stage-3 window, night-image and epoch-image checks). Deep cadence only; the 10-minute tier is the
    # snapshot / sub-LST chat's (cross-tier note in the ledger). P3_ALLOW_OLD_MASTERS=1 bypasses it for archaeology, never production.
    if cadence == 'deep' and os.environ.get('P3_ALLOW_OLD_MASTERS') != '1':
        from exopipe.phase3.lib.masters import membership as _MB
        _mv = _MB.verify(base) + _MB.verify_products(base)   # membership + what the master must carry (flux scale, PB, dewarp)
        if _mv:
            raise RuntimeError(f'{lst} {scope}: REFUSING to search - the master violates the membership rulings: '
                               + ' | '.join(_mv[:5]) + (f' (+{len(_mv) - 5} more)' if len(_mv) > 5 else ''))
        print(f'  {lst} {scope}: master membership verified against the rulings (stage 2 {_MB.STAGE2_METRIC}, stage-3 window)', flush=True)
    em, en, _, _ = fit_empirical_beam(master_I)
    if not np.isfinite(em) or em <= 0:   # beam fit can fail (e.g. 14h Blue) -> NaN propagates into headers
        print(f'  {lst} {scope}: empirical-beam fit returned NaN; using fallback beam=7px')
        em = 7.0; en = 7.0
    beam = em
    ar_max = max(2.0, 1.5 * em / max(en, 1e-3))
    mVp = f'{base}/coadd_final/V_deep_Taper_R0_{scope}.fits'
    if os.path.exists(mVp):
        _opv = fits.getheader(mVp).get('COADDOP')
        if _opv != MASTER_GEN and \
                os.environ.get('P3_ALLOW_OLD_MASTERS') != '1':
            raise RuntimeError(
                f'{lst} {scope}: REFUSING to search a superseded V master. '
                f'{mVp} has COADDOP={_opv!r}, expected {MASTER_GEN!r}.')
    master_V = _load(mVp) if os.path.exists(mVp) else None
    ms_destr, ms_raw = (_mstruct_maps(master_V, mVp, beam)
                        if master_V is not None else (None, None))
    # epoch-product root: the co-add tree (deep) or the 10-minute Stage-1 tree; masters,
    # weights and flows always come from the co-add tree (base)
    ebase = base if cadence == 'deep' else f'{_tr()}/{lst}_{scope}'
    cell_work = f'{WORKROOT}/{lst}_{scope}' + ('' if cadence == 'deep' else '_10min'); os.makedirs(cell_work, exist_ok=True)
    root = os.environ.get('P3_STAGING_ROOT', pc.P.STAGING_ROOT); nwr = 0
    run_root = pc.P.transients_run(root, label, make=True)
    # refusal ledgers (beam gate, sanity gate) live under the RUN root so the
    # fleet ships them (audit R-15: under the input root they were lost on
    # remote nodes and multiplied on lwacalim07); one tag per attempt
    _refdir = f'{run_root}/refusals'; os.makedirs(_refdir, exist_ok=True)
    _attempt = time.strftime('%Y%m%dT%H%M%S', time.gmtime())

    dates = []
    for c in sorted(glob.glob(f'{ebase}/subtracted/*_I_sub_{scope}.fits')):
        m = re.search(r'(\d{4}-\d\d-\d\d(?:T\d\d)?)_I_sub_', os.path.basename(c))
        if m:
            dates.append(m.group(1))
    if len(dates) == 0:
        raise RuntimeError(f'{lst} {scope}: ZERO epochs found — inputs '
                           f'missing/unshipped; REFUSING (a 0-epoch cell '
                           f'is never legitimate)')
    _minE = DEEP_MIN_EPOCHS if cadence == 'deep' else MIN_EPOCHS
    if len(dates) < _minE:
        print(f'  {lst} {scope}: P3-POLICY-SKIP only {len(dates)} epochs (minimum {_minE}, cadence {cadence})'); return 0

    mup = f'{cell_work}/master_up2.fits'
    # FLOW-CACHE GENERATION GATE (constituent-generation ruling 2026-08-12, approved
    # 2026-09-05): a flow cache computed against a different master generation is
    # REFUSED (never silently consumed); a legacy cache without a sidecar is recorded
    # as generation-unverified and the sidecar is written after this search.
    import hashlib as _hl, json as _js
    _gen = _hl.md5(np.ascontiguousarray(master_I.astype(np.float32)).tobytes()).hexdigest()
    _gsc = f'{cell_work}/flow_generation.json'
    if os.path.exists(_gsc):
        _g = _js.load(open(_gsc))
        if _g.get('master_md5_data') != _gen:
            raise RuntimeError(f'{lst} {scope}: REFUSING flow cache of another master generation '
                               f'({_g.get("master_md5_data", "?")[:8]} != current {_gen[:8]}; '
                               f'recompute flows against the current master)')
        print(f'  {lst} {scope}: flow cache generation verified ({_gen[:8]})')
    else:
        print(f'  {lst} {scope}: flow cache generation UNVERIFIED (legacy cache, no sidecar) - sidecar written after this search')
    if not os.path.exists(mup):
        write_up_fits(fft_upsample2(master_I), hdr0, mup)
    items, epoch_I = [], {}
    _binm = {}
    def _recon_master(d):
        # per-bin masters (Gregg 2026-09-09): a 10-minute frame that was subtracted against its bin
        # master is reconstructed with that bin's central master (BINMSTR), not the deep master
        if cadence != '10min':
            return master_I
        _bm = fits.getheader(f'{ebase}/subtracted/{d}_I_sub_{scope}.fits').get('BINMSTR')
        if not _bm:
            return master_I
        if _bm not in _binm:
            _binm[_bm] = _load(f'{ebase}/bin_masters/{_bm}')
        return _binm[_bm]
    if _STREAM:
        epoch_I = _Lazy()
    for d in dates:
        ei = _load(f'{ebase}/subtracted/{d}_I_sub_{scope}.fits') + _recon_master(d)
        if _STREAM:
            dict.__setitem__(epoch_I, d, (lambda _d=d: _load(f'{ebase}/subtracted/{_d}_I_sub_{scope}.fits') + _recon_master(_d)))
        else:
            epoch_I[d] = ei
        ep_up = f'{cell_work}/ep_{d}_up2.fits'
        if not os.path.exists(ep_up):
            write_up_fits(fft_upsample2(ei), hdr0, ep_up)
        items.append((d, ep_up))
    if _REFINE_OFF:
        print(f'  {lst} {scope}: P3_REFINE_OFF=1 - per-epoch x2 flow refinement SKIPPED (battery variant; night flow only)')
        flows = {d_: None for d_, _p in items}
    else:
        flows = _batch_flows(cell_work, mup, items)
    if not os.path.exists(_gsc):
        _js.dump({'master': os.path.basename(mI), 'master_md5_data': _gen,
                  'CONSTDIG': hdr0.get('CONSTDIG', ''), 'written_by': 'phase3_deep', 'n_flows': len(flows)},
                 open(_gsc + f'.part{os.getpid()}', 'w')); os.replace(_gsc + f'.part{os.getpid()}', _gsc)   # pid-unique: V and I searches of one cell may run concurrently (2026-09-12 race)
    dates = [d for d in dates if d in flows]
    # (0) drop data_decisions-excluded epochs — the master already excludes them, but the
    # per-epoch residual still exists on disk, so the search would otherwise pull them back
    # in (e.g. vortex night 2025-05-04; mstruct can't catch it — it's not in the master).
    _excl = load_exclusions()
    _ddrop = [d for d in dates if is_excluded(lst, scope, _kd(d), _excl)]
    if _ddrop:
        dates = [d for d in dates if d not in _ddrop]
        print(f'  {lst} {scope}: dropped {len(_ddrop)} data_decisions-excluded epoch(s): {_ddrop}')
    # (0b) NIGHT-IMAGE GATE (Gregg 2026-10-02 "Approved"): one test per night-image, ahead of the search-admission QA
    # below (which keeps its own logic); a night-image the gate failed is refused from BOTH searches, logged per image;
    # a deep cell without the gate record REFUSES (a gate never skips). Deep cadence: the 10-minute tier is the
    # snapshot / sub-LST chat's.
    if cadence == 'deep':
        from exopipe.phase3.lib.masters import night_image_gate as _NG
        try:
            _ngf = _NG.failed_dates(f'{base}/coadd_final')
        except FileNotFoundError:
            raise RuntimeError(f'{lst} {scope}: no {_NG.GATE_FILE} in {base}/coadd_final - the night-image gate '
                               f'(Gregg 2026-10-02) cannot run; REFUSING')
        _ngd = [d for d in dates if _kd(d) in _ngf]
        if _ngd:
            dates = [d for d in dates if d not in _ngd]
            with open(f'{_refdir}/{lst}_{scope}_nightimage_excluded.csv', 'a') as _nf:
                for d in _ngd:
                    _nf.write(f'{lst},{scope},{d},{_attempt}\n')
            print(f'  {lst} {scope}: NIGHT-IMAGE GATE - {len(_ngd)} epoch(s) REFUSED from both searches: {_ngd}')
        # (0c) EPOCH GATE (Gregg 2026-10-04, option B: "Dont force products for crappy nights"): a night whose epoch image
        # fails its content test (coadd_final/epoch_gate.csv, made against THIS cell's current I master) is refused from BOTH
        # searches; a missing or stale record REFUSES the cell (a gate never skips).
        from exopipe.phase3.lib.masters import epoch_gate as _EG
        try:
            _egf = _EG.failed_dates(f'{base}/coadd_final', scope)
        except FileNotFoundError:
            raise RuntimeError(f'{lst} {scope}: no {_EG.GATE_FILE} in {base}/coadd_final - the epoch gate (Gregg 2026-10-04) '
                               f'cannot run; REFUSING')
        _egd = [d for d in dates if _kd(d) in _egf]
        if _egd:
            dates = [d for d in dates if d not in _egd]
            with open(f'{_refdir}/{lst}_{scope}_epochgate_excluded.csv', 'a') as _ef:
                for d in _egd:
                    _ef.write(f'{lst},{scope},{d},{_attempt}\n')
            print(f'  {lst} {scope}: EPOCH GATE - {len(_egd)} epoch(s) REFUSED from both searches: {_egd}')
    # (1) TWO-TIER QUALITY POLICY (Gregg ruling 2026-08-26, supersedes the
    # earlier QA-outlier search drop / G-02): stage-2 QA rejection omits an
    # epoch from the CO-ADD ONLY; every non-data_decisions epoch is still
    # searched. Egregious nights (vortex, large peel errors) are handled by
    # data_decisions via the QA-3 withhold-for-review flow — the old inline
    # drop (motivated by a 79sigma peeling-blowup flooding the list) is now
    # that flow's job, not the search's.
    qa_out = _qa_outlier_dates(lst, scope)
    _qa_note = [d for d in dates if _kd(d) in qa_out]
    if _qa_note:
        print(f'  {lst} {scope}: {len(_qa_note)} coadd-omitted (QA-outlier) epoch(s) '
              f'REMAIN in search per two-tier policy: {_qa_note}')
    skip_I_dates = set()
    # (1b) SEARCH-FRAME SANITY GATE (Gregg NAMED approval 2026-08-29, the
    # narrow resolution of the open whole-image-discard ruling): an epoch
    # whose flux-alignment QA rms exceeds 10x the cell's median epoch rms
    # is numerically broken (over-cleaning class: 19 products measured at
    # 10^5-10^16 Jy/beam, tranche-2a) and is NOT transient-searched. A
    # refusal, logged per image; the only images the search may drop on
    # its own are ones the QA has MEASURED as broken.
    try:
        import csv as _csv
        _wrows = list(_csv.DictReader(
            open(f'{base}/coadd_final/weights.csv')))
        _wr = {r['date']: float(r['qa3_rms_mjy']) for r in _wrows
               if r.get('qa3_rms_mjy') not in (None, '', 'nan')}
        _fin = sorted(v for v in _wr.values() if np.isfinite(v) and v > 0)
        if len(_fin) >= 5:
            _med = _fin[len(_fin) // 2]
            _bad = [d for d in dates
                    if _wr.get(_kd(d), 0) > 10.0 * _med]
            # Stokes I sanity gate at SANITY_I_X (3x, Gregg 2026-09-05: CLEAN-divergence
            # epochs sit at 4-7x; V has no CLEAN step and keeps the 10x gate)
            _badI = [d for d in dates if d not in _bad and _wr.get(_kd(d), 0) > SANITY_I_X * _med]
            if _badI:
                skip_I_dates.update(_badI)
                with open(f'{_refdir}/{lst}_{scope}_sanity_I_excluded.csv', 'a') as _sf:
                    for d in _badI:
                        _sf.write(f'{lst},{scope},{d},{_wr[_kd(d)]:.1f},{_med:.1f},{_attempt}\n')
                print(f'  {lst} {scope}: SANITY GATE (I only) — {len(_badI)} epoch(s) REFUSED from the I search '
                      f'(alignment rms >{SANITY_I_X:.0f}x cell median {_med:.0f} mJy): {_badI}')
            if _bad:
                dates = [d for d in dates if d not in _bad]
                with open(f'{_refdir}/{lst}_{scope}_sanity_excluded.csv', 'a') as _sf:
                    for d in _bad:
                        _sf.write(f'{lst},{scope},{d},{_wr[_kd(d)]:.1f},'
                                  f'{_med:.1f},{_attempt}\n')
                print(f'  {lst} {scope}: SANITY GATE — {len(_bad)} epoch(s) '
                      f'REFUSED from search (alignment rms >10x cell median '
                      f'{_med:.0f} mJy): {_bad}')
    except FileNotFoundError:
        pass
    # (1c) SEARCH-TIER QUALITY GATE (Gregg 2026-09-09, correcting the reading of the
    # two-tier policy: "we specified a second less stringent threshold for being used
    # for science, but it was still a threshold"; the threshold is the R-GC10 Layer-2
    # search tier: high-rms 4 sigma MAD, no low-side rejection; co-add tier stays 3
    # sigma). Per epoch, either Stokes above median + 4 x 1.4826 x MAD of the cell's
    # stage-1 residual rms (residuals_provisional.csv) REFUSES the epoch from both
    # searches, logged per image. n < 5 nights: gate skips loudly (R-GC10 rider R1).
    # 2026-09-10 (INCIDENT-SEARCH-TIER-GATE-SKIPPED-TABLE-NOT-SHIPPED-AND-STALE): a missing table is a
    # REFUSAL, never a skip; and the table must be THIS generation's - its date set must equal the
    # generation record (weights.csv) - or it is a stale derivative and REFUSES.
    if not os.path.exists(f'{base}/residuals_provisional.csv'):
        raise RuntimeError(f'{lst} {scope}: residuals_provisional.csv missing - the search-tier gate (Gregg '
                           f'2026-09-09, R-GC10 Layer 2) cannot run; REFUSING (never skipped)')
    try:
        import csv as _csv
        _prov = list(_csv.DictReader(open(f'{base}/residuals_provisional.csv')))
        try:
            _wdates = {r['date'] for r in _csv.DictReader(open(f'{base}/coadd_final/weights.csv'))}
        except FileNotFoundError:
            _wdates = None
        _pdates = {r['date'] for r in _prov if r.get('date')}
        if _wdates is not None and _pdates != _wdates:
            raise RuntimeError(f'{lst} {scope}: residuals_provisional.csv is not this generation\'s (table dates '
                               f'{sorted(_pdates ^ _wdates)[:4]} differ from weights.csv) - stale derivative; REFUSING')
        _rmsI = {r['date']: float(r['i_rms_mjy']) for r in _prov if r.get('i_rms_mjy') not in (None, '', 'nan')}
        _rmsV = {r['date']: float(r['v_rms_mjy']) for r in _prov if r.get('v_rms_mjy') not in (None, '', 'nan')}
        _key = _kd
        if cadence == '10min':
            # R-TENMIN-SEARCH-TIER-PER-FRAME (Gregg 2026-09-12 "Per frame is approved"; contract step 56): at 10
            # minutes the admitted unit is the FRAME. Each frame's stage-1 residual (its leave-one-out bin
            # subtraction, subtracted/<key>_<S>_sub) gives an inner rms; the population is every frame of the
            # cell for that Stokes (median + 4 x 1.4826 MAD, no low side; fewer than 5 frames skips loudly);
            # nothing is inherited from the night table (a bad night fails every frame on its own). Either
            # Stokes above threshold refuses the frame from both searches, as at deep.
            from exopipe.phase3.lib.search.identification import inner_rms as _irms
            _rmsI, _rmsV = {}, {}
            for _d in dates:
                for _st, _rm in (('I', _rmsI), ('V', _rmsV)):
                    _fp = f'{ebase}/subtracted/{_d}_{_st}_sub_{scope}.fits'
                    if os.path.exists(_fp):
                        try:
                            _rv = float(_irms(np.squeeze(fits.getdata(_fp)).astype(float)))
                            if np.isfinite(_rv) and _rv > 0:
                                _rm[_d] = _rv
                        except Exception as _e:
                            print(f'  {lst} {scope} {_d}: stage-1 rms unreadable ({_e!r}) - frame keeps no rms')
            _key = lambda d: d
        _tier = []
        for _st, _rm in (('I', _rmsI), ('V', _rmsV)):
            _v = np.array(sorted(x for x in _rm.values() if np.isfinite(x) and x > 0))
            if len(_v) < 5:
                print(f'  {lst} {scope}: SEARCH-TIER GATE ({_st}) SKIPPED - only {len(_v)} '
                      f'{"frames" if cadence == "10min" else "nights"} with a stage-1 rms (needs 5)')
                continue
            _med = float(np.median(_v)); _mad = float(np.median(np.abs(_v - _med))) * 1.4826
            _thr = _med + SEARCH_TIER_SIGMA * _mad
            for d in dates:
                if _rm.get(_key(d), 0) > _thr:
                    _tier.append((d, _st, _rm[_key(d)], _thr))
        # GREGG 2026-09-13 ("this is a mistake. That sounds like the gate for the co-add, not the search!"): the
        # search-tier admission is PER STOKES - an epoch (or 10-minute frame) is refused from the I search on the I
        # stage-1 rms and from the V search on the V rms, never from both on either. Until this fix the deep run
        # refused 472 epochs in 291 cell-Stokes on the other Stokes' rms (EQ Peg 2025-08-01 in six 21h bands).
        _skipI_tier = {d for d, _st, *_ in _tier if _st == 'I'}
        _skipV_tier = {d for d, _st, *_ in _tier if _st == 'V'}
        if _tier:
            with open(f'{_refdir}/{lst}_{scope}_searchtier_excluded.csv', 'a') as _sf:
                for d, _st, _r, _thr in _tier:
                    _sf.write(f'{lst},{scope},{d},{_st},{_r:.1f},{_thr:.1f},{_attempt}\n')
            print(f'  {lst} {scope}: SEARCH-TIER GATE (per Stokes) - I search refuses {len(_skipI_tier)} epoch(s) {sorted(_skipI_tier)}; '
                  f'V search refuses {len(_skipV_tier)} epoch(s) {sorted(_skipV_tier)} '
                  f'(stage-1 rms > median + {SEARCH_TIER_SIGMA:.0f} x MAD of that Stokes)')
            # (an epoch refused by both Stokes stays in dates: its frames are still made; only its search is skipped)
    except FileNotFoundError as _e:
        raise RuntimeError(f'{lst} {scope}: search-tier gate input missing ({_e}); REFUSING')
    skip_I_dates.update({d for d in _skipI_tier if d in dates})      # per-Stokes search-tier refusals, I side (Gregg 2026-09-13)
    # (1d) INPUT-PROVENANCE GATE (2026-09-05, after 02h 36MHz was parked six times on
    # a stale 2025-01-02 product from 2026-08-13 with BMAJ=0 that the 2026-09-02 rebuild
    # never regenerated - the night has no Phase-2 product): an epoch is searched only
    # if its subtracted products are of the current generation, i.e. the epoch is a
    # master constituent OR a current-generation QA-omitted epoch, AND both Stokes
    # headers carry a positive beam. Refused epochs are logged per image, never
    # silently dropped, and never fail the cell.
    try:
        _cons = {str(x).split()[-1] for x in hdr0.get('HISTORY', []) if str(x).startswith('COADD constituent')}
        # current-generation record = the FINAL co-add's weights.csv (lists every epoch the
        # final stage handled, omitted ones included); residuals_provisional.csv can be
        # stale (02h 36MHz: it listed 2025-01-02 and lacked the autumn epochs)
        _wcsv = f'{base}/coadd_final/weights.csv'
        _cur = {r['date'] for r in csv.DictReader(open(_wcsv))} if os.path.exists(_wcsv) else None
        _stale = []
        for d in dates:
            _why = ''
            if _cur is not None and _kd(d) not in _cons and _kd(d) not in _cur:
                _why = 'not a constituent and absent from the final co-add record (weights.csv)'
            for _st in ('I', 'V'):
                try:
                    _h = fits.getheader(f'{ebase}/subtracted/{d}_{_st}_sub_{scope}.fits')
                    if not (float(_h.get('BMAJ') or 0) > 0 and float(_h.get('BMIN') or 0) > 0):
                        _why = (_why + '; ' if _why else '') + f'{_st} product BMAJ/BMIN not positive'
                except Exception as _e:
                    _why = (_why + '; ' if _why else '') + f'{_st} product unreadable'
            if _why:
                _stale.append((d, _why))
        if _stale:
            dates = [d for d in dates if d not in {x[0] for x in _stale}]
            with open(f'{_refdir}/{lst}_{scope}_stale_input_excluded.csv', 'a') as _sf:
                for d, _why in _stale:
                    _sf.write(f'{lst},{scope},{d},"{_why}",{_attempt}\n')
            print(f'  {lst} {scope}: INPUT-PROVENANCE GATE — {len(_stale)} epoch(s) REFUSED from search: {_stale}')
    except Exception as _e:
        raise RuntimeError(f'{lst} {scope}: input-provenance gate could not run ({_e}); REFUSING')
    # (1c) BEAM GATE (Gregg approved 2026-09-02, after the 2024-12-24
    # long-baseline dropout diagnosis): an epoch whose fitted-beam AREA
    # (BMAJ*BMIN of its Phase-2 survey-variant product) exceeds
    # P3_BEAM_GATE_RATIO (default 1.5) x the median of the searched epoch
    # set was imaged from degraded uv coverage (long baselines absent) —
    # searching it against a sharp master floods the list with beam-
    # mismatch FPs (10h Blue 2024-12-24: 147 survivors, all vetoes blind
    # to it). Survey-wide audit 2026-09-02: the healthy population ends
    # ~1.4x and the dropout class starts >= 1.8x, so 1.5 sits in the gap
    # (950/20,376 epochs = 4.7% affected; exopipe_ops/beam_audit.py).
    # A REFUSAL, logged per epoch. P3_BEAM_GATE_OFF=1 for archaeology.
    if os.environ.get('P3_BEAM_GATE_OFF') != '1':
        _p2 = '/lustre/pipeline/exopipe/phase2'
        _irob_f = 'Robust-0.75'
        _areas = {}
        for d in dates:
            # 2026-09-04 23:10 UT: the Stage-1 subtracted product carries the
            # epoch's own fitted beam (same image family, always shipped with
            # the cell) - read it FIRST; the Phase-2 deep image is the fallback.
            try:
                _hs = fits.getheader(f'{ebase}/subtracted/{d}_I_sub_{scope}.fits')
                if _hs.get('BMAJ') and _hs.get('BMIN'):
                    _areas[d] = float(_hs['BMAJ']) * float(_hs['BMIN'])
                    continue
            except Exception:
                pass
            # 2026-09-04: the Phase-2 deep product carries the 'dewarped' token
            # only in some Science runs (e.g. absent on 2024-12 epochs); the
            # old dewarped-only glob silently found NO header for those epochs
            # and the gate never keyed on them. Match the survey variant with
            # or without the token, prefer the dewarped file when both exist.
            _g = sorted([x for x in
                         glob.glob(f'{_p2}/{lst}/{_kd(d)}/*/{scope}/I/deep/'
                                   f'*_I_deep_Taper_{_irob_f}*.fits')
                         + glob.glob(f'{_p2}/{lst}/{_kd(d)}/*/{scope}/I/deep/'
                                     f'*Taper-*-{_irob_f}-*.fits')
                         if 'pbcorr' not in os.path.basename(x)],
                        key=lambda x: (0 if 'dewarped' in os.path.basename(x) else 1, x))
            if _g:
                try:
                    _h = fits.getheader(_g[0])
                    _areas[d] = float(_h['BMAJ']) * float(_h['BMIN'])
                except Exception:
                    pass
        _fin = sorted(_areas.values())
        # 2026-09-04 (audit R-15): the gate REFUSES instead of failing open -
        # an epoch without a readable beam header cannot be admitted, and a
        # set too small to key the median cannot be searched unguarded.
        _nobeam = [d for d in dates if d not in _areas]
        if _nobeam:
            raise RuntimeError(f'{lst} {scope}: no readable Phase-2 beam header for '
                               f'{_nobeam} - beam gate cannot run; REFUSING')
        if len(_fin) < 5:
            # GREGG 2026-09-11 ("Waive the floor. I want a complete Masters set"): a thin cell is
            # SEARCHED with the beam gate skipped LOUDLY (as the search-tier gate does below 5
            # nights, R-GC10 rider), never refused. Every night keeps its readable-beam requirement.
            print(f'  {lst} {scope}: BEAM GATE SKIPPED - only {len(_fin)} nights with a beam area '
                  f'(needs 5 for the cell median); floor waived, Gregg 2026-09-11; all {len(dates)} epochs searched')
        if len(_fin) >= 5:
            _bmed = _fin[len(_fin) // 2]
            _thr = float(os.environ.get('P3_BEAM_GATE_RATIO', '1.5'))
            _bbad = [d for d in dates
                     if _areas.get(d, 0) > _thr * _bmed]
            if _bbad:
                dates = [d for d in dates if d not in _bbad]
                with open(f'{_refdir}/{lst}_{scope}_beam_excluded.csv', 'a') as _bf:
                    for d in _bbad:
                        _bf.write(f'{lst},{scope},{d},'
                                  f'{_areas[d]/_bmed:.2f},{_thr},{_attempt}\n')
                print(f'  {lst} {scope}: BEAM GATE — {len(_bbad)} epoch(s)'
                      f' REFUSED from search (fitted-beam area >{_thr}x '
                      f'cell median): '
                      f'{[(d, round(_areas[d]/_bmed, 2)) for d in _bbad]}')
    if len(dates) == 0:
        raise RuntimeError(f'{lst} {scope}: ZERO flows — REFUSING')
    if len(dates) < _minE:
        print(f'  {lst} {scope}: P3-POLICY-SKIP only {len(dates)} flows (minimum {_minE}, cadence {cadence})'); return 0

    if included_dates:
        # INCREMENTAL MODE (Gregg 2026-09-13 "Proceed"): every admission gate above ran on the whole cell; only the listed
        # epochs are now built and searched, and their rows/frames are APPENDED to the attempt (run under a copy of the
        # published attempt). The sigma-stack annotation comes from the persisted cell map, not from the subset.
        _inc_missing = [d for d in included_dates if d not in dates]
        dates = [d for d in dates if d in included_dates]
        print(f'  {lst} {scope}: INCREMENTAL - {len(dates)} listed epoch(s) to build and search'
              + (f'; {len(_inc_missing)} listed but not admitted by the gates: {_inc_missing}' if _inc_missing else ''), flush=True)
        if not dates:
            print(f'  {lst} {scope}: INCREMENTAL - nothing admitted; done'); return 0
    hdr_ref = _det_header(hdr0, scope, em, cd, None)
    # R-LOO-MASTERS as SEARCH FRAMES (conformance fix 2026-09-03): an epoch
    # is differenced against ITS OWN leave-one-out master where one exists
    # (file existence encodes the ratified per-Stokes [2,20] rule); the
    # central master is the fallback and remains the flow/mstruct/parent
    # reference for all epochs (declared: alignment and structure
    # references stay common across epochs).
    try:
        import csv as _csv2
        _inm = {r['date']: str(r.get('in_master', '')).strip().lower() in ('1', 'true')
                for r in _csv2.DictReader(open(f'{base}/coadd_final/weights.csv'))}
    except FileNotFoundError:
        _inm = {}
    def _loo_masters(central, stokes_tag, variant_tag, master_path):
        out = {}
        _nco = int(fits.getheader(master_path).get('NCOADD', 0) or 0)
        for _d in dates:
            _lp = (f'{base}/coadd_final/loo/'
                   f'{_d}_{stokes_tag}_loo_{variant_tag}_{scope}.fits')
            if cadence == '10min':
                out[_d] = central          # 10-minute frames: central master, no leave-one-out (Gregg 2026-09-08)
                continue
            if not os.path.exists(_lp) and 2 <= _nco <= 20 and _inm.get(_d, False):
                # 2026-09-04 (audit R-19): an in-master epoch of an N<=20 stack
                # without its leave-one-out master is a REFUSAL, never a silent
                # fallback to the central master (the epoch would subtract itself)
                raise RuntimeError(f'{lst} {scope} {_d}: leave-one-out {stokes_tag} master '
                                   f'missing ({_lp}) for an in-master epoch of an '
                                   f'N={_nco} stack; REFUSING')
            if os.path.exists(_lp) and cadence == 'deep':
                # CONSTITUENT-DIGEST REFUSAL (Arch S6 2026-08-12c; R-CONSTITUENT-DIGEST 2026-09-05;
                # landed 2026-09-09 contract step 17): a LOO master is consumed only when its
                # PARENTDG equals the central master's CONSTDIG; a legacy LOO without PARENTDG
                # must at least carry the master's COADDOP. Otherwise REFUSE, never subtract.
                _lh = fits.getheader(_lp)
                _pd = str(_lh.get('PARENTDG', '') or '')
                # 2026-09-11: compare against THIS Stokes' own master (master_path), not hdr0 (the I master):
                # the stacks are per Stokes (R-LOO-PER-STOKES) and a V stack that differs from the I stack has
                # its own CONSTDIG - 02h 55MHz V was parked on a false "stale derivative" refusal.
                _mh_own = fits.getheader(master_path)
                _cd = str(_mh_own.get('CONSTDIG', '') or '')
                if _pd and _cd and _pd != _cd:
                    raise RuntimeError(f'{lst} {scope} {_d}: LOO {stokes_tag} master PARENTDG={_pd} does not '
                                       f'match the {stokes_tag} central master CONSTDIG={_cd} - stale derivative; REFUSING')
                if not _pd and _lh.get('COADDOP') != _mh_own.get('COADDOP'):
                    raise RuntimeError(f'{lst} {scope} {_d}: LOO {stokes_tag} master COADDOP={_lh.get("COADDOP")!r} '
                                       f'!= {stokes_tag} master {_mh_own.get("COADDOP")!r} (no PARENTDG) - stale derivative; REFUSING')
            out[_d] = _load(_lp) if os.path.exists(_lp) else central
        _n = sum(1 for _d in dates if out[_d] is not central)
        if _n:
            print(f'  {lst} {scope}: {_n}/{len(dates)} epochs searched '
                  f'against their leave-one-out {stokes_tag} masters')
        return out
    if 'I' in stokes:
        resid_I = _build_residuals_mp(epoch_I,
                                      _loo_masters(master_I, 'I',
                                                   'Taper_R075', mI),
                                      flows, hdr_ref, cd, dates, NPROC_RESID,
                                      spill=(cell_work if _STREAM else None), tag='I')
        dates = [d for d in dates if d in resid_I]
        _ssI = f'{run_root}/sigstack/{lst}_{scope}_I_sigstack.fits'
        if included_dates and os.path.exists(_ssI):
            sstack_I = np.squeeze(fits.getdata(_ssI)).astype(float); print(f'  {lst} {scope}: INCREMENTAL - sigma-stack I from the persisted cell map')
        else:
            sstack_I = _sigma_stack(resid_I, cell_work, hdr0, 'I')
            _write_sigstack(sstack_I, hdr0, run_root, lst, scope, 'I')
    else:
        resid_I, sstack_I = {}, None   # V-only redo (Gregg 2026-08-21): Stokes I is out
    resid_V = {}
    # 2026-09-01 fix: honor stokes='I' — the V side previously ran whenever
    # a V master existed, so an I-only extraction silently re-searched V
    # (duplicating the survey-complete V ledger into the new run's dir).
    # 'V' and 'IV' behavior unchanged.
    if master_V is not None and 'V' in stokes:
        epoch_V = _Lazy() if _STREAM else {}
        for d in dates:
            vp = f'{ebase}/subtracted/{d}_V_sub_{scope}.fits'
            if os.path.exists(vp):
                if _STREAM:
                    dict.__setitem__(epoch_V, d, (lambda _vp=vp: _load(_vp) + master_V))
                else:
                    epoch_V[d] = _load(vp) + master_V
        if epoch_V:
            vd = list(dict.keys(epoch_V))
            _mV = _loo_masters(master_V, 'V', 'Taper_R0', mVp)
            resid_V = _build_residuals_mp(epoch_V,
                                          {d: _mV[d] for d in vd},
                                          flows,   # the builder reads flows[d] for d in vd only (same in both modes)
                                          hdr_ref, cd, vd, NPROC_RESID,
                                          spill=(cell_work if _STREAM else None), tag='V')
    _ssV = f'{run_root}/sigstack/{lst}_{scope}_V_sigstack.fits'
    if included_dates and os.path.exists(_ssV):
        sstack_V = np.squeeze(fits.getdata(_ssV)).astype(float); print(f'  {lst} {scope}: INCREMENTAL - sigma-stack V from the persisted cell map')
    else:
        sstack_V = _sigma_stack(resid_V, cell_work, hdr0, 'V') if resid_V else None
        _write_sigstack(sstack_V, hdr0, run_root, lst, scope, 'V')

    gled_rows, gimg_rows = [], []
    v_pending, i_pending = [], []
    _gimgI = {}   # I image rows by date, completed after the post-loop curve stage
    _fhV, _fhI = {}, {}   # conformant frame headers per date (cutouts inherit them)
    # ── R-ROUND2-RATIFIED shared per-cell inputs ────────────────────────
    # (1) BDSF parent catalog on the CURRENT I master: parents for the
    # ratified I 2-D curve AND the v110 stage's CATDIR (gate contract
    # name master_R075_{lst}_{scope}.gaul.csv).
    _catdir = f'{cell_work}/catdir'
    os.makedirs(_catdir, exist_ok=True)
    _gaulp = f'{_catdir}/master_R075_{lst}_{scope}.gaul.csv'
    # per-cell lock (2026-09-08): the I and V searches of one cell may run at once and
    # both build this catalog; without the lock one process removed the other's
    # temporary master (10h Blue battery). Second arrival finds the file and skips.
    _glk = open(f'{_catdir}/.gaul.lock', 'w'); fcntl.flock(_glk, fcntl.LOCK_EX)
    # PARENT LIST ON THE ROBUST -0.75 MASTER (Gregg 2026-09-23: "I thought I chose the Robust -0.75 Masters for the source
    # list... The Robust 0 Masters must be used for subtraction of course"). At 10 minutes master_I is the Robust 0 master
    # the frames are subtracted against; the parent catalogue is built on I_deep_Taper_R075 regardless (the file name
    # already said so). Deep I searches the R075 master, so nothing changes there.
    _pm = f'{base}/coadd_final/I_deep_Taper_R075_{scope}.fits'
    if not os.path.exists(_pm):
        raise RuntimeError(f'{lst} {scope}: no Robust -0.75 master {_pm} for the parent catalogue - REFUSING')
    _pdata = master_I if _pm == mI else _load(_pm)
    _phdr = hdr0 if _pm == mI else fits.getheader(_pm)
    # CAT1/CAT2 (R-CAT1-7 2026-08-15; contract step 23, enforced 2026-09-09): a catalog left in
    # the work dir by a previous generation is a STALE derivative - its .gen.json digest must
    # equal the md5 of the master being searched, else it is discarded and rebuilt (never reused).
    if os.path.exists(_gaulp):
        import hashlib as _hl, json as _jsn
        _cur_md5 = _hl.md5(np.ascontiguousarray(_pdata.astype(np.float32)).tobytes()).hexdigest()   # the parent master (R075)
        try:
            _old = _jsn.load(open(_gaulp + '.gen.json')).get('master_md5_data')
        except Exception:
            _old = None
        if _old != _cur_md5:
            print(f'  {lst} {scope}: parent catalog is STALE (digest {str(_old)[:8]} != master {_cur_md5[:8]}) - rebuilding', flush=True)
            for _x in (_gaulp, _gaulp + '.gen.json'):
                try:
                    os.remove(_x)
                except OSError:
                    pass
    if not os.path.exists(_gaulp):
        import bdsf
        _tmpm = f'{cell_work}/_master_tmp_{os.getpid()}.fits'
        fits.PrimaryHDU(data=_pdata.astype(np.float32),
                        header=_det_header(_phdr, scope, (em if _pm == mI else float(_phdr['BMAJ']) / abs(_phdr['CDELT1'])),
                                           abs(_phdr['CDELT1']), None)   # the parent master's own beam
                        ).writeto(_tmpm, overwrite=True)
        _img = bdsf.process_image(_tmpm, thresh_isl=4.0, thresh_pix=5.0,
                                  quiet=True)
        _img.write_catalog(outfile=_gaulp + '.part', format='csv',
                           catalog_type='gaul', clobber=True)
        os.replace(_gaulp + '.part', _gaulp)
        os.remove(_tmpm)
        # CAT1 generation stamp (R-CAT1-7 2026-08-15; landed 2026-09-04): md5 of
        # the master DATA payload + extraction record, as a sidecar shipped
        # with the catalog. CAT4 (FITS table) remains a DOC row.
        import hashlib, json
        _md5 = hashlib.md5(np.ascontiguousarray(_pdata.astype(np.float32)).tobytes()).hexdigest()
        json.dump({'master': os.path.basename(_pm), 'master_md5_data': _md5,
                   'COADDOP': _phdr.get('COADDOP'), 'NCOADD': int(_phdr.get('NCOADD', 0) or 0),
                   'thresh_pix': 5.0, 'thresh_isl': 4.0,
                   'note': 'stored floor 5.0 (curve consumers cut S>=25 from stored SNR); CAT3 7.0 floor is a DOC row',
                   'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},
                  open(_gaulp + '.gen.json', 'w'), indent=1)
    fcntl.flock(_glk, fcntl.LOCK_UN); _glk.close()
    _parents = _read_gaul_parents(_gaulp)
    # (2) I-master structure map: mstruct_I is a RATIFIED I front gate
    # (R-IVETO-POLICY-RATIFIED 2026-08-31: mstruct AND sigma_stack k=5).
    ms_destr_I, _ = _mstruct_maps(master_I, mI, beam)
    # (1e) SANITY GATE per Stokes (Gregg 2026-09-05: 'both steps are Stokes I and V, treated
    # independently'): V epochs whose uniform frame rms exceeds SANITY_I_X x the cell median of
    # the V frames are refused from the V search (I uses the flux-alignment QA rms above)
    skip_V_dates = set()
    try:
        _vr = {}
        for _d, _rv in resid_V.items():
            try:
                _vr[_d] = float(sv2.get_uniform_rms(np.nan_to_num(_rv, nan=0.0), sv2.make_masks(hdr0, _rv.shape)[0]))
            except Exception:
                pass
        _vf = sorted(v for v in _vr.values() if np.isfinite(v) and v > 0)
        if len(_vf) >= 5:
            _vmed = _vf[len(_vf) // 2]
            skip_V_dates = {d for d, v in _vr.items() if v > SANITY_I_X * _vmed}
            if skip_V_dates:
                with open(f'{_refdir}/{lst}_{scope}_sanity_V_excluded.csv', 'a') as _sf:
                    for d in sorted(skip_V_dates):
                        _sf.write(f'{lst},{scope},{d},{_vr[d]*1e3:.1f},{_vmed*1e3:.1f},{_attempt}\n')
                print(f'  {lst} {scope}: SANITY GATE (V only) — {len(skip_V_dates)} epoch(s) REFUSED from the V search '
                      f'(frame rms >{SANITY_I_X:.0f}x cell median {_vmed*1e3:.0f} mJy): {sorted(skip_V_dates)}')
        skip_V_dates |= {d for d in _skipV_tier if d in resid_V}    # per-Stokes search-tier refusals (Gregg 2026-09-13)
    except Exception as _e:
        print(f'  {lst} {scope}: V sanity gate could not run ({_e})')
    # (1f) SOURCE-EXTRACTION REFUSAL (Gregg 2026-10-11, 'Agreed': "a crashed image is recorded as refused, with the error,
    # and is not counted as searched. The rest of the cell is searched normally."; ledger
    # FINDING-SEARCH-SWALLOWS-PYBDSF-FAILURES-SILENT-ZERO-CANDIDATES). Deep cadence: PyBDSF failing on a search image raises
    # sv2.SourceExtractionFailed (no zero-source result, no fallback finder); the loop below catches exactly that, writes one
    # row per refused image to refusals/<lst>_<scope>_extraction_<S>_excluded.csv (lst, scope, date, Stokes, reason, attempt,
    # UTC time) and writes NO image row, gate-ledger row or candidate for it - its search frame stays archived (the frame
    # exists for every epoch, Gregg 2026-09-13) - and the cell's other images are searched as before. The 10-minute cadence
    # (sub-LST tier, the snapshot / sub-LST chat's) keeps the old behaviour until ruled for that tier.
    _xrefuse = cadence == 'deep'
    _xref = {'I': [], 'V': []}

    def _refuse_extraction(st, d, err):
        _xref[st].append(d)
        with open(f'{_refdir}/{lst}_{scope}_extraction_{st}_excluded.csv', 'a', newline='') as _xf_:
            csv.writer(_xf_, lineterminator='\n').writerow(
                [lst, scope, d, st, f'source extraction failed: {err}', _attempt,
                 time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())])
        print(f'  {lst} {scope} {d}: SOURCE-EXTRACTION GATE - Stokes {st} image REFUSED, not searched '
              f'(source extraction failed: {err})', flush=True)
    for d in dates:
        ot = _epoch_time(ebase, d, scope)
        hdr = _det_header(hdr0, scope, em, cd, ot)
        # 2026-09-01 fix: hdr0 is the MASTER header, whose DURATION is the
        # month-scale coverage span. search_i_transients.get_midpoint_time
        # computes DATE-OBS + DURATION/2, which then lands hours off in
        # time-of-day, and the time-based I elevation mask blanks the
        # field (00h Red: fully masked -> rms NaN -> BDSF refused ->
        # fallback 0 peaks, silently, every epoch). Stamp the epoch's own
        # duration. (V is immune: its deep-path mask is zenith geometry.)
        try:
            hdr['DURATION'] = float(fits.getheader(
                f'{ebase}/subtracted/{d}_I_sub_{scope}.fits'
            ).get('DURATION') or (600.0 if cadence == '10min' else 3600.0))
        except Exception:
            hdr['DURATION'] = 600.0 if cadence == '10min' else 3600.0
        wcs = WCS(hdr).celestial
        if 'I' in stokes and d in resid_I:
            rI = np.nan_to_num(resid_I[d], nan=0.0)
            # 2026-09-04: ARCHIVE the I search frame (S15.2 lowest-level file
            # (3), non-PSF-matched; residuals-required ruling 2026-07-13;
            # products audit B-10: never written before). Same writer as V.
            ir_dir = (f"{os.environ.get('P3_IRESID_ROOT', run_root + '/iresid')}"
                      f'/{lst}_{scope}')
            os.makedirs(ir_dir, exist_ok=True)
            ir_path = f'{ir_dir}/{d}_I_redo_resid_{scope}.fits'
            _hI, _wI = rI.shape
            _halfI = min(512, _hI // 4, _wI // 4)
            from astropy.stats import mad_std as _madsI
            _rmsI = float(round(float(_madsI(
                rI[_hI//2-_halfI:_hI//2+_halfI, _wI//2-_halfI:_wI//2+_halfI],
                ignore_nan=True)), 9))
            _loo_i = f'{base}/coadd_final/loo/{d}_I_loo_Taper_R075_{scope}.fits'
            _hdr_I = _archive_frame_header(
                hdr0, ebase, d, scope, 'I', em, cd, None, _rmsI, mI, _loo_i, mI,
                label, f'{lst} {scope}', cadence=cadence)
            _fhI[d] = _hdr_I
            if not os.path.exists(ir_path):
                _write_archive_frame(ir_path, rI, _hdr_I, f'{lst} {scope}', d)
            if not os.path.exists(ir_path.replace('_redo_resid_', '_redo_resid_pbcorr_')):
                _write_pbcorr_sibling(ir_path, rI, _hdr_I, base, scope, f'{lst} {scope}', d, robust='R075',
                                      pbp=(f'{ebase}/bin_masters/PB_bin{int(_hdr_I.get("LSTBIN"))}_{scope}.fits' if cadence == '10min' and _hdr_I.get('LSTBIN') is not None else None))
            _archive_frame_publish(ir_path, lst, scope, d, 'I', cadence, f'{lst} {scope}')
            if os.path.exists(ir_path.replace('_redo_resid_', '_redo_resid_pbcorr_')):
                _archive_frame_publish(ir_path.replace('_redo_resid_', '_redo_resid_pbcorr_'), lst, scope, d, 'I', cadence, f'{lst} {scope}')
            # R-IVETO-POLICY-RATIFIED (2026-08-31) as shipped under
            # R-ROUND2-RATIFIED (2026-09-03): front gates = sigma_stack
            # (applied inside run_identification_i, k=SIGMA_THRESHOLD)
            # AND mstruct on the I master; morphology veto RETIRED to
            # annotation; the 2-D dynamic curve applies POST-LOOP (flood
            # keying needs every night's post-front count first).
            if d in skip_I_dates:
                gimg_rows.append([lst, scope, 'I', d, 0, 0, '', '', 0, 0, _elev_floor('I')]); continue
            _xfI = None
            try:
                _allI = run_identification_i(rI, master_I, hdr, beam, sstack_I,
                                             refuse_on_extraction_failure=_xrefuse)
            except sv2.SourceExtractionFailed as _xf:
                _xfI = _xf
                _refuse_extraction('I', d, _xf)
        # the I chain continues only for an image whose extraction ran (a refused image has no row anywhere); the
        # header is split, not a 'continue', so the V search of the same epoch still runs in a stokes='IV' search
        if 'I' in stokes and d in resid_I and _xfI is None:
            passed = [c for c in _allI if c['status'] == 'PASS']
            # 2026-09-04 (audit R-09): every identification-stage rejection is
            # a gate-ledger row (local-SNR floor, sigma_stack, merge)
            for c in _allI:
                if c['status'] != 'PASS':
                    gled_rows.append(_gled_row(lst, scope, 'I', d, c, 1,
                                               reason=str(c['status']).lower(),
                                               pass_ms=''))
            if _kd(d) in qa_out:                  # two-tier tag, I as well as V (audit R-12)
                for c in passed:
                    c['epoch_qa'] = 'outlier QA'
            _ebI = _epoch_bmaj(ebase, d, scope, 'I')
            _ebI_min = _epoch_bmin(ebase, d, scope, 'I')
            front = []
            for c in passed:
                _s = abs(float(c.get('local_snr', c.get('snr', 0))))
                _mi = float(ms_destr_I[int(c['pix_y']), int(c['pix_x'])])
                c['mstruct_i'] = round(_mi, 3)
                # 2026-09-05: the I gate VALUES were computed but never recorded on the
                # catalog / ledger rows (0 of 4,064 I catalog rows carried mstruct or sstack)
                c['mstruct'] = round(_mi, 2)
                try:
                    c['sstack'] = round(float(sstack_I[int(c['pix_y']), int(c['pix_x'])]), 2)
                except Exception:
                    pass
                if _s < SIGMA_MSTRUCT * max(_mi, 1.0):
                    gled_rows.append(_gled_row(lst, scope, 'I', d, c, 1,
                                               reason='mstruct_I',
                                               pass_ms=0))
                    continue
                front.append(c)
            _ik, _iv = pv.apply_veto([dict(c) for c in front], rI,
                                     master_I, hdr, em, cd, wcs, ot,
                                     ar_max=ar_max)
            _itag = {}
            for _c in _ik:
                _itag[(round(_c['coord'].ra.deg, 5),
                       round(_c['coord'].dec.deg, 5))] = ('1', '')
            for _c in _iv:
                _itag[(round(_c['coord'].ra.deg, 5),
                       round(_c['coord'].dec.deg, 5))] = \
                    ('0', _c.get('veto_reason', 'veto'))
            for c in front:
                _t = _itag.get((round(c['coord'].ra.deg, 5),
                                round(c['coord'].dec.deg, 5)),
                               ('', 'not-evaluated'))
                c['morph_pass'], c['morph_reason'] = _t
            # Gaussian-fit SIZE removal above 8 sigma (Gregg 2026-09-04)
            _fr2 = []
            for c in front:
                _s = abs(float(c.get('local_snr', c.get('snr', 0))))
                _sb, _sv = _size_test(c, _s, _ebI, cd, 'I', rI, beam, _ebI_min)
                if _sv == 'remove-axis':
                    gled_rows.append(_gled_row(lst, scope, 'I', d, c, 0,
                                               reason='axis-ratio', pass_ms=1))
                    continue
                if _sv == 'remove':
                    gled_rows.append(_gled_row(lst, scope, 'I', d, c, 0,
                                               reason='gaussfit-size', pass_ms=1))
                    continue
                _fr2.append(c)
            front = _fr2
            _drI = sv2.get_uniform_rms(rI, sv2.make_masks(
                hdr, rI.shape)[0])
            i_pending.append((d, front, hdr, ot, float(_drI)))
            _gI = [lst, scope, 'I', d, len(passed), len(front), '', '', '', len(_allI), _elev_floor('I')]
            gimg_rows.append(_gI); _gimgI[d] = _gI
        if d in resid_V:
            rV = np.nan_to_num(resid_V[d], nan=0.0)
            # P13 fix (approved 2026-08-21): ARCHIVE the search's own V residual
            # frame — the exact frame detection runs on — with THERMNOI stamped
            # (inner-box MAD, same formula as phase2 get_inner_rms). The FP-gate
            # verdict stage measures on this file.
            vr_dir = (f"{os.environ.get('P3_VRESID_ROOT', run_root + '/vresid')}"
                      f'/{lst}_{scope}')
            os.makedirs(vr_dir, exist_ok=True)
            vr_path = f'{vr_dir}/{d}_V_redo_resid_{scope}.fits'
            _h, _w = rV.shape
            _half = min(512, _h // 4, _w // 4)
            _inner = rV[_h//2-_half:_h//2+_half, _w//2-_half:_w//2+_half]
            from astropy.stats import mad_std as _mads
            _tn = float(round(float(_mads(_inner, ignore_nan=True)), 9))
            # 2026-09-04 conformant-from-birth writer (replaces the master-
            # header copy that gave every frame its master's time block)
            _loo_v = f'{base}/coadd_final/loo/{d}_V_loo_Taper_R0_{scope}.fits'
            _hdr_out = _archive_frame_header(
                hdr0, ebase, d, scope, 'V', em, cd, _tn, _tn, mVp, _loo_v, mI,
                label, f'{lst} {scope}', cadence=cadence)
            _fhV[d] = _hdr_out
            if not os.path.exists(vr_path):
                _write_archive_frame(vr_path, rV, _hdr_out, f'{lst} {scope}', d)
            if not os.path.exists(vr_path.replace('_redo_resid_', '_redo_resid_pbcorr_')):
                _write_pbcorr_sibling(vr_path, rV, _hdr_out, base, scope, f'{lst} {scope}', d, robust='R0',
                                      pbp=(f'{ebase}/bin_masters/PB_bin{int(_hdr_out.get("LSTBIN"))}_{scope}.fits' if cadence == '10min' and _hdr_out.get('LSTBIN') is not None else None))
            _archive_frame_publish(vr_path, lst, scope, d, 'V', cadence, f'{lst} {scope}')
            if os.path.exists(vr_path.replace('_redo_resid_', '_redo_resid_pbcorr_')):
                _archive_frame_publish(vr_path.replace('_redo_resid_', '_redo_resid_pbcorr_'), lst, scope, d, 'V', cadence, f'{lst} {scope}')
            # GREGG 2026-09-13 (decoupling): the search frame is a science product and exists for every epoch of the
            # generation; the search-tier and sanity gates decide only whether it is SEARCHED. Skip detection here.
            if d in skip_V_dates:
                gimg_rows.append([lst, scope, 'V', d, 0, 0, 0, 0, 0, 0, _elev_floor('V')]); continue
            # sstack is INFO-ONLY (quoted on survivors, lines below), NOT a gate
            # (2026-06-23 ruling: mstruct is the sole V gate). Passing None here keeps
            # run_identification_v from filtering on sigma_stack.
            # WRITE FLOOR = 5sigma (Gregg 2026-06-30): retain >=5sigma in the CSV for
            # repeaters/cross-match; the mstruct gate (below) still vetoes structure.
            try:
                pV = run_identification_v(rV, master_I, hdr, beam, None, sigma_thresh=5.0,
                                          refuse_on_extraction_failure=_xrefuse)
            except sv2.SourceExtractionFailed as _xf:
                _refuse_extraction('V', d, _xf); continue
            # 'outlier QA' tag (Gregg 2026-08-26): candidates from coadd-omitted
            # epochs carry the flag through catalogs to transient review
            if _kd(d) in qa_out:
                for _c in pV:
                    _c['epoch_qa'] = 'outlier QA'
            # R-ROUND2-RATIFIED (Gregg 2026-09-03): gaussfit is RETIRED as a
            # removal — against deep injections it cost 15-20% of faint
            # recoveries for <=10% FP removal. It runs in RECORD mode only:
            # the verdict is annotated for the pages (morph_pass /
            # morph_reason) and every candidate proceeds to mstruct, the
            # sole structure gate. (Supersedes G-09's arms=('gaussfit',)
            # removal; bright_mask and bilobe stay retired as before.)
            _mk, _mv = pv.apply_veto([dict(c) for c in pV], rV, master_I,
                                     hdr, em, cd, wcs, ot,
                                     ar_max=ar_max, arms=('gaussfit',))
            _mtag = {}
            for _c in _mk:
                _mtag[(round(_c['coord'].ra.deg, 5),
                       round(_c['coord'].dec.deg, 5))] = ('1', '')
            for _c in _mv:
                _mtag[(round(_c['coord'].ra.deg, 5),
                       round(_c['coord'].dec.deg, 5))] = \
                    ('0', _c.get('veto_reason', 'veto'))
            for c in pV:
                _t = _mtag.get((round(c['coord'].ra.deg, 5),
                                round(c['coord'].dec.deg, 5)),
                               ('', 'not-evaluated'))
                c['morph_pass'], c['morph_reason'] = _t
            # Gaussian-fit SIZE removal above 8 sigma (Gregg 2026-09-04),
            # against the epoch's own beam (Stage-1 V product header)
            _ebV = _epoch_bmaj(ebase, d, scope, 'V')
            _ebV_min = _epoch_bmin(ebase, d, scope, 'V')
            keptv, vetoed_v = [], []
            for c in pV:
                _s = abs(float(c.get('local_snr', c.get('snr', 0))))
                _sb, _sv = _size_test(c, _s, _ebV, cd, 'V', rV, beam, _ebV_min)
                if _sv == 'remove-axis':
                    gled_rows.append(_gled_row(lst, scope, 'V', d, c, 0,
                                               reason='axis-ratio', pass_ms=1))
                    continue
                if _sv == 'remove':
                    gled_rows.append(_gled_row(lst, scope, 'V', d, c, 0,
                                               reason='gaussfit-size', pass_ms=''))
                    continue
                keptv.append(c)
            n_after_morph = len(keptv)
            vetoed_ms, vetoed_jup, vetoed_moon = [], [], []
            # (2) mstruct gate + (3) Jupiter veto, quoting mstruct/sstack on survivors
            if keptv and ms_destr is not None:
                # subband scopes map to their band's Jupiter radius
                _b = {'18':'Red','23':'Red','27':'Red','32':'Red','36':'Red',
                      '41':'Green','46':'Green','50':'Green','55':'Green',
                      '59':'Green','64':'Blue','69':'Blue','73':'Blue',
                      '78':'Blue','82':'Blue'}.get(
                          scope.replace('MHz',''), scope)
                jrad = JUPITER_VETO_DEG.get(_b)
                # Card 7a (ratified 2026-08-11): ephemeris positions at the
                # EPOCH's own observation time, not a fixed convention.
                jup = _body_skycoord('jupiter', ot, _kd(d)) if jrad else None
                moon = _body_skycoord('moon', ot, _kd(d))
                gated = []
                for c in keptv:
                    xx, yy = int(c['pix_x']), int(c['pix_y'])
                    snr = abs(float(c.get('local_snr', c.get('snr', 0))))
                    msv = float(ms_destr[yy, xx])
                    # quote the gate measurables on EVERY candidate (parity
                    # ledger needs them on vetoed rows too, not just survivors)
                    c['mstruct'] = round(msv, 2)
                    c['mstruct_raw'] = round(float(ms_raw[yy, xx]), 2)
                    c['sstack'] = (round(float(sstack_V[yy, xx]), 2)
                                   if sstack_V is not None else '')
                    if snr < SIGMA_MSTRUCT * max(msv, 1.0):       # mstruct sole gate
                        vetoed_ms.append(c)
                        continue
                    if jrad is not None:
                        jdeg = float(wcs.pixel_to_world(xx, yy).separation(jup).deg)
                        c['jupiter_deg'] = round(jdeg, 2)
                        if jdeg < jrad:                           # Jupiter Red+Full veto
                            vetoed_jup.append(c)
                            continue
                    # Moon 15-arcmin removal, all bands, per-epoch (Card 7c)
                    mdeg = float(wcs.pixel_to_world(xx, yy).separation(moon).deg)
                    c['moon_deg'] = round(mdeg, 2)
                    if mdeg < MOON_VETO_DEG:
                        vetoed_moon.append(c)
                        continue
                    # A-team Card 6 ratified table: six sources, V 3.0 deg
                    adeg = float(min(wcs.pixel_to_world(xx, yy)
                                     .separation(_ATEAM6_COORDS).deg))
                    c['nearest_ateam'] = round(adeg, 2)
                    if adeg < ATEAM_VETO_DEG_V:
                        c['veto_reason'] = 'ateam'
                        # ledger row here: the vetoed_v write-out already ran
                        # (pre-existing gap — ateam rows were silently absent)
                        gled_rows.append(_gled_row(lst, scope, 'V', d, c, 1,
                                                   reason='ateam', pass_ms=1))
                        vetoed_v.append(c)
                        continue
                    gated.append(c)
                keptv = gated
                # RECORD the per-epoch surviving-candidate count on each row (Gregg
                # 2026-06-30: this is a diagnostic, NOT a gate — never silently drop).
                # A high count flags a likely artifact-ridden epoch; the user filters.
                ne = len(keptv)
                for c in keptv:
                    c['n_v_epoch'] = ne
                if ne > MAX_V_PER_EPOCH:
                    print(f'  {lst} {scope} {d}: {ne} gated V candidates -> HIGH COUNT '
                          f'(flagged n_v_epoch, NOT dropped)')
            gate_ran = ms_destr is not None and n_after_morph > 0
            for c in vetoed_ms:
                gled_rows.append(_gled_row(lst, scope, 'V', d, c, 1,
                                           reason='mstruct', pass_ms=0))
            for c in vetoed_jup:
                gled_rows.append(_gled_row(lst, scope, 'V', d, c, 1,
                                           reason='jupiter', pass_ms=1,
                                           jup_veto=1))
            for c in vetoed_moon:
                gled_rows.append(_gled_row(lst, scope, 'V', d, c, 1,
                                           reason='moon', pass_ms=1,
                                           jup_veto=0))
            n_after_ms = n_after_morph - len(vetoed_ms)
            gimg_rows.append([lst, scope, 'V', d, len(pV), n_after_morph,
                              n_after_ms if gate_ran else '',
                              (n_after_ms - len(vetoed_jup)) if gate_ran else '',
                              len(keptv), len(pV), _elev_floor('V')])
            # R-ROUND2-RATIFIED: survivors are NOT final until the frozen
            # v110 gate rules on them — stash for the per-cell verdict
            # stage below (one gate invocation per cell; the archived
            # vr_path frame is what the gate measures on).
            if keptv:
                v_pending.append((d, keptv, vr_path, hdr, ot,
                                  1 if gate_ran else '', ))
    # ── I 2-D BRIGHT-PARENT STAGE. The ratified flood-night curve (R25 1.0 nominal / 1.4 on a flood
    # night) was SUPERSEDED on 2026-09-05/06 by the measured per-Stokes per-band table, and flood keying
    # was retired with it (see the table call below): nothing downstream has read a flood classification
    # since. The keying nevertheless survived here as three dead assignments plus a printed line claiming
    # that an uncalibrated tier "falls back to 3x the cell median", which was never applied to anything
    # and which the contract repeated as though it changed 10-minute products. Removed 2026-09-13.
    # GREGG 2026-09-13, on the 10-minute flood threshold: "there is no threshold until we review battery
    # results." There is none, at either cadence: no night is classified and no radius is widened.
    if i_pending:
        from scipy.spatial import cKDTree as _KD
        _pa = np.array(_parents) if _parents else np.zeros((0, 3))

        def _uv(ra_, de_):
            r_, d_ = np.radians(ra_), np.radians(de_)
            return np.stack([np.cos(d_) * np.cos(r_),
                             np.cos(d_) * np.sin(r_), np.sin(d_)], -1)
        _ptree = _KD(_uv(_pa[:, 0], _pa[:, 1])) if len(_pa) else None
        for d_, front, hdr_, ot_, drI_ in i_pending:
            # Ephemeris always-on for I (ratified; omission caught by Gregg
            # 2026-09-04 - Jupiter at 80 sigma on the Blue I pages):
            # Jupiter per band radius + Moon 15 arcmin, per-epoch positions.
            _wI = WCS(hdr_).celestial
            _b = {'18':'Red','23':'Red','27':'Red','32':'Red','36':'Red',
                  '41':'Green','46':'Green','50':'Green','55':'Green',
                  '59':'Green','64':'Blue','69':'Blue','73':'Blue',
                  '78':'Blue','82':'Blue'}.get(scope.replace('MHz',''), scope)
            _jrad = JUPITER_VETO_DEG.get(_b, 0.5)
            _jup = _body_skycoord('jupiter', ot_, _kd(d_))
            _moon = _body_skycoord('moon', ot_, _kd(d_))
            finals = []
            for c in front:
                _sky = _wI.pixel_to_world(int(c['pix_x']), int(c['pix_y']))
                c['jupiter_deg'] = round(float(_sky.separation(_jup).deg), 3)
                c['moon_deg'] = round(float(_sky.separation(_moon).deg), 3)
                if c['jupiter_deg'] < _jrad:
                    gled_rows.append(_gled_row(lst, scope, 'I', d_, c, 1,
                                               reason='jupiter', pass_ms=1,
                                               jup_veto=1))
                    continue
                if c['moon_deg'] < MOON_VETO_DEG:
                    gled_rows.append(_gled_row(lst, scope, 'I', d_, c, 1,
                                               reason='moon', pass_ms=1))
                    continue
                # A-team DOWNSTREAM veto, I 5.0 deg, six sources (Card 6;
                # landed 2026-09-04 - the I chain had none)
                _adeg = float(min(_sky.separation(_ATEAM6_COORDS).deg))
                c['nearest_ateam'] = round(_adeg, 2)
                if _adeg < ATEAM_VETO_DEG_I:
                    gled_rows.append(_gled_row(lst, scope, 'I', d_, c, 1,
                                               reason='ateam', pass_ms=1))
                    continue
                cut = False
                if _ptree is not None:
                    _dd, _ii = _ptree.query(_uv(
                        np.array([c['coord'].ra.deg]),
                        np.array([c['coord'].dec.deg])), k=1)
                    _sep_bw = (2 * np.arcsin(_dd[0] / 2) * 180 / np.pi) \
                        / (em * cd)
                    c['parent_jy'] = round(float(_pa[_ii[0], 2]), 3)
                    c['parent_off_beams'] = round(float(_sep_bw), 3)
                    # PER-STOKES, PER-BAND TABLE R(S) (Gregg 2026-09-05/06, supersedes the single
                    # ratified curve R25 1.0/1.4 gamma 0.35; flood keying retired with it):
                    # measured table from exopipe/common/parent_curve_tables.json
                    # phase centre = the master's CRVAL (as gate_inputs.phase_centre): an image-pixel table
                    # measures in direction cosines about it; a sky table ignores it (Gregg 2026-10-01)
                    # ELEVATION-BAND TABLES IN THE SEARCH (Gregg 2026-10-06 "2. yes"): deep, with P3_TABLE_ELEV=1, each candidate
                    # is judged by the band table of its mid-hour elevation, as the conformance judges it (removals ignores elev
                    # otherwise); other cadences unchanged
                    _tc, _tn = _PCT.removals('I', scope, _pa, max(drI_, 1e-12), em * cd,
                                             [c['coord'].ra.deg], [c['coord'].dec.deg],
                                             elev=(_mid_hour_elev(lst, [c['coord'].ra.deg], [c['coord'].dec.deg]) if cadence == 'deep' else None),
                                             phase_centre=(float(hdr0['CRVAL1']) % 360.0, float(hdr0['CRVAL2'])))
                    if bool(_tc[0]):
                        cut = True; c['table_note'] = _tn[0]
                if cut:
                    gled_rows.append(_gled_row(lst, scope, 'I', d_, c, 1,
                                               reason='table', pass_ms=1))
                    continue
                finals.append(c)   # the final=1 ledger row is written after the row-line veto
            # ROW-LINE veto (Gregg 2026-09-05, ratified): >= ROWLINE_MIN I finals of one epoch on
            # one pixel row +/-1 = the CLEAN-divergence stripe -> all of them removed, ledgered
            if len(finals) >= ROWLINE_MIN:
                _ys = np.array([int(c['pix_y']) for c in finals]); _drop = set()
                for _y in set(_ys.tolist()):
                    _m = np.where(np.abs(_ys - _y) <= 1)[0]
                    if len(_m) >= ROWLINE_MIN: _drop.update(_m.tolist())
                if _drop:
                    _keep = []
                    for _i, c in enumerate(finals):
                        if _i in _drop:
                            gled_rows.append(_gled_row(lst, scope, 'I', d_, c, 1, reason='row-line', pass_ms=1))
                        else:
                            _keep.append(c)
                    print(f'  {lst} {scope} {d_}: ROW-LINE veto removed {len(_drop)} of {len(finals)} I finals on one pixel row')
                    finals = _keep
            for c in finals:
                gled_rows.append(_gled_row(lst, scope, 'I', d_, c, 1, pass_ms=1, final=1))
            if d_ in _gimgI:   # n_after_jupiter (post-body vetoes) and n_final for the I image row
                _gimgI[d_][7] = sum(1 for c in front if c.get('jupiter_deg', 99) >= _jrad and c.get('moon_deg', 99) >= MOON_VETO_DEG)
                _gimgI[d_][8] = len(finals)
            if finals and not GATE_LEDGER_ONLY:
                pc.write_candidates(root, label, finals, resid_I[d_],
                                    master_I, hdr_, lst, d_, scope, 'I',
                                    cadence, em, cd, ot_, frame_hdr=_fhI.get(d_))
                nwr += len(finals)
    # ── IN-CHAIN v110 VERDICT STAGE (R-ROUND2-RATIFIED, Gregg 2026-09-03:
    # V chain = 5sig -> mstruct -> v110 in-chain [MARGINAL reading] ->
    # ephemerides). One frozen-gate invocation over the cell's survivors;
    # fp_prediction in {FP, marginal} REMOVES (reading override
    # P3_V110_READING=strict; escape P3_V110_OFF=1 -> annotate only).
    verdicts = {}
    if v_pending and os.environ.get('P3_V110_OFF') != '1':
        verdicts = _v110_cell_verdicts(lst, scope, base, cell_work,
                                       master_I, hdr0, em, cd, v_pending)
    # Gregg 2026-09-04: FP removed; marginal AND inconclusive RETAINED and
    # LABELLED (fp_prediction on the catalog row). P3_V110_READING=marginal
    # restores the round-2 reading (marginal removed) for comparison only.
    _v110_removes = ({'FP', 'marginal'} if os.environ.get('P3_V110_READING') == 'marginal'
                     else {'FP'})
    for d, keptv, _vr, hdr, ot, _pm in v_pending:
        finals = []
        for c in keptv:
            _vk = f'{d}_{round(c["coord"].ra.deg,5)}_{round(c["coord"].dec.deg,5)}'
            _pred = verdicts.get(_vk, ('' if verdicts else 'gate-off'))
            c['fp_prediction'] = _pred
            if _pred in _v110_removes:
                gled_rows.append(_gled_row(lst, scope, 'V', d, c, 1,
                                           reason=f'v110:{_pred}',
                                           pass_ms=_pm, jup_veto=0))
                continue
            c['_pm'] = _pm
            finals.append(c)   # final=1 ledger row written after the row-line veto
        # PER-STOKES, PER-BAND TABLE R(S) for V (Gregg 2026-09-05/06): bright-parent residuals
        # that v110 passes; parents = the cell's I master catalog, S = parent peak / this
        # epoch's V frame global rms (drV_ when recorded, else the cell median)
        if finals and _parents:
            _pa_v = np.array(_parents)
            _rmsV = float(np.nanmedian([abs(float(c.get('flux', 0))) / abs(float(c.get('global_snr', c.get('snr', 1)) or 1)) for c in finals])) if finals else 0.0
            # phase centre (2026-10-02, INCIDENT-DEEP-V-TABLE-CALL-WITHOUT-PHASE-CENTRE): the deep tables are in the
            # image-pixel metric since 2026-10-01 and refuse without it; the I call above already passed it
            _tcv, _tnv = _PCT.removals('V', scope, _pa_v, max(_rmsV, 1e-12), em * cd,
                                       [c['coord'].ra.deg for c in finals], [c['coord'].dec.deg for c in finals],
                                       elev=(_mid_hour_elev(lst, [c['coord'].ra.deg for c in finals], [c['coord'].dec.deg for c in finals])
                                             if cadence == 'deep' else None),   # elevation-band tables in the search (Gregg 2026-10-06)
                                       phase_centre=(float(hdr0['CRVAL1']) % 360.0, float(hdr0['CRVAL2'])))
            _keepv = []
            for c, _x, _n in zip(finals, _tcv, _tnv):
                if bool(_x):
                    c['table_note'] = _n
                    gled_rows.append(_gled_row(lst, scope, 'V', d, c, 1, reason='table', pass_ms=c.get('_pm', ''), jup_veto=0))
                else:
                    _keepv.append(c)
            finals = _keepv
        if len(finals) >= ROWLINE_MIN:   # ROW-LINE veto, V (Gregg 2026-09-05, independent of I)
            _ys = np.array([int(c['pix_y']) for c in finals]); _drop = set()
            for _y in set(_ys.tolist()):
                _m = np.where(np.abs(_ys - _y) <= 1)[0]
                if len(_m) >= ROWLINE_MIN: _drop.update(_m.tolist())
            if _drop:
                _keep = []
                for _i, c in enumerate(finals):
                    if _i in _drop:
                        gled_rows.append(_gled_row(lst, scope, 'V', d, c, 1, reason='row-line', pass_ms=c.get('_pm', 1)))
                    else:
                        _keep.append(c)
                print(f'  {lst} {scope} {d}: ROW-LINE veto removed {len(_drop)} of {len(finals)} V finals on one pixel row')
                finals = _keep
        for c in finals:
            gled_rows.append(_gled_row(lst, scope, 'V', d, c, 1, pass_ms=c.get('_pm', 1),
                                       jup_veto=0 if c.get('_pm', 1) else '', final=1))
        if finals and not GATE_LEDGER_ONLY:
            pc.write_candidates(root, label, finals, resid_V[d], master_I,
                                hdr, lst, d, scope, 'V', cadence, em, cd, ot,
                                frame_hdr=_fhV.get(d))
            nwr += len(finals)
    _append_locked(f'{run_root}/v_deep_ledger.csv', GLED_COLS, gled_rows)
    _append_locked(f'{run_root}/v_deep_images.csv', GIMG_COLS, gimg_rows)
    # ZERO-PRODUCT REFUSAL (2026-09-13, CLAUDE.md section 3: "a rebuild that silently no-ops is a bug"
    # and "wrote 0 candidates never produces a done marker"). A run that searched epochs but wrote no
    # image row wrote nothing at all - no ledger row, no frame, no residual - and must NOT print the
    # completion line, because the fleets read that line as proof of a publishable attempt. 07h_Blue
    # Stokes V did exactly this on 2026-09-13: 29 minutes, "124 epochs -> 0 candidates (0 gate-ledger
    # rows, 0 image rows)", the done sentinel, and nothing on disk but two gate records; its worker then
    # adopted the attempt four times before parking the cell. An epoch count of zero is a different and
    # legitimate case (every epoch refused upstream) and still reports normally.
    if dates and not gimg_rows:
        # a run in which source extraction failed on EVERY image it searched writes no image row and is refused here as
        # well (the attempt fails and the fleet retries it): the 2026-10-04 test harness made every PyBDSF call fail
        raise RuntimeError(
            f'{lst} {scope}: REFUSING - {len(dates)} epoch(s) entered the search and NOT ONE image row was '
            f'written, so this run produced no ledger row, no frame and no residual. The completion line is '
            f'withheld. Causes seen: the Stokes loop never ran because no residual of that Stokes reached it. '
            f'Source extraction failed on {len(_xref["I"])} I and {len(_xref["V"])} V image(s) of this run '
            f'(refusals/{lst}_{scope}_extraction_<S>_excluded.csv). '
            f'Re-run this cell with unbuffered output to see which stage dropped them.')
    print(f'  {lst} {scope}: {len(dates)} epochs -> {nwr} candidates  '
          f'({len(gled_rows)} gate-ledger rows, {len(gimg_rows)} image rows)  '
          + (f'(source extraction failed, image REFUSED and not searched: I {len(_xref["I"])}, V {len(_xref["V"])})  '
             if _xrefuse else '')
          + f'({(time.time()-t0)/60:.1f} min)', flush=True)
    return nwr


if __name__ == '__main__':
    lst, scope = _ARGV[1], _ARGV[2]
    label = _ARGV[3] if len(_ARGV) > 3 else 'PROD'
    stokes = _ARGV[4] if len(_ARGV) > 4 else 'IV'
    run_deep_cell(lst, scope, label, stokes=stokes)
