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
from exopipe.common.roots import coadd_root
import os, sys, glob, re, subprocess, time, csv, fcntl
_ARGV = list(sys.argv)        # captured before lab imports that reset sys.argv
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from exopipe.phase3.coadd_full_pipeline import apply_flow_field, build_striation_model
from exopipe.phase3.phase3_pipeline import (build_sigma_stack, run_identification_i,
                             run_identification_v, STRIATION_WINDOW_DEG)
from exopipe.phase3.residuals import fft_upsample2, fft_decimate2, write_up_fits
from exopipe.phase3.identify import fit_empirical_beam
from exopipe.phase3.phase3_tenmin import _det_header
from exopipe.phase3 import phase3_veto as pv
from exopipe.phase3 import phase3_catalog as pc
from exopipe.phase3 import search_v_transients_v2 as sv2
from astropy.coordinates import SkyCoord as _SC
_ATEAM_COORDS = _SC([sv2.CONFIG['A_TEAM'][k]['coord'].ra.deg
                     for k in sv2.CONFIG['A_TEAM']],
                    [sv2.CONFIG['A_TEAM'][k]['coord'].dec.deg
                     for k in sv2.CONFIG['A_TEAM']], unit='deg')
from scipy.ndimage import uniform_filter
from astropy.time import Time
from astropy.coordinates import get_body, SkyCoord
from exopipe.phase3.coadd_full_pipeline import OVRO_LOC, load_exclusions, is_excluded

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
             'jupiter_deg', 'jupiter_veto', 'final', 'n_v_epoch', 'moon_deg']
GIMG_COLS = ['lst', 'scope', 'stokes', 'date', 'n_ident5', 'n_after_morph',
             'n_after_mstruct', 'n_after_jupiter', 'n_final']
# P3_GATE_LEDGER_ONLY=1: write ONLY the gate-ledger CSVs, skip master_catalog
# + candidate cutouts. Used by the posterity BACKFILL over already-searched
# cells (Gregg 2026-07-11) so re-runs never duplicate catalog rows.
GATE_LEDGER_ONLY = os.environ.get('P3_GATE_LEDGER_ONLY', '') == '1'


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
            int(final), c.get('n_v_epoch', ''), c.get('moon_deg', '')]


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
ALIGN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'phase2', 'align_images.py')
WORKROOT = os.environ.get('P3_FLOW_ROOT', '/data07/gh/phase3/deep')   # moved off /fast (NVMe too small; was filling it)
COADD_ROOT = os.environ.get('P3_COADD_ROOT')   # NO default (Gregg 2026-09-04); use coadd_root() at point of use
MIN_EPOCHS = 5
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
    """items=[(date, ep_up_path)] -> {date: ×2 flow}, via align_images.py
    (py310_orca GPU). Cached on disk; only missing pairs are recomputed."""
    pairs = []
    for date, ep_up in items:
        fb = os.path.splitext(os.path.basename(ep_up))[0]
        if not os.path.exists(f'{cell_work}/{fb}_flow.npy'):
            pairs.append(f'{ep_up} {mup} {cell_work}')
    if pairs:
        pp = f'{cell_work}/flow_pairs.txt'
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
        r = subprocess.run([flow_py, ALIGN,
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
    flows = {}
    for date, ep_up in items:
        fb = os.path.splitext(os.path.basename(ep_up))[0]
        fp = f'{cell_work}/{fb}_flow.npy'
        if os.path.exists(fp):
            flows[date] = np.load(fp)
    return flows


def _residual(epoch, master, flow, hdr, cd):
    """×2-upflow refine epoch -> direct subtract master -> striation clean."""
    up = fft_decimate2(apply_flow_field(fft_upsample2(epoch), flow))
    invalid = (up == 0) | (master == 0) | ~np.isfinite(up)
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


def _build_residuals_mp(epoch, master, flows, hdr, cd, dates, nproc):
    """Build per-epoch residuals in parallel. Uses fork so the big master +
    flow arrays are shared copy-on-write (not pickled); only the date index is
    passed and the residual returned. Identical output to the serial path."""
    import multiprocessing as mp
    global _RESID_CTX
    _RESID_CTX = dict(epoch=epoch, master=master, flows=flows, hdr=hdr, cd=cd)
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
        tp = f'{cell_work}/ss_{tag}_{d}.fits'
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
                       timeout=14400)
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


def run_deep_cell(lst, scope, label='PROD', included_dates=None, stokes='IV'):
    t0 = time.time()
    if stokes == 'V':
        # G-03 (ratified 2026-07-14/15, deployed here per G-09 redo 2026-08-21):
        # A-team is a DOWNSTREAM veto stage, never a pre-detection blank. The
        # V redo path searches the full sky and vetoes at the ratified radii
        # below. (Left ON for 'IV' runs: the I side has no downstream A-team
        # veto stage yet, so removing its pre-blank would leave I unprotected.)
        os.environ['P3_ATEAM_MASK_OFF'] = '1'
    base = f'{coadd_root()}/{lst}_{scope}'
    # BUG FIX (Gregg order 2026-08-29): wideband used the Taper R0 I master
    # here while subtracted/*_I_sub_* is built on the Taper R-0.75 chain
    # (coadd_full_pipeline.py:2681) — the epoch reconstruction I_sub+master
    # was therefore a variant hybrid, planting master-gradient artifacts at
    # source positions and mis-calibrating beam-based tests (it also fed the
    # V search's co-located-I context). R-VARIANT-DOC: I = Taper_Robust-0.75.
    _irob = 'R075'
    mI = f'{base}/coadd_final/I_deep_Taper_{_irob}_{scope}.fits'
    if not os.path.exists(mI):
        print(f'  {lst} {scope}: no deep I master ({_irob}) — skip'); return 0
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
    # comparison, never for production.
    _op = hdr0.get('COADDOP')
    if _op != 'ivw-mean-v2' and os.environ.get('P3_ALLOW_OLD_MASTERS') != '1':
        raise RuntimeError(
            f'{lst} {scope}: REFUSING to search a superseded master. '
            f'{mI} has COADDOP={_op!r}, expected \'ivw-mean-v2\' '
            f'(plain-mean + LOO policy, ratified 2026-08-25). Rebuild the '
            f'cell, or point P3_COADD_ROOT at the current-generation root '
            f'(exopipe_ops/resolve_master_root.sh resolves it).')
    em, en, _, _ = fit_empirical_beam(master_I)
    if not np.isfinite(em) or em <= 0:   # beam fit can fail (e.g. 14h Blue) -> NaN propagates into headers
        print(f'  {lst} {scope}: empirical-beam fit returned NaN; using fallback beam=7px')
        em = 7.0; en = 7.0
    beam = em
    ar_max = max(2.0, 1.5 * em / max(en, 1e-3))
    mVp = f'{base}/coadd_final/V_deep_Taper_R0_{scope}.fits'
    if os.path.exists(mVp):
        _opv = fits.getheader(mVp).get('COADDOP')
        if _opv != 'ivw-mean-v2' and \
                os.environ.get('P3_ALLOW_OLD_MASTERS') != '1':
            raise RuntimeError(
                f'{lst} {scope}: REFUSING to search a superseded V master. '
                f'{mVp} has COADDOP={_opv!r}, expected \'ivw-mean-v2\'.')
    master_V = _load(mVp) if os.path.exists(mVp) else None
    ms_destr, ms_raw = (_mstruct_maps(master_V, mVp, beam)
                        if master_V is not None else (None, None))
    cell_work = f'{WORKROOT}/{lst}_{scope}'; os.makedirs(cell_work, exist_ok=True)

    dates = []
    for c in sorted(glob.glob(f'{base}/subtracted/*_I_sub_{scope}.fits')):
        m = re.search(r'(\d{4}-\d\d-\d\d)_I_sub_', os.path.basename(c))
        if m:
            dates.append(m.group(1))
    if included_dates:
        dates = [d for d in dates if d in included_dates]
    if len(dates) == 0:
        raise RuntimeError(f'{lst} {scope}: ZERO epochs found — inputs '
                           f'missing/unshipped; REFUSING (a 0-epoch cell '
                           f'is never legitimate)')
    if len(dates) < MIN_EPOCHS:
        print(f'  {lst} {scope}: only {len(dates)} epochs — skip'); return 0

    mup = f'{cell_work}/master_up2.fits'
    if not os.path.exists(mup):
        write_up_fits(fft_upsample2(master_I), hdr0, mup)
    items, epoch_I = [], {}
    for d in dates:
        ei = _load(f'{base}/subtracted/{d}_I_sub_{scope}.fits') + master_I; epoch_I[d] = ei
        ep_up = f'{cell_work}/ep_{d}_up2.fits'
        if not os.path.exists(ep_up):
            write_up_fits(fft_upsample2(ei), hdr0, ep_up)
        items.append((d, ep_up))
    flows = _batch_flows(cell_work, mup, items)
    dates = [d for d in dates if d in flows]
    # (0) drop data_decisions-excluded epochs — the master already excludes them, but the
    # per-epoch residual still exists on disk, so the search would otherwise pull them back
    # in (e.g. vortex night 2025-05-04; mstruct can't catch it — it's not in the master).
    _excl = load_exclusions()
    _ddrop = [d for d in dates if is_excluded(lst, scope, d, _excl)]
    if _ddrop:
        dates = [d for d in dates if d not in _ddrop]
        print(f'  {lst} {scope}: dropped {len(_ddrop)} data_decisions-excluded epoch(s): {_ddrop}')
    # (1) TWO-TIER QUALITY POLICY (Gregg ruling 2026-08-26, supersedes the
    # earlier QA-outlier search drop / G-02): stage-2 QA rejection omits an
    # epoch from the CO-ADD ONLY; every non-data_decisions epoch is still
    # searched. Egregious nights (vortex, large peel errors) are handled by
    # data_decisions via the QA-3 withhold-for-review flow — the old inline
    # drop (motivated by a 79sigma peeling-blowup flooding the list) is now
    # that flow's job, not the search's.
    qa_out = _qa_outlier_dates(lst, scope)
    _qa_note = [d for d in dates if d in qa_out]
    if _qa_note:
        print(f'  {lst} {scope}: {len(_qa_note)} coadd-omitted (QA-outlier) epoch(s) '
              f'REMAIN in search per two-tier policy: {_qa_note}')
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
                    if _wr.get(d, 0) > 10.0 * _med]
            if _bad:
                dates = [d for d in dates if d not in _bad]
                with open(f'{base}/sanity_excluded.csv', 'a') as _sf:
                    for d in _bad:
                        _sf.write(f'{lst},{scope},{d},{_wr[d]:.1f},'
                                  f'{_med:.1f}\n')
                print(f'  {lst} {scope}: SANITY GATE — {len(_bad)} epoch(s) '
                      f'REFUSED from search (alignment rms >10x cell median '
                      f'{_med:.0f} mJy): {_bad}')
    except FileNotFoundError:
        pass
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
            _g = [x for x in
                  glob.glob(f'{_p2}/{lst}/{d}/*/{scope}/I/deep/'
                            f'*Taper_{_irob_f}_dewarped*.fits')
                  + glob.glob(f'{_p2}/{lst}/{d}/*/{scope}/I/deep/'
                              f'*Taper-*-{_irob_f}-*dewarped*.fits')
                  if 'pbcorr' not in os.path.basename(x)]
            if _g:
                try:
                    _h = fits.getheader(_g[0])
                    _areas[d] = float(_h['BMAJ']) * float(_h['BMIN'])
                except Exception:
                    pass
        _fin = sorted(_areas.values())
        if len(_fin) >= 5:
            _bmed = _fin[len(_fin) // 2]
            _thr = float(os.environ.get('P3_BEAM_GATE_RATIO', '1.5'))
            _bbad = [d for d in dates
                     if _areas.get(d, 0) > _thr * _bmed]
            if _bbad:
                dates = [d for d in dates if d not in _bbad]
                with open(f'{base}/beam_excluded.csv', 'a') as _bf:
                    for d in _bbad:
                        _bf.write(f'{lst},{scope},{d},'
                                  f'{_areas[d]/_bmed:.2f},{_thr}\n')
                print(f'  {lst} {scope}: BEAM GATE — {len(_bbad)} epoch(s)'
                      f' REFUSED from search (fitted-beam area >{_thr}x '
                      f'cell median): '
                      f'{[(d, round(_areas[d]/_bmed, 2)) for d in _bbad]}')
    if len(dates) == 0:
        raise RuntimeError(f'{lst} {scope}: ZERO flows — REFUSING')
    if len(dates) < MIN_EPOCHS:
        print(f'  {lst} {scope}: only {len(dates)} flows — skip'); return 0

    hdr_ref = _det_header(hdr0, scope, em, cd, None)
    # R-LOO-MASTERS as SEARCH FRAMES (conformance fix 2026-09-03): an epoch
    # is differenced against ITS OWN leave-one-out master where one exists
    # (file existence encodes the ratified per-Stokes [2,20] rule); the
    # central master is the fallback and remains the flow/mstruct/parent
    # reference for all epochs (declared: alignment and structure
    # references stay common across epochs).
    def _loo_masters(central, stokes_tag, variant_tag):
        out = {}
        for _d in dates:
            _lp = (f'{base}/coadd_final/loo/'
                   f'{_d}_{stokes_tag}_loo_{variant_tag}_{scope}.fits')
            out[_d] = _load(_lp) if os.path.exists(_lp) else central
        _n = sum(1 for _d in dates if out[_d] is not central)
        if _n:
            print(f'  {lst} {scope}: {_n}/{len(dates)} epochs searched '
                  f'against their leave-one-out {stokes_tag} masters')
        return out
    if 'I' in stokes:
        resid_I = _build_residuals_mp(epoch_I,
                                      _loo_masters(master_I, 'I',
                                                   'Taper_R075'),
                                      flows, hdr_ref, cd, dates, NPROC_RESID)
        dates = [d for d in dates if d in resid_I]
        sstack_I = _sigma_stack(resid_I, cell_work, hdr0, 'I')
    else:
        resid_I, sstack_I = {}, None   # V-only redo (Gregg 2026-08-21): Stokes I is out
    resid_V = {}
    # 2026-09-01 fix: honor stokes='I' — the V side previously ran whenever
    # a V master existed, so an I-only extraction silently re-searched V
    # (duplicating the survey-complete V ledger into the new run's dir).
    # 'V' and 'IV' behavior unchanged.
    if master_V is not None and 'V' in stokes:
        epoch_V = {}
        for d in dates:
            vp = f'{base}/subtracted/{d}_V_sub_{scope}.fits'
            if os.path.exists(vp):
                epoch_V[d] = _load(vp) + master_V
        if epoch_V:
            vd = list(epoch_V)
            _mV = _loo_masters(master_V, 'V', 'Taper_R0')
            resid_V = _build_residuals_mp(epoch_V,
                                          {d: _mV[d] for d in vd},
                                          {d: flows[d] for d in vd},
                                          hdr_ref, cd, vd, NPROC_RESID)
    sstack_V = _sigma_stack(resid_V, cell_work, hdr0, 'V') if resid_V else None

    root = os.environ.get('P3_STAGING_ROOT', pc.P.STAGING_ROOT); nwr = 0
    run_root = pc.P.transients_run(root, label, make=True)
    gled_rows, gimg_rows = [], []
    v_pending, i_pending = [], []
    # ── R-ROUND2-RATIFIED shared per-cell inputs ────────────────────────
    # (1) BDSF parent catalog on the CURRENT I master: parents for the
    # ratified I 2-D curve AND the v110 stage's CATDIR (gate contract
    # name master_R075_{lst}_{scope}.gaul.csv).
    _catdir = f'{cell_work}/catdir'
    os.makedirs(_catdir, exist_ok=True)
    _gaulp = f'{_catdir}/master_R075_{lst}_{scope}.gaul.csv'
    if not os.path.exists(_gaulp):
        import bdsf
        _tmpm = f'{cell_work}/_master_tmp.fits'
        fits.PrimaryHDU(data=master_I.astype(np.float32),
                        header=_det_header(hdr0, scope, em, cd, None)
                        ).writeto(_tmpm, overwrite=True)
        _img = bdsf.process_image(_tmpm, thresh_isl=4.0, thresh_pix=5.0,
                                  quiet=True)
        _img.write_catalog(outfile=_gaulp, format='csv',
                           catalog_type='gaul', clobber=True)
        os.remove(_tmpm)
    _parents = _read_gaul_parents(_gaulp)
    # (2) I-master structure map: mstruct_I is a RATIFIED I front gate
    # (R-IVETO-POLICY-RATIFIED 2026-08-31: mstruct AND sigma_stack k=5).
    ms_destr_I, _ = _mstruct_maps(master_I, mI, beam)
    for d in dates:
        ot = _epoch_time(base, d, scope)
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
                f'{base}/subtracted/{d}_I_sub_{scope}.fits'
            ).get('DURATION') or 3600.0)
        except Exception:
            hdr['DURATION'] = 3600.0
        wcs = WCS(hdr).celestial
        if 'I' in stokes and d in resid_I:
            rI = np.nan_to_num(resid_I[d], nan=0.0)
            # R-IVETO-POLICY-RATIFIED (2026-08-31) as shipped under
            # R-ROUND2-RATIFIED (2026-09-03): front gates = sigma_stack
            # (applied inside run_identification_i, k=SIGMA_THRESHOLD)
            # AND mstruct on the I master; morphology veto RETIRED to
            # annotation; the 2-D dynamic curve applies POST-LOOP (flood
            # keying needs every night's post-front count first).
            passed = run_identification_i(rI, master_I, hdr, beam, sstack_I)
            front = []
            for c in passed:
                _s = abs(float(c.get('local_snr', c.get('snr', 0))))
                _mi = float(ms_destr_I[int(c['pix_y']), int(c['pix_x'])])
                c['mstruct_i'] = round(_mi, 3)
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
            _drI = sv2.get_uniform_rms(rI, sv2.make_masks(
                hdr, rI.shape)[0])
            i_pending.append((d, front, hdr, ot, float(_drI)))
            gimg_rows.append([lst, scope, 'I', d, len(passed), len(front),
                              '', '', ''])
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
            if not os.path.exists(vr_path):
                _h, _w = rV.shape
                _half = min(512, _h // 4, _w // 4)
                _inner = rV[_h//2-_half:_h//2+_half, _w//2-_half:_w//2+_half]
                from astropy.stats import mad_std as _mads
                _hdr_out = hdr.copy()
                _hdr_out['THERMNOI'] = (float(round(float(_mads(_inner, ignore_nan=True)), 9)),
                                        'inner rms of this V residual (Jy/beam)')
                # atomic write: a crash mid-write must never leave a partial
                # file that the exists-check above would trust on rerun
                _tmp = vr_path + '.part'
                fits.writeto(_tmp, rV.astype(np.float32), _hdr_out, overwrite=True)
                os.replace(_tmp, vr_path)
            # sstack is INFO-ONLY (quoted on survivors, lines below), NOT a gate
            # (2026-06-23 ruling: mstruct is the sole V gate). Passing None here keeps
            # run_identification_v from filtering on sigma_stack.
            # WRITE FLOOR = 5sigma (Gregg 2026-06-30): retain >=5sigma in the CSV for
            # repeaters/cross-match; the mstruct gate (below) still vetoes structure.
            pV = run_identification_v(rV, master_I, hdr, beam, None, sigma_thresh=5.0)
            # 'outlier QA' tag (Gregg 2026-08-26): candidates from coadd-omitted
            # epochs carry the flag through catalogs to transient review
            if d in qa_out:
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
            keptv, vetoed_v = list(pV), []
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
                jup = _body_skycoord('jupiter', ot, d) if jrad else None
                moon = _body_skycoord('moon', ot, d)
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
                              len(keptv)])
            # R-ROUND2-RATIFIED: survivors are NOT final until the frozen
            # v110 gate rules on them — stash for the per-cell verdict
            # stage below (one gate invocation per cell; the archived
            # vr_path frame is what the gate measures on).
            if keptv:
                v_pending.append((d, keptv, vr_path, hdr, ot,
                                  1 if gate_ran else '', ))
    # ── I 2-D DYNAMIC CURVE STAGE (ratified R-IVETO-FLOOD-CURVE: R25 1.0
    # nominal / 1.4 flood, gamma 0.35, active S>=25 where S = parent peak
    # flux over the night's uniform residual rms; flood = a night whose
    # post-front count exceeds 3x the cell median of post-front counts).
    if i_pending:
        _icnt = {d_: len(fr) for d_, fr, _h, _o, _r in i_pending}
        _isv = sorted(_icnt.values())
        _icmed = max(_isv[len(_isv) // 2], 1)
        from scipy.spatial import cKDTree as _KD
        _pa = np.array(_parents) if _parents else np.zeros((0, 3))

        def _uv(ra_, de_):
            r_, d_ = np.radians(ra_), np.radians(de_)
            return np.stack([np.cos(d_) * np.cos(r_),
                             np.cos(d_) * np.sin(r_), np.sin(d_)], -1)
        _ptree = _KD(_uv(_pa[:, 0], _pa[:, 1])) if len(_pa) else None
        for d_, front, hdr_, ot_, drI_ in i_pending:
            _flood = _icnt[d_] > 3 * _icmed
            _r25 = 1.4 if _flood else 1.0
            # Ephemeris always-on for I (ratified; omission caught by Gregg
            # 2026-09-04 - Jupiter at 80 sigma on the Blue I pages):
            # Jupiter per band radius + Moon 15 arcmin, per-epoch positions.
            _wI = WCS(hdr_).celestial
            _b = {'18':'Red','23':'Red','27':'Red','32':'Red','36':'Red',
                  '41':'Green','46':'Green','50':'Green','55':'Green',
                  '59':'Green','64':'Blue','69':'Blue','73':'Blue',
                  '78':'Blue','82':'Blue'}.get(scope.replace('MHz',''), scope)
            _jrad = JUPITER_VETO_DEG.get(_b, 0.5)
            _jup = _body_skycoord('jupiter', ot_, d_)
            _moon = _body_skycoord('moon', ot_, d_)
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
                cut = False
                if _ptree is not None:
                    _dd, _ii = _ptree.query(_uv(
                        np.array([c['coord'].ra.deg]),
                        np.array([c['coord'].dec.deg])), k=1)
                    _sep_bw = (2 * np.arcsin(_dd[0] / 2) * 180 / np.pi) \
                        / (em * cd)
                    _S = _pa[_ii[0], 2] / max(drI_, 1e-12)
                    c['parent_jy'] = round(float(_pa[_ii[0], 2]), 3)
                    c['parent_off_beams'] = round(float(_sep_bw), 3)
                    if _S >= 25.0 and _sep_bw < _r25 * (_S / 25.0) ** 0.35:
                        cut = True
                if cut:
                    gled_rows.append(_gled_row(lst, scope, 'I', d_, c, 1,
                                               reason='curve'
                                               + ('-flood' if _flood
                                                  else ''), pass_ms=1))
                    continue
                gled_rows.append(_gled_row(lst, scope, 'I', d_, c, 1,
                                           pass_ms=1, final=1))
                finals.append(c)
            if finals and not GATE_LEDGER_ONLY:
                pc.write_candidates(root, label, finals, resid_I[d_],
                                    master_I, hdr_, lst, d_, scope, 'I',
                                    'deep', em, cd, ot_)
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
    _v110_removes = ({'FP'} if os.environ.get('P3_V110_READING') == 'strict'
                     else {'FP', 'marginal'})
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
            gled_rows.append(_gled_row(lst, scope, 'V', d, c, 1,
                                       pass_ms=_pm, jup_veto=0 if _pm else '',
                                       final=1))
            finals.append(c)
        if finals and not GATE_LEDGER_ONLY:
            pc.write_candidates(root, label, finals, resid_V[d], master_I,
                                hdr, lst, d, scope, 'V', 'deep', em, cd, ot)
            nwr += len(finals)
    _append_locked(f'{run_root}/v_deep_ledger.csv', GLED_COLS, gled_rows)
    _append_locked(f'{run_root}/v_deep_images.csv', GIMG_COLS, gimg_rows)
    print(f'  {lst} {scope}: {len(dates)} epochs -> {nwr} candidates  '
          f'({len(gled_rows)} gate-ledger rows, {len(gimg_rows)} image rows)  '
          f'({(time.time()-t0)/60:.1f} min)', flush=True)
    return nwr


if __name__ == '__main__':
    lst, scope = _ARGV[1], _ARGV[2]
    label = _ARGV[3] if len(_ARGV) > 3 else 'PROD'
    stokes = _ARGV[4] if len(_ARGV) > 4 else 'IV'
    run_deep_cell(lst, scope, label, stokes=stokes)
