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
STRIATION_WINDOW_BEAMS = 20.0
JUPITER_VETO_DEG = {'Red': 5.0, 'Green': 0.5, 'Blue': 0.5, 'Full': 5.0}
# A-team veto stage (Gregg ADOPTED 2026-07-15, pilot-measured: handoff §5.15g):
# 2.0 deg all six sources, applied POST-mstruct; replaces the pre-detection
# blank (which stays default-ON until a tier boundary — see P3_ATEAM_MASK_OFF).
ATEAM_VETO_DEG = 2.0
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
             'jupiter_deg', 'jupiter_veto', 'final', 'n_v_epoch']
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
            int(final), c.get('n_v_epoch', '')]


def _master_structure_map(vm, bpix, win_beams=4.0):
    """Local rms of the V master normalized to clean-sky (>=1). Inlined from
    combined_transient_list (which imports phase3_deep, so can't import back)."""
    if not np.isfinite(bpix) or bpix <= 0:   # empirical-beam fit can return NaN (13h/14h Blue/Full)
        bpix = 7.0
    valid = np.isfinite(vm) & (vm != 0); w = max(int(win_beams * bpix), 5)
    vf = np.where(valid, vm, 0.0); m1 = uniform_filter(vf, w); m2 = uniform_filter(vf * vf, w)
    locrms = np.sqrt(np.maximum(m2 - m1 * m1, 0)); n = locrms.shape[0]; q = n // 4
    cen = locrms[q:3 * q, q:3 * q]; sky = np.median(cen[cen > 0]) if np.any(cen > 0) else 1.0
    m = locrms / max(sky, 1e-9); m[~valid] = 1.0; return np.maximum(m, 1.0)


def _mstruct_maps(master_V, mVp, beam):
    """(destriped, raw) mstruct from the deep V master — Gregg's validated gate.
    Falls back to raw if destriping fails (raw alone already separates sidelobes from
    real transients in validation)."""
    if not np.isfinite(beam) or beam <= 0:
        beam = 7.0
    raw = _master_structure_map(master_V, beam)
    try:
        vh = fits.getheader(mVp); mwcs = WCS(vh).celestial
        _hm, _am, iwm, _ot, _ac, _el = sv2.make_masks(vh, master_V.shape, fits_path=mVp)
        sm, _ = build_striation_model(master_V, mwcs, iwm,
                                      ra_window_beams=STRIATION_WINDOW_BEAMS, beam_pix=beam)
        return _master_structure_map(master_V - sm, beam), raw
    except Exception as e:
        print(f'    [mstruct] destripe failed ({e}); using raw mstruct')
        return raw, raw


def _qa_outlier_dates(lst, scope):
    """Dates the two-stage QA flagged outlier==True — must NOT be searched (problem 9)."""
    p = f'{COADD_ROOT}/{lst}_{scope}/residuals_provisional.csv'
    if not os.path.exists(p):
        return set()
    return {r['date'] for r in csv.DictReader(open(p))
            if str(r.get('outlier')).strip().lower() == 'true'}


def _jupiter_skycoord(date):
    """Apparent ra/dec of Jupiter (strip distance -> avoids the separation(get_body)
    barycentric-parallax inflation)."""
    jb = get_body('jupiter', Time(date + ' 00:00:00', scale='utc'), OVRO_LOC)
    return SkyCoord(jb.ra, jb.dec)

# GC-PRE-3 (Gregg 2026-08-12): flow-align script is the repo's own copy
# (exopipe/phase2/align_images.py, byte-identical to the frozen clv15
# original) — never execute frozen-tree code from the production path.
# Shelled into py310_orca (JAX), never imported.
ALIGN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'phase2', 'align_images.py')
WORKROOT = os.environ.get('P3_FLOW_ROOT', '/data07/gh/phase3/deep')   # moved off /fast (NVMe too small; was filling it)
COADD_ROOT = os.environ.get('P3_COADD_ROOT', '/fast/gh/main/phase3/coadd')
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
        r = subprocess.run(['conda', 'run', '-n', 'py310_orca', 'python', ALIGN,
                            '--pairs', pp, '--no_qa', '--save_flow'],
                           capture_output=True, text=True, env=env)
        if r.returncode not in (0, 2):
            print(f'    flow batch rc={r.returncode}: {r.stderr[-400:]}', flush=True)
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
    smodel, _ = build_striation_model(np.nan_to_num(up - master), w, invalid,
                                      ra_window_beams=STRIATION_WINDOW_DEG / cd,
                                      beam_pix=1.0)
    return np.where(invalid, np.nan, (up - master) - smodel)


_RESID_CTX = {}


def _resid_task(d):
    c = _RESID_CTX
    try:
        return d, _residual(c['epoch'][d], c['master'], c['flows'][d], c['hdr'], c['cd'])
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


def run_deep_cell(lst, scope, label='PROD', included_dates=None):
    t0 = time.time()
    base = f'{COADD_ROOT}/{lst}_{scope}'
    _irob = 'R0' if scope in ('Red', 'Green', 'Blue', 'Full') else 'R075'  # subbands co-add I at Robust+0.75
    mI = f'{base}/coadd_final/I_deep_Taper_{_irob}_{scope}.fits'
    if not os.path.exists(mI):
        print(f'  {lst} {scope}: no deep I master ({_irob}) — skip'); return 0
    master_I = _load(mI); hdr0 = fits.getheader(mI); cd = abs(hdr0['CDELT1'])
    em, en, _, _ = fit_empirical_beam(master_I)
    if not np.isfinite(em) or em <= 0:   # beam fit can fail (e.g. 14h Blue) -> NaN propagates into headers
        print(f'  {lst} {scope}: empirical-beam fit returned NaN; using fallback beam=7px')
        em = 7.0; en = 7.0
    beam = em
    ar_max = max(2.0, 1.5 * em / max(en, 1e-3))
    mVp = f'{base}/coadd_final/V_deep_Taper_R0_{scope}.fits'
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
    # (1) drop two-stage-QA outlier epochs — the search must honor the same QA that
    # gates the co-add (a 79sigma-rms peeling-blowup epoch otherwise floods the list)
    qa_out = _qa_outlier_dates(lst, scope)
    _dropped = [d for d in dates if d in qa_out]
    if _dropped:
        dates = [d for d in dates if d not in qa_out]
        print(f'  {lst} {scope}: dropped {len(_dropped)} QA-outlier epoch(s) from search: {_dropped}')
    if len(dates) < MIN_EPOCHS:
        print(f'  {lst} {scope}: only {len(dates)} flows — skip'); return 0

    hdr_ref = _det_header(hdr0, scope, em, cd, None)
    resid_I = _build_residuals_mp(epoch_I, master_I, flows, hdr_ref, cd, dates, NPROC_RESID)
    dates = [d for d in dates if d in resid_I]
    sstack_I = _sigma_stack(resid_I, cell_work, hdr0, 'I')
    resid_V = {}
    if master_V is not None:
        epoch_V = {}
        for d in dates:
            vp = f'{base}/subtracted/{d}_V_sub_{scope}.fits'
            if os.path.exists(vp):
                epoch_V[d] = _load(vp) + master_V
        if epoch_V:
            vd = list(epoch_V)
            resid_V = _build_residuals_mp(epoch_V, master_V,
                                          {d: flows[d] for d in vd},
                                          hdr_ref, cd, vd, NPROC_RESID)
    sstack_V = _sigma_stack(resid_V, cell_work, hdr0, 'V') if resid_V else None

    root = pc.P.STAGING_ROOT; nwr = 0
    run_root = pc.P.transients_run(root, label, make=True)
    gled_rows, gimg_rows = [], []
    for d in dates:
        ot = _epoch_time(base, d, scope)
        hdr = _det_header(hdr0, scope, em, cd, ot); wcs = WCS(hdr).celestial
        rI = np.nan_to_num(resid_I[d], nan=0.0)
        passed = run_identification_i(rI, master_I, hdr, beam, sstack_I)
        kept, vetoed_i = pv.apply_veto(passed, rI, master_I, hdr, em, cd, wcs, ot, ar_max=ar_max)
        for c in kept:
            gled_rows.append(_gled_row(lst, scope, 'I', d, c, 1, final=1))
        for c in vetoed_i:
            gled_rows.append(_gled_row(lst, scope, 'I', d, c, 0,
                                       reason=c.get('veto_reason', '')))
        gimg_rows.append([lst, scope, 'I', d, len(passed), len(kept),
                          '', '', len(kept)])
        if kept and not GATE_LEDGER_ONLY:
            pc.write_candidates(root, label, kept, rI, master_I, hdr, lst, d,
                                scope, 'I', 'deep', em, cd, ot)
            nwr += len(kept)
        if d in resid_V:
            rV = np.nan_to_num(resid_V[d], nan=0.0)
            # sstack is INFO-ONLY (quoted on survivors, lines below), NOT a gate
            # (2026-06-23 ruling: mstruct is the sole V gate). Passing None here keeps
            # run_identification_v from filtering on sigma_stack.
            # WRITE FLOOR = 5sigma (Gregg 2026-06-30): retain >=5sigma in the CSV for
            # repeaters/cross-match; the mstruct gate (below) still vetoes structure.
            pV = run_identification_v(rV, master_I, hdr, beam, None, sigma_thresh=5.0)
            keptv, vetoed_v = pv.apply_veto(pV, rV, master_I, hdr, em, cd, wcs, ot, ar_max=ar_max)
            for c in vetoed_v:
                gled_rows.append(_gled_row(lst, scope, 'V', d, c, 0,
                                           reason=c.get('veto_reason', '')))
            n_after_morph = len(keptv)
            vetoed_ms, vetoed_jup = [], []
            # (2) mstruct gate + (3) Jupiter veto, quoting mstruct/sstack on survivors
            if keptv and ms_destr is not None:
                # subband scopes map to their band's Jupiter radius
                _b = {'18':'Red','23':'Red','27':'Red','32':'Red','36':'Red',
                      '41':'Green','46':'Green','50':'Green','55':'Green',
                      '59':'Green','64':'Blue','69':'Blue','73':'Blue',
                      '78':'Blue','82':'Blue'}.get(
                          scope.replace('MHz',''), scope)
                jrad = JUPITER_VETO_DEG.get(_b)
                jup = _jupiter_skycoord(d) if jrad else None
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
                    adeg = float(min(wcs.pixel_to_world(xx, yy)
                                     .separation(_ATEAM_COORDS).deg))
                    c['nearest_ateam'] = round(adeg, 2)
                    if adeg < ATEAM_VETO_DEG:                     # A-team 2 deg (adopted 2026-07-15)
                        c['veto_reason'] = 'ateam'
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
            for c in keptv:
                gled_rows.append(_gled_row(lst, scope, 'V', d, c, 1,
                                           pass_ms=1 if gate_ran else '',
                                           jup_veto=0 if gate_ran else '',
                                           final=1))
            n_after_ms = n_after_morph - len(vetoed_ms)
            gimg_rows.append([lst, scope, 'V', d, len(pV), n_after_morph,
                              n_after_ms if gate_ran else '',
                              (n_after_ms - len(vetoed_jup)) if gate_ran else '',
                              len(keptv)])
            if keptv and not GATE_LEDGER_ONLY:
                pc.write_candidates(root, label, keptv, rV, master_I, hdr, lst, d,
                                    scope, 'V', 'deep', em, cd, ot)
                nwr += len(keptv)
    _append_locked(f'{run_root}/v_deep_ledger.csv', GLED_COLS, gled_rows)
    _append_locked(f'{run_root}/v_deep_images.csv', GIMG_COLS, gimg_rows)
    print(f'  {lst} {scope}: {len(dates)} epochs -> {nwr} candidates  '
          f'({len(gled_rows)} gate-ledger rows, {len(gimg_rows)} image rows)  '
          f'({(time.time()-t0)/60:.1f} min)', flush=True)
    return nwr


if __name__ == '__main__':
    lst, scope = _ARGV[1], _ARGV[2]
    label = _ARGV[3] if len(_ARGV) > 3 else 'PROD'
    run_deep_cell(lst, scope, label)
