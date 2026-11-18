"""Stages `hybrid` and `full` of the hybrid route (spec plan/specs/hybrid.md 2.5, 2.6): the hybrid master = the Taper master
(striation-restored where the selection lists the unit) + G (x) (reconstruction - Taper), its beam-corrected sibling, its CLEAN
residual, the PB maps; the Full hybrid from the 15 subband hybrids. A close port of production exopipe/phase3/notaper_largescale.py
(NL) at exopipe main 4c7098c: taper_base 1114, hybrid_span 1140, elevation_jump 1186, test_hybrid 1198, small_scale_rel 1232,
test_hybrid_pbcorr 1241, test_readback 1253, _write_product 1265, hybrid_unit 1274, _subband_thermnoi 1390, hybrid_full 1404,
stack_refused 1505, diffuse_pb_map 1517, full_pb_map 1530. The differences are those of lib/hybrid/recon.py (config constants
through bind(), ReconRefusal, memmap=False, records through a temporary name).
"""
import csv
import glob
import json
import os
import re
import time

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import fft as sfft

from exopipe.phase3.core.products import build_name, parse_name, write_standard_header
from exopipe.phase3.lib.hybrid import recon as REC
from exopipe.phase3.lib.hybrid.common import TAPER_OF, VARIANT, diag_header, log, lowpass, rd
from exopipe.phase3.lib.hybrid.common import sky_interior as _sky_interior
from exopipe.phase3.lib.hybrid.recon import (ReconRefusal, _dump, _level_verdict, beam_image, conv_kernel, installed_master,
                                             lscaveat_cards, smooth, test_compact_sources)
from exopipe.phase3.lib.hybrid.stack import constituent_span, members_of
from exopipe.phase3.lib.hybrid.wideband import WCS_KEYS
from exopipe.phase3.lib.install.steps_common import md5
from exopipe.phase3.lib.masters import coadd as CFP
from exopipe.phase3.lib.masters import pb_apply as PBA


def bind():
    """Bind the module's constants from the config (recon.bind() plus the hybrid stage's own; production NL 62-80, 944-949,
    1109-1111, 1681). Call before any computation."""
    from exopipe.phase3.config import CFG
    g = dict(REC.bind())
    H = CFG.hybrid
    g.update(HYB_U_IN=float(H.HYB_U_IN), HYB_U_OUT=float(H.HYB_U_OUT), STRIATION_SELECTION=H.STRIATION_SELECTION,
             STRIATION_RESTORED=H.STRIATION_RESTORED, STRIATION_SCOPES=tuple(H.STRIATION_SCOPES),
             WIDEBANDS=dict(H.WIDEBAND_MEMBERS, Full=tuple(CFG.hybrid_full.MEMBERS)), COADD_SB=CFG.hybrid_stack.COADD_SB,
             RAW_NIGHT_CAVEAT=dict(CFG.hybrid_unit.RAW_NIGHT_CAVEAT), OWN_BEAM_MAX=float(CFG.hybrid_unit.OWN_BEAM_MAX),
             IDENTITY=float(CFG.hybrid_unit.IDENTITY), LARGE_REL=float(CFG.hybrid_unit.LARGE_REL),
             STRIATION_IDENTITY=float(CFG.hybrid_unit.STRIATION_IDENTITY))
    globals().update(g)
    return g


def taper_base(lst, scope, v):
    """The hybrid's Taper part: (master, beam-corrected sibling, striation model or None). A selected cell's restored master must carry
    the installed master's name apart from the restore tokens and give back the installed master with its striation model added
    (max |restored + model - installed| < 1e-6 of the installed peak: made from THIS master), else REFUSE."""
    tpath, tppath = installed_master(lst, scope, v, True)
    if scope not in STRIATION_SCOPES or (lst, scope, f'Taper_{v}') not in {
            (r['lst'], r['band'], r['variant']) for r in csv.DictReader(open(STRIATION_SELECTION))}:
        return tpath, tppath, None
    d = f'{STRIATION_RESTORED}/{lst}_{scope}'; tok = 'Robust-0.75' if v == 'R075' else 'Robust+0.0'
    g = glob.glob(f'{d}/{scope}_I_deep_Taper_{tok}_*ssub-rst8_LST{lst}_*.fits')
    sm = [f for f in g if f.endswith('_striationmodel.fits')]; g = [f for f in g if not f.endswith('_striationmodel.fits')]
    a = [f for f in g if '_pbcorr_' not in f]; p = [f for f in g if '_pbcorr_' in f]
    if len(a) != 1 or len(p) != 1 or len(sm) != 1:
        raise ReconRefusal(f'{lst} {scope} {v}: selected for the striation-restored base but {len(a)} restored, {len(p)} beam-corrected, '
                           f'{len(sm)} striation models in {d} - REFUSING')
    strip = lambda n: re.sub(r'_(ssub-rst8|shflux|mdewarp)', '', n)
    if strip(os.path.basename(a[0])) != strip(os.path.basename(tpath)) or fits.getheader(a[0]).get('PRODNAME') != os.path.basename(a[0]):
        raise ReconRefusal(f'{lst} {scope} {v}: {os.path.basename(a[0])} is not the restored {os.path.basename(tpath)} - REFUSING')
    I, R_, S = rd(tpath), rd(a[0]), rd(sm[0]); ok = np.isfinite(I) & np.isfinite(R_) & np.isfinite(S)
    err = float(np.max(np.abs(R_[ok] + S[ok] - I[ok])) / np.max(np.abs(I[ok])))
    if not err < STRIATION_IDENTITY:
        raise ReconRefusal(f'{lst} {scope} {v}: restored master + striation model differs from the installed master by {err:.2e} of its '
                           f'peak - REFUSING (restore again on the current master)')
    return a[0], p[0], sm[0]


def hybrid_span(run_dir, lst, scope, v):
    """Time block of the hybrid = the union of its inputs' nights: the nights the reconstruction stacked (each subband's NoTaper stack
    record) and the Taper master's constituents (the Taper stack record's span_master; Red / Green: the wideband Taper master's
    constituent nights, found in the subbands' Taper records). Per night the earliest start, latest end and longest integration (Phase 2:
    not a sum over frequency); a Taper constituent found in no record refuses (25 of 576 archived Taper / NoTaper pairs differ by 1-2
    nights, 2026-10-28)."""
    from astropy.time import Time
    tv = TAPER_OF[v]; subs = WIDEBANDS.get(scope, (scope,)); nights = {}

    def add(n):
        d = nights.setdefault(n['date'], dict(start=n['start'], end=n['end'], inttime_s=0.0))
        d['start'] = min(d['start'], n['start'], key=lambda t: Time(t).mjd); d['end'] = max(d['end'], n['end'], key=lambda t: Time(t).mjd)
        d['inttime_s'] = max(d['inttime_s'], float(n['inttime_s']))
    have = {}
    for b in subs:
        for n in json.load(open(f'{run_dir}/{lst}_{b}/{v}_stack.json'))['span']['nights']:
            add(n)
        if os.path.exists(f'{run_dir}/{lst}_{b}/{tv}_stack.json'):
            tn = json.load(open(f'{run_dir}/{lst}_{b}/{tv}_stack.json'))['span_master']['nights']
        elif stack_refused(run_dir, lst, b, tv):  # its rebuild test refused (only the CLEAN residual is lost, 2026-10-31): the Taper master's
            tn = constituent_span(lst, b, members_of(COADD_SB, lst, b, tv)[0], VARIANT[tv][0])['nights']   # span from the records the stack reads
        else:
            raise ReconRefusal(f'{lst} {b} {tv}: no Taper stack record and no recorded refusal - REFUSING')
        for n in tn:
            have.setdefault(n['date'], []).append(n)
    want = CFP._constituent_dates(fits.getheader(installed_master(lst, scope, v, True)[0]))
    miss = [d for d in want if d not in have]; from_wb = []
    if miss and scope in WIDEBANDS:               # nights the wideband master holds but no subband master does (04h Red 2024-12-19,
        wrows = {r['date']: r for r in csv.DictReader(open(f'{COADD_SB}/{lst}_{scope}/coadd_final/weights.csv'))}   # 14h Red 2025-04-05,
        for d in list(miss):                      # 11h Green 2025-04-22; 2026-10-30): the span of the wideband night image itself
            p = CFP.find_wideband_deep_images(wrows[d]['sci_dir'], scope, 'I', VARIANT[tv][0]) if d in wrows else None
            h = fits.getheader(p) if p else {}
            if h.get('DATE-OBS') and h.get('DATE-END') and h.get('INTTIME') is not None:
                have.setdefault(d, []).append(dict(date=d, start=str(h['DATE-OBS']), end=str(h['DATE-END']), inttime_s=float(h['INTTIME'])))
                miss.remove(d); from_wb.append(dict(date=d, image=p))
    if miss:
        raise ReconRefusal(f'{lst} {scope} {v}: Taper constituents {miss} found in no Taper stack record - REFUSING')
    for d in want:
        for n in have[d]:
            add(n)
    dates = sorted(nights)
    return dict(start=min((n['start'] for n in nights.values()), key=lambda t: Time(t).mjd),
                end=max((n['end'] for n in nights.values()), key=lambda t: Time(t).mjd), inttime_s=sum(n['inttime_s'] for n in nights.values()),
                nights=[dict(date=d, **nights[d]) for d in dates], nights_from_wideband_images=from_wb)


def elevation_jump(img, hdr, e0=10.0):
    """rms over 24 position-angle sectors (15 deg) of mean(elevation e0 to e0 + 1.5 deg) - mean(e0 - 1.5 to e0), Jy/beam: an edge at
    elevation e0 shows here (Gregg 2026-10-27: the step at 10 deg)."""
    N = img.shape[0]; yy, xx = np.indices((N, N)); l_ = np.radians((xx + 1 - hdr['CRPIX1']) * hdr['CDELT1']); m_ = np.radians((yy + 1 - hdr['CRPIX2']) * hdr['CDELT2'])
    r = np.hypot(l_, m_); el = np.where(r < 1, 90 - np.degrees(np.arcsin(np.clip(r, 0, 1))), -1.0); pa = np.degrees(np.arctan2(l_, m_)); j = []
    for p0 in range(-180, 180, 15):
        s = (pa >= p0) & (pa < p0 + 15) & np.isfinite(img); a_, b_ = img[s & (el >= e0) & (el < e0 + 1.5)], img[s & (el >= e0 - 1.5) & (el < e0)]
        if a_.size > 50 and b_.size > 50:
            j.append(float(np.mean(a_) - np.mean(b_)))
    return float(np.sqrt(np.mean(np.square(j)))) if j else float('nan')


def test_hybrid(path, A, T, F, ok, hdr, lst, scope, installed_nt):
    """The written hybrid, read back: (1) equals Taper + G (x) (reconstruction - Taper) to 1e-5 of the Taper's peak, NaN exactly outside
    the inputs' common finite area; (2) its large scales are the reconstruction's: smoothed to 4 deg, hybrid - reconstruction has an rms
    below 1e-3 of (reconstruction - Taper) smoothed the same way; (3) its small scales are the Taper's: above HYB_U_OUT wavelengths,
    hybrid - Taper has an rms below 5e-3 of its own (a hybrid that kept the reconstruction's small scales scores
    0.5-1); (2) and (3) over the sky interior (l^2 + m^2 < 0.95^2), away from the image border
    and the Taper master's blank pixels beyond the horizon, whose edges put a few 1e-3 of either into the outermost pixels (18h 46 MHz
    Robust -0.75, 2026-10-28: 9e-5 and 1e-3 inside, 2e-3 and 3e-3 over every pixel); (4) its compact sources are the Taper master's
    (test_compact_sources: the 200 brightest, median within 10%, 20% at 18 MHz and Red). The jump across 10 deg elevation (hybrid and
    installed NoTaper master, elevation_jump) is recorded as a diagnostic, not a gate (Gregg 2026-10-28 "approved": at 18 MHz bright-sky
    gradients dominate it - 01h / 03h Robust 0 hybrids with no edge in their profiles read 1.20 / 1.42 x the master - and the recipe has
    no region boundary at 10 deg). In CAVEAT_SCOPES (3) is a recorded caveat, not a refusal (Gregg 2026-10-31; verdict 'caveat')."""
    H = rd(path); cd = abs(float(hdr['CDELT1'])); res = {}
    res['max_rel_err'] = float(np.max(np.abs(H[ok] - (T[ok] + F[ok]))) / np.max(np.abs(T[ok])))
    res['nan_mask_ok'] = bool(np.array_equal(np.isfinite(H), ok))
    inner = _sky_interior(ok, hdr)
    d4 = smooth(np.where(ok, H - A, np.nan), 4.0, cd, ok); r4 = smooth(np.where(ok, A - T, np.nan), 4.0, cd, ok)
    res['large_scale_rel'] = float(np.nanstd(d4[inner]) / np.nanstd(r4[inner]))
    res['small_scale_rel'] = small_scale_rel(H, T, ok, hdr)
    res['test_compact_sources'] = test_compact_sources(H, T, hdr, lst, scope)
    res['jump10_hybrid'] = elevation_jump(H, hdr); res['jump10_installed_notaper'] = elevation_jump(installed_nt, hdr)
    small_ok = res['small_scale_rel'] < HYB_SMALL_LIMIT
    if not (res['max_rel_err'] < IDENTITY and res['nan_mask_ok'] and res['large_scale_rel'] < LARGE_REL and (small_ok or scope in CAVEAT_SCOPES)):
        raise ReconRefusal(f'test_hybrid FAILED: {res} - REFUSING')
    res['verdict'] = 'pass' if small_ok else 'caveat'
    res['small_scale_caveat'] = not small_ok
    return res


def small_scale_rel(H, T, ok, hdr):
    """test_hybrid (3): rms of (hybrid - Taper) above HYB_U_OUT wavelengths / the rms of (hybrid - Taper), over the sky interior."""
    N = H.shape[0]; cd = abs(float(hdr['CDELT1'])); inner = _sky_interior(ok, hdr)
    Pz = np.zeros((2 * N, 2 * N)); Pz[:N, :N] = np.where(ok, H - T, 0.0)
    u = np.hypot(np.fft.fftfreq(2 * N)[:, None], np.fft.rfftfreq(2 * N)[None, :]) / np.radians(cd)
    hp = sfft.irfft2(sfft.rfft2(Pz, workers=FFTW) * (u > HYB_U_OUT), s=Pz.shape, workers=FFTW)[:N, :N]
    return float(np.std(hp[inner]) / np.std((H - T)[inner]))


def test_hybrid_pbcorr(path, F, pbd, tp_path):
    """The beam-corrected hybrid read back, minus the added large scales / the diffuse beam, must equal the archived Taper beam-corrected
    master to 1e-5 of its largest value, NaN exactly where that master is NaN (the same full beam and floor)."""
    P = rd(path); Tp = rd(tp_path); both = np.isfinite(P) & np.isfinite(Tp)
    res = dict(max_rel_err=float(np.max(np.abs(P[both] - F[both] / pbd[both] - Tp[both])) / float(np.max(np.abs(Tp[both])))),
               nan_masks_equal=bool(np.array_equal(np.isfinite(P), np.isfinite(Tp))), valid_px=int(both.sum()))
    if not (res['max_rel_err'] < IDENTITY and res['nan_masks_equal']):
        raise ReconRefusal(f'test_hybrid_pbcorr FAILED: {res} - REFUSING')
    res['verdict'] = 'pass'
    return res


def test_readback(path, ref, name):
    """A written array product read back equals the array computed (to 1e-5 of its largest value), NaN where it is NaN, and carries its
    PRODNAME."""
    b = rd(path); f = np.isfinite(ref)
    res = dict(max_rel_err=float(np.max(np.abs(b[f] - ref[f])) / float(np.max(np.abs(ref[f])))), nan_masks_equal=bool(np.array_equal(np.isfinite(b), f)),
               prodname_ok=fits.getheader(path).get('PRODNAME') == name)
    if not (res['max_rel_err'] < IDENTITY and res['nan_masks_equal'] and res['prodname_ok']):
        raise ReconRefusal(f'test_readback FAILED ({path}): {res} - REFUSING')
    res['verdict'] = 'pass'
    return res


def _write_product(path, data, h, name, scope, span, beam, pbcorr, gen):
    fits.writeto(path, data.astype(np.float32), h, overwrite=True)
    write_standard_header(path, scope=scope, stokes='I', start=span['start'], end=span['end'], inttime_s=span['inttime_s'], time_basis='coadd-span',
                          beam=beam, imgrms=None if pbcorr else CFP.get_uniform_rms(data, ~np.isfinite(data) | (data == 0)), pbcorr=pbcorr,
                          dewarped=True, aligned=True, msub=False, ssub=False,
                          extra={'PRODNAME': (name, 'Phase 3 product name (grammar; CONTINUE cards)'), 'MSTRGEN': (gen, 'master generation of this product')},
                          validate=True)


def hybrid_unit(lst, scope, v, run_dir):
    """Hybrid NoTaper master of one cell and weighting (Gregg 2026-10-27 proposal; preamble approved 2026-10-28): the installed Taper
    master (WSClean visibility-domain deconvolution: compact sources and small scales) + G (x) (reconstruction - Taper master), G = 1 below
    30 wavelengths, a cosine to 0 at 60 (the reconstruction's large scales). Named as the Taper master + 'lsrecon' + 'lsfill'; time block
    the union of the inputs' nights (hybrid_span). Beside it: the beam-corrected sibling (Taper part / full beam, as the archived Taper
    beam-corrected master; added large scales / diffuse beam) and the hybrid CLEAN residual (token cleanres; Gregg 2026-10-28 d): for a
    subband the Taper residual stack + G (x) (reconstruction final residual - Taper residual stack), for Red / Green the subband hybrid
    residuals combined with the wideband reconstruction's weights."""
    bind()                                        # v2: config constants
    if scope == 'Full':
        return hybrid_full(lst, v, run_dir)
    wd = f'{run_dir}/{lst}_{scope}'; wb = scope in WIDEBANDS
    rrec = json.load(open(f'{wd}/{v}_lsrecon_wb.json' if wb else f'{wd}/{v}_lsrecon.json'))
    A = rd(rrec['product']['path']); rh = fits.getheader(rrec['product']['path'])
    tpath, tppath, smodel = taper_base(lst, scope, v); T = rd(tpath); th = fits.getheader(tpath)   # striation-restored where ruled (2026-10-31)
    for k in WCS_KEYS:
        if rh[k] != th[k]:
            raise ReconRefusal(f'{lst} {scope} {v}: reconstruction and Taper master grids differ in {k} - REFUSING')
    cd = abs(float(th['CDELT1'])); N = A.shape[0]
    # the reconstruction re-restored with the Taper master's beam (the hybrid's declared beam) before the blend: final residual + model
    # (x) Taper beam. With its own NoTaper beam the 18 MHz Robust 0 hybrids carried ~18% more compact-source flux than their Taper
    # masters (beam areas 35.0x25.7 vs 32.3x23.5 arcmin; 07h: compact median 1.199 -> 1.022 re-restored), 2026-10-28.
    subs = [(s['band'], float(s['weight'])) for s in rrec['subbands']] if wb else [(scope, 1.0)]
    Rf = sum(w * rd(f'{run_dir}/{lst}_{b}/{v}_lsrecon_final_residual.fits') for b, w in subs)
    Mo = [(w, np.nan_to_num(rd(f'{run_dir}/{lst}_{b}/{v}_lsrecon_model.fits')), fits.getheader(f'{run_dir}/{lst}_{b}/{v}_lsrecon_model.fits')) for b, w in subs]
    back = Rf + sum(w * conv_kernel(m, beam_image(mh_, (N, N)), N // 2, N // 2) for w, m, mh_ in Mo)   # its own beams: must give back A
    rerr = float(np.nanmax(np.abs(back - A)) / np.nanmax(np.abs(A)))
    if not rerr < OWN_BEAM_MAX:
        raise ReconRefusal(f'{lst} {scope} {v}: final residual + model (x) beam does not give back the reconstruction ({rerr:.2e}) - REFUSING')
    A = Rf + conv_kernel(sum(w * m for w, m, _ in Mo), beam_image(th, (N, N)), N // 2, N // 2)
    ok = np.isfinite(A) & np.isfinite(T)
    F = lowpass(np.where(ok, A - T, 0.0), cd); F[~ok] = np.nan; H = T + F
    cards = []
    if scope in CAVEAT_SCOPES:                    # the recorded caveats (Gregg 2026-10-31): the reconstruction's level, the hybrid's small
        cards = lscaveat_cards(rrec['test_large_scale_level'], small_scale_rel(H.astype(np.float32).astype(np.float64), T, ok, th))   # scales as read back
    span = hybrid_span(run_dir, lst, scope, v); dates = [n['date'] for n in span['nights']]
    pn = parse_name(os.path.basename(tpath)); pn['proc'] = list(pn['proc']) + ['lsrecon', 'lsfill']
    pn.update(date_start=dates[0].replace('-', ''), date_end=dates[-1].replace('-', ''), n_epochs=len(dates)); name = build_name(**pn)
    h = th.copy(); hist = [str(x) for x in h.get('HISTORY', []) if not str(x).startswith('COADD constituent')]
    h.remove('HISTORY', ignore_missing=True, remove_all=True)
    for x in hist:
        h['HISTORY'] = x
    for d in dates:
        h['HISTORY'] = f'COADD constituent {d}'
    h['NCOADD'] = (len(dates), 'distinct nights: Taper master and reconstruction'); h['LSRECON'] = (True, 'large scales from the lsrecon reconstruction')
    h['LSFILL'] = (True, 'hybrid: Taper master + G (x) (reconstruction - Taper)'); h['LSFILLU'] = (f'{HYB_U_IN:g}-{HYB_U_OUT:g}', 'G: 1 below, cosine to 0 at (wavelengths)')
    h['LSRGEN'] = (RECON_GEN, 'reconstruction: CLEAN, step and prior to the true horizon')
    for k, val, com in cards:
        h[k] = (val, com)
    raw_b = [b for b, _ in subs if (lst, b) in RAW_NIGHT_CAVEAT]
    if raw_b:                                     # Gregg 2026-11-04: kept with undewarped nights in the reconstruction stacks, a recorded caveat
        h['LSRAWNT'] = (','.join(raw_b), 'undewarped nights in recon stacks (caveat)')
    adm_b = [b for b, _ in subs if json.load(open(f'{run_dir}/{lst}_{b}/{v}_lsrecon.json')).get('stack_admission')]
    if adm_b:                                     # Gregg 2026-10-31 "Admit them as hybrids": the (subband) stacks admitted at hybrid scales
        h['LSSTKADM'] = (','.join(adm_b), 'stack admitted at hybrid scales (Gregg 10-31)')
    h['HISTORY'] = f'hybrid {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}: Taper master {os.path.basename(tpath)}'
    h['HISTORY'] = f'hybrid large scales from {rrec["product"]["prodname"]}'
    beam = (float(th['BMAJ']), float(th['BMIN']), float(th.get('BPA', 0.0))); gen = 'hybrid-' + RECON_GEN
    path = f'{wd}/I_deep_Taper_{v}_lsrecon_lsfill_{scope}.fits'; _write_product(path, H, h, name, scope, span, beam, False, gen)
    nt = installed_master(lst, scope, v, False)[0]
    rec = dict(lst=lst, scope=scope, weighting=v, taper_master=tpath, striation_model=smodel, reconstruction=rrec['product']['path'], span=span, filter_wavelengths=[HYB_U_IN, HYB_U_OUT],
               stacks_admitted_at_hybrid_scales=adm_b, raw_night_caveat={b: RAW_NIGHT_CAVEAT[(lst, b)] for b in raw_b},
               reconstruction_restoring=dict(method='final residual + model (x) Taper master beam', own_beam_identity_rel_err=rerr))
    rec['test_hybrid'] = test_hybrid(path, A, T, F, ok, th, lst, scope, rd(nt))
    if cards and f"{float(fits.getheader(path).get('LSSMALL', float('nan'))):.4g}" != f"{rec['test_hybrid']['small_scale_rel']:.4g}":   # at the
        # card's 4 significant digits (a FITS card's float comes back e.g. as 0.000993699999999999 for 0.0009937: exact equality refused 3 units)
        raise ReconRefusal(f'{path}: LSSMALL {fits.getheader(path).get("LSSMALL")} is not the tested {rec["test_hybrid"]["small_scale_rel"]} - REFUSING')
    rec['caveats'] = dict(level=rrec['test_large_scale_level'], small_scale_rel=rec['test_hybrid']['small_scale_rel'],
                          small_scale_caveat=rec['test_hybrid']['small_scale_caveat']) if cards else None
    rec['product'] = dict(path=path, prodname=name, md5=md5(path))
    pbf = rd(f'{ARCHIVE_MASTERS}/{lst}/{scope}/{PBA.deep_pb_name(scope, v)}'); pbd = diffuse_pb_map(run_dir, lst, scope, v)[0]
    # blanking = the archived Taper beam-corrected master's: production set the PB floor in float64 before the PB map was stored as
    # float32, so the stored map cannot reproduce it at the floor (00h 36 MHz Robust -0.75: one pixel at float32 0.1, 2026-10-28)
    valid = ok & np.isfinite(rd(tppath)) & np.isfinite(pbf) & (pbf > 0) & np.isfinite(pbd) & (pbd > 0)
    P = np.full(H.shape, np.nan); P[valid] = T[valid] / pbf[valid] + F[valid] / pbd[valid]
    pp = parse_name(name); pp['proc'] = list(pp['proc']) + ['pbcorr']; pname = build_name(**pp); ppath = f'{wd}/I_deep_Taper_{v}_lsrecon_lsfill_pbcorr_{scope}.fits'
    hp = fits.getheader(path).copy(); hp['PBFLOOR'] = (PBA.PB_FLOOR, 'PB floor below which pixels set NaN')
    hp['PBMETHOD'] = ('hybrid-split', 'Taper part / full PB; added large scales / diffuse PB'); hp['PBMAP'] = (PBA.deep_pb_name(scope, v), 'full PB map (archive)')
    _write_product(ppath, P, hp, pname, scope, span, beam, True, gen)
    rec['product_pbcorr'] = dict(path=ppath, prodname=pname, md5=md5(ppath), test_hybrid_pbcorr=test_hybrid_pbcorr(ppath, F, pbd, tppath))
    # a REFUSED Taper residual stack (its rebuild test; recorded by the stack fleet) costs only the CLEAN residual product (Gregg
    # 2026-10-31, "ok on caveats"): no cleanres, recorded; a wideband has one only when all its subbands have theirs
    R = None; rec['cleanres_skipped'] = None
    if wb:
        subs = [(s['band'], s['weight']) for s in rrec['subbands']]
        sh = {b: json.load(open(f'{run_dir}/{lst}_{b}/{v}_hybrid.json')) for b, _ in subs}
        nores = [b for b in sh if not sh[b].get('product_cleanres')]
        if nores:
            rec['cleanres_skipped'] = f'no CLEAN residual of subband(s) {",".join(nores)} (Taper residual stack refused)'
        else:
            R = sum(w * rd(sh[b]['product_cleanres']['path']) for b, w in subs)
            how = 'subband hybrid residuals combined with the wideband reconstruction weights'
    elif not os.path.exists(f'{wd}/{TAPER_OF[v]}_stack.json'):
        why = stack_refused(run_dir, lst, scope, TAPER_OF[v])
        if not why:
            raise ReconRefusal(f'{lst} {scope} {v}: no {TAPER_OF[v]} Taper residual stack and no recorded refusal - REFUSING')
        rec['cleanres_skipped'] = f'{TAPER_OF[v]} Taper residual stack refused: {why[:300]}'
    else:
        Tr = rd(f'{wd}/{TAPER_OF[v]}_residual_stack.fits'); Rf = rd(f'{wd}/{v}_lsrecon_final_residual.fits'); okr = np.isfinite(Tr) & np.isfinite(Rf)
        if smodel:   # the striations live in the residual: the restored Taper part's residual = residual stack - the same striation model
            Sm = rd(smodel)
            if Sm.shape != Tr.shape:
                raise ReconRefusal(f'{lst} {scope} {v}: striation model {Sm.shape} and residual stack {Tr.shape} grids differ - REFUSING')
            Tr = Tr - np.nan_to_num(Sm)
        R = Tr + lowpass(np.where(okr, Rf - Tr, 0.0), cd); R[~okr] = np.nan
        how = 'Taper residual stack' + (' - striation model' if smodel else '') + ' + G (x) (reconstruction final residual - it)'
    if R is None:
        rec['product_cleanres'] = None; log(lst, scope, v, 'no CLEAN residual product:', rec['cleanres_skipped'])
    else:
        pr = parse_name(name); pr['proc'] = list(pr['proc']) + ['cleanres']; rname = build_name(**pr); rpath = f'{wd}/I_deep_Taper_{v}_lsrecon_lsfill_cleanres_{scope}.fits'
        hr = fits.getheader(path).copy(); hr['CLEANRES'] = (True, 'CLEAN residual of the hybrid'); hr['HISTORY'] = f'cleanres: {how}'
        _write_product(rpath, R, hr, rname, scope, span, beam, False, gen)
        rec['product_cleanres'] = dict(path=rpath, prodname=rname, md5=md5(rpath), method=how, test_readback=test_readback(rpath, R, rname))
    _dump(rec, f'{wd}/{v}_hybrid.json')
    return rec


def _subband_thermnoi(lst, b, v, rpath):
    """THERMNOI of a subband for the wideband weight 1/THERMNOI^2 (as combine_wideband): the reconstruction's header, else the cell's
    other installed masters (one value per cell)."""
    tn, src = fits.getheader(rpath).get('THERMNOI'), 'reconstruction (installed NoTaper master)'
    ow = 'R0' if v == 'R075' else 'R075'
    for tp_, v_ in ((True, v), (False, ow), (True, ow)):
        if tn is not None and float(tn) > 0:
            break
        tn = fits.getheader(installed_master(lst, b, v_, tp_)[0]).get('THERMNOI'); src = f"installed {'Taper' if tp_ else 'NoTaper'} {v_} master of the cell"
    if tn is None or not float(tn) > 0:
        raise ReconRefusal(f'{lst} {b} {v}: no THERMNOI for the weight - REFUSING')
    return float(tn), src


def hybrid_full(lst, v, run_dir):
    """FULL-BAND hybrid NoTaper master (Gregg 2026-11-08: "I need as a minimum a Stokes I Hybrid for all subbands and widebands for all LSTs
    for both R0 and R-0.75"). The 15 subbands sit on three grids (Red 1508 px at 0.0767 deg, Green 2358 at 0.049, Blue / Full 3122 at 0.037),
    so the reconstructions cannot be combined on one grid as Red / Green / Blue are. Full hybrid = the installed Taper Full master (striation-
    restored where ruled) + G (x) (sum_s w_s regrid(H_s) - Taper Full): H_s = the cell's 15 passed subband hybrids (each = its Taper master +
    G (x) (its reconstruction - Taper)), regridded bilinearly onto the Taper Full grid; w_s = 1/THERMNOI^2 normalized (the Red / Green / Blue
    weights); G keeps only scales above ~2 deg (1 below 30 wavelengths, cosine to 0 at 60), where the bilinear regridding of the smooth sky is
    far below the noise - its errors at compact scales are filtered out. Beam-corrected sibling = Taper part / full PB + added large scales /
    the w_s mean of the subbands' diffuse PB maps (regridded). No CLEAN residual product (subbands on three grids; recorded). Caveats as Blue:
    the large-scale level = the subbands' (ratio of the weighted 4-deg model and prior means where every subband record keeps them, else the
    weighted mean of the subbands' ratios - the 18-59 MHz records keep the ratios only; method recorded); the small-scale test a caveat. Run
    directory: <run_dir>/<LST>_<subband> must reach each subband's hybrid run (links into Run_20261028_234240 / Run_20261031_172532)."""
    bind()                                        # v2: config constants
    from reproject import reproject_interp
    scope = 'Full'; wd = f'{run_dir}/{lst}_{scope}'; os.makedirs(wd, exist_ok=True)
    tpath, tppath, smodel = taper_base(lst, scope, v); T = rd(tpath); th = fits.getheader(tpath); tw = WCS(th).celestial; cd = abs(float(th['CDELT1']))
    subs = []
    for b in WIDEBANDS[scope]:
        hj = f'{run_dir}/{lst}_{b}/{v}_hybrid.json'
        if not os.path.exists(hj):
            raise ReconRefusal(f'{lst} {scope} {v}: {b} holds no passed hybrid ({hj}) - REFUSING')
        sh = json.load(open(hj)); rr = json.load(open(f'{run_dir}/{lst}_{b}/{v}_lsrecon.json'))
        hh = fits.getheader(sh['product']['path']); sw = WCS(hh).celestial; tn, tsrc = _subband_thermnoi(lst, b, v, rr['product']['path'])
        Hs = reproject_interp((rd(sh['product']['path']), sw), tw, shape_out=T.shape, order='bilinear')[0]
        Pd = reproject_interp((diffuse_pb_map(run_dir, lst, b)[0], sw), tw, shape_out=T.shape, order='bilinear')[0]
        subs.append(dict(band=b, H=Hs, pbd=Pd, w=1.0 / tn ** 2, thermnoi=tn, thermnoi_source=tsrc, name=sh['product']['prodname'],
                         level=rr.get('test_large_scale_level') or {}, admitted=bool(rr.get('stack_admission'))))
    W = np.array([s['w'] for s in subs]); Wn = W / W.sum()
    A = sum(wn * s['H'] for wn, s in zip(Wn, subs)); ok = np.isfinite(A) & np.isfinite(T)
    F = lowpass(np.where(ok, A - T, 0.0), cd); F[~ok] = np.nan; H = T + F
    keys = [f'{lo:+d}..{hi:+d}' for lo, hi in DEC_BANDS]
    if all('model_mean_by_dec_band' in s['level'] for s in subs):
        keys = [k for k in keys if all(k in s['level']['model_mean_by_dec_band'] for s in subs)]
        am = {k: float(sum(wn * s['level']['model_mean_by_dec_band'][k] for wn, s in zip(Wn, subs))) for k in keys}
        bm = {k: float(sum(wn * s['level']['prior_mean_by_dec_band'][k] for wn, s in zip(Wn, subs))) for k in keys}
        level = _level_verdict({k: round(am[k] / bm[k], 4) for k in keys}, scope, model_mean_by_dec_band=am, prior_mean_by_dec_band=bm,
                               method="ratio of the subbands' 4-deg model and prior means summed with the weights")
    elif all('ratio_by_dec_band' in s['level'] for s in subs):
        keys = [k for k in keys if all(k in s['level']['ratio_by_dec_band'] for s in subs)]
        level = _level_verdict({k: round(float(sum(wn * s['level']['ratio_by_dec_band'][k] for wn, s in zip(Wn, subs))), 4) for k in keys}, scope,
                               method="weighted mean of the subbands' level ratios (the 18-59 MHz records keep the ratios only)")
    else:
        raise ReconRefusal(f'{lst} {scope} {v}: a subband record lacks its large-scale level - REFUSING')
    level['subband_verdicts'] = {s['band']: s['level'].get('verdict') for s in subs}
    cards = lscaveat_cards(level, small_scale_rel(H.astype(np.float32).astype(np.float64), T, ok, th))
    span = hybrid_span(run_dir, lst, scope, v); dates = [n['date'] for n in span['nights']]
    pn = parse_name(os.path.basename(tpath)); pn['proc'] = list(pn['proc']) + ['lsrecon', 'lsfill']
    pn.update(date_start=dates[0].replace('-', ''), date_end=dates[-1].replace('-', ''), n_epochs=len(dates)); name = build_name(**pn)
    h = th.copy(); hist = [str(x) for x in h.get('HISTORY', []) if not str(x).startswith('COADD constituent')]
    h.remove('HISTORY', ignore_missing=True, remove_all=True)
    for x in hist:
        h['HISTORY'] = x
    for d in dates:
        h['HISTORY'] = f'COADD constituent {d}'
    h['NCOADD'] = (len(dates), 'distinct nights: Taper master and reconstructions'); h['LSRECON'] = (True, 'large scales from the subband reconstructions')
    h['LSFILL'] = (True, 'hybrid: Taper + G (x) (sum w regrid(subband hybrids) - Taper)'); h['LSFILLU'] = (f'{HYB_U_IN:g}-{HYB_U_OUT:g}', 'G: 1 below, cosine to 0 at (wavelengths)')
    h['LSRGEN'] = (RECON_GEN, 'reconstruction: CLEAN, step and prior to the true horizon')
    h['LSWBSB'] = (','.join(s['band'] for s in subs), 'combined subband hybrids'); h['LSWBW'] = (','.join(f'{x:.4f}' for x in Wn), 'normalized subband weights')
    for k, val, com in cards:
        h[k] = (val, com)
    raw_b = [s['band'] for s in subs if (lst, s['band']) in RAW_NIGHT_CAVEAT]
    if raw_b:
        h['LSRAWNT'] = (','.join(raw_b), 'undewarped nights in recon stacks (caveat)')
    adm_b = [s['band'] for s in subs if s['admitted']]
    if adm_b:
        h['LSSTKADM'] = (','.join(adm_b), 'stack admitted at hybrid scales (Gregg 10-31)')
    h['HISTORY'] = f'hybrid {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}: Taper master {os.path.basename(tpath)}'
    for s in subs:
        h['HISTORY'] = f'Full hybrid large scales from {s["name"]}'
    beam = (float(th['BMAJ']), float(th['BMIN']), float(th.get('BPA', 0.0))); gen = 'hybrid-' + RECON_GEN
    path = f'{wd}/I_deep_Taper_{v}_lsrecon_lsfill_{scope}.fits'; _write_product(path, H, h, name, scope, span, beam, False, gen)
    nt = installed_master(lst, scope, v, False)[0]
    rec = dict(lst=lst, scope=scope, weighting=v, taper_master=tpath, striation_model=smodel, reconstruction=None,
               subband_hybrids=[dict(band=s['band'], product=s['name'], weight=float(wn), thermnoi=s['thermnoi'], thermnoi_source=s['thermnoi_source'])
                                for wn, s in zip(Wn, subs)],
               method='Taper Full + G (x) (sum_s w_s regrid(subband hybrid) - Taper Full); bilinear regrid onto the Taper Full grid; w_s = 1/THERMNOI^2',
               span=span, filter_wavelengths=[HYB_U_IN, HYB_U_OUT], stacks_admitted_at_hybrid_scales=adm_b,
               raw_night_caveat={b: RAW_NIGHT_CAVEAT[(lst, b)] for b in raw_b})
    rec['test_hybrid'] = test_hybrid(path, A, T, F, ok, th, lst, scope, rd(nt))
    if f"{float(fits.getheader(path).get('LSSMALL', float('nan'))):.4g}" != f"{rec['test_hybrid']['small_scale_rel']:.4g}":
        raise ReconRefusal(f'{path}: LSSMALL {fits.getheader(path).get("LSSMALL")} is not the tested {rec["test_hybrid"]["small_scale_rel"]} - REFUSING')
    rec['caveats'] = dict(level=level, small_scale_rel=rec['test_hybrid']['small_scale_rel'], small_scale_caveat=rec['test_hybrid']['small_scale_caveat'])
    rec['product'] = dict(path=path, prodname=name, md5=md5(path))
    pbf = rd(f'{ARCHIVE_MASTERS}/{lst}/{scope}/{PBA.deep_pb_name(scope, v)}')
    pbd = sum(wn * s['pbd'] for wn, s in zip(Wn, subs))
    pbdf = f'{wd}/PBdiffuse_deep_{scope}.fits'
    hd_ = diag_header(fits.getheader(path), 'diffuse primary beam'); hd_['BUNIT'] = ''; hd_['PBMODEL'] = ('w mean of the 15 subband diffuse PBs, regridded', 'diffuse PB')
    # per-process .part: both weightings of a cell write this (weighting-independent) map; a shared .part raced at 01h (2026-11-09)
    part = f'{pbdf}.{os.getpid()}.part'
    fits.writeto(part, pbd.astype(np.float32), hd_, overwrite=True); os.replace(part, pbdf); pbd = rd(pbdf)
    valid = ok & np.isfinite(rd(tppath)) & np.isfinite(pbf) & (pbf > 0) & np.isfinite(pbd) & (pbd > 0)
    P = np.full(H.shape, np.nan); P[valid] = T[valid] / pbf[valid] + F[valid] / pbd[valid]
    pp = parse_name(name); pp['proc'] = list(pp['proc']) + ['pbcorr']; pname = build_name(**pp); ppath = f'{wd}/I_deep_Taper_{v}_lsrecon_lsfill_pbcorr_{scope}.fits'
    hp = fits.getheader(path).copy(); hp['PBFLOOR'] = (PBA.PB_FLOOR, 'PB floor below which pixels set NaN')
    hp['PBMETHOD'] = ('hybrid-split', 'Taper part / full PB; added large scales / diffuse PB'); hp['PBMAP'] = (PBA.deep_pb_name(scope, v), 'full PB map (archive)')
    _write_product(ppath, P, hp, pname, scope, span, beam, True, gen)
    rec['product_pbcorr'] = dict(path=ppath, prodname=pname, md5=md5(ppath), test_hybrid_pbcorr=test_hybrid_pbcorr(ppath, F, pbd, tppath))
    rec['product_cleanres'] = None; rec['cleanres_skipped'] = 'Full: no CLEAN residual product (its subbands sit on three grids)'
    _dump(rec, f'{wd}/{v}_hybrid.json')
    return rec


def stack_refused(run_dir, lst, band, v):
    """The recorded refusal of a Taper residual stack (v = T075 / T0) with no stack record: the reason the stack unit of its NoTaper
    weighting recorded for the refused item (v2: <run_dir>/_records/hybrid/stack.<LST>_<band>_<w>.json, refused_items; production
    NL 1505 read the stack fleet's CSV rows), else None (not run, or passed)."""
    if os.path.exists(f'{run_dir}/{lst}_{band}/{v}_stack.json'):
        return None
    w = {t: n for n, t in TAPER_OF.items()}[v]
    rp = f'{run_dir}/_records/hybrid/stack.{lst}_{band}_{w}.json'
    if not os.path.exists(rp):
        return None
    with open(rp) as f:
        r = json.load(f)
    want = os.path.abspath(f'{run_dir}/{lst}_{band}/{v}_residual_stack.fits')
    why = [i['reason'] for i in r.get('refused_items') or [] if i.get('path') == want]
    return why[-1] if why else None


def diffuse_pb_map(run_dir, lst, scope, v='R075'):
    """The cell's diffuse primary beam (model x residual, no bandwidth-smearing term; identical for both weightings) as used by the
    split correction: a subband's Robust -0.75 PB map / its smearing term; Red / Green / Blue the weighted mean of the subbands' maps
    with the combination's weights (1/THERMNOI^2, the same for both weightings) - read from weighting v's combination record (Blue:
    a weighting's combination can be missing where a subband's stack of that weighting refused, 2026-10-31)."""
    if scope in WIDEBANDS:
        r = json.load(open(f'{run_dir}/{lst}_{scope}/{v}_lsrecon_wb.json'))
        return sum(s['weight'] * diffuse_pb_map(run_dir, lst, s['band'])[0] for s in r['subbands']), f'ivw mean of {r["subbands"][0]["band"]}-{r["subbands"][-1]["band"]}'
    pbf = f'{run_dir}/{lst}_{scope}/inputs/PB_deep_R075_{scope}.fits'; ph = fits.getheader(pbf); pb = rd(pbf)
    ch, _ = PBA.chromatic_gain(ph, pb.shape, float(PBA.cfg.SUBBAND_FREQ_HZ[scope]['center']), float(ph['PBBMAJ']))
    return pb / ch, os.path.basename(pbf) + ' / bandwidth-smearing term'


def full_pb_map(run_dir, lst, scope, v):
    """The weighting's full primary beam (model x bandwidth smearing x residual), the map whose PB=0.1 contour blanks the beam-corrected
    product: a subband's installed PB map (inputs copy); Red / Green the weighted mean of the subbands' maps with the combination's
    weights (the same PBF the split correction divided by)."""
    name = PBA.deep_pb_name(scope, v)
    if scope in WIDEBANDS:
        r = json.load(open(f'{run_dir}/{lst}_{scope}/{v}_lsrecon_wb.json'))
        return sum(s['weight'] * full_pb_map(run_dir, lst, s['band'], v) for s in r['subbands'])
    return rd(f'{run_dir}/{lst}_{scope}/inputs/{PBA.deep_pb_name(scope, v)}')


