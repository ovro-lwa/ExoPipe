"""Stage `fill` of mosaic v5 (spec plan/specs/hybrid.md 2.14, gate HG19): the zero-spacing fill of a hybrid Jy/sr mosaic from the
LWA1 Low Frequency Sky Survey and its Jy/beam companion. A close port of production exopipe/phase3/mosaic_tpfill.py (MF) at exopipe
main 4c7098c: fill_arrays 75, _smooth 80, test_arrays 85, fill_file 106, fill_jybeam_file 141. The fill field itself (donor_mhz,
lfss_jysr, apod, fill_correction) is lib/hybrid/tilefill.py's port of the same module, so the mosaic and the per-tile fill share one
copy (constants bound from hybrid_tilefill there; the test limits here from mosaic_fill).

THE FILL (Gregg 2026-11-10, his July design restored; R-GREGG-20261110-TOTAL-POWER-FILL-HYBRID-MOSAICS-R075-THEN-R0): inside the map's
footprint, the multipoles l < 8 of the map are replaced by the LFSS (scaled by s = sqrt(C_l map / C_l LFSS) over l 10-30), with a
cosine blend to l = 12. The Jy/beam map gets the same correction through its own Jy/beam per Jy/sr conversion (Gregg 2026-11-11 "go
with your recommendations", R-GREGG-20261111-KEEP-LSFILL-NAME-FOR-ZERO-SPACING-MOSAICS).

Differences from production (none changes a pixel):
- production filled the conformed archive-staged map (its name plus the token tpfill), then renamed it back to the plain lsfill name
  with a PRODNAME rewrite; v2 fills the work co-add into a work file named by the caller, writes no PRODNAME (the conform stage names
  the archive file) and never writes the retired token tpfill (PR-1: parse-only);
- TPSRC holds the unfilled map's full name (D8; production cut it at 68 characters) and a new card TPSRCMD5 its md5, so the band field
  (hybrid tilefill, HG13) can prove which unfilled map it differences;
- the test limits come from the config (section mosaic_fill), a refusal raises FillRefusal (production RuntimeError '... - REFUSING'),
  every FITS file is read with memmap=False (HG23).
"""
import os

import numpy as np
from astropy.io import fits

from exopipe.phase3.lib.hybrid import tilefill as TF
from exopipe.phase3.lib.install.steps_common import md5

hp = TF.hp


class FillRefusal(RuntimeError):
    """A gate or content test of the mosaic fill refused (HG19)."""


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG.mosaic_fill


def fill_arrays(t, fmhz):
    """(filled, s, L, mask, am): the filled map, NaN outside the footprint (MF 75)."""
    corr, s, L, mask, am = TF.fill_correction(t, fmhz)
    return np.where(mask, t + corr, np.nan).astype(np.float32), s, L, mask, am


def _smooth(m, mask, fwhm):
    """Normalized convolution inside the mask (MF 80)."""
    a = hp.smoothing(np.where(mask, m, 0.0), fwhm=np.radians(fwhm), iter=1)
    w = hp.smoothing(mask.astype(float), fwhm=np.radians(fwhm), iter=1)
    return np.where(w > 0.5, a / np.maximum(w, 1e-9), np.nan)


def test_arrays(f, t, L, s, mask, am):
    """The content test on arrays, degraded to mosaic_fill.TEST_NSIDE for the smoothing (MF 85): (1) large scales at LS_SMOOTH_DEG
    FWHM in the footprint interior follow s x LFSS within LS_MAX; (2) the change filled - unfilled is smooth: its high-pass part (minus
    the HP_SMOOTH_DEG smoothing) is at most HP_MAX of it in rms and HP_ABS_FRAC of the unfilled map's robust rms at most."""
    C = _cfg()
    n = int(C.TEST_NSIDE)
    dg = lambda x: hp.ud_grade(np.where(np.isfinite(x), x, hp.UNSEEN), n, pess=False)   # noqa: E731
    fm, tm, Lm = dg(f), dg(t), dg(np.where(mask, L, np.nan))
    mk = (fm != hp.UNSEEN) & (tm != hp.UNSEEN) & (Lm != hp.UNSEEN)
    inner = hp.ud_grade(am, n) > float(C.INNER_APOD_MIN)
    f30, t30, L30 = (_smooth(x, mk, float(C.LS_SMOOTH_DEG)) for x in (fm, tm, Lm))
    ok = inner & np.isfinite(f30) & np.isfinite(L30) & np.isfinite(t30)
    ls_f = float(np.median(np.abs(f30[ok] - s * L30[ok])) / np.median(np.abs(s * L30[ok])))
    ls_t = float(np.median(np.abs(t30[ok] - s * L30[ok])) / np.median(np.abs(s * L30[ok])))
    d = np.where(mk, fm - tm, 0.0)
    d5 = _smooth(d, mk, float(C.HP_SMOOTH_DEG))
    hpd = (d - d5)[inner & np.isfinite(d5)]
    rob = 1.4826 * np.median(np.abs(tm[mk] - np.median(tm[mk])))
    hp_rel = float(np.sqrt(np.mean(hpd ** 2)) / np.sqrt(np.mean(d[inner] ** 2)))
    hp_abs = float(np.max(np.abs(hpd)) / rob)
    rec = dict(large_scale_rel_filled=ls_f, large_scale_rel_unfilled=ls_t, highpass_rel=hp_rel, highpass_abs_over_map_rms=hp_abs, s=s,
               limits=dict(LS_MAX=C.LS_MAX, HP_MAX=C.HP_MAX, HP_ABS_FRAC=C.HP_ABS_FRAC))
    rec['result'] = 'PASS' if (ls_f <= C.LS_MAX and hp_rel <= C.HP_MAX and hp_abs <= C.HP_ABS_FRAC) else 'FAIL'
    return rec


def _set_cards(hl, cards, hdus=(0, 1)):
    for i in hdus:
        for k, v in cards:
            hl[i].header[k] = v


def fill_file(src, out):
    """Fill the unfilled Jy/sr co-add `src` into `out` (all HDUs kept, SIGNAL replaced; cards TPFILL, TPDONOR, TPDONMHZ, TPSCALE,
    TPLLO, TPLHI, TPSRC, TPSRCMD5 in HDU 0 and 1, a HISTORY line); return the record with the content test of the WRITTEN file
    (MF 106). Refuses an existing `out` and a source that is already filled."""
    if os.path.exists(out):
        raise FillRefusal(f'{out} exists - REFUSING')
    TF.bind()
    h1 = TF._getheader(src, 1)
    if h1.get('TPFILL'):
        raise FillRefusal(f'{src} is already filled (TPFILL) - REFUSING')
    f_hz = float(h1['FREQ'])
    fmhz = TF.donor_mhz(f_hz)
    src_md5 = md5(src)
    t = TF._read(src)
    f, s, L, mask, am = fill_arrays(t, fmhz)
    with fits.open(src, memmap=False) as hl:
        hl[1].data['SIGNAL'][:] = f
        _set_cards(hl, [('TPFILL', (True, 'total-power fill: LWA1 LFSS replaces l<8, blend to 12')),
                        ('TPDONOR', ('LWA1 LFSS (Dowell+2017), LEDA zero level', 'donor')),
                        ('TPDONMHZ', (fmhz, 'donor map frequency [MHz]')),
                        ('TPSCALE', (round(s, 5), 's = sqrt(Cl map / Cl LFSS), l 10-30')),
                        ('TPLLO', (TF.L_LO, 'l below: survey')), ('TPLHI', (TF.L_HI, 'l above: map')),
                        ('TPSRC', (os.path.basename(src), 'unfilled map')),
                        ('TPSRCMD5', (src_md5, 'md5 of the unfilled map'))])
        hl[0].header['HISTORY'] = (f'zero-spacing fill (lsfill): LFSS {fmhz} MHz x {s:.4f} replaces l<{TF.L_LO}, cosine to l {TF.L_HI} '
                                   f'(Gregg 2026-11-10 July design restored; lsfill name 2026-11-11)')
        hl.writeto(out + '.part', overwrite=True)
    os.replace(out + '.part', out)
    if md5(src) != src_md5:
        raise FillRefusal(f'{src} changed while it was filled - REFUSING')
    fw = TF._read(out)                                       # the test reads the WRITTEN file
    rec = dict(src=src, src_md5=src_md5, out=out, out_md5=md5(out), freq_hz=f_hz, donor_mhz=fmhz, s=s)
    rec['test'] = test_arrays(fw, t, L, s, mask, am)
    rec['result'] = rec['test']['result']
    return rec


def fill_jybeam_file(jb_src, js_unfilled, js_filled, psf, pbeff, out):
    """The Jy/beam companion of a fill (MF 141): the Jy/sr correction dS = filled - unfilled enters the apparent Jy/beam map through
    the map's own conversion K = Omega_eff x PB_eff x r, r = jybeam / (jysr x Omega_eff x PB_eff) at the bright pixels (|jysr| above
    its JB_BRIGHT_PERCENTILE percentile, PB_eff > JB_PBEFF_MIN), weighted by |jysr| and smoothed to JB_SMOOTH_DEG. Content test on the
    written file: (1) identity (written - unfilled jybeam) = dS x K to float32 storage precision; (2) median jybeam / (jysr x K) at the
    bright pixels within JB_MED_TOL of 1 and its 5-95 percentile spread within JB_SPREAD. Cards TPFILL..TPLHI from the filled Jy/sr
    map, TPJBCONV, TPSRC / TPSRCMD5 of the unfilled Jy/beam map."""
    C = _cfg()
    if os.path.exists(out):
        raise FillRefusal(f'{out} exists - REFUSING')
    TF.bind()
    if TF._getheader(jb_src, 1).get('TPFILL'):
        raise FillRefusal(f'{jb_src} is already filled (TPFILL) - REFUSING')
    jb_md5 = md5(jb_src)
    jb, ju, jf, om, pe = (TF._read(p) for p in (jb_src, js_unfilled, js_filled, psf, pbeff))
    ok = np.isfinite(jb) & np.isfinite(ju) & np.isfinite(om) & np.isfinite(pe) & (pe > float(C.JB_PBEFF_MIN)) & (np.abs(ju) > 0)
    thr = np.percentile(np.abs(ju[ok]), float(C.JB_BRIGHT_PERCENTILE))
    br = ok & (np.abs(ju) > thr)
    r = np.where(br, jb / np.where(br, ju * om * pe, 1.0), 0.0)
    wgt = np.where(br, np.abs(ju), 0.0)
    fw = np.radians(float(C.JB_SMOOTH_DEG))
    rs = hp.smoothing(r * wgt, fwhm=fw, iter=1) / np.maximum(hp.smoothing(wgt, fwhm=fw, iter=1), 1e-30)
    K = om * pe * rs
    dS = np.where(np.isfinite(jf) & np.isfinite(ju), jf - ju, 0.0)
    new = np.where(np.isfinite(jb), jb + dS * np.nan_to_num(K), np.nan).astype(np.float32)
    with fits.open(jb_src, memmap=False) as hl, fits.open(js_filled, memmap=False) as hf:
        hl[1].data['SIGNAL'][:] = new
        for k in ('TPFILL', 'TPDONOR', 'TPDONMHZ', 'TPSCALE', 'TPLLO', 'TPLHI'):
            if k in hf[1].header:
                _set_cards(hl, [(k, (hf[1].header[k], hf[1].header.comments[k]))])
        _set_cards(hl, [('TPJBCONV', ('jybeam/jysr measured, 5-deg smooth', 'Jy/beam conversion of the fill')),
                        ('TPSRC', (os.path.basename(jb_src), 'unfilled map')), ('TPSRCMD5', (jb_md5, 'md5 of the unfilled map'))])
        hl[0].header['HISTORY'] = ("zero-spacing fill (lsfill): the Jy/sr fill x K (the map's own Jy/beam per Jy/sr, 5-deg smooth) "
                                   "(Gregg 2026-11-11)")
        hl.writeto(out + '.part', overwrite=True)
    os.replace(out + '.part', out)
    if md5(jb_src) != jb_md5:
        raise FillRefusal(f'{jb_src} changed while it was filled - REFUSING')
    w = TF._read(out)
    dK = dS * np.nan_to_num(K)
    m = np.isfinite(w) & np.isfinite(jb)
    # identity up to float32 storage (MF 182-185): per pixel |(w - jb) - dK| <= 4 x 2^-24 x (|w| + |jb|) + 1e-6 max|dK|
    err = np.abs((w - jb)[m] - dK[m])
    allow = 4 * 2.0 ** -24 * (np.abs(w[m]) + np.abs(jb[m])) + 1e-6 * max(np.max(np.abs(dK[m])), 1e-30)
    ident = float(np.max(err / allow))
    q = jb[br] / (ju[br] * K[br])
    q = q[np.isfinite(q)]
    med, p05, p95 = (float(x) for x in np.percentile(q, [50, 5, 95]))
    res = 'PASS' if (ident <= 1.0 and abs(med - 1) <= C.JB_MED_TOL and (p95 - p05) <= C.JB_SPREAD) else 'FAIL'
    return dict(src=jb_src, src_md5=jb_md5, out=out, out_md5=md5(out), identity_rel=ident, k_median=med, k_p05=p05, k_p95=p95,
                n_bright=int(br.sum()), limits=dict(JB_MED_TOL=C.JB_MED_TOL, JB_SPREAD=C.JB_SPREAD), result=res)
