"""Stage `stack` of `exopipe3 hybrid` (spec plan/specs/hybrid.md 2.1): the NoTaper dirty stack, its two half-stacks and the PSF
stack of a unit, and the Taper CLEAN-residual stack, each rebuilt from the Phase 1 deep images of the installed master's
constituent nights through the production co-add mapping (flows, flux factors, regrid, inverse-variance co-add without clipping,
master dewarp). Port of notaper_largescale.py (NL) at exopipe 4c7098c: cell_paths 149, members_of 154, _dc 177, _p1file 182,
constituent_span 193, stack_variant 304, test_stack_rebuild 461. The co-add functions are v2's port of coadd_full_pipeline
(lib/masters/coadd.py), the master dewarp's resampling lib/dewarp/master_dewarp.resample, the flux factor
lib/masters/flux_correction.pixel_factor.

Gates (each refuses; StackRefusal names its gate): HG1 the installed master's constituents equal the cell's recorded members and
none is excluded in data_decisions.csv (read now); HG4 at most hybrid_stack.MAX_UNDERIVABLE night per stack that cannot be put on
the master's frame, and at least max(2, N - hybrid_stack.MIN_NIGHTS_DROP) nights with Phase 1 images; HG5 the stack rebuild (the
chain on the nights' Phase 2 images rebuilds the installed master), with the flow-recovered limits and the hybrid-scale admission
(hybrid_stack.ADMISSION).

Changes from production (none changes a pixel): limits and paths come from config (same values); outputs are written through a
temporary name and renamed; the scratch directory is the caller's (node-local, removed after each variant); cached derived warp
screens are reused only while their raw / dewarped pair is unchanged (warp.py). Arrays are read with ordinary reads (memmap=False).
"""
import csv
import glob
import json
import os
import re
import shutil

import numpy as np

from exopipe.phase3.lib.hybrid.common import (TAPER_OF, VARIANT, diag_header, inputs_dir, log, lowpass, md5, rd,
                                              sky_interior, write_atomic)
from exopipe.phase3.lib.hybrid.warp import WarpRefusal, derive_warp_screen, offsets_pass, offsets_summary, peak_offsets


class StackRefusal(RuntimeError):
    """A gate of the stack stage refused the variant (gate: HG1, HG4 or HG5)."""

    def __init__(self, msg, gate):
        super().__init__(msg)
        self.gate = gate


def _CFP():
    from exopipe.phase3.lib.masters import coadd as CFP
    return CFP


def cell_paths(coadd_sb, lst, band, v):
    wd = f'{coadd_sb}/{lst}_{band}'
    cf = f'{wd}/coadd_final'
    return wd, cf, f'{cf}/I_deep_{VARIANT[v][0]}_{band}.fits'


def members_of(coadd_sb, lst, band, v, excl=None):
    """Constituent nights of the installed variant master: weights.csv in_master rows minus the nights the per-variant divergence
    gate refused ('VARIANT-GATE refused <date>' HISTORY) and minus nights with no Phase 2 image of this variant (the co-add finds none
    and leaves the night out without a record: 08h and 09h 18 MHz Robust 0, 2026-10-26); must equal the master's constituent dates
    and its NCOADD (HG1), and none may be excluded by data_decisions.csv (`excl`: an exclusion read, default the deep read made NOW).
    Returns (rows, refused + no-image dates)."""
    from astropy.io import fits
    CFP = _CFP()
    wd, cf, mfile = cell_paths(coadd_sb, lst, band, v)
    with open(f'{cf}/weights.csv', newline='') as fh:
        rows = [r for r in csv.DictReader(fh) if str(r.get('in_master', '')).strip() in ('1', 'True', 'true')]
    mh = fits.getheader(mfile)
    refused = sorted({m.group(1) for x in mh.get('HISTORY', []) for m in [re.search(r'VARIANT-GATE refused (20\d\d-\d\d-\d\d)', str(x))] if m})
    noimg = [r['date'] for r in rows if r['date'] not in refused
             and not CFP.find_subband_variant_image(r['sci_dir'], band.replace('MHz', ''), VARIANT[v][0])]
    refused = refused + [f'{d} (no {VARIANT[v][0]} image)' for d in noimg]
    rows = [r for r in rows if r['date'] not in refused and r['date'] not in noimg]
    const = [d for d in CFP._constituent_dates(mh) if d not in refused]        # 'COADD constituent' lines only
    if sorted(r['date'] for r in rows) != const or len(rows) != int(mh.get('NCOADD', len(rows))):
        raise StackRefusal(f'{lst} {band} {v}: weights.csv members {sorted(r["date"] for r in rows)} != master constituents {const} '
                           f'(NCOADD {mh.get("NCOADD")}, gate-refused {refused}) - REFUSING', 'HG1')
    if excl is None:
        from exopipe.phase3.core import exclusions
        excl = exclusions.for_tier('deep', fresh=True)
    bad = excl.excluded_constituents(lst, band, const)
    if bad:
        raise StackRefusal(f'{lst} {band} {v}: the installed master holds nights excluded in data_decisions.csv {bad} - rebuild the '
                           f'master first - REFUSING', 'HG1')
    return rows, refused


def night_dc(r):
    """The night's flux factor as the co-add applied it (flux_dc, else qa3_scale within 0.3-3, else 1)."""
    m = float(r['qa3_scale'])
    f_ = (r.get('flux_dc') or '').strip()
    return float(f_) if f_ not in ('', 'affine') else (m if 0.3 < m < 3.0 else 1.0)


def p1file(phase1_root, lst, band, pvar, r, kind):
    """The night's Phase 1 deep image of `kind` (dirty, psf, residual) and variant; with more than one Phase 1 run, the newest one
    older than the night's Phase 2 science run. Refuses (HG4) unless exactly one."""
    p1 = sorted(f for f in glob.glob(f'{phase1_root}/{lst}/{r["date"]}/Run_*/{band}/I/deep/{band}-I-Deep-{pvar}-{kind}-*.fits')
                if not f.endswith('.pbcorr.fits'))
    if len(p1) > 1:
        ts = os.path.basename(r['sci_dir']).split('_', 1)[1]
        p1 = [f for f in p1 if f.split('/Run_')[1][:15] < ts][-1:]
    if len(p1) != 1:
        raise StackRefusal(f'{lst} {r["date"]} {pvar} {kind}: {len(p1)} Phase 1 files - REFUSING', 'HG4')
    return p1[0]


def constituent_span(lst, band, rows, vk, read=None):
    """Time block of the co-add from the image the co-add used for each night (start of the earliest, end of the latest, summed
    integration); a night whose image lacks any of the three refuses."""
    from astropy.io import fits
    from astropy.time import Time
    CFP = _CFP()
    freq = band.replace('MHz', '')
    t0, t1, it, per = [], [], 0.0, []
    for r in rows:
        p = CFP.find_subband_variant_image(r['sci_dir'], freq, vk)
        h = fits.getheader(p)
        if read is not None:
            read.add(p)
        if not (h.get('DATE-OBS') and h.get('DATE-END') and h.get('INTTIME') is not None):
            raise StackRefusal(f'{lst} {band} {r["date"]}: {p} has no complete time block - REFUSING', 'HG1')
        t0.append(str(h['DATE-OBS']))
        t1.append(str(h['DATE-END']))
        it += float(h['INTTIME'])
        per.append(dict(date=r['date'], image=p, start=str(h['DATE-OBS']), end=str(h['DATE-END']), inttime_s=float(h['INTTIME'])))
    return dict(start=min(t0, key=lambda s: Time(s).mjd), end=max(t1, key=lambda s: Time(s).mjd), inttime_s=it, nights=per)


def outputs_of(v):
    """The files a variant's stack writes in the cell's work directory (inputs/ copies: NoTaper only, see input_copies)."""
    if v in TAPER_OF.values():
        return [f'{v}_residual_stack.fits', f'{v}_stack.json']
    return [f'{v}_dirty_stack.fits', f'{v}_dirty_halfA.fits', f'{v}_dirty_halfB.fits', f'{v}_psf_stack.fits', f'{v}_stack.json']


def input_copies(coadd_sb, lst, band, v):
    """The installed files a NoTaper stack copies into <workdir>/inputs/ for the reconstruction: its master, the Robust -0.75
    master, and the cell's two PB maps."""
    wd, cf, mfile = cell_paths(coadd_sb, lst, band, v)
    return sorted({mfile, cell_paths(coadd_sb, lst, band, 'R075')[2], f'{cf}/PB_deep_R075_{band}.fits', f'{cf}/PB_deep_{band}.fits'})


def stack_variant(lst, band, v, workdir, prep, screen_cache, read=None):
    """Stacks of one cell and weighting (R075 / R0: NoTaper dirty, half and PSF stacks + inputs/ copies; T075 / T0: the Taper CLEAN
    residual stack) through the production co-add mapping. `prep`: an empty node-local directory (the caller removes it);
    `screen_cache`: the node-local derived-screen cache; `read`: a set the paths read are added to. Returns the record (dict, also
    written as <workdir>/<v>_stack.json). Raises StackRefusal (gates HG1, HG4, HG5)."""
    from astropy.io import fits
    from scipy.ndimage import map_coordinates
    from exopipe.phase2.cascade_dewarp import apply_warp
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.dewarp import master_dewarp as MDW
    from exopipe.phase3.lib.masters import flux_correction as FC
    CFP = _CFP()
    S, H = CFG.hybrid_stack, CFG.hybrid
    read = set() if read is None else read
    wd, cf, mfile = cell_paths(S.COADD_SB, lst, band, v)
    vk, pvar, _rob = VARIANT[v]
    freq = band.replace('MHz', '')
    read.update({f'{cf}/weights.csv', mfile})
    rows, refused = members_of(S.COADD_SB, lst, band, v)
    all_dates = CFP.discover_dates(lst, freq)
    post = [x for x in all_dates if x[0] >= CFP.CUTOVER_DATE]
    tmpl, shape = CFP._modal_grid_template(post, all_dates[0]) if post else CFP.build_template_header(all_dates[0][2])
    shape = tuple(shape)
    shdr = fits.getheader(f'{cf}/I_deep_Taper_R075_{band}.fits')
    if shape != tuple(int(shdr[f'NAXIS{i}']) for i in (2, 1)):
        raise StackRefusal(f'{lst} {band}: template grid {shape} != master grid - REFUSING', 'HG5')
    pf = FC.pixel_factor(shdr)
    md = f'{S.MDEWARP}/{lst}_{band}_warp_screen.fits'
    sx = sy = None
    if fits.getheader(mfile).get('MDWARP'):           # the master's own card decides (04h 18 MHz: rebuilt 2026-10-09 without the
        if not os.path.exists(md):                    # master dewarp although a 2026-10-04 screen exists)
            raise StackRefusal(f'{lst} {band} {v}: master is master-dewarped but {md} is missing - REFUSING', 'HG5')
        with fits.open(md, memmap=False) as h:
            sx, sy = h['SCREEN_X'].data.astype(np.float64), h['SCREEN_Y'].data.astype(np.float64)
        read.add(md)
    derived = {}                                      # nights whose warp screen was derived (no saved screen; derive_warp_screen)

    def night_input(r, kind):
        if kind == 'image':                           # the Phase 2 product exactly as production reads it
            p = CFP.find_subband_variant_image(r['sci_dir'], freq, vk)
            if not p:
                raise StackRefusal(f'{r["date"]}: no {vk} Phase 2 image - REFUSING', 'HG1')
            read.add(p)
            return p
        src = p1file(S.PHASE1_ROOT, lst, band, pvar, r, kind)
        read.add(src)
        with fits.open(src, memmap=False) as hd:
            a = np.squeeze(hd[0].data).astype(np.float64)
            h = hd[0].header.copy()
        scr = glob.glob(f'{r["sci_dir"]}/Dewarp_Diagnostics/maps/{r["date"]}_{lst}_{band}_warp_screen.fits')
        if len(scr) > 1:
            raise StackRefusal(f'{r["date"]}: {len(scr)} screens - REFUSING', 'HG4')
        wx = None
        if scr:
            with fits.open(scr[0], memmap=False) as hs:
                wx, wy = hs['SCREEN_X'].data.astype(np.float64), hs['SCREEN_Y'].data.astype(np.float64)
            read.add(scr[0])
        elif '_dewarped' in os.path.basename(night_input(r, 'image')):
            # the Phase 2 image the master uses IS dewarped but no screen was saved (flow cascade; 13h 2025-04/05 nights keep only
            # 73 MHz, 23h 2025-09-06 lacks 69 MHz): the screen derived as Phase 2 recovers it (Gregg 2026-11-01 "All approved"), and
            # the dirty image must then line up with that Phase 2 image (a residual image has no sources: the screen's own test
            # stands for it)
            try:
                wx, wy, drec = derive_warp_screen(r['sci_dir'], r['date'], lst, band, screen_cache)
            except WarpRefusal as e:
                raise StackRefusal(str(e), 'HG4')
            derived.setdefault(r['date'], dict(drec))
            if kind == 'dirty':
                d = peak_offsets(apply_warp(a, wx, wy), rd(night_input(r, 'image')))
                t = offsets_summary(d)
                derived[r['date']][f'test_dirty_vs_phase2_{v}'] = t
                if not offsets_pass(d):
                    raise StackRefusal(f'{r["date"]} {lst} {band} {v}: the dirty image dewarped with the derived screen does not line '
                                       f'up with the Phase 2 image {t} - REFUSING', 'HG4')
        if wx is not None:                            # (no screen and a raw Phase 2 image: the raw image enters, the production failsafe)
            if wx.shape != a.shape:
                raise StackRefusal(f'{r["date"]}: screen {wx.shape} vs {kind} {a.shape} - REFUSING', 'HG4')
            a = apply_warp(a, wx, wy)
        out = f'{prep}/in_{kind}_{r["date"]}.fits'
        for k in ('NAXIS3', 'NAXIS4'):
            h.pop(k, None)
        fits.writeto(out, a.astype(np.float32), h, overwrite=True)
        return out

    def chain(kind, rr):
        paths, wts = [], []
        for r in rr:
            src = night_input(r, kind)
            with fits.open(src, memmap=False) as hh:
                sshape = tuple(np.squeeze(hh[0].data).shape)
            rg = CFP.precompute_regrid_mapping(src, tmpl, shape) if sshape != shape else None
            flow = None
            if (r.get('qa3_guard') or '') != 'reference-identity':
                fp_ = f'{wd}/flow/{r["date"]}_scaled_flow.npy'
                flow = np.load(fp_)
                read.add(fp_)
            out = f'{prep}/al_{kind}_{r["date"]}.fits'
            if os.path.exists(out):
                os.remove(out)
            if not CFP.process_variant_image(src, flow, np.full(shape, night_dc(r), dtype=float), tmpl, shape, out, rg):
                raise StackRefusal(f'{r["date"]} {kind}: alignment failed - REFUSING', 'HG5')
            paths.append(out)
            wts.append(float(r['weight']))
            if kind != 'image':
                os.remove(src)
        cpath = f'{prep}/coadd_{kind}.fits'
        if os.path.exists(cpath):
            os.remove(cpath)
        CFP.coadd_inverse_variance(paths, wts, cpath, tmpl, dates=[r['date'] for r in rr], sigclip=False)
        d = rd(cpath).astype(np.float32).astype(np.float64) * (pf if pf is not None else 1.0)
        if sx is not None:
            d = MDW.resample(d, sx, sy)
        for p in paths + [cpath]:
            os.remove(p)
        return d

    rec = dict(lst=lst, band=band, weighting=v, nights=[dict(date=r['date'], weight=float(r['weight']), dc=night_dc(r)) for r in rows],
               variant_gate_refused=refused, pixel_factor=pf, master_dewarp=sx is not None,
               span=constituent_span(lst, band, rows, vk, read), code='exopipe v2 lib/hybrid/stack.py (port of NL 304, exopipe 4c7098c)')
    admit = S.ADMISSION == 'hybrid-scales' and band in H.CAVEAT_SCOPES and v in TAPER_OF   # admission at hybrid scales: high-band NoTaper only
    rec['test_stack_rebuild'] = test_stack_rebuild(chain('image', rows), rd(mfile), wd, cf, rows, fits.getheader(mfile) if admit else None)
    log(lst, band, v, rec['test_stack_rebuild']['verdict'])
    taper = v in TAPER_OF.values()
    kinds = ('residual',) if taper else ('dirty', 'psf')
    rec['span_master'] = rec['span']                  # all constituents

    def has_p1(r):
        try:
            for k in kinds:
                p1file(S.PHASE1_ROOT, lst, band, pvar, r, k)
            return True
        except StackRefusal:
            return False
    rows_d = [r for r in rows if has_p1(r)]           # Gregg 2026-10-27: a night whose Phase 1 dirty image / PSF was never written is
    nodrv = {}                                        # left out of the dirty stacks, recorded
    for r in rows_d:                                  # Gregg 2026-11-03 "(b) - if it's only dropping one night of 7, the impact is small!": a
        if (not glob.glob(f'{r["sci_dir"]}/Dewarp_Diagnostics/maps/{r["date"]}_{lst}_{band}_warp_screen.fits')  # night whose warp screen
                and '_dewarped' in os.path.basename(night_input(r, 'image'))):   # cannot be derived (23h 69 MHz 2025-09-06: 36 arcmin off the
            try:                                                                 # master's frame) is left out of these stacks - at most
                wx_, wy_, _ = derive_warp_screen(r['sci_dir'], r['date'], lst, band, screen_cache)   # MAX_UNDERIVABLE per stack, recorded -
                if v in TAPER_OF:                     # and a night whose dewarped dirty misses the alignment test (Gregg 2026-11-04 "All
                    src_ = p1file(S.PHASE1_ROOT, lst, band, pvar, r, 'dirty')    # approved": 13h 78 MHz Robust 0 2025-05-06, 90th pct 3.7 px)
                    d_ = peak_offsets(apply_warp(np.squeeze(fits.getdata(src_, memmap=False)).astype(np.float64), wx_, wy_),
                                      rd(night_input(r, 'image')))
                    if not offsets_pass(d_):
                        t_ = offsets_summary(d_)
                        raise WarpRefusal(f'{r["date"]} {lst} {band} {v}: the dewarped dirty image does not line up with the Phase 2 '
                                          f'image (n {t_["n"]}, median {t_["median_px"]} px, 90th pct {t_["p90_px"]} px)')
            except RuntimeError as e:                 # as production: any refusal of the derivation or the test drops the night
                nodrv[r['date']] = str(e)[:300]
    if len(nodrv) > S.MAX_UNDERIVABLE:
        raise StackRefusal(f'{lst} {band} {v}: {len(nodrv)} nights that cannot be put on the master\'s frame {sorted(nodrv)} - REFUSING '
                           f'(at most {S.MAX_UNDERIVABLE} night per stack ruled, Gregg 2026-11-03 / 11-04)', 'HG4')
    rows_d = [r for r in rows_d if r['date'] not in nodrv]
    rec['nights_dropped_no_screen'] = nodrv
    rec['dirty_nights_dropped'] = sorted(r['date'] for r in rows if r not in rows_d)
    if len(rows_d) < max(2, len(rows) - S.MIN_NIGHTS_DROP):
        raise StackRefusal(f'{lst} {band} {v}: Phase 1 dirty images missing for {rec["dirty_nights_dropped"]} - REFUSING', 'HG4')
    if rec['dirty_nights_dropped']:
        rec['span'] = constituent_span(lst, band, rows_d, vk, read)
        rec['nights_stacked'] = [r['date'] for r in rows_d]
    mh = fits.getheader(mfile)
    srt = sorted(rows_d, key=lambda r: r['date'])
    if taper:                                         # hybrid input (Gregg 2026-10-28, d): the Taper master's CLEAN residual, same chain
        write_atomic(f'{workdir}/{v}_residual_stack.fits', chain('residual', rows_d).astype(np.float32),
                     diag_header(mh, 'stack of the Phase 1 Taper CLEAN residual images'))
        log(lst, band, v, 'residual_stack')
        rec['derived_screens'] = derived
        _dump(rec, f'{workdir}/{v}_stack.json')
        return rec
    for name, rr in (('dirty_stack', rows_d), ('dirty_halfA', srt[0::2]), ('dirty_halfB', srt[1::2])):
        write_atomic(f'{workdir}/{v}_{name}.fits', chain('dirty', rr).astype(np.float32),
                     diag_header(mh, f'{name} of the Phase 1 dirty images'))
        log(lst, band, v, name)
    P = np.zeros(shape)
    wsum = 0.0
    for r in rows_d:
        src = p1file(S.PHASE1_ROOT, lst, band, pvar, r, 'psf')
        read.add(src)
        a = np.nan_to_num(rd(src))
        if a.shape != shape:
            y_in, x_in = CFP.precompute_regrid_mapping(src, tmpl, shape)
            a = map_coordinates(a, [y_in, x_in], order=1, mode='constant', cval=0.0)
        P += float(r['weight']) * a / np.max(a)
        wsum += float(r['weight'])
    P /= wsum
    P /= P.max()
    write_atomic(f'{workdir}/{v}_psf_stack.fits', P.astype(np.float32), diag_header(mh, 'weighted mean Phase 1 PSF, peak 1'))
    ind = inputs_dir(workdir)
    os.makedirs(ind, exist_ok=True)
    rec['inputs'] = {}
    for src in input_copies(S.COADD_SB, lst, band, v):
        dst = f'{ind}/{os.path.basename(src)}'
        want = md5(src)                               # the cell's two weightings share inputs/: temp copy + atomic rename
        read.add(src)
        if not (os.path.exists(dst) and md5(dst) == want):
            tmp = f'{dst}.{v}.part'
            shutil.copy2(src, tmp)
            if md5(tmp) != want:
                raise StackRefusal(f'{tmp}: copy differs from {src} - REFUSING', 'HG5')
            os.replace(tmp, dst)
        rec['inputs'][os.path.basename(src)] = dict(installed=src, md5=want)
    rec['derived_screens'] = derived
    _dump(rec, f'{workdir}/{v}_stack.json')
    return rec


def _dump(rec, path):
    tmp = f'{path}.part'
    with open(tmp, 'w') as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, path)


def test_stack_rebuild(stack_img, master, wd, cf, rows, admit_hdr=None):
    """HG5. The production chain on the nights' Phase 2 images must rebuild the installed master to floating-point level: max |diff|
    below max(REBUILD_MAX_ABS_JY, REBUILD_MAX_ABS_PEAK_FRAC x the master's peak), rms below max(REBUILD_RMS_JY, REBUILD_RMS_FRAC x
    the master's rms), identical NaN masks. Cells whose flows were recomputed after the build (every flow file more than
    FLOW_RECOVERED_AGE_S newer than weights.csv; R-GREGG-20261026-FLOW-RECOVERED-CELLS-...) pass instead when the differences are
    those of recomputed alignment, not of a mapping error: global ratio within FLOW_RECOVERED_RATIO_TOL of 1, median |difference|
    below FLOW_RECOVERED_MEDIAN_FRAC of the master's rms, and the largest difference after the ratio below
    FLOW_RECOVERED_MAX_PEAK_FRAC of the master peak. With `admit_hdr` (a high-band NoTaper stack, hybrid_stack.ADMISSION
    'hybrid-scales'), a flow-recovered cell missing those limits is admitted at hybrid scales (R-GREGG-20261031-ADMIT-HB-FLOW-
    RECOVERED-NOTAPER-STACKS-AS-HYBRIDS; the comparison with the half-stack noise is made by the recon stage). Anything else raises
    StackRefusal (HG5). Returns the result dict with its verdict. (All limits: config hybrid_stack.)"""
    from exopipe.phase3.config import CFG
    S = CFG.hybrid_stack
    fin = np.isfinite(master) & np.isfinite(stack_img)
    d = stack_img[fin] - master[fin]
    res = dict(max_abs_diff=float(np.max(np.abs(d))), rms_diff=float(np.sqrt(np.mean(d ** 2))), master_peak=float(np.nanmax(master)),
               master_rms=float(np.std(master[fin])), nan_masks_equal=bool(np.array_equal(np.isfinite(master), np.isfinite(stack_img))))
    if (res['max_abs_diff'] < max(S.REBUILD_MAX_ABS_JY, S.REBUILD_MAX_ABS_PEAK_FRAC * res['master_peak'])
            and res['rms_diff'] < max(S.REBUILD_RMS_JY, S.REBUILD_RMS_FRAC * res['master_rms']) and res['nan_masks_equal']):
        res['verdict'] = 'pass (exact)'
        return res
    fl = [os.path.getmtime(f'{wd}/flow/{r["date"]}_scaled_flow.npy') for r in rows if (r.get('qa3_guard') or '') != 'reference-identity']
    recovered = bool(fl) and min(fl) > os.path.getmtime(f'{cf}/weights.csv') + S.FLOW_RECOVERED_AGE_S
    ratio = float(np.sum(master[fin] * stack_img[fin]) / np.sum(stack_img[fin] ** 2))
    mx = float(np.max(np.abs(master[fin] - ratio * stack_img[fin])))
    res.update(flows_recomputed_after_build=recovered, global_ratio=ratio, max_after_ratio=mx, median_abs_diff=float(np.median(np.abs(d))))
    if (recovered and abs(ratio - 1) < S.FLOW_RECOVERED_RATIO_TOL and res['median_abs_diff'] < S.FLOW_RECOVERED_MEDIAN_FRAC * res['master_rms']
            and mx < S.FLOW_RECOVERED_MAX_PEAK_FRAC * res['master_peak'] and res['nan_masks_equal']):
        res['verdict'] = 'pass (flow-recovered cell)'
        return res
    if admit_hdr is not None and recovered and res['nan_masks_equal']:
        # HYBRID-SCALE ADMISSION (Gregg 2026-10-31, "Admit them as hybrids"): a high-band NoTaper stack of a flow-recovered cell whose
        # rebuild misses the limits above - sub-pixel offsets at the brightest compact sources (Cygnus A, Cassiopeia A, Taurus A, 3C 273,
        # Tycho SNR: 4-72 Jy/beam) - is admitted when its mismatch through the hybrid filter G (the only scales a hybrid takes from the
        # reconstruction) has an rms over the sky interior below the unit's half-stack noise at 1 deg: recorded here, the comparison
        # made by the recon stage, which computes that noise (refuses if not).
        ld = lowpass(np.where(fin, stack_img - master, 0.0), abs(float(admit_hdr['CDELT1'])))
        inner = sky_interior(fin, admit_hdr)
        res['hybrid_scale_admission'] = dict(lowpassed_rms=float(np.std(ld[inner])), lowpassed_max_abs=float(np.max(np.abs(ld[inner]))),
                                             rule='rms of G (x) (stack - master) over the sky interior < half_stack_noise_1deg (checked in recon)')
        res['verdict'] = 'admitted at hybrid scales (flow-recovered cell; Gregg 2026-10-31)'
        return res
    raise StackRefusal(f'test_stack_rebuild FAILED: {res} - REFUSING', 'HG5')


VERDICTS = ('pass (exact)', 'pass (flow-recovered cell)', 'admitted at hybrid scales (flow-recovered cell; Gregg 2026-10-31)')
