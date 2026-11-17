#!/usr/bin/env python3
"""candinv/snapshots.py — T5: 10-second snapshot lightcurve (gated).

Warranted ONLY when the 10-min lightcurve shows sub-10-min structure
(Gregg 2026-06-26). Resolves burst duration / sub-structure by forced
photometry on the 10-second Phase-1 snapshots, which are NOT dewarped — we
apply the per-band warp screen first (same machinery as Phase 2).

SAFETY: the Phase-1 snapshots are fpack-compressed `.fits.fs` in the archive,
and the pipeline's _funpack renames+deletes the original. So we COPY each
snapshot to a scratch workdir and funpack the COPY — the archive is untouched.
funpack lives in the `development` conda env.

FRAMES AND EXCLUSIONS (snapshot chat 2026-11-14, JUDGED-20261114-FOLLOWUP-SNAPSHOT-CUTOUTS-SURVEY-ROOT-ONLY): the frames of a
(LST, night) come from the search's own frame root (core/inputs.frame_root: the GPU pipeline's tree for a cell of
paths.SNAPSHOT_GPU_FRAME_CELLS, the survey's Phase-1 tree otherwise), and a subband-night the snapshot tiers exclude
(core/exclusions.PerNight: 'snapshot-gpu' on a GPU-frame cell, 'snapshot' elsewhere) is refused (``snap_refusal``), never read
from another tree. Production's snapshots.py read the survey tree only and no exclusion.

WARP RECORD (the snapshot chat's review of PR 70, 2026-11-14): the warp screen is applied to a frame only when the frame lies on
the screen's own grid - the same shape, pixel scale (CDELT) and reference pixel (CRPIX) as the screen's SCREEN_X WCS (``_warp``).
The survey's frames (4096 x 4096 at 0.03125 deg) never do, the GPU pipeline's (2358 x 2358 at 0.049 deg) do. ``dewarped`` is
what was done to each frame: a saved burst frame carries it (name and header) only when warped; a spectrum row and a light
curve are dewarped only when every frame read was warped (the light curve also counts them). Production stamped 'dewarped'
whenever a screen existed, warped or not.
"""
import glob
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from . import config as C
from . import imaging as I
from . import resolve as R
from . import photometry as PHOT
from exopipe.phase2.cascade_dewarp import apply_warp

FPACK_ENV = 'development'
# Timestamp after 'Snapshot-' regardless of any Stokes suffix (run conventions
# vary: '-V-image' / '-I-image' on some runs, bare '-image' on others).
_TS = re.compile(r'Snapshot-(\d{8}_\d{6})-')


def snap_refusal(lst, date, scope):
    """Why T5 may not read the 10-s frames of (lst, date, scope), or None: the subband-night is excluded for the snapshot
    tiers (data_decisions.csv read per night as the search reads it, core/exclusions.PerNight). Exclusion is total: such a
    subband-night is refused, not read from another frame tree."""
    from exopipe.phase3.core import exclusions as EX
    if EX.PerNight().is_excluded(lst, scope, date):
        return (f'refused: {lst} {date} {scope} is excluded for the snapshot tiers (data_decisions.csv, scope '
                f'{"snapshot-gpu" if (lst, date) in EX.gpu_frame_cells() else "snapshot"})')
    return None


def _phase1_snap_dir(lst, date, scope, clean=False):
    """snapshots/ (dirty V) or snapshots_clean/ (CLEANed Stokes I) under the night's frame root (core/inputs.frame_root),
    None when the subband-night is excluded (snap_refusal) or has no such directory. Prefer a run
    whose subdir actually CONTAINS frames — a date can have multiple Run_* dirs
    where an earlier one created an EMPTY snapshots/ and a later one holds the
    frames (caused several candidates to be wrongly marked 'no snapshot data')."""
    from exopipe.phase3.core import inputs as IN
    if snap_refusal(lst, date, scope):
        return None
    sub = 'snapshots_clean' if clean else 'snapshots'
    first = None
    for run in sorted(glob.glob(os.path.join(
            IN.frame_root(lst, date), lst, date, 'Run_*'))):
        d = os.path.join(run, scope, sub)
        if os.path.isdir(d):
            if first is None:
                first = d
            if (glob.glob(os.path.join(d, '*image.fits.fs'))
                    or glob.glob(os.path.join(d, '*image.fits.fz'))):
                return d
    return first


def _snap_glob(snapdir, stokes):
    """Frame files for a Stokes, handling both naming conventions:
    Stokes I lives in snapshots_clean as '*-Clean-Snapshot-*-image';
    V is in snapshots as '*-V-image' (some runs) or bare '*-image' (others).
    Excludes psf / horizon-mask."""
    if stokes == 'I':
        pats = ['*-Clean-Snapshot-*-image.fits.fs']
    else:
        pats = [f'*-{stokes}-image.fits.fs', '*-image.fits.fs']
    pats = [q for p_ in pats for q in (p_, p_[:-3] + '.fz')]   # .fs + .fz archives
    for pat in pats:
        hits = [f for f in sorted(glob.glob(os.path.join(snapdir, pat)))
                if 'psf' not in f and 'horizon' not in f and 'mask' not in f]
        if hits:
            return hits
    return []


def _frame_path(lst, date, scope, stokes, timestamp):
    """One snapshot .fits.fs for (scope, stokes, timestamp), using the right dir
    (snapshots_clean for I, snapshots for V) and BOTH naming conventions: some
    runs tag the Stokes ('*-V-image' / '*-I-image'), others are bare
    ('*-Pilot-Snapshot-{ts}-image' for V, '*-Clean-Snapshot-{ts}-image' for I)."""
    clean = (stokes == 'I')
    sd = _phase1_snap_dir(lst, date, scope, clean=clean)
    if sd is None:
        return None
    if clean:
        pats = [f'*Clean-Snapshot-{timestamp}-image.fits.fs',
                f'*Snapshot-{timestamp}-I-image.fits.fs']
    else:
        pats = [f'*Snapshot-{timestamp}-V-image.fits.fs',
                f'*Pilot-Snapshot-{timestamp}-image.fits.fs',
                f'*Snapshot-{timestamp}-image.fits.fs']
    pats = [q for p_ in pats for q in (p_, p_[:-3] + '.fz')]   # .fs + .fz archives
    for pat in pats:
        hits = [f for f in sorted(glob.glob(os.path.join(sd, pat)))
                if all(t not in f for t in ('psf', 'horizon', 'mask'))
                and (clean or 'Clean' not in f)]
        if hits:
            return hits[0]
    return None


def has_frames(lst, date, scope):
    """True if this scope has any Phase-1 V snapshot frames (in a populated run)."""
    d = _phase1_snap_dir(lst, date, scope)
    return bool(d and _snap_glob(d, 'V'))


def _snap_time(path):
    m = _TS.search(os.path.basename(path))
    if not m:
        return None
    return Time(f'{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}T'
                f'{m.group(1)[9:11]}:{m.group(1)[11:13]}:{m.group(1)[13:15]}')


def _grid(w):
    """(CDELT1, CDELT2, CRPIX1, CRPIX2) of a celestial WCS."""
    return tuple(float(v) for v in (*w.wcs.get_cdelt()[:2], *w.wcs.crpix[:2]))


def _load_warp(scope, lst, date):
    """(SCREEN_X, SCREEN_Y, grid) of the subband-night's Phase-2 warp screen: grid = the screen's own pixel grid (_grid of
    SCREEN_X's WCS), None when that header has no celestial WCS; (None, None, None) without a screen."""
    p = R.warp_screen(lst, date, scope)
    if p is None:
        return None, None, None
    with fits.open(p) as h:
        hx = h['SCREEN_X'].header
        grid = _grid(WCS(hx).celestial) if hx.get('CTYPE1') and hx.get('CTYPE2') else None
        return h['SCREEN_X'].data.astype(float), h['SCREEN_Y'].data.astype(float), grid


def _warp(data, wcs, scr):
    """(data, applied): the warp screen ``scr`` (_load_warp) applied to a frame only when the frame lies on the screen's own
    grid - the same shape, CDELT and CRPIX; otherwise the frame as read and False (module docstring, WARP RECORD)."""
    sx, sy, grid = scr
    if sx is None or grid is None or data.shape != sx.shape:
        return data, False
    if not np.allclose(_grid(wcs), grid, rtol=1e-6, atol=1e-9):
        return data, False
    out = apply_warp(data, sx, sy)
    return (data, False) if out is None else (out, True)


def save_burst_allsky(cand, lst, date, scope, ra, dec, frame_times, outdir,
                      stokes=('V', 'I'), verbose=True):
    """Write conformant all-sky snapshot FITS for specific 10-sec frames (e.g. the burst spikes) so they can be inspected
    full-frame; a frame on the warp screen's grid is dewarped, and only then named and stamped so (WARP RECORD).

    frame_times : iterable of 'YYYYMMDD_HHMMSS' tokens.
    Returns list of written paths.
    """
    from . import io_products as IO
    from exopipe.phase3.core.products import build_name
    if _phase1_snap_dir(lst, date, scope) is None:      # excluded (snap_refusal) or no frames
        return []
    scr = _load_warp(scope, lst, date)
    tmp = tempfile.mkdtemp(prefix='candinv_snapsave_', dir=C.SCRATCH_DIR)
    written = []
    try:
        for ts in frame_times:
            for stok in stokes:
                src = _frame_path(lst, date, scope, stok, ts)
                if not src:
                    continue
                dst = os.path.join(tmp, os.path.basename(src)[:-3] + '.fz')
                fp = _copy_funpack(src, dst)
                if fp is None:
                    continue
                data, hdr, wcs = I.load_image(fp)
                data, warped = _warp(data, wcs, scr)
                proc = ['dewarped'] if warped else []
                t = _snap_time(src)
                name = build_name(scope=scope, stokes=stok, cadence='snap',
                                  taper='Taper', robust=0.0, proc=proc,
                                  lst=lst, epoch=t, target=cand.cand_id)
                out = os.path.join(outdir, name)
                nh = wcs.to_header()
                for k in ('BMAJ', 'BMIN', 'BPA', 'BUNIT'):
                    if k in hdr:
                        nh[k] = hdr[k]
                fits.writeto(out, data.astype(np.float32), nh, overwrite=True)
                beam = ((hdr['BMAJ'], hdr['BMIN'], hdr.get('BPA', 0.0))
                        if 'BMAJ' in hdr else None)
                IO._stamp(out, cand, scope, stok, proc, t.isot,
                          (t + 10.0 / 86400.0).isot, 10.0, beam, None,
                          dewarped=warped)
                written.append(out)
                if verbose:
                    print('  saved', os.path.basename(out))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return written


def snapshot_spectrum(lst, date, ra, dec, timestamp, scopes=None,
                      stokes=('V', 'I'), fallback_bpix=6.0, verbose=True):
    """Instantaneous (single 10-sec frame) spectrum of a burst across subbands.

    For a 10-second burst the deep/10-min spectrum is diluted — the true
    narrowband-vs-broadband test is the snapshot itself. For each subband:
    copy+funpack the {timestamp} frame, dewarp (its own warp screen), imfit at
    the fixed position. timestamp = 'YYYYMMDD_HHMMSS'. Returns list of dicts.
    """
    scopes = scopes or C.SUBBANDS
    tmp = tempfile.mkdtemp(prefix='candinv_snapspec_', dir=C.SCRATCH_DIR)
    rows = []
    try:
        for sb in scopes:
            why = snap_refusal(lst, date, sb)
            if why:                                    # listed, so the spectrum shows the subband was refused
                rows.append({'scope': sb, 'freq_mhz': (C.scope_center_hz(sb) or 0) / 1e6, 'refused': why})
                continue
            if _phase1_snap_dir(lst, date, sb) is None:
                continue
            scr = _load_warp(sb, lst, date)
            row = {'scope': sb, 'freq_mhz': (C.scope_center_hz(sb) or 0) / 1e6}
            warped = []
            # snapshots are not perfectly dewarped per-frame -> search the wide
            # snapshot box (6'x74/nu); supply a nominal beam so imfit RUNS (finds
            # the source in the box and returns flux +/- err), not a fixed-pixel
            # matched filter.
            fmhz = row['freq_mhz'] or 74.0
            radius = float(np.clip(6.0 * 74.0 / fmhz, 2.0, 30.0))
            for stok in stokes:
                dw, hdr, w = _funpack_dewarp(lst, date, sb, scr, timestamp,
                                             stok, tmp)
                if dw is None:
                    continue
                warped.append(w)
                try:
                    cd = abs(float(hdr.get('CDELT2', hdr.get('CD2_2', 0.04)) or 0.04))
                    nb = (3.5 * cd, 3.5 * cd, 0.0)
                    m = PHOT.forced_imfit(dw, ra, dec, beam=nb,
                                          fallback_bpix=fallback_bpix,
                                          box_half_arcmin=radius)
                    if m:
                        row[f'{stok}_peak'] = m['peak']
                        row[f'{stok}_err'] = m.get('err')
                        row[f'{stok}_snr'] = m['snr']
                        row[f'{stok}_sign'] = m['sign']
                except Exception:
                    pass
            row['dewarped'] = bool(warped) and all(warped)          # every frame read was warped (WARP RECORD)
            if row.get('I_peak'):
                row['vi_ratio'] = abs(row.get('V_peak', 0)) / abs(row['I_peak'])
            rows.append(row)
            if verbose:
                print('  %-6s %5.1f MHz  V_snr=%6s  I_snr=%6s  V/I=%s' % (
                    sb, row['freq_mhz'],
                    ('%.1f' % row['V_snr']) if 'V_snr' in row else '-',
                    ('%.1f' % row['I_snr']) if 'I_snr' in row else '-',
                    ('%.2f' % row['vi_ratio']) if 'vi_ratio' in row else '-'))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rows


def _copy_funpack(src, dst, retries=3):
    """Copy a `.fits.fs` to scratch `dst` (`.fz`) and funpack it, ROBUST to the
    concurrent-IO truncation that made a single batch funpack fail with CFITSIO
    read/EOF errors (107/108) under parallel workers. Verifies the copy size,
    funpacks with -F (overwrite), and retries with a fresh re-copy. Returns the
    funpacked `.fits` path or None. The archive `src` is never mutated."""
    fp = dst[:-3] if dst.endswith('.fz') else dst + '.fits'
    try:
        ssz = os.path.getsize(src)
    except OSError:
        return None
    for _ in range(retries):
        try:
            shutil.copyfile(src, dst)
            if os.path.getsize(dst) != ssz:          # truncated copy -> retry
                continue
            r = subprocess.run(['conda', 'run', '-n', FPACK_ENV, 'funpack',
                                '-F', dst], capture_output=True, timeout=180)
            if r.returncode == 0 and os.path.exists(fp):
                return fp
        except Exception:
            pass
        try:
            if os.path.exists(fp):
                os.remove(fp)
        except OSError:
            pass
    return None


def _funpack_dewarp(lst, date, scope, scr, timestamp, stokes, tmp):
    """Copy->funpack->dewarp (_warp: only on the screen's grid) one snapshot frame; return (2D FITS path, hdr, warped) or
    (None, None, False). Archive untouched (works on a scratch copy)."""
    src = _frame_path(lst, date, scope, stokes, timestamp)
    if not src:
        return None, None, False
    dst = os.path.join(tmp, f'{scope}_{stokes}.fits.fz')
    fp = _copy_funpack(src, dst)
    if fp is None:
        return None, None, False
    data, hdr, wcs = I.load_image(fp)
    data, warped = _warp(data, wcs, scr)
    nh = wcs.to_header()
    for k in ('BUNIT',):
        if k in hdr:
            nh[k] = hdr[k]
    dw = fp + '.dw.fits'
    fits.writeto(dw, data.astype(np.float32), nh, overwrite=True)
    return dw, hdr, warped


def snapshot_morphology(lst, date, scope, ra, dec, timestamp, verbose=True):
    """Point-source test on the PEAK 10-second frame (Gregg's decider, the test
    that killed rank-48/rank-28): imfit the V (and clean-I) source vs the
    EMPIRICAL beam from unresolved Stokes-I sources in the SAME snapshot frame.

    Snapshot frames carry no restoring beam (BMAJ=0) and have the large
    instantaneous-uv PSF, so we bootstrap imfit with a nominal beam from the
    pixel scale; the comparison uses observed (convolved) FWHMs, which are
    measured from the image and so are robust to that guess. Larger/elongated
    than the empirical PSF => RESOLVED => propagation/lensing artifact.
    """
    out = {'scope': scope, 'timestamp': timestamp}
    why = snap_refusal(lst, date, scope)
    if why:
        out['note'] = out['refused'] = why
        return out
    if _phase1_snap_dir(lst, date, scope, clean=False) is None:
        out['note'] = 'no snapshot dir'
        return out
    scr = _load_warp(scope, lst, date)
    tmp = tempfile.mkdtemp(prefix='candinv_snapmorph_', dir=C.SCRATCH_DIR)
    try:
        vdw, vhdr, vw = _funpack_dewarp(lst, date, scope, scr, timestamp, 'V', tmp)
        idw, ihdr, iw = _funpack_dewarp(lst, date, scope, scr, timestamp, 'I', tmp)
        read = [w for d, w in ((vdw, vw), (idw, iw)) if d]
        out['dewarped'] = bool(read) and all(read)              # every frame read was warped (WARP RECORD)
        # nominal beam from pixel scale (~3.5 px FWHM) to let imfit run
        ref = ihdr or vhdr
        if ref is None:
            out['note'] = 'frame not found'
            return out
        cd = abs(float(ref.get('CDELT2', ref.get('CD2_2', 0.037))))
        nb = (3.5 * cd, 3.5 * cd, 0.0)
        # empirical PSF from unresolved I sources in the clean-I snapshot
        eb = (PHOT.empirical_beam_imfit(idw, exclude_radec=(ra, dec), beam=nb,
                                        fallback_bpix=3.5) if idw else None)
        out['ebeam_maj_deg'] = eb[0] if eb else None
        out['ebeam_min_deg'] = eb[1] if eb else None
        out['ebeam_n_src'] = eb[3] if eb else 0
        for stok, dw in (('v', vdw), ('i', idw)):
            if not dw:
                continue
            m = PHOT.forced_imfit(dw, ra, dec, beam=nb, fallback_bpix=3.5)
            if m and m.get('converged'):
                out[f'{stok}_fwhm_maj_deg'] = m['fwhm_maj_deg']
                out[f'{stok}_fwhm_min_deg'] = m['fwhm_min_deg']
                out[f'{stok}_peak'] = m['peak']
                out[f'{stok}_snr'] = m['snr']
                if eb:
                    out[f'{stok}_maj_ratio'] = m['fwhm_maj_deg'] / max(eb[0], 1e-9)
                    out[f'{stok}_min_ratio'] = m['fwhm_min_deg'] / max(eb[1], 1e-9)
        # verdict from V (fall back to I) major/minor ratio vs empirical PSF
        mr = out.get('v_maj_ratio', out.get('i_maj_ratio'))
        nr = out.get('v_min_ratio', out.get('i_min_ratio'))
        if mr is not None and nr is not None:
            out['point_like'] = bool(mr < C.MORPH_POINT_RATIO and
                                     nr < C.MORPH_POINT_RATIO + 0.05)
            out['extended'] = bool(mr >= C.MORPH_RESOLVED_RATIO or
                                   nr >= C.MORPH_RESOLVED_RATIO)
        if verbose:
            print(f'  T5 snapshot morphology [{scope} @ {timestamp}]: '
                  f'point_like={out.get("point_like")} '
                  f'v_maj_ratio={out.get("v_maj_ratio")} '
                  f'(ebeam n={out.get("ebeam_n_src")})')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def snapshot_lightcurve(lst, date, scope, ra, dec, start_iso, end_iso,
                        pad_s=60.0, workdir=None, fallback_bpix=6.0,
                        verbose=True, stokes='V'):
    """10-second {stokes} lightcurve at (ra,dec) over [start-pad, end+pad].

    Returns dict with the per-frame series and a summary. Cleans up scratch.
    """
    out = {'scope': scope, 'date': date, 'stokes': stokes,
           'available': False, 'frames': []}
    # 10-second snapshots are NOT dewarped per-frame (the warp screen is the
    # deep/10-min-averaged solution -> per-frame residual remains), so the
    # forced-photometry search radius is the SNAPSHOT radius 6'x74/nu, not the
    # 2' dewarped value (Gregg 2026-06-28).
    fmhz = (C.scope_center_hz(scope) or 74e6) / 1e6
    search_radius = float(np.clip(6.0 * 74.0 / fmhz, 2.0, 30.0))
    out['search_radius_arcmin'] = search_radius
    why = snap_refusal(lst, date, scope)
    if why:
        out['note'] = out['refused'] = why
        return out
    snapdir = _phase1_snap_dir(lst, date, scope, clean=(stokes == 'I'))
    if snapdir is None:
        out['note'] = 'no Phase-1 snapshot dir'
        return out
    # Snapshots are NOT dewarped per-frame; apply the per-band warp screen when
    # its grid matches, else proceed un-dewarped (the wide 6'x74/nu search radius
    # absorbs the residual offset). Always run (Gregg 2026-06-28). 'dewarped':
    # every frame read was warped; n_frames_dewarped counts them (WARP RECORD).
    scr = _load_warp(scope, lst, date)
    out['dewarped'] = False
    t0, t1 = Time(start_iso) - pad_s / 86400.0, Time(end_iso) + pad_s / 86400.0
    snaps = []
    for f in _snap_glob(snapdir, stokes):
        t = _snap_time(f)
        if t is not None and t0 <= t <= t1:
            snaps.append((t, f))
    if not snaps:
        out['note'] = f'no {stokes} snapshots in the window'
        return out
    out['available'] = True
    out['n_frames'] = len(snaps)

    tmp = workdir or tempfile.mkdtemp(prefix='candinv_snap_', dir=C.SCRATCH_DIR)
    try:
        # copy+funpack each snapshot PER-FILE (archive untouched), robust to
        # concurrent-IO truncation: a few unreadable frames are skipped rather
        # than aborting the whole lightcurve (the old batch funpack was
        # all-or-nothing and silently killed the snapshot under parallel load).
        local = []
        for t, f in snaps:
            dst = os.path.join(tmp, os.path.basename(f)[:-3] + '.fz')
            fp = _copy_funpack(f, dst)
            if fp:
                local.append((t, fp))
        out['n_frames_read'] = len(local)
        series = []
        n_loaded = n_warped = 0
        for t, fp in local:
            try:
                data, hdr, wcs = I.load_image(fp)
                n_loaded += 1
                data, warped = _warp(data, wcs, scr)
                n_warped += warped
                # write the DEWARPED frame (2D + WCS + beam) so imfit measures
                # it on a cutout — same primitive as everywhere else.
                nh = wcs.to_header()
                for k in ('BMAJ', 'BMIN', 'BPA', 'BUNIT'):
                    if k in hdr:
                        nh[k] = hdr[k]
                dw = fp + '.dw.fits'
                fits.writeto(dw, data.astype(np.float32), nh, overwrite=True)
                m = PHOT.forced_imfit(dw, ra, dec, fallback_bpix=fallback_bpix,
                                      box_half_arcmin=search_radius)
                os.remove(dw)
                if m is None:
                    continue
                series.append({'time': t.isot, 'mjd': float(t.mjd),
                               'peak': m['peak'], 'snr': m['snr'],
                               'sign': m['sign']})
            except Exception:
                continue
        out['n_frames_dewarped'] = n_warped
        out['dewarped'] = bool(n_loaded) and n_warped == n_loaded
        series.sort(key=lambda r: r['mjd'])
        out['frames'] = series
        snrs = [s['snr'] for s in series if np.isfinite(s['snr'])]
        if snrs:
            out['peak_snr'] = max(snrs)
            ipk = int(np.argmax(snrs))
            out['peak_time'] = series[ipk]['time']
            out['peak_sign'] = series[ipk]['sign']
            ndet = sum(1 for s in snrs if s >= C.STRONG_SNR)
            out['n_frames_gt7'] = ndet
            # burst duration: contiguous frames above DETECT around the peak
            det = [s['snr'] >= C.DETECT_SNR for s in series]
            lo = hi = ipk
            while lo > 0 and det[lo - 1]:
                lo -= 1
            while hi < len(det) - 1 and det[hi + 1]:
                hi += 1
            out['burst_dur_s'] = round(
                (series[hi]['mjd'] - series[lo]['mjd']) * 86400.0 + 10.0, 1)
            out['burst_nframes'] = hi - lo + 1
        if verbose:
            print(f'  T5 snapshot [{scope}]: {len(series)} frames, '
                  f'peak {out.get("peak_snr",0):.1f} at {out.get("peak_time")}, '
                  f'burst ~{out.get("burst_dur_s")}s ({out.get("burst_nframes")} frames)')
    finally:
        if workdir is None:
            shutil.rmtree(tmp, ignore_errors=True)
    return out
