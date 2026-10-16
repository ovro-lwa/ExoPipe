"""D-ADMIT (a), slice 4 (plan/wave2/DADMIT_DESIGN.md): the admitted night's products in the installed generation.

So far: shflux.bake - the body of shflux.apply's loop as one function (an admitted night's products are baked with it) -
writes exactly what the loop wrote: the old body is pinned here and both are run on a real product's header with synthetic
pixels, compared byte for byte. apply itself is covered by the parity chains (the fluxscale stage's gen_shflux tree, P5).
Run: python tests/test_masters_admit4.py
"""
import filecmp
import os
import shutil
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from exopipe.phase3.core import env as ENV                          # noqa: E402
ENV.legacy()
from exopipe.phase3.config import CFG                               # noqa: E402

T = '/tmp/exopipe3_tests_admit4'
FAILS, N = [], [0]


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({detail})'))
    if not ok:
        FAILS.append(name)


def _old_bake_body(d, hd, scope, row, corr_t):
    """shflux.apply's loop body as it was before shflux.bake (2026-10-15), pinned here."""
    from exopipe.phase3.lib.masters import flux_correction as FC
    scale = float(row['scale']); corr = float(corr_t[scope]['corr'])
    d = (np.asarray(d, dtype=np.float32) * np.float32(scale * corr))
    for _c, _v, _com in FC.cards_for(scope, corr_t, baked=True):
        hd[_c] = (_v, _com)
    _vlssr = str(row['srcs']).startswith('VLSSr')
    hd['SHSCALE'] = (scale, 'flux scale applied (VLSSr/RCB per LST, 2026-09-07)' if _vlssr
                     else 'SH12 absolute flux scale applied (Gregg 2026-08-28)')
    hd['SHERR'] = (float(row['err']), '1-sigma uncertainty of SHSCALE')
    hd['SHNSRC'] = (int(row['nsrc']), 'VLSSr matches in the window' if _vlssr else 'SH calibrators used (el > 50 deg)')
    hd['SHSRCS'] = (row['srcs'], 'scale reference (window)' if _vlssr else 'SH calibrators used')
    hd['SHMETHOD'] = ('VLSSr-dec20-60' if _vlssr else 'SH12-calibrators', 'flux scale method')
    hd['SHFREQ'] = (float(row['freq_hz']), f"model frequency Hz ({row['freq_src'][:40]})")
    hd['SHDATE'] = (row['measured_at'], 'flux scale measured')
    hd['HISTORY'] = (f"shflux: pixels x {scale:.5f} (VLSSr/RCB median ratio per LST, {row['srcs']}, N {row['nsrc']}; R-FLUX-SCALE-APPLY-VLSSR-HIGH-ELEVATION-APPROVED 2026-09-07)" if _vlssr else
                     f"shflux: pixels x {scale:.5f} (SH12 calibrators {row['srcs']}; I Taper R-0.75 pbcorr master; stage A of MASTER_SHFLUX_VLSSR_DEWARP_SPEC_20260828)")
    hd['HISTORY'] = f"fluxcorr: pixels also x FLUXCORR {corr:.5f} (k x g, {corr_t[scope]['version'][:24]})"
    return d, hd


def test_bake_equivalence():
    from astropy.io import fits
    from exopipe.phase3.lib.masters import deep_stages as DS, flux_correction as FC, shflux as SH
    lst, sc = '13h', '78MHz'
    row = DS.scale_row(lst, sc)
    corr_t = FC.load_table(CFG.masters_fluxscale.FLUXCORR_TABLE)
    hd0 = fits.getheader(os.path.join(CFG.paths.COADD_SB_ROOT, '13h_78MHz', 'coadd_provisional', 'I_deep_Taper_R075_78MHz.fits'))
    rng = np.random.default_rng(4)
    data = rng.normal(0, 1, (64, 64)).astype(np.float32)
    data[3, 5] = np.nan
    a = _old_bake_body(data, hd0.copy(), sc, row, corr_t)
    b = SH.bake(data, hd0.copy(), sc, row, corr_t)
    os.makedirs(T, exist_ok=True)
    pa, pb = os.path.join(T, 'old.fits'), os.path.join(T, 'new.fits')
    fits.writeto(pa, a[0], a[1], overwrite=True)
    fits.writeto(pb, b[0], b[1], overwrite=True)
    check('shflux.bake = the old loop body, byte for byte (13h 78 MHz scale row, a real header, synthetic pixels)',
          filecmp.cmp(pa, pb, shallow=False) and b[1]['FLUXBAKE'] is True, (row.get('scale'), corr_t.get(sc)))


# ---- slice 4, part 2: final_night / measure_scale2 / subtract_night (run_final_lst_freq's per-night bodies) -------------
# The loops of run_final_lst_freq as they were before the extraction (2026-10-15) and the build's new callers, pinned
# verbatim (generated, not retyped); both run in coadd's own namespace, so a module name means the same object in each and a
# monkeypatch of coadd reaches both.
OLD_STEP1 = 'def _old_step1(target_dates, _fref, workdir, prep_dir, post_dates, lst, fl, ref_date, good_dates_deep, _gate_rows, date_params, i_aligned):\n    from exopipe.phase3.lib.masters import night_image_gate as _NG\n    ref_data, ref_wcs, beam_pix, horizon_mask, cd = _fref[\'data\'], _fref[\'wcs\'], _fref[\'beam_pix\'], _fref[\'horizon_mask\'], _fref[\'cd\']\n    tmpl_hdr, target_shape = _fref[\'tmpl_hdr\'], _fref[\'target_shape\']\n    for date, sci_dir, align_img_path in target_dates:\n        needs_regrid = date < CUTOVER_DATE and bool(post_dates)\n        if not needs_regrid and _off_grid(date, align_img_path, target_shape):    # B-03 (a)\n            needs_regrid = True\n        if needs_regrid:\n            work_path = regrid_image(align_img_path, tmpl_hdr,\n                                      f\'{prep_dir}/{date}_regrid.fits\')\n        else:\n            work_path = align_img_path\n\n        tgt_data, tgt_hdr = load_image(work_path)\n        if tgt_data.shape != target_shape:\n            print(f"    {date}: shape mismatch, skipping")\n            continue\n\n        tgt_mask = ~np.isfinite(tgt_data) | (tgt_data == 0)\n        if tgt_mask.shape != horizon_mask.shape:\n            # wrong-grid epoch (e.g. 19h Full 2025-04-05 raw-failsafe built\n            # on the Green grid — Phase-2 anomaly, problem ledger) — skip\n            print(f"    {date}: WRONG GRID {tgt_mask.shape} vs "\n                  f"{horizon_mask.shape}, skipping", flush=True)\n            continue\n        tgt_obs_time = get_midpoint_time(tgt_hdr, work_path)\n        tgt_rms = get_uniform_rms(tgt_data, tgt_mask)\n\n        scale_map, coeffs, n_src = measure_flux_scale(\n            tgt_data, ref_data, ref_wcs, beam_pix,\n            horizon_mask | tgt_mask, tgt_obs_time)\n\n        mean_scale = float(coeffs[0])\n        weight = float(mean_scale**2 / tgt_rms**2) if tgt_rms > 0 else 0.0\n        # STAGE 3 - the flux-alignment window is a co-add MEMBERSHIP VETO (R-THIRD-STAGE-QA-FLUXALIGN 2026-08-25, "gate refuses,\n        # never warns"; R-QA3-BOUNDS-PROCESS-APPROVED 2026-08-26: 0.7-1.6, lower edge 0.35 at 18 MHz and 0.55 at 23/27 MHz). In this\n        # code since 2026-10-07: the August drops were side-script edits that the 2026-09-10 rebuild lost. The scale tested is this\n        # measurement (before flow alignment), the one the window was set on. The night keeps its per-night products (two-tier).\n        from exopipe.phase3.lib.masters import membership as _MB3\n        _qa3_in = _MB3.qa3_inside(fl, mean_scale)\n        if not _qa3_in:\n            print(f"    {date}: STAGE-3 WINDOW REFUSED - flux-alignment scale {mean_scale:.4g} outside "\n                  f"{_MB3.qa3_window(fl)[0]:g}-{_MB3.qa3_window(fl)[1]:g}; leaves the co-add (products still built)", flush=True)\n            good_dates_deep.discard(date)\n\n        flow_path = f\'{workdir}/flow/{date}_scaled_flow.npy\'\n        flow_offsets = _require_flow(flow_path, date, ref_date, f\'{lst} {fl}\')\n\n        h, w = target_shape\n        yg, xg = np.indices((h, w), dtype=float)\n        scale_map_full = (coeffs[0] + coeffs[1] * (2.0 * xg / w - 1.0)\n                          + coeffs[2] * (2.0 * yg / h - 1.0))\n        # BLANK-FIELD REFUSAL (Gregg 2026-09-02: "QA should refuse blank\n        # fields and report them"). Previously a blank epoch satisfied the\n        # n_src<8 arm below and was handed a flat DC of 1.0, i.e. ACCEPTED.\n        # It then left the co-add only by accident: get_uniform_rms returns\n        # 0.0 for an all-zero frame, so the weight line zeroes it - and\n        # nothing recorded that the epoch had dropped out. That accident\n        # does NOT cover a frame that is finite with no positive pixel: it\n        # has a nonzero rms, hence a nonzero weight, and would enter the\n        # co-add. Refuse explicitly and say so.\n        _npos = int(np.count_nonzero(np.nan_to_num(tgt_data) > 0))\n        _finite = float(np.isfinite(tgt_data).mean())\n        if _npos == 0 or _finite == 0.0:\n            print(f"    {date}: BLANK FIELD REFUSED — {_npos} positive "\n                  f"pixels, {100*_finite:.1f}% finite. Not scaled, not "\n                  f"weighted, NOT added to the co-add (Gregg 2026-09-02).",\n                  flush=True)\n            continue\n        # Flux-degeneracy guard (Gregg 2026-06-21): a degenerate affine fit\n        # (few/clustered sources) ramps 2-8x across the field -> sharp-edge\n        # artifact. If out of physical range, drop the tilt, use robust median DC.\n        _mean = float(coeffs[0]); _tilt = float(np.hypot(coeffs[1], coeffs[2]))\n        # R-TILT-REMOVED-NEXTGEN (Gregg 2026-08-28, landed 2026-09-08): the x/y\n        # tilt is REMOVED from production - flat DC always. P3_FLUXSCALE_TILT=1\n        # restores the affine plane for comparison runs only, never production.\n        _mean_aligned = None\n        _degen = n_src < 8 or not (0.7 <= _mean <= 1.4) or _tilt > 0.3\n        if os.environ.get(\'P3_FLUXSCALE_TILT\') != \'1\' or _degen:\n            # EVERY ADMITTED NIGHT IS CORRECTED BY ITS MEASURED SCALE (Gregg 2026-10-02, "1. yes"); the applied scale is\n            # measured on the FLOW-ALIGNED night at the reference peak pixels (Gregg 2026-10-04, "1 approved":\n            # _night_flux_dc), and a night whose scale cannot be measured is REFUSED.\n            _dc, _mean_aligned = _night_flux_dc(_mean, work_path, flow_offsets, tmpl_hdr, target_shape, ref_data, ref_wcs,\n                                                beam_pix, horizon_mask | tgt_mask, tgt_obs_time,\n                                                f\'{prep_dir}/{date}_scale_remeasure.fits\')\n            if not (np.isfinite(_dc) and _dc > 0):\n                print(f"    {date}: FLUX SCALE NOT MEASURABLE (first {_mean:.4g}, on the flow-aligned image "\n                      f"{_mean_aligned}) - REFUSED, not added to the co-add (Gregg 2026-10-02)", flush=True)\n                # recorded in the night-image gate record, so the search refuses it under its true reason\n                _gate_rows[date] = dict(verdict=\'FAIL\', reason=f\'flux scale not measurable (first {_mean:.4g}, \'\n                                                               f\'flow-aligned {_mean_aligned})\')\n                continue\n            if _mean_aligned is not None:\n                weight = float(_dc ** 2 / tgt_rms ** 2) if tgt_rms > 0 else 0.0\n                print(f"    {date}: applied scale {_dc:.4f} (flow-aligned night at the reference peak pixels; "\n                      f"pre-alignment aperture maximum {_mean:.4g}, recorded only)", flush=True)\n            scale_map_full = np.full((h, w), _dc, dtype=float)\n            print(f"    {date}: flat DC={_dc:.3f} (measured tilt {_tilt:.3f} "\n                  f"dropped per R-TILT-REMOVED-NEXTGEN; n={n_src}"\n                  f"{\'; degenerate\' if _degen else \'\'})")\n\n        regrid_coords = (precompute_regrid_mapping(align_img_path, tmpl_hdr, target_shape)\n                         if needs_regrid else None)\n\n        date_params[date] = {\n            \'scale_map\': scale_map_full, \'flow\': flow_offsets, \'weight\': weight,\n            \'regrid\': regrid_coords, \'sci_dir\': sci_dir,\n            # third-stage QA record (R-THIRD-STAGE-QA-FLUXALIGN 2026-08-25; the window veto is applied above since 2026-10-07 -\n            # \'qa3_guard\' is the 2026-06-21 affine-fit degeneracy label, NOT the stage-3 verdict; \'qa3_window\' is)\n            \'qa3_scale\': _mean, \'qa3_tilt\': _tilt, \'qa3_nsrc\': n_src, \'qa3_window\': \'inside\' if _qa3_in else \'OUTSIDE\',\n            \'qa3_guard\': (\'degenerate-flatDC\' if (n_src < 8 or not\n                          (0.7 <= _mean <= 1.4) or _tilt > 0.3) else \'pass\'),\n            \'qa3_rms\': tgt_rms,\n            # the factor actually applied (Gregg 2026-10-02 flux ruling) and the flow-aligned re-measurement, if made\n            \'flux_dc\': (f\'{float(np.ravel(scale_map_full)[0]):.5f}\' if np.ptp(scale_map_full) == 0 else \'affine\'),\n            \'qa3_scale_aligned\': (\'\' if _mean_aligned is None else f\'{_mean_aligned:.5f}\'),\n        }\n\n        # Two-tier policy (Gregg 2026-08-26): per-epoch products are built\n        # for EVERY non-data_decisions epoch; only co-add membership is\n        # gated by the QA verdict.\n        i_out = f\'{prep_dir}/{date}_I.fits\'\n        process_variant_image(align_img_path, flow_offsets, scale_map_full,\n                              tmpl_hdr, target_shape, i_out, regrid_coords)\n        # NIGHT-IMAGE GATE (Gregg 2026-10-02 "Approved"): ahead of co-add membership AND search admission; the aligned\n        # night against the reference it was aligned to; a FAIL leaves good_dates_deep, so the I and V masters, the\n        # variants, the leave-one-out masters and in_master all follow; the record is read by the search.\n        try:\n            _gate_rows[date] = _NG.evaluate(load_image(i_out)[0], ref_data, cd, night_raw=tgt_data)\n        except Exception as _ge:\n            _gate_rows[date] = dict(verdict=\'FAIL\', reason=f\'night-image gate could not be evaluated ({_ge})\'[:200])\n        if _gate_rows[date][\'verdict\'] == \'FAIL\':\n            print(f"    {date}: NIGHT-IMAGE GATE REFUSED - {_gate_rows[date][\'reason\']}", flush=True)\n            good_dates_deep.discard(date)\n        if date in good_dates_deep:\n            i_aligned.append((date, i_out, weight))\n            flow_str = \'flow\' if flow_offsets is not None else \'no-flow\'\n            print(f"    {date}: scale={mean_scale:.3f} ({n_src} src) "\n                  f"w={weight:.3f} {flow_str} ✓")\n        else:\n            print(f"    {date}: coadd-omitted (QA); products still built")\n    return None\n'

NEW_STEP1 = 'def _new_step1(target_dates, _ref_in, workdir, prep_dir, post_dates, lst, fl, ref_date, good_dates_deep, _gate_rows, date_params, i_aligned):\n    from exopipe.phase3.lib.masters import night_image_gate as _NG\n    ref_data, ref_wcs, beam_pix, horizon_mask, cd = _ref_in[\'data\'], _ref_in[\'wcs\'], _ref_in[\'beam_pix\'], _ref_in[\'horizon_mask\'], _ref_in[\'cd\']\n    tmpl_hdr, target_shape = _ref_in[\'tmpl_hdr\'], _ref_in[\'target_shape\']\n    _fref = dict(data=ref_data, wcs=ref_wcs, beam_pix=beam_pix, horizon_mask=horizon_mask, cd=cd, tmpl_hdr=tmpl_hdr,\n                 target_shape=target_shape)\n    for date, sci_dir, align_img_path in target_dates:\n        _nr = final_night(date, sci_dir, align_img_path, _fref, workdir, prep_dir, post_dates, lst, fl, ref_date)\n        if _nr is None:\n            continue\n        if not _nr[\'qa3_in\']:\n            good_dates_deep.discard(date)\n        if _nr[\'gate\'] is not None:\n            _gate_rows[date] = _nr[\'gate\']\n        if _nr[\'params\'] is None:\n            continue\n        date_params[date] = _nr[\'params\']\n        if _nr[\'gate\'][\'verdict\'] == \'FAIL\':\n            good_dates_deep.discard(date)\n        if date in good_dates_deep:\n            i_aligned.append((date, _nr[\'i_out\'], _nr[\'weight\']))\n            flow_str = \'flow\' if _nr[\'flow_offsets\'] is not None else \'no-flow\'\n            print(f"    {date}: scale={_nr[\'mean_scale\']:.3f} ({_nr[\'n_src\']} src) "\n                  f"w={_nr[\'weight\']:.3f} {flow_str} ✓")\n        else:\n            print(f"    {date}: coadd-omitted (QA); products still built")\n    return None\n'

OLD_PASS1 = 'def _old_pass1(dates, date_params, prep_dir, coadd_mask, ref_hdr, final_i_data, final_wcs, beam_pix):\n    scale2_meas = {}\n    for date, sci_dir, align_img_path in dates:\n        if date not in date_params:\n            continue\n        i_path = f\'{prep_dir}/{date}_I.fits\'\n        if not os.path.exists(i_path):\n            continue\n        i_data, _ = load_image(i_path)\n        combined_mask = coadd_mask | (~np.isfinite(i_data)) | (i_data == 0)\n        obs_time = get_midpoint_time(ref_hdr, i_path)\n        # epoch (flow-aligned) against the final master, read at the master\'s peak pixels (Gregg 2026-10-04, "1 approved":\n        # the aperture maximum read high and left a negative imprint of the bright sources in the residuals)\n        _, coeffs2, n2 = measure_flux_scale(\n            i_data, final_i_data, final_wcs, beam_pix, combined_mask, obs_time, at_peak_pixel=True)\n        scale2_meas[date] = (coeffs2, n2)\n    _dcs = [float(c[0][0]) for c in scale2_meas.values() if c[1] >= 8]\n    med_dc = float(np.median(_dcs)) if _dcs else 1.0\n    return scale2_meas, med_dc\n'

NEW_PASS1 = "def _new_pass1(dates, date_params, prep_dir, coadd_mask, ref_hdr, final_i_data, final_wcs, beam_pix):\n    scale2_meas = {}\n    for date, sci_dir, align_img_path in dates:\n        if date not in date_params:\n            continue\n        i_path = f'{prep_dir}/{date}_I.fits'\n        if not os.path.exists(i_path):\n            continue\n        scale2_meas[date] = measure_scale2(i_path, final_i_data, final_wcs, beam_pix, coadd_mask, ref_hdr)\n    _dcs = [float(c[0][0]) for c in scale2_meas.values() if c[1] >= 8]\n    med_dc = float(np.median(_dcs)) if _dcs else 1.0\n    return scale2_meas, med_dc\n"

OLD_PASS2 = 'def _old_pass2(dates, scale2_meas, date_params, prep_dir, tmpl_hdr, target_shape, med_dc, final_i_data, final_v_data, final_wcs, coadd_mask, beam_pix, sub_dir, fl, freq_str):\n    for date, sci_dir, align_img_path in dates:\n        if date not in scale2_meas:\n            continue\n        params = date_params[date]\n        i_path = f\'{prep_dir}/{date}_I.fits\'\n        i_data, _ = load_image(i_path)\n\n        # Per-epoch time block (from this date\'s Phase-2 source) stamped onto every\n        # residual below — keeps tmpl_hdr\'s WCS/grid, fixes the inherited-DATE-OBS bug.\n        # Beam+RESTFRQ are stamped per Stokes at each write via\n        # save_epoch_residual (R1 amendment #1) — never left as tmpl_hdr\'s.\n        epoch_hdr = epoch_time_header(tmpl_hdr, align_img_path)\n\n        coeffs2, n2 = scale2_meas[date]\n        h, w = target_shape\n        yg, xg = np.indices((h, w), dtype=float)\n        if os.environ.get(\'P3_FLUXSCALE_TILT\') != \'1\':\n            coeffs2 = np.array([coeffs2[0], 0.0, 0.0])   # DC-only second pass (R-TILT-REMOVED-NEXTGEN)\n        scale2_full = (coeffs2[0] + coeffs2[1] * (2.0 * xg / w - 1.0)\n                       + coeffs2[2] * (2.0 * yg / h - 1.0))\n        # Flux-degeneracy guard, median-relative (see PASS 1 comment)\n        _dc = float(coeffs2[0]); _tilt = float(np.hypot(coeffs2[1], coeffs2[2]))\n        if (n2 < 8 or not (0.6 * med_dc <= _dc <= 1.6 * med_dc)\n                or _tilt > 0.3 * abs(med_dc)):\n            print(f"    {date}: scale2-degenerate (DC={_dc:.3f} tilt={_tilt:.3f} "\n                  f"n={n2}; cell median DC={med_dc:.3f}) -> flat DC={med_dc:.3f}")\n            scale2_full = np.full((h, w), med_dc if med_dc > 0 else 1.0)\n            coeffs2 = np.array([med_dc, 0.0, 0.0])\n\n        # Apply flux correction and subtract\n        safe_scale = np.where(np.abs(scale2_full) > 0.1, scale2_full, 1.0)\n        i_corrected = i_data / safe_scale\n        # THE ALIGNED, FLUX-CORRECTED EPOCH IMAGE itself - contract 0.2 (1), the archived epoch product (2026-09-30:\n        # it was never written, and the archive published the striation-cleaned residual under its name)\n        save_epoch_residual(f\'{sub_dir}/{date}_I_epoch_{fl}.fits\', i_corrected, epoch_hdr,\n                            align_img_path, tmpl_hdr, \'I\')\n        i_sub = i_corrected - final_i_data\n\n        sub_i_path = f\'{sub_dir}/{date}_I_sub_{fl}.fits\'\n        save_epoch_residual(sub_i_path, i_sub, epoch_hdr,\n                            align_img_path, tmpl_hdr, \'I\')\n\n        # Striation removal on I\n        i_mask = coadd_mask | (~np.isfinite(i_sub)) | (i_sub == 0)\n        # window = 8 x beam (R-STRIATION-WINDOW-APPROVED-8BEAMS 2026-08-25;\n        # supersedes both the 20-beam default and the fixed 4.375 deg)\n        i_stri_model, _ = build_striation_model(\n            i_sub, final_wcs, i_mask, ra_window_beams=8.0, beam_pix=beam_pix)\n        i_clean = i_sub - i_stri_model\n        clean_i_path = f\'{sub_dir}/{date}_I_clean_{fl}.fits\'\n        save_epoch_residual(clean_i_path, i_clean, epoch_hdr,\n                            align_img_path, tmpl_hdr, \'I\')\n\n        i_rms_before = get_uniform_rms(i_sub, i_mask)\n        i_rms_after = get_uniform_rms(i_clean, i_mask)\n\n        # V subtraction + striation removal\n        v_path = f\'{prep_dir}/{date}_V.fits\'\n        if os.path.exists(v_path) and final_v_data is not None:\n            # R1 amendment #1 / restamp finding F2: V residuals carry the V\n            # epoch image\'s OWN beam (Taper_R0 convention), never I-derived.\n            v_src = find_stokes_v_deep_image(sci_dir, freq_str)\n            v_data, _ = load_image(v_path)\n            v_corrected = v_data / safe_scale\n            save_epoch_residual(f\'{sub_dir}/{date}_V_epoch_{fl}.fits\', v_corrected, epoch_hdr,\n                                v_src, tmpl_hdr, \'V\')          # the aligned, flux-corrected V epoch (2026-09-30)\n            v_sub = v_corrected - final_v_data\n\n            sub_v_path = f\'{sub_dir}/{date}_V_sub_{fl}.fits\'\n            save_epoch_residual(sub_v_path, v_sub, epoch_hdr,\n                                v_src, tmpl_hdr, \'V\')\n\n            v_mask = coadd_mask | (~np.isfinite(v_sub)) | (v_sub == 0)\n            v_stri_model, _ = build_striation_model(\n                v_sub, final_wcs, v_mask, ra_window_beams=8.0, beam_pix=beam_pix)\n            v_clean = v_sub - v_stri_model\n\n            clean_v_path = f\'{sub_dir}/{date}_V_clean_{fl}.fits\'\n            save_epoch_residual(clean_v_path, v_clean, epoch_hdr,\n                                v_src, tmpl_hdr, \'V\')\n\n            v_rms_before = get_uniform_rms(v_sub, v_mask)\n            v_rms_after = get_uniform_rms(v_clean, v_mask)\n            print(f"    {date}: scale2={coeffs2[0]:.4f} ({n2} src)  "\n                  f"I {i_rms_before*1000:.1f}→{i_rms_after*1000:.1f}  "\n                  f"V {v_rms_before*1000:.1f}→{v_rms_after*1000:.1f} mJy")\n        else:\n            print(f"    {date}: scale2={coeffs2[0]:.4f} ({n2} src)  "\n                  f"I {i_rms_before*1000:.1f}→{i_rms_after*1000:.1f} mJy")\n    return None\n'

NEW_PASS2 = "def _new_pass2(dates, scale2_meas, date_params, prep_dir, tmpl_hdr, target_shape, med_dc, final_i_data, final_v_data, final_wcs, coadd_mask, beam_pix, sub_dir, fl, freq_str):\n    for date, sci_dir, align_img_path in dates:\n        if date not in scale2_meas:\n            continue\n        params = date_params[date]\n        subtract_night(date, sci_dir, align_img_path, f'{prep_dir}/{date}_I.fits', f'{prep_dir}/{date}_V.fits',\n                       *scale2_meas[date], med_dc, final_i_data, final_v_data, final_wcs, coadd_mask, beam_pix, tmpl_hdr,\n                       target_shape, sub_dir, fl, freq_str, first_scale=params['scale_map'])\n    return None\n"

NEW_CALLER_STEP1 = '    _fref = dict(data=ref_data, wcs=ref_wcs, beam_pix=beam_pix, horizon_mask=horizon_mask, cd=cd, tmpl_hdr=tmpl_hdr,\n                 target_shape=target_shape)\n    for date, sci_dir, align_img_path in target_dates:\n        _nr = final_night(date, sci_dir, align_img_path, _fref, workdir, prep_dir, post_dates, lst, fl, ref_date)\n        if _nr is None:\n            continue\n        if not _nr[\'qa3_in\']:\n            good_dates_deep.discard(date)\n        if _nr[\'gate\'] is not None:\n            _gate_rows[date] = _nr[\'gate\']\n        if _nr[\'params\'] is None:\n            continue\n        date_params[date] = _nr[\'params\']\n        if _nr[\'gate\'][\'verdict\'] == \'FAIL\':\n            good_dates_deep.discard(date)\n        if date in good_dates_deep:\n            i_aligned.append((date, _nr[\'i_out\'], _nr[\'weight\']))\n            flow_str = \'flow\' if _nr[\'flow_offsets\'] is not None else \'no-flow\'\n            print(f"    {date}: scale={_nr[\'mean_scale\']:.3f} ({_nr[\'n_src\']} src) "\n                  f"w={_nr[\'weight\']:.3f} {flow_str} ✓")\n        else:\n            print(f"    {date}: coadd-omitted (QA); products still built")'

NEW_CALLER_PASS1 = "    scale2_meas = {}\n    for date, sci_dir, align_img_path in dates:\n        if date not in date_params:\n            continue\n        i_path = f'{prep_dir}/{date}_I.fits'\n        if not os.path.exists(i_path):\n            continue\n        scale2_meas[date] = measure_scale2(i_path, final_i_data, final_wcs, beam_pix, coadd_mask, ref_hdr)\n    _dcs = [float(c[0][0]) for c in scale2_meas.values() if c[1] >= 8]\n    med_dc = float(np.median(_dcs)) if _dcs else 1.0"

NEW_CALLER_PASS2 = "    for date, sci_dir, align_img_path in dates:\n        if date not in scale2_meas:\n            continue\n        params = date_params[date]\n        subtract_night(date, sci_dir, align_img_path, f'{prep_dir}/{date}_I.fits', f'{prep_dir}/{date}_V.fits',\n                       *scale2_meas[date], med_dc, final_i_data, final_v_data, final_wcs, coadd_mask, beam_pix, tmpl_hdr,\n                       target_shape, sub_dir, fl, freq_str, first_scale=params['scale_map'])"


def _ns():
    from exopipe.phase3.lib.masters import coadd as co
    for k in ('OLD_STEP1', 'NEW_STEP1', 'OLD_PASS1', 'NEW_PASS1', 'OLD_PASS2', 'NEW_PASS2'):
        exec(compile(globals()[k], f'<pinned {k}>', 'exec'), co.__dict__)
    return co


def _state_eq(a, b, pa, pb):
    """date_params / gate rows / membership / aligned list of two runs equal (paths compared with their prep folder
    swapped; arrays element-wise, NaN equal)."""
    def norm(x):
        if isinstance(x, str):
            return x.replace(pb, pa)
        if isinstance(x, dict):
            return {k: norm(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(norm(v) for v in x)
        return x

    def eq(x, y):
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            return (isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and x.shape == y.shape
                    and np.array_equal(x, y, equal_nan=True))
        if isinstance(x, dict) and isinstance(y, dict):
            return set(x) == set(y) and all(eq(x[k], y[k]) for k in x)
        if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            return len(x) == len(y) and all(eq(p, q) for p, q in zip(x, y))
        if isinstance(x, float) and isinstance(y, float) and x != x and y != y:
            return True
        return x == y
    return eq(a, norm(b))


def _dir_eq(a, b):
    import filecmp
    fa, fb = sorted(os.listdir(a)), sorted(os.listdir(b))
    return fa == fb and all(filecmp.cmp(os.path.join(a, f), os.path.join(b, f), shallow=False) for f in fa), (fa, fb)


def test_final_pass_equivalence():
    """The build's per-night final pass, before and after the extraction, on real nights of 13h_78MHz (stored flows; the
    installed master in native units as the reference and as the final master): every state the loop changes
    (membership, night-image gate rows, date_params, the aligned list), every file written and the log are equal - on a
    member night, a night refused by the stage-3 window AND the night-image gate, an unmeasurable flux scale (outside the
    window too: the window's verdict counts before the refusal), a blank field, a shape mismatch, a wrong grid; then
    PASS 1 (the median second-pass DC) and PASS 2 (epoch / sub / clean, I and V) on the aligned nights, and PASS 2 again
    with a median that makes the degeneracy guard fire."""
    import contextlib
    import io
    from astropy.io import fits
    from astropy.wcs import WCS
    from exopipe.phase3.lib.masters import admit as A, admit_stages as AS, membership as MB, night_image_gate as NG
    co = _ns()
    src = co.__dict__
    check('the new callers pinned here are the ones in coadd.run_final_lst_freq (no drift)',
          all(globals()[k] in open(co.__file__).read() for k in ('NEW_CALLER_STEP1', 'NEW_CALLER_PASS1', 'NEW_CALLER_PASS2')))
    cell, fl, freq_str, lst = '13h_78MHz', '78MHz', '78', '13h'
    root = os.path.join(CFG.paths.COADD_SB_ROOT, cell)
    ents = A.installed_reference(cell, root)
    refs = A.write_reference(os.path.join(T, 'ref'), '78MHz', ents)
    r = AS.reference_dict(co, refs['I'], ents[0])
    ref_hdr = fits.getheader(refs['I'])
    cdelt = abs(ref_hdr.get('CDELT1', ref_hdr.get('CD1_1', 0.037)))
    fref = dict(data=r['data'], wcs=r['wcs'], beam_pix=r['beam_pix'], horizon_mask=r['horizon_mask'], cd=cdelt,
                tmpl_hdr=r['tmpl_hdr'], target_shape=r['target_shape'])
    with contextlib.redirect_stdout(io.StringIO()):
        nights = co.discover_dates(lst, freq_str)
    import csv as _csv
    flows = {d for d in os.listdir(os.path.join(root, 'flow')) if d.endswith('_scaled_flow.npy')}
    mem = {w['date'] for w in _csv.DictReader(open(os.path.join(root, 'coadd_final', 'weights.csv')))
           if w['in_master'] == '1' and w['qa3_guard'] == 'pass'}                  # members, not a corrupt night (2025-01-08)
    have = [x for x in nights if f'{x[0]}_scaled_flow.npy' in flows and x[0] >= co.CUTOVER_DATE and x[0] in mem]
    post = [x for x in nights if x[0] >= co.CUTOVER_DATE]
    a_n, b_n, c_n, d_n = have[:4]
    blank = os.path.join(T, 'blank.fits')
    with fits.open(d_n[2]) as h:
        fits.writeto(blank, np.zeros_like(h[0].data), h[0].header)
    cases = [a_n, b_n, c_n, (d_n[0], d_n[1], blank)]
    real_qa3, real_eval, real_dc = MB.qa3_inside, NG.evaluate, src['_night_flux_dc']
    MB.qa3_inside = lambda f, s, _d=[None]: False if _CUR[0] in (b_n[0], c_n[0]) else real_qa3(f, s)
    NG.evaluate = lambda *a, **k: (dict(verdict='FAIL', reason='forced by the test') if _CUR[0] == b_n[0] else real_eval(*a, **k))
    src['_night_flux_dc'] = lambda *a, **k: ((float('nan'), None) if _CUR[0] == c_n[0] else real_dc(*a, **k))
    runs = {}
    try:
        for tag, fname in (('old', '_old_step1'), ('new', '_new_step1')):
            pdir = os.path.join(T, f'prep_{tag}')
            os.makedirs(pdir)
            st = dict(good=set(d for d, _, _ in cases), gate={}, params={}, aligned=[])
            buf = io.StringIO()
            for case in cases:
                _CUR[0] = case[0]
                with contextlib.redirect_stdout(buf):
                    src[fname]([case], fref, root, pdir, post, lst, fl, 'MASTER', st['good'], st['gate'], st['params'],
                               st['aligned'])
            small = dict(fref, target_shape=(fref['target_shape'][0] - 1, fref['target_shape'][1]))
            grid = dict(fref, horizon_mask=fref['horizon_mask'][:-1])
            for ref_x in (small, grid):
                _CUR[0] = None
                with contextlib.redirect_stdout(buf):
                    src[fname]([a_n], ref_x, root, pdir, post, lst, fl, 'MASTER', st['good'], st['gate'], st['params'],
                               st['aligned'])
            runs[tag] = (pdir, st, buf.getvalue())
    finally:
        MB.qa3_inside, NG.evaluate, src['_night_flux_dc'] = real_qa3, real_eval, real_dc
    (po, so, lo), (pn, sn, ln) = runs['old'], runs['new']
    same_files = _dir_eq(po, pn)
    check('Step 1: membership, gate rows, date_params and the aligned list equal; same files; same log',
          _state_eq(so, sn, po, pn) and same_files[0] and lo == ln.replace(pn, po), (sorted(so['good']), sorted(sn['good']),
                                                                                 same_files[1], lo == ln.replace(pn, po)))
    check('Step 1 covered each path: a member, the window + gate refusal, an unmeasurable scale, a blank field, shape and '
          'grid mismatches', a_n[0] in so['good'] and a_n[0] in so['params'] and b_n[0] in so['params']
          and b_n[0] not in so['good'] and so['gate'].get(b_n[0], {}).get('reason') == 'forced by the test'
          and so['gate'].get(c_n[0], {}).get('verdict') == 'FAIL' and c_n[0] not in so['params'] and c_n[0] not in so['good']
          and d_n[0] not in so['params'] and 'BLANK FIELD REFUSED' in lo and 'shape mismatch' in lo and 'WRONG GRID' in lo
          and '✓' in lo, (sorted(so['good']), sorted(so['params']), lo[-900:]))
    with fits.open(refs['V']) as h:
        fin_v = h[0].data.astype(np.float32)
    v_img = co.find_stokes_v_deep_image(a_n[1], freq_str)
    pa = so['params'][a_n[0]]
    for pdir in (po, pn):
        co.process_variant_image(v_img, pa['flow'], pa['scale_map'], fref['tmpl_hdr'], fref['target_shape'],
                                 f'{pdir}/{a_n[0]}_V.fits', co._product_mapping(v_img, a_n[2], pa['regrid']))
    final_wcs = WCS(fref['tmpl_hdr']).celestial
    coadd_mask = ~np.isfinite(fref['data']) | (fref['data'] == 0)
    dates = [x for x in cases if x[2] != blank]
    m_old = src['_old_pass1'](dates, so['params'], po, coadd_mask, ref_hdr, fref['data'], final_wcs, fref['beam_pix'])
    m_new = src['_new_pass1'](dates, sn['params'], pn, coadd_mask, ref_hdr, fref['data'], final_wcs, fref['beam_pix'])
    check('PASS 1: the second-pass scales and the cell median DC equal', _state_eq(m_old, m_new, po, pn), (m_old, m_new))
    for med, label in ((m_old[1], 'measured'), (10.0, 'forced: the guard fires')):
        outs = {}
        for tag, pdir, st, fname in (('old', po, so, '_old_pass2'), ('new', pn, sn, '_new_pass2')):
            sub = os.path.join(T, f'sub_{tag}_{label[:6]}')
            os.makedirs(sub)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                src[fname](dates, m_old[0], st['params'], pdir, fref['tmpl_hdr'], fref['target_shape'], med, fref['data'],
                           fin_v, final_wcs, coadd_mask, fref['beam_pix'], sub, fl, freq_str)
            outs[tag] = (sub, buf.getvalue())
        same = _dir_eq(outs['old'][0], outs['new'][0])
        check(f'PASS 2 (median {label}): the epoch / sub / clean files of I and V equal, same log',
              same[0] and outs['old'][1] == outs['new'][1] and len(same[1][0]) >= 6
              and (label == 'measured' or 'scale2-degenerate' in outs['old'][1]), (same[1], outs['old'][1][-300:]))


_CUR = [None]


def main():
    shutil.rmtree(T, ignore_errors=True)
    try:
        test_bake_equivalence()
        test_final_pass_equivalence()
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
