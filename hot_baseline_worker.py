import argparse
import sys
import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
from casacore.tables import table
import logging
import pipeline_config as config  # IMPORT CONFIG

# --- 1. CONFIGURATION ---
# Loaded from pipeline_config.py
SYSTEM_CONFIG = getattr(config, 'SYSTEM_CONFIG', {})

if not SYSTEM_CONFIG:
    print("WARNING: SYSTEM_CONFIG not found in pipeline_config.py. Mapping will fail.")

# --- 2. HELPERS ---
def extract_ant_number(name):
    match = re.search(r'(\d+)', str(name))
    if match: return int(match.group(1))
    return None

def get_ms_id_map(ms_name):
    try:
        t = table(ms_name + "::ANTENNA", ack=False, readonly=True)
        names = t.getcol("NAME")
        t.close()
        id_map = {}
        for row_idx, name in enumerate(names):
            ant_num = extract_ant_number(name)
            if ant_num is not None: id_map[ant_num] = row_idx
        return id_map
    except Exception as e:
        logging.error(f"Failed to load antenna table: {e}")
        return {}

def build_complete_map(ms_id_map):
    """
    Build a DataFrame mapping correlator numbers to hardware info.
    Uses corr_num directly from SYSTEM_CONFIG (verified against wiring spreadsheet)
    rather than inferring it from MS ANTENNA table row indices.
    ms_id_map is still used to verify which antennas are present in the MS.
    """
    if not SYSTEM_CONFIG:
        return pd.DataFrame()
        
    df = pd.DataFrame.from_dict(SYSTEM_CONFIG, orient='index')
    df['ant_num'] = df.index

    # corr_num comes directly from the config now
    # Filter to only antennas that are present in this MS
    ms_ant_nums = set(ms_id_map.keys())
    df = df[df['ant_num'].isin(ms_ant_nums)]
    
    if 'corr_num' not in df.columns:
        # Fallback for old config format without corr_num
        df['corr_num'] = df['ant_num'].map(ms_id_map)
        df = df.dropna(subset=['corr_num'])
        df['corr_num'] = df['corr_num'].astype(int)

    # SNAP board: use from config if available, else compute
    if 'snap' in df.columns:
        df['snap_id_int'] = df['snap']
    else:
        df['snap_id_int'] = (df['corr_num'] // 32) + 1
    df['snap_id'] = df['snap_id_int'].apply(lambda x: f"SNAP-{x:02d}")
    
    df['arx_id'] = df['arx']
    # Support both old ('chan') and new ('pola_chan') field names
    if 'pola_chan' in df.columns:
        df['arx_chan'] = df['pola_chan']
    elif 'chan' in df.columns:
        df['arx_chan'] = df['chan']
    else:
        df['arx_chan'] = 0
    df['display_name'] = df['name']
    return df.set_index('corr_num').sort_index()

def get_antenna_positions(ms_name):
    t = table(ms_name + "::ANTENNA", ack=False, readonly=True)
    pos = t.getcol("POSITION"); t.close()
    return pos

def get_mean_frequency(ms_name):
    t = table(ms_name + "::SPECTRAL_WINDOW", ack=False, readonly=True)
    freqs = t.getcol("CHAN_FREQ"); t.close()
    return np.mean(freqs)

def get_data_matrix(ms_name, data_col="CORRECTED_DATA"):
    # Wrapped in try/except for safety
    try:
        t_ant = table(ms_name + "::ANTENNA", ack=False, readonly=True)
        n_ant = t_ant.nrows(); t_ant.close()
        
        accum_sum = {pol: np.zeros((n_ant, n_ant), dtype=np.float64) for pol in ['XX', 'XY', 'YX', 'YY']}
        accum_sq  = {pol: np.zeros((n_ant, n_ant), dtype=np.float64) for pol in ['XX', 'XY', 'YX', 'YY']}
        accum_cnt = {pol: np.zeros((n_ant, n_ant), dtype=np.int64) for pol in ['XX', 'XY', 'YX', 'YY']}
        
        t = table(ms_name, ack=False, readonly=True)
        chunk_size = 500000 
        pol_map = {0:'XX', 1:'XY', 2:'YX', 3:'YY'}

        for start in range(0, t.nrows(), chunk_size):
            end = min(start + chunk_size, t.nrows())
            a1 = t.getcol("ANTENNA1", startrow=start, nrow=end-start)
            a2 = t.getcol("ANTENNA2", startrow=start, nrow=end-start)
            try:
                data = t.getcol(data_col, startrow=start, nrow=end-start)
            except RuntimeError:
                # Fallback if CORRECTED_DATA missing
                logging.warning(f"Column {data_col} missing, falling back to DATA.")
                data = t.getcol("DATA", startrow=start, nrow=end-start)
                
            mask = (a1 != a2)
            if not mask.any(): continue
            a1 = a1[mask]; a2 = a2[mask]; data = data[mask]
            amp = np.abs(data); amp_avg = np.nanmean(amp, axis=1)

            for p_idx in range(4):
                pol_name = pol_map[p_idx]; vals = amp_avg[:, p_idx]
                valid = np.isfinite(vals)
                if not valid.any(): continue
                v_a1 = a1[valid]; v_a2 = a2[valid]; v_vals = vals[valid]
                np.add.at(accum_sum[pol_name], (v_a1, v_a2), v_vals)
                np.add.at(accum_sq[pol_name],  (v_a1, v_a2), v_vals**2)
                np.add.at(accum_cnt[pol_name], (v_a1, v_a2), 1)
        t.close()

        mats_mean = {}; mats_std = {}
        for pol in ['XX', 'XY', 'YX', 'YY']:
            cnt = accum_cnt[pol]; s = accum_sum[pol]; ss = accum_sq[pol]
            with np.errstate(invalid='ignore', divide='ignore'):
                mean_mat = s / cnt
                var_mat = (ss / cnt) - (mean_mat ** 2)
                std_mat = np.sqrt(var_mat)
            mean_mat[cnt == 0] = np.nan; std_mat[cnt == 0] = np.nan
            mean_sym = np.copy(mean_mat); std_sym = np.copy(std_mat)
            rows, cols = np.triu_indices(n_ant, k=1)
            mean_sym[cols, rows] = mean_sym[rows, cols]; std_sym[cols, rows] = std_sym[rows, cols]
            mats_mean[pol] = mean_sym; mats_std[pol] = std_sym
        return mats_mean, mats_std, n_ant
    except Exception as e:
        logging.error(f"Error reading MS matrix: {e}")
        # Return dummies to prevent crash
        return {}, {}, 0

def apply_uv_cut(matrix, pos_array, uv_cut_m):
    if uv_cut_m <= 0: return matrix
    dist_mat = np.linalg.norm(pos_array[:, None, :] - pos_array[None, :, :], axis=2)
    matrix[dist_mat < uv_cut_m] = np.nan
    return matrix

def analyze_amp_vs_uv(raw_mean, pos_array, n_ant, mean_freq=None, sigma_cut=5.0, plot_prefix="amp_vs_uv", window_size=100):
    uv_dist_mat = np.linalg.norm(pos_array[:, None, :] - pos_array[None, :, :], axis=2)
    mask = np.triu(np.ones((n_ant, n_ant), dtype=bool), k=1)
    min_dist_m = 0.0
    if mean_freq:
        c = 299792458.0; lam = c / mean_freq; min_dist_m = 4.0 * lam
    
    dead_antennas = set()
    check_pol = 'XX' if 'XX' in raw_mean else list(raw_mean.keys())[0]
    valid_amps_all = raw_mean[check_pol][np.isfinite(raw_mean[check_pol])]
    if len(valid_amps_all) > 0:
        global_median = np.nanmedian(valid_amps_all)
        ant_means = np.nanmedian(raw_mean[check_pol], axis=1)
        dead_threshold = 0.20 * global_median
        dead_indices = np.where(ant_means < dead_threshold)[0]
        for idx in dead_indices: dead_antennas.add(idx)

    uv_dists = uv_dist_mat[mask]; rows, cols = np.where(mask)
    local_bad_baselines = set()
    fig, axes = plt.subplots(2, 2, figsize=(18, 12)); axes = axes.flatten()
    pols = ['XX', 'XY', 'YX', 'YY']
    
    for i, pol in enumerate(pols):
        ax = axes[i]; amps = raw_mean[pol][mask]
        valid_mask = np.isfinite(amps)
        curr_dists = uv_dists[valid_mask]; curr_amps = amps[valid_mask]
        curr_rows = rows[valid_mask]; curr_cols = cols[valid_mask]
        
        is_dead_row = np.isin(curr_rows, list(dead_antennas))
        is_dead_col = np.isin(curr_cols, list(dead_antennas))
        is_weak_baseline = (is_dead_row | is_dead_col)
        is_short_baseline = (curr_dists < min_dist_m)
        is_ignored = (is_weak_baseline | is_short_baseline)

        fit_dists = curr_dists[~is_ignored]; fit_amps = curr_amps[~is_ignored]
        ignored_dists = curr_dists[is_ignored]; ignored_amps = curr_amps[is_ignored]

        if len(fit_amps) < window_size:
            ax.text(0.5, 0.5, "Insufficient Data", ha='center'); continue

        sort_idx = np.argsort(fit_dists)
        s_dists = fit_dists[sort_idx]; s_amps = fit_amps[sort_idx]
        df_roll = pd.DataFrame({'amp': s_amps})
        roll_med = df_roll['amp'].rolling(window=window_size, center=True, min_periods=10).median().bfill().ffill()
        abs_diff = (df_roll['amp'] - roll_med).abs()
        roll_mad = abs_diff.rolling(window=window_size, center=True, min_periods=10).median().bfill().ffill()
        envelope_sorted = roll_med.values + (sigma_cut * 1.4826 * roll_mad.values)

        bad_mask_sorted = (s_amps > envelope_sorted)
        bad_fit_indices_sorted = np.where(bad_mask_sorted)[0]
        bad_indices_in_fit = sort_idx[bad_fit_indices_sorted]
        all_indices = np.arange(len(curr_dists))
        fit_indices_in_full = all_indices[~is_ignored]
        bad_indices_full = fit_indices_in_full[bad_indices_in_fit]
        
        bad_u = curr_rows[bad_indices_full]; bad_v = curr_cols[bad_indices_full]
        for u, v in zip(bad_u, bad_v): local_bad_baselines.add(tuple(sorted((u, v))))

        if len(ignored_dists) > 0:
            plot_idx_w = np.arange(len(ignored_dists))
            if len(plot_idx_w) > 20000: plot_idx_w = np.random.choice(plot_idx_w, 20000, replace=False)
            ax.scatter(ignored_dists[plot_idx_w], ignored_amps[plot_idx_w], c='lightgrey', s=1, alpha=0.5)

        plot_idx = np.arange(len(fit_dists))
        if len(plot_idx) > 40000: plot_idx = np.random.choice(plot_idx, 40000, replace=False)
        ax.scatter(fit_dists[plot_idx], fit_amps[plot_idx], c='royalblue', s=1, alpha=0.5)
        ax.plot(s_dists, envelope_sorted, color='lime', lw=2)
        if len(bad_indices_full) > 0:
            ax.scatter(curr_dists[bad_indices_full], curr_amps[bad_indices_full], c='red', s=10, marker='x')
        ax.set_title(f"Pol {pol}"); ax.set_xlabel("Baseline Length (m)"); ax.set_ylabel("Amplitude")

    plt.tight_layout(); plt.savefig(f"{plot_prefix}.png", dpi=150); plt.close()
    return list(local_bad_baselines)

def identify_bad_components(combined_matrix, df_map, n_ant, extra_bad_baselines=None, sigma_cut=5.0, threshold_percent=0.10):
    detect_mat = combined_matrix.copy()
    mask_tri = np.triu(np.ones_like(detect_mat, dtype=bool), k=1)
    valid_data = detect_mat[mask_tri & np.isfinite(detect_mat)]
    if len(valid_data) == 0: return [], set(), ["No valid data."]
    
    med = np.nanmedian(valid_data)
    mad = np.nanmedian(np.abs(valid_data - med))
    threshold = med + sigma_cut * 1.4826 * mad
    
    rows, cols = np.where((detect_mat > threshold) & mask_tri)
    heatmap_bad_baselines = list(zip(rows, cols))
    all_bad_baselines_set = set(heatmap_bad_baselines)
    if extra_bad_baselines:
        for b in extra_bad_baselines: all_bad_baselines_set.add(b)
            
    all_bad_baselines = list(all_bad_baselines_set)
    ant_counts = {}
    for r, c in all_bad_baselines:
        ant_counts[r] = ant_counts.get(r, 0) + 1; ant_counts[c] = ant_counts.get(c, 0) + 1

    bad_antennas = set()
    threshold_count = (n_ant - 1) * threshold_percent
    for ant, count in ant_counts.items():
        if count > threshold_count: bad_antennas.add(ant)

    lines = [f"\n[HOT BASELINE REPORT]", f"  Total: {len(all_bad_baselines)}"]
    return heatmap_bad_baselines, bad_antennas, lines

def apply_flags_to_ms(ms_name, bad_antennas, bad_baselines):
    print(f"--> Applying Flags to {ms_name}...")
    try:
        t = table(ms_name, readonly=False); nrows = t.nrows()
        bad_pair_set = {tuple(sorted((u, v))) for u, v in bad_baselines if u not in bad_antennas and v not in bad_antennas}
        chunk_size = 1000000 
        for start in range(0, nrows, chunk_size):
            end = min(start + chunk_size, nrows)
            c_a1 = t.getcol("ANTENNA1", startrow=start, nrow=end-start)
            c_a2 = t.getcol("ANTENNA2", startrow=start, nrow=end-start)
            c_mask = np.zeros(end-start, dtype=bool)
            if bad_antennas:
                c_mask |= (np.isin(c_a1, list(bad_antennas)) | np.isin(c_a2, list(bad_antennas)))
            if bad_pair_set:
                N = 100000  # supports up to 100k antennas (was 10000)
                bad_encoded = {u*N + v for u, v in bad_pair_set}
                c_min = np.minimum(c_a1, c_a2); c_max = np.maximum(c_a1, c_a2)
                encoded_chunk = c_min * N + c_max
                c_mask |= np.isin(encoded_chunk, list(bad_encoded))
            if c_mask.any():
                flags = t.getcol("FLAG", startrow=start, nrow=end-start)
                flags[c_mask, :, :] = True
                t.putcol("FLAG", flags, startrow=start, nrow=end-start)
        t.close()
    except Exception as e: print(f"Error flagging: {e}")

def plot_heatmap(matrix, title, filename, vmin=None, vmax=None, grid_lines=None, axis_ticks=None, axis_labels=None):
    plt.figure(figsize=(14, 12)) 
    cmap = plt.cm.inferno.copy(); cmap.set_bad('lightgrey')
    plot_data = matrix.copy(); mask = np.triu(np.ones_like(plot_data, dtype=bool), k=0); plot_data[mask] = np.nan
    
    if vmin is None:
        valid = plot_data[np.isfinite(plot_data)]
        if len(valid) > 0:
            med = np.nanmedian(valid); mad = np.nanmedian(np.abs(valid - med))
            vmin = med - 5*1.4826*mad; vmax = med + 5*1.4826*mad
            if vmin <= 0: vmin = med / 100.0
    try: norm = colors.LogNorm(vmin=vmin, vmax=vmax); im = plt.imshow(plot_data, origin='lower', cmap=cmap, norm=norm, interpolation='nearest')
    except Exception: im = plt.imshow(plot_data, origin='lower', cmap=cmap)
    
    if grid_lines:
        for l in grid_lines:
            plt.axhline(l, color='white', linestyle='--', alpha=0.5)
            plt.axvline(l, color='white', linestyle='--', alpha=0.5)
    if axis_ticks is not None:
        plt.xticks(axis_ticks, axis_labels, rotation=90, fontsize=8); plt.yticks(axis_ticks, axis_labels, fontsize=8)
    plt.colorbar(im); plt.title(title); plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()

def plot_diagnosis_figure(matrix, bad_antennas, bad_baselines, filename, uv_baselines=None):
    plot_data = matrix.copy(); mask = np.triu(np.ones_like(plot_data, dtype=bool), k=0); plot_data[mask] = np.nan
    valid = plot_data[np.isfinite(plot_data)]
    if len(valid) > 0:
        med = np.nanmedian(valid); mad = np.nanmedian(np.abs(valid - med))
        vmin = med - 5*1.4826*mad; vmax = med + 5*1.4826*mad
        if vmin <= 0: vmin = 1e-4
    else: vmin, vmax = 1e-4, 1.0
    cmap = plt.cm.inferno.copy(); cmap.set_bad('lightgrey'); norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    for i, ax in enumerate(axes):
        ax.imshow(plot_data, origin='lower', cmap=cmap, norm=norm, interpolation='nearest')
        if i == 1:
            if uv_baselines:
                uv_mask = np.zeros_like(plot_data, dtype=bool)
                for u, v in uv_baselines: uv_mask[max(u,v), min(u,v)] = True
                m_over = np.zeros((*plot_data.shape, 4)); m_over[uv_mask] = [1,0,1,1]; ax.imshow(m_over, origin='lower')
            bl_mask = np.zeros_like(plot_data, dtype=bool)
            for u, v in bad_baselines: bl_mask[max(u,v), min(u,v)] = True
            c_over = np.zeros((*plot_data.shape, 4)); c_over[bl_mask] = [0,1,1,1]; ax.imshow(c_over, origin='lower')
            ant_mask = np.zeros_like(plot_data, dtype=bool)
            if bad_antennas:
                b = list(bad_antennas); ant_mask[b, :] = True; ant_mask[:, b] = True
            ant_mask[mask] = False
            l_over = np.zeros((*plot_data.shape, 4)); l_over[ant_mask] = [0.2,1,0.2,1]; ax.imshow(l_over, origin='lower')
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()

def run_diagnostics(args_obj, logger):
    logger.info("Starting Hot Baseline Analysis...")
    ms_id_map = get_ms_id_map(args_obj.ms)
    df_map = build_complete_map(ms_id_map)
    pos = get_antenna_positions(args_obj.ms)
    mean_freq = get_mean_frequency(args_obj.ms)
    ms_basename = os.path.basename(os.path.normpath(args_obj.ms))
    
    raw_mean, raw_std, n_ant = get_data_matrix(args_obj.ms, args_obj.col)
    
    uv_bad = []
    if args_obj.run_uv:
        uv_window = getattr(args_obj, 'uv_window_size', 100)
        uv_bad = analyze_amp_vs_uv(raw_mean, pos, n_ant, mean_freq, args_obj.uv_sigma,
                                    f"{ms_basename}_diagnosis_amp_vs_uv",
                                    window_size=uv_window)
        for u, v in uv_bad:
            for pol in raw_mean: raw_mean[pol][u, v] = np.nan; raw_mean[pol][v, u] = np.nan

    cut_meters = args_obj.uv_cut
    if args_obj.uv_cut_lambda > 0 and mean_freq:
        lam = 299792458.0 / mean_freq; cut_meters = max(cut_meters, args_obj.uv_cut_lambda * lam)
    if cut_meters > 0:
        for pol in raw_mean: raw_mean[pol] = apply_uv_cut(raw_mean[pol], pos, cut_meters)

    diag_matrix = (raw_mean['XY'] + raw_mean['YX']) / 2.0
    hm_bad, bad_ants, report = identify_bad_components(diag_matrix, df_map, n_ant, uv_bad, args_obj.sigma, args_obj.threshold)
    
    with open(f"{ms_basename}_hot_component_report.txt", "w") as f: f.write("\n".join(report))
    
    # GENERATE ALL PLOTS
    plot_diagnosis_figure(diag_matrix, bad_ants, hm_bad, f"{ms_basename}_diagnosis_heatmap.png", uv_baselines=uv_bad)
    for pol in ['XX', 'XY', 'YX', 'YY']:
        plot_heatmap(raw_mean[pol], f"Raw Mean {pol}", f"{ms_basename}_heatmap_raw_mean_{pol}.png")
        plot_heatmap(raw_std[pol],  f"Raw Std {pol}",  f"{ms_basename}_heatmap_raw_std_{pol}.png")
        
    # ARX Ordered Plots
    valid_indices = [i for i in range(n_ant) if i in df_map.index]
    df_sub = df_map.loc[valid_indices].sort_values(by=['arx_id', 'arx_chan'])
    arx_order = df_sub.index.values
    arx_ids = df_sub['arx_id'].values
    arx_bounds = []
    curr = 0
    for i in range(len(arx_ids)-1):
        if arx_ids[i] != arx_ids[i+1]: arx_bounds.append(i + 0.5)
            
    for pol in ['XX', 'XY', 'YX', 'YY']:
        mat = raw_mean[pol]; mat_re = mat[np.ix_(arx_order, arx_order)]
        plot_heatmap(mat_re, f"ARX Ordered {pol}", f"{ms_basename}_heatmap_arx_{pol}.png", grid_lines=arx_bounds)

    if args_obj.apply_antenna_flags or args_obj.apply_baseline_flags:
        ants_to_flag = bad_ants if args_obj.apply_antenna_flags else set()
        base_to_flag = list(set(hm_bad + uv_bad)) if args_obj.apply_baseline_flags else []
        if ants_to_flag or base_to_flag:
            apply_flags_to_ms(args_obj.ms, ants_to_flag, base_to_flag)
