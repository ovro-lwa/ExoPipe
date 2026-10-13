#!/usr/bin/env python3
import argparse
import os
import glob
import shutil
import warnings
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.nddata import Cutout2D
from astropy.stats import mad_std

# --- CASA IMPORTS ---
CASA_TASKS_AVAILABLE = False
try:
    from casatasks import imfit, imstat
    CASA_TASKS_AVAILABLE = True
except ImportError:
    print("WARNING: Could not import casatasks (imfit/imstat).", file=sys.stderr)

# --- CONFIGURATION ---
CUTOUT_SIZE = 2.0 * u.deg
SEARCH_RADIUS = 0.25 * u.deg # Constrains imfit/imstat to this radius
MIN_ELEVATION = 20.0
OVRO_LOC = EarthLocation(lat=37.2398*u.deg, lon=-118.2817*u.deg, height=1200*u.m)

# Suppress warnings
warnings.simplefilter('ignore', category=FITSFixedWarning)

# Scaife & Heald (2012) Table 3 Coefficients
SH_CATALOG = {
    '3C48':  {'coords': SkyCoord(24.422, 33.16, unit='deg'),   'coeffs': [63.910, -0.394, -0.391,  0.185, -0.014]},
    '3C147': {'coords': SkyCoord(85.651, 49.852, unit='deg'),  'coeffs': [66.494, -0.041, -0.719,  0.625, -0.124]},
    '3C196': {'coords': SkyCoord(123.400, 48.217, unit='deg'), 'coeffs': [83.776, -0.677, -0.139, -0.027,  0.035]},
    '3C286': {'coords': SkyCoord(202.785, 30.509, unit='deg'), 'coeffs': [27.591, -0.144,  0.005, -0.187,  0.021]},
    '3C295': {'coords': SkyCoord(212.836, 52.203, unit='deg'), 'coeffs': [97.763, -0.582, -0.298,  0.583, -0.363]},
    '3C380': {'coords': SkyCoord(277.382, 48.746, unit='deg'), 'coeffs': [74.386, -0.787,  0.104,  0.030, -0.084]}
}

def get_model_flux(name, freq_mhz):
    if name not in SH_CATALOG: return np.nan
    A = SH_CATALOG[name]['coeffs']
    x = np.log10(freq_mhz / 150.0)
    log_s = np.log10(A[0]) + A[1]*x + A[2]*(x**2) + A[3]*(x**3) + A[4]*(x**4)
    return 10**log_s

def process_cutout(img_path, source_name, source_data, temp_dir):
    try:
        subband = img_path.split('/')[-4]
        
        with fits.open(img_path) as hdul:
            data = hdul[0].data.squeeze()
            header = hdul[0].header
            wcs = WCS(header).celestial
            
            # 1. Elevation Check
            elevation = np.nan
            try:
                t_mid = Time(header['DATE-OBS'], format='isot', scale='utc')
                altaz = source_data['coords'].transform_to(AltAz(obstime=t_mid, location=OVRO_LOC))
                elevation = altaz.alt.deg
                if elevation < MIN_ELEVATION: return None 
            except: pass

            # 2. Create Cutout
            try:
                cutout = Cutout2D(data, source_data['coords'], (CUTOUT_SIZE, CUTOUT_SIZE), wcs=wcs, mode='trim')
            except Exception: return None

            # 3. Save Cutout (Full Header Copy)
            temp_name = f"{source_name}_{subband}.fits"
            temp_path = os.path.join(temp_dir, temp_name)
            cut_header = cutout.wcs.to_header()
            
            # Copy all non-WCS keywords
            wcs_keys = {'NAXIS1','NAXIS2','CRPIX1','CRPIX2','CRVAL1','CRVAL2','CDELT1','CDELT2','CTYPE1','CTYPE2','PC1_1','PC1_2','PC2_1','PC2_2'}
            for card in header.cards:
                if card.keyword not in wcs_keys and card.keyword not in cut_header:
                    cut_header.append(card)
            
            # Force BUNIT if missing
            if 'BUNIT' not in cut_header or not str(cut_header['BUNIT']).strip():
                cut_header['BUNIT'] = 'Jy/beam'
                
            fits.writeto(temp_path, cutout.data, cut_header, overwrite=True)

            # --- Define Search Box (in pixels) for Imfit/Imstat ---
            center_sky = source_data['coords']
            cx, cy = cutout.wcs.world_to_pixel(center_sky)
            
            # Estimate pixel scale (deg/pix)
            cdelt = np.abs(cutout.wcs.wcs.cdelt[0])
            radius_pix = SEARCH_RADIUS.to(u.deg).value / cdelt
            
            # Box format for CASA: 'xmin,ymin,xmax,ymax'
            ny, nx = cutout.data.shape
            x1, y1 = int(cx - radius_pix), int(cy - radius_pix)
            x2, y2 = int(cx + radius_pix), int(cy + radius_pix)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(nx-1, x2), min(ny-1, y2)
            search_box = f"{x1},{y1},{x2},{y2}"

            # Pre-validate before handing anything to CASA: imfit/imstat
            # abort the whole process on fully-blanked regions
            # (casa6core::ArrayError -> C++ terminate -> SIGABRT, which
            # Python CANNOT catch — this killed every 2025-01-02 and
            # 2025-04-04 run). Skip the source instead of crashing.
            region = cutout.data[y1:y2 + 1, x1:x2 + 1]
            region_ok = np.isfinite(region).any() and \
                np.nanmax(np.abs(region)) > 0
            if not region_ok:
                return None
            # imfit additionally aborts on beamless images
            has_beam = float(header.get('BMAJ', 0) or 0) > 0

            # --- METHOD 1: IMSTAT (Max Pixel) ---
            imstat_flux, imstat_err, imstat_ra, imstat_dec = np.nan, np.nan, np.nan, np.nan
            if CASA_TASKS_AVAILABLE:
                try:
                    stats = imstat(imagename=temp_path, box=search_box)
                    if stats and stats['max'].size > 0:
                        imstat_flux = stats['max'][0]
                        imstat_err = stats['rms'][0]
                        
                        max_pix = stats['maxpos']
                        peak_coord = cutout.wcs.pixel_to_world(max_pix[0], max_pix[1])
                        imstat_ra, imstat_dec = peak_coord.ra.deg, peak_coord.dec.deg
                except Exception: pass

            # --- METHOD 2: IMFIT (Gaussian Fit) ---
            imfit_flux, imfit_err, imfit_ra, imfit_dec = np.nan, np.nan, np.nan, np.nan
            if CASA_TASKS_AVAILABLE and has_beam:
                try:
                    fit = imfit(imagename=temp_path, box=search_box)
                    if fit and fit.get('converged', False) and 'results' in fit and 'component0' in fit['results']:
                        comp = fit['results']['component0']
                        imfit_flux = comp['peak']['value']
                        imfit_err = comp['peak']['error']
                        
                        ra_rad = comp['shape']['direction']['m0']['value']
                        dec_rad = comp['shape']['direction']['m1']['value']
                        imfit_ra = np.degrees(ra_rad)
                        imfit_dec = np.degrees(dec_rad)
                except Exception: 
                    imfit_flux = np.nan

            return {
                'imstat_flux': imstat_flux,
                'imstat_err': imstat_err,
                'imstat_ra': imstat_ra,
                'imstat_dec': imstat_dec,
                'imfit_flux': imfit_flux,
                'imfit_err': imfit_err,
                'imfit_ra': imfit_ra,
                'imfit_dec': imfit_dec,
                'cutout_path': temp_path,
                'elevation': elevation # <--- Added Elevation
            }

    except Exception:
        return None

def plot_cutout_panel(ax, freq_target, df_res, source_name):
    """Helper to plot a cutout panel with robust offsets and elevation."""
    df_res['freq_diff'] = abs(df_res['freq'] - freq_target)
    row = df_res.sort_values('freq_diff').iloc[0]
    
    try:
        with fits.open(row['cutout_path']) as h:
            data = h[0].data
            wcs = WCS(h[0].header).celestial
        
        ax.reset_wcs(wcs)
        ax.set_facecolor('black')
        
        vmin, vmax = np.percentile(data, [1, 99.5])
        im = ax.imshow(data, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
        
        # Expected Position
        exp_sky = SH_CATALOG[source_name]['coords']
        ax.plot_coord(exp_sky, '+', color='lime', ms=20, markeredgewidth=2, label='Expected')
        
        # Plot Measured Position (Prefer Imfit, then Imstat)
        meas_ra = row['imfit_ra'] if not np.isnan(row['imfit_ra']) else row['imstat_ra']
        meas_dec = row['imfit_dec'] if not np.isnan(row['imfit_dec']) else row['imstat_dec']
        
        stats_text = f"Freq: {row['freq']:.1f} MHz\n"
        if not np.isnan(row['elevation']):
            stats_text += f"El: {row['elevation']:.1f} deg\n"
        stats_text += f"Model: {row['model_flux']:.1f} Jy\n"

        if not np.isnan(meas_ra):
            meas_sky = SkyCoord(meas_ra, meas_dec, unit='deg')
            ax.plot_coord(meas_sky, 'x', color='red', ms=15, markeredgewidth=2, label='Measured')

            # Robust Offset Calculation using Astropy
            d_ra, d_dec = meas_sky.spherical_offsets_to(exp_sky)
            d_ra_as = -d_ra.to(u.arcsec).value 
            d_dec_as = -d_dec.to(u.arcsec).value
            
            if not np.isnan(row['imfit_flux']):
                stats_text += f"Gauss: {row['imfit_flux']:.1f} $\pm$ {row['imfit_err']:.1f} Jy\n"
            else:
                stats_text += f"Gauss: Fit Failed\n"
                
            stats_text += f"Peak:  {row['imstat_flux']:.1f} $\pm$ {row['imstat_err']:.1f} Jy\n"
            stats_text += f"$\Delta$RA: {d_ra_as:+.1f}\"\n"
            stats_text += f"$\Delta$Dec: {d_dec_as:+.1f}\""
            
        else:
            stats_text += "Fit Failed (Bad Data)"

        props = dict(boxstyle='round', facecolor='white', alpha=0.9)
        ax.text(0.05, 0.05, stats_text, transform=ax.transAxes, fontsize=10, 
                verticalalignment='bottom', bbox=props)

        ax.set_title(f"{row['freq']:.1f} MHz", color='black', fontsize=12)
        ax.coords[0].set_axislabel('RA'); ax.coords[1].set_axislabel('Dec')
        
    except:
        ax.text(0.5, 0.5, "No Data", ha='center', transform=ax.transAxes)

def make_summary_plot(source_name, df_res, save_dir, prefix=""):
    df_res = df_res.sort_values('freq')
    
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    
    # --- TOP: SPECTRUM ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_res['freq'], df_res['model_flux'], 'k--', label='Model (S&H 2012)', linewidth=2, zorder=1)
    
    # STRICT FILTER: Gaussian Fit (Imfit)
    mask_fit = np.isfinite(df_res['imfit_flux']) & np.isfinite(df_res['imfit_err']) & (df_res['imfit_err'] > 0)
    df_fit = df_res[mask_fit]
    
    if not df_fit.empty:
        ax1.errorbar(df_fit['freq'], df_fit['imfit_flux'], yerr=df_fit['imfit_err'], 
                     fmt='rx', label='Gaussian Fit (Imfit)', markersize=8, markeredgewidth=2, zorder=2, capsize=3)
    
    # STRICT FILTER: Max Pixel (Imstat)
    mask_stat = np.isfinite(df_res['imstat_flux']) & np.isfinite(df_res['imstat_err']) & (df_res['imstat_err'] > 0)
    df_stat = df_res[mask_stat]
    
    if not df_stat.empty:
        ax1.errorbar(df_stat['freq'], df_stat['imstat_flux'], yerr=df_stat['imstat_err'],
                     fmt='bo', label='Max Pixel (Imstat)', markersize=6, alpha=0.6, zorder=3, capsize=3)
    
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel('Frequency (MHz)', fontsize=12)
    ax1.set_ylabel('Flux Density (Jy)', fontsize=12)
    ax1.set_title(f"{source_name} Flux Scale Check", fontsize=14, fontweight='bold')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend(fontsize=12)

    # --- BOTTOM LEFT: 36 MHz ---
    ax2 = fig.add_subplot(gs[1, 0], projection=WCS())
    plot_cutout_panel(ax2, 36.0, df_res, source_name)
    
    # --- BOTTOM RIGHT: 73 MHz ---
    ax3 = fig.add_subplot(gs[1, 1], projection=WCS())
    plot_cutout_panel(ax3, 73.0, df_res, source_name)

    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{prefix}Flux_Check_Hybrid_{source_name}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")

def run_flux_check(run_dir):
    print("--- HYBRID FLUX CHECKER (Imfit + Imstat) ---")
    
    if not CASA_TASKS_AVAILABLE:
        print("CRITICAL: casatasks not installed/importable. Cannot run imfit/imstat.", file=sys.stderr)
        return

    # Extract date_LST prefix for unique QA filenames
    import re
    _m = re.search(r'/(\d+h)/(\d{4}-\d{2}-\d{2})/', run_dir.rstrip('/') + '/')
    prefix = f"{_m.group(2)}_{_m.group(1)}_" if _m else ""

    temp_dir = os.path.join(run_dir, "QA", "temp_flux_cutouts")
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Find deep I images: prefer PBcorr dewarped, then PBcorr, then raw
    images = []
    for pat_suffix in ['*pbcorr_dewarped.fits', '*pbcorr.fits', '*image*.fits']:
        pattern = os.path.join(run_dir, "*MHz", "I", "deep",
                               f"*Taper*Robust-0.75*{pat_suffix}")
        hits = sorted(glob.glob(pattern))
        hits = [img for img in hits if "NoTaper" not in img]
        if pat_suffix == '*pbcorr.fits':
            hits = [f for f in hits if 'dewarped' not in f]
        elif pat_suffix == '*image*.fits':
            hits = [f for f in hits if 'pbcorr' not in f and 'dewarped' not in f]
        if hits:
            # One image per subband  deduplicate
            seen_sb = set()
            for h in hits:
                sb = h.split('/')[-4]
                if sb not in seen_sb:
                    images.append(h)
                    seen_sb.add(sb)
            break
    
    results = []
    print(f"Processing {len(images)} bands...", end=" ", flush=True)

    for i, img in enumerate(images):
        subband = img.split('/')[-4]
        try: freq = float(subband.replace('MHz',''))
        except: continue
        print(".", end="", flush=True)
        
        for name, data in SH_CATALOG.items():
            res = process_cutout(img, name, data, temp_dir)
            if res:
                res['source'] = name
                res['freq'] = freq
                res['model_flux'] = get_model_flux(name, freq)
                results.append(res)

    print(" Done.")

    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(run_dir, "QA", f"{prefix}flux_check_hybrid.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved CSV: {csv_path}")
        
        qa_dir = os.path.join(run_dir, "QA")
        for name in df['source'].unique():
            subset = df[df['source'] == name].sort_values('freq')
            if len(subset) > 1:
                make_summary_plot(name, subset, qa_dir, prefix=prefix)
    else:
        print("\nNo valid measurements found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    run_flux_check(args.run_dir)
