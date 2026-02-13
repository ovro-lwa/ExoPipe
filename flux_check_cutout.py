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
    print("WARNING: casatasks not available.", file=sys.stderr)

CUTOUT_SIZE = 2.0 * u.deg; SEARCH_RADIUS = 0.25 * u.deg; MIN_ELEVATION = 20.0
try:
    from pipeline_config import OVRO_LOC
except ImportError:
    OVRO_LOC = EarthLocation(lat=37.23977727*u.deg, lon=-118.2816667*u.deg, height=1222*u.m)
warnings.simplefilter('ignore', category=FITSFixedWarning)

# Use the single source of truth for calibrator data.
# CALIB_DATA in pipeline_config uses Scaife & Heald 2012 coefficients.
try:
    from pipeline_config import CALIB_DATA as SH_CATALOG
except ImportError:
    SH_CATALOG = {
        '3C48':  {'coords': SkyCoord(24.422, 33.16, unit='deg'),   'coeffs': [64.768, -0.387, -0.420, 0.181]},
        '3C147': {'coords': SkyCoord(85.651, 49.852, unit='deg'),  'coeffs': [66.738, -0.022, -1.017, 0.549]},
        '3C196': {'coords': SkyCoord(123.400, 48.217, unit='deg'), 'coeffs': [83.084, -0.699, -0.110]},
        '3C286': {'coords': SkyCoord(202.785, 30.509, unit='deg'), 'coeffs': [27.477, -0.158, 0.032, -0.180]},
        '3C295': {'coords': SkyCoord(212.836, 52.203, unit='deg'), 'coeffs': [97.763, -0.582, -0.298, 0.583, -0.363]},
        '3C380': {'coords': SkyCoord(277.382, 48.746, unit='deg'), 'coeffs': [77.352, -0.767]},
    }

def get_model_flux(name, freq_mhz):
    A = SH_CATALOG[name]['coeffs']
    x = np.log10(freq_mhz / 150.0)
    log_s = np.log10(A[0])
    for i in range(1, len(A)):
        log_s += A[i] * (x ** i)
    return 10**log_s

def process_cutout(img_path, source_name, source_data, temp_dir):
    try:
        subband = img_path.split('/')[-4]
        with fits.open(img_path) as hdul:
            data = hdul[0].data.squeeze(); header = hdul[0].header; wcs = WCS(header).celestial
            try:
                el = source_data['coords'].transform_to(AltAz(obstime=Time(header['DATE-OBS'], format='isot', scale='utc'), location=OVRO_LOC)).alt.deg
                if el < MIN_ELEVATION: return None
            except Exception: el = np.nan
            
            # Create Cutout
            cutout = Cutout2D(data, source_data['coords'], (CUTOUT_SIZE, CUTOUT_SIZE), wcs=wcs, mode='trim')
            
            # Prepare Header for CASA imfit
            cut_header = cutout.wcs.to_header()
            cut_header['BUNIT'] = 'Jy/beam' # Fix for unrecognized intensity unit warnings
            
            # Copy Beam info from parent header
            for kw in ['BMAJ', 'BMIN', 'BPA']:
                if kw in header: cut_header[kw] = header[kw]
            
            temp_path = os.path.join(temp_dir, f"{source_name}_{subband}.fits")
            fits.writeto(temp_path, cutout.data, cut_header, overwrite=True)
            
            cx, cy = cutout.wcs.world_to_pixel(source_data['coords'])
            cdelt = np.abs(cutout.wcs.wcs.cdelt[0])
            r_pix = SEARCH_RADIUS.to(u.deg).value / cdelt
            ny, nx = cutout.data.shape
            x1, y1 = max(0, int(cx-r_pix)), max(0, int(cy-r_pix))
            x2, y2 = min(nx-1, int(cx+r_pix)), min(ny-1, int(cy+r_pix))
            box = f"{x1},{y1},{x2},{y2}"
            
            imfit_flux, imfit_err = np.nan, np.nan
            if CASA_TASKS_AVAILABLE:
                try:
                    fit = imfit(imagename=temp_path, box=box)
                    if fit and fit.get('converged') and 'results' in fit:
                        # Extract Integrated Flux
                        imfit_flux = fit['results']['component0']['flux']['value'][0]
                        imfit_err = fit['results']['component0']['flux']['error'][0]
                except Exception: pass
            
            return {'imfit_flux': imfit_flux, 'imfit_err': imfit_err, 'elevation': el}
    except Exception: return None

def run_flux_check(run_dir):
    print("--- HYBRID FLUX CHECKER ---")
    if not CASA_TASKS_AVAILABLE: print("CRITICAL: casatasks missing.", file=sys.stderr); return
    
    qa_dir = os.path.join(run_dir, "QA")
    os.makedirs(qa_dir, exist_ok=True)
    temp_dir = os.path.join(qa_dir, "temp_flux_cutouts"); os.makedirs(temp_dir, exist_ok=True)
    images = sorted(glob.glob(os.path.join(run_dir, "*MHz", "I", "deep", "*I-Deep-Taper-Robust-0.75*pbcorr_dewarped.fits")))
    
    if not images: print("No matching images found."); return
    print(f"Processing {len(images)} bands...", end=" ", flush=True)
    
    results = []
    for img in images:
        try: freq = float(img.split('/')[-4].replace('MHz',''))
        except Exception: continue
        print(".", end="", flush=True)
        for name, data in SH_CATALOG.items():
            res = process_cutout(img, name, data, temp_dir)
            if res:
                try:
                    res['model_flux'] = get_model_flux(name, freq)
                except Exception as e:
                    print(f"WARNING: Model flux failed for {name} @ {freq} MHz: {e}")
                    continue
                res['source'] = name; res['freq'] = freq
                results.append(res)
    print(" Done.")
    
    if results:
        csv_path = os.path.join(qa_dir, "flux_check_hybrid.csv")
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"\nSaved CSV: {csv_path}")

        # Generate diagnostic plot: measured vs model flux per source
        try:
            df = pd.DataFrame(results)
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Panel 1: Flux ratio vs frequency
            ax1 = axes[0]
            for src in df['source'].unique():
                sub = df[df['source'] == src].sort_values('freq')
                valid = sub['imfit_flux'].notna() & (sub['model_flux'] > 0)
                if valid.any():
                    ratio = sub.loc[valid, 'imfit_flux'] / sub.loc[valid, 'model_flux']
                    ax1.plot(sub.loc[valid, 'freq'], ratio, 'o-', label=src, ms=5)
            ax1.axhline(1.0, color='grey', ls='--', alpha=0.7)
            ax1.set_xlabel('Frequency (MHz)')
            ax1.set_ylabel('Measured / Model Flux')
            ax1.set_title('Flux Scale Check')
            ax1.legend(fontsize=8)
            ax1.set_ylim(0, 3)
            ax1.grid(True, alpha=0.3)

            # Panel 2: Measured vs Model scatter
            ax2 = axes[1]
            valid = df['imfit_flux'].notna() & (df['model_flux'] > 0)
            if valid.any():
                ax2.scatter(df.loc[valid, 'model_flux'], df.loc[valid, 'imfit_flux'],
                            c=df.loc[valid, 'freq'], cmap='viridis', s=30, edgecolors='k', lw=0.5)
                max_flux = max(df.loc[valid, 'model_flux'].max(), df.loc[valid, 'imfit_flux'].max())
                ax2.plot([0, max_flux * 1.1], [0, max_flux * 1.1], 'k--', alpha=0.5)
                ax2.set_xlabel('Model Flux (Jy)')
                ax2.set_ylabel('Measured Flux (Jy)')
                ax2.set_title('Measured vs Model')
                cb = plt.colorbar(ax2.collections[0], ax=ax2, label='Freq (MHz)')
                ax2.grid(True, alpha=0.3)

            png_path = os.path.join(qa_dir, "flux_check_diagnostic.png")
            plt.tight_layout()
            plt.savefig(png_path, dpi=150)
            plt.close()
            print(f"Saved diagnostic plot: {png_path}")
        except Exception as e:
            print(f"WARNING: Diagnostic plot failed: {e}")
            plt.close('all')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    run_flux_check(args.run_dir)
