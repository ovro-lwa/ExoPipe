import os
import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from extractor_pb_75 import BeamModel, BEAM_PATH

# Module-level singleton — loaded once, reused across all calls
_beam_model = None

def _get_beam():
    global _beam_model
    if _beam_model is None:
        _beam_model = BeamModel(BEAM_PATH)
    return _beam_model

def apply_pb_correction(fits_path):
    """
    Applies PB correction and saves a NEW file with suffix .pbcorr.fits
    Does NOT overwrite the original.
    SAFEGUARD: Checks header for 'PBCORR' to prevent double application.
    """
    if not os.path.exists(fits_path):
        print(f"Error: {fits_path} not found.")
        return

    # Skip files that are already corrected (based on filename)
    if fits_path.endswith('.pbcorr.fits'):
        print(f"[PB-Correct] Skipping {fits_path} (Already matches output suffix).")
        return

    try:
        # Check header first without loading data
        header_peek = fits.getheader(fits_path)
        if header_peek.get('PBCORR', False) is True:
             print(f"[PB-Correct] Skipping {fits_path} (Header indicates PBCORR already applied).")
             return

        beam = _get_beam()
        
        with fits.open(fits_path) as hdul:
            data = hdul[0].data
            header = hdul[0].header.copy() 
            
            original_shape = data.shape
            data_sq = data.squeeze()
            
            wcs = WCS(header).celestial
            obs_date = header.get('DATE-OBS')
            freq_hz = header.get('CRVAL3', header.get('RESTFRQ', 50e6))
            
            # Grid & Response
            h, w = data_sq.shape
            y_inds, x_inds = np.indices((h, w))
            ra_map, dec_map = wcs.all_pix2world(x_inds, y_inds, 0)
            
            resp = beam.get_response(ra_map.ravel(), dec_map.ravel(), obs_date, freq_hz)
            resp_map = resp.reshape((h, w))
            
            # Apply
            valid_mask = resp_map > 0.05
            corrected_data = np.zeros_like(data_sq)
            corrected_data[valid_mask] = data_sq[valid_mask] / resp_map[valid_mask]
            corrected_data[~valid_mask] = np.nan
            
            # Save to NEW filename
            final_data = corrected_data.reshape(original_shape)
            header['PBCORR'] = (True, 'Applied OVRO-LWA Beam')
            
            # Insert .pbcorr before .fits
            base, ext = os.path.splitext(fits_path)
            out_name = f"{base}.pbcorr{ext}"
            
            fits.writeto(out_name, final_data, header, overwrite=True)
            print(f"[PB-Correct] Saved corrected image to {os.path.basename(out_name)}")
            
    except Exception as e:
        print(f"[PB-Correct] Failed on {fits_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pb_correct.py <image.fits>")
        sys.exit(1)
    apply_pb_correction(sys.argv[1])
