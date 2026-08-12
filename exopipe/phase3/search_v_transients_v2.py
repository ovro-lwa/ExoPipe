#!/usr/bin/env python3
"""
search_v_transients_v2.py  Stokes V transient search pipeline (self-contained)

Three contamination sources modeled and subtracted:
  1. SIDELOBES  PSF template (flux-weighted V/I stacks, elevation-binned)
  2. BILOBES  XY-delay (tiled templates, nearest-neighbor fallback)
  3. STRIATIONS  RA-aligned noise (running median along Dec strips)

Then candidate extraction with local SNR, exclusion zones, bilobe
cross-match, bright-source proximity, tiered sidelobe exclusion.

Usage:
  python search_v_transients_v2.py \
      --v_image V.fits --i_image I.fits \
      --out_dir QA/search/ --catalog catalog.csv \
      --el_bins --save_intermediates
"""
import argparse, os, sys, csv, re, time, tempfile, shutil
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import mad_std
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time, TimeDelta
from astropy import units as u
from scipy.ndimage import maximum_filter
import warnings
warnings.filterwarnings('ignore')

try:
    from exopipe.common.pipeline_config import OVRO_LOC
except ImportError:
    OVRO_LOC = EarthLocation(lat=37.23977727*u.deg, lon=-118.2816667*u.deg,
                             height=1222*u.m)

class _Tee:
    """Duplicate stdout to a log file."""
    def __init__(self, path):
        self.file = open(path, 'w')
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        sys.stdout = self.stdout
        self.file.close()

CONFIG = {
    'MIN_ELEVATION': 30.0, 'I_MIN_ELEVATION': 29.5,
    'CANDIDATE_SIGMA': 5.0, 'BDSF_THRESH_PIX_V': 4.5,
    'A_TEAM': {
        'Cas A': {'coord': SkyCoord('23h23m24s', '+58d48m54s'), 'radius': 5.0*u.deg},
        'Cyg A': {'coord': SkyCoord('19h59m28s', '+40d44m02s'), 'radius': 5.0*u.deg},
        'Tau A': {'coord': SkyCoord('05h34m31s', '+22d00m52s'), 'radius': 5.0*u.deg},
        'Vir A': {'coord': SkyCoord('12h30m49s', '+12d23m28s'), 'radius': 10.0*u.deg},
        'Hydra A': {'coord': SkyCoord('09h18m05.7s', '-12d05m44s'), 'radius': 5.0*u.deg},
    },
    'BDSF_THRESH_ISL': 5.0, 'I_MIN_SIGMA': 10.0,
    'BIREF_N_BRIGHT': 100, 'BIREF_MAX_FLUX': 25.0,
    'SIDELOBE_TIERS': [(100.0, 1.25), (70.0, 1.0), (40.0, 0.5)],
    'MERGE_RADIUS_BEAMS': 1.0, 'CATALOG_MATCH_RADIUS': 2.0*u.arcmin,
}

# ================================================================
# BEAM MODEL
# ================================================================
_BEAM_MODEL = None; _BEAM_MODEL_TRIED = False

def _get_beam_model():
    global _BEAM_MODEL, _BEAM_MODEL_TRIED
    if _BEAM_MODEL_TRIED: return _BEAM_MODEL
    _BEAM_MODEL_TRIED = True
    try:
        from exopipe.phase2.extractor_pb_75 import BeamModel, BEAM_PATH
        if os.path.exists(BEAM_PATH): _BEAM_MODEL = BeamModel(BEAM_PATH)
    except Exception: pass
    return _BEAM_MODEL

def _pb_at_coords(ra_deg, dec_deg, obs_time, freq_hz):
    bm = _get_beam_model()
    if bm is not None:
        try: return np.asarray(bm.get_response(ra_deg, dec_deg, obs_time, freq_hz), dtype=float)
        except Exception: pass
    coords = SkyCoord(np.atleast_1d(ra_deg), np.atleast_1d(dec_deg), unit='deg')
    altaz = coords.transform_to(AltAz(obstime=obs_time, location=OVRO_LOC))
    za = 90.0 - altaz.alt.deg
    resp = np.clip(np.cos(np.radians(np.clip(za, 0, 89.999)))**1.6, 1e-3, 1.0)
    return np.where(altaz.alt.deg < 10.0, np.nan, resp)

def _get_freq_mhz(header, fits_path=None):
    for n in (3, 4):
        if 'FREQ' in str(header.get(f'CTYPE{n}', '')).upper():
            val = header.get(f'CRVAL{n}')
            if val: return float(val) / 1e6
    for key in ('FREQ', 'RESTFRQ', 'RESTFREQ', 'CRVAL3'):
        val = header.get(key)
        if val and float(val) > 1e6: return float(val) / 1e6
    if fits_path:
        m = re.search(r'(?<!\d)(\d{2,3})MHz', os.path.basename(fits_path), re.IGNORECASE)
        if m: return float(m.group(1)) + 2.3
    return None

# ================================================================
# IMAGE LOADING AND MASKING
# ================================================================
def load_image(path):
    with fits.open(path) as hdul:
        return hdul[0].data.squeeze(), hdul[0].header

def get_midpoint_time(header, fits_path=None):
    duration = float(header.get('DURATION', 0.0))
    if duration <= 0 and fits_path:
        bn = os.path.basename(fits_path).lower()
        path_lower = (fits_path or '').lower()
        if '10min' in bn or '10min' in path_lower:
            duration = 600.0
        elif 'deep' in bn or '/deep/' in path_lower:
            duration = 3600.0
    if duration <= 0:
        duration = 3600.0  # safe default  never use 0
    return Time(header.get('DATE-OBS'), format='isot', scale='utc') + TimeDelta(duration*0.5, format='sec')

def make_masks(header, shape, fits_path=None):
    wcs = WCS(header).celestial
    h, w = shape
    y, x = np.indices((h, w))
    ra, dec = wcs.all_pix2world(x, y, 0)
    coords = SkyCoord(ra, dec, unit='deg')
    t_mid = get_midpoint_time(header, fits_path)
    altaz = coords.transform_to(AltAz(obstime=t_mid, location=OVRO_LOC))
    el_deg = altaz.alt.deg
    horizon_mask = el_deg < CONFIG['MIN_ELEVATION']
    invalid_wcs_mask = np.isnan(ra) | np.isnan(dec) | np.isnan(el_deg)
    ateam_mask = np.zeros(shape, dtype=bool)
    for info in CONFIG['A_TEAM'].values():
        ateam_mask |= (coords.separation(info['coord']) < info['radius'])
    return horizon_mask, ateam_mask, invalid_wcs_mask, t_mid, coords, el_deg

def get_uniform_rms(data, horizon_mask):
    h, w = data.shape; half = min(512, h//4, w//4); cy, cx = h//2, w//2
    region = data[cy-half:cy+half, cx-half:cx+half].copy()
    region[horizon_mask[cy-half:cy+half, cx-half:cx+half]] = np.nan
    return mad_std(region, ignore_nan=True)

# ================================================================
# BDSF EXTRACTION
# ================================================================
def _run_bdsf(fits_path, thresh_pix, thresh_isl, quiet=True, rms_value=None, mean_map=None):
    import bdsf
    kw = dict(thresh_pix=thresh_pix, thresh_isl=thresh_isl, quiet=quiet)
    if rms_value is not None: kw['rms_value'] = rms_value; kw['rms_map'] = False
    else: kw['adaptive_rms_box'] = True
    if mean_map is not None: kw['mean_map'] = mean_map
    img = bdsf.process_image(fits_path, **kw)
    wcs_img = WCS(fits.getheader(fits_path)).celestial
    sources = []
    for s in img.sources:
        try: ra_d = float(s.posn_sky_centroid[0]); dec_d = float(s.posn_sky_centroid[1])
        except: ra_d = float(s.posn_sky_max[0]); dec_d = float(s.posn_sky_max[1])
        if np.isnan(ra_d) or np.isnan(dec_d): continue
        try: px, py = wcs_img.all_world2pix([[ra_d, dec_d]], 0)[0]; px, py = int(round(px)), int(round(py))
        except: continue
        sources.append({'coord': SkyCoord(ra_d, dec_d, unit='deg'), 'flux': float(s.peak_flux_max),
                        'total_flux': float(getattr(s, 'total_flux', s.peak_flux_max)),
                        'snr': float(s.peak_flux_max / getattr(s, 'rms', 1e-3)),
                        'pix_x': px, 'pix_y': py})
    return sources

def _filter_by_mask(sources, mask, shape):
    h, w = shape
    return [s for s in sources if 0 <= s['pix_y'] < h and 0 <= s['pix_x'] < w and not mask[s['pix_y'], s['pix_x']]]

# ================================================================
# V PEAK EXTRACTION
# ================================================================
def extract_v_peaks(v_data, v_header, horizon_mask, ateam_mask, invalid_wcs_mask, wcs, obs_time, bdsf_thresh, v_rms):
    candidates = []; post_mask = horizon_mask | ateam_mask | invalid_wcs_mask
    tmp_dir = tempfile.mkdtemp(prefix='bdsf_v_')
    try:
        for sign_label, sign_data, sign_mult in [('positive', v_data, 1), ('negated', -v_data, -1)]:
            path = os.path.join(tmp_dir, f'V_{sign_label}.fits')
            fits.writeto(path, sign_data, v_header, overwrite=True)
            print(f"    Running BDSF on V ({sign_label}, rms={v_rms*1000:.2f}mJy)...")
            try:
                srcs = _run_bdsf(path, thresh_pix=bdsf_thresh, thresh_isl=3.0, rms_value=v_rms, mean_map='zero')
                n_pre = len(srcs); srcs = _filter_by_mask(srcs, post_mask, v_data.shape)
                print(f"    BDSF {sign_label}: {n_pre} -> {len(srcs)} after mask")
            except Exception as e:
                print(f"    BDSF {sign_label} failed: {e}"); srcs = []
            for s in srcs:
                snr = sign_mult * s['flux'] / v_rms
                if abs(snr) < bdsf_thresh: continue
                try:
                    altaz = s['coord'].transform_to(AltAz(obstime=obs_time, location=OVRO_LOC))
                    el = altaz.alt.deg
                except: continue
                candidates.append({'coord': s['coord'], 'flux': sign_mult*s['flux'], 'snr': snr,
                                   'el': el, 'pix_x': s['pix_x'], 'pix_y': s['pix_y'],
                                   'n_pix': 0, 'status': 'PASS', 'reject_reason': ''})
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    candidates.sort(key=lambda c: abs(c['snr']), reverse=True)
    n_total = len(candidates)
    print(f"    Found {n_total} peaks ({sum(1 for c in candidates if c['snr']>0)} +, {sum(1 for c in candidates if c['snr']<0)} -)")
    return candidates

# ================================================================
# LOCAL SNR
# ================================================================
def assign_local_snr(candidates, v_data, exclusion_mask, beam_pix, sigma_thresh,
                     inner_beams=2.0, outer_beams=10.0, n_sectors=4, min_sector_pix=400):
    h, w = v_data.shape
    box = int(np.ceil(outer_beams * beam_pix))
    dy, dx = np.indices((2*box+1, 2*box+1), dtype=float); dy -= box; dx -= box
    d_sq = dx**2 + dy**2
    ann = (d_sq >= (inner_beams*beam_pix)**2) & (d_sq <= (outer_beams*beam_pix)**2)
    angles = np.arctan2(dx, dy) % (2*np.pi)
    sec_w = 2*np.pi/n_sectors
    sec_ids = (angles/sec_w).astype(int) % n_sectors
    ann_dy, ann_dx, ann_sec = dy[ann].astype(int), dx[ann].astype(int), sec_ids[ann]
    n_rej = 0
    for c in candidates:
        if c['status'] != 'PASS': continue
        c['global_snr'] = c['snr']
        cy, cx = c['pix_y'], c['pix_x']
        ay, ax, asec = cy+ann_dy, cx+ann_dx, ann_sec.copy()
        v = (ay>=0)&(ay<h)&(ax>=0)&(ax<w); ay,ax,asec = ay[v],ax[v],asec[v]
        nm = ~exclusion_mask[ay,ax]; ay,ax,asec = ay[nm],ax[nm],asec[nm]
        if len(ay)==0: c['local_rms']=np.nan; continue
        vals = v_data[ay,ax]
        mads = [float(mad_std(vals[asec==s])) for s in range(n_sectors) if np.sum(asec==s)>=min_sector_pix]
        if not mads:
            if len(vals)>=min_sector_pix: lr = float(mad_std(vals))
            else: c['local_rms']=np.nan; continue
        else: lr = min(mads)
        if lr <= 0: c['local_rms']=np.nan; continue
        c['local_rms'] = lr; lsnr = c['flux']/lr; c['snr'] = float(lsnr)
        if abs(lsnr) < sigma_thresh:
            c['status']='REJECT_LOCAL_RMS'; c['reject_reason']=f"Local SNR={lsnr:+.1f}"; n_rej+=1
    return n_rej

# ================================================================
# I SOURCE EXTRACTION
# ================================================================
def extract_i_sources(i_data, i_header, i_rms, wcs, horizon_mask, invalid_wcs_mask, min_sigma, beam_pix):
    post_mask = horizon_mask | invalid_wcs_mask
    tmp_dir = tempfile.mkdtemp(prefix='bdsf_i_')
    try:
        path = os.path.join(tmp_dir, 'I.fits'); fits.writeto(path, i_data, i_header, overwrite=True)
        print("    Running BDSF on I...")
        try:
            srcs = _run_bdsf(path, thresh_pix=min_sigma, thresh_isl=CONFIG['BDSF_THRESH_ISL'])
            n_pre = len(srcs); srcs = _filter_by_mask(srcs, post_mask, i_data.shape)
            print(f"    BDSF: {n_pre} total, {len(srcs)} above horizon")
        except Exception as e:
            print(f"    BDSF failed: {e}, fallback...")
            snr_map = i_data/i_rms; snr_map[horizon_mask]=0
            ms = max(int(beam_pix*1.5),5); lm = maximum_filter(snr_map,size=ms)
            py,px = np.where((snr_map==lm)&(snr_map>min_sigma)); srcs=[]
            for y,x in zip(py,px):
                try:
                    co = wcs.pixel_to_world(x,y)
                    if not np.isnan(co.ra.deg):
                        srcs.append({'pix_x':int(x),'pix_y':int(y),'coord':co,'flux':float(i_data[y,x]),
                                     'total_flux':float(i_data[y,x]),'snr':float(snr_map[y,x])})
                except: pass
    finally: shutil.rmtree(tmp_dir, ignore_errors=True)
    srcs.sort(key=lambda s: s['flux'], reverse=True)
    return srcs

def apply_pb_correction(i_sources, obs_time, freq_mhz):
    if freq_mhz is None:
        for s in i_sources: s['pb']=1.0; s['true_flux']=s.get('flux',0); s['true_total_flux']=s.get('total_flux',s.get('flux',0))
        return 0
    ra = np.array([s['coord'].ra.deg for s in i_sources]); dec = np.array([s['coord'].dec.deg for s in i_sources])
    try: pb_vals = _pb_at_coords(ra, dec, obs_time, freq_mhz*1e6)
    except:
        for s in i_sources: s['pb']=1.0; s['true_flux']=s.get('flux',0); s['true_total_flux']=s.get('total_flux',s.get('flux',0))
        return 0
    n=0
    for s, pb in zip(i_sources, pb_vals):
        pb = float(pb)
        if not np.isfinite(pb) or pb<=0: pb=1.0
        s['pb']=pb; s['true_flux']=s.get('flux',0)/pb; s['true_total_flux']=s.get('total_flux',s.get('flux',0))/pb; n+=1
    return n

# ================================================================
# SOURCE CLASSIFICATION
# ================================================================
def effective_apparent_flux(src):
    return max(src.get('flux', 0), src.get('total_flux', 0))

def classify_i_sources(i_sources, beam_pix, obs_time, total_peak_ratio=2.0, exclusion_flux=250.0, exclusion_radius_deg=2.0):
    n_pt=n_res=n_ex=0
    for s in i_sources:
        altaz = s['coord'].transform_to(AltAz(obstime=obs_time, location=OVRO_LOC))
        s['el_deg'] = float(altaz.alt.deg)
        app = effective_apparent_flux(s); pk = s.get('flux',0); tot = s.get('total_flux',pk)
        if app >= exclusion_flux: s['source_class']='exclusion'; s['exclusion_radius_deg']=exclusion_radius_deg; n_ex+=1
        elif pk>0 and tot>total_peak_ratio*pk: s['source_class']='resolved'; s['exclusion_radius_deg']=max(exclusion_radius_deg,0.5); n_res+=1
        else: s['source_class']='point'; n_pt+=1
    return n_pt, n_res, n_ex

# ================================================================
# XY-DELAY AXIS
# ================================================================
def measure_xy_delay_axis(v_data, i_sources, v_rms, beam_pix, n_bright, max_flux=None):
    if max_flux is None: max_flux = CONFIG.get('BIREF_MAX_FLUX', np.inf)
    eligible = sorted([s for s in i_sources if s['flux']<=max_flux], key=lambda s: s['flux'], reverse=True)[:n_bright]
    if not eligible: return None, None, None, None, None, None, None
    print(f"    Using {len(eligible)} I sources ({eligible[-1]['flux']:.1f}{eligible[0]['flux']:.1f} Jy, cap={max_flux:.0f} Jy)")
    pp, np_, pd, nd = [], [], [], []
    vi_pos, vi_neg = [], []  # V/I ratios per lobe
    sr = int(4*beam_pix); h, w = v_data.shape
    for src in eligible:
        sy,sx = src['pix_y'],src['pix_x']; i_flux = effective_apparent_flux(src)
        y1,y2 = max(0,sy-sr),min(h,sy+sr+1); x1,x2 = max(0,sx-sr),min(w,sx+sr+1)
        if y2-y1<3 or x2-x1<3: continue
        vs = v_data[y1:y2,x1:x2].copy(); yy,xx = np.indices(vs.shape)
        cl,cr = sy-y1,sx-x1; dist = np.sqrt((xx-cr)**2+(yy-cl)**2)
        mask = (dist<0.5*beam_pix)|(dist>sr)
        vp = vs.copy(); vp[mask]=-np.inf
        if np.any(vp>-np.inf):
            idx = np.unravel_index(np.argmax(vp),vp.shape)
            if vs[idx]>3*v_rms:
                dy,dx = (y1+idx[0])-sy,(x1+idx[1])-sx
                pp.append(float(np.degrees(np.arctan2(dx,dy))%360)); pd.append(float(np.sqrt(dx**2+dy**2)))
                if i_flux > 0: vi_pos.append(float(vs[idx]/i_flux*100))
        vn = vs.copy(); vn[mask]=np.inf
        if np.any(vn<np.inf):
            idx = np.unravel_index(np.argmin(vn),vn.shape)
            if vs[idx]<-3*v_rms:
                dy,dx = (y1+idx[0])-sy,(x1+idx[1])-sx
                np_.append(float(np.degrees(np.arctan2(dx,dy))%360)); nd.append(float(np.sqrt(dx**2+dy**2)))
                if i_flux > 0: vi_neg.append(float(vs[idx]/i_flux*100))
    if not pp: return None, None, None, None, None, None, None
    ad = pd+nd
    return (float(np.median(pp)), float(np.median(np_)), float(mad_std(pp)),
            float(np.median(ad)), float(mad_std(ad)),
            vi_pos, vi_neg)

# ================================================================
# STACKING
# ================================================================
def stack_cutouts(v_data, i_sources, exclusion_mask, cutout_half_pix, min_flux, max_flux,
                  min_el=None, max_el=None, flux_type='apparent', source_indices=None):
    h, w = v_data.shape; size = 2*cutout_half_pix+1
    sum_sv = np.zeros((size,size), dtype=np.float64)
    sum_s2 = np.zeros((size,size), dtype=np.float64)
    n_pix = np.zeros((size,size), dtype=np.int32)
    pool = [i_sources[i] for i in source_indices] if source_indices is not None else i_sources
    selected = []
    for s in pool:
        if s.get('source_class') != 'point': continue
        app = effective_apparent_flux(s)
        if not (min_flux <= app < max_flux): continue
        if min_el is not None and s.get('el_deg',90)<min_el: continue
        if max_el is not None and s.get('el_deg',0)>=max_el: continue
        selected.append(s)
    if not selected: return np.zeros((size,size)), np.zeros((size,size),dtype=int), 0, []
    for s in selected:
        sy,sx = s['pix_y'],s['pix_x']; flux = effective_apparent_flux(s)
        y1=max(0,sy-cutout_half_pix); y2=min(h,sy-cutout_half_pix+size)
        x1=max(0,sx-cutout_half_pix); x2=min(w,sx-cutout_half_pix+size)
        ky1=y1-(sy-cutout_half_pix); ky2=size-((sy-cutout_half_pix+size)-y2)
        kx1=x1-(sx-cutout_half_pix); kx2=size-((sx-cutout_half_pix+size)-x2)
        if y2<=y1 or x2<=x1: continue
        vc = v_data[y1:y2,x1:x2]; mc = exclusion_mask[y1:y2,x1:x2]; v = ~mc
        if np.sum(v)==0: continue
        sum_sv[ky1:ky2,kx1:kx2][v] += flux*vc[v]
        sum_s2[ky1:ky2,kx1:kx2][v] += flux*flux
        n_pix[ky1:ky2,kx1:kx2][v] += 1
    g = sum_s2>0; vfrac = np.zeros((size,size),dtype=np.float64); vfrac[g]=sum_sv[g]/sum_s2[g]
    return vfrac, n_pix, len(selected), [effective_apparent_flux(s) for s in selected]

# ================================================================
# PSF TEMPLATE
# ================================================================
def build_psf_template(v_data, i_sources, exclusion_mask, cutout_half_pix, beam_pix, cdelt_deg,
                       min_flux=10.0, max_flux=200.0, bilobe_mask_arcmin=15.0, min_el=None, max_el=None):
    el_str = f", el {min_el or 'any'}-{max_el or 'any'}°" if (min_el or max_el) else ""
    print(f"  Building PSF template ({min_flux:.0f}-{max_flux:.0f} Jy{el_str})...")
    vfrac, n_map, n_src, _ = stack_cutouts(v_data, i_sources, exclusion_mask, cutout_half_pix,
                                            min_flux, max_flux, min_el=min_el, max_el=max_el)
    if n_src == 0:
        sz = 2*cutout_half_pix+1; return np.zeros((sz,sz)), np.zeros((sz,sz)), 0
    tpl_full = vfrac.copy(); sz = vfrac.shape[0]; c = cutout_half_pix
    yy,xx = np.indices((sz,sz),dtype=float); dist = np.sqrt((xx-c)**2+(yy-c)**2)
    tpl = vfrac.copy(); tpl[dist < bilobe_mask_arcmin/(cdelt_deg*60)] = 0.0
    tpl[n_map < max(3,n_src//10)] = 0.0
    print(f"    {n_src} sources")
    return tpl, tpl_full, n_src

# ================================================================
# SIDELOBE PREDICTION
# ================================================================
def build_sidelobe_prediction(v_shape, i_sources, template, cutout_half_pix, el_templates=None):
    h,w = v_shape; sz = 2*cutout_half_pix+1; pred = np.zeros((h,w),dtype=np.float64)
    for s in i_sources:
        if s.get('source_class')!='point': continue
        sy,sx = s['pix_y'],s['pix_x']; flux = effective_apparent_flux(s)
        if flux<=0: continue
        tpl = template
        if el_templates:
            el = s.get('el_deg',90)
            for lo,hi,t in el_templates:
                if lo<=el<hi: tpl=t; break
        y1=sy-cutout_half_pix; y2=y1+sz; x1=sx-cutout_half_pix; x2=x1+sz
        ky1,ky2,kx1,kx2 = 0,sz,0,sz
        if y1<0: ky1=-y1;y1=0
        if y2>h: ky2=sz-(y2-h);y2=h
        if x1<0: kx1=-x1;x1=0
        if x2>w: kx2=sz-(x2-w);x2=w
        if y2<=y1 or x2<=x1: continue
        pred[y1:y2,x1:x2] += flux*tpl[ky1:ky2,kx1:kx2]
    return pred

# ================================================================
# BILOBE PREDICTION
# ================================================================
def build_bilobe_prediction(v_shape, i_sources, global_tpl, tile_tpls, fb_map, grid_n, cutout_half_pix):
    h,w = v_shape; sz = 2*cutout_half_pix+1; th,tw = h/grid_n,w/grid_n
    pred = np.zeros((h,w),dtype=np.float64); nd=nn=ng=0
    for s in i_sources:
        if s.get('source_class')!='point': continue
        sy,sx = s['pix_y'],s['pix_x']; flux = effective_apparent_flux(s)
        if flux<=0: continue
        ty,tx = min(int(sy/th),grid_n-1),min(int(sx/tw),grid_n-1); key=(ty,tx)
        if key in tile_tpls: tpl=tile_tpls[key]; nd+=1
        elif key in fb_map: tpl=tile_tpls[fb_map[key]]; nn+=1
        else: tpl=global_tpl; ng+=1
        y1=sy-cutout_half_pix; y2=y1+sz; x1=sx-cutout_half_pix; x2=x1+sz
        ky1,ky2,kx1,kx2 = 0,sz,0,sz
        if y1<0: ky1=-y1;y1=0
        if y2>h: ky2=sz-(y2-h);y2=h
        if x1<0: kx1=-x1;x1=0
        if x2>w: kx2=sz-(x2-w);x2=w
        if y2<=y1 or x2<=x1: continue
        pred[y1:y2,x1:x2] += flux*tpl[ky1:ky2,kx1:kx2]
    print(f"    {nd+nn+ng} stamped ({nd} direct, {nn} neighbor, {ng} global)")
    return pred

# ================================================================
# STRIATION MODEL
# ================================================================
def build_striation_model(v_data, wcs, exclusion_mask, ra_window_beams=20, beam_pix=7.0):
    h,w = v_data.shape; y_idx,x_idx = np.indices((h,w))
    ra_map = np.full((h,w),np.nan,dtype=np.float64); dec_map = np.full((h,w),np.nan,dtype=np.float64)
    for y0 in range(0,h,512):
        ye = min(y0+512,h)
        try: r,d = wcs.all_pix2world(x_idx[y0:ye],y_idx[y0:ye],0); ra_map[y0:ye]=r; dec_map[y0:ye]=d
        except: pass
    valid = ~exclusion_mask & np.isfinite(dec_map) & np.isfinite(ra_map)
    if np.sum(valid)==0: return np.zeros_like(v_data), 0
    dmin,dmax = np.nanmin(dec_map[valid]),np.nanmax(dec_map[valid])
    cd = abs(wcs.wcs.cdelt[1]) if hasattr(wcs.wcs,'cdelt') else 0.03125
    de = np.arange(dmin,dmax+cd,cd); nb = len(de)-1
    dbi = np.clip(np.digitize(dec_map,de)-1,0,nb-1)
    wp = int(ra_window_beams*beam_pix);
    if wp%2==0: wp+=1
    hw = wp//2
    print(f"    {nb} Dec strips, window={wp} pix")
    model = np.zeros_like(v_data); na=0
    for b in range(nb):
        ib = (dbi==b)&valid; n=np.sum(ib)
        if n<20: continue
        na+=1; py,px = np.where(ib); vals=v_data[py,px]; ras=ra_map[py,px]
        o = np.argsort(ras); py,px,vals = py[o],px[o],vals[o]; ns=len(vals)
        if ns<wp: model[py,px]=float(np.median(vals))
        else:
            mv = np.zeros(ns)
            for i in range(ns): mv[i]=np.median(vals[max(0,i-hw):min(ns,i+hw+1)])
            model[py,px]=mv
    print(f"    Active: {na}/{nb}")
    return model, nb

# ================================================================
# EXCLUSION AND REJECTION
# ================================================================
def build_exclusion_zone_mask(v_shape, i_sources, cdelt_deg):
    h,w = v_shape; em = np.zeros((h,w),dtype=bool)
    for s in i_sources:
        if s.get('source_class') not in ('resolved','exclusion'): continue
        sy,sx = s['pix_y'],s['pix_x']; rp = s.get('exclusion_radius_deg',2.0)/cdelt_deg
        y1=max(0,int(sy-rp)); y2=min(h,int(sy+rp)+1); x1=max(0,int(sx-rp)); x2=min(w,int(sx+rp)+1)
        if y2<=y1 or x2<=x1: continue
        yy=np.arange(y1,y2,dtype=float)-sy; xx=np.arange(x1,x2,dtype=float)-sx
        XX,YY = np.meshgrid(xx,yy); em[y1:y2,x1:x2][np.sqrt(XX**2+YY**2)<rp]=True
    return em

def reject_exclusion_zones(candidates, excl_mask):
    h,w = excl_mask.shape; n=0
    for c in candidates:
        if c['status']!='PASS': continue
        cy,cx = c['pix_y'],c['pix_x']
        if 0<=cy<h and 0<=cx<w and excl_mask[cy,cx]:
            c['status']='REJECT_EXCLUSION'; c['reject_reason']='Exclusion zone'; n+=1
    return n

def reject_bilobe_crossmatch(candidates, i_sources, i_data, i_rms, beam_pix, offset_pix,
                              pa_pos, pa_neg, pa_tol=45.0, search_radius_pix=None, min_i_flux=10.0, min_i_sigma=3.0):
    h,w = i_data.shape; it = min_i_sigma*i_rms
    if search_radius_pix is None: search_radius_pix = 1.5*beam_pix
    sr_sq = search_radius_pix**2; n=0
    bm = np.array([effective_apparent_flux(s)>=min_i_flux for s in i_sources])
    ipx = np.array([s['pix_x'] for s in i_sources],dtype=float)[bm]
    ipy = np.array([s['pix_y'] for s in i_sources],dtype=float)[bm]
    ifl = np.array([effective_apparent_flux(s) for s in i_sources])[bm]
    for c in candidates:
        if c['status']!='PASS': continue
        cy,cx = c['pix_y'],c['pix_x']; epa = pa_pos if c['snr']>0 else pa_neg
        dx=cx-ipx; dy=cy-ipy; dsq=dx**2+dy**2; ir=np.where(dsq<sr_sq)[0]
        matched=False
        for idx in ir:
            idx=int(idx)
            if dsq[idx]<1: continue
            pa=float(np.degrees(np.arctan2(dx[idx],dy[idx]))%360)
            pd=abs(pa-epa)%360
            if pd>180: pd=360-pd
            if pd<pa_tol:
                c['status']='REJECT_BILOBE'; c['reject_reason']=f"Bilobe: I {ifl[idx]:.1f}Jy, PA={pa:.0f}°"
                n+=1; matched=True; break
        if matched: continue
        # Direct image check at expected parent
        pr=np.radians(pa_neg if c['snr']>0 else pa_pos)
        epy=int(round(cy+offset_pix*np.cos(pr))); epx=int(round(cx+offset_pix*np.sin(pr)))
        sr=max(int(np.ceil(beam_pix*0.5)),3)
        py1=max(0,epy-sr);py2=min(h,epy+sr+1);px1=max(0,epx-sr);px2=min(w,epx+sr+1)
        if py2>py1 and px2>px1:
            st=i_data[py1:py2,px1:px2]
            if st.size>0:
                pv=float(np.max(st))
                if pv>=it and pv>=min_i_flux:
                    pi=np.unravel_index(np.argmax(st),st.shape)
                    dxc,dyc = cx-(px1+pi[1]),cy-(py1+pi[0])
                    if dxc!=0 or dyc!=0:
                        pac=float(np.degrees(np.arctan2(dxc,dyc))%360)
                        pd=abs(pac-epa)%360
                        if pd>180: pd=360-pd
                        if pd<pa_tol:
                            c['status']='REJECT_BILOBE'; c['reject_reason']=f"Bilobe: I peak {pv:.1f}Jy"; n+=1
    return n

# ================================================================
# MERGE AND CATALOG
# ================================================================
def merge_close_multiples(candidates, beam_arcmin, merge_radius_beams):
    mr = merge_radius_beams*beam_arcmin*u.arcmin
    passed = sorted([c for c in candidates if c['status']=='PASS'], key=lambda c: abs(c['snr']), reverse=True)
    n=0
    for i in range(len(passed)):
        if passed[i]['status']!='PASS': continue
        for j in range(i+1,len(passed)):
            if passed[j]['status']!='PASS': continue
            if passed[i]['coord'].separation(passed[j]['coord'])<mr:
                passed[j]['status']='REJECT_MERGED'; passed[j]['reject_reason']=f"Merged with SNR={passed[i]['snr']:+.1f}"; n+=1
    return n

def crossmatch_catalog(candidates, catalog_path):
    if not catalog_path or not os.path.exists(catalog_path): return
    try:
        import pandas as pd
        try: df = pd.read_csv(catalog_path)
        except: df = pd.read_csv(catalog_path, sep=r'\s+')
        df.columns = [c.lower() for c in df.columns]
        rak = 'ra_current' if 'ra_current' in df.columns else 'ra_deg'
        dek = 'dec_current' if 'dec_current' in df.columns else 'dec_deg'
        nk = 'common_name' if 'common_name' in df.columns else 'name'
        if rak not in df.columns: return
        cc = SkyCoord(df[rak].values, df[dek].values, unit='deg'); cn = df[nk].values
        for c in candidates:
            if c['status']!='PASS': continue
            idx,d2d,_ = c['coord'].match_to_catalog_sky(cc)
            c['catalog_match'] = f"{cn[int(idx)]} ({float(d2d.arcmin):.1f}')" if d2d<CONFIG['CATALOG_MATCH_RADIUS'] else ''
    except Exception as e: print(f"  Catalog failed: {e}")

# ================================================================
# OUTPUT
# ================================================================
def write_output(candidates, out_dir, label):
    csv_path = os.path.join(out_dir, f"search_{label}.csv")
    flds = ['status','reject_reason','ra_deg','dec_deg','ra_hms','dec_dms','snr','global_snr',
            'local_rms_mjy','flux_jy','el_deg','nearest_i_sep','nearest_i_snr','catalog_match']
    with open(csv_path,'w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=flds,extrasaction='ignore'); w.writeheader()
        for c in candidates:
            lr = c.get('local_rms',np.nan)
            w.writerow({'status':c['status'],'reject_reason':c.get('reject_reason',''),
                'ra_deg':f"{c['coord'].ra.deg:.5f}",'dec_deg':f"{c['coord'].dec.deg:.5f}",
                'ra_hms':c['coord'].ra.to_string(unit=u.hour,sep=':',precision=1,pad=True),
                'dec_dms':c['coord'].dec.to_string(unit=u.deg,sep=':',precision=1,alwayssign=True,pad=True),
                'snr':f"{c['snr']:.1f}",'global_snr':f"{c.get('global_snr',c['snr']):.1f}",
                'local_rms_mjy':f"{lr*1000:.2f}" if np.isfinite(lr) else "",
                'flux_jy':f"{c['flux']:.4f}",'el_deg':f"{c['el']:.1f}",
                'nearest_i_sep':f"{c.get('nearest_i_sep',np.nan):.1f}",
                'nearest_i_snr':f"{c.get('nearest_i_snr',0):.0f}",
                'catalog_match':c.get('catalog_match','')})
    print(f"  CSV: {csv_path}")
    # Regions
    reg = os.path.join(out_dir,f"search_{label}.reg")
    cm = {'PASS':'green','REJECT_LOCAL_RMS':'cyan','REJECT_EXCLUSION':'orange',
          'REJECT_BILOBE':'red','REJECT_BRIGHT_PROXIMITY':'yellow','REJECT_SIDELOBE_TIER':'magenta','REJECT_MERGED':'gray'}
    with open(reg,'w') as f:
        f.write("# Region file format: DS9 version 4.1\nfk5\n")
        for c in candidates:
            co = 'green' if c['snr']>0 else 'blue' if c['status']=='PASS' else cm.get(c['status'],'white')
            f.write(f"circle({c['coord'].ra.deg:.5f},{c['coord'].dec.deg:.5f},24') # color={co} text={{{c['snr']:+.1f}}}\n")
    print(f"  Regions: {reg}")
    passed = [c for c in candidates if c['status']=='PASS']
    if passed:
        pr = os.path.join(out_dir,f"passed_{label}.reg")
        with open(pr,'w') as f:
            f.write("# Region file format: DS9 version 4.1\nfk5\n")
            for c in passed:
                co = 'green' if c['snr']>0 else 'blue'
                m = c.get('catalog_match',''); lbl = f"SNR={c['snr']:+.1f}"
                if m: lbl = f"{m} {lbl}"
                f.write(f"circle({c['coord'].ra.deg:.5f},{c['coord'].dec.deg:.5f},24') # color={co} text={{{lbl}}}\n")
        print(f"  PASS regions: {pr}")

def write_stack_fits(stack, cdelt_deg, out_path, label, cutout_half_pix):
    hdr = fits.Header(); sz = 2*cutout_half_pix+1; ctr = cutout_half_pix+1
    hdr['NAXIS']=2;hdr['NAXIS1']=sz;hdr['NAXIS2']=sz
    hdr['CRPIX1']=float(ctr);hdr['CRPIX2']=float(ctr);hdr['CRVAL1']=0.0;hdr['CRVAL2']=0.0
    hdr['CDELT1']=cdelt_deg*60;hdr['CDELT2']=cdelt_deg*60
    hdr['CTYPE1']='DX---CAR';hdr['CTYPE2']='DY---CAR';hdr['CUNIT1']='arcmin';hdr['CUNIT2']='arcmin'
    fits.writeto(out_path, stack.astype(np.float32), hdr, overwrite=True)

# ================================================================
# MAIN PIPELINE
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Stokes V transient search v2")
    parser.add_argument("--v_image", required=True); parser.add_argument("--i_image", required=True)
    parser.add_argument("--catalog", default=None); parser.add_argument("--out_dir", default=None)
    parser.add_argument("--sigma", type=float, default=5.0)
    parser.add_argument("--sidelobe_radius_deg", type=float, default=2.0)
    parser.add_argument("--bilobe_mask_arcmin", type=float, default=15.0)
    parser.add_argument("--min_template_flux", type=float, default=10.0)
    parser.add_argument("--max_template_flux", type=float, default=200.0)
    parser.add_argument("--el_bins", action='store_true')
    parser.add_argument("--bilobe_radius_arcmin", type=float, default=15.0)
    parser.add_argument("--bilobe_grid", type=int, default=5)
    parser.add_argument("--bilobe_min_sources", type=int, default=10)
    parser.add_argument("--striation_window_beams", type=float, default=20.0)
    parser.add_argument("--pa_tolerance", type=float, default=45.0)
    parser.add_argument("--freq_mhz", type=float, default=None)
    parser.add_argument("--beam", type=float, default=None)
    parser.add_argument("--save_intermediates", action='store_true')
    args = parser.parse_args()
    t0 = time.time()

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(args.v_image))),"detections")
    os.makedirs(out_dir, exist_ok=True)
    v_base = os.path.splitext(os.path.basename(args.v_image))[0]

    # Tee stdout to log file
    log_path = os.path.join(out_dir, f"search_{v_base}.log")
    tee = _Tee(log_path)
    sys.stdout = tee

    print(f"\n{'='*65}\n  STOKES V TRANSIENT SEARCH v2\n{'='*65}")
    print(f"  V: {os.path.basename(args.v_image)}\n  I: {os.path.basename(args.i_image)}")
    print(f"  Output: {out_dir}")

    # PHASE 1: SETUP
    print(f"\n{'-'*65}\n  PHASE 1: SETUP\n{'-'*65}")
    v_data, v_hdr = load_image(args.v_image); i_data, i_hdr = load_image(args.i_image)
    wcs = WCS(v_hdr).celestial
    if v_data.shape != i_data.shape: print("ERROR: shape mismatch"); sys.exit(1)
    bmaj = v_hdr.get('BMAJ',0)
    if bmaj<=0:
        bmaj=i_hdr.get('BMAJ',0)
        if bmaj>0: v_hdr['BMAJ']=i_hdr['BMAJ']; v_hdr['BMIN']=i_hdr.get('BMIN',bmaj); v_hdr['BPA']=i_hdr.get('BPA',0)
    cd = abs(wcs.wcs.cdelt[0]) if hasattr(wcs.wcs,'cdelt') else 0.03125
    if args.beam: bmaj=args.beam/60.0
    if bmaj<=0: bmaj=5*cd
    bpix=bmaj/cd; bam=bmaj*60
    print(f"  Beam = {bam:.1f}' ({bpix:.1f} pix)")
    hm,am,iwm,obs_time,all_coords,el_deg = make_masks(v_hdr,v_data.shape,fits_path=args.v_image)
    em = hm|am|iwm; nv = np.sum(~(hm|iwm))
    vrms = get_uniform_rms(v_data,hm); irms = get_uniform_rms(i_data,hm)
    print(f"  V RMS={vrms*1000:.2f}mJy, I RMS={irms*1000:.2f}mJy")
    iel = CONFIG['I_MIN_ELEVATION']; ihm = hm.copy()
    if iel<CONFIG['MIN_ELEVATION']: ihm = (el_deg<iel)|iwm
    print(f"  Extracting I sources...")
    isrc = extract_i_sources(i_data,i_hdr,irms,wcs,ihm,iwm,CONFIG['I_MIN_SIGMA'],bpix)
    print(f"  Found {len(isrc)} I sources")
    fm = args.freq_mhz or _get_freq_mhz(v_hdr,args.v_image) or _get_freq_mhz(i_hdr,args.i_image)
    if fm: apply_pb_correction(isrc,obs_time,fm)
    npt,nre,nex = classify_i_sources(isrc,bpix,obs_time)
    print(f"  Point:{npt} Resolved:{nre} Exclusion:{nex}")
    ic = SkyCoord([s['coord'] for s in isrc])

    # PHASE 2: SIDELOBE SUBTRACTION
    print(f"\n{'-'*65}\n  PHASE 2: SIDELOBE SUBTRACTION\n{'-'*65}")
    t2=time.time(); slh = int(np.ceil(args.sidelobe_radius_deg/cd))
    elt = None
    if args.el_bins:
        elt = []
        for lo,hi in [(30,60),(60,90)]:
            t,_,n = build_psf_template(v_data,isrc,em,slh,bpix,cd,args.min_template_flux,args.max_template_flux,args.bilobe_mask_arcmin,min_el=lo,max_el=hi)
            elt.append((lo,hi,t))
            if args.save_intermediates: write_stack_fits(t,cd,os.path.join(out_dir,f"psf_tpl_el{lo}-{hi}.fits"),f"el{lo}-{hi} N={n}",slh)
    slt,sltf,nt = build_psf_template(v_data,isrc,em,slh,bpix,cd,args.min_template_flux,args.max_template_flux,args.bilobe_mask_arcmin)
    if nt==0: print("ERROR: no template"); sys.exit(1)
    if args.save_intermediates: write_stack_fits(slt,cd,os.path.join(out_dir,"psf_template.fits"),f"N={nt}",slh)
    print(f"  Building sidelobe prediction...")
    vslp = build_sidelobe_prediction(v_data.shape,isrc,slt,slh,el_templates=elt)
    vr1 = v_data - vslp; rms1 = get_uniform_rms(vr1, hm)
    print(f"  RMS: {vrms*1000:.2f} -> {rms1*1000:.2f} mJy")
    if args.save_intermediates:
        v_hdr_out = fits.getheader(args.v_image)
        fits.writeto(os.path.join(out_dir,f"{v_base}_residual_sidelobe.fits"),vr1.astype(np.float32),v_hdr_out,overwrite=True)
        fits.writeto(os.path.join(out_dir,f"{v_base}_sidelobe_model.fits"),vslp.astype(np.float32),v_hdr_out,overwrite=True)
    print(f"  Phase 2: {time.time()-t2:.0f}s")

    # PHASE 3: BILOBE SUBTRACTION
    print(f"\n{'-'*65}\n  PHASE 3: BILOBE SUBTRACTION\n{'-'*65}")
    t3=time.time(); blh = int(np.ceil(args.bilobe_radius_arcmin/(cd*60))); gn = args.bilobe_grid
    print(f"  Measuring bilobe axis...")
    pap,pan,pas,opix,ost,vi_pos,vi_neg = measure_xy_delay_axis(vr1,isrc,rms1,bpix,CONFIG['BIREF_N_BRIGHT'],max_flux=CONFIG['BIREF_MAX_FLUX'])
    if pap is not None:
        print(f"  PA: +{pap:.0f}°/-{pan:.0f}°, offset: {opix*cd*60:.1f}' ({opix:.1f} pix)")
        if vi_pos:
            print(f"  + lobe V/I: median={np.median(vi_pos):.2f}%, "
                  f"range={min(vi_pos):.2f}{max(vi_pos):.2f}%, "
                  f"N={len(vi_pos)}")
        if vi_neg:
            print(f"  - lobe V/I: median={np.median(vi_neg):.2f}%, "
                  f"range={min(vi_neg):.2f}{max(vi_neg):.2f}%, "
                  f"N={len(vi_neg)}")
    else: print("  WARNING: no bilobe axis")
    print(f"  Building bilobe templates ({gn}x{gn})...")
    vfg,nmg,nsg,_ = stack_cutouts(vr1,isrc,em,blh,args.min_template_flux,args.max_template_flux)
    vfg[nmg<max(3,nsg//10)]=0.0
    tiles = {}
    th,tw = v_data.shape[0]/gn, v_data.shape[1]/gn
    for i,s in enumerate(isrc):
        ty,tx = min(int(s['pix_y']/th),gn-1),min(int(s['pix_x']/tw),gn-1)
        tiles.setdefault((ty,tx),[]).append(i)
    ttpl = {}
    for ty in range(gn):
        for tx in range(gn):
            if (ty,tx) not in tiles: continue
            pi = [i for i in tiles[(ty,tx)] if isrc[i].get('source_class')=='point' and args.min_template_flux<=effective_apparent_flux(isrc[i])<args.max_template_flux]
            if len(pi)<args.bilobe_min_sources: continue
            vf,nm,ns,_ = stack_cutouts(vr1,isrc,em,blh,args.min_template_flux,args.max_template_flux,source_indices=pi)
            if ns==0: continue
            vf[nm<max(3,ns//5)]=0.0; ttpl[(ty,tx)]=vf
    print(f"  Built {len(ttpl)} tile templates")
    fbm = {}
    for ty in range(gn):
        for tx in range(gn):
            if (ty,tx) in ttpl: continue
            bd,bk = np.inf,None
            for ry,rx in ttpl:
                d=(ty-ry)**2+(tx-rx)**2
                if d<bd: bd,bk=d,(ry,rx)
            if bk: fbm[(ty,tx)]=bk
    print(f"  Building bilobe prediction...")
    vblp = build_bilobe_prediction(vr1.shape,isrc,vfg,ttpl,fbm,gn,blh)
    vr2 = vr1 - vblp; rms2 = get_uniform_rms(vr2, hm)
    print(f"  RMS: {rms1*1000:.2f} -> {rms2*1000:.2f} mJy")
    if args.save_intermediates:
        v_hdr_out = fits.getheader(args.v_image)
        fits.writeto(os.path.join(out_dir,f"{v_base}_residual_bilobe.fits"),vr2.astype(np.float32),v_hdr_out,overwrite=True)
        fits.writeto(os.path.join(out_dir,f"{v_base}_bilobe_model.fits"),vblp.astype(np.float32),v_hdr_out,overwrite=True)
    print(f"  Phase 3: {time.time()-t3:.0f}s")

    # PHASE 4: STRIATION SUBTRACTION
    print(f"\n{'-'*65}\n  PHASE 4: STRIATION SUBTRACTION\n{'-'*65}")
    t4=time.time()
    print(f"  Building striation model...")
    sm,nb = build_striation_model(vr2,wcs,iwm,ra_window_beams=args.striation_window_beams,beam_pix=bpix)
    vr3 = vr2 - sm; rms3 = get_uniform_rms(vr3, hm)
    print(f"  RMS: {rms2*1000:.2f} -> {rms3*1000:.2f} mJy (total: {(1-rms3/vrms)*100:.1f}% reduction)")
    # Always save the final residual  it supersedes the original V for stacking
    v_hdr_out = fits.getheader(args.v_image)
    resid_final_path = os.path.join(out_dir, f"{v_base}_residual_final.fits")
    fits.writeto(resid_final_path, vr3.astype(np.float32), v_hdr_out, overwrite=True)
    print(f"  Final residual: {resid_final_path}")
    if args.save_intermediates:
        fits.writeto(os.path.join(out_dir,f"{v_base}_striation_model.fits"),sm.astype(np.float32),v_hdr_out,overwrite=True)
    print(f"  Phase 4: {time.time()-t4:.0f}s")

    # PHASE 5: EXTRACTION
    print(f"\n{'-'*65}\n  PHASE 5: CANDIDATE EXTRACTION\n{'-'*65}")
    t5=time.time()

    # Build all exclusion masks and compute sky area breakdown
    ezm_sources = build_exclusion_zone_mask(v_data.shape,isrc,cd)
    ncpm = (all_coords.dec.deg>85.0)&~iwm

    # Bright source proximity mask (I>10Jy, <10')
    prox_radius_pix = 10.0 / (cd*60)
    prox_mask = np.zeros(v_data.shape, dtype=bool)
    for s in isrc:
        if effective_apparent_flux(s) < 10.0: continue
        sy,sx = s['pix_y'],s['pix_x']
        rp = int(np.ceil(prox_radius_pix))
        y1=max(0,sy-rp);y2=min(v_data.shape[0],sy+rp+1)
        x1=max(0,sx-rp);x2=min(v_data.shape[1],sx+rp+1)
        if y2<=y1 or x2<=x1: continue
        yy=np.arange(y1,y2,dtype=float)-sy; xx=np.arange(x1,x2,dtype=float)-sx
        XX,YY = np.meshgrid(xx,yy)
        prox_mask[y1:y2,x1:x2][np.sqrt(XX**2+YY**2)<prox_radius_pix] = True

    # Tiered sidelobe exclusion mask
    tier_mask = np.zeros(v_data.shape, dtype=bool)
    for mf,rd in CONFIG['SIDELOBE_TIERS']:
        rp = rd/cd
        for s in isrc:
            if effective_apparent_flux(s) < mf: continue
            sy,sx = s['pix_y'],s['pix_x']
            rpi = int(np.ceil(rp))
            y1=max(0,sy-rpi);y2=min(v_data.shape[0],sy+rpi+1)
            x1=max(0,sx-rpi);x2=min(v_data.shape[1],sx+rpi+1)
            if y2<=y1 or x2<=x1: continue
            yy=np.arange(y1,y2,dtype=float)-sy; xx=np.arange(x1,x2,dtype=float)-sx
            XX,YY = np.meshgrid(xx,yy)
            tier_mask[y1:y2,x1:x2][np.sqrt(XX**2+YY**2)<rp] = True

    # Combine exclusion zone mask
    ezm = ezm_sources | ncpm

    # Sky area breakdown (cumulative, in order of application)
    total_pix = v_data.shape[0] * v_data.shape[1]
    pix_to_sqdeg = cd * cd  # square degrees per pixel

    n_horizon = int(np.sum(hm | iwm))
    n_ateam = int(np.sum(am & ~hm & ~iwm))
    n_resolved = int(np.sum(ezm_sources & ~hm & ~iwm & ~am))
    n_ncp = int(np.sum(ncpm & ~ezm_sources & ~hm & ~iwm & ~am))
    n_prox = int(np.sum(prox_mask & ~ezm & ~hm & ~iwm & ~am))
    n_tier = int(np.sum(tier_mask & ~prox_mask & ~ezm & ~hm & ~iwm & ~am))

    searchable = total_pix - n_horizon - n_ateam - n_resolved - n_ncp - n_prox - n_tier
    visible = total_pix - n_horizon

    print(f"\n  SKY AREA BUDGET:")
    print(f"  {'-'*50}")
    print(f"  {'Total image:':30s} {total_pix:>8} pix  ({total_pix*pix_to_sqdeg:>8.0f} sq deg)")
    print(f"  {'Below horizon / invalid:':30s}-{n_horizon:>8} pix  ({n_horizon*pix_to_sqdeg:>8.0f} sq deg)")
    print(f"  {'A-team exclusion:':30s}-{n_ateam:>8} pix  ({n_ateam*pix_to_sqdeg:>8.0f} sq deg)")
    print(f"  {'Resolved/bright sources:':30s}-{n_resolved:>8} pix  ({n_resolved*pix_to_sqdeg:>8.0f} sq deg)")
    print(f"  {'NCP (Dec>85°):':30s}-{n_ncp:>8} pix  ({n_ncp*pix_to_sqdeg:>8.0f} sq deg)")
    lbl_prox = "Bright I>10Jy prox (<10'):"
    print(f"  {lbl_prox:30s}-{n_prox:>8} pix  ({n_prox*pix_to_sqdeg:>8.0f} sq deg)")
    print(f"  {'Sidelobe tiers (>40Jy):':30s}-{n_tier:>8} pix  ({n_tier*pix_to_sqdeg:>8.0f} sq deg)")
    print(f"  {'-'*50}")
    print(f"  {'SEARCHABLE:':30s} {searchable:>8} pix  ({searchable*pix_to_sqdeg:>8.0f} sq deg)")
    print(f"  {'':30s} ({searchable/visible*100:.1f}% of visible sky)")

    beam_area_pix = np.pi * (bpix/2)**2 / np.log(2)  # Gaussian beam area
    n_beams = searchable / beam_area_pix
    print(f"  Independent beams: ~{n_beams:.0f}")
    print()

    sig = args.sigma; vt = CONFIG['BDSF_THRESH_PIX_V']
    print(f"\n  Step 1: Extract candidates...")
    cands = extract_v_peaks(vr3,v_hdr,hm,am,iwm,wcs,obs_time,vt,rms3)
    if not cands: print("  No candidates"); tee.close(); return

    print(f"  Step 1b: Local SNR...")
    nlr = assign_local_snr(cands,vr3,em,bpix,sig)
    nrem = sum(1 for c in cands if c['status']=='PASS')
    print(f"    Rejected: {nlr}, Remaining: {nrem}")

    print(f"  Step 2: Exclusion zones...")
    nez = reject_exclusion_zones(cands,ezm); print(f"    Rejected: {nez}")

    if pap is not None:
        print(f"  Step 3: Bilobe cross-match (I>10Jy)...")
        nbl = reject_bilobe_crossmatch(cands,isrc,i_data,irms,bpix,opix,pap,pan,pa_tol=args.pa_tolerance,min_i_flux=10.0)
        print(f"    Rejected: {nbl}")

    print(f"  Step 3b: Bright proximity (I>10Jy, <10')...")
    prsq = (10.0/(cd*60))**2
    bi = [(s['pix_x'],s['pix_y'],effective_apparent_flux(s)) for s in isrc if effective_apparent_flux(s)>=10.0]
    bpx=np.array([b[0] for b in bi],dtype=float); bpy=np.array([b[1] for b in bi],dtype=float); bfl=np.array([b[2] for b in bi])
    nbp=0
    for c in cands:
        if c['status']!='PASS': continue
        dsq=(c['pix_x']-bpx)**2+(c['pix_y']-bpy)**2; m=np.where(dsq<prsq)[0]
        if len(m)>0:
            best=m[np.argmin(dsq[m])]; c['status']='REJECT_BRIGHT_PROXIMITY'
            c['reject_reason']=f"Near I {bfl[best]:.1f}Jy at {np.sqrt(dsq[best])*cd*60:.1f}'"; nbp+=1
    print(f"    Rejected: {nbp}")

    print(f"  Step 3c: Tiered sidelobe exclusion...")
    tiers=sorted(CONFIG['SIDELOBE_TIERS'],key=lambda t:-t[0]); ntr=0
    for c in cands:
        if c['status']!='PASS': continue
        cx,cyc = c['pix_x'],c['pix_y']
        for mf,rd in tiers:
            rpsq=(rd/cd)**2
            for s in isrc:
                if effective_apparent_flux(s)<mf: continue
                if (cx-s['pix_x'])**2+(cyc-s['pix_y'])**2<rpsq:
                    c['status']='REJECT_SIDELOBE_TIER'; c['reject_reason']=f"Tier >{mf:.0f}Jy->{rd:.2f}°"; ntr+=1; break
            if c['status']!='PASS': break
    print(f"    Rejected: {ntr}")

    print(f"  Step 4: Merge..."); nmg = merge_close_multiples(cands,bam,CONFIG['MERGE_RADIUS_BEAMS']); print(f"    Merged: {nmg}")

    for c in cands:
        if c['status']=='PASS':
            idx,d2d,_ = c['coord'].match_to_catalog_sky(ic)
            c['nearest_i_sep']=float(d2d.arcmin); c['nearest_i_snr']=float(isrc[int(idx)]['snr'])

    if args.catalog: print(f"  Step 5: Catalog..."); crossmatch_catalog(cands,args.catalog)
    print(f"  Phase 5: {time.time()-t5:.0f}s")

    # SUMMARY
    np_ = sum(1 for c in cands if c['status']=='PASS')
    cts={}
    for c in cands: cts[c['status']]=cts.get(c['status'],0)+1
    print(f"\n{'='*65}\n  RESULTS\n{'='*65}")
    print(f"  RMS: {vrms*1000:.2f} -> {rms3*1000:.2f} mJy ({(1-rms3/vrms)*100:.1f}%)")
    print(f"  Initial: {len(cands)}")
    for st in ['REJECT_LOCAL_RMS','REJECT_EXCLUSION','REJECT_BILOBE','REJECT_BRIGHT_PROXIMITY','REJECT_SIDELOBE_TIER','REJECT_MERGED']:
        n=cts.get(st,0)
        if n>0: print(f"  {st.replace('REJECT_',''):<22} -{n}")
    print(f"  {'-'*30}\n  PASSED:                {np_}")

    if np_>0:
        passed=sorted([c for c in cands if c['status']=='PASS'],key=lambda c:abs(c['snr']),reverse=True)
        pp=sum(1 for c in passed if c['snr']>0); print(f"  +/- : {pp}/{np_-pp}")
        print(f"\n  {'SNR':>7} {'Flux':>8} | {'RA':>12} {'Dec':>12} | {'El':>5} | {'NearI':>7} | {'Cat':>20}")
        print("  "+"-"*80)
        for c in passed:
            rs=c['coord'].ra.to_string(unit=u.hour,sep=':',precision=1,pad=True)
            ds=c['coord'].dec.to_string(unit=u.deg,sep=':',precision=1,alwayssign=True,pad=True)
            ni=c.get('nearest_i_sep',np.nan); ns=f"{ni:.1f}'" if np.isfinite(ni) else "N/A"
            cat=c.get('catalog_match','')
            print(f"  {c['snr']:>+7.1f} {c['flux']*1000:>7.1f}m | {rs:>12} {ds:>12} | {c['el']:>5.1f} | {ns:>7} | {cat:>20}")

    print(); write_output(cands,out_dir,v_base)
    print(f"\n{'='*65}")
    print(f"  DONE  {time.time()-t0:.0f}s")
    print(f"  Log: {log_path}")
    print(f"{'='*65}")

    tee.close()

if __name__ == "__main__":
    main()
