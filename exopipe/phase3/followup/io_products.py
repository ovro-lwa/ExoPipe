#!/usr/bin/env python3
"""candinv/io_products.py — conformant output writers.

Every FITS goes through phase3_products.build_name + write_standard_header
(time sourced from the Phase-2 originals, since the bundle/_clean headers are
incomplete and carry the co-add date). CSV/.reg use a documented name +
a .provenance.json sidecar. .reg circles use a 1-degree radius (standing rule).
"""
import json
import os
import shutil

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from . import config as C
from exopipe.common.phase3_products import build_name, write_standard_header


# --- naming -----------------------------------------------------------------
def product_name(cand, scope, stokes, proc, epoch_iso, date_range=None,
                 n_epochs=None, cutout=False):
    kw = dict(scope=scope, stokes=stokes, cadence=cand.cadence, taper='Taper',
              robust=0.0, proc=proc, lst=cand.lst, target=cand.cand_id,
              cutout=cutout)
    if cand.cadence == '10min' and cand.iv is not None:
        kw['lst_bin'] = int(cand.iv)
    if date_range:
        kw['date_start'], kw['date_end'] = date_range
        kw['n_epochs'] = n_epochs or 1
    else:
        kw['epoch'] = epoch_iso
    return build_name(**kw)


def _stamp(path, cand, scope, stokes, proc, start, end, inttime, beam, ebeam,
           validate=True, **flags):
    try:
        write_standard_header(
            path, scope=scope, stokes=stokes, start=start, end=end,
            inttime_s=inttime, time_basis='phase2', beam=beam, ebeam=ebeam,
            p2run='PROD', validate=validate,
            extra={'CANDID': (cand.cand_id[:68], 'candinv candidate id'),
                   'CANDRANK': (int(cand.rank), 'top50 rank')},
            **flags)
        return True
    except Exception as e:
        # Don't let one header failure kill a 50-candidate batch; degrade to
        # an unvalidated stamp and flag it loudly.
        if validate:
            try:
                write_standard_header(
                    path, scope=scope, stokes=stokes, start=start, end=end,
                    inttime_s=inttime, time_basis='phase2', beam=beam,
                    ebeam=ebeam, p2run='PROD', validate=False, **flags)
            except Exception:
                pass
        print(f'  [io] WARN header stamp non-conformant for {os.path.basename(path)}: {e}')
        return False


# --- all-sky conformant copy ------------------------------------------------
def conform_allsky(src, outdir, cand, scope, stokes, proc, times, beam=None,
                   ebeam=None, **flags):
    """Copy an all-sky residual and stamp the full standard header in place."""
    start, end, inttime = times
    epoch_iso = start
    name = product_name(cand, scope, stokes, proc, epoch_iso)
    out = os.path.join(outdir, name)
    shutil.copyfile(src, out)
    _stamp(out, cand, scope, stokes, proc, start, end, inttime, beam, ebeam,
           **flags)
    return out


# --- cutout -----------------------------------------------------------------
def write_cutout(src, outdir, cand, scope, stokes, proc, ra, dec, times,
                 half_pix=80, beam=None, ebeam=None, **flags):
    with fits.open(src) as hdul:
        hdr = hdul[0].header.copy()
        data = np.squeeze(hdul[0].data).astype(np.float32)
    wcs = WCS(hdr).celestial
    x, y = wcs.all_world2pix(ra, dec, 0)
    x, y = int(round(float(x))), int(round(float(y)))
    n0, n1 = data.shape
    y0, y1 = max(0, y - half_pix), min(n0, y + half_pix + 1)
    x0, x1 = max(0, x - half_pix), min(n1, x + half_pix + 1)
    sub = data[y0:y1, x0:x1]
    subw = wcs[y0:y1, x0:x1]
    nh = subw.to_header()
    for k in ('BMAJ', 'BMIN', 'BPA', 'BUNIT'):
        if k in hdr:
            nh[k] = hdr[k]
    start, end, inttime = times
    name = product_name(cand, scope, stokes, proc, start, cutout=True)
    out = os.path.join(outdir, name)
    fits.writeto(out, sub, nh, overwrite=True)
    _stamp(out, cand, scope, stokes, proc, start, end, inttime, beam, ebeam,
           **flags)
    return out


# --- sidecars: json / csv / reg --------------------------------------------
def _provenance(out, inputs, params):
    side = out + '.provenance.json' if not out.endswith('.json') else \
        out.replace('.json', '.provenance.json')
    from exopipe.common.phase2_config import PIPELINE_GITTAG
    with open(side, 'w') as f:
        json.dump({'gittag': PIPELINE_GITTAG, 'inputs': inputs,
                   'params': params, 'generator': 'candinv'}, f, indent=2)
    return side


def write_json(outdir, cand, payload, inputs=None, params=None):
    out = os.path.join(outdir, f'{cand.cand_id}_evidence.json')
    with open(out, 'w') as f:
        json.dump(payload, f, indent=2, default=_jsafe)
    _provenance(out, inputs or [], params or {})
    return out


def write_spectrum_csv(outdir, cand, rows, inputs=None):
    import csv
    out = os.path.join(outdir, f'{cand.cand_id}_spectrum.csv')
    cols = ['scope', 'freq_mhz', 'is_subband', 'v_source', 'v_peak', 'v_err',
            'v_snr', 'v_sign', 'i_peak', 'i_err', 'i_snr', 'vi_ratio',
            'v_fwhm_maj_deg', 'v_axis_ratio', 'v_ra', 'v_dec', 'v_detected',
            'i_detected']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    _provenance(out, inputs or [], {'candidate': cand.cand_id})
    return out


def write_snapshot_csv(outdir, cand, snap):
    import csv
    out = os.path.join(outdir, f'{cand.cand_id}_snapshot_lc.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['time', 'mjd', 'peak', 'snr', 'sign'],
                           extrasaction='ignore')
        w.writeheader()
        for r in snap.get('frames', []):
            w.writerow(r)
    _provenance(out, [], {'scope': snap.get('scope'), 'date': snap.get('date'),
                          'peak_snr': snap.get('peak_snr'),
                          'burst_dur_s': snap.get('burst_dur_s')})
    return out


def write_snapshot_spectrum_csv(outdir, cand, rows):
    """Instantaneous (single 10-second frame) spectrum across subbands."""
    import csv
    out = os.path.join(outdir, f'{cand.cand_id}_snapshot_spectrum.csv')
    cols = ['scope', 'freq_mhz', 'dewarped', 'V_peak', 'V_err', 'V_snr',
            'V_sign', 'I_peak', 'I_err', 'I_snr', 'I_sign', 'vi_ratio']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    _provenance(out, [], {'candidate': cand.cand_id})
    return out


def write_reg(outdir, cand, entries):
    """DS9 region file. entries: list of (ra, dec, color, label).
    All circles use a 1-degree radius so the ring never obscures the source."""
    out = os.path.join(outdir, f'{cand.cand_id}.reg')
    with open(out, 'w') as f:
        f.write('global dashlist=8 3 width=1 font="helvetica 10 normal"\n')
        f.write('fk5\n')
        for ra, dec, color, label in entries:
            f.write(f'circle({ra:.6f},{dec:.6f},{C.REG_RADIUS_DEG}d) '
                    f'# color={color} text={{{label}}}\n')
    _provenance(out, [], {'radius_deg': C.REG_RADIUS_DEG})
    return out


def _jsafe(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
