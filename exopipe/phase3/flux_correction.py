#!/usr/bin/env python3
"""Flux-scale correction table (R-FLUX-SCALE-POWERLAW-SMOOTHNESS-THEN-SH-FIT, Gregg 2026-09-11; go 2026-09-11):
  (1) per-band smoothing vector k_b (15 subbands, 1.0 at 73 MHz) from the single power-law VLSSr sub-sample,
      iterated until every band's median residual is within 0.3 per cent (iterate_band_vector.sh);
  (2) one global absolute factor g = inverse-variance mean over cells of (SH12 stage-A factor on the scaled
      generation-4 master) / k_b, i.e. the best fit to the S&H calibrators AFTER the smoothing;
  (3) per cell: FLUXCORR = k_b * g (wideband cells: k of the constituent subbands weighted equally... see below),
      carried as header cards (FLUXBND, FLUXABS, FLUXCORR, FLUXVER) and catalog columns; pixels unchanged.
Wideband bands: k_band = mean of k_b over the band's constituent subbands (Red 18-36, Green 41-59, Blue 64-82,
Full all 15) - the wideband stack is an inverse-variance combination of the subbands, so a flat mean is the
first-order correction; recorded as such in FLUXVER.
Usage: build_table(vector_csv, stageA_csvs, out_csv) -> dict; apply_to_catalog_rows(rows, table) adds the columns.
"""
import csv, glob, os
import numpy as np
SB = ['18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz', '64MHz', '69MHz', '73MHz', '78MHz', '82MHz']
WB = {'Red': SB[:5], 'Green': SB[5:10], 'Blue': SB[10:], 'Full': SB}
VERSION = 'fluxcorr-v1-20260911 (power-law smoothing k_b normalised at 73 MHz; global S&H factor g; wideband k = mean over constituent subbands)'


def read_vector(path):
    k = {r['subband']: float(r['factor']) for r in csv.DictReader(open(path))}
    for b, subs in WB.items():
        k[b] = float(np.mean([k[s] for s in subs]))
    return k


def sh_fit(stageA_csvs, k):
    """Global S&H factor after smoothing: inverse-variance mean over OK stage-A cells of f_cell / k_scope."""
    num = den = 0.0; n = 0; rows = []
    for p in stageA_csvs:
        for r in csv.DictReader(open(p)):
            if r.get('status') != 'OK' or r['scope'] not in k:
                continue
            f = float(r['scale']) / k[r['scope']]; e = float(r['err']) / k[r['scope']]
            if not (e > 0):
                continue
            w = 1.0 / e ** 2; num += w * f; den += w; n += 1; rows.append((r['lst'], r['scope'], f, e))
    if not den:
        raise RuntimeError('S&H fit: no OK stage-A cells')
    g = num / den; ge = den ** -0.5
    resid = np.array([(f - g) / e for _, _, f, e in rows])
    return g, ge, n, float(np.sqrt(np.mean(resid ** 2))), rows


def build_table(vector_csv, stageA_glob, out_csv):
    k = read_vector(vector_csv)
    files = sorted(glob.glob(stageA_glob))
    g, ge, n, chi, rows = sh_fit(files, k)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['scope', 'k_band', 'g_abs', 'g_err', 'fluxcorr', 'n_sh_cells', 'sh_reduced_chi', 'version'])
        for sc in SB + list(WB):
            w.writerow([sc, f'{k[sc]:.6f}', f'{g:.6f}', f'{ge:.6f}', f'{k[sc] * g:.6f}', n, f'{chi:.3f}', VERSION])
    with open(out_csv.replace('.csv', '_shcells.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['lst', 'scope', 'sh_over_k', 'err']); w.writerows(rows)
    return dict(k=k, g=g, g_err=ge, n=n, chi=chi, out=out_csv)


def load_table(path):
    return {r['scope']: dict(k=float(r['k_band']), g=float(r['g_abs']), corr=float(r['fluxcorr']), version=r['version'])
            for r in csv.DictReader(open(path))}


def apply_to_catalog_rows(rows, table, flux_cols=('peak_jy', 'peak_jy_pbcorr', 'flux_jy', 'fp_flux_jy')):
    """Add fluxcorr_factor and <col>_corr for every flux column present; rows of unknown band get factor '' (never 1.0 silently)."""
    for r in rows:
        t = table.get(r.get('band', ''))
        r['fluxcorr_factor'] = f'{t["corr"]:.5f}' if t else ''
        for c in flux_cols:
            if c in r:
                try:
                    r[c + '_corr'] = f'{float(r[c]) * t["corr"]:.4f}' if t and r[c] not in ('', None) else ''
                except ValueError:
                    r[c + '_corr'] = ''
    return rows


CARDS = (('FLUXBND', 'k', 'per-band power-law smoothing factor (Gregg 2026-09-11)'),
         ('FLUXABS', 'g', 'global S&H absolute factor after smoothing'),
         ('FLUXCORR', 'corr', 'multiply fluxes by this (pixels unchanged)'))


def cards_for(scope, table):
    t = table[scope]
    out = [(c, t[key], com) for c, key, com in CARDS]
    out.append(('FLUXVER', t['version'][:68], 'flux correction table version'))
    return out
