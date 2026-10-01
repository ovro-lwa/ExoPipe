#!/usr/bin/env python3
"""Flux-scale correction table (R-FLUX-SCALE-POWERLAW-SMOOTHNESS-THEN-SH-FIT, Gregg 2026-09-11; go 2026-09-11):
  (1) per-band smoothing vector k_b (15 subbands, 1.0 at 73 MHz) from the single power-law VLSSr sub-sample,
      iterated until every band's median residual is within 0.3 per cent (iterate_band_vector.sh);
  (2) one global absolute factor g = inverse-variance mean over cells of (SH12 stage-A factor on the scaled
      generation-4 master) / k_b, i.e. the best fit to the S&H calibrators AFTER the smoothing;
  (3) per cell: FLUXCORR = k_b * g (wideband cells: k of the constituent subbands weighted equally... see below),
      carried as header cards (FLUXBND, FLUXABS, FLUXCORR, FLUXVER, FLUXBAKE); from the 2026-10-01 re-bake it is IN THE
      PIXELS (Gregg 2026-10-01) and catalogues no longer apply it to baked products.
Wideband bands: k_band = mean of k_b over the band's constituent subbands (Red 18-36, Green 41-59, Blue 64-82,
Full all 15) - the wideband stack is an inverse-variance combination of the subbands, so a flat mean is the
first-order correction; recorded as such in FLUXVER.
Usage: build_table(vector_csv, stageA_csvs, out_csv) -> dict; apply_to_catalog_rows(rows, table) adds the columns.
"""
import csv, glob, os
import numpy as np
SB = ['18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz', '64MHz', '69MHz', '73MHz', '78MHz', '82MHz']
WB = {'Red': SB[:5], 'Green': SB[5:10], 'Blue': SB[10:], 'Full': SB}
VERSION = ('fluxcorr-v2-20261001 (LOO power-law smoothing k_b at 73 MHz=1, PB per robust; '
                         'global S&H factor g = robust median over cells on the 2026-10-01 per-LST VLSSr scale; wideband k = mean of '
                         'constituent subbands; BAKED INTO THE PIXELS)')


def read_vector(path):
    k = {r['subband']: float(r['factor']) for r in csv.DictReader(open(path))}
    for b, subs in WB.items():
        k[b] = float(np.mean([k[s] for s in subs]))
    return k


def sh_fit(stageA_csvs, k, lst_rescale=None):
    """Global S&H factor after smoothing: inverse-variance mean over OK stage-A cells of f_cell / k_scope.
    lst_rescale {lst: r}: the stage-A factors were measured on pixels that will be multiplied by r per LST (the 2026-10-01
    re-measured VLSSr factor relative to the one in the pixels), so f_cell / r is the factor on the re-baked pixels; an LST
    missing from it is REFUSED (never treated as 1)."""
    num = den = 0.0; n = 0; rows = []
    for p in stageA_csvs:
        for r in csv.DictReader(open(p)):
            if r.get('status') != 'OK' or r['scope'] not in k:
                continue
            if lst_rescale is not None and r['lst'] not in lst_rescale:
                raise RuntimeError(f"S&H fit: {r['lst']} {r['scope']} has no per-LST rescale factor; REFUSING")
            rs = lst_rescale[r['lst']] if lst_rescale is not None else 1.0
            f = float(r['scale']) / k[r['scope']] / rs; e = float(r['err']) / k[r['scope']] / rs
            if not (e > 0):
                continue
            w = 1.0 / e ** 2; num += w * f; den += w; n += 1; rows.append((r['lst'], r['scope'], f, e))
    if not den:
        raise RuntimeError('S&H fit: no OK stage-A cells')
    g_ivw = num / den; ge_ivw = den ** -0.5
    resid = np.array([(f - g_ivw) / e for _, _, f, e in rows])
    chi = float(np.sqrt(np.mean(resid ** 2)))
    # 2026-09-11: the per-cell calibrator factors scatter far beyond their formal errors (reduced chi-square 18
    # on generation 4), so inverse-variance weights are dominated by a handful of cells; the applied global
    # factor is the ROBUST MEDIAN over cells with its robust error, the inverse-variance mean kept for reference.
    v = np.array([f for _, _, f, _ in rows]); g = float(np.median(v))
    ge = float(1.2533 * 1.4826 * np.median(np.abs(v - g)) / np.sqrt(len(v)))
    return g, ge, n, chi, rows, g_ivw, ge_ivw


def build_table(vector_csv, stageA_glob, out_csv, lst_rescale=None):
    k = read_vector(vector_csv)
    files = sorted(glob.glob(stageA_glob))
    g, ge, n, chi, rows, g_ivw, ge_ivw = sh_fit(files, k, lst_rescale)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['scope', 'k_band', 'g_abs', 'g_err', 'fluxcorr', 'n_sh_cells', 'sh_reduced_chi', 'g_ivw_reference', 'version'])
        for sc in SB + list(WB):
            w.writerow([sc, f'{k[sc]:.6f}', f'{g:.6f}', f'{ge:.6f}', f'{k[sc] * g:.6f}', n, f'{chi:.3f}', f'{g_ivw:.6f}', VERSION])
    with open(out_csv.replace('.csv', '_shcells.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['lst', 'scope', 'sh_over_k', 'err']); w.writerows(rows)
    return dict(k=k, g=g, g_err=ge, n=n, chi=chi, g_ivw=g_ivw, out=out_csv)


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
         ('FLUXCORR', 'corr', 'k x g'))
# BAKED INTO THE PIXELS (Gregg 2026-10-01: "Yes, bake it into the pixels in the same re-bake. It was always my intention to
# have the flux scale correction applied to pixels."): a product whose pixels already carry k x g has FLUXBAKE = True, and
# no consumer may apply the table to it again (build_source_catalogs_v3 refuses).
BAKE_CARD = 'FLUXBAKE'


def pixels_carry_correction(hdr):
    return bool(hdr.get(BAKE_CARD, False))


def cards_for(scope, table, baked=False):
    t = table[scope]
    out = [(c, t[key], com) for c, key, com in CARDS]
    out.append(('FLUXVER', t['version'][:68], 'flux correction table version'))
    out.append((BAKE_CARD, bool(baked), 'k x g already in the pixels' if baked else 'pixels unchanged: multiply fluxes by FLUXCORR'))
    return out


# Rebuild paths that start from UN-SCALED inputs (variant rebuild from the generation record, un-warped reconstructions)
# multiply by the cell's TOTAL pixel factor and carry these cards: SHSCALE alone before the 2026-10-01 re-bake, SHSCALE x
# FLUXCORR once FLUXBAKE is set.
CARRY_CARDS = ('SHSCALE', 'SHERR', 'SHNSRC', 'SHSRCS', 'SHMETHOD', 'SHFREQ', 'SHDATE',
               'FLUXBND', 'FLUXABS', 'FLUXCORR', 'FLUXVER', BAKE_CARD)


def pixel_factor(hdr):
    """Total factor in a product's pixels relative to the un-scaled build; None when it carries no flux scale."""
    if hdr.get('SHSCALE') is None:
        return None
    f = float(hdr['SHSCALE'])
    if pixels_carry_correction(hdr):
        if hdr.get('FLUXCORR') is None:
            raise RuntimeError('FLUXBAKE set without FLUXCORR - header inconsistent; REFUSING')
        f *= float(hdr['FLUXCORR'])
    return f
