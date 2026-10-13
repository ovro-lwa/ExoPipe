"""Deep pages input, one row per served source of a page set.

``singles(U, label, sets)`` is the PORT of the production heredoc p3s3_pass.sh:137-169 (exopipe_ops 0c1580c), kept 1:1:
the same 46 columns in the same order, the same row filter (rows without a date or an LST are dropped silently - counted
here), the same in-place (non-atomic) write. The production heredoc hard-codes U=/data07/gh/phase3/transients/Run_<label>;
the port takes U. The fork sets keep the production script p3s3_tier_pages_input.py IN PLACE (tier_deep.py); ``fork_rows``
here recomputes what that script must write, for the pages_input content test.

DUP-11: the singles heredoc, p3s3_tier_pages_input.py and p3t_pages_input.py are three versions of one function; they stay
separate until a parity-tested merge at absorption.
"""
import csv
import glob
import os

# The 46 columns of a deep pages_input file, in the heredoc's order (p3s3_pass.sh:150-164).
SINGLES_COLUMNS = ('_sid', '_rank', 'lst', 'date', 'band', 'ra', 'dec', 'snr', 'peak_jy', 'pass_ms', 'run', 'common_name',
                   'ra_current', 'dec_current', 'label', 'stokes', 'n_det', 'identity_bands', 'size_beams', 'size_flag',
                   'axis_ratio', 'table_note', 'peak_jy_pbcorr', 'forced_jy', 'forced_pixel_jy', 'forced_method', 'xm_best',
                   'xm_dist_pc', 'xm_sep_arcmin', 'xm_offset_pct', 'xm_E_at_snr', 'xm_contam_pct', 'xm_all', 'promotion',
                   'gal_lat_deg', 'ecl_lat_deg', 'elevation_deg', 'gfit_fwhm_beams', 'n_v_epoch', 'pos_err_arcsec',
                   'source_class', 'identity_id', 'identity_tiers', 'n_cover', 'e_rep', 'snr_rep')
SINGLES_SETS = ('V_wb', 'I_wb', 'V_sb', 'I_sb')


def master_context(U):
    """(fp_prediction by cand_id, conformance columns by cand_id) from <U>/master_catalog.csv (p3s3_pass.sh:141-143)."""
    fpl, mcx = {}, {}
    for r in csv.DictReader(open(f'{U}/master_catalog.csv')):
        fpl[r['cand_id']] = r.get('fp_prediction', '')
        mcx[r['cand_id']] = {k: r.get(k, '') for k in ('size_beams', 'size_flag', 'axis_ratio', 'conform_note')}
    return fpl, mcx


def _table_note(mcx, bc):
    return '; '.join(x.strip() for x in (mcx.get(bc, {}).get('conform_note', '') or '').split('|') if 'table-exempt' in x)


def _row(r, st, label_value, run_label, mcx, identity_tiers):
    bc = r.get('best_cand_id', '')
    return dict(_sid=r.get('source_id', ''), _rank='',
                lst=(bc.split('_')[1] if bc.count('_') >= 3 else (r.get('lst_cells', '') or '').split('+')[0]),
                date=r.get('best_date', ''), band=r.get('band', ''), ra=r.get('ra', ''), dec=r.get('dec', ''),
                snr=r.get('best_snr', ''), peak_jy=r.get('peak_jy_best', '') or '', pass_ms='1', run=run_label,
                common_name=r.get('xm_best', ''), ra_current=r.get('ra', ''), dec_current=r.get('dec', ''),
                label=label_value, stokes=st, n_det=r.get('n_det', ''), identity_bands=r.get('identity_bands', ''),
                size_beams=mcx.get(bc, {}).get('size_beams', ''), size_flag=mcx.get(bc, {}).get('size_flag', ''),
                axis_ratio=mcx.get(bc, {}).get('axis_ratio', ''), table_note=_table_note(mcx, bc),
                peak_jy_pbcorr=r.get('peak_jy_pbcorr_best', ''), forced_jy=r.get('forced_jy', ''),
                forced_pixel_jy=r.get('forced_pixel_jy', ''), forced_method=r.get('forced_method', ''),
                xm_best=r.get('xm_best', ''), xm_dist_pc=r.get('xm_dist_pc', ''), xm_sep_arcmin=r.get('xm_sep_arcmin', ''),
                xm_offset_pct=r.get('xm_offset_pct', ''), xm_E_at_snr=r.get('xm_E_at_snr', ''),
                xm_contam_pct=r.get('xm_contam_pct', ''), xm_all=r.get('xm_all', ''), promotion=r.get('promotion', ''),
                gal_lat_deg=r.get('gal_lat_deg', ''), ecl_lat_deg=r.get('ecl_lat_deg', ''),
                elevation_deg=r.get('elevation_deg', ''), gfit_fwhm_beams=r.get('gfit_fwhm_beams', ''),
                n_v_epoch=r.get('n_v_epoch', ''), pos_err_arcsec=r.get('pos_err_arcsec', ''),
                source_class=r.get('source_class', ''), identity_id=r.get('identity_id', ''),
                identity_tiers=identity_tiers if identity_tiers is not None else r.get('identity_tiers', ''),
                n_cover=r.get('n_cover', ''), e_rep=r.get('e_rep', ''), snr_rep=r.get('snr_rep', ''))


def survivors_files(U, K):
    """The set's survivors catalogues in the order the heredoc reads them (sorted glob, ALL_ files skipped)."""
    return [p for p in sorted(glob.glob(f'{U}/source_catalogs_{K}/*_sources_survivors.csv')) if 'ALL_' not in p]


def singles_rows(U, K, run_label, ctx=None):
    """(rows kept, rows dropped for a missing date or LST) of one singles set, exactly as the heredoc builds them."""
    fpl, mcx = ctx or master_context(U)
    st = K.split('_')[0]
    out = []
    for p in survivors_files(U, K):
        for r in csv.DictReader(open(p)):
            out.append(_row(r, st, fpl.get(r.get('best_cand_id', ''), ''), run_label, mcx, None))
    kept = [r for r in out if r['date'] and r['lst']]
    return kept, len(out) - len(kept)


def fork_rows(U, K, label, run_label, mcx=None):
    """What ops/p3s3_tier_pages_input.py <U> <K> <label> writes (p3s3_tier_pages_input.py:8-26): rows kept, dropped."""
    if mcx is None:
        mcx = {}
        if os.path.exists(f'{U}/master_catalog.csv'):
            mcx = master_context(U)[1]
    st = K.split('_')[0]
    out = []
    for p in survivors_files(U, K):
        for r in csv.DictReader(open(p)):
            out.append(_row(r, st, label, run_label, mcx, K))
    kept = [r for r in out if r['date'] and r['lst']]
    return kept, len(out) - len(kept)


def write_rows(path, rows, atomic=False):
    """The heredoc's write (p3s3_pass.sh:166-167): DictWriter on the first row's keys, or a header of '_sid' alone."""
    tmp = path + '.new' if atomic else path
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['_sid'])
        w.writeheader()
        w.writerows(rows)
    if atomic:
        os.replace(tmp, path)
    return path


def singles(U, run_label, sets=SINGLES_SETS, atomic=False):
    """Write <U>/source_catalogs/pages_input_<K>.csv for the singles sets; returns {K: (path, n kept, n dropped)}."""
    ctx = master_context(U)
    os.makedirs(f'{U}/source_catalogs', exist_ok=True)
    res = {}
    for K in SINGLES_SETS:            # the heredoc's loop order (p3s3_pass.sh:144)
        if K not in sets:
            continue
        kept, dropped = singles_rows(U, K, run_label, ctx)
        res[K] = (write_rows(f'{U}/source_catalogs/pages_input_{K}.csv', kept, atomic=atomic), len(kept), dropped)
    return res
