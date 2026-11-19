"""Stages `conform` and `tables` of `exopipe3 mosaic --generation v5` (spec plan/specs/hybrid.md 2.15, 2.16 item 5, gates HG15-HG17;
PR-7c): the archive-staging copies of one (scope, weighting), and the run's tables. A port of the v5 branch of production
mosaic_conformance_pass.py (MC, exopipe 4c7098c) and of the drivers that wrote the V5 tables (run_v5_pkcorr.sh 29-41,
lostsky_hybrids_20261113.py tables 470-486).

conform, per (scope, weighting), on the unit's own records (v5.conform_unit checks them):
  membership  MC 219-262: per LST in order, the staged tile and its weighting map, read whole (memmap=False), with the build's
              guards (at least 100 finite apparent pixels where PB > PBCORR_FLOOR; sigma = 1.4826 MAD >= 1e-4; the coadd added
              every tile, so a guard failing now refuses). PROVENANCE rows: FILE = mosaic_v5.PROV_FILE (the archived hybrid master
              of the tile's name; MC 242 with MOSAIC_V5_PROV_FILE, bb679e2), MJD_OBS, MJD_MID, INTTIME, WEIGHT = 1 / sigma^2,
              THERMNOI, P1RUN = the tile's GITTAG. Time block: MJD-OBS of the first tile, DATE-END of the last, INTTIME summed (a
              tile without MJD-OBS or DATE-END refuses, R-07). Scalar beam: the median of the tiles' BMAJ and BMIN, BPA 0.
  products    MC 355-470 with SUFFIX_PROC v5 (tokens lsrecon, lsfill): the filled Jy/sr map and its two renders, the filled
              Jy/beam map and its two renders, the PB_eff map, the PSF map, the PKCORR map, and in hybrid.CAVEAT_SCOPES the caveat
              map. Names by core.products.build_name (lst 'allsky'; dates and N from the membership).
  stamp       MC 295-305: write_standard_header (time_basis mosaic_conform.TIME_BASIS; the renders with validate=True; the HEALPix
              maps with validate=mosaic_conform.HEADER_VALIDATE, False while decision C-08 is pending: the validator reads pixels
              from HDU 0 only and a HEALPix map's are in HDU 1, its one error on the conformed Blue R075 maps being 'No data in
              primary HDU', so the product test and the push audit check their cards), PRODNAME, NSTACK, PROVBAS, BEAMNOTE, the cards MC copies from HDU 1,
              append_provenance_table, and the HISTORY line 'Mosaic <run>: N constituent masters ...' wrapped at 72. A render
              carries its filled map's fill cards TPFILL ... TPJBCONV (TP_RENDER_KEYS) in HDU 0, as production's renders.
  Every file is written under a temporary name and renamed into place only when the whole unit succeeded.
Changes from production (spec section 4 notes):
  - every product carries the HISTORY line (the archived filled maps and their renders lack it: the fill replaced them after the
    11-11 header repair), and a HEALPix map carries PRODNAME in HDU 1 as well as HDU 0;
  - the Jy/beam map's PSFMAP card names the staged PSF product (bug B14: production's names a work-tree file);
  - PROVBAS and BEAMNOTE state the v2 basis (production's name its own staging tool, and a beam repair ledger the v5 beams never
    used: MC 257 takes them from the master headers);
  - no THERMNOI: MC takes it from the cleaned Stokes V mosaic, and the hybrid route makes none (spec section 6, item 17).

tables, one unit per run: each V5 table is the archived table (hybrid_tilefill.FILLED_MOSAIC_RUN/tables) with the rows of every
(variant, scope) conformed in this run replaced by this run's rows, in production's row order. summary.csv (run_v5_pkcorr.sh 37-40:
the content reports, sorted by <variant>_<scope>_lsfill_<domain>), pkcorr_curves.csv (29-35: the curve files, sorted by
<variant>_<scope>), summary_tpfill_Taper_<w>.csv (lostsky 479-484: the fill tests, sorted by map name; the map column names the
staged Jy/sr product), caveats_<variant>_<scope>_lsfill.csv (the caveat table, its staged column mapped through PROV_FILE as the
PROVENANCE rows are), taper_only.csv (HG14 admits no Taper-only tile: this run adds no row). A table refuses unless it holds a row
for every scope of mosaic_v5.SCOPES and both weightings, and every result is PASS.
"""
import csv
import io
import os

import numpy as np

from exopipe.phase3.lib.mosaic import gen_common as GC

SUMMARY_COLS = ('map', 'variant', 'band', 'domain', 'n_tiles', 'n_sources', 'median_ratio', 'p05', 'p95', 'frac_within_5pct', 'result')
CURVE_COLS = ('variant', 'band', 'dec', 'rec_median', 'rec_mad', 'n', 'n_rounds')
TPFILL_COLS = ('map', 'donor_mhz', 's', 'large_scale_rel_filled', 'large_scale_rel_unfilled', 'highpass_rel', 'highpass_abs_over_map_rms',
               'result')
TAPER_ONLY_COLS = ('lst', 'scope', 'weighting', 'staged', 'striation_restored', 'reason')
RENDERS = (('mollweide', 'molrender'), ('radec', 'carrender'))
# the fill cards a render carries in HDU 0, with their comments, from HDU 1 of its filled map (production's 2026-11-13 carry_renders,
# hybrid_highband_20261031/lostsky_hybrids_20261113.py 317-346)
TP_RENDER_KEYS = ('TPFILL', 'TPDONOR', 'TPDONMHZ', 'TPSCALE', 'TPLLO', 'TPLHI', 'TPJBCONV')


class ConformRefusal(RuntimeError):
    pass


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def archive_run():
    """The mosaic archive run of the hybrid mosaics (its name in each product's HISTORY line)."""
    return _cfg().hybrid_tilefill.FILLED_MOSAIC_RUN


def prov_file(lst, scope, prodname):
    C = _cfg()
    return C.mosaic_v5.PROV_FILE.format(archive_hybrid=C.hybrid.ARCHIVE_HYBRID, lst=lst, scope=scope, name=prodname)


def membership(tiles, scope, floor, file_of=None):
    """MC 219-262 on the staged tiles: (PROVENANCE rows, t0 MJD, t1 MJD, summed INTTIME s, beam (bmaj, bmin, 0) deg). file_of(tile row)
    gives a row's FILE (default: prov_file of the tile's name; v3: the tile master's real path)."""
    from astropy.io import fits
    from astropy.time import Time
    rows, mjd0, mjd1, itt, beams = [], [], [], 0.0, []
    for t in sorted(tiles, key=lambda r: int(r['lst'][:2])):
        with fits.open(t['path'], memmap=False) as h:
            hdr = h[0].header.copy()
            d = np.nan_to_num(h[0].data.squeeze().astype(np.float32))
        with fits.open(t['pb'], memmap=False) as h:
            pb = np.nan_to_num(h[0].data.squeeze().astype(np.float32))
        app = (d * pb)[pb > floor]
        app = app[np.isfinite(app)]
        if app.size < 100:
            raise ConformRefusal(f'{t["lst"]}: {app.size} finite apparent pixels (< 100) in {t["path"]}, which the coadd added')
        med = np.median(app)
        sig = 1.4826 * np.median(np.abs(app - med)) or 1e-6
        if not np.isfinite(sig) or sig < 1e-4:
            raise ConformRefusal(f'{t["lst"]}: sigma {sig:.2e} (flat or corrupt) in {t["path"]}, which the coadd added')
        if hdr.get('MJD-OBS') is None or not hdr.get('DATE-END'):
            raise ConformRefusal(f'{t["lst"]}: {t["path"]} lacks MJD-OBS or DATE-END: the time block would be incomplete (R-07)')
        rows.append({'FILE': file_of(t) if file_of else prov_file(t['lst'], scope, t['prodname']), 'MJD_OBS': hdr.get('MJD-OBS'),
                     'MJD_MID': hdr.get('MJD-MID'),
                     'INTTIME': hdr.get('INTTIME'), 'WEIGHT': 1.0 / sig ** 2, 'THERMNOI': hdr.get('THERMNOI'),
                     'P1RUN': str(hdr.get('GITTAG', ''))})
        mjd0.append(hdr['MJD-OBS'])
        mjd1.append(Time(hdr['DATE-END']).mjd)
        itt += float(hdr.get('INTTIME') or 0)
        if hdr.get('BMAJ') and hdr.get('BMIN'):
            beams.append((float(hdr['BMAJ']), float(hdr['BMIN'])))
    b = np.array(beams)
    beam = (float(np.median(b[:, 0])), float(np.median(b[:, 1])), 0.0) if len(b) else None
    return rows, min(mjd0), max(mjd1), itt, beam


def classic_missing(scope):
    """HG16, MC 163-177 (v5): the weightings of mosaic_v5.VARIANT_OF whose CLASSIC Taper Jy/sr map of the scope is not on Lustre: an
    earlier mosaic run (a Run_* beside hybrid_tilefill.FILLED_MOSAIC_RUN, not a *_QUARANTINE_* run: a change from MC, which also
    took a superseded map) must hold it by name with a PASS jysr row in its
    tables/summary.csv (the hybrids stand for NoTaper; Gregg 2026-10-23: 'all products are required on Lustre'). The pattern takes
    no hybrid map: their names carry lsfill after feather."""
    import glob
    C = _cfg()
    root = os.path.dirname(archive_run())
    miss = []
    for w, v in C.mosaic_v5.VARIANT_OF.items():
        tok = f"{v.split('_')[0]}_{'Robust-0.75' if v.endswith('R075') else 'Robust+0.0'}"
        found = False
        for run in sorted(glob.glob(os.path.join(glob.escape(root), 'Run_*'))):
            if '_QUARANTINE_' in os.path.basename(run):                  # superseded files: never the archive's classic map
                continue
            g = glob.glob(os.path.join(glob.escape(run), scope, 'I', 'healpix',
                                       f'{scope}_I_deep_{tok}_pbcorr_coadd_*mosaic_jysr_feather_allsky_*.fits'))
            summ = os.path.join(run, 'tables', 'summary.csv')
            if g and os.path.exists(summ):
                with open(summ, newline='') as f:
                    if any(r.get('variant') == v and r.get('band') == scope and r.get('domain') == 'jysr'
                           and (r.get('result') or '').strip() == 'PASS' for r in csv.DictReader(f)):
                        found = True
                        break
        if not found:
            miss.append(w)
    return miss


def product_plan(P, caveat):
    """[(kind, source path, subdirectory, proc tokens, PBCORR)] in MC's order (355-470), then the unfilled Jy/sr and Jy/beam twins under
    the filled maps' names (subdirectory 'unfilled': production's mosaic/Run_HYBRID_QUARANTINE_20261113/unfilled/, the pair the tilefill
    band field is made from, hybrid_tilefill.UNFILLED_MOSAIC_RUNS)."""
    C = _cfg()
    K = C.mosaic_conform
    base = list(K.BASE_PROC)
    suf = list(C.mosaic_v5.SUFFIX_PROC)
    app = [p for p in base if p not in K.APPARENT_DROP_PROC] + suf
    noj = [p for p in base if p != 'jysr'] + suf
    items = [('jysr', P['filled_jysr'], 'healpix', base + suf, True)]
    items += [(f'jysr_{k}', P[f'render_jysr_{k}'], k, base + suf + [tok], True) for k, tok in RENDERS]
    items += [('jybeam', P['filled_jybeam'], 'healpix', app + ['jybeam'], False),
              ('pbmap', P['unfilled_pbeff'], 'healpix', app + ['pbmap'], False)]
    items += [(f'jybeam_{k}', P[f'render_jybeam_{k}'], k, app + ['jybeam', tok], False) for k, tok in RENDERS]
    items += [('psfmap', P['unfilled_psf'], 'healpix', noj + ['psfmap'], True),
              ('pkcorr', P['pkcorr_map'], 'healpix', noj + ['pkcorr'], True)]
    if caveat:
        items.append(('caveat', P['caveat_map'], 'healpix', noj + ['caveat'], True))
    items += [('unfilled_jysr', P['unfilled_jysr'], 'unfilled', base + suf, True),
              ('unfilled_jybeam', P['unfilled_jybeam'], 'unfilled', app + ['jybeam'], False)]
    return items


def names(scope, w, P, caveat, t0, t1, n):
    """{kind: (source, subdirectory, product name, PBCORR)}."""
    from astropy.time import Time
    from exopipe.phase3.core.products import build_name
    C = _cfg()
    K = C.mosaic_conform
    v = C.mosaic_v5.VARIANT_OF[w]
    common = dict(scope=scope, stokes='I', cadence=K.CADENCE, taper=v.split('_')[0], robust=-0.75 if v.endswith('R075') else 0.0,
                  lst=K.LST_TOKEN, date_start=Time(t0, format='mjd').strftime('%Y%m%d'),
                  date_end=Time(t1, format='mjd').strftime('%Y%m%d'), n_epochs=n)
    return {k: (src, sub, build_name(proc=proc, **common), pbc) for k, src, sub, proc, pbc in product_plan(P, caveat)}


def _subdir(dest, scope, sub):
    """<dest>/<scope>/I/<sub>/ (the archive run's layout); the unfilled twins: <dest>/unfilled/<scope>/I/healpix/."""
    return os.path.join(dest, 'unfilled', scope, 'I', 'healpix') if sub == 'unfilled' else os.path.join(dest, scope, 'I', sub)


def _hdr1(src, keys, ex):
    from astropy.io import fits
    out = dict(ex)
    h1 = fits.getheader(src, 1, memmap=False)
    for k in keys:
        if k in h1 and k not in ex:
            out[k] = h1[k]
    return out


def stage(scope, w, P, tiles, caveat, dest, floor, plan=None):
    """Stage the unit's products under dest/<scope>/I/<subdirectory>/ (the archive run's layout; the unfilled twins under
    dest/unfilled/<scope>/I/healpix/). Returns ([(final path, source,
    kind)], info). plan(final paths) is called once the names are known, before anything is written. Raises ConformRefusal;
    nothing is renamed into place unless every product was written (gen_common.stamp_products)."""
    from astropy.io import fits
    C = _cfg()
    K = C.mosaic_conform
    V5 = C.mosaic_v5
    rows, t0, t1, itt, beam = membership(tiles, scope, floor)
    n = len(rows)
    nm = names(scope, w, P, caveat, t0, t1, n)
    if plan:
        plan([os.path.join(_subdir(dest, scope, sub), name) for _src, sub, name, _p in nm.values()])
    provbas = V5.PROVBAS.format(archive_hybrid=os.path.relpath(C.hybrid.ARCHIVE_HYBRID, C.paths.PHASE3_ARCHIVE_ROOT))
    if caveat:
        provbas += V5.PROVBAS_CAVEAT
    items = []
    for kind, (src, sub, name, pbc) in nm.items():
        base = {'PRODNAME': (name, 'Phase 3 product name (grammar; CONTINUE cards)'), 'NSTACK': (n, 'Constituent LST tiles'),
                'PROVBAS': provbas, 'BEAMNOTE': V5.BEAMNOTE}
        if kind in ('jysr', 'unfilled_jysr'):
            ex = _hdr1(src, K.COPY_HEADER_KEYS, base)
        elif kind in ('jybeam', 'unfilled_jybeam', 'pbmap'):
            ex = _hdr1(src, K.COPY_HEADER_KEYS_APPARENT, dict(base, **({'BUNIT': ''} if kind == 'pbmap' else {})))
        elif kind == 'psfmap':
            ex = dict(base, BUNIT='sr')
        elif kind == 'pkcorr':
            ex = dict(base, BUNIT='')
        elif kind == 'caveat':
            ex = dict(base, BUNIT='', NSIDE=fits.getheader(src, 1, memmap=False)['NSIDE'])
        else:                                                      # a render: its filled map's fill cards
            h1 = fits.getheader(nm[kind.split('_')[0]][0], 1, memmap=False)
            ex = dict(base, **{k: (h1[k], h1.comments[k]) for k in TP_RENDER_KEYS if k in h1})
        items.append((kind, src, sub, os.path.join(_subdir(dest, scope, sub), name), pbc, ex))
    out = GC.stamp_products(items, rows, GC.mosaic_history(os.path.basename(archive_run()), n), scope=scope, stokes='I', t0=t0, t1=t1,
                            itt=itt, beam=beam, thermnoi=None, psf_name=nm['psfmap'][2])
    info = {'n': n, 't0': t0, 't1': t1, 'inttime_s': itt, 'beam': beam, 'names': {k: v[2] for k, v in nm.items()},
            'prov_files': [r['FILE'] for r in rows]}
    return out, info


# ---- tables -----------------------------------------------------------------------------------------------------------------------

def _read(path):
    with open(path, newline='') as f:
        r = csv.reader(f)
        head = next(r)
        return head, [row for row in r]


def _write(path, head, rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(head)
    for row in rows:
        w.writerow(row)
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w', newline='') as f:
        f.write(buf.getvalue())
    os.replace(tmp, path)


def merge(archived_rows, new_rows, key, sort_key):
    """The archived rows less those whose key is in new_rows' keys, plus new_rows, sorted by sort_key."""
    keys = {key(r) for r in new_rows}
    return sorted([r for r in archived_rows if key(r) not in keys] + list(new_rows), key=sort_key)


def summary_rows(content_json):
    """run_v5_pkcorr.sh 37-40: one row per domain of a content report."""
    import json
    with open(content_json) as f:
        rep = json.load(f)
    return [[str(x) if not isinstance(x, str) else x for x in
             (os.path.basename(r['map']), r['variant'], r['band'], r['domain'], r['n_tiles'], r['n_sources'], r['median_ratio'], r['p05'],
              r['p95'], r['frac_within_5pct'], r['result'])] for r in (rep[d] for d in ('jybeam', 'jysr'))]


def summary_sort(r):
    return f'{r[1]}_{r[2]}_lsfill_{r[3]}.json'


def curve_rows(variant, band, curve_csv):
    """run_v5_pkcorr.sh 29-35."""
    with open(curve_csv, newline='') as f:
        return [[variant, band, r['dec'], r['rec_median'], r['rec_mad'], r['n'], r['n_rounds']] for r in csv.DictReader(f)]


def curve_sort(r):
    return f'peak_corr_curve_{r[0]}_{r[1]}_lsfill.csv'


def tpfill_row(jysr_name, fill_meta):
    """lostsky_hybrids_20261113.py 479-484 on the fill record (meta fill.jysr)."""
    t = fill_meta['test']
    return [jysr_name] + [str(x) for x in (fill_meta['donor_mhz'], round(fill_meta['s'], 5), t['large_scale_rel_filled'],
                                           t['large_scale_rel_unfilled'], t['highpass_rel'], t['highpass_abs_over_map_rms'],
                                           t['result'])]


def caveat_rows(caveat_csv, scope):
    """The caveat table with its staged column (the staged tile) mapped through PROV_FILE, as the PROVENANCE rows are."""
    head, rows = _read(caveat_csv)
    i, j = head.index('staged'), head.index('lst')
    out = []
    for r in rows:
        r = list(r)
        if r[i]:
            r[i] = prov_file(r[j], scope, os.path.basename(r[i]))
        out.append(r)
    return head, out
