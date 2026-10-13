#!/usr/bin/env python
"""
mosaic_conformance_pass.py <wb|sb> — build the ARCHIVE STAGING tree of
conformant mosaic products (agreed grammar 2026-07-21; Gregg class ruling:
signal + psfmap + renders + tables; NO rgbcomp).

For every product: copy -> conformant name (build_name, allsky token) ->
write_standard_header (v2 time block aggregated over constituents, post-repair
beam values from the repair ledger / repaired headers, THERMNOI from the
cleaned-V mosaic zenith strip) -> append_provenance_table (one row per
constituent master, canonical coadd_final path, verified data-identical
2026-07-21: prov_hash_{staged,source}.csv 734/734 match).

Constituent membership is RECOMPUTED with the same guards as the build
(missing / census-skip / <100 valid px / CORRUPT-FLAT sigma<1e-4), not parsed
from logs. Read-only on sources; writes only under archive_staging/.
"""
import os, sys, csv, shutil
# 2026-10-13 (refactor chat, verified): this import had been inserted INSIDE the docstring by ecc6557, so coadd_root was undefined
# and mode 'wb' raised NameError at CANON. ('wb' covers 00h-22h because the July 2026 v1 mosaic masters had no 23h; 'wb2' covers 24.)
from exopipe.common.roots import coadd_root
import numpy as np
from astropy.io import fits
from astropy.time import Time

from exopipe.phase3 import mosaic_coadd_jysr as mcj
from exopipe.common.phase3_products import build_name, write_standard_header, append_provenance_table

def main():
    MODE = sys.argv[1]
    if MODE == 'wb':
        SRC = "/fast/gh/main/phase3/mosaic/healpix_jysr_v1"
        MASTERS = "/fast/gh/main/phase3/mosaic/masters"
        STG = "/fast/gh/main/phase3/mosaic/archive_staging/mosaic"
        SUB = "wideband"
        SCOPES = ["Blue", "Green", "Red", "Full"]
        LSTS = [f"{h:02d}h" for h in range(23)]
        CANON = coadd_root() + "/{lst}_{band}/coadd_final/{name}"
        mcj.load_beam_csv("/lustre/gh/main/clv15/master_beam_repair_20260711_wb_apply.csv")
        PROVBAS = ("02-staged snapshot 20260711T1807 + beam ledger "
                   "master_beam_repair_20260711_wb_apply.csv; data blocks verified "
                   "identical to coadd_final 2026-07-21 (734/734 sha256)")
    elif MODE == 'wb2':
        # MOSAIC v2 (Gregg 2026-09-26 Case A). Push approved 2026-09-29 ('when the rebuild and renders are done'),
        # then 'push Taper only' - NoTaper held until the diverged NoTaper masters are rebuilt - and 'jybeam approved'.
        SRC = "/fast/gh/main/phase3/mosaic_v2/healpix"
        MASTERS = "/fast/gh/main/phase3/mosaic_v2/masters"
        STG = "/fast/gh/main/phase3/mosaic_v2/archive_staging/mosaic"
        RENDERS = "/fast/gh/main/phase3/mosaic_v2/renders"
        SUB = "wideband"
        SCOPES = ["Blue", "Green", "Red", "Full"]
        LSTS = [f"{h:02d}h" for h in range(24)]
        CANON = None     # provenance FILE = the Lustre master each staged symlink resolves to
        PROVBAS = ("Lustre Run_20260909_000000 current masters (symlink stage, mosaic_stage_from_lustre.py); "
                   "beams from the masters' own headers (no repair ledger); mosaic v2 2026-09-26..29")
    else:
        SRC = "/fast/gh/main/phase3/mosaic_sb/healpix_jysr_v1"
        MASTERS = "/fast/gh/main/phase3/mosaic_sb/masters"
        STG = "/fast/gh/main/phase3/mosaic_sb/archive_staging/mosaic"
        SUB = "subband"
        SCOPES = ['18MHz','23MHz','27MHz','32MHz','36MHz','41MHz','46MHz','50MHz',
                  '55MHz','59MHz','64MHz','69MHz','73MHz','78MHz','82MHz']
        LSTS = [f"{h:02d}h" for h in range(1, 23)]
        CANON = "/data07/gh/phase3/coadd_sb/{lst}_{band}/coadd_final/{name}"
        mcj.load_beam_csv("/lustre/gh/main/clv15/master_beam_repair_20260711_full_dryrun.csv")
        mcj.load_skip_csv("/data07/gh/phase3/coadd_sb/corrupt_subband_masters_20260711.csv")
        PROVBAS = ("masters symlink tree -> /data07 coadd_sb (post beam repair "
                   "2026-07-13) + dryrun-census beam values")

    # product identity: current-stem prefix -> (stokes, taper, robust, base proc, product)
    VAR = {"NoTaper_R075": ("NoTaper", -0.75), "Taper_R075": ("Taper", -0.75),
           "Taper_R0": ("Taper", 0.0), "NoTaper_R0": ("NoTaper", 0.0)}
    if MODE == 'wb2':
        VAR = {k: v for k, v in VAR.items() if k.startswith('Taper')}   # 'push Taper only' (2026-09-29)
    PRODUCTS = []
    for v, (tp, rb) in VAR.items():
        PRODUCTS.append((f"I_deep_{v}", "I", "I", tp, rb, ["pbcorr", "coadd", "mosaic", "jysr", "feather"]))
    if MODE != 'wb2':                                  # v2 builds Stokes I only
        PRODUCTS.append(("V_deep_Taper_R0_clean", "Vclean", "V", "Taper", 0.0,
                         ["vclean", "pbcorr", "coadd", "mosaic", "jysr", "feather"]))
    if MODE == 'wb':
        PRODUCTS.append(("Mstruct_Taper_R0", "Mstruct", "V", "Taper", 0.0,
                         ["mstruct", "pbcorr", "coadd", "mosaic", "jysr", "feather"]))
    SUFFIX_PROC = {"": [], "_tcorr": ["tcorr"], "_filled": ["lsfill"],
                   "_tcorr_filled": ["tcorr", "lsfill"]}

    for d in (f"{STG}/{SUB}", f"{STG}/renders/{SUB}/mollweide",
              f"{STG}/renders/{SUB}/radec", f"{STG}/tables"):
        os.makedirs(d, exist_ok=True)


    def membership(stem, product, band):
        """Recompute constituent tiles + provenance rows with the build guards."""
        rows, mjd0, mjd1, itt = [], [], [], 0.0
        vname = stem if product == 'I' else mcj._PRODUCT_VARIANT[product]
        variant = stem.replace("I_deep_", "") if product == 'I' else "Taper_R0"
        for lst in LSTS:
            mp, pbp = mcj._paths(MASTERS, lst, variant, band, product)
            key = f"{lst}_{band}|{vname}"
            if key in mcj.SKIP_CELLS:
                continue
            if not (os.path.exists(mp) and os.path.exists(pbp)):
                continue
            d = np.nan_to_num(fits.getdata(mp).squeeze().astype(np.float32))
            pb = np.nan_to_num(fits.getdata(pbp).squeeze().astype(np.float32))
            app = (d * pb)[pb > mcj.PBCORR_FLOOR]
            app = app[np.isfinite(app)]
            if app.size < 100:
                continue
            med = np.median(app)
            sig = 1.4826 * np.median(np.abs(app - med)) or 1e-6
            if not np.isfinite(sig) or sig < 1e-4:
                continue
            h = fits.getheader(mp)
            rows.append({'FILE': CANON.format(lst=lst, band=band, name=os.path.basename(mp)) if CANON
                         else os.path.realpath(mp),
                         'MJD_OBS': h.get('MJD-OBS'), 'MJD_MID': h.get('MJD-MID'),
                         'INTTIME': h.get('INTTIME'), 'WEIGHT': 1.0 / sig ** 2,
                         'THERMNOI': h.get('THERMNOI'), 'P1RUN': str(h.get('GITTAG', ''))})
            if h.get('MJD-OBS') is not None:
                mjd0.append(h['MJD-OBS'])
            for k in ('DATE-END',):
                if h.get(k):
                    mjd1.append(Time(h[k]).mjd)
            itt += float(h.get('INTTIME') or 0)
            bo = mcj.BEAM_OVERRIDE.get(f"{lst}_{band}|{vname}")
            if bo:
                rows[-1]['_beam'] = bo
            elif MODE == 'wb2' and h.get('BMAJ') and h.get('BMIN'):     # v2: master headers are the beam record
                rows[-1]['_beam'] = (float(h['BMAJ']), float(h['BMIN']))
        return rows, (min(mjd0) if mjd0 else None), (max(mjd1) if mjd1 else None), itt


    def thermnoi_for(band):
        """Cleaned-V mosaic zenith-strip (dec 20-55) MAD x median Omega_eff -> Jy/beam."""
        try:
            import healpy as hp
            v = hp.read_map(f"{SRC}/healpix_V_deep_Taper_R0_clean_{band}_jysr_nside2048.fits",
                            verbose=False)
            p = hp.read_map(f"{SRC}/healpix_PSF_V_deep_Taper_R0_clean_{band}_nside2048.fits",
                            verbose=False)
            th, _ = hp.pix2ang(2048, np.arange(len(v)))
            dec = 90.0 - np.degrees(th)
            s = np.isfinite(v) & np.isfinite(p) & (dec > 20) & (dec < 55)
            mad = 1.4826 * np.median(np.abs(v[s] - np.median(v[s])))
            return float(mad * np.median(p[s]))
        except Exception:
            return None


    THERM = {b: thermnoi_for(b) for b in SCOPES}
    print("THERMNOI [Jy/beam]:", {k: (round(v, 4) if v else v) for k, v in THERM.items()},
          flush=True)

    report = {'staged': 0, 'failed': []}

    def stamp(dst, scope, stokes, prov, t0, t1, itt, beam, extra, pbcorr=True):
        write_standard_header(dst, scope=scope, stokes=stokes,
                              start=Time(t0, format='mjd').isot if t0 else None,
                              end=Time(t1, format='mjd').isot if t1 else None,
                              inttime_s=itt or None, time_basis='mosaic-constituents',
                              beam=beam, thermnoi=THERM.get(scope),
                              pbcorr=pbcorr, extra=extra, validate=False)
        append_provenance_table(dst, [{k: v for k, v in r.items() if not k.startswith('_')}
                                      for r in prov])

    for stem0, product, stokes, taper, robust, baseproc in PRODUCTS:
        for band in SCOPES:
            prov, t0, t1, itt = membership(stem0, product, band)
            if not prov:
                print(f"SKIP {stem0} {band}: no constituents", flush=True)
                continue
            beams = np.array([r['_beam'] for r in prov if '_beam' in r])
            beam = ((float(np.median(beams[:, 0])), float(np.median(beams[:, 1])), 0.0)
                    if len(beams) else None)
            n = len(prov)
            d0 = Time(t0, format='mjd').strftime('%Y%m%d') if t0 else '00000000'
            d1 = Time(t1, format='mjd').strftime('%Y%m%d') if t1 else '00000000'
            common = dict(scope=band, stokes=stokes, cadence='deep', taper=taper,
                          robust=robust, lst='allsky', date_start=d0, date_end=d1,
                          n_epochs=n)
            extra_base = {'NSTACK': (n, 'Constituent LST tiles'),
                          'PROVBAS': PROVBAS,
                          'BEAMNOTE': (('scalar beam = median constituent master header; ' if MODE == 'wb2' else
                                        'scalar beam = median constituent (repair ledger); ')
                                       + 'per-pixel Omega_eff in the psfmap product')}
            for suf, extraproc in SUFFIX_PROC.items():
                srcf = f"{SRC}/healpix_{stem0}_{band}{suf}_jysr_nside2048.fits"
                if not os.path.exists(srcf):
                    continue
                name = build_name(proc=baseproc + extraproc, **common)
                dst = f"{STG}/{SUB}/{name}"
                try:
                    shutil.copy2(srcf, dst)
                    src1 = fits.getheader(srcf, 1)
                    ex = dict(extra_base)
                    for k in ('FLUXSCL', 'FEATHER', 'NSIDE', 'JY2K', 'BUNIT'):
                        if k in src1:
                            ex[k] = src1[k]
                    stamp(dst, band, stokes, prov, t0, t1, itt, beam, ex)
                    report['staged'] += 1
                except Exception as e:
                    report['failed'].append((srcf, str(e)[:80]))
                    continue
                # renders of this variant
                base_render = f"{stem0}_{band}{suf}"
                for rdir, tok, rsuf in (("mollweide", "molrender", "_Mollweide"),
                                        ("radec", "carrender", "_RADec")):
                    rsrc = (f"{RENDERS}/{band}_{stem0.replace('I_deep_', '')}{suf}_jysr{rsuf}.fits" if MODE == 'wb2'
                            else f"{SRC}/{rdir}/{base_render}_jysr{rsuf}.fits")
                    if not os.path.exists(rsrc):
                        continue
                    rname = build_name(proc=baseproc + extraproc + [tok], **common)
                    rdst = f"{STG}/renders/{SUB}/{rdir}/{rname}"
                    try:
                        shutil.copy2(rsrc, rdst)
                        stamp(rdst, band, stokes, prov, t0, t1, itt, beam, extra_base)
                        report['staged'] += 1
                    except Exception as e:
                        report['failed'].append((rsrc, str(e)[:80]))
            if MODE == 'wb2':
                # the APPARENT-flux (Jy/beam) domain and its effective-PB companion: the PB-corrected masters x PB, PB
                # used as weight only and never divided out - so neither name carries pbcorr ('jybeam approved' 2026-09-29)
                app = [p for p in baseproc if p not in ('jysr', 'pbcorr')]
                for srcf, proc, ex in ((f"{SRC}/healpix_{stem0}_{band}_jybeam_nside2048.fits", app + ['jybeam'], {}),
                                       (f"{SRC}/healpix_PBeff_{stem0}_{band}_nside2048.fits", app + ['pbmap'], {'BUNIT': ''})):
                    if not os.path.exists(srcf):
                        report['failed'].append((srcf, 'MISSING')); continue
                    dst = f"{STG}/{SUB}/{build_name(proc=proc, **common)}"
                    try:
                        shutil.copy2(srcf, dst)
                        e2 = dict(extra_base, **ex)
                        for k in ('FEATHER', 'NSIDE', 'BUNIT'):
                            if k in fits.getheader(srcf, 1) and k not in ex:
                                e2[k] = fits.getheader(srcf, 1)[k]
                        stamp(dst, band, stokes, prov, t0, t1, itt, beam, e2, pbcorr=False)
                        report['staged'] += 1
                    except Exception as e:
                        report['failed'].append((srcf, str(e)[:80])); continue
                    if proc[-1] == 'jybeam':
                        for rdir, tok, rsuf in (("mollweide", "molrender", "_Mollweide"), ("radec", "carrender", "_RADec")):
                            rsrc = f"{RENDERS}/{band}_{stem0.replace('I_deep_', '')}_jybeam{rsuf}.fits"
                            if not os.path.exists(rsrc):
                                report['failed'].append((rsrc, 'MISSING')); continue
                            rdst = f"{STG}/renders/{SUB}/{rdir}/{build_name(proc=proc + [tok], **common)}"
                            try:
                                shutil.copy2(rsrc, rdst)
                                stamp(rdst, band, stokes, prov, t0, t1, itt, beam, extra_base, pbcorr=False)
                                report['staged'] += 1
                            except Exception as e:
                                report['failed'].append((rsrc, str(e)[:80]))
            # psfmap (base variant only)
            psrc = f"{SRC}/healpix_PSF_{stem0}_{band}_nside2048.fits"
            if os.path.exists(psrc):
                pname = build_name(proc=[p for p in baseproc if p != 'jysr'] + ['psfmap'],
                                   **common)
                pdst = f"{STG}/{SUB}/{pname}"
                try:
                    shutil.copy2(psrc, pdst)
                    stamp(pdst, band, stokes, prov, t0, t1, itt, beam,
                          dict(extra_base, BUNIT='sr'))
                    report['staged'] += 1
                except Exception as e:
                    report['failed'].append((psrc, str(e)[:80]))
            print(f"OK {stem0} {band}: N={n} {d0}-{d1}", flush=True)

    # tables
    import glob
    for pat in ("*.csv", "vsearch/*.csv", "vsearch/*.reg"):
        for f in glob.glob(f"{SRC}/{pat}"):
            shutil.copy2(f, f"{STG}/tables/{os.path.basename(f)}")

    print(f"staged={report['staged']} failed={len(report['failed'])}", flush=True)
    for f, e in report['failed'][:20]:
        print(f"  FAIL {os.path.basename(f)}: {e}", flush=True)
    print("CONFORMANCE_DONE", flush=True)


if __name__ == '__main__':
    main()
