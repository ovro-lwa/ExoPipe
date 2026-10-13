"""`exopipe3 mosaic` stage conform: the archive-staging copies of one job (unit 'conform/<job>') and the QA tables (unit
'conform/_tables'), under <out>/archive_staging/mosaic/ (the tree `exopipe3 archive push --class mosaic` transfers).

PORT of the 'wb2' branch of the production mosaic_conformance_pass.py (CP:43-55, 70-89, 92-131, 155-276; exopipe 927a836)
as a function with arguments, with the same calls in the same order: core.products.build_name (allsky token), shutil.copy2,
write_standard_header(time_basis 'mosaic-constituents', beam = median constituent header, THERMNOI from the cleaned-V mosaic
strip = absent in v2, validate=False) and append_provenance_table. Differences, all from the spec's gates (no product byte
changes on a clean run):
- membership is the coadd record's contributing tiles (census with the engine's guards, checked against the engine's own
  verdicts) instead of a recomputation over the staged links - same tiles, same arithmetic (CP:104-113 = CJ:200-221);
- every failure refuses the unit (G13; production printed the first 20 failures and exited 0), and nothing half-staged
  survives: every file is written under a temporary name and renamed only when the whole job succeeded;
- a missing time block refuses (G14) before anything is written (production: write_standard_header raised, the copy stayed);
- the VLSSr gate G16 refuses only when mosaic_qa.VLSSR_GATE_MAX_MEDIAN_OFFSET_ARCMIN is set (None today).
"""
import os
import shutil
import sys

from exopipe.phase3.lib.mosaic import steps_common as SC


def _say(msg):
    print(msg, file=sys.stderr, flush=True)


def membership(tiles):
    """CP:92-131 on the coadd record: provenance rows of the contributing tiles in LST order, t0 (min MJD-OBS), t1 (max
    DATE-END as MJD), summed INTTIME, and the beam (median BMAJ, median BMIN, BPA 0) of the tiles that carry both."""
    import numpy as np
    from astropy.time import Time
    rows, mjd0, mjd1, itt = [], [], [], 0.0
    for t in sorted((t for t in tiles if t.get('verdict') == 'added'), key=lambda t: int(t['lst'][:2])):
        rows.append({'FILE': t['realpath'], 'MJD_OBS': t.get('MJD-OBS'), 'MJD_MID': t.get('MJD-MID'),
                     'INTTIME': t.get('INTTIME'), 'WEIGHT': t.get('weight'), 'THERMNOI': t.get('THERMNOI'),
                     'P1RUN': str(t.get('GITTAG') if t.get('GITTAG') is not None else '')})
        if t.get('MJD-OBS') is not None:
            mjd0.append(t['MJD-OBS'])
        if t.get('DATE-END'):
            mjd1.append(Time(t['DATE-END']).mjd)
        itt += float(t.get('INTTIME') or 0)
        if t.get('BMAJ') and t.get('BMIN'):
            rows[-1]['_beam'] = (float(t['BMAJ']), float(t['BMIN']))
    beams = np.array([r['_beam'] for r in rows if '_beam' in r])
    beam = (float(np.median(beams[:, 0])), float(np.median(beams[:, 1])), 0.0) if len(beams) else None
    return rows, (min(mjd0) if mjd0 else None), (max(mjd1) if mjd1 else None), itt, beam


def thermnoi_for(src_dir, band, nside):
    """CP:134-148 verbatim in effect: cleaned-V mosaic zenith-strip MAD x median Omega_eff -> Jy/beam; None when the
    cleaned-V mosaic is absent (always in v2)."""
    C = SC.cfg()
    try:
        import healpy as hp
        import numpy as np
        v = hp.read_map(f"{src_dir}/healpix_V_deep_Taper_R0_clean_{band}_jysr_nside{nside}.fits", verbose=False)
        p = hp.read_map(f"{src_dir}/healpix_PSF_V_deep_Taper_R0_clean_{band}_nside{nside}.fits", verbose=False)
        th, _ = hp.pix2ang(nside, np.arange(len(v)))
        dec = 90.0 - np.degrees(th)
        lo, hi = C.mosaic_conform.THERMNOI_DEC_STRIP_DEG
        s = np.isfinite(v) & np.isfinite(p) & (dec > lo) & (dec < hi)
        mad = 1.4826 * np.median(np.abs(v[s] - np.median(v[s])))
        return float(mad * np.median(p[s]))
    except Exception:
        return None


def stage_job(job, coadd_rec, render_rec, out_root):
    """Stage one job. Returns [(final path, source build product, kind)] after renaming every temporary file into place;
    raises RuntimeError (G13/G14) with every failure, leaving nothing staged."""
    from astropy.io import fits
    from astropy.time import Time
    from exopipe.phase3.core.products import build_name, write_standard_header, append_provenance_table
    C = SC.cfg()
    K = C.mosaic_conform
    nside = C.mosaic_coadd.NSIDE
    stg = SC.layout(out_root, 'staging')
    src_dir = SC.layout(out_root, 'healpix')
    sub = K.SUBDIR
    files = {os.path.basename(o['path']): o['path'] for o in coadd_rec.get('outputs', [])}
    rfiles = {os.path.basename(o['path']): o['path'] for o in (render_rec or {}).get('outputs', [])}
    tiles = (coadd_rec.get('meta') or {}).get('tiles') or []
    prov, t0, t1, itt, beam = membership(tiles)
    if not prov:
        raise RuntimeError(f'{job.name}: no contributing tile in the coadd record')
    if t0 is None or t1 is None:
        raise RuntimeError(f'G14: {job.name}: no contributing master carries MJD-OBS and DATE-END (time block '
                           f'incomplete, R-07)')
    therm = thermnoi_for(src_dir, job.scope, nside)
    n = len(prov)
    d0 = Time(t0, format='mjd').strftime('%Y%m%d')
    d1 = Time(t1, format='mjd').strftime('%Y%m%d')
    common = dict(scope=job.scope, stokes='I', cadence=K.CADENCE, taper=job.taper, robust=job.robust_value,
                  lst=K.LST_TOKEN, date_start=d0, date_end=d1, n_epochs=n)
    extra_base = {'NSTACK': (n, 'Constituent LST tiles'), 'PROVBAS': K.PROVBAS, 'BEAMNOTE': K.BEAMNOTE}
    base = list(K.BASE_PROC)
    app = [p for p in base if p not in K.APPARENT_DROP_PROC]
    for d in (f'{stg}/{sub}', f'{stg}/renders/{sub}/mollweide', f'{stg}/renders/{sub}/radec', f'{stg}/tables'):
        os.makedirs(d, exist_ok=True)
    pid = os.getpid()
    done, failed = [], []

    def stamp(dst, prov_rows, extra, pbcorr=True):
        write_standard_header(dst, scope=job.scope, stokes='I',
                              start=Time(t0, format='mjd').isot if t0 else None,
                              end=Time(t1, format='mjd').isot if t1 else None,
                              inttime_s=itt or None, time_basis=K.TIME_BASIS,
                              beam=beam, thermnoi=therm, pbcorr=pbcorr, extra=extra, validate=K.HEADER_VALIDATE)
        append_provenance_table(dst, [{k: v for k, v in r.items() if not k.startswith('_')} for r in prov_rows])

    def one(src, dst, extra, pbcorr, kind):
        if not src or not os.path.exists(src):
            failed.append((src or dst, 'MISSING'))
            return
        tmp = os.path.join(os.path.dirname(dst), f'.{os.path.basename(dst)}.tmp.{pid}')
        try:
            shutil.copy2(src, tmp)
            stamp(tmp, prov, extra, pbcorr)
            done.append((tmp, dst, src, kind))
        except Exception as e:
            failed.append((src, f'{type(e).__name__}: {str(e)[:160]}'))
            if os.path.exists(tmp):
                os.remove(tmp)

    def hdr1(src, keys, ex):
        out = dict(ex)
        try:
            h1 = fits.getheader(src, 1)
        except Exception as e:
            failed.append((src, f'HDU 1 unreadable: {e}'))
            return out
        for k in keys:
            if k in h1 and k not in ex:
                out[k] = h1[k]
        return out

    stem = job.stem
    (jysr_n, _), (psf_n, _) = SC.coadd_products(job, 'jysr')
    (jyb_n, _), (pbe_n, _) = SC.coadd_products(job, 'jybeam')
    # Jy/sr map (CP:188-204): extra = base + FLUXSCL, FEATHER, NSIDE, JY2K, BUNIT of the map's HDU 1
    srcf = files.get(jysr_n)
    if srcf:
        one(srcf, f'{stg}/{sub}/{build_name(proc=base, **common)}', hdr1(srcf, K.COPY_HEADER_KEYS, extra_base), True,
            'jysr')
        for rdir, tok, rsuf in (('mollweide', 'molrender', '_Mollweide'), ('radec', 'carrender', '_RADec')):   # CP:206-220
            rsrc = rfiles.get(f'{job.scope}_{job.variant}_jysr{rsuf}.fits')
            if rsrc:
                one(rsrc, f'{stg}/renders/{sub}/{rdir}/{build_name(proc=base + [tok], **common)}', dict(extra_base), True,
                    f'jysr_{rdir}')
    else:
        failed.append((jysr_n, 'MISSING from the coadd record'))
    # apparent-flux domain and its PB_eff companion (CP:221-251): no pbcorr token, PBCORR False
    for src_n, proc, ex in ((jyb_n, app + ['jybeam'], {}), (pbe_n, app + ['pbmap'], {'BUNIT': ''})):
        srcf = files.get(src_n)
        if not srcf:
            failed.append((src_n, 'MISSING'))
            continue
        one(srcf, f'{stg}/{sub}/{build_name(proc=proc, **common)}',
            hdr1(srcf, K.COPY_HEADER_KEYS_APPARENT, dict(extra_base, **ex)), False, proc[-1])
        if proc[-1] == 'jybeam':
            for rdir, tok, rsuf in (('mollweide', 'molrender', '_Mollweide'), ('radec', 'carrender', '_RADec')):
                rsrc = rfiles.get(f'{job.scope}_{job.variant}_jybeam{rsuf}.fits')
                if not rsrc:
                    failed.append((f'{job.scope}_{job.variant}_jybeam{rsuf}.fits', 'MISSING'))
                    continue
                one(rsrc, f'{stg}/renders/{sub}/{rdir}/{build_name(proc=proc + [tok], **common)}', dict(extra_base),
                    False, f'jybeam_{rdir}')
    # psfmap (CP:252-264)
    psrc = files.get(psf_n)
    if psrc:
        one(psrc, f'{stg}/{sub}/{build_name(proc=[p for p in base if p != "jysr"] + ["psfmap"], **common)}',
            dict(extra_base, BUNIT='sr'), True, 'psfmap')
    if failed:
        for tmp, _dst, _src, _k in done:
            if os.path.exists(tmp):
                os.remove(tmp)
        raise RuntimeError('G13: staging failed for ' + '; '.join(f'{os.path.basename(str(f))}: {e}'
                                                                  for f, e in failed[:20]))
    out = []
    for tmp, dst, src, kind in done:
        os.replace(tmp, dst)
        out.append((dst, src, kind))
    _say(f'OK {stem} {job.scope}: N={n} {d0}-{d1}')
    return out, {'n': n, 'd0': d0, 'd1': d1, 'beam': beam, 'thermnoi': therm, 'inttime_s': itt, 't0': t0, 't1': t1}


def _vlssr_gate(run, job):
    """G16: (passed, detail) - the job's median VLSSr offset from the latest done qa record vs the gate (None = measure
    only)."""
    import csv
    from exopipe.phase3.core.record import load_record
    C = SC.cfg()
    gate = C.mosaic_qa.VLSSR_GATE_MAX_MEDIAN_OFFSET_ARCMIN
    if gate is None:
        return True, 'measure only (mosaic_qa.VLSSR_GATE_MAX_MEDIAN_OFFSET_ARCMIN None)'
    rec = load_record(run.out_root, SC.STEP, 'qa/_all')
    if not rec or rec.get('status') != 'done':
        return False, 'G16: no done qa record (the VLSSr gate is set)'
    for o in rec.get('outputs', []):
        if 'vlssr' in os.path.basename(o['path']):
            for r in csv.DictReader(open(o['path'])):
                if r['variant'] == job.variant and r['band'] == job.scope:
                    try:
                        med = float(r['offset_med_arcmin'])
                    except (TypeError, ValueError):
                        return False, f'G16: no VLSSr offset for {job.name}'
                    return med <= gate, f'G16: median VLSSr offset {med:.3f} arcmin vs gate {gate}'
    return False, f'G16: {job.name} not in the VLSSr table'


def run_conform(args, run, jobs, explicit_jobs=False):
    from exopipe.phase3.core.record import load_record
    C = SC.cfg()
    K = C.mosaic_conform
    state = {}
    for job in jobs:
        name = SC.unit_name('conform', job)
        if job.variant not in K.VARIANTS or job.scope not in K.SCOPES:
            why = (f'not staged: {job.variant} not in mosaic_conform.VARIANTS {K.VARIANTS} (push Taper only, '
                   f'R-JYBEAM-TOKEN-AND-MOSAIC-V2-PUSH-TAPER-ONLY)' if job.variant not in K.VARIANTS else
                   f'not staged: {job.scope} not in mosaic_conform.SCOPES (no sub-band staging mode exists)')
            if explicit_jobs and job.variant not in K.VARIANTS and not run.waived('mosaic.conform_taper_only'):
                with run.unit(name, job=job.name) as u:
                    u.refuse('G12: ' + why + ' - staging it needs --waive mosaic.conform_taper_only --ruling "..."')
                state[job.name] = 'refused'
            else:
                _say(f'mosaic conform {job.name}: {why}')
            continue
        crec = load_record(run.out_root, SC.STEP, SC.unit_name('coadd', job))
        rrec = load_record(run.out_root, SC.STEP, SC.unit_name('render', job))
        inputs = [o['path'] for o in (crec or {}).get('outputs', [])] + [o['path'] for o in (rrec or {}).get('outputs', [])]
        if run.resume and not run.fresh and crec and rrec:
            ok, why = run.resume_check(name, inputs=inputs, admitted=crec.get('admitted'))
            if ok:
                u = run.unit(name, job=job.name)
                u.status, u.reason = 'skipped-valid', why
                _say(f'mosaic conform {job.name}: skipped (resume: {why})')
                state[job.name] = 'skipped-valid'
                continue
        try:
            with run.unit(name, job=job.name) as u:
                if not crec or crec.get('status') != 'done':
                    u.refuse(f"conform needs a done coadd record ({'none' if not crec else crec.get('status')})")
                if not rrec or rrec.get('status') != 'done':
                    u.refuse(f"conform needs a done render record ({'none' if not rrec else rrec.get('status')})")
                if set(crec['meta'].get('domains') or ()) != {'jysr', 'jybeam'}:
                    u.refuse(f"conform stages both domains; the coadd record has {crec['meta'].get('domains')}")
                for p in inputs:
                    u.add_input(p)
                u.set_admitted(crec.get('admitted') or [])
                ok, detail = _vlssr_gate(run, job)
                u.check('mosaic.G16_vlssr', ok, None, detail)
                if not ok:
                    u.refuse(detail)
                if crec.get('meta', {}).get('skipped'):
                    u.note(f"tiles skipped by the engine guard: {crec['meta']['skipped']}")
                try:
                    staged, info = stage_job(job, crec, rrec, run.out_root)
                except RuntimeError as e:
                    u.refuse(str(e))
                u.meta.update(info)
                tiles = crec['meta'].get('tiles') or []
                for dst, src, kind in staged:
                    u.add_output(dst, 'mosaic', source=src, tiles=tiles, kind=kind)
            state[job.name] = u.status
        except Exception as e:
            _say(f'mosaic conform {job.name}: error {type(e).__name__}: {e}')
            state[job.name] = 'error'
    return state


def run_tables(args, run):
    """conform/_tables: the qa record's tables copied into <staging>/tables/ (push_mosaic_v2_20260929.py:22, :52-53)."""
    from exopipe.phase3.core.record import load_record, NoInput
    stg = os.path.join(SC.layout(run.out_root, 'staging'), 'tables')
    rec = load_record(run.out_root, SC.STEP, 'qa/_all')
    try:
        with run.unit('conform/_tables') as u:
            if not rec or rec.get('status') != 'done':
                raise NoInput('no done qa record: run the qa stage first')
            os.makedirs(stg, exist_ok=True)
            for o in rec.get('outputs', []):
                src = o['path']
                u.add_input(src)
                dst = os.path.join(stg, os.path.basename(src))
                tmp = os.path.join(stg, f'.{os.path.basename(src)}.tmp.{os.getpid()}')
                shutil.copy2(src, tmp)
                os.replace(tmp, dst)
                u.add_output(dst, 'mosaic_qa_table', kind='anchor' if 'anchor' in os.path.basename(src) else 'vlssr')
        return u.status
    except Exception as e:
        _say(f'mosaic conform tables: error {type(e).__name__}: {e}')
        return 'error'
