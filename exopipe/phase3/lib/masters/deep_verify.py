"""T1-T8 of a deep master generation (class `master_generation`): port of ops/membership_rebuild_20261007/check_rebuild.py
(the content test production runs on <scratch>/<cell> before install). One function per test, each returning
(passed, value, detail) with production's own measurement and wording, so `--waive master_generation.T<n> --ruling ...`
replaces the hand-edited REBUILD_CONTENT_OK waiver lines. Differences from check_rebuild.py, all pure (no product change):
  - the scale table is read from its Lustre copy (masters_fluxscale.SCALE_TABLE, sha1 equal to the lwacalim07 /fast copy
    check_rebuild.py:28 reads; pinned) - the test runs on any node;
  - T3 tests Stokes V too (masters_verify.T3_STOKES; R-LOO-PER-STOKES-RATIFIED is per Stokes; check_rebuild.py:70 tests I);
  - T6 reads each master once per Stokes (fitsio cache), not once per night (review item 14);
  - the cell directory and the generation directory are arguments (no hard-coded scratch root).
Reads only; writes nothing.

Layout (production's scratch cell): R = <root>/<cell>; G = R/<OUTGEN> (gen_shflux) with coadd_final/ (+ loo/) and
subtracted/ carrying the '_shflux' token before the scope; R/coadd_final/{weights.csv,night_image_gate.csv};
R/residuals_provisional.csv; the native reference R/prep/ref_master_prev.fits or R/coadd_final_ref/I_deep_Taper_R075_<sc>.fits.
"""
import csv
import glob
import os

import numpy as np

_CACHE = {}


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def cell_of_generation(gen_dir):
    """(cell_dir R, cell name, lst, scope) of a generation directory <R>/<OUTGEN>."""
    R = os.path.dirname(os.path.abspath(gen_dir.rstrip('/')))
    cell = os.path.basename(R)
    lst, sc = cell.split('_', 1)
    return R, cell, lst, sc


def _rd(p):
    from astropy.io import fits
    return np.squeeze(fits.getdata(p)).astype(np.float64)


def _tables():
    """(v4 {lst: scale}, corr table) - pinned sha1 checked."""
    from exopipe.phase3.lib.masters import deep_inplace as IP
    from exopipe.phase3.lib.masters import flux_correction as FC
    s = _cfg().masters_fluxscale
    key = ('tables', s.SCALE_TABLE, s.FLUXCORR_TABLE)
    if key not in _CACHE:
        IP.sha1_pin(s.SCALE_TABLE, s.SCALE_TABLE_SHA1, 'scale table v4')
        IP.sha1_pin(s.FLUXCORR_TABLE, s.FLUXCORR_TABLE_SHA1, 'flux correction table v2')
        v4 = {r['lst']: float(r['scale']) for r in csv.DictReader(open(s.SCALE_TABLE))}
        _CACHE[key] = (v4, FC.load_table(s.FLUXCORR_TABLE))
    return _CACHE[key]


def sname(base, sc, token=None):
    token = _cfg().masters_fluxscale.TOKEN if token is None else token
    return base[:-len(f'_{sc}.fits')] + f'{token}_{sc}.fits'


def _ctx(gen_dir):
    R, cell, lst, sc = cell_of_generation(gen_dir)
    return dict(R=R, G=os.path.abspath(gen_dir.rstrip('/')), cell=cell, lst=lst, sc=sc)


def _weights(R):
    return {x['date']: x for x in csv.DictReader(open(f'{R}/coadd_final/weights.csv'))}


def t1(gen_dir):
    """T1 flux scale: scaled survey master / (native reference x v4[LST] x v2[scope]) at the reference's compact peaks."""
    from scipy.ndimage import maximum_filter
    c = _ctx(gen_dir)
    R, G, lst, sc = c['R'], c['G'], c['lst'], c['sc']
    vs = _cfg().masters_verify
    v4, corr = _tables()
    if lst not in v4 or sc not in corr:
        return False, None, f'no scale-table row for {lst} ({lst in v4}) or correction row for {sc} ({sc in corr})'
    want = v4[lst] * corr[sc]['corr']
    ref = None
    for p in (f'{R}/prep/ref_master_prev.fits', f'{R}/coadd_final_ref/I_deep_Taper_R075_{sc}.fits'):
        if os.path.exists(p):
            ref = _rd(p)
            break
    if ref is None:
        return False, None, 'no reference copy to compare against'
    new = _rd(f'{G}/coadd_final/{sname(f"I_deep_Taper_R075_{sc}.fits", sc)}')
    pk_ = vs.T1_PEAKS
    rr = 1.4826 * np.nanmedian(np.abs(ref - np.nanmedian(ref)))
    rf = np.nan_to_num(ref)
    pk = (rf == maximum_filter(rf, size=int(pk_['box_px']))) & (rf > pk_['snr'] * rr) & (rf < pk_['max_jy'])
    yy, xx = np.nonzero(pk)
    h, w = ref.shape
    k = np.hypot(xx - w / 2, yy - h / 2) / (w / 2) < pk_['radius']
    yy, xx = yy[k], xx[k]
    r = new[yy, xx] / (ref[yy, xx] * want)
    med = float(np.nanmedian(r)) if len(r) else float('nan')
    ok = bool(np.isfinite(med) and abs(med - 1) <= vs.T1_TOL)
    det = (f'scaled master / (native reference x {want:.6f}) at {len(yy)} peaks: median {med:.4f} (16-84%: '
           f'{np.nanpercentile(r, 16):.4f}-{np.nanpercentile(r, 84):.4f})' if len(r) else 'no reference peaks')
    return ok, round(med, 5) if np.isfinite(med) else None, det


def t2(gen_dir):
    """T2 gate: a night-image record row for every weights.csv night; no FAIL night in the master; flux_dc for members."""
    c = _ctx(gen_dir)
    R = c['R']
    w = _weights(R)
    g = {x['date']: x for x in csv.DictReader(open(f'{R}/coadd_final/night_image_gate.csv'))}
    miss = sorted(set(w) - set(g))
    fin = sorted(d for d, x in g.items() if x['verdict'] == 'FAIL' and w.get(d, {}).get('in_master') == '1')
    nodc = sorted(d for d, x in w.items() if x.get('in_master') == '1' and not (x.get('flux_dc') or '').strip())
    nfail = sum(1 for x in g.values() if x['verdict'] == 'FAIL')
    return (not (miss or fin or nodc), len(g),
            f'{len(g)} gate rows ({nfail} FAIL); missing {miss}; FAIL in master {fin}; in-master without flux_dc {nodc}')


def t3(gen_dir):
    """T3 LOO: every LOO master's PARENTDG = the master's CONSTDIG and NCOADD = master's - 1; one per in-master night when
    2 <= N <= 20 (per Stokes: I and V)."""
    from astropy.io import fits
    c = _ctx(gen_dir)
    R, G, sc = c['R'], c['G'], c['sc']
    vs = _cfg().masters_verify
    lo, hi = _cfg().masters_build.LOO_N_RANGE
    w = _weights(R)
    inm = sorted(d for d, x in w.items() if x.get('in_master') == '1')
    parts, ok = [], True
    for st in vs.T3_STOKES:
        mvar = 'I_deep_Taper_R075' if st == 'I' else 'V_deep_Taper_R0'
        mh = fits.getheader(f'{G}/coadd_final/{sname(f"{mvar}_{sc}.fits", sc)}')
        n, cdig = mh.get('NCOADD'), mh.get('CONSTDIG')
        loos = sorted(glob.glob(f'{G}/coadd_final/loo/*_{st}_loo_*.fits'))
        lbad = []
        for p in loos:
            hh = fits.getheader(p)
            if hh.get('PARENTDG') != cdig or hh.get('NCOADD') != n - 1:
                lbad.append(os.path.basename(p))
        need = (lo <= n <= hi)
        # I: production's rule (check_rebuild.py:77) - one LOO per in-master night = NCOADD; V (added, pure): each master
        # its own NCOADD (R-LOO-PER-STOKES-RATIFIED: a V frame dropped on grid mismatch leaves V with fewer constituents)
        if st == 'I':
            good = not lbad and (not need or len(loos) == n == len(inm))
        else:
            good = not lbad and (not need or len(loos) == n)
        ok = ok and good
        parts.append(f'{st}: NCOADD {n}, in-master nights {len(inm)}, LOO masters {len(loos)} (needed: {need}); '
                     f'not derivatives: {lbad[:3]}')
    return ok, len(inm), '; '.join(parts)


def t4(gen_dir):
    """T4 PB siblings of the scaled generation: sibling x its robust's PB map = its plain; exactly T4_SIBLINGS siblings."""
    from exopipe.phase3.lib.masters import pb_apply as PB
    c = _ctx(gen_dir)
    G, sc = c['G'], c['sc']
    sb = []
    sibs = sorted(glob.glob(f'{G}/coadd_final/*_pbcorr_*.fits'))
    maps = {}
    for p in sibs:
        b = os.path.basename(p)
        plain = p.replace('_pbcorr_', '_', 1)
        mp = f'{G}/coadd_final/PB_deep_R075_{sc}.fits' if '_R075_' in b else f'{G}/coadd_final/PB_deep_{sc}.fits'
        if mp not in maps:
            maps[mp] = _rd(mp)
        okk, why = PB.sibling_check(p, plain, maps[mp])
        if not okk:
            sb.append(f'{b}: {why}')
    n = len(sibs)
    want = _cfg().masters_verify.T4_SIBLINGS
    return (not sb and n == want), n, f'{n} siblings ({want} expected); failing {sb}'


def t5(gen_dir):
    """T5 flux cards: every scaled FITS except PB_deep carries FLUXBAKE, SHSCALE = v4[LST], FLUXCORR = v2[scope]."""
    from astropy.io import fits
    from exopipe.phase3.lib.masters import flux_correction as FC
    c = _ctx(gen_dir)
    G, lst, sc = c['G'], c['lst'], c['sc']
    v4, corr = _tables()
    tol = _cfg().masters_fluxscale.CARD_TOL
    allf = glob.glob(f'{G}/**/*.fits', recursive=True)
    cb = []
    for p in allf:
        if os.path.basename(p).startswith(tuple(_cfg().masters_fluxscale.UNSCALED_PREFIX)):
            continue
        hh = fits.getheader(p)
        if not (FC.pixels_carry_correction(hh) and abs(float(hh.get('SHSCALE', -1)) - v4[lst]) < tol['SHSCALE']
                and abs(float(hh.get('FLUXCORR', -1)) - corr[sc]['corr']) < tol['FLUXCORR']):
            cb.append(os.path.basename(p))
    return not cb, len(allf), f'{len(allf)} files; without the re-bake cards: {cb[:3]}'


def t6(gen_dir):
    """T6 epochs: every in-master night's scaled epoch passes coadd._epoch_content_check (identity + I ratio), both Stokes."""
    from exopipe.phase3.core import fitsio
    from exopipe.phase3.lib.masters import coadd as CFP
    c = _ctx(gen_dir)
    R, G, sc = c['R'], c['G'], c['sc']
    inm = sorted(d for d, x in _weights(R).items() if x.get('in_master') == '1')
    from exopipe.phase3.lib.masters import input_resolver
    refused_v = {r['date'] for r in input_resolver.read_refusals(R) if r.get('product') == 'V'}   # B-03 same-state: no V epoch
    eb = []
    for st in ('I', 'V'):
        m = f'{G}/coadd_final/' + sname(('I_deep_Taper_R075' if st == 'I' else 'V_deep_Taper_R0') + f'_{sc}.fits', sc)
        mdat = np.asarray(fitsio.read_image(m)[0], dtype=np.float64)          # once per Stokes (review item 14)
        for d in inm:
            if st == 'V' and d in refused_v:
                continue
            e = f'{G}/subtracted/{d}_{st}_epoch{_cfg().masters_fluxscale.TOKEN}_{sc}.fits'
            s_ = f'{G}/subtracted/{d}_{st}_sub{_cfg().masters_fluxscale.TOKEN}_{sc}.fits'
            if not os.path.exists(e):
                eb.append(f'{d} {st}: no epoch')
                continue
            r = _rd(s_) if os.path.exists(s_) else None
            okk, why, _m, _n = CFP._epoch_content_arrays(_rd(e), mdat, r, st)
            if not okk:
                eb.append(f'{d} {st}: {why}')
    return not eb, len(inm), (f'{len(inm)} in-master nights x 2 Stokes; failing {eb[:4]}'
                              + (f'; V input refused (B-03 same-state) on {sorted(refused_v)}' if refused_v else ''))


def t7(gen_dir):
    """T7 date sets: residuals_provisional.csv dates = weights.csv dates."""
    c = _ctx(gen_dir)
    R = c['R']
    w = _weights(R)
    rp = {x['date'] for x in csv.DictReader(open(f'{R}/residuals_provisional.csv'))}
    return rp == set(w), len(w), f'residuals_provisional {len(rp)} dates, weights {len(w)}; differ {sorted(rp ^ set(w))[:4]}'


def t8(gen_dir):
    """T8 membership rulings on the cell (membership.verify) and what the scaled generation must carry
    (membership.verify_products with the '_shflux' token), RBF master dewarp ignored before the dewarp."""
    from exopipe.phase3.lib.masters import admit as AD, membership as MB
    c = _ctx(gen_dir)
    R, G, sc, cell = c['R'], c['G'], c['sc'], c['cell']
    tok = _cfg().masters_fluxscale.TOKEN
    mv = AD.verify(R)[0]                               # D-ADMIT: a night admitted into the master is not a violation
    tok_ = tok if os.path.exists(f'{G}/coadd_final/I_deep_Taper_R075{tok}_{sc}.fits') else ''
    ign = tuple(_cfg().masters_verify.T8_IGNORE)
    pv = [x for x in MB.verify_products(G, token=tok_, cell=cell) if not any(i in x for i in ign)]
    return (not mv and not pv), len(mv) + len(pv), f'membership {len(mv)} violations {mv[:2]}; products {len(pv)} {pv[:2]}'


TESTS = (('T1', t1), ('T2', t2), ('T3', t3), ('T4', t4), ('T5', t5), ('T6', t6), ('T7', t7), ('T8', t8))


def run_all(gen_dir):
    """[(name, passed, value, detail)] for T1-T8 (an exception is a failed test with its text)."""
    out = []
    for name, fn in TESTS:
        try:
            ok, val, det = fn(gen_dir)
        except Exception as e:
            ok, val, det = False, None, f'test raised {type(e).__name__}: {e}'
        out.append((name, bool(ok), val, det))
    return out
