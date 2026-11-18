"""Stage `wideband` of the hybrid route (spec plan/specs/hybrid.md 2.4): the Red / Green / Blue NoTaper reconstructions, the
inverse-variance (1/THERMNOI^2) mean of the members' reconstructions, with the split beam correction carried through. A close
port of production exopipe/phase3/notaper_largescale.py (NL) at exopipe main 4c7098c: combine_wideband NL 954-1074 (WCS_KEYS /
CARRY_KEYS 950-951) and test_wideband_identity 1868-1879; the differences are those listed in lib/hybrid/recon.py (config
constants through recon.bind(), ReconRefusal, memmap=False, the record through a temporary name), and WIDEBANDS = the config's
hybrid.WIDEBAND_MEMBERS (production NL 944-947, Full excluded: the full stage makes Full).
"""
import glob
import json
import os

import numpy as np
from astropy.io import fits

from exopipe.phase3.core.products import build_name, parse_name, write_standard_header
from exopipe.phase3.lib.hybrid import recon as REC
from exopipe.phase3.lib.hybrid.common import VARIANT, diag_header, rd
from exopipe.phase3.lib.hybrid.recon import (ReconRefusal, _dump, _level_verdict, compact_ratio, installed_master, lscaveat_cards,
                                             test_compact_sources, test_pbcorr_split)
from exopipe.phase3.lib.install.steps_common import md5
from exopipe.phase3.lib.masters import coadd as CFP
from exopipe.phase3.lib.masters import pb_apply as PBA

WCS_KEYS = ('NAXIS1', 'NAXIS2', 'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2')
CARRY_KEYS = WCS_KEYS[2:] + ('CUNIT1', 'CUNIT2', 'LONPOLE', 'LATPOLE', 'RADESYS', 'EQUINOX', 'SPECSYS', 'TIMESYS', 'MJDREF', 'BUNIT', 'DEWARPED', 'ALIGNED')


def combine_wideband(lst, scope, v, run_dir):
    """Red / Green / Blue NoTaper reconstruction = the inverse-variance mean of the cell's five subband reconstructions (Gregg 2026-10-26
    "Aproved"; Blue 2026-10-31 "ok on caveats. Go on everything", its large-scale level a recorded caveat):
    weights 1/THERMNOI^2 of each subband master (V-based thermal noise; Phase 2 weights each night's subband images by V rms),
    on the shared grid (identical WCS required); NaN wherever a subband is NaN (as phase2_science_agg._stack_images); beam = the
    weighted mean BMAJ/BMIN/BPA. The split beam correction is carried through: compact = weighted mean of the subbands' restored point
    components; full and diffuse beams = weighted means of the subbands' maps; sibling = compact / full + rest / diffuse. Time block =
    the union of the subbands' nights: earliest start, latest end, INTTIME = per night the longest subband integration, summed (Phase 2:
    not a sum over frequency). All five subbands must hold a passed reconstruction."""
    from astropy.time import Time
    from exopipe.phase3.config import CFG
    g = REC.bind(); CAVEAT_SCOPES, DEC_BANDS, ARCHIVE_MASTERS = g['CAVEAT_SCOPES'], g['DEC_BANDS'], g['ARCHIVE_MASTERS']   # v2: config
    WIDEBANDS = CFG.hybrid.WIDEBAND_MEMBERS
    if scope == 'Full':
        raise ReconRefusal(f'{lst} Full {v}: the Full band is made by hybrid_full (three subband grids), not combine_wideband - REFUSING')
    out = f'{run_dir}/{lst}_{scope}'; os.makedirs(out, exist_ok=True); rob = VARIANT[v][2]; subs = []
    for b in WIDEBANDS[scope]:
        wd = f'{run_dir}/{lst}_{b}'; jf = f'{wd}/{v}_lsrecon.json'
        if not os.path.exists(jf):
            raise ReconRefusal(f'{lst} {scope} {v}: {b} holds no passed reconstruction ({jf}) - REFUSING')
        r = json.load(open(jf)); s = json.load(open(f'{wd}/{v}_stack.json')); h = fits.getheader(r['product']['path'])
        tn, tsrc = h.get('THERMNOI'), 'reconstruction (installed NoTaper master)'
        ow = 'R0' if v == 'R075' else 'R075'       # THERMNOI is one value per cell, identical in its four masters (NoTaper / Taper 1.0000 in
        for tp_, v_ in ((True, v), (False, ow), (True, ow)):   # 225 pairs at 41-59 MHz, R075 / R0 1.0000 in 71 at 50-59 MHz; 2026-10-30): when
            if tn is not None and float(tn) > 0:   # the reconstruction's (= installed NoTaper master's) header lacks it, read the cell's others
                break
            tn = fits.getheader(installed_master(lst, b, v_, tp_)[0]).get('THERMNOI')
            tsrc = f"installed {'Taper' if tp_ else 'NoTaper'} {v_} master of the cell (the NoTaper {v} header lacks it)"
        if tn is None or not float(tn) > 0:
            raise ReconRefusal(f'{lst} {b} {v}: no THERMNOI for the weight - REFUSING')
        pbf = f'{wd}/inputs/{PBA.deep_pb_name(b, v)}'; ph = fits.getheader(pbf); pb = rd(pbf)
        ch, _ = PBA.chromatic_gain(ph, pb.shape, float(PBA.cfg.SUBBAND_FREQ_HZ[b]['center']), float(ph['PBBMAJ']))
        subs.append(dict(band=b, hdr=h, span=s['span'], name=r['product']['prodname'], A=rd(r['product']['path']),
                         C=rd(f'{wd}/{v}_lsrecon_compact_restored.fits'), pbf=pb, pbd=pb / ch, w=1.0 / float(tn) ** 2, thermnoi=float(tn),
                         thermnoi_source=tsrc, level=r.get('test_large_scale_level') or {}))
    for k in WCS_KEYS:
        if len({s['hdr'][k] for s in subs}) != 1:
            raise ReconRefusal(f'{lst} {scope} {v}: subband grids differ in {k} {[s["hdr"][k] for s in subs]} - REFUSING')
    W = np.array([s['w'] for s in subs]); Wn = W / W.sum()
    A = sum(wn * s['A'] for wn, s in zip(Wn, subs)); C = sum(wn * s['C'] for wn, s in zip(Wn, subs))
    PBF = sum(wn * s['pbf'] for wn, s in zip(Wn, subs)); PBD = sum(wn * s['pbd'] for wn, s in zip(Wn, subs))
    nights = {}
    for s in subs:
        for n in s['span']['nights']:
            d = nights.setdefault(n['date'], dict(start=n['start'], end=n['end'], inttime_s=0.0))
            d['start'] = min(d['start'], n['start'], key=lambda t: Time(t).mjd); d['end'] = max(d['end'], n['end'], key=lambda t: Time(t).mjd)
            d['inttime_s'] = max(d['inttime_s'], float(n['inttime_s']))
    span = dict(start=min((n['start'] for n in nights.values()), key=lambda t: Time(t).mjd),
                end=max((n['end'] for n in nights.values()), key=lambda t: Time(t).mjd), inttime_s=sum(n['inttime_s'] for n in nights.values()))
    procs = [set(parse_name(s['name'])['proc']) for s in subs]; dates = sorted(nights)
    name = build_name(scope, 'I', 'deep', 'NoTaper', rob, proc=sorted(set.intersection(*procs)), lst=lst, date_start=dates[0].replace('-', ''),
                      date_end=dates[-1].replace('-', ''), n_epochs=len(dates))
    h0 = subs[0]['hdr']; h = fits.Header()
    for k in CARRY_KEYS:
        if k in h0:
            h[k] = (h0[k], h0.comments[k])
    h['NCOADD'] = (len(dates), 'distinct nights over the combined subbands'); h['COADDOP'] = ('lsrecon-sbcomb-v1', 'subband lsrecon combination')
    h['LSRECON'] = (True, 'NoTaper large-scale reconstruction (notaper_largescale)')
    h['LSWBMETH'] = ('ivw mean of subband lsrecon masters', 'weights 1/THERMNOI^2 per subband')
    h['LSWBSB'] = (','.join(s['band'] for s in subs), 'combined subbands')
    h['LSWBW'] = (','.join(f'{x:.4f}' for x in Wn), 'normalized subband weights')
    level = None
    if scope in CAVEAT_SCOPES:                    # Blue's large-scale level (recorded caveat, 2026-10-31): the ratio of the subbands' 4-deg model
        if any('model_mean_by_dec_band' not in s['level'] for s in subs):              # and prior means summed with the combination weights
            raise ReconRefusal(f'{lst} {scope} {v}: a subband record lacks the large-scale level means - REFUSING')
        keys = [k for k in (f'{lo:+d}..{hi:+d}' for lo, hi in DEC_BANDS) if all(k in s['level']['model_mean_by_dec_band'] for s in subs)]
        am = {k: float(sum(wn * s['level']['model_mean_by_dec_band'][k] for wn, s in zip(Wn, subs))) for k in keys}
        bm = {k: float(sum(wn * s['level']['prior_mean_by_dec_band'][k] for wn, s in zip(Wn, subs))) for k in keys}
        level = _level_verdict({k: round(am[k] / bm[k], 4) for k in keys}, scope, model_mean_by_dec_band=am, prior_mean_by_dec_band=bm,
                               method="ratio of the subbands' 4-deg model and prior means summed with the combination weights (LSWBW)",
                               subband_verdicts={s['band']: s['level'].get('verdict') for s in subs})
        for k, val, com in lscaveat_cards(level):
            h[k] = (val, com)
    for d in dates:
        h['HISTORY'] = f'COADD constituent {d}'
    for s in subs:
        h['HISTORY'] = f'lsrecon subband {s["band"]}: {s["name"]}'
    beam = tuple(float(sum(wn * float(s['hdr'].get(k, 0.0)) for wn, s in zip(Wn, subs))) for k in ('BMAJ', 'BMIN', 'BPA'))
    thermnoi = float(1.0 / np.sqrt(W.sum()))
    path = f'{out}/I_deep_NoTaper_{v}_lsrecon_{scope}.fits'; fits.writeto(path, A.astype(np.float32), h, overwrite=True)
    write_standard_header(path, scope=scope, stokes='I', start=span['start'], end=span['end'], inttime_s=span['inttime_s'], time_basis='coadd-span',
                          beam=beam, thermnoi=thermnoi, imgrms=CFP.get_uniform_rms(A, ~np.isfinite(A) | (A == 0)), pbcorr=False, dewarped=True,
                          aligned=True, msub=False, ssub=False, extra={'PRODNAME': (name, 'Phase 3 product name (grammar; CONTINUE cards)'),
                                                                     'MSTRGEN': ('lsrecon-sbcomb-v1', 'master generation of this product')}, validate=True)
    rec = dict(lst=lst, scope=scope, weighting=v, subbands=[dict(band=s['band'], weight=float(wn), product=s['name'], thermnoi=s['thermnoi'],
                                                             thermnoi_source=s['thermnoi_source']) for wn, s in zip(Wn, subs)],
               span=span, n_nights=len(dates), beam=beam, thermnoi=thermnoi)
    if level:
        rec['test_large_scale_level'] = level
    rec['test_wideband_identity'] = test_wideband_identity(path, subs, Wn, name)
    ph_ = fits.getheader(path)                    # v2: the written product's header, read once (production re-read it per use)
    fits.writeto(f'{out}/{v}_lsrecon_compact_restored.fits', C.astype(np.float32), diag_header(ph_, 'combined restored point components'),
                 overwrite=True)
    inst = [f for f in glob.glob(f'{ARCHIVE_MASTERS}/{lst}/{scope}/{scope}_I_deep_NoTaper_{"Robust-0.75" if v == "R075" else "Robust+0.0"}_*coadd_LST{lst}_*.fits')
            if '_pbcorr_' not in f]
    if len(inst) != 1:
        raise ReconRefusal(f'{lst} {scope} {v}: {len(inst)} installed wideband masters {inst} - REFUSING')
    ih = fits.getheader(inst[0])
    if any(ih[k] != ph_[k] for k in WCS_KEYS):
        raise ReconRefusal(f'{lst} {scope} {v}: installed wideband master grid differs - REFUSING')
    rec['installed_wideband_master'] = inst[0]
    ref = rd(inst[0])
    if scope in CAVEAT_SCOPES:                    # Gregg 2026-11-04 "a" (ledger R-GREGG-20261104-BLUE-COMPACT-TEST-REFERENCE-SUBBAND-MASTERS): where
        sub_ref = sum(wn * rd(installed_master(lst, s['band'], v, False)[0]) for wn, s in zip(Wn, subs))   # the installed Blue master is
        cons = compact_ratio(sub_ref, ref, ph_, lst, scope); rec['installed_wideband_vs_subbands'] = cons      # inconsistent with its own
        if not (cons['n'] >= 50 and abs(cons['median'] - 1) < cons['tolerance']):                              # subbands (NoTaper Robust 0 at
            ref = sub_ref                                                                                      # 11h, 13h, 15h-22h: 0.76-0.89),
            rec['compact_reference'] = (f'1/THERMNOI^2 mean of the installed subband masters (the installed {scope} master is inconsistent '  # the
                                        f'with them: median {cons["median"]:.3f})')                        # test's reference is their mean
    rec['test_compact_sources'] = test_compact_sources(rd(path), ref, ph_, lst, scope)
    valid = np.isfinite(PBF) & (PBF > PBA.PB_FLOOR) & np.isfinite(A) & np.isfinite(PBD) & (PBD > 0)
    P = np.full(A.shape, np.nan); P[valid] = C[valid] / PBF[valid] + (A[valid] - C[valid]) / PBD[valid]
    pn = parse_name(name); pn['proc'] = list(pn['proc']) + ['pbcorr']; pname = build_name(**pn); ppath = f'{out}/I_deep_NoTaper_{v}_lsrecon_pbcorr_{scope}.fits'
    hp = ph_.copy(); hp['PBFLOOR'] = (PBA.PB_FLOOR, 'PB floor below which pixels set NaN')
    hp['PBMETHOD'] = ('lsrecon-split', 'compact / full PB; extended + step + residual / diffuse PB')
    hp['PBMAP'] = ('ivw mean of subband PB maps', 'same weights as the images (LSWBW)')
    fits.writeto(ppath, P.astype(np.float32), hp, overwrite=True)
    write_standard_header(ppath, scope=scope, stokes='I', start=span['start'], end=span['end'], inttime_s=span['inttime_s'], time_basis='coadd-span',
                          beam=beam, thermnoi=thermnoi, pbcorr=True, dewarped=True, aligned=True, msub=False, ssub=False,
                          extra={'PRODNAME': (pname, 'Phase 3 product name (grammar; CONTINUE cards)')}, validate=True)
    rec['product'] = dict(path=path, prodname=name, md5=md5(path))
    rec['product_pbcorr'] = dict(path=ppath, prodname=pname, md5=md5(ppath), test_pbcorr_split=test_pbcorr_split(ppath, A, C, PBF, PBF / PBD, valid))
    _dump(rec, f'{out}/{v}_lsrecon_wb.json')
    return rec


def test_wideband_identity(path, subs, Wn, name):
    """The written wideband file must equal the weighted mean of the subband reconstructions read from disk (to 1e-6 of the largest
    value), be NaN exactly where any subband is NaN, and carry its PRODNAME."""
    back = rd(path); ref = sum(wn * s['A'] for wn, s in zip(Wn, subs))                # subband files as read from disk
    nan_ok = bool(np.array_equal(np.isfinite(back), np.isfinite(ref))); f = np.isfinite(back) & np.isfinite(ref)
    res = dict(max_rel_err=float(np.max(np.abs(back[f] - ref[f])) / np.max(np.abs(ref[f]))), nan_masks_equal=nan_ok,
               prodname_ok=fits.getheader(path).get('PRODNAME') == name)
    if not (res['max_rel_err'] < 1e-6 and nan_ok and res['prodname_ok']):
        raise ReconRefusal(f'test_wideband_identity FAILED: {res} - REFUSING')
    res['verdict'] = 'pass'
    return res


