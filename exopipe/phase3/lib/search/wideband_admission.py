"""Wideband member admission (R-GREGG-20261023-WIDEBAND-MEMBER-SEARCH-ADMISSION).

Gregg 2026-10-23, on the CLEAN-diverged 59 MHz member of 11h 2024-12-23 Green: "approved and it will prevent that wideband image from
being created. Perhaps use the same QA criteria used for assessing if an image is included in the transient search." and "If a subband
is missing the wideband does not get created. There would be wideband Green Stokes V, but not Stokes I for this 1 hour block." With the
standing rules on the wideband's own set: 2026-08-21 "No wideband product is ever made if any required sub-band is missing"; the one
allowance 18 MHz (SUBBAND_ONLY_EXCL; 2026-10-17 "Red can be built without 18 MHz", "Full can also exist if 18 MHz is excluded",
"18 MHz data fails QA regularly, and we have the rule that Red can still be formed if 18 MHz data are missing").

A wideband epoch (Red / Green / Blue / Full) is searched in a Stokes only when (lib/masters/coadd helpers, headers only):
  IMAGE   that Stokes' Phase 2 image of the night (I: the alignment image, V: the Taper Robust-0 deep image, from the night's sci_dir
          in the wideband cell's weights.csv) shows its stacked subband set (coadd.wideband_stacked_subbands); the set holds every
          required subband (the group minus 18 MHz);
  MEMBERS every subband IN THAT SET except 18 MHz passes, in that Stokes, the deep search's own per-epoch admission, mirrored from the member cell's
          records in the search's order (lib/search/deep_search.run_deep_cell):
            present      the night has an I product in the member's subtracted/ (the search's epoch list) or is an admitted night;
            data_decisions  not excluded;                                                         (both Stokes)
            night-image and epoch gates  passed (incl. D-ADMIT verdicts of admitted nights);       (both Stokes)
            sanity       qa3 rms <= 10 x the cell median (both Stokes) and, I only, <= SANITY_I_X x it (weights.csv; median =
                         the upper middle of the sorted finite values, as the engine takes it; fewer than SANITY_MIN nights: skipped);
            search tier  stage-1 rms <= median + SIGMA x 1.4826 MAD of the cell's nights, PER STOKES (residuals_provisional.csv,
                         which must be this generation's - its dates equal weights.csv's - or the member REFUSES; an rms of inf is
                         tested and refused, as the engine does; fewer than TIER_MIN nights: skipped; admitted nights tested, never
                         in the population);
            input        both Stokes' products carry a positive beam (a same-state V refusal: the V side only);  (both Stokes)
            beam gate    I-product beam area <= BEAM_RATIO x the median over the member nights the gates above leave (fewer than
                         BEAM_MIN nights: skipped); a night without a readable beam REFUSES the member, as the engine refuses the cell.
  NOT mirrored (they need the member's search products): alignment flows (a night without a flow), the V sanity gate (the V
  residual frame's rms), extraction failures. A member cell without its records REFUSES (RuntimeError naming the member).
THE ONE EXCEPTION, 18 MHz (Gregg 2026-10-23: "No! The one exception to the exclusion rule is 18 MHz"): an 18 MHz member that is
excluded or fails admission never refuses Red / Full; going forward Red / Full are built without it, and an existing image holding it
is kept (as the masters: 2026-10-17 "enforce going forward, no rebuild now"; 18 MHz is always downweighted - 0.5% median, 2% max of
a Red master's weight). Until 2026-10-23 this module refused an image holding an excluded 18 MHz and judged an 18 MHz member like
the others (CORRECTION-12H-RED-18MHZ-IS-THE-CARVE-OUT-NOT-A-DEEP-ISSUE-20261023).
A failing member refuses the wideband epoch in that Stokes - never a restack from a subset. Refusals are written one row per (night,
Stokes, member or image, reason) by the caller (refusals/<LST>_<band>_wbmember_excluded.csv).
"""
import csv
import glob
import math
import os
import re

GROUPS = ('Red', 'Green', 'Blue', 'Full')
EXCEPTION = '18MHz'          # the one member whose exclusion or failure never refuses a wideband (Gregg; module docstring)


def members(scope, subband_bands):
    """The member subbands of a wideband scope (Full = every subband of Red, Green and Blue), or () for a subband."""
    if scope == 'Full':
        return tuple(sb for g in ('Red', 'Green', 'Blue') for sb in subband_bands[g])
    return tuple(subband_bands.get(scope, ()))


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v               # NaN -> None; inf kept (the engine refuses it)


def _median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def tier_failures(rows, sigma, min_n, extra=None):
    """{stokes: {date: (rms, threshold)}} of the search-tier gate on one cell's residuals_provisional rows, per Stokes, as the engine:
    the population = the positive finite values (np.median and its MAD); every value with a date is TESTED (inf refused); a Stokes
    with fewer than min_n population values: {} (the engine skips the gate). extra = {stokes: {date: rms}}: nights tested WITHOUT
    joining the population (D-ADMIT's admitted nights, admissions.csv stage2_<s>_rms_mjy)."""
    out = {'I': {}, 'V': {}}
    for st, col in (('I', 'i_rms_mjy'), ('V', 'v_rms_mjy')):
        vals = {}
        for r in rows:
            if r.get(col) in (None, '', 'nan') or not r.get('date'):
                continue
            x = _num(r.get(col))
            if x is not None:
                vals[r['date']] = x
        pop = sorted(x for x in vals.values() if math.isfinite(x) and x > 0)
        if len(pop) < min_n:
            continue
        med = _median(pop)
        thr = med + sigma * 1.4826 * _median([abs(x - med) for x in pop])
        tested = dict(vals)
        tested.update((d, x) for d, x in ((extra or {}).get(st) or {}).items() if x is not None)
        out[st] = {d: (x, thr) for d, x in tested.items() if x > thr}
    return out


def sanity_failures(weights_rows, x_both, x_i, min_n, admitted_qa3=None):
    """(both, i_only): {date: (qa3, median)} of the Stokes I alignment sanity gates on weights.csv qa3_rms_mjy, as the engine: the
    median = the upper-middle element of the sorted positive finite values of the record (_fin[len // 2]); admitted nights' values
    added after the median; > x_both x median refuses both searches, > x_i x median (and not already) the I search; fewer than min_n
    values: no refusal (the engine skips the gate)."""
    vals = {}
    for r in weights_rows:
        if r.get('qa3_rms_mjy') in (None, '', 'nan') or not r.get('date'):
            continue
        x = _num(r.get('qa3_rms_mjy'))
        if x is not None:
            vals[r['date']] = x
    fin = sorted(v for v in vals.values() if math.isfinite(v) and v > 0)
    if len(fin) < min_n:
        return {}, {}
    med = fin[len(fin) // 2]
    vals.update((d, x) for d, x in (admitted_qa3 or {}).items() if x is not None)
    both = {d: (x, med) for d, x in vals.items() if x > x_both * med}
    i_only = {d: (x, med) for d, x in vals.items() if d not in both and x > x_i * med}
    return both, i_only


def beam_area(cell_dir, member, lst, date, phase2_root='/lustre/pipeline/exopipe/phase2'):
    """The night's fitted-beam area (deg^2) as the engine's beam gate reads it: the I subtracted product's BMAJ x BMIN, else the
    Phase 2 deep I Taper Robust-0.75 image's (dewarped preferred). None when no header shows it."""
    from astropy.io import fits
    try:
        h = fits.getheader(os.path.join(cell_dir, 'subtracted', f'{date}_I_sub_{member}.fits'))
        if h.get('BMAJ') and h.get('BMIN'):
            return float(h['BMAJ']) * float(h['BMIN'])
    except Exception:
        pass
    g = sorted([x for x in glob.glob(f'{phase2_root}/{lst}/{date}/*/{member}/I/deep/*_I_deep_Taper_Robust-0.75*.fits')
                + glob.glob(f'{phase2_root}/{lst}/{date}/*/{member}/I/deep/*Taper-*-Robust-0.75-*.fits')
                if 'pbcorr' not in os.path.basename(x)], key=lambda x: (0 if 'dewarped' in os.path.basename(x) else 1, x))
    if g:
        try:
            h = fits.getheader(g[0])
            return float(h['BMAJ']) * float(h['BMIN'])
        except Exception:
            pass
    return None


def _beam_ok(path):
    from astropy.io import fits
    try:
        h = fits.getheader(path)
        return float(h.get('BMAJ') or 0) > 0 and float(h.get('BMIN') or 0) > 0
    except Exception:
        return False


def member_record(cell_dir, member, lst, p, readers, admitted=({}, {})):
    """The deep search's admission of every night of one member cell, mirrored from its records (module docstring). p = dict(sigma,
    tier_min, sanity_x_both, sanity_x_i, sanity_min, beam_ratio, beam_min); readers = dict(night_failed(final_dir), epoch_failed(
    final_dir, band), excluded(lst, band, date) -> bool, same_state_v(cell_dir) -> set, beam_area(cell_dir, member, lst, date)).
    admitted = (accepted, refused) of lib/search/admitted.nights(cell_dir, member). Returns dict(present, both={night: [reasons]},
    i_only={night: [reasons]}, v_only={night: [reasons]}, tier={stokes: {night: (rms, thr)}}). REFUSES (RuntimeError naming the
    member) when its records are missing or stale, or a night it would search has no readable beam."""
    final = os.path.join(cell_dir, 'coadd_final')
    wp = os.path.join(final, 'weights.csv')
    pp = os.path.join(cell_dir, 'residuals_provisional.csv')
    for f in (wp, pp):
        if not os.path.exists(f):
            raise RuntimeError(f'wideband member {member}: {f} missing - the member admission cannot run; REFUSING')
    adm, adm_bad = admitted
    wrows = list(csv.DictReader(open(wp)))
    prows = list(csv.DictReader(open(pp)))
    wdates = {r['date'] for r in wrows if r.get('date')}
    pdates = {r['date'] for r in prows if r.get('date')}
    if wdates != pdates:
        raise RuntimeError(f'wideband member {member}: residuals_provisional.csv is not this generation\'s (dates '
                           f'{sorted(wdates ^ pdates)[:4]} differ from weights.csv) - stale derivative; REFUSING')
    prod = set()
    for c in glob.glob(os.path.join(cell_dir, 'subtracted', f'*_I_sub_{member}.fits')):
        m = re.search(r'(\d{4}-\d\d-\d\d)_I_sub_', os.path.basename(c))
        if m:
            prod.add(m.group(1))
    present = prod | set(adm)
    both, i_only, v_only = {}, {}, {}
    try:
        ng = set(readers['night_failed'](final))
        eg = set(readers['epoch_failed'](final, member))
    except FileNotFoundError as e:
        raise RuntimeError(f'wideband member {member}: gate record missing ({e}); REFUSING')
    ng |= {d for d, r in adm.items() if r.get('night_image') == 'FAIL'}
    eg |= {d for d, r in adm.items() if r.get('epoch_gate') == 'FAIL'}
    for d in sorted(present):
        if readers['excluded'](lst, member, d):
            both.setdefault(d, []).append('data_decisions exclusion')
    for d in sorted(ng):
        both.setdefault(d, []).append('night-image gate')
    for d in sorted(eg):
        both.setdefault(d, []).append('epoch gate')
    for d in sorted(adm_bad):
        both.setdefault(d, []).append('admitted night refused by the search')
    s_both, s_i = sanity_failures(wrows, p['sanity_x_both'], p['sanity_x_i'], p['sanity_min'],
                                  {d: _num(r.get('qa3_rms_mjy')) for d, r in adm.items()})
    for d, (x, med) in s_both.items():
        both.setdefault(d, []).append('sanity gate (qa3 rms %.1f > %g x median %.1f mJy)' % (x, p['sanity_x_both'], med))
    for d, (x, med) in s_i.items():
        i_only.setdefault(d, []).append('I sanity gate (qa3 rms %.1f > %g x median %.1f mJy)' % (x, p['sanity_x_i'], med))
    vref = set(readers['same_state_v'](cell_dir)) | {d for d, r in adm.items()
                                                     if str(r.get('v_products') or '').startswith('refused (B-03 same-state)')}
    for d in sorted(present):
        for st in ('I', 'V'):
            fp = os.path.join(cell_dir, 'subtracted', f'{d}_{st}_sub_{member}.fits')
            if st == 'V' and d in vref and not os.path.exists(fp):
                v_only.setdefault(d, []).append('no V product by the build\'s rule (B-03 same-state)')
                continue
            if d in prod and not _beam_ok(fp):
                both.setdefault(d, []).append('%s product without a positive beam' % st)
    extra = {st: {d: _num(r.get('stage2_%s_rms_mjy' % st.lower())) for d, r in adm.items()} for st in ('I', 'V')}
    tier = tier_failures(prows, p['sigma'], p['tier_min'], extra=extra)
    # the beam gate, over the nights the gates before it leave (the engine's order: products, data_decisions, night-image, epoch,
    # the 10x sanity, the input-provenance gate; the 3x I sanity and the search tier do not remove a night from that list)
    left = [d for d in sorted(present) if d not in both]
    areas = {}
    for d in left:
        a = readers['beam_area'](cell_dir, member, lst, d)
        if a is None:
            raise RuntimeError(f'wideband member {member}: no readable beam for {d} - the member\'s beam gate cannot run; REFUSING')
        areas[d] = a
    fin = sorted(v for d, v in areas.items() if d not in adm)
    if len(fin) >= p['beam_min']:
        bmed = fin[len(fin) // 2]
        for d, a in areas.items():
            if a > p['beam_ratio'] * bmed:
                both.setdefault(d, []).append('beam gate (area %.2f x the median)' % (a / bmed))
    return dict(present=present, both=both, i_only=i_only, v_only=v_only, tier=tier)


def image_findings(lst, scope, sci_dir, night, ex18=None):
    """{stokes: (stacked subband names {'23MHz', ...} or None, [reasons])} of the wideband's own Phase 2 images of the night (I: the
    alignment image, V: the Taper Robust-0 deep image), with lib/masters/coadd's helpers: a missing image, a set the headers cannot
    show, or a required subband missing (the group minus 18 MHz). An excluded 18 MHz the image holds is NOT a reason (the 18 MHz
    exception, module docstring); ex18 is accepted and ignored."""
    from exopipe.phase3.lib.masters import coadd as C
    req = C._WB_SUBS[scope] - C.SUBBAND_ONLY_EXCL
    out = {}
    for st, path in (('I', C.find_wideband_alignment_image(sci_dir, scope, lst, night) if sci_dir else None),
                     ('V', C.find_stokes_v_deep_image(sci_dir, scope) if sci_dir else None)):
        if path is None:
            out[st] = (None, ['wideband %s image not found' % st])
            continue
        subs = C.wideband_stacked_subbands(path)
        if subs is None:
            out[st] = (None, ['%s cannot show its subbands' % os.path.basename(path)])
            continue
        why = []
        miss = sorted(req - subs, key=int)
        if miss:
            why.append('stacked from an incomplete subband set (missing %s)' % ' '.join(m + 'MHz' for m in miss))
        out[st] = ({s + 'MHz' for s in subs}, why)
    return out


def refusals(lst, scope, nights, records, images):
    """{(night, stokes): [(who, reason), ...]}. records: {member: member_record(...)}; images: {night: image_findings(...)}. A night
    is refused in a Stokes when its image of that Stokes fails, or any member IN THAT IMAGE'S SET other than 18 MHz fails in that
    Stokes (a member the image does not hold is not judged; 18 MHz is the one exception, module docstring)."""
    out = {}
    for night in nights:
        img = images.get(night) or {}
        for st in ('I', 'V'):
            subs, why = img.get(st, (None, ['no image record']))
            for w in why:
                out.setdefault((night, st), []).append((scope, w))
            for m, rec in records.items():
                if (subs is not None and m not in subs) or m == EXCEPTION:
                    continue
                rs = []
                if night not in rec['present']:
                    rs.append('member epoch missing')
                rs += rec['both'].get(night, [])
                rs += rec['i_only' if st == 'I' else 'v_only'].get(night, [])
                t = rec['tier'].get(st, {}).get(night)
                if t:
                    rs.append('search tier (%s stage-1 rms %.1f > %.1f mJy)' % (st, t[0], t[1]))
                for r in rs:
                    out.setdefault((night, st), []).append((m, r))
    return out
