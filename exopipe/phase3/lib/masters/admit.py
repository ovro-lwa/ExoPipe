"""D-ADMIT (a): admitting a night newer than its cell's installed master (ruling R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-
B03-B05-B30-B32: "admit stage into the INSTALLED master's frame and scale, gates against the build's recorded statistics,
admission record beside weights.csv, E15 accepts admitted nights, membership.verify reports 'admitted, rebuild pending'; the
master rebuild is NOT triggered automatically"). Design: plan/wave2/DADMIT_DESIGN.md (slices 2-4).

This module: which nights an installed cell admits (nights_to_admit) and the reference they are admitted against
(installed_reference: the INSTALLED master divided by its pixel factor - native flux units, in the installed frame, so in a
dewarped cell the night's flow aligns it to the dewarped master directly and no screen is applied afterwards). Everything
it reads is the cell tree it is given; nothing here writes outside <cell>/admit/.
"""
import csv
import os

from exopipe.phase3.core.record import Refusal

ADMIT_DIR = 'admit'                      # <cell>/admit/: reference copies, scaled flow inputs, flows, native products
ADMISSIONS = 'admissions.csv'            # <cell>/coadd_final/admissions.csv: one row per admitted (or refused) night
BUILD_NIGHTS = 'residuals_provisional.csv'   # every night the installed build measured (member or refused)
# every record of the installed cell that names a night the build saw - measured, refused before measuring, or refused as an
# input (2026-10-15 second review: of the 37 nights in no residuals_provisional.csv, 29 are in night_image_gate.csv, 3 in
# refusals_grid.csv) - (path relative to the cell, date column)
BUILD_RECORDS = ((BUILD_NIGHTS, 'date'), ('coadd_final/weights.csv', 'date'), ('coadd_final/night_image_gate.csv', 'date'),
                 ('coadd_final/epoch_gate.csv', 'date'), ('sanity_excluded.csv', None), ('refusals_grid.csv', None),
                 ('refusals_input.csv', 'date'))


def _rows(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def build_record(cell_dir):
    """(path, mtime, {nights}) of the installed build's night record - the nights it measured; its mtime is when the build
    fixed its night set. Refuses without it (a cell with no build record has no population to admit into)."""
    p = os.path.join(cell_dir, BUILD_NIGHTS)
    if not os.path.exists(p):
        raise Refusal(f'{p} missing: the installed build\'s night record (no population to admit a night into)')
    return p, os.path.getmtime(p), {r['date'] for r in _rows(p)}


def build_time(cell_dir):
    """(path, time) when the installed build fixed its night set: the write time of its final stage's record
    coadd_final/weights.csv - the final stage discovers the cell's nights again and records every night it measures. Not
    residuals_provisional.csv, which later passes rewrite (in 315 of 456 cells it is 32-60 h newer than weights.csv; fourth
    review). Refuses without it."""
    p = os.path.join(cell_dir, 'coadd_final', 'weights.csv')
    if not os.path.exists(p):
        raise Refusal(f'{p} missing: the installed build\'s final record (no time the build fixed its nights)')
    return p, os.path.getmtime(p)


def seen_nights(cell_dir):
    """{date: [records that name it]} over BUILD_RECORDS: the nights the installed build recorded. A record with a header
    is read by its date column - a header without that column refuses (a record this rule cannot read is never skipped);
    one without a header (refusals_grid.csv, sanity_excluded.csv: lst,scope,date,...) by its third column."""
    import re
    out = {}
    for rel, col in BUILD_RECORDS:
        p = os.path.join(cell_dir, rel)
        if not os.path.exists(p):
            continue
        with open(p, newline='', encoding='utf-8-sig') as f:
            rows = list(csv.reader(f))
        if col:
            if not rows or col not in rows[0]:
                raise Refusal(f'{p}: no {col!r} column in its first row ({(rows or [[]])[0][:6]}) - the build record cannot '
                              f'be read, so no night of the cell can be shown new')
            i, body = rows[0].index(col), rows[1:]
        else:
            i, body = 2, rows
        for r in body:
            if len(r) > i and re.match(r'^\d{4}-\d\d-\d\d$', r[i].strip()):
                names = out.setdefault(r[i].strip(), [])
                if rel not in names:
                    names.append(rel)
    return out


SCIENCE_DIR = r'^Science_(\d{8})_(\d{6})$'       # Phase 2's Science dir: the UTC time it was made, in its name


def science_dirs_before(sci_dir, t, lst, freq_str, date, co):
    """The night's Phase-2 Science dirs (the siblings of ``sci_dir``) named before ``t`` that hold the cell's survey image by
    the co-add's own finder: where a build that fixed its night set at ``t`` could have seen the night. A Science dir whose
    name carries no time counts as before (it cannot be dated). Re-dewarped or restored images keep their dir's name, so
    file times play no part (2026-10-15 review: 6,463 of 11,191 survey images were rewritten more than a day after their
    Science dir was made; every Science dir an installed build used is named before that build's night record)."""
    import calendar
    import contextlib
    import io
    import re
    import time as _time
    night = os.path.dirname(sci_dir)
    out = []
    for name in sorted(os.listdir(night)) if os.path.isdir(night) else []:
        if not name.startswith('Science_'):
            continue
        m = re.match(SCIENCE_DIR, name)
        when = calendar.timegm(_time.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')) if m else None
        if when is not None and when >= t:
            continue
        sd = os.path.join(night, name)
        with contextlib.redirect_stdout(io.StringIO()):          # the finders' failsafe notes
            img = (co.find_subband_alignment_image(sd, int(freq_str)) if co.is_subband(freq_str)
                   else co.find_wideband_alignment_image(sd, freq_str, lst, date))
        if img:
            out.append(name)
    return out


def admitted_nights(cell_dir):
    """Nights coadd_final/admissions.csv holds (admitted or refused there): never admitted twice."""
    p = os.path.join(cell_dir, 'coadd_final', ADMISSIONS)
    return {r['date'] for r in _rows(p)} if os.path.exists(p) else set()


def nights_to_admit(lst, scope, cell_dir, entries, ex, co, nights=None):
    """The nights of an installed cell to admit, and why each other night is not admitted.

    entries: the cell's Phase-2 nights as the co-add discovers them, [(date, science dir, survey image)]; ex: the exclusions
    (core.exclusions); co: the co-add module (its finders). A night is admitted when data_decisions.csv does not exclude it,
    admissions.csv does not hold it, no record of the installed build names it (seen_nights: measured, refused before
    measuring, refused as an input) and Phase 2 first had it after the build fixed its night set (build_time: the
    final stage's weights.csv): no Science dir of the night named before that time holds the cell's survey image
    (science_dirs_before) - so a night the build saw stays out however it was refused, recorded or not, and whatever was
    rewritten since (a rebuild decides). nights: when given (dates; the stage converts --epochs; [] = none), only those; a
    named night that does not qualify is reported with its reason.
    Returns ([(date, sci_dir, img)], {date: reason})."""
    import time as _time
    from exopipe.phase3.lib.masters import deep_stages as DS
    build_record(cell_dir)                                   # the night record must exist
    t_path, t_build = build_time(cell_dir)
    seen = seen_nights(cell_dir)
    done = admitted_nights(cell_dir)
    want = set(nights) if nights is not None else None
    freq_str = DS._freq(scope)
    take, why = [], {}
    for d, sd, img in entries:
        if want is not None and d not in want:
            continue
        if ex.is_excluded(lst, scope, d):
            why[d] = 'excluded in data_decisions.csv'
        elif d in done:
            why[d] = f'already in coadd_final/{ADMISSIONS}'
        elif d in seen:
            why[d] = (f'the installed build recorded it ({", ".join(seen[d])}): a member or a night it refused - a rebuild '
                      f'decides')
        elif not img or not os.path.exists(img):
            why[d] = 'no survey image'
        else:
            before = science_dirs_before(sd, t_build, lst, freq_str, d, co)
            if before:
                why[d] = (f'Phase 2 had it before the build fixed its night set (coadd_final/weights.csv written '
                          f'{_time.strftime("%Y-%m-%dT%H:%MZ", _time.gmtime(t_build))}; {", ".join(before)} hold its survey '
                          f'image): the build saw this night - a rebuild decides')
            else:
                take.append((d, sd, img))
    if want is not None:
        for d in sorted(want - {x[0] for x in entries}):
            why[d] = 'no Phase-2 night of this cell'
    return take, why


def installed_reference(cell, cell_dir, read_data=True):
    """The reference a night is admitted against, per Stokes (I survey master, V master): the INSTALLED master divided by
    its pixel factor (SHSCALE x FLUXCORR when baked) - the installed master's frame and, in native units, its scale. Not
    the pre-dewarp master: of the 166 dewarped cells' masters only the Stokes I input of the dewarp survives (node-local
    copies, no V), so admission aligns the night to the installed frame itself and bakes at the end. Refuses when a master
    is missing or its pixel factor is outside masters_stagein's range. Each Stokes keeps its own constituents (the V
    master may hold fewer nights: per-Stokes sets, R-LOO-PER-STOKES-RATIFIED - 6 installed cells, 2026-10-15). Returns one
    dict per Stokes: stokes, var, path, factor, constdig, mdwarp, data (float32), header (SH*/FLUX* cards stripped as
    stage-in strips them), history."""
    from astropy.io import fits
    import numpy as np
    from exopipe.phase3.lib.masters import deep_stagein as DS, flux_correction as FC
    sc = cell.split('_', 1)[1]
    out = []
    for st, var, _rob in DS.REF_VARS:
        p = os.path.join(cell_dir, 'coadd_final', f'{var}_{sc}.fits')
        if not os.path.exists(p):
            raise Refusal(f'{cell}: installed master {p} missing - nothing to admit a night into')
        h = fits.getheader(p)
        f = FC.pixel_factor(h)
        if f is None:                       # stage-in refuses an unscaled current master too (deep_stagein, pixel factor)
            raise Refusal(f'{cell}: installed master {p} carries no flux scale (SHSCALE) - not an installed generation')
        if not DS._factor_ok(f):
            raise Refusal(f'{cell}: pixel factor {f} of {p} outside masters_stagein.REF_PIXEL_FACTOR_RANGE')
        e = dict(stokes=st, var=var, path=p, factor=f, constdig=h.get('CONSTDIG'), mdwarp=bool(h.get('MDWARP')),
                 data=None, header=None,
                 history=f'ADMIT reference: installed {var} {p} / pixel factor {f:.6f} (CONSTDIG {h.get("CONSTDIG")}; '
                         f'MDWARP {bool(h.get("MDWARP"))})')
        if read_data:
            with fits.open(p) as hd:
                e['data'] = (hd[0].data.astype(np.float64) / f).astype(np.float32)
                hh = hd[0].header.copy()
            DS._strip(hh)
            e['header'] = hh
        out.append(e)
    return out


def header_differences(a, b):
    """The keys whose value differs between headers ``a`` and ``b``: every card of either but HISTORY, COMMENT, blank
    cards (the reference copy carries its own HISTORY line) and the structure cards the FITS writer sets (SIMPLE, BITPIX,
    NAXIS*, EXTEND) - so a header-only change of the master (WCS, beam) is seen (fifth review)."""
    skip = ('HISTORY', 'COMMENT', '', 'SIMPLE', 'BITPIX', 'EXTEND')
    keys = {k for k in list(a.keys()) + list(b.keys()) if k not in skip and not k.startswith('NAXIS')}
    return sorted(k for k in keys if a.get(k) != b.get(k))


def write_reference(cell_dir, scope, ents):
    """The reference copies under <cell>/admit/ref/<var>_<scope>.fits (the canonical names), each with its resolution in
    HISTORY, written to a temporary name and renamed (never a partial copy). Returns {stokes: path}; refuses to overwrite
    an existing copy whose data or any header card differ (header_differences: CONSTDIG, MDWARP, the WCS, the beam ...)."""
    from astropy.io import fits
    import numpy as np
    d = os.path.join(cell_dir, ADMIT_DIR, 'ref')
    os.makedirs(d, exist_ok=True)
    out = {}
    for e in ents:
        if e['data'] is None:
            continue
        p = os.path.join(d, f'{e["var"]}_{scope}.fits')
        hh = e['header'].copy()
        hh['HISTORY'] = e['history']
        if os.path.exists(p):
            with fits.open(p) as h:
                old, oh = h[0].data, h[0].header
                same = (old.shape == e['data'].shape and np.array_equal(old, e['data'], equal_nan=True)
                        and not header_differences(oh, hh))
            if not same:
                raise Refusal(f'{p} exists with other content: the installed master changed since the admission reference '
                              f'was written - admit into the new master from a fresh <cell>/{ADMIT_DIR}/')
        else:
            tmp = f'{p}.part{os.getpid()}'
            fits.writeto(tmp, e['data'], hh)
            os.replace(tmp, p)
        out[e['stokes']] = p
    return out


# ── admitted nights in every check that reads a cell (slices 5 and 6; plan/wave2/DADMIT_DESIGN.md "Slices 5 and 6 as they will
# be written"). One switch for every step, masters_admit.ADMITTED_NIGHTS: 'admitted' (the ruling) or 'absent' (production: an
# admitted night is what production calls a non-excluded night absent from the co-add record).
ADMITTED_NIGHTS = ('admitted', 'absent')


def admitted_policy(value=None):
    """masters_admit.ADMITTED_NIGHTS, or ``value`` checked; an unknown value refuses (a typo never selects either reading)."""
    if value is None:
        from exopipe.phase3.config import CFG
        value = CFG.masters_admit.ADMITTED_NIGHTS
    if value not in ADMITTED_NIGHTS:
        raise Refusal(f'masters_admit.ADMITTED_NIGHTS = {value!r} is not one of {ADMITTED_NIGHTS}')
    return value


def master_path(cell_dir):
    sc = os.path.basename(cell_dir.rstrip('/')).split('_', 1)[1]
    return os.path.join(cell_dir, 'coadd_final', f'I_deep_Taper_R075_{sc}.fits')


def master_identity(cell_dir):
    """(CONSTDIG, md5 of the float32 data) of the cell's I master, read as installed_reference and the epoch gate read them
    (('', '') without a master). CONSTDIG is the digest of the constituent dates only - a re-dewarped or re-baked master keeps
    it - so an admission records both."""
    from astropy.io import fits
    from exopipe.phase3.lib.masters import epoch_gate as EG
    p = master_path(cell_dir)
    if not os.path.exists(p):
        return '', ''
    return str(fits.getheader(p).get('CONSTDIG') or ''), EG.master_md5(p)


def admitted_rows(cell_dir):
    """{date: admissions.csv row} of the nights admitted into the cell's installed master: verdict 'admitted', made into this
    master (reference_constdig and reference_md5 its CONSTDIG and data md5 - an admission into another master counts for
    nothing). {} without the record."""
    p = os.path.join(cell_dir, 'coadd_final', ADMISSIONS)
    if not os.path.exists(p):
        return {}
    rows = [r for r in _rows(p) if r.get('verdict') == 'admitted']
    if not rows:
        return {}
    cdg, md5 = master_identity(cell_dir)
    return {r['date']: r for r in rows
            if cdg and r.get('reference_constdig') == cdg and r.get('reference_md5') == md5}


def absent_violation(lst, band, date):
    """membership.verify's sentence for a non-excluded night with a Stokes I residual in the cell but no weights.csv row -
    production's text (membership.py, the 2026-10-11 absent-night rule), pinned by tests/test_masters_admit7.py against
    membership.verify's own output: if production rewords it, the violation stays a violation."""
    return (f'{lst} {band} {date}: has a Stokes I residual in the cell but is ABSENT from the co-add record (weights.csv) '
            f'- never tested against the membership rulings; the master must be rebuilt')


def membership(cell_dir, violations, policy=None):
    """(violations, reports) of a cell from membership.verify's ``violations`` (this process's or a child's): under
    'admitted' each admitted night's absent-night violation leaves the violations and is reported as '<lst> <band> <date>:
    admitted <utc> into the installed master, rebuild pending'. Every other violation stays; under 'absent' nothing
    changes."""
    viol = list(violations)
    if admitted_policy(policy) != 'admitted' or not viol:
        return viol, []
    rows = admitted_rows(cell_dir)
    if not rows:
        return viol, []
    from exopipe.phase3.lib.masters.membership import cell_band
    lst, band = cell_band(cell_dir)
    which = {absent_violation(lst, band, d): d for d in rows}
    keep, reports = [], []
    for v in viol:
        d = which.get(v)
        if d is None:
            keep.append(v)
        else:
            reports.append(f'{lst} {band} {d}: admitted {rows[d].get("utc", "")} into the installed master, rebuild pending')
    return keep, reports


def verify(cell_dir, exclusions=None, policy=None, mb=None):
    """membership.verify of the cell read with its admitted nights - (violations, reports), see membership(). The one entry
    point of every branch step that verifies a cell (``mb``: the membership module a caller loaded, default the branch's);
    membership.py stays production's text (S-G6)."""
    if mb is None:
        from exopipe.phase3.lib.masters import membership as mb
    return membership(cell_dir, mb.verify(cell_dir, exclusions), policy)
