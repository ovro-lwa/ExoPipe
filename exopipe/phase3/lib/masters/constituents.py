"""THE MEMBERSHIP CHECK AGAINST THE CONSTITUENT LIST of a deep cell's masters (refactor FOLD_IN rows of 2026-10-26, the deep
chat's findings 'SILENT DROP' and 'MORE SILENT DROPS': installed Robust 0 masters 08h and 09h 18 MHz, and the Robust -0.75
masters 06h 46 MHz, 06h 55 MHz and 04h 59 MHz, each lack one weights.csv member night with no record of why).

WHY: the co-add's final stage (lib/masters/coadd.run_final_lst_freq, = production coadd_full_pipeline) builds the V master
and the three I variants from the cell's member nights (weights.csv in_master 1), and leaves a member night out of one of
them WITHOUT A RECORD when the night has no Phase 2 image of that product (the finder returns None: `if not img: continue`)
or when the night's image cannot be aligned (process_variant_image returns False). Only the per-variant divergence gate
(HISTORY 'VARIANT-GATE refused <date>', NVARREF) and the B-03 same-state rule (<cell>/refusals_input.csv) record what they
leave out. The variant-only rebuild (coadd.rebuild_variants_from_record) does record a missing image ('no <variant> input'
in its VARIANT-GATE HISTORY lines); the final stage does not.

THE RULE (one rule, every master of the cell): a master's constituents (its 'COADD constituent <date>' HISTORY lines, NCOADD)
must equal the cell's RECORDED members for that product:
    survey I Taper R-0.75     the members (weights.csv in_master 1)
    V Taper R0, I variants    the members minus the nights RECORDED as refused for that product:
                              B-03 same-state refusals (refusals_input.csv), the variant gate (HISTORY; I variants only),
                              the nights whose image could not be regridded (process_variant_image returned False: dropped
                              as production drops them, recorded by the build - deep_stages.regrid_watch; meta
                              'regrid_failed' and the content test's context 'build_refused'), and the nights with no Phase
                              2 image of the product - the build records those too (meta 'no_image', one note per product,
                              the content test's context 'no_image'); without that record (a cell built by production, or
                              `exopipe3 verify`, which passes no context) they are found again in the Phase 2 tree by the
                              co-add's finder choice (core.inputs.product_image), as production's NoTaper large-scale
                              reconstruction does (notaper_largescale.members_of)
A night in a master that is not a recorded member, a night in a master although recorded as refused, and a member missing
from a master with no recorded reason (a silent drop) are violations; the master REFUSES (the content test
'master.members_equal_constituents' fails, so the unit that wrote or re-tests it is refused).

KNOWN LIMITS (documented, not handled - the 2026-11-02 masters review):
  - Without the build's record the no-image nights are re-derived from TODAY's Phase 2 tree, not the tree the master was built
    from. A later Phase 2 change - a variant image quarantined or renamed, a Science directory retired, a night re-imaged - can
    then make a correct installed master read as violating the rule (or hide a drop). On 2026-11-02, 0 of the 2,280 installed
    masters violate it and 34 of them pass only through this re-derivation (their left-out nights have no Phase 2 image).
  - Without the build's record a member dropped because its image could not be regridded reads as a silent drop (none among
    the installed masters on 2026-11-02).
  Both go once the build's refusals are kept beside the cell's records and carried by the install (a record file, no product
  byte changed) and read here before any re-derivation.

Read only. Light at import (stdlib + astropy inside the functions).
"""
import csv
import os
import re

# master stem -> the Phase 2 product it is built from (None: the survey master, built from the members' survey images)
PRODUCT_OF = {'I_deep_Taper_R075': None, 'V_deep_Taper_R0': 'V', 'I_deep_Taper_R0': 'Taper_R0',
              'I_deep_NoTaper_R0': 'NoTaper_R0', 'I_deep_NoTaper_R075': 'NoTaper_R075'}
SURVEY = 'I_deep_Taper_R075'
NON_SURVEY = tuple(s for s in PRODUCT_OF if s != SURVEY)
_GATE_RE = re.compile(r'VARIANT-GATE refused (\d{4}-\d\d-\d\d)')
_STEM_RE = re.compile(r'^(?P<stem>[IV]_deep_(?:Taper|NoTaper)_(?:R075|R0))(?:_shflux)?(?:_mdewarp)?_(?P<fl>[A-Za-z0-9]+)\.fits$')


def stem_of(path):
    """(master stem, scope label) of a plain master file name - I_deep_Taper_R0_23MHz.fits, the scaled generation's
    I_deep_Taper_R0_shflux_23MHz.fits, a dewarped generation's ..._mdewarp_23MHz.fits - else (None, None). Clean and
    PB-corrected siblings are not masters of this check."""
    m = _STEM_RE.match(os.path.basename(path))
    if not m or m.group('stem') not in PRODUCT_OF:
        return None, None
    return m.group('stem'), m.group('fl')


def refusal_product(product):
    """The product name of a B-03 refusal row (lib/masters/input_resolver: 'V' or 'I_<variant>')."""
    return 'V' if product == 'V' else f'I_{product}'


def _history(hdr):
    return [str(x) for x in hdr.get('HISTORY', [])]


def constituent_dates(hdr):
    """The 'COADD constituent <date>' HISTORY dates (the co-add writer's own convention; coadd._constituent_dates)."""
    return sorted(x.split()[-1] for x in _history(hdr) if x.startswith('COADD constituent'))


def gate_refused_dates(hdr):
    """Nights the per-variant divergence gate refused, from the master's 'VARIANT-GATE refused <date>' HISTORY lines (the final
    stage and the variant-only rebuild stamp them: coadd._stamp_variant_gate)."""
    return sorted({m.group(1) for x in _history(hdr) for m in [_GATE_RE.search(x)] if m})


def members(cell_dir):
    """{date: sci_dir} of the cell's member nights (coadd_final/weights.csv in_master 1), or None without weights.csv."""
    p = os.path.join(cell_dir, 'coadd_final', 'weights.csv')
    if not os.path.isfile(p):
        return None
    with open(p, newline='') as f:
        return {r['date']: (r.get('sci_dir') or '') for r in csv.DictReader(f) if str(r.get('in_master', '')).strip() == '1'}


def input_refused(cell_dir, product):
    """Nights whose ``product`` image the B-03 same-state rule refused in this cell (<cell>/refusals_input.csv)."""
    from exopipe.phase3.lib.masters import input_resolver
    want = refusal_product(product)
    return sorted({r['date'] for r in input_resolver.read_refusals(cell_dir) if r.get('product') == want})


def _finder_scope(scope):
    """The scope as the co-add's finders take it: a subband's MHz number ('23'), a band name ('Red')."""
    s = str(scope)
    return s[:-3] if s.endswith('MHz') else s


_PINS = {}


def finder_copy_current():
    """[] when core.inputs' copy of the co-add's finders is current - FINDER_PINS equal the finders' source text in
    lib/masters/coadd.py (read as text, not imported) - else what changed. Under production values the B-03 resolver is not
    installed and checks nothing, so the image lookup here checks for itself: a stale copy could name a night 'no image'
    that the co-add found, or the reverse."""
    if 'bad' not in _PINS:
        from exopipe.phase3.core import inputs as I
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coadd.py')) as f:
            _PINS['bad'] = I.pin_mismatches(f.read())
    return list(_PINS['bad'])


def missing_images(member_dirs, scope, product):
    """(no_image, unjudged): the member nights with no Phase 2 image of ``product`` ('V' or an I variant) - the image the co-add's
    finder would choose (core.inputs.product_image, pinned equal to the co-add's finders by FINDER_PINS; a stale copy raises
    RuntimeError) - and the member nights without a Science directory in the record (cannot be judged)."""
    from exopipe.phase3.core import inputs as I
    stale = finder_copy_current()
    if stale:
        raise RuntimeError(f'core.inputs copies the co-add finders {stale} as they were when it was written; '
                           f'lib/masters/coadd.py has changed since - re-port core.inputs (and FINDER_PINS) before images '
                           f'can be looked up (B-03)')
    fq = _finder_scope(scope)
    no, unjudged = [], []
    for d, sci in sorted(member_dirs.items()):
        if not sci or not os.path.isdir(sci):
            unjudged.append(d)
        elif I.product_image(sci, fq, product) is None:
            no.append(d)
    return no, unjudged


def missing_images_cell(cell_dir, scope):
    """{stem: [member nights with no Phase 2 image of its product]} for the V master and the three I variants, and
    {stem: [member nights with no Science directory]} - what the build records (its meta 'no_image') after the co-add."""
    mem = members(cell_dir) or {}
    out, unjudged = {}, {}
    for stem in NON_SURVEY:
        out[stem], unjudged[stem] = missing_images(mem, scope, PRODUCT_OF[stem])
    return out, unjudged


def pbcorr_only(sci_dir, scope, product):
    """MESSAGE ONLY: the PB-corrected files in the night's Phase 2 directory of ``product`` that match its taper and robust
    weighting - the finders never take a PB-corrected image, so a night holding only those has 'no image' (the 08h and 09h
    18 MHz NoTaper Robust 0 nights of the finding). The finders' own name rules decide membership, never this."""
    fq = _finder_scope(scope)
    st = 'V' if product == 'V' else 'I'
    d = os.path.join(sci_dir or '', f'{fq}MHz' if fq.isdigit() else fq, st, 'deep')
    taper, rob = ('Taper', 'R0') if product == 'V' else product.split('_')
    try:
        files = sorted(os.listdir(d))
    except OSError:
        return []
    out = []
    for f in files:
        if 'pbcorr' not in f or not f.endswith('.fits') or ('NoTaper' in f) != (taper == 'NoTaper') or 'Taper' not in f:
            continue
        if (rob == 'R075') != ('Robust-0.75' in f) or 'Robust-0' not in f:
            continue
        out.append(f)
    return out


def no_image_reason(sci_dir, scope, product):
    """'no Phase 2 <product> image', naming the PB-corrected files of the product when only those exist (pbcorr_only)."""
    pc = pbcorr_only(sci_dir, scope, product)
    return f'no Phase 2 {product} image' + (f' (only PB-corrected: {", ".join(pc[:2])})' if pc else '')


def check(master, cell_dir, no_image=None, build_refused=None):
    """THE CHECK of one master file. ``cell_dir`` holds coadd_final/weights.csv and refusals_input.csv (the cell the master was
    built for; its scaled or dewarped generation keeps the cell's records). ``no_image``: the build's RECORDED list of member
    nights with no Phase 2 image of this master's product (None = not recorded: found again in the Phase 2 tree).
    ``build_refused``: {night: reason} the build RECORDED as left out of this master for another reason (a regrid failure:
    deep_stages.regrid_watch). Returns (violations, info): violations is [] when the master's constituents equal its recorded
    members."""
    from astropy.io import fits
    stem, scope = stem_of(master)
    if stem is None:
        return [f'{os.path.basename(master)}: not a master file name (I/V_deep_<taper>_<robust>[_shflux][_mdewarp]_<scope>)'], {}
    mem = members(cell_dir)
    if mem is None:
        return [f'{stem}_{scope}: no {os.path.join(cell_dir, "coadd_final", "weights.csv")} - membership unknown'], {}
    h = fits.getheader(master)
    const = constituent_dates(h)
    product = PRODUCT_OF[stem]
    info = dict(stem=stem, scope=scope, members=len(mem), constituents=len(const), ncoadd=h.get('NCOADD'))
    viol = []
    try:
        if int(h.get('NCOADD')) != len(const):
            viol.append(f'NCOADD {h.get("NCOADD")} != its {len(const)} COADD constituent lines')
    except (TypeError, ValueError):
        viol.append(f'NCOADD {h.get("NCOADD")!r} unreadable')
    refused = {}
    if product is not None:
        for d in input_refused(cell_dir, product):
            refused.setdefault(d, []).append('B-03 same-state refusal (refusals_input.csv)')
        if product != 'V':
            for d in gate_refused_dates(h):
                refused.setdefault(d, []).append('VARIANT-GATE (HISTORY)')
        for d, why in sorted((build_refused or {}).items()):
            refused.setdefault(d, []).append(f'{why} (recorded by the build)')
        if no_image is None:
            todo = {d: s for d, s in mem.items() if d not in refused}
            noimg, unjudged = missing_images(todo, scope, product)
            how = 'found again in the Phase 2 tree (no build record)'
            if unjudged:
                viol.append(f'member night(s) {unjudged} have no Science directory in weights.csv - the {product} image '
                            f'cannot be looked up')
        else:
            noimg, how = sorted(no_image), 'recorded by the build'
        for d in noimg:
            refused.setdefault(d, []).append(f'{no_image_reason(mem.get(d), scope, product)} ({how})')
        info.update(refused={d: '; '.join(w) for d, w in sorted(refused.items())}, no_image=sorted(noimg), no_image_source=how)
    expected = sorted(d for d in mem if d not in refused)
    extra = sorted(set(const) - set(mem))
    if extra:
        viol.append(f'night(s) {extra} in the master are not members of the cell (weights.csv in_master 1)')
    held = sorted(d for d in const if d in refused)
    if held:
        viol.append('night(s) in the master although recorded as refused: '
                    + '; '.join(f'{d} ({refused[d][0]})' for d in held))
    silent = sorted(set(expected) - set(const))
    if silent:
        why = ('no VARIANT-GATE line, no B-03 refusal, no recorded regrid failure, ' if product not in (None, 'V') else
               'no B-03 refusal, no recorded regrid failure, ' if product == 'V' else '')
        why += (('and its Phase 2 image exists' if no_image is None else 'not recorded as without a Phase 2 image')
                if product is not None else 'the survey master holds every member by construction')
        viol.append(f'member night(s) {silent} missing from the master with no recorded reason (a silent drop: {why})')
    info['expected'] = len(expected)
    return [f'{stem}_{scope}: {v}' for v in viol], info


def check_cell(cell_dir, final_dir=None, no_image=None, build_refused=None):
    """check() of every master present in ``final_dir`` (default <cell_dir>/coadd_final). ``no_image``: {stem: [nights]} as
    the build recorded it (None = found again); ``build_refused``: {stem: {night: reason}}. Returns (violations,
    {file name: info})."""
    fd = final_dir or os.path.join(cell_dir, 'coadd_final')
    viol, infos = [], {}
    for f in sorted(os.listdir(fd)) if os.path.isdir(fd) else []:
        stem, _ = stem_of(f)
        if stem is None:
            continue
        v, info = check(os.path.join(fd, f), cell_dir, None if no_image is None else no_image.get(stem, []),
                        (build_refused or {}).get(stem))
        viol += v
        infos[f] = info
    return viol, infos
