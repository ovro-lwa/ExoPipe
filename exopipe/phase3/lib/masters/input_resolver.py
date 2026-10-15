"""B-03 (a), same-state (Gregg 2026-10-13; the same-state reading 2026-10-14): the masters build's Phase 2 inputs through ONE
resolver, core.inputs.

masters_build.INPUT_RESOLVER
  'core.inputs'   (ruled) the co-add's five image finders in the in-place module (lib/masters/coadd.py) are replaced by
                  wrappers over core.inputs' copy of them: the same file is chosen (core.inputs.FINDER_PINS pin the finders'
                  source text; plan/parity_scripts/b03_resolver_equality.py: equal over every deep cell's nights, 2026-10-14)
                  and the co-add's '[failsafe]' lines are printed as before.
  'coadd-finders' (production) the co-add's own finders.
masters_build.RAW_INPUT_FLOW_RULE
  'same-state'        (ruled; needs the resolver) a night's V or I-variant image whose dewarp state differs from the night's
                      survey image - whose flow aligns every product of the night - is refused (core.inputs.state_mismatch):
                      the finder returns None, as for a night without that image, and the refusal is written once to
                      <cell workdir>/refusals_input.csv and printed. The survey image itself is never refused here.
                      A caller that is not a build of the cell (the admission stages, their plans and --resume checks)
                      runs inside read_only(): the refusal is printed, never recorded.
  'survey-image-flow' (production) every product takes the survey image's flow, whatever its state.
"""
import contextlib
import csv
import os

from exopipe.phase3.core import inputs as I
from exopipe.phase3.core.record import Refusal

FINDERS = ('find_subband_alignment_image', 'find_wideband_alignment_image', 'find_subband_variant_image',
           'find_wideband_deep_images', 'find_stokes_v_deep_image')
RESOLVERS = ('core.inputs', 'coadd-finders')
RULES = ('same-state', 'survey-image-flow')
REFUSALS_FILE = 'refusals_input.csv'
REFUSALS_HEADER = ['lst', 'scope', 'date', 'product', 'image', 'reason']
_STATE = {'co': None, 'rule': None, 'record': True}
_WRITTEN = set()          # (refusals file, date, scope, product) written or found in this process
_LOADED = set()           # refusals files whose earlier rows were read
_PRINTED = set()          # (lst, scope, date, product) refusals printed in this process (once each, not at every lookup)
GRID_VALUES = {'PRODUCT_GRID_MISMATCH': ('regrid', 'drop'), 'OFF_GRID_SURVEY': ('regrid', 'refuse')}


def check_grid_values(b):
    """Refuse an unknown masters_build.PRODUCT_GRID_MISMATCH / OFF_GRID_SURVEY value (the co-add tests only for 'regrid':
    any other spelling would silently mean production)."""
    for key, ok in GRID_VALUES.items():
        v = getattr(b, key)
        if v not in ok:
            raise Refusal(f'masters_build.{key} {v!r}: one of {ok}')


def _night_of(sci_dir):
    """(lst, date) of a Phase 2 Science_* directory, <PHASE2_ROOT>/<lst>/<date>/Science_*."""
    night = os.path.dirname(os.path.normpath(sci_dir))
    return os.path.basename(os.path.dirname(night)), os.path.basename(night)


def _scope_label(freq):
    return f'{int(freq)}MHz' if I.is_subband(freq) else str(freq)


def refusals_path(cell_dir):
    return os.path.join(cell_dir, REFUSALS_FILE)


def read_refusals(cell_dir):
    """The rows of a cell's refusals_input.csv ([] when none)."""
    p = refusals_path(cell_dir)
    if not os.path.exists(p):
        return []
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def _record(sci_dir, freq, product, path, why):
    lst, date = _night_of(sci_dir)
    fl = _scope_label(freq)
    if (lst, fl, date, product) not in _PRINTED:
        _PRINTED.add((lst, fl, date, product))
        print(f'    INPUT REFUSED {lst} {fl} {date} {product}: {why} [{os.path.basename(path)}] (B-03 same-state)', flush=True)
    if not _STATE['record']:                  # read_only(): a caller that is not a build of the cell - printed only
        return
    cell_dir = os.path.join(_STATE['co']._workdir_base(), f'{lst}_{fl}')
    if not os.path.isdir(cell_dir):           # not inside a build of this cell (a read-only caller): printed only
        return
    p = refusals_path(cell_dir)
    if p not in _LOADED:
        _LOADED.add(p)
        _WRITTEN.update((p, r['date'], r['scope'], r['product']) for r in read_refusals(cell_dir))
    key = (p, date, fl, product)
    if key in _WRITTEN:
        return
    new = not os.path.exists(p)
    with open(p, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(REFUSALS_HEADER)
        w.writerow([lst, fl, date, product, os.path.basename(path), why])
    _WRITTEN.add(key)


@contextlib.contextmanager
def read_only():
    """A block in which a refused input is printed, never recorded - every caller that is not a build of the cell. D-ADMIT
    seventh review: the admission stages ran the finders with the co-add's work root at their --out; with the cell tree
    there, a refusal went into that tree's build record (refusals_input.csv), the night rule then counted the night as
    one the build saw and dropped it, for good - a dry run, a --resume check or a refused run did the same."""
    old = _STATE['record']
    _STATE['record'] = False
    try:
        yield
    finally:
        _STATE['record'] = old


def _admit(sci_dir, freq, product, path):
    if path is None or _STATE['rule'] != 'same-state':
        return path
    why = I.state_mismatch(sci_dir, freq, path, product)
    if why is None:
        return path
    _record(sci_dir, freq, product, path, why)
    return None


# ---- the wrappers: the co-add finders' signatures, core.inputs' choice ------------------------------------------------

def find_subband_alignment_image(sci_dir, freq_mhz):
    p, raw = I.survey_subband(sci_dir, freq_mhz)
    if p and raw:
        print(f"    [failsafe] no dewarped {freq_mhz}MHz deep in {os.path.basename(sci_dir)}; "
              f"using RAW {os.path.basename(p)} (flow-align to reference will dewarp)")
    return p


def find_wideband_alignment_image(sci_dir, band, lst, date):
    p, raw = I.survey_wideband(sci_dir, band)
    if p and raw:
        print(f"    [failsafe] no dewarped {band} wideband in {os.path.basename(sci_dir)}; "
              f"using RAW {os.path.basename(p)} (flow-align to reference will dewarp)")
    return p


def find_subband_variant_image(sci_dir, freq_mhz, variant):
    return _admit(sci_dir, freq_mhz, f'I_{variant}', I.variant_subband(sci_dir, freq_mhz, variant))


def find_wideband_deep_images(sci_dir, band, stokes, variant):
    return _admit(sci_dir, band, f'{stokes}_{variant}', I.variant_wideband(sci_dir, band, stokes, variant))


def find_stokes_v_deep_image(sci_dir, freq_str):
    return _admit(sci_dir, freq_str, 'V', I.stokes_v_image(sci_dir, freq_str))


_WRAPPERS = {n: globals()[n] for n in FINDERS}


def install(co, resolver, rule):
    """Point the co-add module ``co``'s finders at core.inputs (resolver 'core.inputs') or back at its own ('coadd-finders'),
    and set the raw-input rule. Refuses an unknown value, the same-state rule without the resolver, and a finder whose
    source text changed since core.inputs copied it. Returns the resolver in force."""
    if resolver not in RESOLVERS:
        raise Refusal(f'masters_build.INPUT_RESOLVER {resolver!r}: one of {RESOLVERS}')
    if rule not in RULES:
        raise Refusal(f'masters_build.RAW_INPUT_FLOW_RULE {rule!r}: one of {RULES}')
    if rule == 'same-state' and resolver != 'core.inputs':
        raise Refusal("masters_build.RAW_INPUT_FLOW_RULE 'same-state' needs masters_build.INPUT_RESOLVER 'core.inputs' (the rule "
                      "is applied by the resolver)")
    orig = co.__dict__.setdefault('_B03_FINDERS', {n: getattr(co, n) for n in FINDERS})   # the co-add's own, first seen
    _STATE['co'] = co
    if resolver == 'coadd-finders':
        for n in FINDERS:
            setattr(co, n, orig[n])
        _STATE['rule'] = None
        return resolver
    bad = I.pin_mismatches(open(co.__file__).read())
    if sorted(co.SUBBANDS) != sorted(int(str(x).replace('MHz', '')) for x in __import__('exopipe.phase3.config', fromlist=['CFG']).CFG.instrument.ALL_SUBBANDS):
        bad.append('SUBBANDS (the co-add\'s subband list differs from the config\'s, which core.inputs.is_subband reads)')
    if bad:
        raise Refusal(f'core.inputs copies the co-add finders {bad} as they were when it was written; their source in '
                      f'{co.__file__} has changed since - re-port core.inputs (and its FINDER_PINS) before using it (B-03)')
    for n in FINDERS:
        setattr(co, n, _WRAPPERS[n])
    _STATE['rule'] = rule
    return resolver


def preview(lst, scope, dates):
    """Dry run, read only: {product: [night]} of the products the same-state rule would refuse among (date, sci_dir, img)."""
    fq = scope.replace('MHz', '') if scope.endswith('MHz') else scope
    out = {}
    for d, sci, _img in dates:
        for prod in ('V',) + tuple(v for v in I.VARIANTS if v != 'Taper_R075'):
            p = I.product_image(sci, fq, prod)
            if I.state_mismatch(sci, fq, p, prod):
                out.setdefault('V' if prod == 'V' else f'I_{prod}', []).append(d)
    return out


def off_grid_nights(entries, cutover_date):
    """[(night, shape, modal shape)] of the post-cutover nights among (date, sci_dir, survey image) whose survey image is off
    the modal post-cutover grid - the test production's coadd._grid_refused_dates applies (shape; ties broken the same
    way). An unreadable header is left to the co-add's own grid-header site."""
    from astropy.io import fits
    shapes = {}
    for d, _sd, img in entries:
        if d < cutover_date:
            continue
        try:
            h = fits.getheader(img)
            shapes[d] = (int(h['NAXIS2']), int(h['NAXIS1']))
        except Exception:
            shapes[d] = None
    vals = [x for x in shapes.values() if x]
    modal = max(set(vals), key=vals.count) if vals else None
    return sorted((d, x, modal) for d, x in shapes.items() if x and modal and x != modal)


def wrong_grid_refusal(lst, scope, admitted_entries, cutover_date):
    """B-03 (a), masters_build.OFF_GRID_SURVEY 'regrid': the refusal text when an ADMITTED post-cutover night of the cell has
    its survey image off the cell's modal grid, else None. Under 'regrid' such a night would be regridded into the master
    with no record; the ruling keeps the wrong-grid nights out by data_decisions.csv rows (Gregg's), and the data hold more
    of them than the four ruled 'leave excluded' (R-WRONG-GRID-NIGHTS-LEAVE-EXCLUDED; refactor review 2026-10-15: 02h 50MHz
    2025-12-26, 11h Full 2025-03-09 / 04-07, 22h Full 2025-08-08), so the cell refuses until the night is excluded.
    None under 'refuse' (production refuses such a night by grid)."""
    from exopipe.phase3.config import CFG
    if CFG.masters_build.OFF_GRID_SURVEY != 'regrid':
        return None
    off = off_grid_nights(admitted_entries, cutover_date)
    if not off:
        return None
    return (f'B-03 (a): {lst} {scope} admitted night(s) ' + ', '.join(f'{d} ({x[0]}x{x[1]})' for d, x, _m in off)
            + f' have a survey image off the cell\'s modal grid {off[0][2][0]}x{off[0][2][1]} - Phase 2 anomalies of the kind '
            f'ruled "leave excluded"; with masters_build.OFF_GRID_SURVEY regrid they would be regridded into the master with no '
            f'record, so the cell refuses until data_decisions.csv excludes them (Gregg\'s rows; DECISIONS_PENDING.md)')


PRODUCT_MASTERS = {'V': 'V_deep_Taper_R0', 'I_Taper_R0': 'I_deep_Taper_R0', 'I_NoTaper_R0': 'I_deep_NoTaper_R0',
                   'I_NoTaper_R075': 'I_deep_NoTaper_R075'}


def masters_below_minimum(cell_dir, scope, refused, min_coadd, min_variant):
    """After the co-add: the products whose master was not built although the same-state rule refused some of their nights
    (``refused``: product -> [night]) - the nights left fell below the minimum (V: MIN_DATES_FOR_COADD; an I variant:
    MIN_DATES_FOR_VARIANT). Returns one sentence per such product ([] when every refused product still has its master)."""
    out = []
    for prod, nights in sorted(refused.items()):
        stem = PRODUCT_MASTERS.get(prod)
        if stem and not os.path.exists(os.path.join(cell_dir, 'coadd_final', f'{stem}_{scope}.fits')):
            need = min_coadd if prod == 'V' else min_variant
            out.append(f'{stem}_{scope} not built: the {prod} image of {len(nights)} night(s) {sorted(nights)} was refused and '
                       f'fewer than {need} master nights are left')
    return out


def nights_left(lst, scope, dates):
    """Dry run, read only: {product: nights among (date, sci_dir, img) whose image the same-state rule keeps} for V and the
    three I variants (an upper bound on the master's nights: the stage-2 and epoch checks come after)."""
    fq = scope.replace('MHz', '') if scope.endswith('MHz') else scope
    out = {}
    for prod in ('V',) + tuple(v for v in I.VARIANTS if v != 'Taper_R075'):
        n = 0
        for _d, sci, _img in dates:
            p = I.product_image(sci, fq, prod)
            n += p is not None and not I.state_mismatch(sci, fq, p, prod)
        out['V' if prod == 'V' else f'I_{prod}'] = n
    return out

