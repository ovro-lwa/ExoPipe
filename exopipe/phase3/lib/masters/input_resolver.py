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
  'survey-image-flow' (production) every product takes the survey image's flow, whatever its state.
"""
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
_STATE = {'co': None, 'rule': None}
_WRITTEN = set()          # (refusals file, date, scope, product) written or found in this process
_LOADED = set()           # refusals files whose earlier rows were read


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
    print(f'    INPUT REFUSED {lst} {fl} {date} {product}: {why} [{os.path.basename(path)}] (B-03 same-state)', flush=True)
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
