"""Content tests - one registry for every product class (CLAUDE.md 4b line 3; architecture 11.1).

Every product class a step writes has an automatic content test that runs on every file the step writes
(``UnitRecord.add_output``), on every file ``--resume`` would keep (``record.resume_check``), and alone through
``exopipe3 verify <class> <paths>``. A test checks the file's CONTENT (pixels, rows, header values against the data),
never only its presence, name or a marker.

Registering a test (in the module that owns the product class)::

    from exopipe.phase3.core.validate import content_test

    @content_test('pbcorr_sibling')
    def sibling_equals_plain_over_pb(path, **ctx):
        ...
        return passed, value, detail      # bool, number or short text, one line

A test returns ``(passed, value, detail)``; raising is recorded as a failed test with the exception text.

Which module owns which class is listed in ``OWNERS`` so ``verify`` can import only what it needs.
"""
import importlib
import traceback

_TESTS = {}      # class -> [(name, fn)]

# product class -> module that registers its tests (imported on demand by run()).
OWNERS = {
    # masters / dewarp / install (deep, sub-LST, snapshot masters)
    'master': 'exopipe.phase3.steps.masters', 'loo_master': 'exopipe.phase3.steps.masters',
    'vclean_master': 'exopipe.phase3.steps.masters', 'pb_map': 'exopipe.phase3.steps.masters',
    'pbcorr_sibling': 'exopipe.phase3.steps.masters', 'master_catalog': 'exopipe.phase3.steps.masters',
    'membership_record': 'exopipe.phase3.steps.masters', 'weights_table': 'exopipe.phase3.steps.masters',
    'flow': 'exopipe.phase3.steps.flow', 'dewarped_master': 'exopipe.phase3.steps.dewarp',
    'dewarp_screen': 'exopipe.phase3.steps.dewarp', 'install_manifest': 'exopipe.phase3.steps.install',
    # frames / search / catalogues
    'epoch_image': 'exopipe.phase3.steps.frames', 'search_frame': 'exopipe.phase3.steps.frames',
    'noise_map': 'exopipe.phase3.steps.frames', 'detections': 'exopipe.phase3.steps.search',
    'gate_ledger': 'exopipe.phase3.steps.search', 'candidate_catalog': 'exopipe.phase3.steps.catalog',
    'source_catalog': 'exopipe.phase3.steps.catalog', 'forced_table': 'exopipe.phase3.steps.forced',
    'light_curve': 'exopipe.phase3.steps.curves', 'review_page': 'exopipe.phase3.steps.pages',
    'served_queue': 'exopipe.phase3.steps.serve', 'archive_manifest': 'exopipe.phase3.steps.archive',
    'mosaic': 'exopipe.phase3.steps.mosaic', 'photometry_table': 'exopipe.phase3.steps.photometry',
    # generic
    'fits_header': 'exopipe.phase3.core.products', 'csv_table': 'exopipe.phase3.core.validate',
}


def content_test(cls, name=None):
    """Decorator: register ``fn(path, **ctx) -> (passed, value, detail)`` as a content test of ``cls``."""
    def deco(fn):
        _TESTS.setdefault(cls, []).append((name or fn.__name__, fn))
        return fn
    return deco


def classes():
    return sorted(set(OWNERS) | set(_TESTS))


def tests_for(cls):
    if cls not in _TESTS and cls in OWNERS:
        importlib.import_module(OWNERS[cls])
    if cls not in _TESTS:
        raise KeyError(f'no content test registered for product class {cls!r} - every class a step writes must '
                       f'have one (known classes: {", ".join(classes())})')
    return _TESTS[cls]


def run(cls, path, **ctx):
    """Run every test of ``cls`` on ``path``. Returns [{'name', 'passed', 'value', 'detail'}]."""
    out = []
    for name, fn in tests_for(cls):
        try:
            passed, value, detail = fn(path, **ctx)
        except Exception as e:                       # a test that crashes is a failed test, recorded
            passed, value, detail = False, None, f'test raised {type(e).__name__}: {e}'
            detail += ' | ' + traceback.format_exc(limit=2).replace('\n', ' ')[-300:]
        out.append({'name': f'{cls}.{name}', 'passed': bool(passed), 'value': value, 'detail': detail})
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Generic classes
# ---------------------------------------------------------------------------------------------------------------------

@content_test('csv_table')
def csv_has_header_and_parses(path, required_columns=(), min_rows=0, **ctx):
    import csv
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        n = sum(1 for _ in rd)
    missing = [c for c in required_columns if c not in cols]
    if missing:
        return False, n, f'missing columns {missing}'
    if n < min_rows:
        return False, n, f'{n} rows < {min_rows}'
    return True, n, f'{n} rows, {len(cols)} columns'
