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

Tests live in lib/<step>/tests*.py and are discovered on first use; ``OWNERS`` lists the few registered elsewhere.
"""
import importlib
import traceback

_TESTS = {}      # class -> [(name, fn)]

# product class -> module that registers its tests, only for classes registered OUTSIDE lib/<step>/tests*.py (those
# are found by discovery, _discover below).
OWNERS = {
    'fits_header': 'exopipe.phase3.core.products', 'csv_table': 'exopipe.phase3.core.validate',
}


def content_test(cls, name=None):
    """Decorator: register ``fn(path, **ctx) -> (passed, value, detail)`` as a content test of ``cls``."""
    def deco(fn):
        _TESTS.setdefault(cls, []).append((name or fn.__name__, fn))
        return fn
    return deco


def classes():
    _discover()
    return sorted(set(OWNERS) | set(_TESTS))


_DISCOVERED = False
_DISCOVERY_ERRORS = {}   # module -> error text (a broken tests module must not hide every other class's tests)


def _discover():
    """Import every lib/<step>/tests*.py module (where steps register their content tests)."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    import pkgutil
    import exopipe.phase3.lib as lib
    for sub in pkgutil.iter_modules(lib.__path__):
        if not sub.ispkg:
            continue
        pkg = importlib.import_module(f'exopipe.phase3.lib.{sub.name}')
        for m in pkgutil.iter_modules(pkg.__path__):
            if m.name == 'tests' or m.name.startswith('tests_'):
                name = f'exopipe.phase3.lib.{sub.name}.{m.name}'
                try:
                    importlib.import_module(name)
                except Exception as e:
                    _DISCOVERY_ERRORS[name] = f'{type(e).__name__}: {e}'


def tests_for(cls):
    # Always discover first: a class may register tests in several modules (e.g. a record class shared by two steps),
    # and stopping at the first module that registers it would silently drop the others.
    _discover()
    if cls not in _TESTS and cls in OWNERS:
        try:
            importlib.import_module(OWNERS[cls])
        except ImportError:
            pass
    if cls not in _TESTS:
        broken = '; '.join(f'{k} ({v})' for k, v in _DISCOVERY_ERRORS.items())
        raise KeyError(f'no content test registered for product class {cls!r} - every class a step writes must '
                       f'have one (known classes: {", ".join(classes())})' + (f'; test modules that failed to import: '
                                                                              f'{broken}' if broken else ''))
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
