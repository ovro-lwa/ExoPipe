#!/usr/bin/env python3
"""STANDING TEST: no memory maps (Gregg 2026-11-17, R-GREGG-20261117-NO-MMAP-OF-LUSTRE-FILES-READ-OR-WRITE: "never memory-map a
Lustre file, for reading OR writing"; the 11-09 and 11-16 lwacalim07 Lustre client crashes were on mapped pages).

1. Importing exopipe turns astropy's FITS memory map off (fits.conf.use_memmap False), so default fits.open/getdata reads.
2. No source file under exopipe/, tools/ or tests/ asks for a map: memmap=True, mmap_mode=, np.memmap / numpy.memmap,
   mmap.mmap(, or use_memmap = True. Read slices with ordinary reads (ImageHDU.section, f.seek + f.read) instead.
3. An entry point that calls astropy.io.fits (any file under tools/, or a module under exopipe/ with an
   `if __name__ == '__main__':` block, since it can be run by path, often in another conda env, where exopipe/__init__.py
   never runs) imports exopipe or sets fits.conf.use_memmap = False at module level (code review of PR 81). A library
   module imported as exopipe.<...> is covered by the package's __init__.

Run standalone: python tests/test_no_mmap.py    Under pytest: pytest tests/test_no_mmap.py
"""
import ast
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
SCAN = ('exopipe', 'tools', 'tests')
PATTERN = re.compile(r"memmap\s*=\s*True|mmap_mode\s*=|\bnp\.memmap\b|\bnumpy\.memmap\b|\bmmap\.mmap\s*\(|use_memmap\s*=\s*True")


def offending_lines():
    out = []
    for top in SCAN:
        for d, _sub, files in os.walk(os.path.join(REPO, top)):
            for fn in files:
                p = os.path.join(d, fn)
                if not fn.endswith('.py') or fn.endswith('test_no_mmap.py'):     # this file names the forms it refuses
                    continue
                with open(p, encoding='utf-8', errors='replace') as f:
                    for i, line in enumerate(f, 1):
                        code = line.split('#', 1)[0]
                        if PATTERN.search(code):
                            out.append(f'{os.path.relpath(p, REPO)}:{i}: {line.strip()}')
    return out


FITS_CALLS = {'open', 'getdata', 'getheader', 'getval', 'setval', 'update', 'append', 'writeto'}


def _top_level(body):
    """Statements run at import: the module body and the bodies of top-level try/if blocks."""
    for n in body:
        yield n
        for attr in ('body', 'orelse', 'finalbody'):
            if isinstance(n, (ast.Try, ast.If)):
                yield from _top_level(getattr(n, attr, []))
        if isinstance(n, ast.Try):
            for hnd in n.handlers:
                yield from _top_level(hnd.body)


def _is_main_guard(n):
    return (isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
            and n.test.left.id == '__name__')


def entry_points_without_switch():
    """Entry points (tools/*, or exopipe/ modules with a __main__ block) that call fits.<open|getdata|...> but neither
    import exopipe nor set use_memmap = False at module level."""
    out = []
    for top in ('exopipe', 'tools'):
        for d, _sub, files in os.walk(os.path.join(REPO, top)):
            for fn in files:
                if not fn.endswith('.py'):
                    continue
                p = os.path.join(d, fn)
                with open(p, encoding='utf-8', errors='replace') as f:
                    t = ast.parse(f.read(), p)
                calls = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in FITS_CALLS
                            and isinstance(n.func.value, ast.Name) and n.func.value.id in ('fits', 'pyfits')
                            for n in ast.walk(t))
                if not calls or (top == 'exopipe' and not any(_is_main_guard(n) for n in t.body)):
                    continue
                ok = False
                for n in _top_level(t.body):
                    if isinstance(n, ast.Import) and any(a.name.split('.')[0] == 'exopipe' for a in n.names):
                        ok = True
                    elif isinstance(n, ast.ImportFrom) and (n.module or '').split('.')[0] == 'exopipe' and not n.level:
                        ok = True
                    elif (isinstance(n, ast.Assign) and any(isinstance(x, ast.Attribute) and x.attr == 'use_memmap'
                                                            for x in n.targets)
                          and isinstance(n.value, ast.Constant) and n.value.value is False):
                        ok = True
                if not ok:
                    out.append(os.path.relpath(p, REPO))
    return sorted(out)


def test_import_turns_memmap_off():
    import exopipe  # noqa: F401
    from astropy.io import fits
    assert fits.conf.use_memmap is False


def test_no_memory_map_requested_in_source():
    bad = offending_lines()
    assert not bad, 'memory maps requested (refused by R-GREGG-20261117-NO-MMAP-OF-LUSTRE-FILES-READ-OR-WRITE):\n' + '\n'.join(bad)


def test_every_fits_entry_point_turns_memmap_off():
    bad = entry_points_without_switch()
    assert not bad, ('modules calling astropy.io.fits without importing exopipe or setting fits.conf.use_memmap = False at '
                     'module level (R-GREGG-20261117-NO-MMAP-OF-LUSTRE-FILES-READ-OR-WRITE):\n' + '\n'.join(bad))


def test_pattern_catches_the_forms():
    for s in ("fits.open(p, memmap=True)", "np.load(p, mmap_mode='r')", "np.memmap(p)", "mmap.mmap(fd, 0)",
              "fits.conf.use_memmap = True"):
        assert PATTERN.search(s), s
    for s in ("fits.open(p, memmap=False)", "np.load(p)", "fits.conf.use_memmap = False"):
        assert not PATTERN.search(s), s


if __name__ == '__main__':
    test_pattern_catches_the_forms()
    test_import_turns_memmap_off()
    test_no_memory_map_requested_in_source()
    test_every_fits_entry_point_turns_memmap_off()
    print('PASS: no memory maps')
