#!/usr/bin/env python3
"""STANDING TEST: no memory maps (Gregg 2026-11-17, R-GREGG-20261117-NO-MMAP-OF-LUSTRE-FILES-READ-OR-WRITE: "never memory-map a
Lustre file, for reading OR writing"; the 11-09 and 11-16 lwacalim07 Lustre client crashes were on mapped pages).

1. Importing exopipe turns astropy's FITS memory map off (fits.conf.use_memmap False), so default fits.open/getdata reads.
2. No source file under exopipe/, tools/ or tests/ asks for a map: memmap=True, mmap_mode=, np.memmap / numpy.memmap,
   mmap.mmap(, or use_memmap = True. Read slices with ordinary reads (ImageHDU.section, f.seek + f.read) instead.

Run standalone: python tests/test_no_mmap.py    Under pytest: pytest tests/test_no_mmap.py
"""
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


def test_import_turns_memmap_off():
    import exopipe  # noqa: F401
    from astropy.io import fits
    assert fits.conf.use_memmap is False


def test_no_memory_map_requested_in_source():
    bad = offending_lines()
    assert not bad, 'memory maps requested (refused by R-GREGG-20261117-NO-MMAP-OF-LUSTRE-FILES-READ-OR-WRITE):\n' + '\n'.join(bad)


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
    print('PASS: no memory maps')
