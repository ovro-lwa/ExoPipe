"""OVRO-LWA ExoPipe v2 — package root (PHASE3_ARCHITECTURE.md §12.0).

Layout: common/ (config, products, astrometry, exclusions, validate,
residuals, qa substrate — §2), phase1/, phase2/ (relocated, light cleanup),
phase3/ (six capabilities + followup/ subpackage — §1, §9).
The git tree here IS the GitHub tree; operational files live in
/lustre/gh/main/exopipe_ops/ and structurally cannot leak (§12.0).
"""

# Never memory-map a Lustre file, for reading or writing (Gregg 2026-11-17, R-GREGG-20261117-NO-MMAP-OF-LUSTRE-FILES-READ-OR-
# WRITE, after the 11-09 and 11-16 lwacalim07 Lustre client crashes on mapped pages): astropy's default for fits.open/getdata
# is a memory map, so it is turned off for every process that imports exopipe. tests/test_no_mmap.py refuses explicit maps.
try:
    from astropy.io import fits as _fits
    _fits.conf.use_memmap = False
    del _fits
except ImportError:          # environments without astropy (no FITS reads there)
    pass

# Safe copies: shutil copies with ordinary reads and writes (no sendfile, which silently truncated Lustre-to-Lustre copies on
# lwacalim07 on 2026-11-20), and a copy shorter than its source is refused (exopipe/_safe_copy.py; the in-place children get the
# same from core/parity_sitecustomize/sitecustomize.py). tests/test_no_sendfile.py.
from exopipe import _safe_copy
_safe_copy.install()
