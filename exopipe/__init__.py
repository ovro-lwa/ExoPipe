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

# No sendfile copies (2026-11-20, FINDING-20261120-SHUTIL-SENDFILE-COPIES-TRUNCATE-ON-LWACALIM07). On Linux, Python 3.8+
# shutil.copyfile (and copy, copy2, copytree, a move across filesystems) copies through os.sendfile and takes a return of 0 as
# the end of the file, without an error. On lwacalim07 that silently truncated 14 of 560 Lustre-to-Lustre copies of 9-39 MB
# FITS files (to 0-85 percent of their size; 28 units of the 11-20 hybrid archive push refused at G6), while ordinary reads
# and writes truncated 0 of 448 copies of the same files there. So every process that imports exopipe copies through
# shutil's own read/write fallback (copyfileobj), in 16 MiB blocks. tests/test_no_sendfile.py; the in-place children (which
# import no exopipe) get the same switch from core/parity_sitecustomize/sitecustomize.py.
import shutil as _shutil
_shutil._USE_CP_SENDFILE = False
if hasattr(_shutil, '_USE_CP_COPY_FILE_RANGE'):     # newer Pythons: the same fast path through copy_file_range
    _shutil._USE_CP_COPY_FILE_RANGE = False
_shutil.COPY_BUFSIZE = 16 << 20
del _shutil
