"""Process environment defaults applied by `exopipe3` BEFORE anything imports numpy (no imports here).

NUMPY_MADVISE_HUGEPAGE=0: numpy asks for transparent hugepages on large arrays; on nodes with fragmented memory every
such page fault stalls in kernel compaction - one 5.5-Mpixel array step took 14.3 s with the request and 0.09 s
without on lwacalim07 (2026-10-04, ledger FINDING-LWACALIM07-NUMPY-HUGEPAGE-COMPACTION-STALL); arithmetic unchanged.
Thread caps are NOT set by default (they can change floating-point reduction order); use a step's --threads N.
"""
import os

ENV_DEFAULTS = {'NUMPY_MADVISE_HUGEPAGE': '0'}


def apply(threads=None):
    for k, v in ENV_DEFAULTS.items():
        os.environ.setdefault(k, v)
    if threads:
        for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[k] = str(threads)


def legacy():
    """Refactor window (DESIGN 11.1): let clv15 / exopipe_ops modules be called in place. Appends the site file's
    LEGACY_CODE_ROOTS to sys.path (after the package, so nothing in them shadows it) and installs the import finder that
    maps pre-refactor dotted names (exopipe.phase3.coadd_full_pipeline, ...) to their new modules without any file at
    the old paths (core/_shim.py). Idempotent."""
    import sys
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import _shim
    for root in CFG.site.LEGACY_CODE_ROOTS:
        if root not in sys.path:
            sys.path.append(root)
    _shim.install_legacy_finder()
