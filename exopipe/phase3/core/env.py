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
    the old paths (core/_shim.py), and the guard that refuses to import a Run_P3S4 -> Run_P3S5 patch module (B-01 b,
    core/legacy.PATCH_NAME: a plain import or importlib.import_module of p3s5_patch_cell, p3s5_repatch_epochgate, ...).
    Idempotent."""
    import sys
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import _shim, legacy as _legacy
    for root in CFG.site.LEGACY_CODE_ROOTS:
        if root not in sys.path:
            sys.path.append(root)
    _shim.install_legacy_finder()
    _legacy.install_patch_import_guard()


def legacy_root(name):
    """The site's legacy code root named 'clv15' or 'exopipe_ops' (refactor window only)."""
    from exopipe.phase3.config import CFG
    for r in CFG.site.LEGACY_CODE_ROOTS:
        if r.rstrip('/').endswith('/' + name):
            return r
    raise KeyError(f'no legacy code root {name!r} in the site file LEGACY_CODE_ROOTS')


def host():
    """This host's short name (lwacalim07), the key of every per-host site table."""
    import socket
    return socket.gethostname().split('.')[0]


def node_local_holder(path):
    """The host whose site NODE_LOCAL_ROOTS hold ``path`` (None: shared storage, readable from every node)."""
    from exopipe.phase3.config import CFG
    p = os.path.abspath(path)
    for h, roots in (CFG.site.NODE_LOCAL_ROOTS or {}).items():
        if any(p == r.rstrip('/') or p.startswith(r.rstrip('/') + '/') for r in roots):
            return h
    return None


def readable_here(path):
    """False when ``path`` is node-local to ANOTHER host: the same path here is a different disk (or absent)."""
    h = node_local_holder(path)
    return h is None or h == host()
