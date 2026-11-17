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


def code_identity_problems():
    """The code trees a step runs whose commit cannot be read ([] = every one names a commit): this package's GITTAG
    (phase2_config.PIPELINE_GITTAG, stamped into every v2 product header; its static fallback 'clv15' when `git rev-parse`
    fails) and each site LEGACY_CODE_ROOTS tree that exists (production modules run in place stamp GITTAG from their own
    tree). Causes: not a git checkout (an export), or git refuses the tree for this user - 'detected dubious ownership'
    when user pipeline runs a tree owned by gh (git >= 2.35.2): a safe.directory entry in that user's git config, or the
    tree owned by that user."""
    import re
    import subprocess
    from exopipe.common import phase2_config
    from exopipe.phase3.config import CFG
    commit = re.compile(r'^[0-9a-f]{7,40}$')
    out = []
    g = str(phase2_config.PIPELINE_GITTAG)
    if not commit.match(g):
        out.append(f'{os.path.dirname(os.path.abspath(phase2_config.__file__))}: GITTAG would be {g!r}')
    for root in CFG.site.LEGACY_CODE_ROOTS:
        if not os.path.isdir(root):
            continue
        try:
            r = subprocess.run(['git', '-C', root, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                               timeout=60)
            sha, err = (r.stdout.strip() if r.returncode == 0 else ''), r.stderr.strip()
        except (OSError, subprocess.SubprocessError) as e:
            sha, err = '', str(e)
        if not commit.match(sha):
            out.append(f'{root}: git rev-parse failed ({(err.splitlines() or ["no output"])[0][:200]})')
    return out


def check_code_identity():
    """B-30 site GITTAG_FALLBACK, at step start (cli.main, before the step's run(): no work done, nothing written): the
    step refuses - under 'fallback' (parity, production behavior) it is counted once - when code_identity_problems() is
    not empty. Checked once per process because the code identity is fixed at import; a check at each header write would
    refuse after the product's pixels were already on disk."""
    bad = code_identity_problems()
    if bad:
        from exopipe.phase3.core import fallback
        fallback.site('fallbacks.GITTAG_FALLBACK', 'code tree(s) whose commit cannot be read, so products would carry no '
                      'commit: ' + '; '.join(bad))

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
