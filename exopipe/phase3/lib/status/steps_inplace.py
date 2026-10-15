"""In-place calls of production status tools (exopipe_ops / clv15 / package) from `exopipe3 status` - refactor window.

Three ways a production tool is called (spec section 7.2):
  attr   import the module by name (legacy roots are on sys.path: core/env.legacy), override its module globals, call
         its main() - for tools whose main reads the globals at call time (conformance_check, build_qa_pages,
         build_data_processing_page, build_ratified_register, phase_status);
  argv   run the script in a CHILD through core.legacy.run (the one helper; site interpreter, P3_* cleared, child code
         fingerprint in the record) with its own output arguments (build_qa_table, beam_audit, qa_noise_vs_freq_plot,
         coadd_exclusion_conformance, bandpass_phase_warp, snapshot_batch_audit, snapshot_batch_lists);
  remap  run a MODULE-LEVEL script in-process (runpy) while ``redirect`` maps its literal output paths into --out
         (_inventory_p1p2 / _build_ledger, solar_contamination_audit, candidate_provenance_audit, perframe_solar_check,
         the fragment literal of build_data_processing_page). Spec blocker 2 proposed a core helper; core has none, so
         this status-local one is used and offered to core (it touches nothing outside these calls).

Every in-process call (attr, remap) runs under the parity write guard (core.guard) with --out (and --work) as the only
writable roots unless a guard is already active (--parity): a production path the tool writes that is not mapped is
REFUSED (PermissionError), never written. sys.path and the environment are restored afterwards; each script run by path
is added to the unit as an input of kind 'code' (record.code_fingerprint covers only imported modules; DESIGN 11.5).
"""
import builtins
import contextlib
import io
import os
import runpy
import sys
import traceback


def _real(p):
    return os.path.realpath(os.path.abspath(os.fsdecode(os.fspath(p))))


class _Mapper:
    def __init__(self, mapping):
        # exact path -> target; a path + suffix ('.part', '.new', '.tmp...') maps to target + the same suffix
        self.m = {_real(k): v for k, v in (mapping or {}).items()}

    def __call__(self, path):
        if isinstance(path, int):
            return path
        try:
            r = _real(path)
        except TypeError:
            return path
        if r in self.m:
            return self.m[r]
        for k, v in self.m.items():
            if r.startswith(k) and r[len(k):len(k) + 1] in ('.', '_', '~'):
                return v + r[len(k):]
        return path


@contextlib.contextmanager
def redirect(mapping, allowed_roots, env=None):
    """Map literal paths (reads and writes) to their --out places and refuse every other write outside
    ``allowed_roots`` (core.guard), for the duration of an in-process call. Restores sys.path and os.environ."""
    from exopipe.phase3.core import guard
    mp = _Mapper(mapping)
    own_guard = not guard.active()
    if own_guard:
        guard.install([r for r in allowed_roots if r])
    saved = {}
    o_open, o_ioopen = builtins.open, io.open
    o_replace, o_rename, o_makedirs = os.replace, os.rename, os.makedirs
    o_exists, o_isfile, o_remove, o_unlink, o_symlink = os.path.exists, os.path.isfile, os.remove, os.unlink, os.symlink
    o_getmtime = os.path.getmtime

    def _open(file, *a, **k):
        return o_open(mp(file), *a, **k)

    def _ioopen(file, *a, **k):
        return o_ioopen(mp(file), *a, **k)

    patches = [(builtins, 'open', _open), (io, 'open', _ioopen),
               (os, 'replace', lambda s, d, *a, **k: o_replace(mp(s), mp(d), *a, **k)),
               (os, 'rename', lambda s, d, *a, **k: o_rename(mp(s), mp(d), *a, **k)),
               (os, 'makedirs', lambda p, *a, **k: o_makedirs(mp(p), *a, **k)),
               (os, 'remove', lambda p, *a, **k: o_remove(mp(p), *a, **k)),
               (os, 'unlink', lambda p, *a, **k: o_unlink(mp(p), *a, **k)),
               (os, 'symlink', lambda s, d, *a, **k: o_symlink(s, mp(d), *a, **k)),
               (os.path, 'exists', lambda p: o_exists(mp(p))),
               (os.path, 'isfile', lambda p: o_isfile(mp(p))),
               (os.path, 'getmtime', lambda p: o_getmtime(mp(p)))]
    sys_path = list(sys.path)
    env_saved = {k: os.environ.get(k) for k in (env or {})}
    for obj, name, fn in patches:
        saved[(obj, name)] = getattr(obj, name)
        setattr(obj, name, fn)
    for k, v in (env or {}).items():
        os.environ[k] = str(v)
    try:
        yield mp
    finally:
        for (obj, name), fn in saved.items():
            setattr(obj, name, fn)
        sys.path[:] = sys_path
        for k, v in env_saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if own_guard:
            guard.uninstall()


def run_path(script, argv=(), mapping=None, allowed_roots=(), env=None, log=None, unit=None):
    """Run a module-level script in-process (runpy, __main__) under ``redirect``; stdout/stderr captured (and appended
    to ``log``). Returns (rc, text): rc 0 on normal end, the SystemExit code otherwise, 1 on an exception."""
    from exopipe.phase3.core.legacy import refuse_patch_path
    refuse_patch_path(script, 'steps_inplace.run_path')
    if unit is not None:
        unit.add_input(script, extra={'kind': 'code'})
    buf = io.StringIO()
    rc = 0
    old_argv = sys.argv
    sys.argv = [script] + [str(a) for a in argv]
    try:
        with redirect(mapping, allowed_roots, env=env):
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                try:
                    runpy.run_path(script, run_name='__main__')
                except SystemExit as e:
                    rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
                    if not isinstance(e.code, int) and e.code is not None:
                        buf.write(str(e.code) + '\n')
                except Exception:
                    rc = 1
                    buf.write(traceback.format_exc())
    finally:
        sys.argv = old_argv
    text = buf.getvalue()
    _log(log, script, argv, rc, text)
    return rc, text


def call_attr(module_name, attrs, fn_name='main', args=(), mapping=None, allowed_roots=(), env=None, log=None,
              unit=None, argv=None):
    """Import ``module_name`` (fresh: a cached copy is dropped first so its import-time env reads see ``env``), set
    ``attrs`` on it, call ``fn_name(*args)`` under ``redirect``. Returns (rc, text, module)."""
    from exopipe.phase3.core.legacy import refuse_patch_path
    refuse_patch_path(module_name, 'steps_inplace.call_attr')
    buf = io.StringIO()
    rc = 0
    mod = None
    old_argv = sys.argv
    if argv is not None:
        sys.argv = list(argv)
    try:
        with redirect(mapping, allowed_roots, env=env):
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                try:
                    sys.modules.pop(module_name, None)
                    import importlib
                    mod = importlib.import_module(module_name)
                    if unit is not None and getattr(mod, '__file__', None):
                        unit.add_input(mod.__file__, extra={'kind': 'code'})
                    for k, v in (attrs or {}).items():
                        setattr(mod, k, v)
                    getattr(mod, fn_name)(*args)
                except SystemExit as e:
                    rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
                except Exception:
                    rc = 1
                    buf.write(traceback.format_exc())
    finally:
        sys.argv = old_argv
    text = buf.getvalue()
    _log(log, module_name, args, rc, text)
    return rc, text, mod


def import_quiet(module_name, env=None):
    """Import a legacy module by name with ``env`` set during the import (modules that read env at import time) and
    sys.path restored afterwards (some insert production roots at import)."""
    import importlib
    from exopipe.phase3.core.legacy import refuse_patch_path
    refuse_patch_path(module_name, 'steps_inplace.import_quiet')
    sys_path = list(sys.path)
    env_saved = {k: os.environ.get(k) for k in (env or {})}
    for k, v in (env or {}).items():
        os.environ[k] = str(v)
    try:
        if env:
            sys.modules.pop(module_name, None)
        return importlib.import_module(module_name)
    finally:
        sys.path[:] = sys_path
        for k, v in env_saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def search_state_dir(label):
    """The deep search's published-attempt register of run ``label`` (config catalog.DEEP_STATE_DIR - the folder the deep
    catalogue harvest reads)."""
    from exopipe.phase3.config import CFG
    lab = label[4:] if label.startswith('Run_') else label
    return CFG.catalog.DEEP_STATE_DIR.format(ops_root=CFG.paths.OPS_ROOT, label_lower=lab.lower())


def search_env(label, state=None):
    """The environment p3s3_sources reads at import, pinned from config (see search_sources)."""
    from exopipe.phase3.config import CFG
    return {'P3S_LABEL': label, 'P3S3_STATE_DIR': state or search_state_dir(label),
            'P3S3_COLLECT_ROOTS': ':'.join(CFG.catalog.COLLECT_ROOTS)}


def search_sources(label, state=None):
    """ops/p3s3_sources imported for run ``label`` with its register folder and collect roots PINNED from config, as the
    deep catalogue harvest sets them (deep_chain.DeepRun.env): an inherited P3S3_STATE_DIR or P3S3_COLLECT_ROOTS would
    otherwise change the attempt set silently (2026-10-15 review). ``state``: another register folder (a copy)."""
    return import_quiet('p3s3_sources', env=search_env(label, state))


def legacy_script(name):
    """Absolute path of an exopipe_ops script (site LEGACY_CODE_ROOTS)."""
    from exopipe.phase3.core import env as E
    from exopipe.phase3.core.legacy import refuse_patch_path
    refuse_patch_path(name, 'steps_inplace.legacy_script')
    p = os.path.join(E.legacy_root('exopipe_ops'), name)
    if not os.path.exists(p):
        from exopipe.phase3.core.record import Refusal
        raise Refusal(f'production tool {p} not found (legacy root exopipe_ops)')
    return p


def _log(log, what, argv, rc, text):
    if not log:
        return
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, 'a') as f:
        f.write(f'===== {what} {" ".join(map(str, argv or ()))} -> rc {rc}\n{text}\n')
