"""Run a clv15 / exopipe_ops script or module in a child process - the ONE helper every step uses during the refactor
window (DESIGN 11.1: steps call production code in place; crosscheck 4.6 found six home-grown variants).

    res = legacy.run('/lustre/gh/main/exopipe_ops/p3s3_merge.py', ['--run', run_dir], python='default',
                     env={'P3_COADD_ROOT': root}, log=run_dir + '/merge.log', unit=u)
    if res.refused: u.refuse(res.reason)

What the child gets:
- the interpreter named by the site file's PYTHON[python] ('default', 'flow', 'py36', 'casa', 'tess');
- the parent's environment with inherited production switches CLEARED (runner.CHILD_ENV_CLEAR_PREFIXES: P3_*, P3S*,
  P3T*, SUBLST_*, SNAP_*, ... - a stale P3_COADD_ROOT or SUBLST_TABLES never leaks in), then ``env``,
  then runner.ENV_DEFAULTS (NUMPY_MADVISE_HUGEPAGE=0) and EXOPIPE3_SITE;
- PYTHONPATH prefixed with core/parity_sitecustomize: under ``parity_root`` (or the parent's --parity) every write outside
  the parity roots is refused in the child too; and the child writes at exit the sha1 of every imported legacy / exopipe
  source file (returned in ``res.fingerprint`` and added to the unit's record, so --resume notices a production edit);
- B-30: the policy of every fallback key (EXOPIPE3_FALLBACK_POLICY, the parent's config with its overrides); a branch
  module in the child that reaches a fallback site counts or refuses there, and its counts and refusals come back in
  ``res.fallbacks`` / ``res.fallback_refusals`` and into this process's fallback state, so the running unit records
  them (a refusal makes ``refused`` True even at exit 0, and the unit refuses - core/fallback.py);
- the script's own sys.path handling is left alone: it runs exactly as production runs it.
The child's stdout and stderr go to ``log`` (and a tail is kept). ``refused`` is True when the exit code is in
``refusal_codes`` or the output contains one of ``refusal_text``; ``ok`` only when rc == 0 and not refused.
"""
import glob
import json
import os
import re
import subprocess
import tempfile
import time

# B-01 (b), Gregg 2026-10-13 ("no patch path in the new code; patched Run_P3S4 results stay in Run_P3S5 until the next full
# re-search"): the production scripts that patch Run_P3S4 attempts into Run_P3S5 - p3s5_patch_cell.py, p3s5_patch_runner.sh,
# p3s5_patch_untouched.sh, p3s5_repatch_epochgate.py and any later p3s5_*patch* - and the ops runners that call them are
# never run or imported by the branch: every in-place entry point refuses them (refuse_patch_path).
PATCH_NAME = re.compile(r'^p3s5_(re)?patch')
PATCH_RUNNERS = ('mdewarp_rbf_20261004/install_runner.sh', 'mdewarp_rbf_20261004/lst_watch.sh',
                 'mdewarp_rbf_20261004/chain_11h.sh', 'membership_rebuild_20261007/install_runner.sh')


def patch_path(target):
    """True when ``target`` - a script path, a module name ('p3s5_patch_cell', 'a.b.p3s5_repatch_epochgate') or a command
    line ('bash -c "p3s5_patch_runner.sh cells.txt 4"': each word) - names one of the patch scripts or of the runners
    that call them."""
    for t in str(target or '').replace('"', ' ').replace("'", ' ').replace(';', ' ').split():
        name = os.path.basename(t.rstrip('/'))
        if '/' not in t and not t.endswith(('.py', '.sh')):
            name = name.rsplit('.', 1)[-1]                    # a dotted module name: its last component
        if PATCH_NAME.match(name) or any(os.path.normpath(t).endswith(r) for r in PATCH_RUNNERS):
            return True
    return False


def refuse_patch_path(target, where):
    """Raise Refusal when ``target`` is a patch path (B-01 b)."""
    if patch_path(target):
        from exopipe.phase3.core.record import Refusal
        raise Refusal(f'{where}: {target} is a Run_P3S4 -> Run_P3S5 patch path - the new code has none (B-01 b, Gregg '
                      f'2026-10-13: patched results stay in Run_P3S5 until the next full re-search)')


class _PatchImportGuard:
    """sys.meta_path finder (first): importing a patch module refuses - whatever spelling reached the import system."""

    def find_spec(self, name, path=None, target=None):
        if PATCH_NAME.match(name.rsplit('.', 1)[-1]):
            refuse_patch_path(name, 'import')
        return None


def install_patch_import_guard():
    """Put _PatchImportGuard first on sys.meta_path (idempotent; core/env.legacy installs it for every step)."""
    import sys
    if not any(isinstance(f, _PatchImportGuard) for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchImportGuard())


class LegacyResult:
    def __init__(self, rc, log, tail, fingerprint, refused, reason, wall_s, argv, fallbacks=None, fallback_refusals=()):
        self.rc, self.log, self.tail, self.fingerprint = rc, log, tail, fingerprint
        self.refused, self.reason, self.wall_s, self.argv = refused, reason, wall_s, argv
        self.fallbacks, self.fallback_refusals = dict(fallbacks or {}), list(fallback_refusals)

    @property
    def ok(self):
        return self.rc == 0 and not self.refused


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def interpreter(name='default'):
    py = _cfg().site.PYTHON.get(name)
    if not py:
        from exopipe.phase3.core.record import Refusal
        raise Refusal(f"no interpreter {name!r} in the site file PYTHON (keys: {sorted(_cfg().site.PYTHON)})")
    return py


def run(target, args=(), python='default', module=False, env=None, cwd=None, log=None, parity_root=None,
        timeout=None, refusal_codes=(2,), refusal_text=('REFUSED', 'REFUSING', 'REFUSE:'), unit=None, tail_lines=40):
    """Run ``target`` (a script path, or a module name with module=True) with ``args``. See the module docstring."""
    refuse_patch_path(target, 'core.legacy.run')
    for a in args:                                            # a patch script handed to an interpreter or shell as an argument
        refuse_patch_path(a, 'core.legacy.run')
    cfg = _cfg()
    py = interpreter(python)
    argv = [py] + (['-m', target] if module else [target]) + [str(a) for a in args]
    clear = tuple(getattr(cfg.runner, 'CHILD_ENV_CLEAR_PREFIXES', ('P3_',)))
    child = {k: v for k, v in os.environ.items() if not k.startswith(clear)}
    child.update({k: str(v) for k, v in (env or {}).items()})
    child.update(cfg.runner.ENV_DEFAULTS)
    if os.environ.get('EXOPIPE3_SITE'):
        child['EXOPIPE3_SITE'] = os.environ['EXOPIPE3_SITE']
    here = os.path.dirname(os.path.abspath(__file__))
    boot = os.path.join(here, 'parity_sitecustomize')
    child['PYTHONPATH'] = boot + (os.pathsep + child['PYTHONPATH'] if child.get('PYTHONPATH') else '')
    proot = parity_root or os.environ.get('EXOPIPE3_PARITY_ROOT')
    if proot:
        child['EXOPIPE3_PARITY_ROOT'] = proot
    fp_fd, fp_path = tempfile.mkstemp(prefix='exopipe3_fp_', suffix='.json')
    os.close(fp_fd)
    child['EXOPIPE3_FINGERPRINT_OUT'] = fp_path
    child['EXOPIPE3_FINGERPRINT_ROOTS'] = os.pathsep.join(cfg.site.LEGACY_CODE_ROOTS)
    from exopipe.phase3.core import fallback, wrap
    child.update(fallback.child_env())
    child.pop('EXOPIPE3_WRAPS', None)          # the wraps of THIS config only (core/wrap.py; B-32)
    child.update(wrap.child_env())
    fb_fd, fb_path = tempfile.mkstemp(prefix='exopipe3_fb_', suffix='.json')
    os.close(fb_fd)
    os.remove(fb_path)                  # written by the child only when it imported core/fallback
    child['EXOPIPE3_FALLBACK_OUT'] = fb_path
    t0 = time.time()
    lf = open(log, 'a') if log else tempfile.TemporaryFile(mode='w+')
    try:
        if log:
            lf.write(f'\n===== {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} {" ".join(argv)}\n')
            lf.flush()
        p = subprocess.run(argv, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=cwd, env=child,
                           timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = -9
    finally:
        lf.flush()
    tail = _tail(log, tail_lines) if log else _tail_file(lf, tail_lines)
    lf.close()
    fp = {}
    for p in [fp_path] + sorted(glob.glob(glob.escape(fp_path) + '.*')):   # one file per process of the child (sitecustomize)
        try:
            with open(p) as f:
                fp.update(json.load(f))
        except (OSError, ValueError):
            pass
        finally:
            try:
                os.remove(p)
            except OSError:
                pass
    fb_counts, fb_refs = fallback.load(fb_path)        # the child's and its descendants' (<fb_path>.<pid>), removed
    hit = next((t for t in refusal_text if t in tail), None)
    refused = rc in refusal_codes or (hit is not None and rc != 0)
    reason = '' if rc == 0 and not refused else (
        f"{os.path.basename(target)} exit {rc}" + (f" ({hit}: {_line_with(tail, hit)})" if hit else '')
        + ('' if refused else ' - error'))
    fallback.absorb(fb_counts, fb_refs)    # into this process's state: the running unit takes them (with or without unit=)
    if fb_refs:                         # B-30: sticky - refused in the child even if the child carried on
        refused = True
        reason = (f'{os.path.basename(target)}: B-30 fallback site {fb_refs[0][0]}: {fb_refs[0][1]} - REFUSED'
                  + (f' (exit {rc})' if rc else ''))
    res = LegacyResult(rc, log, tail, fp, refused, reason, round(time.time() - t0, 1), argv, fb_counts, fb_refs)
    if unit is not None:
        unit.meta.setdefault('legacy_calls', []).append({'argv': argv, 'rc': rc, 'wall_s': res.wall_s, 'log': log})
        unit.meta.setdefault('legacy_code', {}).update(fp)
    return res


def _tail(path, n):
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            return '\n'.join(f.read().decode('utf-8', 'replace').splitlines()[-n:])
    except OSError:
        return ''


def _tail_file(f, n):
    try:
        f.seek(0)
        return '\n'.join(f.read().splitlines()[-n:])
    except Exception:
        return ''


def _line_with(text, needle):
    for line in reversed(text.splitlines()):
        if needle in line:
            return line.strip()[:200]
    return ''
