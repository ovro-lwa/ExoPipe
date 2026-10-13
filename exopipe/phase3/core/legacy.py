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
- the script's own sys.path handling is left alone: it runs exactly as production runs it.
The child's stdout and stderr go to ``log`` (and a tail is kept). ``refused`` is True when the exit code is in
``refusal_codes`` or the output contains one of ``refusal_text``; ``ok`` only when rc == 0 and not refused.
"""
import json
import os
import subprocess
import tempfile
import time


class LegacyResult:
    def __init__(self, rc, log, tail, fingerprint, refused, reason, wall_s, argv):
        self.rc, self.log, self.tail, self.fingerprint = rc, log, tail, fingerprint
        self.refused, self.reason, self.wall_s, self.argv = refused, reason, wall_s, argv

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
    try:
        with open(fp_path) as f:
            fp = json.load(f)
    except (OSError, ValueError):
        fp = {}
    finally:
        try:
            os.remove(fp_path)
        except OSError:
            pass
    hit = next((t for t in refusal_text if t in tail), None)
    refused = rc in refusal_codes or (hit is not None and rc != 0)
    reason = '' if rc == 0 and not refused else (
        f"{os.path.basename(target)} exit {rc}" + (f" ({hit}: {_line_with(tail, hit)})" if hit else '')
        + ('' if refused else ' - error'))
    res = LegacyResult(rc, log, tail, fp, refused, reason, round(time.time() - t0, 1), argv)
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
