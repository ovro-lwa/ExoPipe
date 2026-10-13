"""The production scripts serve uses in place during the refactor window (DESIGN 11.1; spec serve.md 7.2) - WHERE they are.

Every Python script is RUN through the one child runner, ``exopipe.phase3.core.legacy.run`` (refactor-chat decision 1,
2026-10-13: site interpreter, inherited P3_* cleared, parity guard, child code fingerprint in the unit record). This module
only names the scripts (resolved under ``core.env.legacy_root('exopipe_ops')``) and holds the two things core.legacy.run
does not do, each documented where it is used:
  run_shell()   the one bash script serve calls in place (snapshot_localize_tiers.sh): core.legacy.run always puts a
                Python interpreter in front of its target, so the script runs through bash with the SAME environment rules
                (P3_* cleared, runner.ENV_DEFAULTS, EXOPIPE3_SITE, the parity sitecustomize on PYTHONPATH so its Python
                children are guarded)
  server_env()  the environment of a long-lived server (procs.start): the same rules, without the exit-time fingerprint
                file (a server never exits normally); its code identity is recorded by the App probe child instead
The scripts move into lib/serve/ at absorption (MOVE_PLAN 2.11), when the owning chats stop editing them.
"""
import os
import subprocess
import time

# name -> (path relative to the exopipe_ops root, where production calls it)
SCRIPTS = {
    'classify_transients': ('classify_transients.py', 'p3s3_serve_stokes.sh:79 (the review server)'),
    'jupiter_server': ('jupiter_server.py', 'the live 8800 server (no launcher)'),
    'stokes_queue': ('p3s3_stokes_queue.py', 'p3s3_serve_stokes.sh:11; sublst_serve.sh:73'),
    'merge_review_dir': ('p3s3_merge_review_dir.py', 'sublst_serve.sh:52 (sub-LST cumulative build; deep: pages owns it)'),
    'multiples_fork': ('multiples_fork.py', 'p3s3_serve_multiples.sh:136 (served <S>)'),
    'build_review_hub': ('build_review_hub.py', 'p3s3_serve_stokes.sh:84 and every serve launcher'),
    'build_inspection_hub': ('build_inspection_hub.py', 'run_inspection_hub.sh:5'),
    'deep_carry': ('deep_carry_judgements.py', 'deep_carry_judgements.py docstring (by hand, 2026-10-06)'),
    'snapshot_carry': ('snapshot_carry_judgements.py', 'snapshot_serve.sh:37; snapshot_rechain_batch.sh:374; '
                                                       'snapshot_forced_serve_20261007.sh:77'),
    'snapshot_gate': ('snapshot_serve_exclusion_gate.py', 'snapshot_serve.sh:26-27, 47-48'),
    'snapshot_localize': ('snapshot_localize_tiers.sh', 'snapshot_serve.sh:44-46'),
    'sublst_cumulative': ('sublst_cumulative.py', 'sublst_cumulative.sh:26'),
    'build_localizations': ('build_localizations.py', 'sublst_cumulative.sh:29'),
    'astrometry_panel': ('source_astrometry_panel.py', 'sublst_cumulative.sh:53'),
    'pages_index': ('p3t_pages_index.py', 'sublst_cumulative.sh:62-64'),
    'sublst_served_test': ('sublst_reserve_test_served.py', 'sublst_catalog_publish.sh:106'),
    'forced_adleo': ('test_snapshot_forced_reserve.py', 'snapshot_forced_serve_20261007.sh:68'),
}


def ops_root():
    from exopipe.phase3.core.env import legacy_root
    from exopipe.phase3.core.record import Refusal
    try:
        return os.path.normpath(legacy_root('exopipe_ops'))
    except KeyError as e:
        raise Refusal(f'site file: {e} - serve calls the exopipe_ops scripts in place during the refactor window')


def clv15_root():
    from exopipe.phase3.core.env import legacy_root
    return os.path.normpath(legacy_root('clv15'))


def path(name):
    return os.path.join(ops_root(), SCRIPTS[name][0])


def interpreter(key='default'):
    from exopipe.phase3.core import legacy
    return legacy.interpreter(key)


def site_value(name, default=None):
    from exopipe.phase3.config import CFG
    return getattr(CFG.site, name, default)


def _base_env(extra=None, with_parity=True):
    """os.environ with inherited P3_* cleared, then ``extra`` (a value None or '' = left unset), runner.ENV_DEFAULTS,
    EXOPIPE3_SITE; in a parity run the parity sitecustomize stays first on PYTHONPATH (core.legacy.run's rules)."""
    from exopipe.phase3.config import CFG
    env = {k: v for k, v in os.environ.items() if not k.startswith('P3_')}
    for k, v in (extra or {}).items():
        if v is None or v == '':
            env.pop(k, None)
        else:
            env[k] = str(v)
    env.update(CFG.runner.ENV_DEFAULTS)
    if os.environ.get('EXOPIPE3_SITE'):
        env['EXOPIPE3_SITE'] = os.environ['EXOPIPE3_SITE']
    if with_parity and os.environ.get('EXOPIPE3_PARITY_ROOT'):
        from exopipe.phase3.core import guard
        env['EXOPIPE3_PARITY_ROOT'] = os.environ['EXOPIPE3_PARITY_ROOT']
        parts = [p for p in (env.get('PYTHONPATH') or '').split(os.pathsep) if p and p != guard.SITECUSTOMIZE_DIR]
        env['PYTHONPATH'] = os.pathsep.join([guard.SITECUSTOMIZE_DIR] + parts)
    return env


def server_env(extra):
    return _base_env(extra)


def run_shell(name, args, env=None, cwd=None, timeout=None, log=None, unit=None):
    """bash <ops>/<script> args (see the module docstring). Returns (rc, combined output)."""
    from exopipe.phase3.config import CFG
    argv = ['bash', path(name)] + [str(a) for a in args]
    to = timeout or CFG.serve.LEGACY_TIMEOUT_S
    t0 = time.time()
    try:
        p = subprocess.run(argv, cwd=cwd or ops_root(), env=_base_env(env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, timeout=to)
        out, rc = p.stdout.decode('utf-8', 'replace'), p.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = (e.stdout or b'').decode('utf-8', 'replace') + f'\nTIMEOUT after {to} s', 124
    if log:
        with open(log, 'a') as f:
            f.write(out if out.endswith('\n') or not out else out + '\n')
    if unit is not None:
        unit.meta.setdefault('legacy_calls', []).append({'argv': argv, 'rc': rc, 'wall_s': round(time.time() - t0, 1),
                                                         'log': log})
    return rc, out


def last_line(text):
    lines = [l for l in (text or '').strip().splitlines() if l.strip()]
    return lines[-1] if lines else ''


def run(name, args=(), python='default', env=None, cwd=None, log=None, unit=None, timeout=None, **kw):
    """core.legacy.run on the named script, cwd the exopipe_ops root (as every serve launcher), timeout
    serve.LEGACY_TIMEOUT_S. Returns the LegacyResult (rc, tail, ok, refused, reason)."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import legacy
    clean = {k: v for k, v in (env or {}).items() if v not in (None, '')}
    return legacy.run(path(name), args, python=python, env=clean, cwd=cwd or ops_root(), log=log, unit=unit,
                      timeout=timeout or CFG.serve.LEGACY_TIMEOUT_S, **kw)
