"""Import wraps of IN-PLACE production modules in child processes (wave 2, WP-C1 mechanism M3) - how a ruling reaches
production code that the new steps call in place without editing it. First user: B-32 (a), exopipe_ops/source_localization.py.

WRAPS maps a production module name (as the child imports it) to the branch patch file that changes it and the config keys
the patch reads, each with its PRODUCTION value and the values it may take. The step process (cli.main) refuses an unknown
value (check_values), then exports EXOPIPE3_WRAPS - JSON {module: {'patch': path, 'sha1': ..., 'cfg': {key: value}}} for
the modules with any key off its production value, '{}' when none (said explicitly, so no descendant falls back to its own
defaults) - with EXOPIPE3_WRAPS_PID. core.legacy.run puts into every child the value this process was given when it is not
the exporter (a nested child passes the step's wraps on), else the wraps of its own config (child_env). The child's
core/parity_sitecustomize/sitecustomize.py - first on PYTHONPATH of every core.legacy.run child and of serve's shell scripts
- installs a meta-path finder for those names: the PRODUCTION module loads as always, then the patch file loads BY PATH (a
child's `exopipe` can be the production package), is registered in sys.modules as exopipe3_wrap.<module> (so the child's
code fingerprint names it and a changed patch invalidates --resume) and applies - after checking its pin, the sha1 of the
whole production source file, refusing at import when production changed it. With every key at its production value nothing
is wrapped and the child runs production code byte for byte.
"""
import hashlib
import json
import os

_PHASE3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# module -> patch file and {config key: (production value, allowed values)}
WRAPS = {
    'source_localization': {
        'patch': os.path.join(_PHASE3, 'lib', 'catalog', 'wraps', 'source_localization.py'),
        'keys': {'astrometry.LOC_ESTIMATOR': ('production', ('ivw-sandwich', 'production')),
                 'astrometry.LOC_ERROR_SOURCE': ('production', ('fitted', 'production')),
                 'fallbacks.LOC_MASTER_BEAM_FALLBACK': ('fallback', ('refuse', 'fallback')),
                 'fallbacks.LOCALIZATION_FALLBACK': ('fallback', ('refuse', 'fallback')),
                 'fallbacks.LOC_FORK_ONLY_FALLBACK': ('fallback', ('refuse', 'fallback'))},
    },
}


def _values(w, cfg):
    out = {}
    for key in w['keys']:
        sec, _, name = key.partition('.')
        out[key] = getattr(getattr(cfg, sec), name)
    return out


def check_values(cfg=None):
    """[problems]: each wrap key whose value is none of its allowed values (a misspelt value would otherwise wrap nothing
    while the record says it did). cli.main refuses the step on any."""
    if cfg is None:
        from exopipe.phase3.config import CFG as cfg
    bad = []
    for w in WRAPS.values():
        for key, v in _values(w, cfg).items():
            allowed = w['keys'][key][1]
            if v not in allowed:
                bad.append(f'{key} = {v!r} (one of {allowed})')
    return bad


def _sha1(path):
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()


def active(cfg=None):
    """{module: {'patch': path, 'sha1': patch sha1, 'cfg': {key: value}}} of the wraps whose keys are not all at their
    production value."""
    if cfg is None:
        from exopipe.phase3.config import CFG as cfg
    out = {}
    for mod, w in WRAPS.items():
        vals = _values(w, cfg)
        if any(vals[k] != prod for k, (prod, _) in w['keys'].items()):
            out[mod] = {'patch': w['patch'], 'sha1': _sha1(w['patch']), 'cfg': vals}
    return out


def export_env():
    """A step process (cli.main, after its overrides and check_values): every descendant wraps the same modules the same
    way - '{}' exported when nothing is wrapped."""
    os.environ['EXOPIPE3_WRAPS'] = json.dumps(active(), sort_keys=True)
    os.environ['EXOPIPE3_WRAPS_PID'] = str(os.getpid())


def child_env():
    """EXOPIPE3_WRAPS for a child (core.legacy.run): the value this process was given when it did not export it itself (a
    legacy.run inside a child passes the step's wraps on, never the child's own config defaults); else this config's."""
    raw = os.environ.get('EXOPIPE3_WRAPS')
    if raw is not None and os.environ.get('EXOPIPE3_WRAPS_PID') != str(os.getpid()):
        return {'EXOPIPE3_WRAPS': raw}
    return {'EXOPIPE3_WRAPS': json.dumps(active(), sort_keys=True)}
