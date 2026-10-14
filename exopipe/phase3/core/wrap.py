"""Import wraps of IN-PLACE production modules in child processes (wave 2, WP-C1 mechanism M3) - how a ruling reaches
production code that the new steps call in place without editing it. First user: B-32 (a), exopipe_ops/source_localization.py.

WRAPS maps a production module name (as the child imports it) to the branch patch file that changes it and the config keys
the patch reads, each with its PRODUCTION value. The step process exports EXOPIPE3_WRAPS - JSON {module: {'patch': path,
'cfg': {key: value}}} for the modules with any key off its production value - and core.legacy.run passes it to every child
(export_env, like core/fallback.py's policies). The child's core/parity_sitecustomize/sitecustomize.py installs a meta-path
finder for those names: the PRODUCTION module loads as always, then the patch file loads BY PATH (a child's `exopipe` can
be the production package) and applies - after checking its pins (the sha1 of the source TEXT of every production function
it replaces or wraps: the same in every Python version a child may run, where ast.dump is not), refusing at import when
production changed them. With every key at its production value nothing is wrapped and the child runs production code byte
for byte.
"""
import json
import os

_PHASE3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WRAPS = {
    'source_localization': {
        'patch': os.path.join(_PHASE3, 'lib', 'catalog', 'wraps', 'source_localization.py'),
        'keys': {'astrometry.LOC_ESTIMATOR': 'production', 'astrometry.LOC_ERROR_SOURCE': 'production',
                 'fallbacks.LOC_MASTER_BEAM_FALLBACK': 'fallback', 'fallbacks.LOCALIZATION_FALLBACK': 'fallback'},
    },
}


def active(cfg=None):
    """{module: {'patch': path, 'cfg': {key: value}}} of the wraps whose keys are not all at their production value."""
    if cfg is None:
        from exopipe.phase3.config import CFG as cfg
    out = {}
    for mod, w in WRAPS.items():
        vals = {}
        for key in w['keys']:
            sec, _, name = key.partition('.')
            vals[key] = getattr(getattr(cfg, sec), name)
        if any(vals[k] != prod for k, prod in w['keys'].items()):
            out[mod] = {'patch': w['patch'], 'cfg': vals}
    return out


def export_env():
    """A step process (cli.main, after its overrides): every descendant wraps the same modules the same way."""
    a = active()
    if a:
        os.environ['EXOPIPE3_WRAPS'] = json.dumps(a, sort_keys=True)
    else:
        os.environ.pop('EXOPIPE3_WRAPS', None)


def child_env():
    """The entries core.legacy.run puts into a child's environment (the wraps of the current config)."""
    a = active()
    return {'EXOPIPE3_WRAPS': json.dumps(a, sort_keys=True)} if a else {}
