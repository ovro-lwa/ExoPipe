"""Config adapter of the follow-up package for the refactor window (`exopipe3 followup investigate | summary`).

The package reads its paths from CANDINV_<NAME> environment variables ONCE, when ``exopipe.phase3.followup.config`` is
first imported (config.py:14-15, module-level ``_env``). The step therefore sets them from ``CFG.followup_investigate``
(and the run's --out) BEFORE that first import, here and nowhere else, and checks afterwards that the package took them.
Every other CANDINV_* variable inherited from the caller's shell is removed first, so the package's own defaults apply
exactly as in production (a stale CANDINV_VLSS_FULL or CANDINV_TRANSIENTS_DIR never leaks in).

Gate G13 (spec followup.md section 5): if the package config is already imported when ``apply`` is called, the paths would
not apply - refuse. The adapter changes nothing in the package modules; it disappears at absorption, when the package
reads ``CFG`` directly.
"""
import os
import sys

# config keys of followup_investigate passed as CANDINV_<KEY> (spec section 3, the ENV keys)
ENV_KEYS = ('TOP50_CSV', 'BUNDLE_DIR', 'PROD_ROOT', 'PROD_LABEL', 'REDO1_LABEL', 'PHASE2_ARCHIVE', 'PHASE2_FAST',
            'OUT_ROOT', 'OUT_LABEL', 'OUT_SUBDIR', 'VLSS_UNRESOLVED', 'FLARE_STAR_CSV', 'ATNF_CSV')
PKG_CONFIG = 'exopipe.phase3.followup.config'


def env_for(section, out_root=None):
    """{CANDINV_<KEY>: value} from a followup_investigate section; OUT_ROOT is the run's product root when given."""
    env = {f'CANDINV_{k}': str(getattr(section, k)) for k in ENV_KEYS}
    if out_root:
        env['CANDINV_OUT_ROOT'] = os.path.abspath(out_root)
    return env


def apply(section, out_root=None):
    """Set the CANDINV_* environment for the package's first import; returns the mapping. Raises record.Refusal (G13) when
    the package config was imported before (its paths would silently stay the earlier ones)."""
    from exopipe.phase3.core.record import Refusal
    env = env_for(section, out_root)
    if PKG_CONFIG in sys.modules:
        mod = sys.modules[PKG_CONFIG]
        stale = {k: (getattr(mod, k[len('CANDINV_'):], None), v) for k, v in env.items()
                 if str(getattr(mod, k[len('CANDINV_'):], None)) != v}
        if stale:
            raise Refusal('G13: exopipe.phase3.followup.config was imported before the step set its paths; these would '
                          'not apply: ' + '; '.join(f'{k} is {a!r}, wanted {b!r}' for k, (a, b) in sorted(stale.items())))
        return env
    for k in [k for k in os.environ if k.startswith('CANDINV_') and k not in env]:
        del os.environ[k]
    os.environ.update(env)
    return env


def check_applied(env):
    """After the import: every CANDINV_* value is what the package config holds. Returns the list of mismatches."""
    mod = sys.modules.get(PKG_CONFIG)
    if mod is None:
        return ['package config not imported']
    return [f'{k}: package has {getattr(mod, k[len("CANDINV_"):], None)!r}, wanted {v!r}' for k, v in sorted(env.items())
            if str(getattr(mod, k[len('CANDINV_'):], None)) != v]
