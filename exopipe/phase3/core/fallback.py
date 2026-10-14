"""B-30 (a), Gregg 2026-10-13 (R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32): no silent fallbacks.

A fallback SITE is a place where code substitutes a default for missing, unreadable or degenerate input and carries on
without saying so (coadd.get_midpoint_time returned 2025-01-01T00:00 for an image with no readable time; every
time-dependent gate downstream then worked on a wrong time). Each site names a key of the config section ``fallbacks``
(config/shared.py; ``<SITE>_FALLBACK`` - the section, not the suffix, makes it a fallback key: other sections hold
ordinary tunables named *_FALLBACK):

    'refuse'    (ruled)       the site raises FallbackRefused naming itself and the input;
    'fallback'  (production)  the site keeps production behaviour and is COUNTED in the unit record.

    from exopipe.phase3.core import fallback
    ...
    except Exception:
        fallback.site('fallbacks.IMAGE_TIME_FALLBACK', f'{path}: no DATE-OBS / DATE-BEG / MJD-OBS')
        return Time('2025-01-01T00:00:00', scale='utc')          # production behaviour, reached only under 'fallback'

A refusal is STICKY: it is also recorded here, and the unit that was running refuses at its end even when code between
the site and the unit caught the exception (the moved production modules hold many broad ``except Exception``).
In a child process (core.legacy.run) the policies arrive as JSON in EXOPIPE3_FALLBACK_POLICY, and the counts and
refusals go back through EXOPIPE3_FALLBACK_OUT (written at exit by core/parity_sitecustomize/sitecustomize.py).
Design: exopipe_ops/refactor_20261012/plan/wave2/B30_DESIGN.md.
"""
import collections
import json
import os

from exopipe.phase3.core.record import Refusal

POLICIES = ('refuse', 'fallback')
SECTION = 'fallbacks'
SUFFIX = '_FALLBACK'
COUNTS = collections.Counter()      # key -> fallbacks taken in this process (policy 'fallback')
REFUSALS = []                       # [(key, detail)] refused in this process (policy 'refuse'), in order
_ENV = None


class FallbackRefused(Refusal):
    """A fallback site under policy 'refuse'."""


def _env_policies():
    global _ENV
    if _ENV is None:
        raw = os.environ.get('EXOPIPE3_FALLBACK_POLICY')
        _ENV = json.loads(raw) if raw else {}
    return _ENV


def policy(key):
    """'refuse' or 'fallback' for ``fallbacks.<SITE>_FALLBACK``: the parent's JSON in a child, else config."""
    sec, _, name = key.partition('.')
    if sec != SECTION or not name.endswith(SUFFIX):
        raise KeyError(f'fallback key {key!r}: a fallback site names a key {SECTION}.<SITE>{SUFFIX} (config/shared.py)')
    env = _env_policies()
    if key in env:
        v = env[key]
    else:
        from exopipe.phase3.config import CFG
        v = getattr(getattr(CFG, sec), name)
    if v not in POLICIES:
        raise Refusal(f'{key} = {v!r}: one of {POLICIES}')
    return v


def site(key, detail, n=1):
    """A fallback site was reached. 'fallback': count ``n`` and return (the caller keeps production behaviour);
    'refuse': record and raise FallbackRefused."""
    if policy(key) == 'fallback':
        COUNTS[key] += n
        return
    REFUSALS.append((key, str(detail)))
    raise FallbackRefused(f'B-30 fallback site {key}: {detail} - REFUSED (no silent fallbacks, Gregg 2026-10-13; '
                          f"{key}='fallback' keeps the production behaviour, counted)")


def mark():
    """The state to compare against later (a unit takes it when it starts)."""
    return len(REFUSALS), COUNTS.copy()


def since(m):
    """(refusals, counts) recorded after mark ``m``."""
    n, c = m
    out = COUNTS.copy()
    out.subtract(c)
    return REFUSALS[n:], {k: v for k, v in out.items() if v > 0}


def absorb(counts, refusals):
    """Add a child's counts and refusals to this process's state (core.legacy.run), so the running unit records them."""
    COUNTS.update({k: int(n) for k, n in (counts or {}).items()})
    REFUSALS.extend(tuple(r) for r in refusals or ())


def all_keys(cfg=None):
    """{'fallbacks.KEY': value} of every fallback key (the keys of the config section ``fallbacks``)."""
    if cfg is None:
        from exopipe.phase3.config import CFG as cfg
    return {f'{SECTION}.{k}': v for k, v in getattr(cfg, SECTION).items()}


def child_env():
    """The environment entries a child needs: every fallback key's policy as JSON (the parent's config, overrides
    included)."""
    return {'EXOPIPE3_FALLBACK_POLICY': json.dumps(all_keys(), sort_keys=True)}


def dump(path):
    """Write this process's counts and refusals (a child, at exit)."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({'counts': dict(COUNTS), 'refusals': REFUSALS}, f)
    os.replace(tmp, path)


def load(path):
    """(counts, refusals) a child wrote; ({}, []) when it wrote none (it never imported this module)."""
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return {}, []
    return dict(d.get('counts') or {}), [tuple(r) for r in d.get('refusals') or []]
