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


# ---- comparing a moved module with its production original ----------------------------------------------------------

def _is_site_stmt(n):
    import ast
    if isinstance(n, ast.ImportFrom) and n.module == 'exopipe.phase3.core' and [a.name for a in n.names] == ['fallback']:
        return True
    if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
        f = n.value.func
        return isinstance(f, ast.Attribute) and f.attr == 'site' and isinstance(f.value, ast.Name) and f.value.id == 'fallback'
    return False


def site_lines(text):
    """Line numbers (1-based) of the B-30 site statements of a module: `from exopipe.phase3.core import fallback`,
    `fallback.site(...)`, an `if` (no else) whose body is only those, and an `else:` branch that is only those (the
    `else:` line included when it stands alone). A site is a pure insertion of whole lines."""
    import ast
    lines = text.split('\n')
    out = set()

    def only_sites(stmts):
        return bool(stmts) and all(_is_site_stmt(x) or (isinstance(x, ast.If) and not x.orelse and only_sites(x.body))
                                   for x in stmts)
    for node in ast.walk(ast.parse(text)):
        for field in ('body', 'orelse', 'finalbody'):
            stmts = getattr(node, field, None)
            if not (isinstance(stmts, list) and stmts and isinstance(stmts[0], ast.stmt)):
                continue
            for st in stmts:
                if _is_site_stmt(st) or (isinstance(st, ast.If) and not st.orelse and only_sites(st.body)):
                    out.update(range(st.lineno, st.end_lineno + 1))
        if isinstance(node, ast.If) and node.orelse and only_sites(node.orelse) and not isinstance(node.orelse[0], ast.If):
            k = node.orelse[0].lineno - 1                     # the `else:` line, when it stands alone
            if lines[k - 1].strip() == 'else:':
                out.add(k)
                for st in node.orelse:
                    out.update(range(st.lineno, st.end_lineno + 1))
    return out


def strip_site_lines(text):
    """The module text without its B-30 site lines (site_lines): what is left is the production text."""
    drop = site_lines(text)
    return '\n'.join(x for i, x in enumerate(text.split('\n'), 1) if i not in drop)


def same_code_modulo_sites(production_text, branch_text):
    """True when the branch module is the production module plus B-30 site lines only: byte equality once they are
    removed (a comment-only or any other edit of production still differs - the gates that use this stay byte-strict)."""
    return production_text == strip_site_lines(branch_text)
