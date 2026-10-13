"""Phase 3 configuration - EVERY tunable of every `exopipe3` step, in one file.

How to use
----------
- Read:      ``from exopipe.phase3.config import CFG``; ``CFG.search.V_FLOOR_SIGMA``.
- Inspect:   ``exopipe3 config`` lists every section; ``exopipe3 config --section search`` prints that section
             with its comments (unit and ruling of every value).
- Override:  per run, on any step: ``--set search.V_FLOOR_SIGMA=4.5`` (repeatable) or ``--config my_overrides.py``
             (a python file of ``section.KEY = value`` lines). Every override is written into the run record
             and the products' HISTORY. Changing a ruled value without Gregg's ruling is a conformance breach;
             the run record makes it visible.

Conventions
-----------
- One class per step (``masters``, ``search``, ...) plus shared sections (``instrument``, ``paths``,
  ``exclusions``, ``astrometry``, ``products``, ``qa``, ``runner``, ``site``).
- Every value carries its unit (in the name or the comment) and the ledger ID of the ruling that set it.
- Tables that are data (scale tables, bright-parent tables, dewarp acceptance) are referenced by versioned path,
  never edited in place.
- Host names, node pools, GPU hosts and python environments are NOT here: they live in the site file named by
  the environment variable ``EXOPIPE3_SITE`` (ops, not git), loaded into ``CFG.site`` below. No production
  module names a host (PHASE3_ARCHITECTURE.md 0.1).
- Library modules read ``CFG`` when they are imported; the CLI applies ``--set`` / ``--config`` overrides
  before it imports any step, so overrides reach every module.
"""
import ast
import hashlib
import importlib.util
import inspect
import json
import os


class _Section:
    """Base class of a config section: UPPER_CASE class attributes are the tunables."""

    @classmethod
    def items(cls):
        return [(k, getattr(cls, k)) for k in sorted(vars(cls)) if k.isupper()]


# =====================================================================================================================
# Shared sections
# =====================================================================================================================

class instrument(_Section):
    """OVRO-LWA instrument constants used by Phase 3 (authoritative values live in common/phase2_config.py;
    architecture 0.1). Imported, not copied, so Phase 2 and Phase 3 cannot disagree."""
    from exopipe.common import phase2_config as _p2
    ALL_SUBBANDS = tuple(_p2.ALL_SUBBANDS)            # subband labels, nominal ('18MHz' ... '82MHz')
    SUBBAND_FREQ_HZ = dict(_p2.SUBBAND_FREQ_HZ)       # Hz, exact subband centres (label -> Hz)
    SUBBAND_BANDS = {k: tuple(v) for k, v in _p2.SUBBAND_BANDS.items()}   # band label -> member subbands
    INTEGRATION_S = _p2.LWA_DT                        # s, one integration
    del _p2


class paths(_Section):
    """Roots of inputs and products. A step's --out / --work flags override the product and scratch roots
    for one run (parity and scratch runs never write into these)."""
    PHASE1_ROOT = '/lustre/pipeline/exopipe/phase1'
    PHASE2_ROOT = '/lustre/pipeline/exopipe/phase2'
    PHASE3_ARCHIVE_ROOT = '/lustre/pipeline/exopipe/phase3'
    DATA_DECISIONS = '/lustre/gh/main/clv15/data_decisions.csv'    # Gregg's exclusions (read only; core/exclusions)
    REFERENCE_DIR = '/lustre/gh/main/exopipe/reference'            # small publishable reference CSVs
    # Product roots that hold PRODUCTION data. While PRODUCTION_WRITES is False (the refactor window, until each
    # step's parity run passes and Gregg approves its cutover) a step refuses to write under any of them: parity and
    # scratch runs pass --out to their own roots.
    PRODUCTION_ROOTS = ('/lustre/pipeline/exopipe/phase3', '/data07/gh/phase3', '/lustre/gh/main/phase3')
    PRODUCTION_WRITES = False


class runner(_Section):
    """Defaults for `exopipe3 run` and for every step's process environment."""
    from exopipe.phase3.core import env as _env
    ENV_DEFAULTS = dict(_env.ENV_DEFAULTS)       # applied by exopipe3 before numpy loads (core/env.py explains why)
    del _env
    FREE_CORES_CMD = 'bash /lustre/gh/main/exopipe_ops/node_free_cores.sh'   # prints the recorder-free core list
    JOBS_ROOT = '/lustre/gh/main/exopipe_ops/jobs'                            # job records of `exopipe3 run`
    SSH_PREFIX_ENV = {'SSH_AUTH_SOCK': ''}                                     # ssh from lwacalim07 (CLAUDE.md 8)
    POLL_S = 30                                                                # s, --after / status polling


class site(_Section):
    """Filled from the site file (EXOPIPE3_SITE). Expected names (all optional; a step that needs one refuses
    with a clear message when it is unset): NODE_POOLS {pool: [host, ...]}, GPU_HOSTS [host, ...],
    NODE_LOCAL_ROOTS {host: [root, ...]}, PYTHON {'default': path, 'flow': path}, SCRATCH_ROOT {host: path}."""
    NODE_POOLS = {}
    GPU_HOSTS = []
    NODE_LOCAL_ROOTS = {}
    PYTHON = {}
    SCRATCH_ROOT = {}


# =====================================================================================================================
# Step sections (one per command, in pipeline order). Populated from the code each step absorbs; every value keeps
# the default the production code had at the refactor baseline and names its ruling.
# =====================================================================================================================

# @@STEP-SECTIONS@@


# =====================================================================================================================
# Machinery: the CFG namespace, overrides, hash, printing. Nothing below is a tunable.
# =====================================================================================================================

class _Config:
    """The namespace of all sections: CFG.<section>.<KEY>."""

    def __init__(self, sections):
        self._sections = dict(sections)
        for name, sec in sections.items():
            setattr(self, name, sec)
        self.overrides = []          # [(dotted key, old value, new value, source)]

    def sections(self):
        return dict(self._sections)

    def get(self, dotted):
        sec, key = _split(dotted)
        return getattr(self._section(sec), key)

    def set(self, dotted, value, source='--set'):
        sec, key = _split(dotted)
        s = self._section(sec)
        if not key.isupper() or not hasattr(s, key):
            raise KeyError(f'unknown config key {dotted!r} (exopipe3 config --section {sec} lists the keys)')
        old = getattr(s, key)
        new = _coerce(value, old) if isinstance(value, str) else value
        setattr(s, key, new)
        self.overrides.append((dotted, old, new, source))

    def apply_override_strings(self, items, source='--set'):
        for it in items or []:
            if '=' not in it:
                raise ValueError(f'--set expects SECTION.KEY=VALUE, got {it!r}')
            k, v = it.split('=', 1)
            self.set(k.strip(), v.strip(), source)

    def apply_override_file(self, path):
        """A python file of `section.KEY = value` lines (literals only)."""
        for n, line in enumerate(open(path), 1):
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            if '=' not in line:
                raise ValueError(f'{path}:{n}: expected section.KEY = value')
            k, v = line.split('=', 1)
            self.set(k.strip(), v.strip(), f'{path}:{n}')

    def snapshot(self):
        return {name: {k: _jsonable(v) for k, v in sec.items()} for name, sec in self._sections.items()
                if name != 'site'}

    def hash(self, sections=None):
        snap = self.snapshot()
        if sections is not None:
            snap = {k: v for k, v in snap.items() if k in sections}
        return hashlib.sha1(json.dumps(snap, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def describe(self, section=None):
        if section is None:
            out = []
            for name, sec in self._sections.items():
                doc = (inspect.getdoc(sec) or '').splitlines()
                out.append(f'{name:14s} {len(sec.items()):4d} keys  {doc[0] if doc else ""}')
            return '\n'.join(out)
        sec = self._section(section)
        try:
            return inspect.getsource(sec)
        except (OSError, TypeError):
            return '\n'.join(f'{k} = {v!r}' for k, v in sec.items())

    def _section(self, name):
        if name not in self._sections:
            raise KeyError(f'unknown config section {name!r}; sections: {", ".join(self._sections)}')
        return self._sections[name]


def _split(dotted):
    if '.' not in dotted:
        raise KeyError(f'config key must be SECTION.KEY, got {dotted!r}')
    return dotted.split('.', 1)


def _coerce(text, old):
    """Parse an override string to the type of the current value."""
    if isinstance(old, bool):
        t = text.lower()
        if t in ('1', 'true', 'yes', 'on'):
            return True
        if t in ('0', 'false', 'no', 'off'):
            return False
        raise ValueError(f'expected a boolean, got {text!r}')
    if isinstance(old, str) or old is None:
        try:
            v = ast.literal_eval(text)
            return v if (old is None or isinstance(v, str)) else text
        except (ValueError, SyntaxError):
            return text
    v = ast.literal_eval(text)
    if isinstance(old, float) and isinstance(v, int):
        v = float(v)
    if isinstance(old, tuple) and isinstance(v, list):
        v = tuple(v)
    if type(v) is not type(old):
        raise ValueError(f'expected {type(old).__name__}, got {text!r}')
    return v


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_jsonable(x) for x in v]
    return repr(v)


def _load_site(cfg):
    path = os.environ.get('EXOPIPE3_SITE')
    if not path:
        return None
    spec = importlib.util.spec_from_file_location('exopipe3_site', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for k in dir(mod):
        if k.isupper():
            setattr(cfg.site, k, getattr(mod, k))
    return path


def _collect():
    secs = {}
    for name, obj in list(globals().items()):
        if inspect.isclass(obj) and issubclass(obj, _Section) and obj is not _Section and not name.startswith('_'):
            secs[name] = obj
    return secs


CFG = _Config(_collect())
SITE_FILE = _load_site(CFG)
