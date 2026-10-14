"""Phase 3 configuration - EVERY tunable of every `exopipe3` step, in this package.

Where things are
----------------
  config/shared.py         the sections every step reads: instrument, paths (and PROTECTED), exclusions, astrometry,
                           fitting, runner, site (site = filled from the site file EXOPIPE3_SITE)
  config/steps/frames.py    `exopipe3 frames`
  config/steps/flow.py      `exopipe3 flow`
  config/steps/search.py    `exopipe3 search`
  config/steps/catalog.py   `exopipe3 catalog`
  config/steps/forced.py    `exopipe3 forced`
  config/steps/curves.py    `exopipe3 curves`
  config/steps/pages.py     `exopipe3 pages`
  config/steps/serve.py     `exopipe3 serve`
  config/steps/archive.py   `exopipe3 archive`
  config/steps/masters.py   `exopipe3 masters`
  config/steps/dewarp.py    `exopipe3 dewarp`
  config/steps/install.py   `exopipe3 install`
  config/steps/calibrate.py `exopipe3 calibrate`
  config/steps/mosaic.py    `exopipe3 mosaic`
  config/steps/photometry.py `exopipe3 photometry`
  config/steps/followup.py  `exopipe3 followup`
  config/steps/jupiter.py   `exopipe3 jupiter`
  config/steps/status.py    `exopipe3 status`
  config/__init__.py       this file: the machinery (CFG, overrides, hash, describe) - nothing here is a tunable
A key listed in its section's PROTECTED_KEYS (or in shared.PROTECTED) is locked: a mirror of in-place code, a pinned input
or a ruling-fixed value - it changes only by a commit (`exopipe3 config --section S` marks them).

How to use
----------
- Read:      ``from exopipe.phase3.config import CFG``; ``CFG.search.V_FLOOR_SIGMA``.
- Inspect:   ``exopipe3 config`` lists every section; ``exopipe3 config --section search`` prints that section
             with its comments (unit and ruling of every value).
- Override:  per run, on any step: ``--set search.V_FLOOR_SIGMA=4.5`` (repeatable) or ``--config my_overrides.py``
             (a python file of ``section.KEY = value`` lines). Every override is written into the run record
             and the products' HISTORY.
- One entry: ``--set fitting.K_TEMPLATE[sublst,V]=(0.66,0.50)`` changes ONE entry of a dict (tuple keys written
             comma-separated, nested dicts with one bracket per level); a whole-dict override that omits any key
             of the current value is refused (it would silently delete the omitted entries).
- Protected: ``PROTECTED`` (config/shared.py) (the exclusion file and rule parameters, the production-root guard, the ledger,
             pinned inputs) - no override can change them; change them in this file, by a commit.
- Ruled:     a section's ``RULED = {'KEY': 'ledger ID'}`` names the keys a ruling of Gregg's set. Overriding one needs
             ``--ruling-for section.KEY=<ledger ID>`` (the ID must be a row of ``paths.FIX_LEDGER``) or a ``--out``
             outside every production root; the override and its authority go into the run record.

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
import importlib
import importlib.util
import inspect
import json
import os
import re


_META = ('RULED', 'PROTECTED_KEYS')      # UPPER_CASE section attributes that DESCRIBE tunables; never tunables themselves


class _Section:
    """Base class of a config section: UPPER_CASE class attributes are the tunables. ``RULED`` (optional) maps a key to
    the ledger ID(s) of the ruling that set it."""

    @classmethod
    def items(cls):
        return [(k, getattr(cls, k)) for k in sorted(vars(cls)) if k.isupper() and k not in _META]


# The settings: shared.py (every step) and one module per command (STEP_CONFIG_MODULES), all in this package. They import
# _Section from here, so they are imported after it.
from exopipe.phase3.config.shared import (PROTECTED, astrometry, exclusions, fitting, instrument, paths,  # noqa: E402
                                          runner, site)









# ---------------------------------------------------------------------------------------------------------------------
# Data roots and the Phase 3 directory layout of the pre-refactor package, absorbed VERBATIM (refactor 2026-10-13,
# MOVE_PLAN.md 2.0) from exopipe/common/roots.py and exopipe/common/phase3_paths.py. Module-level names, not a section:
# they read the environment exactly as before (required_root / coadd_root at call time, STAGING_ROOT at import).
#
# From exopipe/common/roots.py - "Canonical data roots - REQUIRED from the environment, no defaults.
# Gregg 2026-09-04: the superseded sigma-clipped co-add tree (/fast/gh/main/phase3/coadd) was deleted and every module
# default that pointed at it was removed ("remove the defaults"). A run that does not state its co-add root explicitly
# must fail loudly here, never fall through to a stale generation." (R-COADD-ROOT-NO-DEFAULTS)
# ---------------------------------------------------------------------------------------------------------------------

class RootUnset(RuntimeError):
    pass


def required_root(var, purpose):
    v = os.environ.get(var)
    if not v:
        raise RootUnset(
            f"{var} is not set ({purpose}). No default exists: the retired "
            f"/fast/gh/main/phase3/coadd tree was deleted 2026-09-04 and module "
            f"defaults were removed on Gregg's order. Export {var} to the "
            f"current-generation cell root (wideband: rebuild_w2/rebuild_v2 "
            f"on the node's /fast; subband: /data07/gh/phase3/coadd_sb).")
    return v


def coadd_root():
    """Root holding <lst>_<scope>/coadd_final/ and <lst>_<scope>/subtracted/."""
    return required_root('P3_COADD_ROOT', 'co-add cell root: masters + subtracted epochs')


# From exopipe/common/phase3_paths.py - "canonical Phase 3 directory layout (PHASE3_STRUCTURE.md).
# Staging on /fast MIRRORS the final /lustre tree so promotion is a plain copy. One builder for every write site;
# nothing constructs paths ad hoc.
#     {root}/{LST}/{date}/P3_{label}/{scope}/{Stokes}/{cadence}/    per-epoch
#     {root}/{LST}/coadd/Run_{label}_{lo}-{hi}/{scope}/{Stokes}/{cadence}/   masters
#     {root}/transients/Run_{label}/...                            survey-wide blind search
#     {root}/samples/{sample}/{target}/                            targeted forced photometry
# scope in {18MHz...82MHz, Red, Green, Blue, Full}; Stokes in {I, V}; cadence in {deep, 10min}"

STAGING_ROOT = os.environ.get('P3_STAGING_ROOT',
                              '/data07/gh/phase3')  # conformant FINAL products (lwacalim07;
                                                # /fast too small). Flat working staging
                                                # (WORKDIR_BASE) stays on /fast for I/O.
                                                # P3_STAGING_ROOT: staged-node override
                                                # (lwacalim01/02 have no /data07).
ARCHIVE_ROOT = '/lustre/pipeline/exopipe/phase3'  # promotion target


def epoch_dir(root, lst, date, label, scope, stokes, cadence, make=False):
    """Per-epoch product dir: aligned+scaled variants + final subtracted images."""
    p = os.path.join(root, lst, date, f'P3_{label}', scope, stokes, cadence)
    if make:
        os.makedirs(p, exist_ok=True)
    return p


def coadd_dir(root, lst, label, date_lo, date_hi, scope, stokes, cadence,
              make=False):
    """Co-add Run dir (sibling of date dirs; spans dates; date range in name)."""
    run = f'Run_{label}_{date_lo}-{date_hi}'
    p = os.path.join(root, lst, 'coadd', run, scope, stokes, cadence)
    if make:
        os.makedirs(p, exist_ok=True)
    return p


def coadd_run_root(root, lst, label, date_lo, date_hi, make=False):
    """The co-add Run root (holds QA csvs spanning scopes/Stokes)."""
    p = os.path.join(root, lst, 'coadd', f'Run_{label}_{date_lo}-{date_hi}')
    if make:
        os.makedirs(p, exist_ok=True)
    return p


def transients_run(root, label, make=False):
    """Survey-wide blind-search Run (spans ALL LSTs+dates; de-siloed)."""
    p = os.path.join(root, 'transients', f'Run_{label}')
    if make:
        os.makedirs(p, exist_ok=True)
    return p


def candidate_dir(root, label, cand_id, make=False):
    p = os.path.join(transients_run(root, label), 'candidates', cand_id)
    if make:
        os.makedirs(p, exist_ok=True)
    return p


def sample_dir(root, sample, target, make=False):
    """Targeted forced-photometry per (sample, target). sample e.g.
    solar_system, 10pc, Hot_Warm_Jupiters, Local_Volume."""
    safe = target.replace(' ', '_')
    p = os.path.join(root, 'samples', sample, safe)
    if make:
        os.makedirs(p, exist_ok=True)
    return p














# =====================================================================================================================
# Machinery: the CFG namespace, overrides, hash, printing. Nothing below is a tunable.
# =====================================================================================================================

class ConfigRefusal(ValueError):
    """An override the config refuses: a protected key, a ruled key without a ruling, an unknown ledger ID, a whole-dict
    override that omits entries. The CLI prints it and exits 2."""


class _Config:
    """The namespace of all sections: CFG.<section>.<KEY>."""

    def __init__(self, sections):
        self._sections = dict(sections)
        for name, sec in sections.items():
            setattr(self, name, sec)
        # [(dotted key, old value, new value, source, authority)]; authority is None for an unruled key, else
        # 'ruling:<ledger ID>', 'non-production out:<dir>', 'display only' or 'forwarded (checked by the step)'
        self.overrides = []
        self._undo = []              # [(section, key, previous value)] for reset()
        self._auth = {'rulings': {}, 'out': None, 'mode': 'run'}

    def sections(self):
        return dict(self._sections)

    def get(self, dotted):
        sec, key, path = _parse_key(dotted)
        v = getattr(self._section(sec), key)
        for toks in path:
            v = v[_entry_key(v, toks, dotted)]
        return v

    # ---- protected and ruled keys -------------------------------------------------------------------------------
    def protected(self, dotted):
        """True when no override may change ``section.KEY`` (PROTECTED, 'section.*', paths.PINNED_INPUTS)."""
        sec, key, _ = _parse_key(dotted)
        base = f'{sec}.{key}'
        pinned = tuple(getattr(self._sections.get('paths'), 'PINNED_INPUTS', ()) or ())
        own = tuple(getattr(self._sections.get(sec), 'PROTECTED_KEYS', ()) or ())   # a section's own protected keys
        return base in PROTECTED or f'{sec}.*' in PROTECTED or base in pinned or key in own

    def ruled(self, dotted):
        """The ledger ID(s) of the ruling that set ``section.KEY`` (from the section's RULED), or None."""
        sec, key, _ = _parse_key(dotted)
        r = (getattr(self._section(sec), 'RULED', None) or {}).get(key)
        if r is None:
            return None
        return tuple(r) if isinstance(r, (list, tuple)) else (r,)

    def production_root_of(self, path):
        """The production root (paths.PRODUCTION_ROOTS) that holds ``path``, or None."""
        real = os.path.realpath(path)
        for root in self.paths.PRODUCTION_ROOTS:
            r = os.path.realpath(root)
            if real == r or real.startswith(r.rstrip('/') + '/'):
                return root
        return None

    def authorize(self, rulings=(), out=None, mode='run'):
        """How ruled keys may be overridden in this process (the CLI calls it before applying overrides).

        rulings: ['section.KEY=LEDGER_ID', ...] (--ruling-for) or a dict; every ID must be a row of paths.FIX_LEDGER.
        out:     the run's product root; outside every production root, a ruled key may be overridden without a ruling.
        mode:    'run' (check), 'display' (`exopipe3 config`: shown, nothing written), 'forward' (`run submit`: the
                 overrides are forwarded to each step process, which checks them against its own --out)."""
        if isinstance(rulings, dict):
            items = list(rulings.items())
        else:
            items = []
            for r in rulings or ():
                if '=' not in r:
                    raise ConfigRefusal(f'--ruling-for expects SECTION.KEY=LEDGER_ID, got {r!r}')
                k, v = r.split('=', 1)
                items.append((k.strip(), v.strip()))
        from exopipe.phase3.core import ledger
        for k, lid in items:
            _parse_key(k)
            try:
                ok = ledger.has_id(lid, self.paths.FIX_LEDGER)
            except OSError as e:
                raise ConfigRefusal(f'--ruling-for {k}={lid}: the ledger {self.paths.FIX_LEDGER} is not readable ({e})')
            if not ok:
                raise ConfigRefusal(f'--ruling-for {k}={lid}: no row with fix_id {lid!r} in {self.paths.FIX_LEDGER}')
        self._auth = {'rulings': dict(items), 'out': out, 'mode': mode}

    def _authority(self, dotted, base):
        ids = self.ruled(base)
        if not ids:
            return None
        lid = self._auth['rulings'].get(dotted) or self._auth['rulings'].get(base)
        if lid:
            return f'ruling:{lid}'
        if self._auth['mode'] == 'display':
            return 'display only'
        if self._auth['mode'] == 'forward':
            return 'forwarded (checked by the step)'
        out = self._auth['out']
        if out and self.production_root_of(out) is None:
            return f'non-production out:{os.path.abspath(out)}'
        raise ConfigRefusal(
            f'{base} carries a ruling ({", ".join(ids)}): override it only with --ruling-for {base}=<ledger ID> '
            f'(a row of {self.paths.FIX_LEDGER}) or into a --out outside every production root '
            f'(this run writes to {out or "the step production root"})')

    def check_out(self, out):
        """Re-check, once the run's final --out is known, every ruled override that was allowed for a non-production
        out (a step whose default root depends on an override could otherwise move it into production)."""
        for o in self.overrides:
            if o[4] and o[4].startswith('non-production out:') and self.production_root_of(out) is not None:
                raise ConfigRefusal(f'{o[0]} carries a ruling and was overridden without one, but this run writes '
                                    f'under the production root {self.production_root_of(out)} ({out})')

    # ---- overrides ----------------------------------------------------------------------------------------------
    def set(self, dotted, value, source='--set'):
        """Override one tunable: 'section.KEY' (whole value) or 'section.KEY[entry]...' (one entry of a dict)."""
        sec, key, path = _parse_key(dotted)
        s = self._section(sec)
        if not key.isupper() or key in _META or not hasattr(s, key):
            raise KeyError(f'unknown config key {dotted!r} (exopipe3 config --section {sec} lists the keys)')
        base = f'{sec}.{key}'
        if self.protected(base) and not getattr(self, 'parity_replay', False):
            raise ConfigRefusal(f'{base} is PROTECTED (config.PROTECTED / paths.PINNED_INPUTS): no --set or --config '
                                f'can change it; it changes only by a commit to exopipe/phase3/config.py (a --parity run '
                                f'into a non-production --out may replay a historical value)')
        if self.protected(base):
            # PARITY REPLAY: a shadow of a production run must see the inputs that run saw (e.g. the exclusion list of
            # that moment). Allowed only under --parity (every write confined to the parity root by core/guard) and
            # a non-production --out; recorded with this authority so no product of such a run can be installed.
            authority = 'parity replay: protected key overridden inside a --parity run (writes confined to --out)'
        else:
            authority = self._authority(dotted, base)
        top = getattr(s, key)
        if path:
            new_top, old, new = _set_entry(top, path, value, dotted)
        else:
            old = top
            new = _coerce(value, old) if isinstance(value, str) else _like(value, old)
            if isinstance(old, dict):
                _refuse_omitted(old, new, dotted)
            new_top = new
        setattr(s, key, new_top)
        self._undo.append((s, key, top))
        self.overrides.append((dotted, old, new, source, authority))

    def reset(self):
        """Undo every override applied in this process (tests; `exopipe3 config`)."""
        while self._undo:
            s, key, prev = self._undo.pop()
            setattr(s, key, prev)
        self.overrides = []

    def ruled_overrides(self):
        """[{key, ruled_by, authority}] for every override of a ruled key (written into each run record)."""
        out = []
        for o in self.overrides:
            ids = self.ruled(o[0])
            if ids:
                out.append({'key': o[0], 'ruled_by': list(ids), 'authority': o[4]})
        return out

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
    sec, key, _ = _parse_key(dotted)
    return sec, key


_KEY_RE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)((?:\[[^\[\]]*\])*)\s*$')


def _parse_key(dotted):
    """'section.KEY[a,b][c]' -> ('section', 'KEY', [['a', 'b'], ['c']])."""
    m = _KEY_RE.match(dotted or '')
    if not m:
        raise KeyError(f'config key must be SECTION.KEY or SECTION.KEY[entry], got {dotted!r}')
    path = [[t.strip().strip('\'"') for t in sub.split(',')] for sub in re.findall(r'\[([^\[\]]*)\]', m.group(3))]
    return m.group(1), m.group(2), path


def _entry_key(d, toks, dotted):
    """The existing key of dict ``d`` written as ``toks`` ('deep,I' -> ('deep', 'I'); '75,90' -> (75, 90))."""
    if not isinstance(d, dict):
        raise KeyError(f'{dotted}: {type(d).__name__} has no entries (only dicts take [entry])')
    for k in d:
        if isinstance(k, tuple):
            if len(k) == len(toks) and all(str(x) == t for x, t in zip(k, toks)):
                return k
        elif len(toks) == 1 and str(k) == toks[0]:
            return k
    raise KeyError(f'{dotted}: no entry [{",".join(toks)}]; entries: '
                   + ', '.join('[' + (','.join(map(str, k)) if isinstance(k, tuple) else str(k)) + ']' for k in d))


def _set_entry(d, path, value, dotted):
    """Copy of dict ``d`` with the entry at ``path`` replaced; returns (new dict, old entry, new entry)."""
    k = _entry_key(d, path[0], dotted)
    new_d = dict(d)
    if len(path) > 1:
        new_d[k], old, new = _set_entry(d[k], path[1:], value, dotted)
        return new_d, old, new
    old = d[k]
    new = _coerce(value, old) if isinstance(value, str) else _like(value, old)
    if isinstance(old, dict):
        _refuse_omitted(old, new, dotted)
    new_d[k] = new
    return new_d, old, new


def _refuse_omitted(old, new, dotted):
    if not isinstance(new, dict):
        return
    missing = [k for k in old if k not in new]
    if missing:
        raise ConfigRefusal(f'whole-dict override of {dotted} omits {len(missing)} of its {len(old)} entries '
                            f'({", ".join(map(repr, missing[:6]))}{" ..." if len(missing) > 6 else ""}); give every '
                            f'entry, or change one entry with {dotted}[entry]=value')


def _like(v, old):
    """Normalise a parsed value to the shape of the current one (list -> tuple, int -> float), recursively."""
    if isinstance(old, bool) or v is None or old is None:
        return v
    if isinstance(old, float) and isinstance(v, int) and not isinstance(v, bool):
        return float(v)
    if isinstance(old, tuple) and isinstance(v, (list, tuple)):
        if len(v) == len(old):
            return tuple(_like(x, o) for x, o in zip(v, old))
        return tuple(v)
    if isinstance(old, dict) and isinstance(v, dict):
        return {k: (_like(x, old[k]) if k in old else x) for k, x in v.items()}
    return v


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
    v = _like(ast.literal_eval(text), old)
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
    cfg.site_sha1 = hashlib.sha1(open(path, 'rb').read()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location('exopipe3_site', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for k in dir(mod):
        if k.isupper():
            setattr(cfg.site, k, getattr(mod, k))
    return path


STEP_CONFIG_MODULES = ('frames', 'flow', 'search', 'catalog', 'forced', 'curves', 'pages', 'serve', 'archive', 'masters', 'dewarp', 'install', 'calibrate', 'mosaic', 'photometry', 'followup', 'jupiter', 'status')   # exopipe/phase3/config/steps/<step>.py, pipeline order


def _load_step_configs():
    """The sections of every command: config/steps/<step>.py (one module per step). A section name is defined once."""
    for m in STEP_CONFIG_MODULES:
        mod = importlib.import_module(f'exopipe.phase3.config.steps.{m}')
        for k, v in vars(mod).items():
            if inspect.isclass(v) and issubclass(v, _Section) and v is not _Section and not k.startswith('_') \
                    and v.__module__ == mod.__name__:
                if k in globals() and globals()[k] is not v:
                    raise RuntimeError(f'config/steps/{m}.py redefines section {k!r}')
                globals()[k] = v


def _collect():
    secs = {}
    for name, obj in list(globals().items()):
        if inspect.isclass(obj) and issubclass(obj, _Section) and obj is not _Section and not name.startswith('_'):
            secs[name] = obj
    return secs


_load_step_configs()
CFG = _Config(_collect())
SITE_FILE = _load_site(CFG)

# Sections every step reads, whatever it declares: they are always part of a step's resume hash (critique_rulings 6.5:
# a per-section hash must not miss the shared sections).
SHARED_SECTIONS = ('instrument', 'paths', 'exclusions', 'astrometry', 'fitting')


def declared_sections(mod):
    """The sections a step module declares in ``CONFIG`` (a tuple of section names), or None when it declares none.
    An unknown section name raises KeyError."""
    decl = getattr(mod, 'CONFIG', None)
    if decl is None:
        return None
    decl = (decl,) if isinstance(decl, str) else tuple(decl)
    unknown = [s for s in decl if s not in CFG.sections()]
    if unknown:
        raise KeyError(f'step {getattr(mod, "__name__", mod)} declares unknown config section(s) {unknown} in CONFIG; '
                       f'sections: {", ".join(CFG.sections())}')
    return decl


def step_sections(mod):
    """The sections whose values make a step's config hash (DESIGN 11.4): its CONFIG plus SHARED_SECTIONS, or None
    (every section) for a step that declares no CONFIG."""
    decl = declared_sections(mod)
    if decl is None:
        return None
    return tuple(dict.fromkeys(decl + SHARED_SECTIONS))
