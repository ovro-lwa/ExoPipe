"""Window machinery of `exopipe3 masters --tier deep`: how config reaches the IN-PLACE masters modules, and the gates that
guard that hand-over (DESIGN 11.1: during the refactor window the step calls the git-mv'd lib/masters modules and the
exopipe_ops scripts in place; nothing is copied). The deep tier calls them IN PROCESS (spec 7.2); a clv15 / exopipe_ops
script run as a CHILD goes through core/legacy.run (refactor decision 1) - the deep tier has none.

``WIRING`` (below) says, per config section and key, how the value reaches the in-place code (it is code knowledge, so it
lives here, not in the config):
  attr:<module>.<NAME>     module attribute set before the call (read at call time); a dict value with several names
                           (attr:night_image_gate.THRESHOLD,MAX_SHIFT_DEG,...) sets NAME = value[name.lower()]
  env-import:<VAR>         environment variable read when the module is FIRST imported (set by the step before import)
  env-call:<VAR>           environment of the in-place call (bool -> '1' / '0')
  arg                      an argument of the in-place call
  step                     used only by steps/masters.py / lib/masters/deep_*.py / tests.py (wrapper, gates, content tests)
  mirror                   a literal / a default argument bound inside the in-place code: the config value documents it.
                           Every mirror key is in its section's PROTECTED_KEYS, so config.py refuses an override (exit 2)
                           until the module is absorbed into lib/ and reads config (test_masters-deep checks the pairing).

- ``apply_attrs(section_name, modules)``: every ``attr:<module>.<NAME>`` key of a config section is set on the imported
  module before the call (read at call time by the in-place code).
- ``call_env(...)``: the ``env-call`` keys (and the fixed launcher exports of rebuild_cell.sh:26) as the environment of
  one in-place call, restored afterwards.
- ``set_import_env()`` / ``check_import_env()``: the ``env-import`` keys set BEFORE pb_apply is first imported (it reads
  them at import and binds them as default arguments); a module already imported with other values refuses.
- ``refuse_forbidden_overrides(cfg)``: belt and braces behind config.py's PROTECTED_KEYS - an applied override of a mirror
  key (one a section forgot to protect) refuses the run (exit 2); also honours the sub-LST/snapshot drafts' lower-case
  ``_apply`` mirror / ``_protected`` markers until those drafts move to PROTECTED_KEYS.
- ``clear_inherited_p3()``: P3_* variables inherited from the caller's shell are removed before the in-place modules are
  imported (they read P3_COADD_ROOT, P3_LOW_OUTLIER_SIGMA, P3_REF_MODE, P3_RESIDUAL_PB ... at import; core/legacy.run does
  the same for children); the step then sets exactly the config values.
- ``check_mirrors(...)``: where a mirror value is a module attribute of the in-place code, it must equal config.
- ``Tee``: the in-place co-add prints its verdicts and returns None on refusal (exit 0 in production); stdout is teed to
  the cell's log and kept, so the wrapper can copy the printed reason into the record. The VERDICT always comes from the
  return value or the exception, never from a grep of success text (spec 7.3).
- ``sha1_pin(path, want, what)``: a pinned table whose sha1 differs from config refuses (G-M11, G-M12, G-M23, G-M25).
- ``exclusion_readers_agree(...)``: core.exclusions and the co-add's own reader must agree on every night (G-M01).
"""
import collections
import contextlib
import hashlib
import io
import os
import sys

from exopipe.phase3.core.record import Refusal

MASTERS_SECTIONS = ('masters', 'masters_stagein', 'masters_build', 'masters_pbmaps', 'masters_vclean', 'masters_fluxscale',
                    'masters_verify', 'masters_sources')
_MODULE_PATHS = {'coadd': 'exopipe.phase3.lib.masters.coadd', 'membership': 'exopipe.phase3.lib.masters.membership',
                 'night_image_gate': 'exopipe.phase3.lib.masters.night_image_gate',
                 'pbcorr_masters': 'exopipe.phase3.lib.masters.pbcorr_masters',
                 'vclean_masters': 'exopipe.phase3.lib.masters.vclean_masters',
                 'shflux': 'exopipe.phase3.lib.masters.shflux', 'pb_apply': 'exopipe.phase3.lib.masters.pb_apply',
                 'epoch_gate': 'exopipe.phase3.lib.masters.epoch_gate', 'p3s5_parents': 'p3s5_parents'}
# the launcher exports of rebuild_cell.sh:26 that are not config keys of their own (P3_REF_MODE mirrors masters_build.REF_MODE;
# P3_ARCHIVE_RUN = paths.ARCHIVE_RUN; XLA_PYTHON_CLIENT_PREALLOCATE only matters to the GPU flow stage, exported by production
# for every co-add stage)
FIXED_CALL_ENV = {'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'}

WIRING = {
    'masters': {
        'TIERS': 'step', 'STAGES': 'step', 'DEFAULT_STAGES': 'step', 'DEEP_GEN_ROOT': 'step', 'PYTHON_SITE_KEY': 'step',
        'FREE_FLOOR_GB': 'step', 'EXTRA_PRODUCTION_ROOTS': 'step'},
    'masters_stagein': {
        'REF_PIXEL_FACTOR_RANGE': 'step', 'REF_OVERRIDE_TABLE': 'step', 'REF_OVERRIDE_TABLE_SHA1': 'step',
        'REF_OVERRIDE_I_TABLE': 'step', 'REF_PREVGEN_TABLE': 'step', 'REFERENCE_COPY_ROOTS': 'step',
        'PREVIOUS_GENERATION_REFERENCE': 'step', 'V_OPTIONAL_BRANCHES': 'step', 'PREDEWARP_QUARANTINE': 'step',
        'SUPERSEDED_ARCHIVE': 'step', 'EPOCH_GATE_CARRY_DIR': 'step', 'EPOCH_GATE_CARRY': 'step', 'STRIP_CARD_PREFIXES': 'step'},
    'masters_build': {
        'CUTOVER_DATE': 'attr:coadd.CUTOVER_DATE', 'FLOW_MAX_NAN_FRACTION': 'mirror',
        'ALLOW_STALE_FLOW': 'env-call:P3_ALLOW_STALE_FLOW', 'RAW_INPUT_FLOW_RULE': 'mirror',
        'PROVISIONAL_V_GRID_MISMATCH': 'mirror', 'REF_MODE': 'attr:coadd.REF_MODE',
        'REFERENCE_DATES': 'attr:coadd.REFERENCE_DATES', 'REF_ASTROM_MAX_ARCMIN': 'attr:coadd.REF_ASTROM_MAX_AMIN',
        'REF_ASTROM_PIVOT_MHZ': 'mirror', 'REF_ASTROM_EXPONENT': 'mirror', 'REF_ASTROM_MIN_ANCHORS': 'mirror',
        'REF_ASTROM_N_PEAKS': 'mirror', 'REF_ASTROM_MATCH_MAX_ARCMIN': 'mirror', 'VLSSR_PEAKS': 'attr:coadd.VLSSR_REF',
        'FLUXSCALE_ESTIMATOR': 'mirror', 'FLUXSCALE_SNR_MIN': 'mirror', 'FLUXSCALE_PEAK_JY': 'mirror',
        'FLUXSCALE_MIN_ELEV_DEG': 'mirror', 'FLUXSCALE_APERTURE_BEAMS': 'mirror', 'FLUXSCALE_MAXFILTER_BEAMS': 'mirror',
        'FLUXSCALE_COMPACTNESS': 'mirror', 'FLUXSCALE_MIN_SOURCES': 'mirror', 'FLUXSCALE_IRLS': 'mirror',
        'FLUXSCALE_TILT': 'env-call:P3_FLUXSCALE_TILT', 'DEGENERACY': 'mirror', 'SECOND_PASS': 'mirror',
        'SAFE_SCALE_FLOOR': 'mirror', 'STAGE2_METRIC': 'attr:membership.STAGE2_METRIC', 'STAGE2_MARGIN_FRAC': 'mirror',
        'STAGE2_CLIP_SIGMA': 'mirror', 'OUTLIER_SIGMA': 'attr:coadd.OUTLIER_SIGMA',
        'LOW_OUTLIER_SIGMA': 'attr:coadd.LOW_OUTLIER_SIGMA', 'MAD_MIN_VALUES': 'mirror', 'BEAM_GATE_RATIO': 'mirror',
        'BEAM_GATE_MIN_AREAS': 'mirror', 'BEAM_GATE_IMAGE': 'mirror', 'PROVISIONAL_SIGCLIP': 'mirror',
        'LEGACY_STAGE2_RECORD': 'mirror', 'EPOCH_GATE_SOURCE': 'step', 'EPOCH_GATE_MAX_PASSES': 'step', 'SOLAR_AUDIT': 'attr:membership.SOLAR_AUDIT',
        'SOLAR_AUDIT_SHA1': 'step', 'NAMED_OUT_OF_MASTER': 'attr:membership.NAMED_OUT_OF_MASTER',
        'NAMED_OUT_OF_MASTER_SHA1': 'step', 'QA3_HI': 'attr:membership.QA3_HI', 'QA3_LO': 'attr:membership.QA3_LO',
        'QA3_LO_DEFAULT': 'attr:membership.QA3_LO_DEFAULT',
        'NIGHT_GATE': 'attr:night_image_gate.THRESHOLD,MAX_SHIFT_DEG,BLOWUP_JY,BLOCK,RADIUS',
        'MIN_DATES_FOR_COADD': 'attr:coadd.MIN_DATES_FOR_COADD',
        'MIN_DATES_FOR_VARIANT': 'attr:coadd.MIN_DATES_FOR_VARIANT', 'VARIANT_MAX_ABS_JY': 'mirror',
        'VARIANT_MIN_FINITE': 'mirror', 'VARIANT_REL_K': 'mirror', 'VARIANT_BAND_MEDIANS': 'mirror',
        'VARIANT_BAND_MEDIANS_SHA1': 'step', 'VARIANT_FLUXDC_LEGACY': 'mirror', 'VARIANTS_ONLY': 'arg',
        'COADDOP': 'mirror', 'FLUXFIT': 'mirror', 'LOO_N_RANGE': 'mirror', 'STRIATION_WINDOW_BEAMS': 'mirror',
        'EPOCH_CHECK': 'mirror', 'JUPITER_MASK_DEEP': 'mirror', 'ARCHIVE_WRITE': 'env-call:P3_ARCHIVE_WRITE',
        'FORCE_REBUILD': 'env-call:P3_FORCE_REBUILD', 'KEEP_PREP': 'env-call:P3_KEEP_PREP',
        'MASTER_SPAN_SOURCE': 'attr:coadd.MASTER_SPAN_SOURCE'},
    'masters_pbmaps': {
        'PB_LIBRARY': 'env-import:P3_PB_LIBRARY', 'PB_THREE_FACTOR': 'env-import:P3_PB_THREE_FACTOR',
        'PB_CHAN_AVG_HZ': 'env-import:P3_CHAN_AVG_HZ', 'PB_CHROM_VERSION': 'mirror',
        'PB_RESIDUAL_V2_TABLE': 'env-import:P3_RESIDUAL_PB_V2', 'PB_FLOOR': 'mirror', 'PBLIBV': 'mirror',
        'BEAM_MODEL': 'attr:pbcorr_masters.BEAMMOD', 'PB_VARIANTS': 'attr:pbcorr_masters.VARIANTS',
        'PB_MAP_NAMES': 'mirror', 'CORRUPT_MAX_JY': 'mirror', 'SIBLING_TOL': 'mirror', 'SIBLINGS_AFTER_PBMAPS': 'step',
        'PB_VALUE_RANGE': 'step'},
    'masters_vclean': {
        'TIMEOUT_S': 'attr:vclean_masters.SV2_TIMEOUT_S', 'THREADS': 'mirror', 'SANE_MAX_JY': 'mirror',
        'I_MASTER_PREFERENCE': 'mirror', 'ENGINE_ELEV_DEG': 'mirror', 'ENGINE_ATEAM_PREBLANK': 'mirror'},
    'masters_fluxscale': {
        'SCALE_TABLE': 'step', 'SCALE_TABLE_SHA1': 'step', 'FLUXCORR_TABLE': 'step', 'FLUXCORR_TABLE_SHA1': 'step',
        'REQUIRED_FLUXVER': 'attr:membership.REQUIRED_FLUXVER', 'OUTGEN': 'arg', 'TOKEN': 'step',
        'SCALE_MODEL_FREQ_HZ': 'step', 'SCALED_DIRS': 'attr:shflux.SCALED_DIRS',
        'UNSCALED_PREFIX': 'attr:shflux.UNSCALED_PREFIX', 'CARD_TOL': 'step'},
    'masters_verify': {
        'T1_TOL': 'step', 'T1_PEAKS': 'step', 'T3_STOKES': 'step', 'T4_SIBLINGS': 'step', 'T8_IGNORE': 'step',
        'DEWARP_SCREEN_RECORD': 'attr:membership.DEWARP_ACCEPTED', 'DEWARP_SCREEN_RECORD_SHA1': 'step',
        'DEWARP_INTERP': 'attr:membership.DEWARP_INTERP', 'INTEGRITY': 'step'},
    'masters_sources': {
        'THRESH_ISL': 'mirror', 'THRESH_PIX': 'mirror', 'CATALOG_TYPE': 'mirror', 'BEAM': 'mirror',
        'BEAM_FALLBACK_PX': 'mirror', 'BDSF_NCORES': 'env-call:BDSF_NCORES', 'DEEP_COLLECT': 'step',
        'SUBLST_NPROC': 'arg', 'REQUIRE_DEWARP_SCREEN': 'step'},
}


def cfg():
    from exopipe.phase3.config import CFG
    return CFG


def section(name):
    return getattr(cfg(), name)


def apply_map(name):
    """The wiring of one section: WIRING (masters deep sections), else the draft's own lower-case _apply (tier drafts)."""
    if name in WIRING:
        return dict(WIRING[name])
    return dict(getattr(section(name), '_apply', {}) or {})


def protected_keys(name):
    s = section(name)
    return tuple(getattr(s, 'PROTECTED_KEYS', ()) or ()) + tuple(getattr(s, '_protected', ()) or ())


def clear_inherited_p3():
    """Remove every P3_* variable of the caller's environment (returns {name: value} removed). Called once, before the
    in-place modules are first imported; the step sets the config values itself (call_env, set_import_env)."""
    gone = {k: v for k, v in os.environ.items() if k.startswith('P3_')}
    for k in gone:
        os.environ.pop(k, None)
    return gone


# ---------------------------------------------------------------------------------------------------------------------
# overrides that the window cannot honour
# ---------------------------------------------------------------------------------------------------------------------

def _split_override_key(dotted):
    base = dotted.split('[', 1)[0]
    sec, _, key = base.partition('.')
    return sec, key


def forbidden_overrides(c=None):
    """[(dotted key, why)] for every applied override of a masters section that the window cannot honour: a key in the
    section's PROTECTED_KEYS (or a tier draft's lower-case _protected) or a mirror key (WIRING / a draft's _apply). config.py
    already refuses PROTECTED_KEYS before any step runs; this catches what a section forgot to protect."""
    c = c or cfg()
    bad = []
    for o in getattr(c, 'overrides', ()):
        sec, key = _split_override_key(o[0])
        if not sec.startswith('masters') or getattr(c, sec, None) is None:
            continue
        kind = apply_map(sec).get(key, '')
        replay = len(o) > 4 and isinstance(o[4], str) and o[4].startswith('parity replay')
        if key in protected_keys(sec) and not replay:   # config.py admits a protected key only as a --parity replay
            bad.append((o[0], f'{sec}.{key} is protected (pinned input / ruling-fixed / mirror value)'))
        elif kind == 'mirror':
            bad.append((o[0], f'{sec}.{key} is a MIRROR of a literal inside the in-place code: an override would be '
                              f'recorded but not applied - refused until the module is absorbed into lib/ and reads config'))
    return bad


def refuse_forbidden_overrides(c=None):
    bad = forbidden_overrides(c)
    if bad:
        raise Refusal('override refused: ' + '; '.join(f'{k}: {why}' for k, why in bad))


# ---------------------------------------------------------------------------------------------------------------------
# attributes, environment
# ---------------------------------------------------------------------------------------------------------------------

def _module(short):
    import importlib
    return importlib.import_module(_MODULE_PATHS.get(short, short))


def attr_targets(name):
    """[(key, module short name, [attribute names])] of the attr: keys of one section."""
    out = []
    for key, how in apply_map(name).items():
        if not how.startswith('attr:'):
            continue
        mod, _, attrs = how[5:].partition('.')
        out.append((key, mod, [a.strip() for a in attrs.split(',') if a.strip()]))
    return out


def apply_attrs(name, only_modules=None):
    """Set every attr: key of section ``name`` on its in-place module (imported here). Returns [(module.attr, value)]."""
    s = section(name)
    done = []
    for key, mod, attrs in attr_targets(name):
        if only_modules is not None and mod not in only_modules:
            continue
        m = _module(mod)
        val = getattr(s, key)
        if len(attrs) == 1:
            v = dict(val) if isinstance(val, dict) else (list(val) if isinstance(val, tuple) and mod == 'pbcorr_masters'
                                                         else val)
            setattr(m, attrs[0], v)
            done.append((f'{mod}.{attrs[0]}', v))
        else:                                   # one dict -> several module attributes (NIGHT_GATE)
            for a in attrs:
                if a.lower() not in val:
                    raise Refusal(f'config {name}.{key} has no entry {a.lower()!r} for {mod}.{a}')
                setattr(m, a, val[a.lower()])
                done.append((f'{mod}.{a}', val[a.lower()]))
    return done


def _env_str(v):
    if isinstance(v, bool):
        return '1' if v else '0'
    return str(v)


def call_env_values(sections):
    env = dict(FIXED_CALL_ENV)
    for name in sections:
        s = section(name)
        for key, how in apply_map(name).items():
            if how.startswith('env-call:'):
                env[how[len('env-call:'):]] = _env_str(getattr(s, key))
    if 'masters_build' in sections:
        env['P3_REF_MODE'] = str(cfg().masters_build.REF_MODE)      # rebuild_cell.sh:26 P3_REF_MODE=master
        env['P3_ARCHIVE_RUN'] = str(cfg().paths.ARCHIVE_RUN)        # rebuild_cell.sh:26 (the co-add refuses without it)
    return env


@contextlib.contextmanager
def call_env(sections, extra=None):
    """Environment of one in-place call: the env-call keys of ``sections`` (+ extra), restored afterwards."""
    env = call_env_values(sections)
    env.update(extra or {})
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield env
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def import_env_values():
    s = cfg().masters_pbmaps
    out = {}
    for key, how in apply_map('masters_pbmaps').items():
        if how.startswith('env-import:'):
            out[how[len('env-import:'):]] = _env_str(getattr(s, key))
    return out


def set_import_env():
    """Set the env-import keys before pb_apply is first imported (pb_apply.py:31, 57-60 read them at import)."""
    for k, v in import_env_values().items():
        os.environ[k] = v


def check_import_env():
    """pb_apply as imported must carry the config values (a module imported earlier with other values refuses)."""
    s = cfg().masters_pbmaps
    pb = _module('pb_apply')
    want = {'PB_DIR': s.PB_LIBRARY, 'THREE_FACTOR': bool(s.PB_THREE_FACTOR), 'CHAN_AVG_HZ': float(s.PB_CHAN_AVG_HZ),
            'RESIDUAL_V2_TABLE': s.PB_RESIDUAL_V2_TABLE}
    bad = [f'pb_apply.{k} = {getattr(pb, k, None)!r} != config {v!r}' for k, v in want.items() if getattr(pb, k, None) != v]
    if bad:
        raise Refusal('pb_apply was imported with other values than config (env-import keys): ' + '; '.join(bad))
    return want


# ---------------------------------------------------------------------------------------------------------------------
# mirror checks: a mirror key whose literal is a module attribute of the in-place code must equal config
# ---------------------------------------------------------------------------------------------------------------------

def mirror_pairs():
    """[(config 'section.KEY', module short name, attribute, transform)] checkable by attribute."""
    return [('masters_build.VARIANT_REL_K', 'coadd', 'VARIANT_REL_K', float),
            ('masters_pbmaps.PB_CHROM_VERSION', 'pb_apply', 'CHROM_VERSION', str),
            ('masters_pbmaps.PB_FLOOR', 'pb_apply', 'PB_FLOOR', float),
            ('paths.PHASE2_ROOT', 'pb_apply', 'QA_ROOT', str),
            ('paths.PHASE2_ROOT', 'coadd', 'PHASE2_ROOT', str)]


def check_mirrors(modules=None):
    """Refuse when an in-place module attribute differs from the config value it mirrors."""
    c = cfg()
    bad = []
    for dotted, mod, attr, tf in mirror_pairs():
        if modules is not None and mod not in modules:
            continue
        m = _module(mod)
        have = getattr(m, attr, None)
        want = c.get(dotted)
        if have is None or tf(have) != tf(want):
            bad.append(f'{mod}.{attr} = {have!r} but config {dotted} = {want!r}')
    if bad:
        raise Refusal('in-place literal differs from its config mirror: ' + '; '.join(bad))


# ---------------------------------------------------------------------------------------------------------------------
# pinned tables, production roots
# ---------------------------------------------------------------------------------------------------------------------

def sha1_file(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def sha1_pin(path, want, what):
    """sha1 of a pinned table; refuses when missing or different from config."""
    if not os.path.exists(path):
        raise Refusal(f'{what}: pinned table {path} missing - refusing (a gate never admits what it cannot judge)')
    got = sha1_file(path)
    if want and got != want:
        raise Refusal(f'{what}: {path} sha1 {got} differs from the pinned {want} (config) - refusing; re-pin by a commit '
                      f'after checking the change')
    return got


def is_production_path(path):
    """The production root holding ``path`` (paths.PRODUCTION_ROOTS + masters.EXTRA_PRODUCTION_ROOTS), or None."""
    c = cfg()
    r = c.production_root_of(path)
    if r:
        return r
    real = os.path.realpath(path)
    for root in tuple(c.masters.EXTRA_PRODUCTION_ROOTS) + (c.paths.OPS_ROOT,):
        rr = os.path.realpath(root)
        if real == rr or real.startswith(rr.rstrip('/') + '/'):
            return root
    return None


def refuse_production_write(path, what):
    if cfg().paths.PRODUCTION_WRITES:
        return
    r = is_production_path(path)
    if r:
        raise Refusal(f'{what}: {path} is under the production root {r} and paths.PRODUCTION_WRITES is False '
                      f'(refactor window) - pass --out to a parity/scratch root')


# ---------------------------------------------------------------------------------------------------------------------
# exclusions: the two readers must agree (G-M01)
# ---------------------------------------------------------------------------------------------------------------------

def exclusion_readers_agree(lst, scope, dates, coadd_mod=None):
    """core.exclusions vs the in-place co-add reader on every (lst, scope, night): returns the core Exclusions; refuses
    when the two readers or the two file paths disagree. Loads the co-add's exclusions through coadd.load_exclusions() so
    its module state (_PERBAND_EXCL) is filled (FIX-COADD-SINGLE-MODULE-INSTANCE-PERBAND)."""
    from exopipe.phase3.core import exclusions
    c = cfg()
    ex = exclusions.load()
    co = coadd_mod or _module('coadd')
    if os.path.realpath(co.DECISIONS_CSV) != os.path.realpath(c.paths.DATA_DECISIONS):
        raise Refusal(f'co-add exclusion list {co.DECISIONS_CSV} != paths.DATA_DECISIONS {c.paths.DATA_DECISIONS} (U-1c)')
    if {s.replace('MHz', '') for s in c.exclusions.SUBBAND_ONLY} != set(co.SUBBAND_ONLY_EXCL):
        raise Refusal(f'exclusions.SUBBAND_ONLY {c.exclusions.SUBBAND_ONLY} != co-add SUBBAND_ONLY_EXCL '
                      f'{sorted(co.SUBBAND_ONLY_EXCL)} (R-18MHZ-CARVEOUT-REAFFIRMED)')
    excl = co.load_exclusions()
    fs = scope.replace('MHz', '')
    bad = [d for d in dates if ex.is_excluded(lst, scope, d) != co.is_excluded(lst, fs, d, excl)]
    if bad:
        raise Refusal(f'{lst} {scope}: core.exclusions and the co-add reader disagree on {len(bad)} night(s) {bad[:5]} - '
                      f'refusing (one rule for every tier, G-M01)')
    return ex, excl


# ---------------------------------------------------------------------------------------------------------------------
# stdout capture of an in-place call
# ---------------------------------------------------------------------------------------------------------------------

class Tee(io.TextIOBase):
    """Write-through to a log file (and optionally the real stdout), keeping the last ``keep`` lines."""

    def __init__(self, log_path=None, echo=None, keep=400):
        self.lines = collections.deque(maxlen=keep)
        self._buf = ''
        self._f = open(log_path, 'a') if log_path else None
        self._echo = echo

    def write(self, s):
        if self._f:
            self._f.write(s)
        if self._echo is not None:
            self._echo.write(s)
        self._buf += s
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            self.lines.append(line)
        return len(s)

    def flush(self):
        if self._f:
            self._f.flush()
        if self._echo is not None:
            self._echo.flush()

    def close_log(self):
        if self._buf:
            self.lines.append(self._buf)
            self._buf = ''
        if self._f:
            self._f.close()
            self._f = None

    def find(self, *needles):
        """The last captured line containing any of ``needles`` (or '')."""
        for line in reversed(self.lines):
            if any(n in line for n in needles):
                return line.strip()
        return ''


@contextlib.contextmanager
def capture_stdout(log_path=None, echo=True):
    """stdout (Python-level prints of the in-place code) teed to ``log_path``; yields the Tee."""
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    tee = Tee(log_path, echo=sys.__stdout__ if echo else None)
    old = sys.stdout
    sys.stdout = tee
    try:
        yield tee
    finally:
        sys.stdout = old
        tee.close_log()


@contextlib.contextmanager
def saved_sys_path():
    """Restore sys.path after importing a legacy script that inserts the PRODUCTION package root at position 0
    (p3s5_parents.py:15): modules already imported keep resolving to the branch; later imports are not redirected."""
    old = list(sys.path)
    try:
        yield
    finally:
        sys.path[:] = old
