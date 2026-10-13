"""Shared parts of `exopipe3 mosaic` (steps/mosaic.py): the job grammar, the LST list (the masters that exist), the master
resolver, the per-job input plan (gates G1, G3, G4, G8), the in-place literal checks (gate G0), product paths, and the lane
runner that executes worker tasks (lib/mosaic/steps_worker.py) in child processes through core.legacy.run.

Nothing here writes at import. Heavy imports (numpy, astropy) stay inside the functions.
"""
import glob
import json
import os
import re
import sys

STAGES = ('inputs', 'coadd', 'render', 'qa', 'conform')
JOB_STAGES = ('inputs', 'coadd', 'render')
STEP = 'mosaic'

# In-place scripts of exopipe_ops run during the window (DESIGN 11.1). Resolved under core.env.legacy_root('exopipe_ops');
# SCRIPT_PATHS overrides a resolution (tests point it at a synthetic stand-in; never set in production code).
STAGE_SCRIPT = 'mosaic_stage_from_lustre.py'
ANCHOR_SCRIPT = 'mosaic_v2_flux_anchor_measure.py'
SCRIPT_PATHS = {}

# Environment variables pb_apply reads at import (lib/masters/pb_apply.py:31, 38, 57, 59, 60): set in this process they would
# change the weighting PB away from the ruled three-factor map (gate G7). core.legacy.run clears P3_* in the children anyway.
PB_ENV = ('P3_PB_LIBRARY', 'P3_RESIDUAL_PB', 'P3_CHAN_AVG_HZ', 'P3_PB_THREE_FACTOR', 'P3_RESIDUAL_PB_V2')

_LST_DIR = re.compile(r'^\d\dh$')


def cfg():
    from exopipe.phase3.config import CFG
    return CFG


def script_path(name):
    if name in SCRIPT_PATHS:
        return SCRIPT_PATHS[name]
    from exopipe.phase3.core import env
    return os.path.join(env.legacy_root('exopipe_ops'), name)


# ---------------------------------------------------------------------------------------------------------------------
# Jobs: <scope>_<variant>[_<product>]
# ---------------------------------------------------------------------------------------------------------------------

class Job:
    """One mosaic job = (scope, imaging variant, product) with all its domains (spec 1)."""

    def __init__(self, scope, variant, product='I', domains=None):
        self.scope, self.variant, self.product = scope, variant, product
        self.domains = tuple(domains) if domains else None

    @property
    def name(self):
        return f'{self.scope}_{self.variant}' + ('' if self.product == 'I' else f'_{self.product}')

    @property
    def robust(self):
        return 'R075' if self.variant.endswith('R075') else 'R0'

    @property
    def taper(self):
        return self.variant.split('_')[0]

    @property
    def robust_value(self):
        return -0.75 if self.robust == 'R075' else 0.0

    @property
    def stem(self):
        return f'I_deep_{self.variant}_{self.scope}'

    def __repr__(self):
        return f'Job({self.name}{"" if not self.domains else " " + ",".join(self.domains)})'


def parse_job(token):
    """'Blue_Taper_R075' | '27MHz_NoTaper_R0' | 'Blue_Taper_R075_I' | 'Taper_R075 Blue jysr' (run_mosaic_jobs.sh line form,
    one domain) -> Job. Raises ValueError for anything else."""
    C = cfg()
    tok = token.strip()
    fields = tok.split()
    if len(fields) == 3:                                   # 'variant band domain'
        variant, scope, domain = fields
        job = parse_job(f'{scope}_{variant}')
        if domain not in ('jysr', 'jybeam'):
            raise ValueError(f'{token!r}: domain must be jysr or jybeam')
        job.domains = (domain,)
        return job
    if len(fields) != 1:
        raise ValueError(f'not a mosaic job: {token!r} (expected <scope>_<variant>[_<product>], e.g. Blue_Taper_R075)')
    scope, _, rest = tok.partition('_')
    if scope not in C.mosaic.SCOPES:
        raise ValueError(f'{token!r}: unknown scope {scope!r} (mosaic.SCOPES: {", ".join(C.mosaic.SCOPES)})')
    variants = sorted(C.mosaic.VARIANT_TOKENS, key=len, reverse=True)
    for v in variants:
        if rest == v:
            return Job(scope, v)
        if rest.startswith(v + '_'):
            return Job(scope, v, rest[len(v) + 1:])
    raise ValueError(f'{token!r}: unknown variant in {rest!r} (variants: {", ".join(C.mosaic.VARIANT_TOKENS)})')


def job_universe():
    C = cfg()
    return [f'{s}_{v}' for v in C.mosaic.JOB_ORDER_VARIANTS for s in C.mosaic.SCOPES]


def _csv(text):
    return [t.strip() for t in str(text).split(',') if t.strip()]


def select_jobs(args):
    """The jobs of this run, in scheduling order (mosaic.JOB_ORDER_VARIANTS, then scope order): --jobs (item grammar of
    core/cells: list, shell patterns over the job universe, @FILE, record:STATUS:STEP@OUT, JOB:STATUS) or every scope of
    --scopes x every variant of --variants; filtered by --scopes/--variants/--products when both are given."""
    from exopipe.phase3.core import cells
    C = cfg()
    variants = _csv(args.variants) if getattr(args, 'variants', None) else list(C.mosaic.VARIANTS)
    scopes = _csv(args.scopes) if getattr(args, 'scopes', None) else list(C.mosaic.SCOPES)
    products = _csv(args.products) if getattr(args, 'products', None) else list(C.mosaic.PRODUCTS)
    domains = _csv(args.domains) if getattr(args, 'domains', None) else list(C.mosaic.DOMAINS)
    for v in variants:
        if v not in C.mosaic.VARIANT_TOKENS:
            raise ValueError(f'--variants: unknown variant {v!r} ({", ".join(C.mosaic.VARIANT_TOKENS)})')
    for s in scopes:
        if s not in C.mosaic.SCOPES:
            raise ValueError(f'--scopes: unknown scope {s!r} ({", ".join(C.mosaic.SCOPES)})')
    for d in domains:
        if d not in ('jysr', 'jybeam'):
            raise ValueError(f'--domains: unknown domain {d!r} (jysr, jybeam)')
    if getattr(args, 'jobs', None):
        jobs = []
        for it in cells.expand(args.jobs, universe=job_universe(), kind='mosaic job'):
            j = parse_job(it)
            if (getattr(args, 'scopes', None) and j.scope not in scopes) or \
                    (getattr(args, 'variants', None) and j.variant not in variants):
                continue
            jobs.append(j)
    else:
        jobs = [Job(s, v, p) for v in C.mosaic.JOB_ORDER_VARIANTS if v in variants for s in C.mosaic.SCOPES
                if s in scopes for p in products]
    out, seen = [], {}
    for j in jobs:
        dom = tuple(d for d in (j.domains or domains) if d in domains) if j.domains else tuple(domains)
        if j.name in seen:                                   # two @file lines for one job: union of their domains
            prev = seen[j.name]
            prev.domains = tuple(d for d in ('jysr', 'jybeam') if d in set(prev.domains) | set(dom))
            continue
        j.domains = dom
        seen[j.name] = j
        out.append(j)
    order = {v: i for i, v in enumerate(C.mosaic.JOB_ORDER_VARIANTS)}
    sidx = {s: i for i, s in enumerate(C.mosaic.SCOPES)}
    return sorted(out, key=lambda j: (order.get(j.variant, 99), sidx.get(j.scope, 99), j.product))


# ---------------------------------------------------------------------------------------------------------------------
# LSTs: the masters that exist
# ---------------------------------------------------------------------------------------------------------------------

def lst_universe():
    """The LST directories (NNh) of mosaic.INPUT_COADD_RUN - the masters that exist (deep chat 927a836) - or the tuple
    mosaic.LSTS when one is configured."""
    C = cfg()
    if C.mosaic.LSTS:
        return [str(x) for x in C.mosaic.LSTS]
    root = C.mosaic.INPUT_COADD_RUN
    try:
        names = os.listdir(root)
    except OSError as e:
        from exopipe.phase3.core.record import Refusal
        raise Refusal(f'G3: the input master generation {root} is not readable ({e})')
    return sorted(n for n in names if _LST_DIR.match(n) and os.path.isdir(os.path.join(root, n)))


def select_lsts(args):
    from exopipe.phase3.core import cells
    uni = lst_universe()
    if getattr(args, 'lsts', None):
        want = cells.expand(args.lsts, universe=uni, kind='LST')
        bad = [x for x in want if x not in uni]
        if bad:
            raise ValueError(f'--lsts: {bad} not among the LSTs of {cfg().mosaic.INPUT_COADD_RUN} ({len(uni)}: '
                             f'{", ".join(uni)})')
        return [x for x in uni if x in want], uni
    return uni, uni


# ---------------------------------------------------------------------------------------------------------------------
# Masters and the per-job input plan
# ---------------------------------------------------------------------------------------------------------------------

def master_candidates(lst, scope, variant):
    """Every file matching mosaic.MASTER_PATTERNS of (lst, scope, variant), in the stage script's order (STG:44-46)."""
    C = cfg()
    tok = C.mosaic.VARIANT_TOKENS[variant]
    d = os.path.join(C.mosaic.INPUT_COADD_RUN, lst, scope)
    pats = C.mosaic.MASTER_PATTERNS['I']
    first = sorted(glob.glob(os.path.join(d, pats[0].format(scope=scope, var=tok, lst=lst))))
    rest = []
    for p in pats[1:]:
        rest += glob.glob(os.path.join(d, p.format(scope=scope, var=tok, lst=lst)))
    return sorted(first + rest)


def _freq(h):
    for k in ('RESTFRQ', 'RESTFREQ', 'CRVAL3'):
        if k in h and h[k] and h[k] > cfg().mosaic_coadd.MIN_FREQ_HZ:
            return float(h[k])
    return None


_HDR_CACHE = {}


def tile_header(path):
    """The header facts the plan, the gates and the conform provenance need (read header-only; cached per process by
    path, size and mtime - a Taper master is also the weighting-PB reference of its NoTaper sibling)."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key not in _HDR_CACHE:
        _HDR_CACHE[key] = _tile_header(path)
    return dict(_HDR_CACHE[key])


def _tile_header(path):
    from astropy.io import fits
    h = fits.getheader(path)
    hist = '\n'.join(str(c) for c in h.get('HISTORY', []))
    dates = sorted(set(re.findall(r'constituent\s+(20\d\d-\d\d-\d\d)', hist)))   # = pb_apply.constituent_dates (:309-313)

    def val(k):
        v = h.get(k)
        return v if isinstance(v, (bool, int, float, str)) or v is None else str(v)
    return {'path': os.path.abspath(path), 'realpath': os.path.realpath(path), 'dates': dates,
            'PBCORR': val('PBCORR'), 'PBMODEL': val('PBMODEL'), 'PBCHROM': val('PBCHROM'), 'PBRESID': val('PBRESID'),
            'BMAJ': val('BMAJ'), 'BMIN': val('BMIN'), 'freq_hz': _freq(h), 'MJD-OBS': val('MJD-OBS'),
            'MJD-MID': val('MJD-MID'), 'INTTIME': val('INTTIME'), 'THERMNOI': val('THERMNOI'), 'GITTAG': val('GITTAG'),
            'DATE-END': val('DATE-END'), 'CONSTDIG': val('CONSTDIG'), 'NAXIS1': val('NAXIS1'), 'NAXIS2': val('NAXIS2'),
            'mdewarp': '_mdewarp_' in os.path.basename(path)}


class JobPlan:
    """The resolved inputs of one job and the gate verdicts that need no pixels (G1, G3, G4, G8 header part)."""

    def __init__(self, job):
        self.job = job
        self.tiles = {}          # lst -> {'master': path|None, 'n_candidates': int, 'hdr': {...}, 'excluded': [...]}
        self.refusals = []       # [(gate, message)]
        self.notes = []
        self.lsts = []

    @property
    def admitted(self):
        return sorted(f'{l}:{d}' for l, t in self.tiles.items() if t.get('hdr') for d in t['hdr']['dates'])

    @property
    def masters(self):
        return {l: t['master'] for l, t in self.tiles.items() if t.get('master')}

    def refuse(self, gate, msg):
        self.refusals.append((gate, msg))

    def reason(self):
        return '; '.join(f'{g}: {m}' for g, m in self.refusals)


def plan_job(job, lsts, universe, ex, run=None):
    """JobPlan of one job: one master per LST of the universe (G3), the LST count (G4), every constituent night admitted
    (G1, not waivable), and the header checks of G8 (PBCORR True; PBMODEL, PBCHROM, PBRESID, BMAJ, BMIN present; a frequency
    above mosaic_coadd.MIN_FREQ_HZ)."""
    C = cfg()
    P = JobPlan(job)
    P.lsts = list(lsts)
    if job.product not in C.mosaic.PRODUCTS or job.product != 'I':
        P.refuse('G-PRODUCT', f'product {job.product!r} has no production input pattern (mosaic.PRODUCTS = '
                              f'{C.mosaic.PRODUCTS}; only I is built in v2)')
        return P
    need = C.mosaic.REQUIRE_N_LSTS
    if len(lsts) < need:
        why = (f'{len(lsts)} LST tiles planned < mosaic.REQUIRE_N_LSTS {need} (LSTs of the input generation: '
               f'{len(universe)})')
        if run is not None and run.waived('mosaic.inputs_complete'):
            P.notes.append(f'G4 WAIVED: {why} [{run.waived("mosaic.inputs_complete")}]')
        else:
            P.refuse('G4', why + ' - a partial mosaic needs --waive mosaic.inputs_complete --ruling "<Gregg\'s words, UTC, '
                                 'ledger ID>"')
    for lst in lsts:
        cands = master_candidates(lst, job.scope, job.variant)
        t = {'master': cands[0] if len(cands) == 1 else None, 'n_candidates': len(cands), 'candidates': cands}
        P.tiles[lst] = t
        if len(cands) != 1:
            P.refuse('G3', f'{lst}: {len(cands)} current-master candidates for {job.scope} '
                           f'{C.mosaic.VARIANT_TOKENS[job.variant]}' + (f' {[os.path.basename(c) for c in cands]}'
                                                                        if cands else ''))
            continue
        try:
            h = tile_header(cands[0])
        except Exception as e:
            P.refuse('G8', f'{lst}: header unreadable ({type(e).__name__}: {e})')
            continue
        t['hdr'] = h
        if not h['dates']:
            P.refuse('G1', f"{lst}: no 'COADD constituent' HISTORY in {os.path.basename(cands[0])} - the exclusion "
                           f"state of its nights cannot be judged")
        else:
            bad = ex.excluded_constituents(lst, job.scope, h['dates'])
            t['excluded'] = bad
            if bad:
                P.refuse('G1', f'{lst}: master {os.path.basename(cands[0])} contains night(s) {", ".join(bad)} that '
                               f'data_decisions.csv excludes for {lst} {job.scope} (not waivable; rebuild/reinstall '
                               f'the cell first)')
        ref_var = C.mosaic.WEIGHT_PB_REF_VARIANT[job.robust]
        if ref_var != job.variant:
            # the weighting PB of this tile is derived from the Taper master of the same robust (STG:61-67), with ITS
            # constituent nights' subband weights: an excluded night there would reach this mosaic through its weights
            rc = master_candidates(lst, job.scope, ref_var)
            if len(rc) == 1:
                try:
                    rd = tile_header(rc[0])['dates']
                except Exception as e:
                    rd = None
                    P.refuse('G1', f'{lst}: weighting-PB reference master {os.path.basename(rc[0])} unreadable ({e})')
                if rd is not None:
                    rbad = ex.excluded_constituents(lst, job.scope, rd)
                    t['pb_ref'] = {'master': rc[0], 'excluded': rbad, 'n_nights': len(rd)}
                    if rbad:
                        P.refuse('G1', f'{lst}: the weighting PB comes from {os.path.basename(rc[0])} ({ref_var}), which '
                                       f'contains night(s) {", ".join(rbad)} that data_decisions.csv excludes')
            else:
                P.refuse('G5', f'{lst}: {len(rc)} candidates for the weighting-PB reference master ({ref_var}); the stage '
                               f'script derives no PB for {job.robust} without exactly one')
        miss = [k for k in ('PBMODEL', 'PBCHROM', 'PBRESID', 'BMAJ', 'BMIN') if h.get(k) in (None, '')]
        if h.get('PBCORR') is not True:
            P.refuse('G8', f'{lst}: PBCORR is {h.get("PBCORR")!r}, not True')
        if miss:
            P.refuse('G8', f'{lst}: header lacks {", ".join(miss)}')
        if h.get('freq_hz') is None:
            P.refuse('G8', f'{lst}: no RESTFRQ/RESTFREQ/CRVAL3 above {C.mosaic_coadd.MIN_FREQ_HZ:g} Hz (the engine would '
                           f'write FREQ NaN and JY2K 0.0)')
    return P


def scope_duplicates(scope, lsts):
    """[(lst, variant, n)] for every variant of the scope with more than one candidate - the in-place stage script stages
    all four variants of a scope and REFUSES the whole scope on any of them (STG:47-48)."""
    C = cfg()
    out = []
    for lst in lsts:
        for v in C.mosaic.VARIANT_TOKENS:
            n = len(master_candidates(lst, scope, v))
            if n > 1:
                out.append((lst, v, n))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Product paths (today's mosaic_v2 layout; no renames in the window)
# ---------------------------------------------------------------------------------------------------------------------

def layout(out, key):
    return os.path.join(out, cfg().mosaic.LAYOUT[key])


def coadd_products(job, domain, nside=None):
    """[(file name, product class)] the engine writes for one domain (CJ:261-283)."""
    n = nside or cfg().mosaic_coadd.NSIDE
    if domain == 'jysr':
        return [(f'healpix_{job.stem}_jysr_nside{n}.fits', 'mosaic_map'),
                (f'healpix_PSF_{job.stem}_nside{n}.fits', 'mosaic_psfmap')]
    return [(f'healpix_{job.stem}_jybeam_nside{n}.fits', 'mosaic_map'),
            (f'healpix_PBeff_{job.stem}_nside{n}.fits', 'mosaic_pbeff')]


def render_products(job, domain):
    """[(file name, kind)] band_products.build writes for one domain (BP:63, :69 with suffix '_<domain>')."""
    return [(f'{job.scope}_{job.variant}_{domain}_Mollweide.fits', 'mollweide'),
            (f'{job.scope}_{job.variant}_{domain}_RADec.fits', 'radec')]


def unit_name(stage, job_or_name):
    return f'{stage}/{getattr(job_or_name, "name", job_or_name)}'


# ---------------------------------------------------------------------------------------------------------------------
# Gate G0: config values equal the in-place literals (the window's in-place code reads its own constants)
# ---------------------------------------------------------------------------------------------------------------------

def _num(x):
    return float(x)


# (config key, module or script, regex whose group(s) hold the literal, expected value from config)
def _literal_table(C):
    cc, rq, rr = C.mosaic_coadd, C.mosaic_qa, C.mosaic_render
    return [
        ('mosaic_coadd.PB_WEIGHT_EXPONENT', 'coadd_jysr', r'pb \*\* (\d+) / sigma \*\* (\d+)',
         (cc.PB_WEIGHT_EXPONENT, 2)),
        ('mosaic_coadd.MIN_VALID_TILE_PIXELS', 'coadd_jysr', r'if app\.size < (\d+):', (cc.MIN_VALID_TILE_PIXELS,)),
        ('mosaic_coadd.CORRUPT_FLAT_SIGMA_JYBEAM', 'coadd_jysr', r'or sigma < ([0-9.eE+-]+):',
         (cc.CORRUPT_FLAT_SIGMA_JYBEAM,)),
        ('mosaic_coadd.COMMON_GRID_RA_OFFSET_H', 'coadd_jysr', r'int\(lst\[:2\]\) \+ ([0-9.]+)\)',
         (cc.COMMON_GRID_RA_OFFSET_H,)),
        ('mosaic_coadd.COMMON_GRID_CRVAL2_DEG', 'coadd_jysr', r'% 360\.0, ([0-9.]+)\]', (cc.COMMON_GRID_CRVAL2_DEG,)),
        ('mosaic_coadd.MIN_FREQ_HZ', 'coadd_jysr', r'hdr\[k\] > ([0-9.eE+-]+):', (cc.MIN_FREQ_HZ,)),
        ('mosaic_coadd.NSIDE', 'band_products', r'_nside(\d+)\.fits', (cc.NSIDE,)),
        ('mosaic_coadd.NSIDE', 'verify_mosaic', r'query_disc\((\d+),', (cc.NSIDE,)),
        ('mosaic_coadd.NSIDE', ANCHOR_SCRIPT, r'NSIDE = (\d+);', (cc.NSIDE,)),
        ('mosaic_render.CAR_DEC_MIN_DEG', 'band_products', r'dec_lo=([0-9.+-]+), dec_hi=([0-9.+-]+)',
         (rr.CAR_DEC_MIN_DEG, rr.CAR_DEC_MAX_DEG)),
        ('mosaic_render.OMEGA_PER_FWHM2', 'band_products', r'np\.median\(om\) / ([0-9.]+)\)', (rr.OMEGA_PER_FWHM2,)),
        ('mosaic_qa.ANCHOR_APERTURE_FWHM', ANCHOR_SCRIPT, r'rad_scale=([0-9.]+)\)', (rq.ANCHOR_APERTURE_FWHM,)),
        ('mosaic_qa.ANCHOR_PSF_DISC_DEG', ANCHOR_SCRIPT, r'np\.radians\(([0-9.]+)\)\)\]\)', (rq.ANCHOR_PSF_DISC_DEG,)),
        ('mosaic_qa.ANCHOR_ANNULUS_FWHM', ANCHOR_SCRIPT,
         r'np\.radians\(([0-9.]+) \* fwhm\)\), hp\.query_disc\(NSIDE, v, np\.radians\(([0-9.]+) \* fwhm\)',
         (rq.ANCHOR_ANNULUS_FWHM[1], rq.ANCHOR_ANNULUS_FWHM[0])),
        ('mosaic_qa.VLSSR_DEC_RANGE_DEG', 'verify_mosaic', r'cat\[:, 1\] > ([0-9.+-]+)\) & \(cat\[:, 1\] < ([0-9.+-]+)\)',
         tuple(rq.VLSSR_DEC_RANGE_DEG)),
        ('mosaic_qa.VLSSR_CENTROID_DISC_DEG', 'verify_mosaic', r'pix = hp\.query_disc\(\d+, v, np\.radians\(([0-9.]+)\)\)',
         (rq.VLSSR_CENTROID_DISC_DEG,)),
    ]


def _source(name):
    if name.endswith('.py'):
        p = script_path(name)
    else:
        import importlib
        p = importlib.import_module(f'exopipe.phase3.lib.mosaic.{name}').__file__
    with open(p) as f:
        return p, f.read()


def g0_check():
    """[problem text] for every config value that differs from the literal the in-place code uses, plus the stage
    script's ROOT / VARMAP / name patterns / PB reference variants and the anchor script's calibrator choice."""
    C = cfg()
    probs = []
    cache = {}
    for key, mod, rx, want in _literal_table(C):
        try:
            if mod not in cache:
                cache[mod] = _source(mod)
            path, src = cache[mod]
        except (OSError, ImportError, KeyError) as e:
            probs.append(f'{key}: cannot read the in-place code {mod} ({e})')
            continue
        m = re.search(rx, src)
        if not m:
            probs.append(f'{key}: literal not found in {path} (pattern {rx!r}) - the in-place code changed')
            continue
        got = tuple(_num(g) for g in m.groups())
        if any(abs(a - float(b)) > 1e-12 * max(1.0, abs(float(b))) for a, b in zip(got, want)):
            probs.append(f'{key}: config {want if len(want) > 1 else want[0]!r} but {os.path.basename(path)} uses '
                         f'{got if len(got) > 1 else got[0]!r}')
    # stage script: ROOT, VARMAP, both name forms, PB reference = Taper master of each robust
    try:
        path, src = _source(STAGE_SCRIPT)
        m = re.search(r"^ROOT = '([^']*)'", src, re.M)
        if not m or m.group(1).rstrip('/') != C.mosaic.INPUT_COADD_RUN.rstrip('/'):
            probs.append(f'mosaic.INPUT_COADD_RUN {C.mosaic.INPUT_COADD_RUN!r} but {path} stages ROOT '
                         f'{m.group(1) if m else "(not found)"!r}')
        for long, short in ((v, k) for k, v in C.mosaic.VARIANT_TOKENS.items()):
            if f"'{long}': '{short}'" not in src:
                probs.append(f'mosaic.VARIANT_TOKENS {short}={long} not in the stage script VARMAP ({path})')
        for form in ('_shflux_pbcorr_dewarped_aligned_', '_shflux_mdewarp_'):
            if form not in src:
                probs.append(f'mosaic.MASTER_PATTERNS: name form {form!r} not globbed by {path}')
        if "refs.get(f'Taper_{rob}')" not in src or set(C.mosaic.WEIGHT_PB_REF_VARIANT.values()) != {'Taper_R075',
                                                                                                    'Taper_R0'}:
            probs.append(f'mosaic.WEIGHT_PB_REF_VARIANT {C.mosaic.WEIGHT_PB_REF_VARIANT} differs from the stage script '
                         f'(Taper master of each robust, {path})')
    except OSError as e:
        probs.append(f'stage script unreadable ({e})')
    try:
        path, src = _source(ANCHOR_SCRIPT)
        if "('3C196', np.nan) * r.get('3C380'" not in src or tuple(C.mosaic_qa.ANCHOR_SOURCES) != ('3C196', '3C380'):
            probs.append(f'mosaic_qa.ANCHOR_SOURCES {C.mosaic_qa.ANCHOR_SOURCES} differs from {path}')
        if "!= 'SH12'" not in src or C.mosaic_qa.CALIBRATOR_SCALE != 'SH12':
            probs.append(f'mosaic_qa.CALIBRATOR_SCALE {C.mosaic_qa.CALIBRATOR_SCALE!r} differs from {path}')
    except OSError as e:
        probs.append(f'anchor script unreadable ({e})')
    try:
        from exopipe.phase3.lib.mosaic import verify_mosaic as vm
        if vm.VLSSR != C.mosaic_qa.VLSSR_CATALOG:
            probs.append(f'mosaic_qa.VLSSR_CATALOG {C.mosaic_qa.VLSSR_CATALOG!r} but verify_mosaic reads {vm.VLSSR!r}')
    except ImportError as e:
        probs.append(f'verify_mosaic not importable ({e})')
    # values with exactly one implementation in the window
    if C.mosaic_render.RENDER_DEC_CUT_DEG is not None:
        probs.append('mosaic_render.RENDER_DEC_CUT_DEG: only None (no cut) is implemented (batch row B-46 pending)')
    if C.mosaic.WEIGHT_PB_SOURCE != 'derive':
        probs.append(f"mosaic.WEIGHT_PB_SOURCE {C.mosaic.WEIGHT_PB_SOURCE!r}: only 'derive' is implemented (archived maps "
                     f"wait for the parity check P-PB)")
    if C.mosaic.MISSING_TILE_ACTION != 'refuse':
        probs.append(f"mosaic.MISSING_TILE_ACTION {C.mosaic.MISSING_TILE_ACTION!r}: only 'refuse' is implemented")
    if C.mosaic.TILE_GUARD_ACTION not in ('skip-and-record', 'refuse'):
        probs.append(f"mosaic.TILE_GUARD_ACTION {C.mosaic.TILE_GUARD_ACTION!r}: 'skip-and-record' or 'refuse'")
    if C.mosaic_coadd.REPROJECT_ORDER != 'bilinear' or C.mosaic_coadd.SIGNAL_PATH != 'native-sin-healpix':
        probs.append('mosaic_coadd.REPROJECT_ORDER / SIGNAL_PATH: only bilinear / native-sin-healpix are implemented')
    return probs


def engine_globals():
    """Module globals of coadd_jysr the step sets before each call (the WINDOW-GLOBAL keys)."""
    cc = cfg().mosaic_coadd
    return {'NSIDE': int(cc.NSIDE), 'DEC_FLOOR': float(cc.DEC_FLOOR_DEG), 'PBCORR_FLOOR': float(cc.PB_FLOOR),
            'PB_FEATHER': float(cc.PB_FEATHER_START), 'N_FLOOR': float(cc.COSZ_FLOOR),
            'GRID_N': int(cc.COMMON_GRID_N_PIX), 'CDELT': float(cc.COMMON_GRID_CDELT_DEG)}


# ---------------------------------------------------------------------------------------------------------------------
# Lanes and worker tasks
# ---------------------------------------------------------------------------------------------------------------------

def lanes(n_lanes, cores_per_lane):
    """[[cpu, ...], ...]: up to n_lanes disjoint core sets of cores_per_lane from this process's affinity (the cores
    --cores pinned; production pinned each co-add to two cores with taskset)."""
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except AttributeError:
        cpus = list(range(os.cpu_count() or 1))
    k = max(1, int(cores_per_lane))
    n = max(1, min(int(n_lanes), len(cpus) // k if len(cpus) >= k else 1))
    if len(cpus) < k:
        return [cpus]
    return [cpus[i * k:(i + 1) * k] for i in range(n)]


def default_workers(args):
    C = cfg()
    if getattr(args, 'workers', None):
        return int(args.workers)
    try:
        ncpu = len(os.sched_getaffinity(0))
    except AttributeError:
        ncpu = os.cpu_count() or 1
    return max(1, min(C.mosaic.MAX_JOBS_PER_NODE, ncpu // max(1, C.mosaic.CORES_PER_JOB)))


def _pkg_root():
    here = os.path.dirname(os.path.abspath(__file__))           # .../exopipe/phase3/lib/mosaic
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))


def run_tasks(tasks, lane_sets, task_dir, label='task'):
    """Run worker tasks (dicts with 'kind', 'key', 'log', ...) on the lanes, at most one task per lane at a time. Each runs
    `python -m exopipe.phase3.lib.mosaic.steps_worker TASK.json` through core.legacy.run (site interpreter 'default',
    P3_* cleared, NUMPY_MADVISE_HUGEPAGE=0, the parity guard and the child code fingerprint). Returns {key: result} where
    result is the worker's JSON plus 'rc', 'wall_s', 'legacy_code' (sha1 of the child's imported legacy/exopipe files)."""
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from exopipe.phase3.core import legacy
    os.makedirs(task_dir, exist_ok=True)
    free = queue.Queue()
    for ls in lane_sets:
        free.put(ls)
    lock = threading.Lock()
    results = {}
    env = {'PYTHONPATH': _pkg_root() + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')}

    def one(task):
        cores = free.get()
        try:
            t = dict(task, cores=list(cores))
            tp = os.path.join(task_dir, f"{label}_{t['key'].replace('/', '_')}.json")
            t['result'] = tp + '.result'
            with open(tp, 'w') as f:
                json.dump(t, f, indent=1, default=str)
            if t.get('log'):
                os.makedirs(os.path.dirname(t['log']), exist_ok=True)
            res = legacy.run('exopipe.phase3.lib.mosaic.steps_worker', [tp], python='default', module=True, env=env,
                             log=t.get('log'), refusal_codes=(), refusal_text=())
            try:
                with open(t['result']) as f:
                    out = json.load(f)
            except (OSError, ValueError):
                out = {'status': 'error', 'message': f'worker wrote no result (exit {res.rc}): {res.tail[-800:]}'}
            out.update(rc=res.rc, wall_s=res.wall_s, legacy_code=res.fingerprint, log=t.get('log'),
                       argv=' '.join(res.argv))
            with lock:
                results[t['key']] = out
            print(f"mosaic {label} {t['key']}: {out.get('status')} ({res.wall_s:.0f} s)"
                  + (f" - {out.get('message')}" if out.get('status') != 'ok' else ''), file=sys.stderr, flush=True)
        finally:
            free.put(cores)

    with ThreadPoolExecutor(max_workers=max(1, len(lane_sets))) as ex:
        list(ex.map(one, tasks))
    return results


def utc_date():
    import time
    return time.strftime('%Y%m%d', time.gmtime())
