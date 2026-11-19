"""`exopipe3 mosaic --generation v3`: the FEATHER3 mosaics of the deep masters (spec plan/specs/hybrid.md 2.9-2.13, 2.15; PR-6b).
Unit of work: (stage, scope, job); records <out>/_records/mosaic/v3-<stage>.<scope>_<job>.json; files under <out>/<scope>_<job>/.
Jobs of a scope (config mosaic_v3): I_<variant> for each of mosaic_v3.VARIANTS (the plain Stokes I maps; --striation without|both),
I_<variant>_ssub-rst8 for each (variant, scope) pair of hybrid.STRIATION_SELECTION (--striation with|both), and Vclean (the cleaned
Stokes V maps; --products). Production: mosaic_v3_20261022/ step2_stage.sh (STG = mosaic_stage_from_lustre.py), step2b_stage_ssub.py
(S2B), step3_coadd.sh, step4_renders.sh, mosaic_pkcorr.py (MP), mosaic_content_test.py (MT); vclean_refresh_20261110/
release1_v_mosaics_20261110.sh (the Stokes V maps, TEST 1R).

  stage   plain: the 24 current masters of mosaic.INPUT_COADD_RUN (steps_common.plan_job: G1 every constituent night admitted, G3 one
          master per LST, G4 every LST, G8 header cards) and, per LST, the weighting PB map of the job's robust weighting derived with
          pb_apply.wideband_pb from that weighting's Taper master and written with STG's header -> stage/pb/<LST>/PB_deep[_R075]_<S>.fits
          [mosaic_weight_pb], stage/tiles.json [mosaic_v3_tiles]
          ssub-rst8: the plain job's tiles with, at each LST the selection names for (scope, variant), the archived striation-
          corrected master (mosaic_v3.SSUB_PATTERN, exactly one; G1 on its own nights, G8; its nights must be the plain master's);
          weighting maps = the plain job's (S2B); refuses unless every selected LST was substituted -> stage/tiles.json
          Vclean: per LST the archived cleaned Stokes V master (mosaic_v3.VCLEAN_PATTERN, exactly one; G1, G8), weighted by the maps
          of the plain mosaic_v3.VCLEAN_VARIANT job (the Robust 0 map serves Stokes V: coadd_jysr._paths); LSTs whose V and I
          masters hold other nights are recorded (meta nights_differ_from_weighting_master). Only the plain masters that stay tiles
          and the weighting maps are re-verified. A substituted job's stage needs the plain job's stage (refused before any unit).
  coadd   a node-local link tree of the staged tiles under the engine's names (coadd_jysr._paths), then coadd_jysr in process with the
          mosaic_coadd globals, product I or Vclean, tag '' or '_ssub-rst8', Jy/sr and Jy/beam. Engine verdicts: a MISSING tile
          refuses (mosaic.MISSING_TILE_ACTION); a tile the engine guard skips refuses (G9: every staged LST a tile); G1 is re-checked
          against today's data_decisions.csv (also at render).
          -> healpix/healpix_<stem>_<S><tag>_{jysr,jybeam}_nside2048.fits [mosaic_map], healpix_PSF_<stem>_<S><tag>_nside2048.fits
          [mosaic_psfmap], healpix_PBeff_<stem>_<S><tag>_nside2048.fits [mosaic_pbeff]; md5s in the record
  pkcorr  Stokes I jobs only (production ran none for Stokes V): lib/mosaic/pkcorr.run_band (MP) on the co-added Jy/sr map and its PSF
          map, mosaic_pkcorr.ROUNDS injection rounds through the real engine on the staged tiles; test_pkcorr. -> pkcorr/
          healpix_PKCORR_I_deep_<V>_<S><tag>_nside2048.fits [mosaic_pkcorr_map], peak_corr_curve_*.csv, peak_recovery_*.csv,
          pkcorr_*.json
  render  lib/mosaic/band_products.build of the two maps with their PSF map (the beam card), mosaic_render's width, pixel and
          southern edge. -> renders/<S>_<label><tag>_{jysr,jybeam}_{Mollweide,RADec}.fits [mosaic_render]
  content Stokes I: TEST 1 (lib/mosaic/content.run, MT run) at the bright isolated cataloged sources; Stokes V: TEST 1R
          (content.run_random, MT run_random) at mosaic_content.RAND_N seeded random positions; both domains; TEST 2: the files the
          test read are the tile list's, at the md5s verified in the unit. -> content/content_<stem>_<S><tag>.json
          [mosaic_content_report | mosaic_content_random_report]
  pkcorr and content read the coadd's record and maps (at its md5s), refuse when the engine weights now differ from the coadd's, and
  re-check G1.

The frame is v5's: a stage reads its predecessor's done record (never a glob) and refuses unless it was made from the tile list now in
stage/tiles.json (its md5); a unit removes its own earlier outputs first and, when it does not end done, everything it placed. A failed
content test or test_pkcorr removes the unit's outputs (the record keeps the results). Conformance v3 and the tables are PR-6d, the
archive push PR-6e.
"""
import glob
import json
import os

from exopipe.phase3.lib.mosaic import gen_common as GC                     # shared with v5: one copy of each rule
from exopipe.phase3.lib.mosaic.gen_common import _clear, _engine_as_coadd, _md5   # one md5 cache per process for v3 and v5
from exopipe.phase3.lib.mosaic.v5 import LSTS

STAGES = ('stage', 'coadd', 'pkcorr', 'render', 'content')
I_ONLY = ('pkcorr',)                  # stages a Stokes V job does not have (production ran no PKCORR for Vclean): has_stage
STEP = 'mosaic'
STRIATION_CHOICES = ('with', 'without', 'both')
_STG_GRID = ('NAXIS1', 'NAXIS2', 'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2', 'EQUINOX', 'RADESYS')


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


class Job:
    """One v3 mosaic job of a scope: product I (variant, plain or striation-corrected) or Vclean."""

    def __init__(self, product, variant, ssub=False):
        if product not in ('I', 'Vclean') or (product == 'Vclean' and ssub):
            raise ValueError(f'v3 job: product {product!r} ssub {ssub!r}')
        self.product, self.variant, self.ssub = product, variant, bool(ssub)

    def __eq__(self, o):
        return isinstance(o, Job) and (self.product, self.variant, self.ssub) == (o.product, o.variant, o.ssub)

    def __hash__(self):
        return hash((self.product, self.variant, self.ssub))

    def __repr__(self):
        return f'Job({self.label})'

    @property
    def kind(self):
        return 'vclean' if self.product == 'Vclean' else 'ssub' if self.ssub else 'plain'

    @property
    def tag(self):
        """The engine's run tag ('_ssub-rst8' for the striation-corrected maps; step3_coadd.sh --tag)."""
        return f'_{_cfg().masters_striation.TAG}' if self.ssub else ''

    @property
    def label(self):
        return 'Vclean' if self.product == 'Vclean' else f'I_{self.variant}{self.tag}'

    @property
    def stem(self):
        """coadd_jysr's output stem before the scope: V_deep_Taper_R0_clean (Vclean) or I_deep_<variant>."""
        return 'V_deep_Taper_R0_clean' if self.product == 'Vclean' else f'I_deep_{self.variant}'

    @property
    def render_label(self):
        """band_products' label: the variant (I) or V_deep_Taper_R0_clean (production make_band_products 1f719ce)."""
        return 'V_deep_Taper_R0_clean' if self.product == 'Vclean' else self.variant

    @property
    def robust(self):
        """The robust weighting of the job's weighting map (coadd_jysr._paths: every Stokes V product takes the Robust 0 map)."""
        return 'R075' if (self.product == 'I' and str(self.variant).endswith('R075')) else 'R0'

    def engine_name(self, scope):
        """The master's link name in the engine's tree (coadd_jysr._paths)."""
        return f'V_deep_Taper_R0_clean_pbcorr_{scope}.fits' if self.product == 'Vclean' else f'I_deep_{self.variant}_pbcorr_{scope}.fits'

    def pb_job(self):
        """The plain job whose stage holds this job's weighting maps (itself when plain)."""
        if self.product == 'Vclean':
            return Job('I', _cfg().mosaic_v3.VCLEAN_VARIANT)
        return Job('I', self.variant)


def unit_name(st, scope, job):
    return f'v3-{st}.{scope}_{job.label}'


def unit_dir(out, scope, job):
    return os.path.join(out, f'{scope}_{job.label}')


def paths(out, scope, job):
    """Every file path of the unit, by key (the weighting maps of a plain job under 'pb' by LST)."""
    from exopipe.phase3.lib.masters.pb_apply import deep_pb_name
    C = _cfg()
    d = unit_dir(out, scope, job)
    n = int(C.mosaic_coadd.NSIDE)
    s = f'{job.stem}_{scope}{job.tag}'
    p = {'tiles': os.path.join(d, 'stage', 'tiles.json'),
         'jysr': os.path.join(d, 'healpix', f'healpix_{s}_jysr_nside{n}.fits'),
         'psf': os.path.join(d, 'healpix', f'healpix_PSF_{s}_nside{n}.fits'),
         'jybeam': os.path.join(d, 'healpix', f'healpix_{s}_jybeam_nside{n}.fits'),
         'pbeff': os.path.join(d, 'healpix', f'healpix_PBeff_{s}_nside{n}.fits'),
         'pb': {lst: os.path.join(d, 'stage', 'pb', lst, deep_pb_name(scope, job.robust)) for lst in LSTS} if job.kind == 'plain' else {}}
    for dom in ('jysr', 'jybeam'):
        for k, suf in (('mollweide', 'Mollweide'), ('radec', 'RADec')):
            p[f'render_{dom}_{k}'] = os.path.join(d, 'renders', f'{scope}_{job.render_label}{job.tag}_{dom}_{suf}.fits')
    if job.product == 'I':
        from exopipe.phase3.lib.mosaic import pkcorr as PK
        for k, f in PK.names(job.variant, scope, job.tag, n).items():
            p[f'pkcorr_{k}'] = os.path.join(d, 'pkcorr', f)
    p['content'] = os.path.join(d, 'content', f'content_{job.stem}_{scope}{job.tag}.json')
    return p


def has_stage(job, st):
    """Does the job have stage st? Every stage but I_ONLY's for a Stokes V job (the one place that decides)."""
    return not (st in I_ONLY and job.product != 'I')


def outputs_of(st, out, scope, job):
    """[(path, product class)] the stage owes."""
    if not has_stage(job, st):
        raise ValueError(f'v3 stage {st}: no {st.upper()} for {job.label} (has_stage)')
    p = paths(out, scope, job)
    if st == 'stage':
        return [(p['tiles'], 'mosaic_v3_tiles')] + [(p['pb'][lst], 'mosaic_weight_pb') for lst in sorted(p['pb'])]
    if st == 'coadd':
        return [(p['jysr'], 'mosaic_map'), (p['psf'], 'mosaic_psfmap'), (p['jybeam'], 'mosaic_map'), (p['pbeff'], 'mosaic_pbeff')]
    if st == 'pkcorr':
        return [(p['pkcorr_map'], 'mosaic_pkcorr_map'), (p['pkcorr_curve'], 'mosaic_pkcorr_curve'),
                (p['pkcorr_recovery'], 'mosaic_pkcorr_recovery'), (p['pkcorr_json'], 'mosaic_pkcorr_report')]
    if st == 'content':
        return [(p['content'], 'mosaic_content_report' if job.product == 'I' else 'mosaic_content_random_report')]
    if st == 'render':
        return [(p[f'render_{d}_{k}'], 'mosaic_render') for d in ('jysr', 'jybeam') for k in ('mollweide', 'radec')]
    raise ValueError(f'unknown v3 stage {st!r}')


# ---- job selection ----------------------------------------------------------------------------------------------------------------

def _list(val, default, what, allowed):
    got = [x.strip() for x in str(val).split(',') if x.strip()] if val else list(default)
    bad = [x for x in got if x not in allowed]
    if bad:
        raise ValueError(f'{what}: unknown {bad} ({", ".join(allowed)})')
    return got


def striation_selection():
    """{(lst, band): [variant, ...]} of hybrid.STRIATION_SELECTION (refuses unless its md5 is the pinned one)."""
    from exopipe.phase3.lib.masters.deep_stages import striation_selection as sel
    return sel()


def selected_lsts(sel, scope, variant):
    return sorted(lst for (lst, band), vs in sel.items() if band == scope and variant in vs)


def jobs_for(scope, products, striation, variants, sel):
    """The jobs of a scope in dependency order: the plain jobs, then the striation-corrected ones, then Vclean (both read the plain
    stage's weighting maps)."""
    C = _cfg().mosaic_v3
    jobs = []
    if 'I' in products:
        if striation in ('without', 'both'):
            jobs += [Job('I', v) for v in variants]
        if striation in ('with', 'both'):
            jobs += [Job('I', v, ssub=True) for v in variants if selected_lsts(sel, scope, v)]
    if 'Vclean' in products:
        jobs.append(Job('Vclean', C.VCLEAN_VARIANT))
    return jobs


def select_stages(args):
    return GC.select_stages(args, STAGES, 'v3')


def choices(args):
    """(scopes, products, variants, striation) of the command line, defaults from mosaic_v3."""
    C = _cfg().mosaic_v3
    scopes = _list(getattr(args, 'scopes', None), C.SCOPES, '--scopes', C.SCOPES)
    products = _list(getattr(args, 'products', None), C.PRODUCTS, '--products', C.PRODUCTS)
    variants = _list(getattr(args, 'variants', None), C.VARIANTS, '--variants', C.VARIANTS)
    striation = getattr(args, 'striation', None) or C.STRIATION
    if striation not in STRIATION_CHOICES:
        raise ValueError(f'--striation {striation!r} (with, without, both)')
    return scopes, products, variants, striation


def selection_for(products, striation):
    """The striation selection when striation-corrected jobs are wanted, else {} (never read)."""
    return striation_selection() if ('I' in products and striation in ('with', 'both')) else {}


def select_units(args, sel=None):
    """[(stage, scope, job)] in pipeline order: every stage of a job before the next job, the jobs of a scope in jobs_for's order."""
    scopes, products, variants, striation = choices(args)
    if sel is None:
        sel = selection_for(products, striation)
    stages = select_stages(args)
    return [(st, sc, j) for sc in scopes for j in jobs_for(sc, products, striation, variants, sel) for st in stages
            if has_stage(j, st)]


def missing_plain_stages(units, out):
    """A substituted job's stage reads the plain job's weighting maps: the plain stage must be in this run or done in out."""
    from exopipe.phase3.core.record import load_record
    made = {(sc, j) for st, sc, j in units if st == 'stage' and j.kind == 'plain'}
    probs = []
    for st, sc, j in units:
        if st != 'stage' or j.kind == 'plain' or (sc, j.pb_job()) in made:
            continue
        pj = j.pb_job()
        r = load_record(out, STEP, unit_name('stage', sc, pj))
        if not r or r.get('status') != 'done':
            probs.append(f'{sc} {j.label} needs the done stage of {sc} {pj.label} (its weighting maps): add it to this run '
                         f'(--products I --variants {pj.variant} --striation without|both) or run it first')
    return probs


# ---- records and tile lists -------------------------------------------------------------------------------------------------------

def _upstream(u, out, st, scope, job):
    """The done record of stage st of (scope, job), else refuse."""
    return GC.upstream(u, out, STEP, unit_name(st, scope, job), f'{scope} {job.label}', st)


def _tiles(u, out, scope, job, verify=True, masters=None):
    """The staged tiles of (scope, job) (gen_common.staged_tiles) of its done stage unit. verify: every weighting map, and the tile
    master of every LST in masters (default all), must still have the md5 the stage recorded; those are the unit's inputs."""
    _upstream(u, out, 'stage', scope, job)
    return GC.staged_tiles(u, paths(out, scope, job)['tiles'], f'{scope} {job.label}', verify, masters)


def _same_tiles(u, scope, job, cur, rec, st):
    GC.same_tiles(u, f'{scope} {job.label}', cur, rec, st)


def _link_tree(t, scope, job, sd):
    """The staged tiles and their weighting maps linked under the engine's names (coadd_jysr._paths) in the scratch dir sd."""
    from exopipe.phase3.lib.masters.pb_apply import deep_pb_name
    return GC.link_tree(t, sd, job.engine_name(scope), deep_pb_name(scope, job.robust))


def _coadd_maps(u, out, scope, job, co, keys):
    """The coadd's maps (keys of paths) at their recorded md5s, recorded as inputs; else refuse."""
    P = paths(out, scope, job)
    GC.coadd_maps(u, {k: P[k] for k in keys}, co)


def _from_coadd(u, out, scope, job, keys):
    """(tiles, coadd record) for a stage after the coadd: the coadd done and made from the tile list now staged (every tile and
    weighting map at its staged md5), no tile night excluded since, the coadd's maps of keys at their md5s, the engine weights the
    coadd's."""
    co = _upstream(u, out, 'coadd', scope, job)
    t = _tiles(u, out, scope, job)
    _same_tiles(u, scope, job, t['json_md5'], co, 'coadd')
    _recheck_exclusions(u, scope, job, t)
    _coadd_maps(u, out, scope, job, co, keys)
    _engine_as_coadd(u, scope, job.label, co)                  # v5's check, labeled with the v3 job
    return t, co


def _write_tiles(u, p, scope, job, tiles, **extra):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + '.part', 'w') as f:
        json.dump(dict(scope=scope, job=job.label, product=job.product, variant=job.variant, tag=job.tag, tiles=tiles, **extra), f,
                  indent=1)
    os.replace(p + '.part', p)
    u.add_output(p, 'mosaic_v3_tiles', scope=scope, job=job.label)


def _archived(lst, scope, pattern, **fmt):
    """Every file of mosaic.INPUT_COADD_RUN/<lst>/<scope>/ matching pattern."""
    C = _cfg()
    d = os.path.join(C.mosaic.INPUT_COADD_RUN, lst, scope)
    return sorted(glob.glob(os.path.join(glob.escape(d), pattern.format(scope=scope, lst=lst, **fmt))))


def _gate_master(lst, scope, path, ex, pb_cards=True):
    """([problems], tile_header) of an archived master a v3 job substitutes: G1 (its own constituent nights, every one admitted) and
    G8 (PBCORR True, beam, frequency; the PB factor cards when pb_cards - the striation-corrected masters carry none)."""
    from exopipe.phase3.lib.mosaic import steps_common as SC
    C = _cfg()
    try:
        h = SC.tile_header(path)
    except Exception as e:
        return [f'G8 {lst}: header of {os.path.basename(path)} unreadable ({type(e).__name__}: {e})'], None
    probs = []
    if not h['dates']:
        probs.append(f"G1 {lst}: no 'COADD constituent' HISTORY in {os.path.basename(path)}")
    else:
        bad = ex.excluded_constituents(lst, scope, h['dates'])
        if bad:
            probs.append(f'G1 {lst}: {os.path.basename(path)} contains night(s) {", ".join(bad)} that data_decisions.csv excludes for '
                         f'{lst} {scope}')
    miss = [k for k in (('PBMODEL', 'PBCHROM', 'PBRESID') if pb_cards else ()) + ('BMAJ', 'BMIN') if h.get(k) in (None, '')]
    if h.get('PBCORR') is not True:
        probs.append(f'G8 {lst}: PBCORR {h.get("PBCORR")!r} in {os.path.basename(path)}, not True')
    if miss:
        probs.append(f'G8 {lst}: {os.path.basename(path)} lacks {", ".join(miss)}')
    if h.get('freq_hz') is None:
        probs.append(f'G8 {lst}: no frequency above {C.mosaic_coadd.MIN_FREQ_HZ:g} Hz in {os.path.basename(path)}')
    return probs, h


def _recheck_exclusions(u, scope, job, t):
    """G1 again at coadd and render: a tile night that data_decisions.csv excludes NOW refuses, though the stage admitted it (no
    mosaic re-admits an excluded night)."""
    from exopipe.phase3.core import exclusions
    ex = exclusions.load()
    bad = [f'{r["lst"]} {", ".join(b)}' for r in t['tiles'] for b in [ex.excluded_constituents(r['lst'], scope, r['nights'])] if b]
    _refuse_list(u, f'G1 {scope} {job.label}: night(s) data_decisions.csv excludes since the stage', bad)


def _refuse_list(u, what, probs):
    if probs:
        u.refuse(f'{what}: ' + '; '.join(probs[:6]) + (f' (+{len(probs) - 6} more)' if len(probs) > 6 else '') + ' - REFUSED')


# ---- stage ------------------------------------------------------------------------------------------------------------------------

def derive_weight_pb(scope, lst, ref, robust, dest):
    """The weighting PB map of one tile (STG 'PB step 2'): pb_apply.wideband_pb of the scope from the Taper master `ref` of the robust
    weighting (its constituent nights set the subband weights, its beam the bandwidth smearing), written float32 with STG's header
    (grid cards of ref, PBSCOPE, PBFOR, PBROBUST, PBNDATE, PBCHROM, PBRESID, the never-divide COMMENT). Returns wideband_pb's meta."""
    import numpy as np
    from astropy.io import fits
    from exopipe.phase3.lib.masters import pb_apply
    pb, meta = pb_apply.wideband_pb(scope, lst, ref)
    if pb is None:
        raise RuntimeError(f'wideband_pb returned no map for {lst} {scope} ({os.path.basename(ref)})')
    h = fits.getheader(ref, memmap=False)
    hdr = fits.Header()
    for c in _STG_GRID:
        if c in h:
            hdr[c] = h[c]
    hdr['PBSCOPE'] = (scope, 'band this PB is for')
    hdr['PBFOR'] = (os.path.basename(ref)[:60], 'master whose constituents set the weights')
    hdr['PBROBUST'] = ('-0.75' if robust == 'R075' else '0', 'robust weighting whose beam sets factor 2')
    hdr['PBNDATE'] = (len(meta.get('used_dates', []) or []), 'constituent nights used')
    hdr['PBCHROM'] = (str(meta.get('chromatic', '')), 'factor 2: bandwidth smearing')
    hdr['PBRESID'] = (str(meta.get('residual_pb', '')), 'factor 3: measured residual gain')
    hdr['COMMENT'] = 'Weighting PB for the mosaic; masters are ALREADY pbcorr - never divide by this.'
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    fits.PrimaryHDU(np.asarray(pb, np.float32), hdr).writeto(dest + '.part', overwrite=True)
    os.replace(dest + '.part', dest)
    return meta


def plan_plain(scope, job):
    """steps_common.plan_job of the plain job over the 24 LSTs (G1, G3, G4, G8; the weighting map's reference master and G1 on its
    nights where it is another variant)."""
    from exopipe.phase3.core import exclusions
    from exopipe.phase3.lib.mosaic import steps_common as SC
    return SC.plan_job(SC.Job(scope, job.variant, 'I'), list(LSTS), SC.lst_universe(), exclusions.load())


def stage_plain(u, scope, job, out):
    C = _cfg()
    P = paths(out, scope, job)
    plan = plan_plain(scope, job)
    _refuse_list(u, f'{scope} {job.label}', [f'{g} {m}' for g, m in plan.refusals])
    tiles, probs = [], []
    for lst in LSTS:
        t = plan.tiles[lst]
        m = t['master']
        ref = (t.get('pb_ref') or {}).get('master') or m
        try:
            meta = derive_weight_pb(scope, lst, ref, job.robust, P['pb'][lst])
        except Exception as e:
            probs.append(f'G5 {lst}: no weighting PB ({type(e).__name__}: {str(e)[:160]})')
            continue
        h = t['hdr']
        row = dict(lst=lst, path=m, md5=_md5(m), pb=P['pb'][lst], pb_md5=_md5(P['pb'][lst]), pbfor=ref, nights=h['dates'],
                   BMAJ=h.get('BMAJ'), BMIN=h.get('BMIN'), kind='plain', pb_used_dates=len(meta.get('used_dates', []) or []))
        u.add_input(m, extra={'md5': row['md5']})
        if ref != m:
            u.add_input(ref)
        tiles.append(row)
    _refuse_list(u, f'{scope} {job.label}', probs)
    n = int(C.mosaic_v3.REQUIRE_N_LSTS)
    if len(tiles) < n:
        u.refuse(f'{len(tiles)} tiles < mosaic_v3.REQUIRE_N_LSTS {n} - REFUSED')
    u.set_admitted(plan.admitted)
    for r in tiles:
        u.add_output(r['pb'], 'mosaic_weight_pb', master=r['pbfor'], robust=job.robust)
    _write_tiles(u, P['tiles'], scope, job, tiles)


def stage_substituted(u, scope, job, out, sel):
    """The striation-corrected and the Stokes V tile lists: the plain job's weighting maps, other masters (module docstring)."""
    from exopipe.phase3.core import exclusions
    C = _cfg()
    P = paths(out, scope, job)
    pj = job.pb_job()
    want = selected_lsts(sel, scope, job.variant) if job.kind == 'ssub' else list(LSTS)
    pt = _tiles(u, out, scope, pj, masters=set(LSTS) - set(want))    # the plain masters that stay tiles; every weighting map
    ex = exclusions.load()
    tok = C.mosaic.VARIANT_TOKENS[job.variant]
    tiles, probs, n_sub, nd = [], [], 0, {}
    for r in pt['tiles']:
        lst = r['lst']
        row = dict(r, kind='plain')
        if lst in want:
            if job.kind == 'ssub':
                g = _archived(lst, scope, C.mosaic_v3.SSUB_PATTERN, var=tok, tag=C.masters_striation.TAG)
            else:
                g = _archived(lst, scope, C.mosaic_v3.VCLEAN_PATTERN)
            if len(g) != 1:
                probs.append(f'{lst}: {len(g)} archived {job.kind} masters' + (f' {[os.path.basename(x) for x in g]}' if g else ''))
                continue
            bad, h = _gate_master(lst, scope, g[0], ex, pb_cards=job.kind == 'vclean')
            if bad:
                probs += bad
                continue
            if h['dates'] != r['nights']:
                diff = dict(only_substitute=sorted(set(h['dates']) - set(r['nights'])), only_plain=sorted(set(r['nights']) - set(h['dates'])))
                if job.kind == 'ssub':     # a corrected copy of the plain master: other nights = made from an older or newer master
                    probs.append(f'G1 {lst}: {os.path.basename(g[0])} nights differ from the plain master it replaces {diff}')
                    continue
                # Stokes V: production weights the cleaned-V tile with the map derived from the Taper R0 Stokes I master's nights
                # (coadd_jysr._paths); the two masters' nights differ in 26 of the 456 archived (scope, LST) cells (2026-11-18)
                nd[lst] = diff
            row.update(path=g[0], md5=_md5(g[0]), nights=h['dates'], BMAJ=h.get('BMAJ'), BMIN=h.get('BMIN'), kind=job.kind,
                       plain_path=r['path'], plain_md5=r['md5'], plain_nights=r['nights'])
            n_sub += 1
            u.add_input(g[0], extra={'md5': row['md5']})
        tiles.append(row)
    _refuse_list(u, f'{scope} {job.label}', probs)
    if n_sub != len(want) or len(tiles) != len(pt['tiles']):
        u.refuse(f'{scope} {job.label}: {n_sub} of the {len(want)} LSTs substituted - REFUSED')
    u.set_admitted(sorted({f'{r["lst"]}:{d}' for r in tiles for d in r['nights']}))
    u.meta.update(n_substituted=n_sub, substituted=want, plain_tiles_md5=pt['json_md5'], nights_differ_from_weighting_master=nd)
    if nd:
        u.note(f'{len(nd)} LST(s) whose weighting map comes from a master with other nights (production\'s Stokes V weighting): '
               f'{sorted(nd)}')
    _write_tiles(u, P['tiles'], scope, job, tiles, plain_job=pj.label, plain_tiles_md5=pt['json_md5'])


def stage_unit(u, scope, job, out, sel):
    from exopipe.phase3.lib.mosaic import tests_v3  # noqa: F401  (registers mosaic_v3_tiles)
    for pp, c in outputs_of('stage', out, scope, job):
        u.plan(pp, c)
    _clear([pp for pp, _c in outputs_of('stage', out, scope, job)])
    if job.kind == 'plain':
        stage_plain(u, scope, job, out)
    else:
        stage_substituted(u, scope, job, out, sel)


# ---- coadd ------------------------------------------------------------------------------------------------------------------------

def coadd_unit(u, scope, job, out, work):
    import shutil
    from exopipe.phase3.core import fitsio
    C = _cfg()
    P = paths(out, scope, job)
    own = [pp for pp, _c in outputs_of('coadd', out, scope, job)]
    for pp, c in outputs_of('coadd', out, scope, job):
        u.plan(pp, c)
    _clear(own)
    t = _tiles(u, out, scope, job)
    _recheck_exclusions(u, scope, job, t)
    lsts = [r['lst'] for r in t['tiles']]
    sd = fitsio.scratch_dir(f'mosaic_v3_{scope}_{job.label}', work)
    tmp = os.path.join(unit_dir(out, scope, job), f'.tmp_coadd_{os.getpid()}')
    try:
        _link_tree(t, scope, job, sd)
        verdicts = GC.run_engine(u, lsts, job.variant, scope, sd, tmp, job.tag, job.product)
        missing = {dom: [l for l in lsts if vd.get(l) in (None, 'MISSING')] for dom, vd in verdicts.items()}
        missing = {k: x for k, x in missing.items() if x}
        if missing:
            u.refuse(f'G3/G5 {scope} {job.label}: tiles missing or without a verdict {missing} (mosaic.MISSING_TILE_ACTION '
                     f'{C.mosaic.MISSING_TILE_ACTION}) - REFUSED')
        # every staged LST a tile (mosaic_v3.REQUIRE_N_LSTS): a tile the engine guard skips refuses, whatever mosaic.TILE_GUARD_ACTION
        # (v2's) says - as v5's HG14; production's 270 FEATHER3 co-add logs never skip one
        skipped = {dom: sorted((l, x) for l, x in vd.items() if x != 'added') for dom, vd in verdicts.items()}
        skipped = {k: x for k, x in skipped.items() if x}
        if skipped:
            u.refuse(f'G9 {scope} {job.label}: tiles the engine did not add {skipped} - REFUSED')
        for k in ('jysr', 'psf', 'jybeam', 'pbeff'):
            if not os.path.exists(os.path.join(tmp, os.path.basename(P[k]))):
                u.refuse(f'the engine did not write {os.path.basename(P[k])} - REFUSED')
        os.makedirs(os.path.dirname(P['jysr']), exist_ok=True)
        for k in ('jysr', 'psf', 'jybeam', 'pbeff'):
            os.replace(os.path.join(tmp, os.path.basename(P[k])), P[k])
    finally:
        fitsio.cleanup(sd)
        if os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
    u.meta['md5'] = {k: _md5(P[k]) for k in ('jysr', 'psf', 'jybeam', 'pbeff')}
    u.meta['tiles_md5'] = t['json_md5']
    vj = verdicts.get('jysr', {})
    tiles = [dict(lst=r['lst'], verdict=vj.get(r['lst']), realpath=os.path.realpath(r['path']), BMAJ=r.get('BMAJ'), BMIN=r.get('BMIN'))
             for r in t['tiles']]
    ctx = dict(tiles=tiles, partial=False, scope=scope, tag=job.tag, psfmap=P['psf'], jysr_map=P['jysr'], jybeam_map=P['jybeam'])
    for pp, c in outputs_of('coadd', out, scope, job):
        u.add_output(pp, c, domain='jybeam' if pp in (P['jybeam'], P['pbeff']) else 'jysr', **ctx)


# ---- render -----------------------------------------------------------------------------------------------------------------------

def render_unit(u, scope, job, out, work):
    import shutil
    from exopipe.phase3.core import fitsio
    from exopipe.phase3.lib.mosaic import band_products as BP
    R = _cfg().mosaic_render
    P = paths(out, scope, job)
    own = [pp for pp, _c in outputs_of('render', out, scope, job)]
    for pp, c in outputs_of('render', out, scope, job):
        u.plan(pp, c)
    _clear(own)
    co = _upstream(u, out, 'coadd', scope, job)
    if not os.path.exists(P['tiles']):
        u.refuse(f'{P["tiles"]} is missing - REFUSED')
    cur = _md5(P['tiles'])
    _same_tiles(u, scope, job, cur, co, 'coadd')
    u.add_input(P['tiles'], extra={'md5': cur})
    with open(P['tiles']) as f:
        _recheck_exclusions(u, scope, job, json.load(f))
    _coadd_maps(u, out, scope, job, co, ('jysr', 'jybeam', 'psf'))
    ld = fitsio.scratch_dir(f'mosaic_v3_render_{scope}_{job.label}', work)
    tmp = os.path.join(unit_dir(out, scope, job), f'.tmp_render_{os.getpid()}')
    try:
        for k in ('jysr', 'jybeam', 'psf'):
            os.symlink(P[k], os.path.join(ld, os.path.basename(P[k])))
        BP.BEAMS = dict(R.BEAM_FALLBACK_ARCMIN)
        os.makedirs(tmp, exist_ok=True)
        for dom in ('jysr', 'jybeam'):
            BP.build(job.variant, ld, f'{job.tag}_{dom}', tmp, bands=[scope], moll_W=int(R.MOLLWEIDE_WIDTH_PX),
                     car_cdelt=float(R.CAR_CDELT_ARCMIN) / 60.0, dec_cut=R.RENDER_DEC_CUT_DEG, product=job.product)
        for pp in own:
            if not os.path.exists(os.path.join(tmp, os.path.basename(pp))):
                u.refuse(f'band_products did not write {os.path.basename(pp)} - REFUSED')
        os.makedirs(os.path.dirname(own[0]), exist_ok=True)
        for pp in own:
            os.replace(os.path.join(tmp, os.path.basename(pp)), pp)
    finally:
        fitsio.cleanup(ld)
        shutil.rmtree(tmp, ignore_errors=True)
    u.meta['tiles_md5'] = cur
    for dom in ('jysr', 'jybeam'):
        for k in ('mollweide', 'radec'):
            u.add_output(P[f'render_{dom}_{k}'], 'mosaic_render', source=P[dom], psfmap=P['psf'], kind=k, scope=scope,
                         dec_cut=R.RENDER_DEC_CUT_DEG)


# ---- pkcorr -----------------------------------------------------------------------------------------------------------------------

def pkcorr_unit(u, scope, job, out, work):
    """MP run_band on the co-added Jy/sr map and its PSF map (production FEATHER3: healpix/healpix_PKCORR_*, ROUNDS 5, the plain and
    the striation-corrected Stokes I maps; no Stokes V map)."""
    from exopipe.phase3.lib.mosaic import tests_v5  # noqa: F401
    P = paths(out, scope, job)
    own = [pp for pp, _c in outputs_of('pkcorr', out, scope, job)]
    for pp, c in outputs_of('pkcorr', out, scope, job):
        u.plan(pp, c)
    _clear(own)
    t, co = _from_coadd(u, out, scope, job, ('jysr', 'psf'))
    GC.pkcorr(u, what=f'{scope} {job.label}', scope=scope, variant=job.variant, tag=job.tag, t=t,
              maps={k: P[k] for k in ('jysr', 'psf')}, outp={k: P[f'pkcorr_{k}'] for k in ('map', 'curve', 'recovery', 'json')},
              link=lambda sd: _link_tree(t, scope, job, sd),
              scratch=(f'mosaic_v3_pkcorr_{scope}_{job.label}', f'mosaic_v3_pkcorr_inj_{scope}_{job.label}'),
              udir=unit_dir(out, scope, job), work=work, label=job.label, co=co)


# ---- content ----------------------------------------------------------------------------------------------------------------------

def content_unit(u, scope, job, out, work):
    """Stokes I: TEST 1 at the bright cataloged sources (content.run); Stokes V: TEST 1R at random positions (content.run_random); both
    domains, then TEST 2 (the files read = the tile list's, at the md5s verified in this unit)."""
    import glob
    from exopipe.phase3.core import fitsio
    from exopipe.phase3.lib.mosaic import content as CT
    from exopipe.phase3.lib.mosaic import tests_v3, tests_v5  # noqa: F401
    C = _cfg()
    P = paths(out, scope, job)
    for pp, c in outputs_of('content', out, scope, job):
        u.plan(pp, c)
    _clear([P['content']])
    t, _co = _from_coadd(u, out, scope, job, ('jysr', 'jybeam'))
    maps = {dom: P[dom] for dom in ('jysr', 'jybeam')}
    sd = fitsio.scratch_dir(f'mosaic_v3_content_{scope}_{job.label}', work)
    try:
        _link_tree(t, scope, job, sd)
        if job.product == 'I':
            parents = C.mosaic_content.PARENTS
            pat = f'master_R075_*_{scope}.gaul.csv'
            cats = sorted(os.path.basename(x) for x in glob.glob(os.path.join(glob.escape(parents), pat)))
            if not cats:
                u.refuse(f'{scope} {job.label}: no parent catalog {pat} in mosaic_content.PARENTS {parents} - REFUSED')
            u.meta['parents'] = {'dir': parents, 'n_catalogs': len(cats)}
            rep = CT.run(maps, sd, job.variant, scope, parents)
        else:
            rep = CT.run_random(maps, sd, job.variant, scope, product=job.product)
    finally:
        fitsio.cleanup(sd)
    content_md5 = GC.test2_write(rep, t, P['content'])          # TEST 2 provenance (shared with v5)
    cls = outputs_of('content', out, scope, job)[0][1]
    for dom, r in rep.items():
        if job.product == 'I':
            u.check(f'TEST1.{dom}', r['result'] == 'PASS', path=P['content'], cls=cls,
                    value={k: r[k] for k in ('n_tiles', 'n_sources', 'median_ratio', 'p05', 'p95', 'frac_within_5pct', 'result')},
                    detail=f'{r["n_sources"]} sources, median map / tiles {r["median_ratio"]:.5f}, within 5 % {r["frac_within_5pct"]:.3f}')
        else:
            u.check(f'TEST1R.{dom}', r['result'] == 'PASS', path=P['content'], cls=cls,
                    value={k: r[k] for k in ('n_tiles', 'n_positions', 'scale', 'median_absdiff_over_scale', 'p90_absdiff_over_scale',
                                             'result')},
                    detail=f'{r["n_positions"]} positions, median |map - tiles| / scale {r["median_absdiff_over_scale"]:.4f}, '
                           f'90th percentile {r["p90_absdiff_over_scale"]:.4f}')
        GC.test2_check(u, dom, r, len(t['tiles']), P['content'], cls)
    u.meta['tiles_md5'] = t['json_md5']
    u.meta['md5'] = {'content': content_md5}                  # as v5.content_unit: a later stage (conform) checks the report it reads
    u.add_output(P['content'], cls, n_tiles=len(t['tiles']), scope=scope, variant=job.variant, job=job.label)


# ---- the step ---------------------------------------------------------------------------------------------------------------------

def out_problems(out):
    """--out is a work run of its own: never inside or above the masters it reads or a mosaic archive run."""
    C = _cfg()
    held = {'mosaic.INPUT_COADD_RUN': C.mosaic.INPUT_COADD_RUN, 'hybrid.ARCHIVE_MASTERS': C.hybrid.ARCHIVE_MASTERS,
            'hybrid_tilefill.FILLED_MOSAIC_RUN': C.hybrid_tilefill.FILLED_MOSAIC_RUN}
    held.update({f'hybrid_tilefill.UNFILLED_MOSAIC_RUNS[{i}]': r for i, r in enumerate(C.hybrid_tilefill.UNFILLED_MOSAIC_RUNS)})
    return GC.held_problems(out, held)


def run(args, run):
    from exopipe.phase3.core.record import NoInput, Refusal
    bad = out_problems(run.out_root)
    if bad:
        raise Refusal('; '.join(bad))
    _scopes, products, _variants, striation = choices(args)
    sel = selection_for(products, striation)
    units = select_units(args, sel)
    run.meta = {'generation': 'v3', 'units': len(units), 'striation': striation, 'products': products}
    if not units:
        if 'I' not in products and set(select_stages(args)) <= set(I_ONLY):
            raise NoInput(f'no unit: Stokes V jobs have no {",".join(I_ONLY)} stage (production ran no PKCORR for Stokes V)')
        raise NoInput('no unit: the selected products, variants and striation choice give no job')
    need = missing_plain_stages(units, run.out_root)
    if need:
        raise Refusal('; '.join(need[:4]) + (f' (+{len(need) - 4} more)' if len(need) > 4 else ''))
    if run.dry_run:
        for st, sc, j in units:
            what = ', '.join(os.path.relpath(p, run.out_root) for p, _c in outputs_of(st, run.out_root, sc, j)[:4])
            if st == 'stage' and j.kind == 'plain':
                plan = plan_plain(sc, j)
                what = (f'refused: {"; ".join(f"{g} {m}" for g, m in plan.refusals[:3])}' if plan.refusals else
                        f'{len(plan.masters)} masters, {len(LSTS)} weighting maps -> {what} ...')
            elif st == 'stage' and j.kind == 'ssub':
                what = f'LSTs {",".join(selected_lsts(sel, sc, j.variant))} substituted -> {what}'
            print(f'  plan {unit_name(st, sc, j)}: {what}', flush=True)
        print(f'exopipe3 mosaic --generation v3: DRY RUN, {len(units)} units, nothing written', flush=True)
        return 0
    work = getattr(args, 'work', None)
    for st, sc, j in units:
        own = [p for p, _c in outputs_of(st, run.out_root, sc, j)]
        with run.unit(unit_name(st, sc, j), stage=st, scope=sc, job=j.label, generation='v3') as u:
            try:
                if st == 'stage':
                    stage_unit(u, sc, j, run.out_root, sel)
                elif st == 'coadd':
                    coadd_unit(u, sc, j, run.out_root, work)
                elif st == 'pkcorr':
                    pkcorr_unit(u, sc, j, run.out_root, work)
                elif st == 'render':
                    render_unit(u, sc, j, run.out_root, work)
                else:
                    content_unit(u, sc, j, run.out_root, work)
            except BaseException:
                _clear(own)                   # nothing of a unit that did not end done stays in place
                raise
            if u.failed_tests():
                _clear(own)
                u.note(f'a content test failed: the {len(own)} outputs were removed (the record keeps the results)')
    return run.exit_code()
