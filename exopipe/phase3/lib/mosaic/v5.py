"""`exopipe3 mosaic --generation v5`: the hybrid mosaics (spec plan/specs/hybrid.md 2.8, 2.10-2.16; PR-7a, 7b, 7c). Unit of work: (stage,
scope, weighting); records <out>/_records/mosaic/v5-<stage>.<scope>_<w>.json; files under <out>/<scope>_<w>/.

  stage   the 24 UNFILLED beam-corrected hybrid masters of the scope and weighting (HG14: every LST, no Taper-only stand-in), one per LST
          through steps/hybrid.tilefill_source: 'work' when --hybrid-out holds the cell's hybrid record (its v2 unit done, md5 = the
          record's), else 'pinned' (the installed filled master's TPSRCMD5 names its unfilled source in a pinned manifest, HG13); a
          night now excluded refuses (no mosaic re-admits an excluded night). Each tile's weighting map is the archived full PB of its
          weighting (hybrid.ARCHIVE_MASTERS/<LST>/<scope>/PB_deep[_R075]_<scope>.fits; NL stage_mosaic 1704). A filled tile refuses
          (the fill is the mosaic's own stage). -> stage/tiles.json                                            [mosaic_v5_tiles]
  coadd   a node-local link tree of the staged tiles under the engine's names (coadd_jysr._paths), then lib/mosaic/coadd_jysr.
          coadd_jysr in process with the mosaic_coadd globals, tag mosaic_v5.TAG, in Jy/sr and Jy/beam; every tile must be 'added'
          (HG14). -> unfilled/healpix_I_deep_<V>_<S>_lsfill_{jysr,jybeam}_nside2048.fits [mosaic_map], healpix_PSF_..._lsfill_
          nside2048.fits [mosaic_psfmap], healpix_PBeff_..._lsfill_nside2048.fits [mosaic_pbeff]; md5s in the record
  pkcorr  lib/mosaic/pkcorr (MP): mosaic_pkcorr.ROUNDS rounds of injected 50 Jy sources in node-local copies of the staged tiles, each
          co-added by the engine and differenced against the UNFILLED Jy/sr map (production's --clean-map), Omega_eff from the PSF
          map; test_pkcorr inline. -> pkcorr/healpix_PKCORR_I_deep_<V>_<S>_lsfill_nside2048.fits [mosaic_pkcorr_map],
          peak_corr_curve_<V>_<S>_lsfill.csv [mosaic_pkcorr_curve], peak_recovery_... .csv [mosaic_pkcorr_recovery], pkcorr_... .json
          [mosaic_pkcorr_report]
  caveat  hybrid.CAVEAT_SCOPES only: lib/mosaic/caveat.caveat_map with the mosaic's tile weights; test_caveat_map against the unfilled
          PB_eff map. -> caveat/healpix_CAVEAT_I_deep_<V>_<S>_lsfill_nside128.fits [mosaic_caveat_map], caveats_<V>_<S>_lsfill.csv
          [mosaic_caveat_table]
  fill    lib/mosaic/fill.fill_file of the unfilled Jy/sr map (HG19: the test on the written file must PASS), then fill_jybeam_file.
          -> healpix/healpix_I_deep_<V>_<S>_lsfill_{jysr,jybeam}_nside2048.fits [mosaic_filled_map]; md5s in the record
  render  lib/mosaic/band_products.build of the FILLED maps with their PSF map (the beam card), mosaic_render's width, pixel and
          southern edge. -> renders/<S>_<V>_lsfill_{jysr,jybeam}_{Mollweide,RADec}.fits [mosaic_render]
  content lib/mosaic/content (MT): TEST 1 of the UNFILLED Jy/sr and Jy/beam maps against their own staged tiles, and TEST 2 (the
          links and the md5 of each tile's first 64 MB), on the link tree of the coadd. -> content/content_I_deep_<V>_<S>_lsfill.json
          [mosaic_content_report]. Production's V5 tables/summary.csv rows are the unfilled maps' values; TEST 1R is the Stokes V
          test (the hybrid mosaics are Stokes I only)
  conform lib/mosaic/conform_v5 (MC v5): the archive-staging copies of the unit's products under the archive run's names and layout,
          stamped (PRODNAME, the time block of the 24 tiles, the median beam, PROVBAS, BEAMNOTE, the PROVENANCE rows, the Mosaic
          HISTORY line). Refuses unless: the coadd, pkcorr, fill, render, content (and, in hybrid.CAVEAT_SCOPES, caveat) records are
          done, made from the current tiles.json, and hold every test passed (HG15 test_pkcorr, HG17 TEST 1 and TEST 2, HG18
          test_caveat_map, HG19); every file it copies has its stage's md5; the tiles resolved now (sources, nights: an excluded night
          refuses) equal tiles.json; the classic Taper Jy/sr maps of the scope are on Lustre with a PASS row (HG16, MC on_lustre).
          -> conform/<S>/I/{healpix,mollweide,radec}/<product name>, and the unfilled Jy/sr and Jy/beam twins under the filled maps'
          names in conform/unfilled/<S>/I/healpix/ [mosaic_v5_product]; md5s and the table rows in the record (the band field of
          `exopipe3 hybrid --stages tilefill --mosaic-out <out>` reads the conformed Jy/sr pair, so the filled masters' TPMOSAIC and
          TPMOSMD5 name the archived mosaic)
  tables  one unit per run (v5-tables): the V5 tables of hybrid_tilefill.FILLED_MOSAIC_RUN with the rows of every done conform unit
          of --out replaced; refuses unless every scope of mosaic_v5.SCOPES in both weightings has its rows and every result is PASS.
          -> tables/{summary.csv, pkcorr_curves.csv, summary_tpfill_Taper_<w>.csv, caveats_<V>_<S>_lsfill.csv, taper_only.csv};
          their md5s in the record (`exopipe3 archive push --class mosaic_v5` checks them)
          [mosaic_v5_table]

A stage reads its predecessors' unit records (status done), never a glob, and refuses unless they were made from the tile list now
in stage/tiles.json (its md5 in the coadd and fill records). A unit removes its own earlier outputs first; the fill renames a map
into place only when its inline test passed; a unit that does not end done (a refusal, an error, a failed content test) removes the
outputs it placed, and its record keeps the reason. The stages that read the tiles (coadd, pkcorr, caveat, content) bind the
mosaic_coadd globals and refuse unless they equal the coadd record's. The push class (mosaic_v5) is PR-7d.
"""
import json
import os

LSTS = tuple(f'{h:02d}h' for h in range(24))
STAGES = ('stage', 'coadd', 'pkcorr', 'caveat', 'fill', 'render', 'content', 'conform', 'tables')
#   spec 3.1 (exopipe3 mosaic --generation v5 --stages stage,coadd,pkcorr,caveat,fill,render,content,conform,tables)
STEP = 'mosaic'


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def unit_name(st, scope, w):
    return f'v5-{st}.{scope}_{w}'


def unit_dir(out, scope, w):
    return os.path.join(out, f'{scope}_{w}')


def variant_of(w):
    return _cfg().mosaic_v5.VARIANT_OF[w]


def coadd_names(scope, w, nside=None):
    """{key: file name} the engine writes for the unit (coadd_jysr: stem I_deep_<variant>_<scope>, tag mosaic_v5.TAG)."""
    C = _cfg()
    n = int(nside or C.mosaic_coadd.NSIDE)
    s, tag = f'I_deep_{variant_of(w)}_{scope}', C.mosaic_v5.TAG
    return {'jysr': f'healpix_{s}{tag}_jysr_nside{n}.fits', 'psf': f'healpix_PSF_{s}{tag}_nside{n}.fits',
            'jybeam': f'healpix_{s}{tag}_jybeam_nside{n}.fits', 'pbeff': f'healpix_PBeff_{s}{tag}_nside{n}.fits'}


def paths(out, scope, w):
    """Every file path of the unit, by key."""
    from exopipe.phase3.lib.mosaic import caveat as CV
    from exopipe.phase3.lib.mosaic import pkcorr as PK
    C = _cfg()
    d = unit_dir(out, scope, w)
    n = coadd_names(scope, w)
    v, tag = variant_of(w), C.mosaic_v5.TAG
    cm, ct = CV.names(v, scope, int(C.mosaic_caveat.NSIDE))
    p = {'tiles': os.path.join(d, 'stage', 'tiles.json'),
         **{f'unfilled_{k}': os.path.join(d, 'unfilled', f) for k, f in n.items()},
         'filled_jysr': os.path.join(d, 'healpix', n['jysr']), 'filled_jybeam': os.path.join(d, 'healpix', n['jybeam']),
         'caveat_map': os.path.join(d, 'caveat', cm), 'caveat_table': os.path.join(d, 'caveat', ct),
         'content': os.path.join(d, 'content', f'content_I_deep_{v}_{scope}{tag}.json'),
         'conform': os.path.join(d, 'conform')}
    for k, f in PK.names(v, scope, tag, int(C.mosaic_coadd.NSIDE)).items():
        p[f'pkcorr_{k}'] = os.path.join(d, 'pkcorr', f)
    for dom in ('jysr', 'jybeam'):
        for k, suf in (('mollweide', 'Mollweide'), ('radec', 'RADec')):
            p[f'render_{dom}_{k}'] = os.path.join(d, 'renders', f'{scope}_{v}{tag}_{dom}_{suf}.fits')
    return p


def outputs_of(st, out, scope, w):
    """[(path, product class)] the stage owes."""
    p = paths(out, scope, w)
    if st == 'stage':
        return [(p['tiles'], 'mosaic_v5_tiles')]
    if st == 'coadd':
        return [(p['unfilled_jysr'], 'mosaic_map'), (p['unfilled_psf'], 'mosaic_psfmap'), (p['unfilled_jybeam'], 'mosaic_map'),
                (p['unfilled_pbeff'], 'mosaic_pbeff')]
    if st == 'caveat':
        return [(p['caveat_map'], 'mosaic_caveat_map'), (p['caveat_table'], 'mosaic_caveat_table')]
    if st == 'fill':
        return [(p['filled_jysr'], 'mosaic_filled_map'), (p['filled_jybeam'], 'mosaic_filled_map')]
    if st == 'render':
        return [(p[f'render_{d}_{k}'], 'mosaic_render') for d in ('jysr', 'jybeam') for k in ('mollweide', 'radec')]
    if st == 'pkcorr':
        return [(p['pkcorr_map'], 'mosaic_pkcorr_map'), (p['pkcorr_curve'], 'mosaic_pkcorr_curve'),
                (p['pkcorr_recovery'], 'mosaic_pkcorr_recovery'), (p['pkcorr_json'], 'mosaic_pkcorr_report')]
    if st == 'content':
        return [(p['content'], 'mosaic_content_report')]
    if st == 'conform':                 # the names carry the tiles' dates: the files the unit's directory holds now
        return [(f, 'mosaic_v5_product') for f in _files_under(p['conform'])]
    raise ValueError(f'unknown v5 stage {st!r}')


def _files_under(d):
    out = []
    for root, _dirs, files in os.walk(d):
        out += [os.path.join(root, f) for f in files]
    return sorted(out)


def tables_dir(out):
    return os.path.join(out, 'tables')


def stages_for(scope, stages):
    """The selected stages that take this scope (caveat: hybrid.CAVEAT_SCOPES; render: mosaic_v5.RENDER_SCOPES)."""
    C = _cfg()
    return [s for s in stages if s != 'tables' and not (s == 'caveat' and scope not in C.hybrid.CAVEAT_SCOPES)
            and not (s == 'render' and scope not in C.mosaic_v5.RENDER_SCOPES)]


from exopipe.phase3.lib.mosaic import gen_common as GC                     # noqa: E402  (shared with v3: one copy of each rule)
from exopipe.phase3.lib.mosaic.gen_common import (_MD5, _bind_engine, _clear, _engine_as_coadd,  # noqa: E402,F401  (one md5 cache
                                                  _key, _md5, _seed)                               # per process for v3 and v5)


def _upstream(u, out, st, scope, w):
    """The done record of an earlier stage of the unit, else refuse."""
    return GC.upstream(u, out, STEP, unit_name(st, scope, w), f'{scope} {w}', st)


def _tiles(u, out, scope, w, verify=True):
    """The staged tiles of the done stage unit (gen_common.staged_tiles). verify (the units that read the tiles: coadd, caveat):
    every tile and weighting map must still have the md5 the stage recorded."""
    _upstream(u, out, 'stage', scope, w)
    return GC.staged_tiles(u, paths(out, scope, w)['tiles'], f'{scope} {w}', verify)


def _same_tiles(u, scope, w, cur, rec, st):
    """Refuse unless the done record rec (of stage st) was made from the tile list whose md5 is cur."""
    GC.same_tiles(u, f'{scope} {w}', cur, rec, st)


def _link_tree(t, scope, w, sd):
    """The staged tiles and their weighting maps linked under the engine's names (coadd_jysr._paths) in the scratch dir sd."""
    from exopipe.phase3.lib.masters.pb_apply import deep_pb_name
    return GC.link_tree(t, sd, f'I_deep_{variant_of(w)}_pbcorr_{scope}.fits', deep_pb_name(scope, w))


def _coadd_maps(u, out, scope, w, co, keys):
    """The coadd's unfilled maps (keys of coadd_names) at their recorded md5s, recorded as inputs; else refuse."""
    P = paths(out, scope, w)
    GC.coadd_maps(u, {k: P[f'unfilled_{k}'] for k in keys}, co)


# ---- stage ------------------------------------------------------------------------------------------------------------------------

def resolve_tiles(scope, w, hybrid_out, plan_only=False):
    """([tile rows], [problems]): one unfilled beam-corrected hybrid master per LST (module docstring, stage)."""
    from astropy.io import fits
    from exopipe.phase3.core.products import parse_name
    from exopipe.phase3.core.record import NoInput, Refusal
    from exopipe.phase3.lib.hybrid.tilefill import TilefillRefusal
    from exopipe.phase3.lib.masters.pb_apply import deep_pb_name
    from exopipe.phase3.steps import hybrid as HS
    C = _cfg()
    rows = HS._manifest_rows(C.hybrid_tilefill)
    root = hybrid_out or os.path.join(os.sep, 'nonexistent', 'no-hybrid-out')
    tiles, probs = [], []
    for lst in LSTS:
        mode = 'work' if hybrid_out and os.path.exists(os.path.join(hybrid_out, f'{lst}_{scope}', f'{w}_hybrid.json')) else 'pinned'
        try:
            s = HS.tilefill_source(root, lst, scope, w, rows, mode, plan_only=True, need_pb=False)   # the mosaic reads no diffuse PB
        except (TilefillRefusal, Refusal, NoInput, OSError, KeyError, ValueError) as e:
            probs.append(f'{lst} ({mode}): {type(e).__name__}: {e}')
            continue
        path, prodname, want = s['src']['pbcorr']
        pb = os.path.join(C.hybrid.ARCHIVE_MASTERS, lst, scope, deep_pb_name(scope, w))
        row = dict(lst=lst, source=mode, path=path, prodname=prodname, md5=want, pb=pb, nights=s['nights'],
                   nights_from=s['nights_from'], kind='hybrid')
        if not os.path.exists(pb):
            probs.append(f'{lst}: the weighting map {pb} is missing')
            continue
        if not plan_only:
            got = _md5(path)
            if got != want:
                probs.append(f'{lst} ({mode}): {path} has md5 {got}, not {want} (HG13 / the hybrid record)')
                continue
            with fits.open(path, memmap=False) as h:
                hh = h[0].header
            if hh.get('TPFILL'):
                probs.append(f'{lst}: {path} is a filled master (TPFILL): the mosaic is made from the unfilled tiles')
                continue
            if 'lsfill' not in (parse_name(os.path.basename(prodname)).get('proc') or ()):
                probs.append(f'{lst}: {prodname} is not a hybrid master name (no lsfill token)')
                continue
            row.update(BMAJ=hh.get('BMAJ'), BMIN=hh.get('BMIN'), pb_md5=_md5(pb))
        tiles.append(row)
    n = int(C.mosaic_v5.REQUIRE_N_LSTS)
    if not probs and len(tiles) < n:
        probs.append(f'{len(tiles)} tiles < mosaic_v5.REQUIRE_N_LSTS {n}')
    return tiles, probs


def stage_unit(u, scope, w, out, hybrid_out):
    from exopipe.phase3.lib.mosaic import tests_v5  # noqa: F401  (registers mosaic_v5_tiles)
    p = paths(out, scope, w)['tiles']
    u.plan(p, 'mosaic_v5_tiles')
    _clear([p])
    tiles, probs = resolve_tiles(scope, w, hybrid_out)
    if probs:
        u.refuse(f'HG14 {scope} {w}: ' + '; '.join(probs[:6]) + (f' (+{len(probs) - 6} more)' if len(probs) > 6 else '') + ' - REFUSED')
    for t in tiles:
        u.add_input(t['path'], extra={'md5': t['md5']})
        u.add_input(t['pb'], extra={'md5': t['pb_md5']})
        if t['nights_from']:
            u.add_input(t['nights_from'])
    nights = sorted({n for t in tiles for n in t['nights']})
    u.set_admitted(nights)
    u.meta['sources'] = {k: sum(t['source'] == k for t in tiles) for k in ('work', 'pinned')}
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + '.part', 'w') as f:
        json.dump(dict(scope=scope, weighting=w, variant=variant_of(w), hybrid_out=hybrid_out, tiles=tiles), f, indent=1)
    os.replace(p + '.part', p)
    u.add_output(p, 'mosaic_v5_tiles', scope=scope, weighting=w)


# ---- coadd ------------------------------------------------------------------------------------------------------------------------

def coadd_unit(u, scope, w, out, work):
    from exopipe.phase3.core import fitsio
    C = _cfg()
    P = paths(out, scope, w)
    own = [pp for pp, _c in outputs_of('coadd', out, scope, w)]
    for pp, c in outputs_of('coadd', out, scope, w):
        u.plan(pp, c)
    _clear(own)
    t = _tiles(u, out, scope, w)
    v = variant_of(w)
    sd = fitsio.scratch_dir(f'mosaic_v5_{scope}_{w}', work)
    tmp = os.path.join(unit_dir(out, scope, w), f'.tmp_coadd_{os.getpid()}')
    try:
        lsts = _link_tree(t, scope, w, sd)
        verdicts = GC.run_engine(u, lsts, v, scope, sd, tmp, C.mosaic_v5.TAG, 'I')
        bad = {dom: sorted((l, x) for l, x in vd.items() if x != 'added') + [(l, 'absent') for l in lsts if l not in vd]
               for dom, vd in verdicts.items()}
        bad = {k: x for k, x in bad.items() if x}
        if bad:
            u.refuse(f'HG14 {scope} {w}: tiles the engine did not add {bad} - REFUSED')
        names = coadd_names(scope, w)
        for k in ('jysr', 'psf', 'jybeam', 'pbeff'):
            if not os.path.exists(os.path.join(tmp, names[k])):
                u.refuse(f'the engine did not write {names[k]} - REFUSED')
        os.makedirs(os.path.dirname(P['unfilled_jysr']), exist_ok=True)
        for k in ('jysr', 'psf', 'jybeam', 'pbeff'):
            os.replace(os.path.join(tmp, names[k]), P[f'unfilled_{k}'])
    finally:
        fitsio.cleanup(sd)
        if os.path.isdir(tmp):
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
    u.meta['md5'] = {k: _md5(P[f'unfilled_{k}']) for k in ('jysr', 'psf', 'jybeam', 'pbeff')}
    u.meta['tiles_md5'] = t['json_md5']
    ctx = _map_ctx(t, scope, P['unfilled_psf'], P['unfilled_jysr'], P['unfilled_jybeam'])
    for pp, c in outputs_of('coadd', out, scope, w):
        dom = 'jybeam' if pp in (P['unfilled_jybeam'], P['unfilled_pbeff']) else 'jysr'
        u.add_output(pp, c, domain=dom, **ctx)


def _map_ctx(t, scope, psf, jysr, jybeam):
    tiles = [dict(lst=r['lst'], verdict='added', realpath=os.path.realpath(r['path']), BMAJ=r.get('BMAJ'), BMIN=r.get('BMIN'))
             for r in t['tiles']]
    return dict(tiles=tiles, partial=False, scope=scope, tag=_cfg().mosaic_v5.TAG, psfmap=psf, jysr_map=jysr, jybeam_map=jybeam)


# ---- caveat -----------------------------------------------------------------------------------------------------------------------

def caveat_unit(u, scope, w, out):
    from exopipe.phase3.lib.mosaic import caveat as CV
    from exopipe.phase3.lib.mosaic import tests_v5  # noqa: F401
    P = paths(out, scope, w)
    for pp, c in outputs_of('caveat', out, scope, w):
        u.plan(pp, c)
    _clear([P['caveat_map'], P['caveat_table']])
    co = _upstream(u, out, 'coadd', scope, w)
    if _md5(P['unfilled_pbeff']) != co['meta']['md5']['pbeff']:
        u.refuse(f'{P["unfilled_pbeff"]} changed since the coadd (md5) - REFUSED')
    u.add_input(P['unfilled_pbeff'], extra={'md5': co['meta']['md5']['pbeff']})
    t = _tiles(u, out, scope, w)
    _same_tiles(u, scope, w, t['json_md5'], co, 'coadd')
    _engine_as_coadd(u, scope, w, co)
    try:
        rec = CV.caveat_map(t['tiles'], variant_of(w), scope, os.path.dirname(P['caveat_map']), P['unfilled_pbeff'])
    except CV.CaveatRefusal as e:
        _clear([P['caveat_map'], P['caveat_table']])
        u.refuse(f'{e}')
    u.meta['caveat'] = {k: rec[k] for k in ('n_hybrid', 'n_taper_only', 'taper_only')}
    u.meta['md5'] = {'map': _md5(P['caveat_map']), 'table': _md5(P['caveat_table'])}
    u.check('test_caveat_map', rec['test'].get('verdict') == 'pass', value=rec['test'], path=P['caveat_map'],
            cls='mosaic_caveat_map', detail='readback, TAPERONLY range, LEVEL within the tiles, weights = the mosaic PB_eff')
    u.add_output(P['caveat_map'], 'mosaic_caveat_map', scope=scope)
    u.add_output(P['caveat_table'], 'mosaic_caveat_table', n_tiles=len(t['tiles']))


# ---- fill -------------------------------------------------------------------------------------------------------------------------

def fill_unit(u, scope, w, out):
    from exopipe.phase3.lib.mosaic import fill as MF
    from exopipe.phase3.lib.mosaic import tests_v5  # noqa: F401
    P = paths(out, scope, w)
    for pp, c in outputs_of('fill', out, scope, w):
        u.plan(pp, c)
    _clear([P['filled_jysr'], P['filled_jybeam']])
    co = _upstream(u, out, 'coadd', scope, w)
    t = _tiles(u, out, scope, w, verify=False)
    _same_tiles(u, scope, w, t['json_md5'], co, 'coadd')
    for k in ('jysr', 'psf', 'jybeam', 'pbeff'):
        want = co['meta']['md5'][k]
        if _md5(P[f'unfilled_{k}']) != want:
            u.refuse(f'{P[f"unfilled_{k}"]} changed since the coadd (md5) - REFUSED')
        u.add_input(P[f'unfilled_{k}'], extra={'md5': want})
    os.makedirs(os.path.dirname(P['filled_jysr']), exist_ok=True)
    tj, tb = f'{P["filled_jysr"]}.tmp.{os.getpid()}', f'{P["filled_jybeam"]}.tmp.{os.getpid()}'
    try:
        try:
            rj = MF.fill_file(P['unfilled_jysr'], tj)
        except MF.FillRefusal as e:
            u.refuse(f'HG19: {e}')
        u.check('HG19.fill_jysr', rj['result'] == 'PASS', value=rj['test'], path=P['filled_jysr'], cls='mosaic_filled_map',
                detail='large scales follow s x LFSS; the change is smooth (MF test_tpfill limits, mosaic_fill)')
        if rj['result'] != 'PASS':
            u.refuse(f'HG19: the Jy/sr fill test failed {rj["test"]} - REFUSED')
        try:
            rb = MF.fill_jybeam_file(P['unfilled_jybeam'], P['unfilled_jysr'], tj, P['unfilled_psf'], P['unfilled_pbeff'], tb)
        except MF.FillRefusal as e:
            u.refuse(f'HG19: {e}')
        u.check('HG19.fill_jybeam', rb['result'] == 'PASS', value={k: rb[k] for k in ('identity_rel', 'k_median', 'k_p05', 'k_p95',
                                                                                      'n_bright')},
                path=P['filled_jybeam'], cls='mosaic_filled_map', detail='identity to float32; K median and 5-95 spread (mosaic_fill)')
        if rb['result'] != 'PASS':
            u.refuse(f'HG19: the Jy/beam fill test failed {rb} - REFUSED')
        os.replace(tj, P['filled_jysr'])
        _seed(P['filled_jysr'], rj['out_md5'])
        os.replace(tb, P['filled_jybeam'])
        _seed(P['filled_jybeam'], rb['out_md5'])
    finally:
        _clear([tj, tb])
    u.meta['fill'] = {'jysr': {k: rj[k] for k in ('donor_mhz', 's', 'test', 'src_md5')},
                      'jybeam': {k: rb[k] for k in ('identity_rel', 'k_median', 'k_p05', 'k_p95', 'n_bright', 'src_md5')}}
    u.meta['md5'] = {'unfilled_jysr': co['meta']['md5']['jysr'], 'unfilled_jybeam': co['meta']['md5']['jybeam'],
                     'filled_jysr': _md5(P['filled_jysr']), 'filled_jybeam': _md5(P['filled_jybeam'])}
    u.meta['tiles_md5'] = t['json_md5']
    ctx = _map_ctx(t, scope, P['unfilled_psf'], P['filled_jysr'], P['filled_jybeam'])
    u.add_output(P['filled_jysr'], 'mosaic_filled_map', domain='jysr', unfilled=P['unfilled_jysr'], **ctx)
    u.add_output(P['filled_jybeam'], 'mosaic_filled_map', domain='jybeam', unfilled=P['unfilled_jybeam'],
                 filled_jysr=P['filled_jysr'], **ctx)


# ---- render -----------------------------------------------------------------------------------------------------------------------

def render_unit(u, scope, w, out, work):
    import shutil
    from exopipe.phase3.core import fitsio
    from exopipe.phase3.lib.mosaic import band_products as BP
    C = _cfg()
    R = C.mosaic_render
    P = paths(out, scope, w)
    own = [pp for pp, _c in outputs_of('render', out, scope, w)]
    for pp, c in outputs_of('render', out, scope, w):
        u.plan(pp, c)
    _clear(own)
    fi = _upstream(u, out, 'fill', scope, w)
    co = _upstream(u, out, 'coadd', scope, w)
    if not os.path.exists(P['tiles']):
        u.refuse(f'{P["tiles"]} is missing - REFUSED')
    cur = _md5(P['tiles'])
    _same_tiles(u, scope, w, cur, co, 'coadd')
    _same_tiles(u, scope, w, cur, fi, 'fill')
    u.add_input(P['tiles'], extra={'md5': cur})
    srcs = {'jysr': (P['filled_jysr'], fi['meta']['md5']['filled_jysr']), 'jybeam': (P['filled_jybeam'], fi['meta']['md5']['filled_jybeam']),
            'psf': (P['unfilled_psf'], co['meta']['md5']['psf'])}
    for k, (pp, want) in srcs.items():
        if _md5(pp) != want:
            u.refuse(f'{pp} changed since its stage (md5) - REFUSED')
        u.add_input(pp, extra={'md5': want})
    v, tag, names = variant_of(w), C.mosaic_v5.TAG, coadd_names(scope, w)
    ld = fitsio.scratch_dir(f'mosaic_v5_render_{scope}_{w}', work)
    tmp = os.path.join(unit_dir(out, scope, w), f'.tmp_render_{os.getpid()}')
    try:
        for k in ('jysr', 'jybeam', 'psf'):
            os.symlink(srcs[k][0], os.path.join(ld, names[k]))
        BP.BEAMS = dict(R.BEAM_FALLBACK_ARCMIN)
        os.makedirs(tmp, exist_ok=True)
        for dom in ('jysr', 'jybeam'):
            BP.build(v, ld, f'{tag}_{dom}', tmp, bands=[scope], moll_W=int(R.MOLLWEIDE_WIDTH_PX), car_cdelt=float(R.CAR_CDELT_ARCMIN) / 60.0,
                     dec_cut=R.RENDER_DEC_CUT_DEG)
        for pp in own:
            if not os.path.exists(os.path.join(tmp, os.path.basename(pp))):
                u.refuse(f'band_products did not write {os.path.basename(pp)} - REFUSED')
        os.makedirs(os.path.dirname(own[0]), exist_ok=True)
        for pp in own:
            os.replace(os.path.join(tmp, os.path.basename(pp)), pp)
    finally:
        fitsio.cleanup(ld)
        shutil.rmtree(tmp, ignore_errors=True)
    u.meta['md5'] = {f'{dom}_{k}': _md5(P[f'render_{dom}_{k}']) for dom in ('jysr', 'jybeam') for k in ('mollweide', 'radec')}
    for dom in ('jysr', 'jybeam'):
        for k in ('mollweide', 'radec'):
            u.add_output(P[f'render_{dom}_{k}'], 'mosaic_render', source=P[f'filled_{dom}'], psfmap=P['unfilled_psf'], kind=k,
                         scope=scope, dec_cut=R.RENDER_DEC_CUT_DEG)


# ---- pkcorr -----------------------------------------------------------------------------------------------------------------------

def pkcorr_unit(u, scope, w, out, work):
    from exopipe.phase3.lib.mosaic import tests_v5  # noqa: F401
    C = _cfg()
    P = paths(out, scope, w)
    own = [pp for pp, _c in outputs_of('pkcorr', out, scope, w)]
    for pp, c in outputs_of('pkcorr', out, scope, w):
        u.plan(pp, c)
    _clear(own)
    co = _upstream(u, out, 'coadd', scope, w)
    t = _tiles(u, out, scope, w)
    _same_tiles(u, scope, w, t['json_md5'], co, 'coadd')
    _coadd_maps(u, out, scope, w, co, ('jysr', 'psf'))
    _engine_as_coadd(u, scope, w, co)
    GC.pkcorr(u, f'{scope} {w}', scope, variant_of(w), C.mosaic_v5.TAG, t, {k: P[f'unfilled_{k}'] for k in ('jysr', 'psf')},
              {k: P[f'pkcorr_{k}'] for k in ('map', 'curve', 'recovery', 'json')}, lambda sd: _link_tree(t, scope, w, sd),
              (f'mosaic_v5_pkcorr_{scope}_{w}', f'mosaic_v5_pkcorr_inj_{scope}_{w}'), unit_dir(out, scope, w), work)


# ---- content ----------------------------------------------------------------------------------------------------------------------

def content_unit(u, scope, w, out, work):
    import glob
    from exopipe.phase3.core import fitsio
    from exopipe.phase3.lib.mosaic import content as CT
    from exopipe.phase3.lib.mosaic import tests_v5  # noqa: F401
    C = _cfg()
    P = paths(out, scope, w)
    for pp, c in outputs_of('content', out, scope, w):
        u.plan(pp, c)
    _clear([P['content']])
    co = _upstream(u, out, 'coadd', scope, w)
    t = _tiles(u, out, scope, w)
    _same_tiles(u, scope, w, t['json_md5'], co, 'coadd')
    _coadd_maps(u, out, scope, w, co, ('jysr', 'jybeam'))
    _engine_as_coadd(u, scope, w, co)
    parents = C.mosaic_content.PARENTS
    cats = sorted(os.path.basename(x) for x in glob.glob(os.path.join(glob.escape(parents), f'master_R075_*_{scope}.gaul.csv')))
    if not cats:
        u.refuse(f'{scope} {w}: no parent catalog master_R075_*_{scope}.gaul.csv in mosaic_content.PARENTS {parents} - REFUSED')
    u.meta['parents'] = {'dir': parents, 'n_catalogs': len(cats)}
    v = variant_of(w)
    sd = fitsio.scratch_dir(f'mosaic_v5_content_{scope}_{w}', work)
    try:
        _link_tree(t, scope, w, sd)
        rep = CT.run({dom: P[f'unfilled_{dom}'] for dom in ('jysr', 'jybeam')}, sd, v, scope, parents)
    finally:
        fitsio.cleanup(sd)
    # TEST 2: the tile and PB files TEST 1 read are exactly the tile list's, each at the md5 _tiles verified in this unit
    u.meta['md5'] = {'content': GC.test2_write(rep, t, P['content'])}
    for dom, r in rep.items():
        u.check(f'TEST1.{dom}', r['result'] == 'PASS', value={k: r[k] for k in ('n_tiles', 'n_sources', 'median_ratio', 'p05', 'p95',
                                                                              'frac_within_5pct', 'result')},
                path=P['content'], cls='mosaic_content_report',
                detail=f'{r["n_sources"]} sources, median map / tiles {r["median_ratio"]:.5f}, within 5 % {r["frac_within_5pct"]:.3f}')
        GC.test2_check(u, dom, r, len(t['tiles']), P['content'], 'mosaic_content_report')
    u.add_output(P['content'], 'mosaic_content_report', n_tiles=len(t['tiles']), scope=scope, variant=v)


# ---- conform ----------------------------------------------------------------------------------------------------------------------

CONFORM_NEEDS = {'pkcorr': ('test_pkcorr',),                                            # HG15
                 'content': ('TEST1.jysr', 'TEST1.jybeam', 'TEST2.jysr', 'TEST2.jybeam'),  # HG17
                 'fill': ('HG19.fill_jysr', 'HG19.fill_jybeam'),                       # HG19
                 'caveat': ('test_caveat_map',)}                                       # HG18
#   the tests each upstream record must hold, passed, before conform stages the unit's products


def _rec_ok(u, scope, w, rec, st, tiles_path):
    """Refuse unless the done record rec of stage st was made from the current stage/tiles.json (its recorded fingerprint), every
    test it records passed (a waived failure never reaches the archive) and CONFORM_NEEDS[st] are among them."""
    from exopipe.phase3.core.record import fingerprint, same_fingerprint
    fp = (rec.get('inputs') or {}).get(os.path.abspath(tiles_path))
    if fp is None or not same_fingerprint(fp, fingerprint(tiles_path)):
        u.refuse(f'{scope} {w}: the {st} unit was not made from the current stage/tiles.json: rerun --stages {st} - REFUSED')
    tests = rec.get('tests') or []
    bad = sorted({str(t.get('name')) for t in tests if not t.get('passed')})
    if bad:
        u.refuse(f'{scope} {w}: the {st} record holds failed test(s) {bad[:6]} - REFUSED')
    miss = [n for n in CONFORM_NEEDS.get(st, ()) if not any(t.get('name') == n for t in tests)]
    if miss:
        u.refuse(f'{scope} {w}: the {st} record lacks the test(s) {miss} - REFUSED')


def _tile_key(r):
    return (r['lst'], r['source'], os.path.realpath(r['path']), r['md5'], tuple(r['nights']), r['pb'])


def conform_unit(u, scope, w, out, work=None):
    import shutil
    from exopipe.phase3.lib.mosaic import coadd_jysr as mcj
    from exopipe.phase3.lib.mosaic import conform_v5 as CF
    from exopipe.phase3.lib.mosaic import tests_v5  # noqa: F401  (registers mosaic_v5_product)
    C = _cfg()
    P = paths(out, scope, w)
    caveat = scope in C.hybrid.CAVEAT_SCOPES
    if os.path.lexists(P['conform']):
        shutil.rmtree(P['conform'])
    sts = ('coadd', 'pkcorr', 'fill', 'render', 'content') + (('caveat',) if caveat else ())
    recs = {st: _upstream(u, out, st, scope, w) for st in sts}
    t = _tiles(u, out, scope, w)                   # every tile and weighting map at its staged md5 (the membership reads them)
    for st, rec in recs.items():
        _rec_ok(u, scope, w, rec, st, P['tiles'])
    _engine_as_coadd(u, scope, w, recs['coadd'])    # PBCORR_FLOOR of the membership guards = the coadd's
    now, probs = resolve_tiles(scope, w, t.get('hybrid_out'), plan_only=True)
    if probs:
        u.refuse(f'HG14 {scope} {w}: the tiles resolved now: ' + '; '.join(probs[:4]) + ' - REFUSED')
    diff = sorted(set(map(_tile_key, t['tiles'])) ^ set(map(_tile_key, now)))
    if diff:
        u.refuse(f'HG13 {scope} {w}: the tiles resolved now differ from stage/tiles.json in {len(diff)} row(s), e.g. LST {diff[0][0]} '
                 f'(a source replaced, or a night now excluded): rerun from --stages stage - REFUSED')
    miss = CF.classic_missing(scope)
    if miss:
        u.refuse(f'HG16 {scope}: the classic Taper Jy/sr map of weighting(s) {miss} is not on Lustre with a PASS content-test row '
                 f'(MC on_lustre) - REFUSED')
    co, fi, pk, rn, ct = (recs[k] for k in ('coadd', 'fill', 'pkcorr', 'render', 'content'))
    md5_of = lambda rec, k: (rec['meta'].get('md5') or {}).get(k)   # noqa: E731
    want = {P['filled_jysr']: md5_of(fi, 'filled_jysr'), P['filled_jybeam']: md5_of(fi, 'filled_jybeam'),
            P['unfilled_pbeff']: md5_of(co, 'pbeff'), P['unfilled_psf']: md5_of(co, 'psf'), P['pkcorr_map']: md5_of(pk, 'map'),
            P['pkcorr_curve']: md5_of(pk, 'curve'), P['content']: md5_of(ct, 'content'), P['unfilled_jysr']: md5_of(co, 'jysr'),
            P['unfilled_jybeam']: md5_of(co, 'jybeam')}
    for dom in ('jysr', 'jybeam'):
        for k in ('mollweide', 'radec'):
            want[P[f'render_{dom}_{k}']] = md5_of(rn, f'{dom}_{k}')
    if caveat:
        want[P['caveat_map']], want[P['caveat_table']] = md5_of(recs['caveat'], 'map'), md5_of(recs['caveat'], 'table')
    for pp, m in want.items():
        if not m:
            u.refuse(f'{pp}: its stage recorded no md5 (a record of earlier code): rerun that stage - REFUSED')
        if not os.path.exists(pp) or _md5(pp) != m:
            u.refuse(f'{pp} missing or changed since its stage (md5) - REFUSED')
        u.add_input(pp, extra={'md5': m})
    try:
        staged, info = CF.stage(scope, w, P, t['tiles'], caveat, P['conform'], float(mcj.PBCORR_FLOOR),
                                plan=lambda ps: [u.plan(x, 'mosaic_v5_product') for x in ps])
    except CF.ConformRefusal as e:
        u.refuse(f'{scope} {w}: {e} - REFUSED')
    u.meta['tiles_md5'] = t['json_md5']
    u.meta['membership'] = {k: info[k] for k in ('n', 't0', 't1', 'inttime_s', 'beam')}
    u.meta['prov_files'] = info['prov_files']
    u.meta['products'] = {kind: {'path': dst, 'md5': _md5(dst), 'source': src, 'source_md5': want[src]} for dst, src, kind in staged}
    u.meta['tables'] = {'variant': variant_of(w), 'band': scope, 'content': [P['content'], want[P['content']]],
                        'curve': [P['pkcorr_curve'], want[P['pkcorr_curve']]],
                        'caveat_table': [P['caveat_table'], want[P['caveat_table']]] if caveat else None,
                        'tpfill': CF.tpfill_row(info['names']['jysr'], fi['meta']['fill']['jysr'])}
    for dst, src, kind in staged:
        u.add_output(dst, 'mosaic_v5_product', kind=kind, scope=scope, nstack=info['n'], source=src)


# ---- tables -----------------------------------------------------------------------------------------------------------------------

TABLES_UNIT = 'v5-tables'


def tables_unit(u, out):
    """One per run (module docstring, tables): the V5 tables with the rows of every done conform unit of --out."""
    import shutil
    from exopipe.phase3.core.record import load_record
    from exopipe.phase3.lib.mosaic import conform_v5 as CF
    from exopipe.phase3.lib.mosaic import tests_v5  # noqa: F401  (registers mosaic_v5_table)
    C = _cfg()
    V5 = C.mosaic_v5
    td = tables_dir(out)
    if os.path.lexists(td):
        shutil.rmtree(td)
    arch = os.path.join(CF.archive_run(), 'tables')
    recs = {}
    for sc in V5.SCOPES:
        for w in V5.WEIGHTINGS:
            r = load_record(out, STEP, unit_name('conform', sc, w))
            if r is not None and r.get('status') == 'done':
                recs[(sc, w)] = r
    if not recs:
        u.refuse(f'no done conform unit in {out}: run --stages conform first - REFUSED')
    T = {}
    for (sc, w), r in sorted(recs.items()):
        bad = sorted({str(x.get('name')) for x in r.get('tests') or [] if not x.get('passed')})
        if bad:
            u.refuse(f'the conform record of {sc} {w} holds failed test(s) {bad[:6]} - REFUSED')
        T[(sc, w)] = tt = r['meta']['tables']
        for k in ('content', 'curve', 'caveat_table'):
            if tt.get(k):
                pp, m = tt[k]
                if not os.path.exists(pp) or _md5(pp) != m:
                    u.refuse(f'{pp} missing or changed since the conform of {sc} {w} (md5) - REFUSED')
                u.add_input(pp, extra={'md5': m})

    def archived(name, cols=None):
        pa = os.path.join(arch, name)
        if not os.path.exists(pa):
            u.refuse(f'the archived table {pa} is missing - REFUSED')
        u.add_input(pa, extra={'md5': _md5(pa)})
        head, rows = CF._read(pa)
        if cols is not None and tuple(head) != tuple(cols):
            u.refuse(f'{pa}: columns {head} are not {list(cols)} - REFUSED')
        return head, rows

    vb = {(tt['variant'], tt['band']) for tt in T.values()}
    tables = {}
    _h, rows = archived('summary.csv', CF.SUMMARY_COLS)
    new = [row for tt in T.values() for row in CF.summary_rows(tt['content'][0])]
    tables['summary.csv'] = (CF.SUMMARY_COLS, CF.merge([x for x in rows if (x[1], x[2]) not in vb], new, lambda x: (x[1], x[2]),
                                                         CF.summary_sort))
    _h, rows = archived('pkcorr_curves.csv', CF.CURVE_COLS)
    new = [row for tt in T.values() for row in CF.curve_rows(tt['variant'], tt['band'], tt['curve'][0])]
    tables['pkcorr_curves.csv'] = (CF.CURVE_COLS, CF.merge([x for x in rows if (x[0], x[1]) not in vb], new, lambda x: (x[0], x[1]),
                                                           CF.curve_sort))
    for w in V5.WEIGHTINGS:
        v = V5.VARIANT_OF[w]
        name = f'summary_tpfill_{v}.csv'
        _h, rows = archived(name, CF.TPFILL_COLS)
        new = [tt['tpfill'] for (sc, w_), tt in T.items() if w_ == w]
        mine = {sc for (sc, w_) in T if w_ == w}
        tables[name] = (CF.TPFILL_COLS, CF.merge([x for x in rows if x[0].split('_')[0] not in mine], new, lambda x: x[0].split('_')[0],
                                                 lambda x: x[0]))
    for (sc, w), tt in sorted(T.items()):
        if tt.get('caveat_table'):
            name = os.path.basename(tt['caveat_table'][0])
            head, rows = CF.caveat_rows(tt['caveat_table'][0], sc)
            ah, _r = archived(name)                                  # refuses when the archived table is missing
            if tuple(ah) != tuple(head):
                u.refuse(f'{name}: columns {head} are not the archived table\'s {ah} - REFUSED')
            tables[name] = (tuple(head), rows)
    # the caveat table of every other (caveat scope, weighting): the archived table, carried forward whole
    want_cav = {os.path.basename(paths(out, sc, w)['caveat_table']): (sc, w) for sc in C.hybrid.CAVEAT_SCOPES for w in V5.WEIGHTINGS}
    for name in sorted(set(want_cav) - set(tables)):
        tables[name] = tuple(archived(name))
    _h, rows = archived('taper_only.csv', CF.TAPER_ONLY_COLS)
    tables['taper_only.csv'] = (CF.TAPER_ONLY_COLS, [x for x in rows if (x[1], x[2]) not in set(T)])
    # completeness: every scope of mosaic_v5.SCOPES in both weightings, every result PASS (the archived rows included)
    want_vb = {(V5.VARIANT_OF[w], sc) for sc in V5.SCOPES for w in V5.WEIGHTINGS}
    probs = [f'the conform record of {sc} {w} names no caveat table' for (sc, w), tt in sorted(T.items())
             if sc in C.hybrid.CAVEAT_SCOPES and not tt.get('caveat_table')]
    probs += [f'{name}: not a caveat table of hybrid.CAVEAT_SCOPES' for name in sorted(tables)
              if name.startswith('caveats_') and name not in want_cav]
    srows = tables['summary.csv'][1]
    got = {(x[1], x[2], x[3]) for x in srows}
    probs += [f'summary.csv: no {d} row of {v} {b}' for v, b in sorted(want_vb) for d in ('jysr', 'jybeam') if (v, b, d) not in got]
    probs += [f'summary.csv: {x[0]} {x[-1]}' for x in srows if x[-1].strip() != 'PASS']
    got = {(x[0], x[1]) for x in tables['pkcorr_curves.csv'][1]}
    probs += [f'pkcorr_curves.csv: no curve of {v} {b}' for v, b in sorted(want_vb - got)]
    for w in V5.WEIGHTINGS:
        name = f'summary_tpfill_{V5.VARIANT_OF[w]}.csv'
        rows = tables[name][1]
        sc_ = [x[0].split('_')[0] for x in rows]
        probs += [f'{name}: no row of {sc}' for sc in V5.SCOPES if sc not in sc_]
        probs += [f'{name}: {sc_.count(sc)} rows of {sc}' for sc in sorted(set(sc_)) if sc_.count(sc) > 1]
        probs += [f'{name}: {x[0]} {x[-1]}' for x in rows if x[-1].strip() != 'PASS']
    if probs:
        u.refuse(f'the V5 tables would be incomplete or hold a failure: {probs[:6]}' + (f' (+{len(probs) - 6} more)' if len(probs) > 6
                                                                                           else '') + ' - REFUSED')
    os.makedirs(td, exist_ok=True)
    for name in sorted(tables):
        u.plan(os.path.join(td, name), 'mosaic_v5_table')
    for name, (head, rows) in sorted(tables.items()):
        CF._write(os.path.join(td, name), head, rows)
    u.meta['units'] = sorted(f'{sc}_{w}' for sc, w in T)
    u.meta['archived_tables'] = arch
    u.meta['md5'] = {name: _md5(os.path.join(td, name)) for name in sorted(tables)}     # the push (archive --class mosaic_v5) checks them
    for name, (head, rows) in sorted(tables.items()):
        u.add_output(os.path.join(td, name), 'mosaic_v5_table', columns=list(head), n_rows=len(rows))


# ---- the step ---------------------------------------------------------------------------------------------------------------------

def select_stages(args):
    return GC.select_stages(args, STAGES, 'v5')


def select_units(args):
    """[(stage, scope, weighting)] in pipeline order: every stage of a (scope, weighting) before the next pair; the run's tables unit
    ('tables', None, None) last."""
    C = _cfg()
    scopes = [s.strip() for s in str(args.scopes).split(',') if s.strip()] if getattr(args, 'scopes', None) else list(C.mosaic_v5.SCOPES)
    bad = [s for s in scopes if s not in C.mosaic_v5.SCOPES]
    if bad:
        raise ValueError(f'--scopes: unknown scope(s) {bad} ({", ".join(C.mosaic_v5.SCOPES)})')
    ws = [x.strip() for x in str(args.weightings).split(',') if x.strip()] if getattr(args, 'weightings', None) else list(C.mosaic_v5.WEIGHTINGS)
    bad = [x for x in ws if x not in C.mosaic_v5.WEIGHTINGS]
    if bad:
        raise ValueError(f'--weightings: unknown weighting(s) {bad} ({", ".join(C.mosaic_v5.WEIGHTINGS)})')
    stages = select_stages(args)
    return [(st, sc, w) for sc in scopes for w in ws for st in stages_for(sc, stages)] + ([('tables', None, None)] if 'tables' in stages
                                                                                         else [])


def _own(st, out, sc, w):
    """The files a unit owns now (conform and tables: their directories' files; their names are known only once computed)."""
    if st == 'tables':
        return _files_under(tables_dir(out))
    return [p for p, _c in outputs_of(st, out, sc, w)]


def out_problems(out, hybrid_out):
    """--out is a work run of its own: never inside or above an archive run of the hybrid route (the mosaic runs, the hybrid
    archive, the masters), nor the hybrid work run --hybrid-out (its records are read only)."""
    C = _cfg()
    held = {'ARCHIVE_HYBRID': C.hybrid.ARCHIVE_HYBRID, 'ARCHIVE_MASTERS': C.hybrid.ARCHIVE_MASTERS,
            'FILLED_MOSAIC_RUN': C.hybrid_tilefill.FILLED_MOSAIC_RUN}
    held.update({f'UNFILLED_MOSAIC_RUNS[{i}]': r for i, r in enumerate(C.hybrid_tilefill.UNFILLED_MOSAIC_RUNS)})
    if hybrid_out:
        held['--hybrid-out'] = hybrid_out
    return GC.held_problems(out, held)


def run(args, run):
    from exopipe.phase3.core.record import NoInput, Refusal
    hybrid_out = getattr(args, 'hybrid_out', None)
    if hybrid_out and not os.path.isdir(hybrid_out):
        raise Refusal(f'--hybrid-out {hybrid_out} is not a directory')
    bad = out_problems(run.out_root, hybrid_out)
    if bad:
        raise Refusal('; '.join(bad))
    units = select_units(args)
    run.meta = {'generation': 'v5', 'units': len(units), 'hybrid_out': hybrid_out}
    if not units:
        raise NoInput('no unit: the selected stages take none of the selected scopes')
    if run.dry_run:
        for st, sc, w in units:
            if st == 'tables':
                print(f'  plan {TABLES_UNIT}: {os.path.relpath(tables_dir(run.out_root), run.out_root)}/ (the V5 tables with the rows of '
                      f'every done conform unit)', flush=True)
                continue
            if st == 'conform':
                what = f'{os.path.relpath(paths(run.out_root, sc, w)["conform"], run.out_root)}/{sc}/I/{{healpix,mollweide,radec}}/'
            elif st == 'stage':
                tiles, probs = resolve_tiles(sc, w, hybrid_out, plan_only=True)
                src = {k: sum(t['source'] == k for t in tiles) for k in ('work', 'pinned')}
                what = (f'refused: {"; ".join(probs[:3])}' if probs else
                        f'{len(tiles)} tiles (work {src["work"]}, pinned {src["pinned"]}) -> ')
                if not probs:
                    what += os.path.relpath(paths(run.out_root, sc, w)['tiles'], run.out_root)
            else:
                what = ', '.join(os.path.relpath(p, run.out_root) for p, _c in outputs_of(st, run.out_root, sc, w))
            print(f'  plan {unit_name(st, sc, w)}: {what}', flush=True)
        print(f'exopipe3 mosaic --generation v5: DRY RUN, {len(units)} units, nothing written', flush=True)
        return 0
    work = getattr(args, 'work', None)
    for st, sc, w in units:
        name, ctx = (TABLES_UNIT, {}) if st == 'tables' else (unit_name(st, sc, w), dict(scope=sc, weighting=w))
        with run.unit(name, stage=st, generation='v5', **ctx) as u:
            try:
                if st == 'stage':
                    stage_unit(u, sc, w, run.out_root, hybrid_out)
                elif st == 'coadd':
                    coadd_unit(u, sc, w, run.out_root, work)
                elif st == 'caveat':
                    caveat_unit(u, sc, w, run.out_root)
                elif st == 'fill':
                    fill_unit(u, sc, w, run.out_root)
                elif st == 'render':
                    render_unit(u, sc, w, run.out_root, work)
                elif st == 'pkcorr':
                    pkcorr_unit(u, sc, w, run.out_root, work)
                elif st == 'content':
                    content_unit(u, sc, w, run.out_root, work)
                elif st == 'conform':
                    conform_unit(u, sc, w, run.out_root)
                else:
                    tables_unit(u, run.out_root)
            except BaseException:
                _clear(_own(st, run.out_root, sc, w))   # nothing of a unit that did not end done stays in place
                raise
            own = _own(st, run.out_root, sc, w)
            if u.failed_tests():
                _clear(own)
                u.note(f'a content test failed: the {len(own)} outputs were removed (the record keeps the results)')
    return run.exit_code()
