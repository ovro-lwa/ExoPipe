"""`exopipe3 mosaic --generation v5`: the hybrid mosaics (spec plan/specs/hybrid.md 2.8, 2.10-2.14; PR-7a). Unit of work: (stage,
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
  caveat  hybrid.CAVEAT_SCOPES only: lib/mosaic/caveat.caveat_map with the mosaic's tile weights; test_caveat_map against the unfilled
          PB_eff map. -> caveat/healpix_CAVEAT_I_deep_<V>_<S>_lsfill_nside128.fits [mosaic_caveat_map], caveats_<V>_<S>_lsfill.csv
          [mosaic_caveat_table]
  fill    lib/mosaic/fill.fill_file of the unfilled Jy/sr map (HG19: the test on the written file must PASS), then fill_jybeam_file.
          -> healpix/healpix_I_deep_<V>_<S>_lsfill_{jysr,jybeam}_nside2048.fits [mosaic_filled_map]; md5s in the record (the band
          field of `exopipe3 hybrid --stages tilefill --mosaic-out <out>` reads them)
  render  lib/mosaic/band_products.build of the FILLED maps with their PSF map (the beam card), mosaic_render's width, pixel and
          southern edge. -> renders/<S>_<V>_lsfill_{jysr,jybeam}_{Mollweide,RADec}.fits [mosaic_render]

A stage reads its predecessors' unit records (status done), never a glob. A unit removes its own earlier outputs first; nothing is
renamed into place unless its test passed. Content tests (TEST 1/1R/2), PKCORR, conformance v5 and the tables are PR-7b.
"""
import json
import os

LSTS = tuple(f'{h:02d}h' for h in range(24))
STAGES = ('stage', 'coadd', 'caveat', 'fill', 'render')
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
    C = _cfg()
    d = unit_dir(out, scope, w)
    n = coadd_names(scope, w)
    v, tag = variant_of(w), C.mosaic_v5.TAG
    cm, ct = CV.names(v, scope, int(C.mosaic_caveat.NSIDE))
    p = {'tiles': os.path.join(d, 'stage', 'tiles.json'),
         **{f'unfilled_{k}': os.path.join(d, 'unfilled', f) for k, f in n.items()},
         'filled_jysr': os.path.join(d, 'healpix', n['jysr']), 'filled_jybeam': os.path.join(d, 'healpix', n['jybeam']),
         'caveat_map': os.path.join(d, 'caveat', cm), 'caveat_table': os.path.join(d, 'caveat', ct)}
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
    return [(p[f'render_{d}_{k}'], 'mosaic_render') for d in ('jysr', 'jybeam') for k in ('mollweide', 'radec')]


def stages_for(scope, stages):
    """The selected stages that take this scope (caveat: hybrid.CAVEAT_SCOPES; render: mosaic_v5.RENDER_SCOPES)."""
    C = _cfg()
    return [s for s in stages if not (s == 'caveat' and scope not in C.hybrid.CAVEAT_SCOPES)
            and not (s == 'render' and scope not in C.mosaic_v5.RENDER_SCOPES)]


def _md5(p):
    from exopipe.phase3.lib.install.steps_common import md5
    return md5(p)


def _clear(paths_):
    import glob
    for p in paths_:
        for q in [p, f'{p}.part'] + glob.glob(f'{glob.escape(p)}.tmp.*'):
            if os.path.lexists(q):
                os.remove(q)


def _upstream(u, out, st, scope, w):
    """The done record of an earlier stage of the unit, else refuse."""
    from exopipe.phase3.core.record import load_record
    r = load_record(out, STEP, unit_name(st, scope, w))
    if r is None:
        u.refuse(f'no {st} record for {scope} {w} in {out} (run --stages {st} first) - REFUSED')
    if r.get('status') != 'done':
        u.refuse(f'the {st} unit of {scope} {w} ended {r.get("status")}: {str(r.get("reason"))[:200]} - REFUSED')
    return r


def _tiles(u, out, scope, w, verify=True):
    """The staged tiles (tiles.json of the done stage unit), each tile's md5 re-computed (verify) against the stage's."""
    _upstream(u, out, 'stage', scope, w)
    p = paths(out, scope, w)['tiles']
    with open(p) as f:
        t = json.load(f)
    u.add_input(p)
    if verify:
        bad = [r['lst'] for r in t['tiles'] if _md5(r['path']) != r['md5']]
        if bad:
            u.refuse(f'{scope} {w}: staged tile(s) changed since the stage (md5): {bad} - REFUSED')
    for r in t['tiles']:
        u.add_input(r['path'], extra={'md5': r['md5']})
        u.add_input(r['pb'])
    return t


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
            s = HS.tilefill_source(root, lst, scope, w, rows, mode, plan_only=True)
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
    import sys
    from exopipe.phase3.core import fitsio
    from exopipe.phase3.lib.masters.pb_apply import deep_pb_name
    from exopipe.phase3.lib.mosaic import coadd_jysr as mcj
    from exopipe.phase3.lib.mosaic import steps_common as SC
    from exopipe.phase3.lib.mosaic.steps_worker import _VERDICT, _Tee
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
        for r in t['tiles']:
            d = os.path.join(sd, r['lst'])
            os.makedirs(d, exist_ok=True)
            os.symlink(r['path'], os.path.join(d, f'I_deep_{v}_pbcorr_{scope}.fits'))
            os.symlink(r['pb'], os.path.join(d, deep_pb_name(scope, w)))
        g = SC.engine_globals()
        for k, val in g.items():
            setattr(mcj, k, val)
        mcj.BEAM_OVERRIDE.clear()             # July repair-ledger modes are never used by v2 (retired keys, spec 3)
        mcj.SKIP_CELLS.clear()
        u.meta['engine_globals'] = g
        os.makedirs(tmp, exist_ok=True)
        lsts = [r['lst'] for r in t['tiles']]
        verdicts = {}
        for dom in ('jysr', 'jybeam'):
            tee = _Tee(sys.stdout)
            sys.stdout = tee
            try:
                mcj.coadd_jysr(lsts, v, scope, root=sd, workroot=None, last_it=None, out_dir=tmp, tag=C.mosaic_v5.TAG, product='I',
                               domain=dom)
            except SystemExit as e:           # 'NO TILES CONTRIBUTED'
                u.refuse(f'G10 {dom}: {e.code} - REFUSED')
            finally:
                sys.stdout = tee.real
            verdicts[dom] = {m.group(1): m.group(2) for m in (_VERDICT.match(x) for x in tee.lines) if m}
        u.meta['verdicts'] = verdicts
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
    try:
        rec = CV.caveat_map(t['tiles'], variant_of(w), scope, os.path.dirname(P['caveat_map']), P['unfilled_pbeff'])
    except CV.CaveatRefusal as e:
        _clear([P['caveat_map'], P['caveat_table']])
        u.refuse(f'{e}')
    u.meta['caveat'] = {k: rec[k] for k in ('n_hybrid', 'n_taper_only', 'taper_only')}
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
        os.replace(tb, P['filled_jybeam'])
    finally:
        _clear([tj, tb])
    u.meta['fill'] = {'jysr': {k: rj[k] for k in ('donor_mhz', 's', 'test', 'src_md5')},
                      'jybeam': {k: rb[k] for k in ('identity_rel', 'k_median', 'k_p05', 'k_p95', 'n_bright', 'src_md5')}}
    u.meta['md5'] = {'unfilled_jysr': co['meta']['md5']['jysr'], 'unfilled_jybeam': co['meta']['md5']['jybeam'],
                     'filled_jysr': _md5(P['filled_jysr']), 'filled_jybeam': _md5(P['filled_jybeam'])}
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
    for dom in ('jysr', 'jybeam'):
        for k in ('mollweide', 'radec'):
            u.add_output(P[f'render_{dom}_{k}'], 'mosaic_render', source=P[f'filled_{dom}'], psfmap=P['unfilled_psf'], kind=k,
                         scope=scope, dec_cut=R.RENDER_DEC_CUT_DEG)


# ---- the step ---------------------------------------------------------------------------------------------------------------------

def select_stages(args):
    st = list(STAGES)
    if getattr(args, 'stages', None):
        want = [s.strip() for s in args.stages.split(',') if s.strip()]
        bad = [s for s in want if s not in STAGES]
        if bad:
            raise ValueError(f'--stages: unknown v5 stage(s) {bad} (stages: {",".join(STAGES)})')
        st = [s for s in STAGES if s in want]
    for flag, val in (('--from', getattr(args, 'from_stage', None)), ('--until', getattr(args, 'until_stage', None))):
        if val and val not in STAGES:
            raise ValueError(f'{flag}: unknown v5 stage {val!r} (stages: {",".join(STAGES)})')
    if getattr(args, 'from_stage', None):
        st = [s for s in st if STAGES.index(s) >= STAGES.index(args.from_stage)]
    if getattr(args, 'until_stage', None):
        st = [s for s in st if STAGES.index(s) <= STAGES.index(args.until_stage)]
    return st


def select_units(args):
    """[(stage, scope, weighting)] in pipeline order: every stage of a (scope, weighting) before the next pair."""
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
    return [(st, sc, w) for sc in scopes for w in ws for st in stages_for(sc, stages)]


def out_problems(out, hybrid_out):
    """--out is a work run of its own: never inside or above an archive run of the hybrid route (the mosaic runs, the hybrid
    archive, the masters), nor the hybrid work run --hybrid-out (its records are read only)."""
    C = _cfg()
    from exopipe.phase3.steps.hybrid import _within
    o = os.path.realpath(out)
    held = {'ARCHIVE_HYBRID': C.hybrid.ARCHIVE_HYBRID, 'ARCHIVE_MASTERS': C.hybrid.ARCHIVE_MASTERS,
            'FILLED_MOSAIC_RUN': C.hybrid_tilefill.FILLED_MOSAIC_RUN}
    held.update({f'UNFILLED_MOSAIC_RUNS[{i}]': r for i, r in enumerate(C.hybrid_tilefill.UNFILLED_MOSAIC_RUNS)})
    if hybrid_out:
        held['--hybrid-out'] = hybrid_out
    bad = []
    for k, v in sorted(held.items()):
        r = os.path.realpath(v)
        if _within(o, r):
            bad.append(f'--out {out} is inside {k} = {v} (read only)')
        elif _within(r, o):
            bad.append(f'--out {out} contains {k} = {v} (read only)')
    return bad


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
            if st == 'stage':
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
        with run.unit(unit_name(st, sc, w), stage=st, scope=sc, weighting=w, generation='v5') as u:
            if st == 'stage':
                stage_unit(u, sc, w, run.out_root, hybrid_out)
            elif st == 'coadd':
                coadd_unit(u, sc, w, run.out_root, work)
            elif st == 'caveat':
                caveat_unit(u, sc, w, run.out_root)
            elif st == 'fill':
                fill_unit(u, sc, w, run.out_root)
            else:
                render_unit(u, sc, w, run.out_root, work)
    return run.exit_code()
