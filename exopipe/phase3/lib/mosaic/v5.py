"""`exopipe3 mosaic --generation v5`: the hybrid mosaics (spec plan/specs/hybrid.md 2.8, 2.10-2.14; PR-7a, PR-7b). Unit of work: (stage,
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
          -> healpix/healpix_I_deep_<V>_<S>_lsfill_{jysr,jybeam}_nside2048.fits [mosaic_filled_map]; md5s in the record (the band
          field of `exopipe3 hybrid --stages tilefill --mosaic-out <out>` reads them)
  render  lib/mosaic/band_products.build of the FILLED maps with their PSF map (the beam card), mosaic_render's width, pixel and
          southern edge. -> renders/<S>_<V>_lsfill_{jysr,jybeam}_{Mollweide,RADec}.fits [mosaic_render]
  content lib/mosaic/content (MT): TEST 1 of the UNFILLED Jy/sr and Jy/beam maps against their own staged tiles, and TEST 2 (the
          links and the md5 of each tile's first 64 MB), on the link tree of the coadd. -> content/content_I_deep_<V>_<S>_lsfill.json
          [mosaic_content_report]. Production's V5 tables/summary.csv rows are the unfilled maps' values; TEST 1R is the Stokes V
          test (the hybrid mosaics are Stokes I only)

A stage reads its predecessors' unit records (status done), never a glob, and refuses unless they were made from the tile list now
in stage/tiles.json (its md5 in the coadd and fill records). A unit removes its own earlier outputs first; the fill renames a map
into place only when its inline test passed; a unit that does not end done (a refusal, an error, a failed content test) removes the
outputs it placed, and its record keeps the reason. The stages that read the tiles (coadd, pkcorr, caveat, content) bind the
mosaic_coadd globals and refuse unless they equal the coadd record's. Conformance v5, the tables and the push class are PR-7c.
"""
import json
import os

LSTS = tuple(f'{h:02d}h' for h in range(24))
STAGES = ('stage', 'coadd', 'pkcorr', 'caveat', 'fill', 'render', 'content')
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
         'content': os.path.join(d, 'content', f'content_I_deep_{v}_{scope}{tag}.json')}
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
    raise ValueError(f'unknown v5 stage {st!r}')


def stages_for(scope, stages):
    """The selected stages that take this scope (caveat: hybrid.CAVEAT_SCOPES; render: mosaic_v5.RENDER_SCOPES)."""
    C = _cfg()
    return [s for s in stages if not (s == 'caveat' and scope not in C.hybrid.CAVEAT_SCOPES)
            and not (s == 'render' and scope not in C.mosaic_v5.RENDER_SCOPES)]


_MD5 = {}


def _key(p):
    st = os.stat(p)
    return (os.path.realpath(p), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def _md5(p):
    """md5 of a file, kept for the process by (real path, device, inode, size, mtime): the units of one run hash a staged tile (on
    Lustre) once; a later process, or a file changed since, hashes again."""
    from exopipe.phase3.lib.install.steps_common import md5
    k = _key(p)
    if k not in _MD5:
        _MD5[k] = md5(p)
    return _MD5[k]


def _seed(p, m):
    """The md5 a writer computed for the file now at p (renamed into place: same inode and mtime)."""
    _MD5[_key(p)] = m


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
    """The staged tiles (tiles.json of the done stage unit; t['json_md5'] = its md5). verify (the units that read the tiles: coadd,
    caveat): every tile and weighting map must still have the md5 the stage recorded."""
    _upstream(u, out, 'stage', scope, w)
    p = paths(out, scope, w)['tiles']
    with open(p) as f:
        t = json.load(f)
    t['json_md5'] = _md5(p)
    u.add_input(p, extra={'md5': t['json_md5']})
    if verify:
        bad = [f'{r["lst"]} {k}' for r in t['tiles'] for k, f_, m in (('tile', r['path'], r['md5']), ('PB', r['pb'], r.get('pb_md5')))
               if not os.path.exists(f_) or _md5(f_) != m]
        if bad:
            u.refuse(f'{scope} {w}: staged file(s) missing or changed since the stage (md5): {bad} - REFUSED')
    for r in t['tiles']:
        u.add_input(r['path'], extra={'md5': r['md5']})
        u.add_input(r['pb'], extra={'md5': r.get('pb_md5')})
    return t


def _same_tiles(u, scope, w, cur, rec, st):
    """Refuse unless the done record rec (of stage st) was made from the tile list whose md5 is cur."""
    if rec['meta'].get('tiles_md5') != cur:
        u.refuse(f'{scope} {w}: the {st} unit was made from another tile list (stage/tiles.json md5 {rec["meta"].get("tiles_md5")}, '
                 f'now {cur}): rerun --stages {st} - REFUSED')


def _bind_engine():
    """Set coadd_jysr's module globals from mosaic_coadd (steps_common.engine_globals) and clear the retired July override tables
    (spec 3); returns the globals. The coadd and the caveat (its weights) both run under them."""
    from exopipe.phase3.lib.mosaic import coadd_jysr as mcj
    from exopipe.phase3.lib.mosaic import steps_common as SC
    g = SC.engine_globals()
    for k, val in g.items():
        setattr(mcj, k, val)
    mcj.BEAM_OVERRIDE.clear()
    mcj.SKIP_CELLS.clear()
    return g


def _engine_as_coadd(u, scope, w, co):
    """Bind the engine globals and refuse unless they equal the coadd record's (the weights the unfilled maps were made with)."""
    g = json.loads(json.dumps(_bind_engine()))
    if g != co['meta'].get('engine_globals'):
        u.refuse(f'{scope} {w}: the mosaic_coadd weights now {g} are not the coadd\'s {co["meta"].get("engine_globals")} - REFUSED')
    u.meta['engine_globals'] = g
    return g


def _link_tree(t, scope, w, sd):
    """The staged tiles and their weighting maps linked under the engine's names (coadd_jysr._paths) in the scratch dir sd."""
    from exopipe.phase3.lib.masters.pb_apply import deep_pb_name
    v = variant_of(w)
    for r in t['tiles']:
        d = os.path.join(sd, r['lst'])
        os.makedirs(d, exist_ok=True)
        os.symlink(r['path'], os.path.join(d, f'I_deep_{v}_pbcorr_{scope}.fits'))
        os.symlink(r['pb'], os.path.join(d, deep_pb_name(scope, w)))
    return [r['lst'] for r in t['tiles']]


def _coadd_maps(u, out, scope, w, co, keys):
    """The coadd's unfilled maps (keys of coadd_names) at their recorded md5s, recorded as inputs; else refuse."""
    P = paths(out, scope, w)
    for k in keys:
        pp, want = P[f'unfilled_{k}'], co['meta']['md5'][k]
        if not os.path.exists(pp) or _md5(pp) != want:
            u.refuse(f'{pp} missing or changed since the coadd (md5) - REFUSED')
        u.add_input(pp, extra={'md5': want})


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
    import sys
    from exopipe.phase3.core import fitsio
    from exopipe.phase3.lib.mosaic import coadd_jysr as mcj
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
        lsts = _link_tree(t, scope, w, sd)
        u.meta['engine_globals'] = _bind_engine()
        os.makedirs(tmp, exist_ok=True)
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
    for dom in ('jysr', 'jybeam'):
        for k in ('mollweide', 'radec'):
            u.add_output(P[f'render_{dom}_{k}'], 'mosaic_render', source=P[f'filled_{dom}'], psfmap=P['unfilled_psf'], kind=k,
                         scope=scope, dec_cut=R.RENDER_DEC_CUT_DEG)


# ---- pkcorr -----------------------------------------------------------------------------------------------------------------------

def pkcorr_unit(u, scope, w, out, work):
    import shutil
    from exopipe.phase3.core import fitsio
    from exopipe.phase3.lib.mosaic import coadd_jysr as mcj
    from exopipe.phase3.lib.mosaic import pkcorr as PK
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
    nside, rounds, v, tag = int(mcj.NSIDE), int(C.mosaic_pkcorr.ROUNDS), variant_of(w), C.mosaic_v5.TAG
    mos_c, psf = PK.read_map(P['unfilled_jysr']), PK.read_map(P['unfilled_psf'])
    if len(mos_c) != 12 * nside ** 2 or len(psf) != 12 * nside ** 2:
        u.refuse(f'{scope} {w}: the unfilled maps hold {len(mos_c)} / {len(psf)} pixels, not NSIDE {nside} - REFUSED')
    sd = fitsio.scratch_dir(f'mosaic_v5_pkcorr_{scope}_{w}', work)
    inj = fitsio.scratch_dir(f'mosaic_v5_pkcorr_inj_{scope}_{w}', work)
    tmp = os.path.join(unit_dir(out, scope, w), f'.tmp_pkcorr_{os.getpid()}')
    try:
        lsts = _link_tree(t, scope, w, sd)
        os.makedirs(tmp, exist_ok=True)
        try:
            _T, res = PK.run_band(sd, scope, v, tmp, lsts, rounds, tag, mos_c, psf, inj, nside,
                                  record_dir=os.path.dirname(P['pkcorr_map']))
        except PK.InjectionRefusal as e:
            u.refuse(f'{scope} {w}: {e} - REFUSED')
        os.makedirs(os.path.dirname(P['pkcorr_map']), exist_ok=True)
        for k in ('map', 'curve', 'recovery', 'json'):
            os.replace(os.path.join(tmp, os.path.basename(P[f'pkcorr_{k}'])), P[f'pkcorr_{k}'])
    finally:
        fitsio.cleanup(sd)
        fitsio.cleanup(inj)
        shutil.rmtree(tmp, ignore_errors=True)
    tp = res['test_pkcorr']
    u.meta['pkcorr'] = {k: res[k] for k in ('rounds', 'rec_median', 'rec_min', 'rec_max', 'ra_scatter_median_mad')}
    u.meta['md5'] = {k: _md5(P[f'pkcorr_{k}']) for k in ('map', 'curve', 'recovery', 'json')}
    u.meta['tiles_md5'] = t['json_md5']
    u.check('test_pkcorr', tp.get('verdict') == 'pass', value=tp, path=P['pkcorr_map'], cls='mosaic_pkcorr_map',
            detail=f'nodes {tp["nodes"]}, few rounds {tp["missing_or_few_rounds"][:5]}, out of range {tp["out_of_range"][:5]}, '
                   f'map vs curve {tp["max_map_err"]:.2e}')
    ctx = dict(rounds=rounds, scope=scope, variant=v)
    u.add_output(P['pkcorr_map'], 'mosaic_pkcorr_map', curve=P['pkcorr_curve'], **ctx)
    u.add_output(P['pkcorr_curve'], 'mosaic_pkcorr_curve', **ctx)
    u.add_output(P['pkcorr_recovery'], 'mosaic_pkcorr_recovery', **ctx)
    u.add_output(P['pkcorr_json'], 'mosaic_pkcorr_report', map_path=P['pkcorr_map'], **ctx)


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
    listed = {r['lst']: (os.path.realpath(r['path']), os.path.realpath(r['pb']), r['md5']) for r in t['tiles']}
    for dom, r in rep.items():
        read = r.pop('read')
        r['provenance'] = [dict(x, md5=listed[x['lst']][2] if x['lst'] in listed else None) for x in read]
        r['provenance_matches_tiles'] = (len(read) == len(listed)
                                         and all(listed.get(x['lst'], ())[:2] == (x['file'], x['pb']) for x in read))
    os.makedirs(os.path.dirname(P['content']), exist_ok=True)
    tmp = f'{P["content"]}.tmp.{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(rep, f, indent=1)
    os.replace(tmp, P['content'])
    for dom, r in rep.items():
        u.check(f'TEST1.{dom}', r['result'] == 'PASS', value={k: r[k] for k in ('n_tiles', 'n_sources', 'median_ratio', 'p05', 'p95',
                                                                              'frac_within_5pct', 'result')},
                path=P['content'], cls='mosaic_content_report',
                detail=f'{r["n_sources"]} sources, median map / tiles {r["median_ratio"]:.5f}, within 5 % {r["frac_within_5pct"]:.3f}')
        u.check(f'TEST2.{dom}', r['provenance_matches_tiles'] and all(x['symlink'] for x in r['provenance']),
                value=len(r['provenance']), path=P['content'], cls='mosaic_content_report',
                detail=(f'TEST 1 read the {len(r["provenance"])} listed tiles and PB maps, each at its verified md5'
                        if r['provenance_matches_tiles'] else
                        f'TEST 1 read {len(r["provenance"])} files, not the {len(t["tiles"])} listed tiles and PB maps'))
    u.add_output(P['content'], 'mosaic_content_report', n_tiles=len(t['tiles']), scope=scope, variant=v)


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
        own = [p for p, _c in outputs_of(st, run.out_root, sc, w)]
        with run.unit(unit_name(st, sc, w), stage=st, scope=sc, weighting=w, generation='v5') as u:
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
                else:
                    content_unit(u, sc, w, run.out_root, work)
            except BaseException:
                _clear(own)                   # nothing of a unit that did not end done stays in place
                raise
            if u.failed_tests():
                _clear(own)
                u.note(f'a content test failed: the {len(own)} outputs were removed (the record keeps the results)')
    return run.exit_code()
