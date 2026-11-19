"""Helpers shared by `exopipe3 mosaic --generation v3` (lib/mosaic/v3.py) and `--generation v5` (lib/mosaic/v5.py): one copy of
each rule both generations apply (spec plan/specs/hybrid.md section 4, the PR-6 note deferred until PR-7d and PR-6c merged).
  - the md5 cache (one per process for both generations), clearing a unit's outputs, binding the co-add engine's globals and
    checking them against the coadd record;
  - the records a unit reads: an earlier stage's done record, the staged tile list (each tile and weighting map at its staged
    md5), the tile list a record was made from, the coadd's maps at their recorded md5s;
  - the tile tree the engine reads, stage selection, the --out check;
  - the PKCORR stage body (after its gates) and TEST 2 of the content stage (the files the test read = the tile list's).
A unit is named in refusals by `what`: '<scope> <weighting>' (v5) or '<scope> <job label>' (v3).
"""
import json
import os

_MD5 = {}


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


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


def _bind_engine():
    """Set coadd_jysr's module globals from mosaic_coadd (steps_common.engine_globals) and clear the retired July override tables
    (spec 3); returns the globals. Every stage that runs the engine (coadd, pkcorr, caveat, content) runs under them."""
    from exopipe.phase3.lib.mosaic import coadd_jysr as mcj
    from exopipe.phase3.lib.mosaic import steps_common as SC
    g = SC.engine_globals()
    for k, val in g.items():
        setattr(mcj, k, val)
    mcj.BEAM_OVERRIDE.clear()
    mcj.SKIP_CELLS.clear()
    return g


def _engine_as_coadd(u, scope, label, co):
    """Bind the engine globals and refuse unless they equal the coadd record's (the weights the co-added maps were made with)."""
    g = json.loads(json.dumps(_bind_engine()))
    if g != co['meta'].get('engine_globals'):
        u.refuse(f'{scope} {label}: the mosaic_coadd weights now {g} are not the coadd\'s {co["meta"].get("engine_globals")} - REFUSED')
    u.meta['engine_globals'] = g
    return g


def upstream(u, out, step, name, what, st):
    """The done record `name` (stage st of the unit `what`), else refuse."""
    from exopipe.phase3.core.record import load_record
    r = load_record(out, step, name)
    if r is None:
        u.refuse(f'no {st} record for {what} in {out} (run --stages {st} for it first) - REFUSED')
    if r.get('status') != 'done':
        u.refuse(f'the {st} unit of {what} ended {r.get("status")}: {str(r.get("reason"))[:200]} - REFUSED')
    return r


def staged_tiles(u, p, what, verify=True, masters=None):
    """The staged tile list at p (a done stage unit's tiles.json; t['json_md5'] = its md5), recorded as inputs with the files it lists.
    verify: every weighting map, and the tile master of every LST in masters (default all), must still have the md5 the stage
    recorded; those are the unit's inputs."""
    with open(p) as f:
        t = json.load(f)
    t['json_md5'] = _md5(p)
    u.add_input(p, extra={'md5': t['json_md5']})
    use = (lambda lst: True) if masters is None else (lambda lst: lst in masters)
    if verify:
        bad = [f'{r["lst"]} {k}' for r in t['tiles'] for k, f_, m in (('tile', r['path'], r['md5']), ('PB', r['pb'], r.get('pb_md5')))
               if (k == 'PB' or use(r['lst'])) and (not os.path.exists(f_) or _md5(f_) != m)]
        if bad:
            u.refuse(f'{what}: staged file(s) missing or changed since the stage (md5): {bad} - REFUSED')
    for r in t['tiles']:
        if use(r['lst']):
            u.add_input(r['path'], extra={'md5': r['md5']})
        u.add_input(r['pb'], extra={'md5': r.get('pb_md5')})
    return t


def same_tiles(u, what, cur, rec, st):
    """Refuse unless the done record rec (of stage st) was made from the tile list whose md5 is cur."""
    if rec['meta'].get('tiles_md5') != cur:
        u.refuse(f'{what}: the {st} unit was made from another tile list (stage/tiles.json md5 {rec["meta"].get("tiles_md5")}, '
                 f'now {cur}): rerun --stages {st} - REFUSED')


def coadd_maps(u, maps, co):
    """The coadd's maps {key: path} at the md5s its record holds for those keys, recorded as inputs; else refuse."""
    for k, pp in maps.items():
        want = co['meta']['md5'][k]
        if not os.path.exists(pp) or _md5(pp) != want:
            u.refuse(f'{pp} missing or changed since the coadd (md5) - REFUSED')
        u.add_input(pp, extra={'md5': want})


def files_under(d):
    """Every file under the directory d, sorted (a conform unit's products: their names carry the tiles' dates)."""
    out = []
    for root, _dirs, files in os.walk(d):
        out += [os.path.join(root, f) for f in files]
    return sorted(out)


def link_tree(t, sd, tile_name, pb_name):
    """The staged tiles and their weighting maps linked under the engine's names (coadd_jysr._paths: <sd>/<lst>/<tile_name>,
    <pb_name>) in the scratch dir sd; returns the LSTs in tile-list order."""
    for r in t['tiles']:
        d = os.path.join(sd, r['lst'])
        os.makedirs(d, exist_ok=True)
        os.symlink(r['path'], os.path.join(d, tile_name))
        os.symlink(r['pb'], os.path.join(d, pb_name))
    return [r['lst'] for r in t['tiles']]


def select_stages(args, stages, gen):
    """The stages of --stages / --from / --until, in pipeline order (all by default); an unknown name raises ValueError."""
    st = list(stages)
    if getattr(args, 'stages', None):
        want = [s.strip() for s in args.stages.split(',') if s.strip()]
        bad = [s for s in want if s not in stages]
        if bad:
            raise ValueError(f'--stages: unknown {gen} stage(s) {bad} (stages: {",".join(stages)})')
        st = [s for s in stages if s in want]
    for flag, val in (('--from', getattr(args, 'from_stage', None)), ('--until', getattr(args, 'until_stage', None))):
        if val and val not in stages:
            raise ValueError(f'{flag}: unknown {gen} stage {val!r} (stages: {",".join(stages)})')
    if getattr(args, 'from_stage', None):
        st = [s for s in st if stages.index(s) >= stages.index(args.from_stage)]
    if getattr(args, 'until_stage', None):
        st = [s for s in st if stages.index(s) <= stages.index(args.until_stage)]
    return st


def run_engine(u, lsts, variant, scope, sd, tmp, tag, product):
    """The co-add engine (coadd_jysr.coadd_jysr) on the tile tree sd, both domains (Jy/sr, Jy/beam), writing into tmp; binds the engine
    globals first (u.meta['engine_globals']). Returns {domain: {lst: verdict}} from the engine's per-tile lines (u.meta['verdicts']).
    The engine's 'NO TILES CONTRIBUTED' exit refuses (G10); the tile-guard rule on the verdicts is the caller's (v3: G3/G5 and G9;
    v5: HG14)."""
    import sys
    from exopipe.phase3.lib.mosaic import coadd_jysr as mcj
    from exopipe.phase3.lib.mosaic.steps_worker import _VERDICT, _Tee
    u.meta['engine_globals'] = _bind_engine()
    os.makedirs(tmp, exist_ok=True)
    verdicts = {}
    for dom in ('jysr', 'jybeam'):
        tee = _Tee(sys.stdout)
        sys.stdout = tee
        try:
            mcj.coadd_jysr(lsts, variant, scope, root=sd, workroot=None, last_it=None, out_dir=tmp, tag=tag, product=product, domain=dom)
        except SystemExit as e:           # 'NO TILES CONTRIBUTED'
            u.refuse(f'G10 {dom}: {e.code} - REFUSED')
        finally:
            sys.stdout = tee.real
        verdicts[dom] = {m.group(1): m.group(2) for m in (_VERDICT.match(x) for x in tee.lines) if m}
    u.meta['verdicts'] = verdicts
    return verdicts


def held_problems(out, held):
    """--out against the read-only roots held {name: path}: never inside nor above one."""
    from exopipe.phase3.steps.hybrid import _within
    o = os.path.realpath(out)
    bad = []
    for k, v in sorted(held.items()):
        r = os.path.realpath(v)
        if _within(o, r):
            bad.append(f'--out {out} is inside {k} = {v} (read only)')
        elif _within(r, o):
            bad.append(f'--out {out} contains {k} = {v} (read only)')
    return bad


def pkcorr(u, *, what, scope, variant, tag, t, maps, outp, link, scratch, udir, work, label, co):
    """The PKCORR stage after its gates: MP run_band (lib/mosaic/pkcorr.py) on the co-added Jy/sr and PSF maps (maps['jysr'],
    maps['psf']), mosaic_pkcorr.ROUNDS injection rounds through the engine on the tile tree link(sd) builds, then test_pkcorr. The
    four outputs (outp: map, curve, recovery, json -> path) are written under a temporary directory in udir and renamed into place;
    scratch = the names of the unit's two scratch directories under work (the tile tree, the injections). The engine is bound to the
    coadd record co's globals first and refused unless they match (_engine_as_coadd, label = the job): the injections run with
    the coadd's weights whatever unit ran before in the process."""
    import shutil
    from exopipe.phase3.core import fitsio
    from exopipe.phase3.lib.mosaic import coadd_jysr as mcj
    from exopipe.phase3.lib.mosaic import pkcorr as PK
    C = _cfg()
    _engine_as_coadd(u, scope, label, co)
    nside, rounds = int(mcj.NSIDE), int(C.mosaic_pkcorr.ROUNDS)
    mos_c, psf = PK.read_map(maps['jysr']), PK.read_map(maps['psf'])
    if len(mos_c) != 12 * nside ** 2 or len(psf) != 12 * nside ** 2:
        u.refuse(f'{what}: the co-added maps hold {len(mos_c)} / {len(psf)} pixels, not NSIDE {nside} - REFUSED')
        return
    sd = fitsio.scratch_dir(scratch[0], work)
    inj = fitsio.scratch_dir(scratch[1], work)
    tmp = os.path.join(udir, f'.tmp_pkcorr_{os.getpid()}')
    try:
        lsts = link(sd)
        os.makedirs(tmp, exist_ok=True)
        try:
            _T, res = PK.run_band(sd, scope, variant, tmp, lsts, rounds, tag, mos_c, psf, inj, nside,
                                  record_dir=os.path.dirname(outp['map']))
        except PK.InjectionRefusal as e:
            u.refuse(f'{what}: {e} - REFUSED')
            return
        os.makedirs(os.path.dirname(outp['map']), exist_ok=True)
        for k in ('map', 'curve', 'recovery', 'json'):
            os.replace(os.path.join(tmp, os.path.basename(outp[k])), outp[k])
    finally:
        fitsio.cleanup(sd)
        fitsio.cleanup(inj)
        shutil.rmtree(tmp, ignore_errors=True)
    tp = res['test_pkcorr']
    u.meta['pkcorr'] = {k: res[k] for k in ('rounds', 'rec_median', 'rec_min', 'rec_max', 'ra_scatter_median_mad')}
    u.meta['md5'] = {k: _md5(outp[k]) for k in ('map', 'curve', 'recovery', 'json')}
    u.meta['tiles_md5'] = t['json_md5']
    u.check('test_pkcorr', tp.get('verdict') == 'pass', value=tp, path=outp['map'], cls='mosaic_pkcorr_map',
            detail=f'nodes {tp["nodes"]}, few rounds {tp["missing_or_few_rounds"][:5]}, out of range {tp["out_of_range"][:5]}, '
                   f'map vs curve {tp["max_map_err"]:.2e}')
    ctx = dict(rounds=rounds, scope=scope, variant=variant)
    u.add_output(outp['map'], 'mosaic_pkcorr_map', curve=outp['curve'], **ctx)
    u.add_output(outp['curve'], 'mosaic_pkcorr_curve', **ctx)
    u.add_output(outp['recovery'], 'mosaic_pkcorr_recovery', **ctx)
    u.add_output(outp['json'], 'mosaic_pkcorr_report', map_path=outp['map'], **ctx)


def test2_write(rep, t, path):
    """TEST 2 of the content stage: per domain, the files the test read ('read', removed from the report) against the tile list t
    (each LST's real tile and PB paths): r['provenance'] (each read file with its tile's verified md5) and
    r['provenance_matches_tiles']. Mutates rep in place (test2_check reads the keys added here). The report is written to path
    (temporary name, renamed into place); returns its md5."""
    listed = {r['lst']: (os.path.realpath(r['path']), os.path.realpath(r['pb']), r['md5']) for r in t['tiles']}
    for _dom, r in rep.items():
        read = r.pop('read')
        r['provenance'] = [dict(x, md5=listed[x['lst']][2] if x['lst'] in listed else None) for x in read]
        r['provenance_matches_tiles'] = (len(read) == len(listed)
                                         and all(listed.get(x['lst'], ())[:2] == (x['file'], x['pb']) for x in read))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(rep, f, indent=1)
    os.replace(tmp, path)
    return _md5(path)


def test2_check(u, dom, r, n_listed, path, cls):
    """The TEST2.<dom> check of one domain of a report test2_write wrote: the test read exactly the listed tiles and PB maps,
    through the tile tree's links."""
    ok = r['provenance_matches_tiles']
    u.check(f'TEST2.{dom}', ok and all(x['symlink'] for x in r['provenance']), value=len(r['provenance']), path=path, cls=cls,
            detail=(f'the test read the {len(r["provenance"])} listed tiles and PB maps, each at its verified md5' if ok else
                    f'the test read {len(r["provenance"])} files, not the {n_listed} listed tiles and PB maps'))
