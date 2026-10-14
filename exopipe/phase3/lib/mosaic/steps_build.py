"""`exopipe3 mosaic` stages inputs, coadd and render (parent side; the work runs in steps_worker child processes).

inputs   per job: the plan's gates (G1 exclusions, G3 one master per tile, G4 tile count, G8 header), the node-local stage
         made by the IN-PLACE stage script (one per scope, all its variants and both robusts, as production staged it), the
         stage checks (every link resolves to the planned master, G3; the weighting PB of the variant's robust exists, G5,
         and passes mosaic_weight_pb incl. its grid, G6), and the census of every tile with the engine's own guard
         arithmetic (G9 verdicts; TILE_GUARD_ACTION). Outputs: the staged weighting PB maps (class mosaic_weight_pb; node-
         local scratch, removed at the end of the run - an inputs unit is never resumed, it runs again with its coadd).
coadd    per job and domain: coadd_jysr in place on the stage; the engine's printed tile verdicts must equal the census
         (G9); 'NO TILES CONTRIBUTED' refuses (G10); the maps are placed atomically into <out>/healpix/ only when every
         domain of the job succeeded. Outputs: mosaic_map, mosaic_psfmap (jysr), mosaic_pbeff (jybeam).
render   per job in mosaic_render.SCOPES and domain: band_products.build in place on links to the coadd record's map and
         psfmap; Mollweide + RA/Dec renders placed atomically into <out>/renders/. Output class mosaic_render.
"""
import os
import shutil
import sys

from exopipe.phase3.lib.mosaic import steps_common as SC


def _cfg():
    return SC.cfg()


def _say(msg):
    print(msg, file=sys.stderr, flush=True)


def _rm(path):
    shutil.rmtree(path, ignore_errors=True)


def record_inputs(u, P, scripts=()):
    """Inputs common to the inputs and coadd units: the masters (FITS identity: size + DATASUM/processing cards) and the
    in-place scripts (their sha1 = code identity of the in-place code); the admitted nights ('<lst>:<date>' of every
    constituent) are the exclusion identity - data_decisions.csv itself is not an input, so an unrelated exclusion row
    does not force a rebuild of all 76 jobs (its fingerprint is kept in the inputs unit's meta)."""
    for lst, m in sorted(P.masters.items()):
        u.add_input(m)
    for s in scripts:
        u.add_input(s)
    u.set_admitted(P.admitted)


def _child_code(u, res):
    """sha1 of every legacy / exopipe source file the child imported (core/legacy fingerprint) into meta['legacy_code'],
    which record.resume_check re-hashes - a changed in-place module makes the unit stale."""
    if res and res.get('legacy_code'):
        u.meta.setdefault('legacy_code', {}).update(res['legacy_code'])


def tile_meta(P, census):
    """Per tile: the facts conform needs (provenance row, time block, beam) and the census verdict."""
    out = []
    for lst in P.lsts:
        t = P.tiles.get(lst) or {}
        h = t.get('hdr') or {}
        c = (census or {}).get(lst) or {}
        out.append({'lst': lst, 'master': t.get('master'), 'realpath': c.get('master_realpath') or h.get('realpath'),
                    'verdict': c.get('verdict'), 'sigma': c.get('sigma'), 'weight': c.get('weight'),
                    'n_app': c.get('n_app'), 'n_nights': len(h.get('dates') or ()),
                    'MJD-OBS': h.get('MJD-OBS'), 'MJD-MID': h.get('MJD-MID'), 'INTTIME': h.get('INTTIME'),
                    'THERMNOI': h.get('THERMNOI'), 'GITTAG': h.get('GITTAG'), 'DATE-END': h.get('DATE-END'),
                    'BMAJ': h.get('BMAJ'), 'BMIN': h.get('BMIN'), 'freq_hz': h.get('freq_hz'), 'mdewarp': h.get('mdewarp')})
    return out


def _resume_coadd(run, job, P):
    ok, why = run.resume_check(SC.unit_name('coadd', job), inputs=None, admitted=P.admitted)
    return ok, why


def run_inputs_coadd(args, run, jobs, plans, stages, ctx):
    """Stages inputs (+ coadd when selected) for the jobs. ctx: dict shared with the step (scratch root, lanes, task dir,
    the in-place script paths). Returns {job name: 'done' | 'skipped-valid' | 'refused' | 'error'} of the LAST stage run."""
    from exopipe.phase3.core.record import Refusal, load_record
    C = _cfg()
    want_coadd = 'coadd' in stages
    state = {}
    todo = []
    for job in jobs:
        P = plans[job.name]
        if want_coadd and run.resume and not run.fresh and not P.refusals:
            ok, why = _resume_coadd(run, job, P)
            if ok:
                for st in ('inputs', 'coadd'):
                    u = run.unit(SC.unit_name(st, job), job=job.name)
                    u.status, u.reason = 'skipped-valid', (why if st == 'coadd' else 'coadd resumed: ' + why)
                _say(f'mosaic coadd {job.name}: skipped (resume: {why})')
                state[job.name] = 'skipped-valid'
                continue
            _say(f'mosaic coadd {job.name}: not resumed ({why})')
        if P.refusals:
            with run.unit(SC.unit_name('inputs', job), job=job.name, lsts=P.lsts) as u:
                for n in P.notes:
                    u.note(n)
                u.meta['tiles'] = tile_meta(P, None)
                u.refuse(P.reason())
            state[job.name] = 'refused'
            continue
        todo.append(job)
    if not todo:
        return state

    stage_script = SC.script_path(SC.STAGE_SCRIPT)
    stage_root = ctx['stage_root']()                      # node-local scratch (refuses on Lustre / no scratch root)
    by_scope = {}
    for job in todo:
        by_scope.setdefault(job.scope, []).append(job)
    # one stage per scope; the in-place script refuses the WHOLE scope on a duplicate candidate of ANY variant
    stage_res, scope_refusal = {}, {}
    tasks = []
    for scope, sjobs in by_scope.items():
        dups = SC.scope_duplicates(scope, plans[sjobs[0].name].lsts)
        if dups:
            scope_refusal[scope] = ('G3: the in-place stage script refuses scope ' + scope + ' (more than one current '
                                    'master: ' + ', '.join(f'{l} {v} x{n}' for l, v, n in dups[:6]) + ')')
            continue
        sdir = os.path.join(stage_root, scope)
        os.makedirs(sdir, exist_ok=True)
        tasks.append({'kind': 'stage', 'key': scope, 'script': stage_script, 'stage': sdir, 'scope': scope,
                      'log': os.path.join(SC.layout(run.out_root, 'logs'), f'stage_{scope}.log')})
    if tasks:
        stage_res = SC.run_tasks(tasks, ctx['lanes'](len(tasks), C.mosaic.CORES_PER_JOB), ctx['task_dir'], 'stage')
    # census of every job whose scope staged
    census_tasks, census_res, prechk = [], {}, {}
    for job in todo:
        P = plans[job.name]
        sdir = os.path.join(stage_root, job.scope)
        if job.scope in scope_refusal or (stage_res.get(job.scope) or {}).get('status') != 'ok':
            continue
        from exopipe.phase3.lib.masters.pb_apply import deep_pb_name
        pbn = deep_pb_name(job.scope, job.robust)
        mname = f'I_deep_{job.variant}_pbcorr_{job.scope}.fits'
        bad = []
        for lst in P.lsts:
            link = os.path.join(sdir, lst, mname)
            if not os.path.lexists(link):
                bad.append(f'G3: {lst}: no staged master link {mname}')
            elif os.path.realpath(link) != os.path.realpath(P.tiles[lst]['master']):
                bad.append(f'G3: {lst}: staged {os.path.realpath(link)} is not the planned master '
                           f'{P.tiles[lst]["master"]}')
            if not os.path.exists(os.path.join(sdir, lst, pbn)):
                bad.append(f'G5: {lst}: no weighting PB {pbn} (derivation failed or skipped; see the stage log)')
        prechk[job.name] = (bad, pbn, mname)
        if bad:
            continue
        census_tasks.append({'kind': 'census', 'key': job.name, 'stage': sdir, 'lsts': P.lsts, 'master_name': mname,
                             'pb_name': pbn, 'pb_floor': C.mosaic_coadd.PB_FLOOR,
                             'min_valid_px': C.mosaic_coadd.MIN_VALID_TILE_PIXELS,
                             'flat_sigma': C.mosaic_coadd.CORRUPT_FLAT_SIGMA_JYBEAM,
                             'log': os.path.join(SC.layout(run.out_root, 'logs'), f'census_{job.name}.log')})
    if census_tasks:
        census_res = SC.run_tasks(census_tasks, ctx['lanes'](len(census_tasks), C.mosaic.CORES_PER_JOB),
                                  ctx['task_dir'], 'census')
    # inputs units
    ready = []
    for job in todo:
        P = plans[job.name]
        sdir = os.path.join(stage_root, job.scope)
        try:
            with run.unit(SC.unit_name('inputs', job), job=job.name, lsts=P.lsts, stage=sdir) as u:
                record_inputs(u, P, (stage_script,))
                for n in P.notes:
                    u.note(n)
                ex = ctx['exclusions']
                u.meta['exclusions'] = ex.fingerprint()
                u.check('mosaic.G1_exclusions', True, len(P.admitted), 'every constituent night of every tile admitted')
                u.check('mosaic.G3_one_master_per_tile', True, len(P.masters),
                        f"{sum(1 for t in P.tiles.values() if (t.get('hdr') or {}).get('mdewarp'))} _mdewarp_ form")
                u.check('mosaic.G8_headers', True, len(P.masters), 'PBCORR, PB factor cards, beam, frequency present')
                if job.scope in scope_refusal:
                    u.refuse(scope_refusal[job.scope])
                sr = stage_res.get(job.scope) or {}
                u.meta['stage_task'] = {k: sr.get(k) for k in ('status', 'message', 'links', 'absent', 'pb_failures',
                                                               'wall_s', 'log')}
                _child_code(u, sr)
                if sr.get('status') == 'refused':
                    u.refuse(f'stage script refused: {sr.get("message")}')
                if sr.get('status') != 'ok':
                    raise RuntimeError(f'stage script failed: {sr.get("message")}')
                if sr.get('ROOT') and sr['ROOT'].rstrip('/') != C.mosaic.INPUT_COADD_RUN.rstrip('/'):
                    u.refuse(f"G0: the stage script staged ROOT {sr['ROOT']} != mosaic.INPUT_COADD_RUN")
                bad, pbn, mname = prechk[job.name]
                if bad:
                    u.refuse('; '.join(bad[:6]) + (f' ... ({len(bad)} problems)' if len(bad) > 6 else ''))
                cr = census_res.get(job.name) or {}
                _child_code(u, cr)
                if cr.get('status') != 'ok':
                    raise RuntimeError(f'census failed: {cr.get("message")}')
                tiles = cr['tiles']
                u.meta['tiles'] = tile_meta(P, tiles)
                skipped = {l: r['verdict'] for l, r in tiles.items() if r['verdict'] != 'added'}
                missing = [l for l, v in skipped.items() if v == 'MISSING']
                if missing:
                    u.refuse(f'G3/G5: tiles {missing} missing at census (MISSING_TILE_ACTION refuse)')
                if skipped:
                    msg = ', '.join(f'{l} {v}' for l, v in sorted(skipped.items()))
                    if C.mosaic.TILE_GUARD_ACTION == 'refuse':
                        u.refuse(f'G9: tiles the engine would skip: {msg} (mosaic.TILE_GUARD_ACTION refuse)')
                    u.note(f'G9: tiles the engine skips (recorded; TILE_GUARD_ACTION skip-and-record): {msg}')
                if not any(r['verdict'] == 'added' for r in tiles.values()):
                    u.refuse('G10: no tile would contribute')
                u.check('mosaic.G9_tile_guard', True, sum(1 for r in tiles.values() if r['verdict'] == 'added'),
                        f'{len(skipped)} tile(s) skipped by the engine guard')
                for lst in P.lsts:
                    u.add_output(os.path.join(sdir, lst, pbn), 'mosaic_weight_pb', master=P.tiles[lst]['master'],
                                 robust=job.robust)
                u.meta['scratch_outputs'] = True
            if u.status == 'done':
                ready.append(job)
                state[job.name] = 'done'
            else:
                state[job.name] = u.status
        except Exception as e:                            # recorded as 'error' in the unit; go on with the other jobs
            _say(f'mosaic inputs {job.name}: error {type(e).__name__}: {e}')
            state[job.name] = 'error'
    if not want_coadd or not ready:
        return state
    return run_coadd(args, run, ready, plans, ctx, state)


def run_coadd(args, run, jobs, plans, ctx, state):
    """coadd_jysr in place per (job, domain); records the coadd units."""
    from exopipe.phase3.core.record import load_record
    C = _cfg()
    hp_dir = SC.layout(run.out_root, 'healpix')
    logs = SC.layout(run.out_root, 'logs')
    os.makedirs(hp_dir, exist_ok=True)
    glob_vals = SC.engine_globals()
    tasks, tmps = [], {}
    for job in jobs:
        P = plans[job.name]
        for dom in job.domains:
            tmp = os.path.join(hp_dir, f'.tmp_{job.name}_{dom}_{os.getpid()}')
            _rm(tmp)
            tmps[(job.name, dom)] = tmp
            tasks.append({'kind': 'coadd', 'key': f'{job.name}_{dom}', 'variant': job.variant, 'scope': job.scope,
                          'domain': dom, 'lsts': P.lsts, 'stage': os.path.join(ctx['stage_root'](), job.scope),
                          'out_tmp': tmp, 'globals': glob_vals,
                          'log': os.path.join(logs, f'{job.variant}_{job.scope}_{dom}.log')})   # run_mosaic_v2.sh:24
    res = SC.run_tasks(tasks, ctx['lanes'](len(tasks), C.mosaic.CORES_PER_JOB), ctx['task_dir'], 'coadd')
    stage_script = SC.script_path(SC.STAGE_SCRIPT)
    for job in jobs:
        P = plans[job.name]
        inp = load_record(run.out_root, SC.STEP, SC.unit_name('inputs', job)) if not run.dry_run else None
        tiles = (inp or {}).get('meta', {}).get('tiles') or []
        census = {t['lst']: t['verdict'] for t in tiles}
        try:
            with run.unit(SC.unit_name('coadd', job), job=job.name, domains=list(job.domains), lsts=P.lsts) as u:
                record_inputs(u, P, (stage_script,))
                u.meta['tiles'] = tiles
                u.meta['engine_globals'] = glob_vals
                planned = []
                for dom in job.domains:
                    for name, cls in SC.coadd_products(job, dom):
                        planned.append((dom, name, cls))
                        u.plan(os.path.join(hp_dir, name), cls)
                problems, refusals = [], []
                for dom in job.domains:
                    r = res.get(f'{job.name}_{dom}') or {}
                    _child_code(u, r)
                    u.meta.setdefault('coadd_tasks', {})[dom] = {k: r.get(k) for k in ('status', 'message', 'wall_s',
                                                                                       'log', 'wrote')}
                    if r.get('status') == 'refused':
                        refusals.append(f'{dom}: {r.get("message")}')
                        continue
                    if r.get('status') != 'ok':
                        problems.append(f'{dom}: {r.get("message")}')
                        continue
                    eng = r.get('verdicts') or {}
                    diff = sorted(l for l in set(eng) | set(census) if eng.get(l) != census.get(l))
                    if diff:
                        refusals.append(f'G9 {dom}: engine tile verdicts differ from the census at '
                                        + ', '.join(f'{l} (engine {eng.get(l)}, census {census.get(l)})' for l in diff[:6]))
                if problems:
                    raise RuntimeError('; '.join(problems))
                if refusals:
                    u.refuse('; '.join(refusals))
                contributing = [t['lst'] for t in tiles if t.get('verdict') == 'added']
                skipped = [(t['lst'], t.get('verdict')) for t in tiles if t.get('verdict') != 'added']
                u.meta['contributing'] = contributing
                u.meta['skipped'] = skipped
                if skipped:
                    u.note('tiles skipped by the engine guard (recorded): ' + ', '.join(f'{l} {v}' for l, v in skipped))
                # every domain succeeded: place atomically, then test
                for dom, name, cls in planned:
                    src = os.path.join(tmps[(job.name, dom)], name)
                    if not os.path.exists(src):
                        u.refuse(f'{dom}: the engine did not write {name}')
                for dom, name, cls in planned:
                    os.replace(os.path.join(tmps[(job.name, dom)], name), os.path.join(hp_dir, name))
                # review 2026-10-13: a map with fewer CONTRIBUTING tiles than REQUIRE_N_LSTS (a waived --lsts subset, or
                # tiles the engine guard skipped under TILE_GUARD_ACTION 'skip-and-record' = today's behaviour) is a
                # partial build for the coverage content test - else the test turned skip-and-record into a refusal
                partial = len(contributing) < C.mosaic.REQUIRE_N_LSTS
                psf_name = SC.coadd_products(job, 'jysr')[1][0]
                jysr_name = SC.coadd_products(job, 'jysr')[0][0]
                jyb_name = SC.coadd_products(job, 'jybeam')[0][0]
                for dom, name, cls in planned:
                    u.add_output(os.path.join(hp_dir, name), cls, tiles=tiles, partial=partial, domain=dom,
                                 psfmap=os.path.join(hp_dir, psf_name), jysr_map=os.path.join(hp_dir, jysr_name),
                                 jybeam_map=os.path.join(hp_dir, jyb_name))
            state[job.name] = u.status
        except Exception as e:
            _say(f'mosaic coadd {job.name}: error {type(e).__name__}: {e}')
            state[job.name] = 'error'
        finally:
            for dom in job.domains:
                _rm(tmps.get((job.name, dom), ''))
    return state


# ---------------------------------------------------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------------------------------------------------

def _coadd_outputs(run, job):
    """{(class, domain-or-None): path} of the job's done coadd record, or (None, reason)."""
    from exopipe.phase3.core.record import load_record
    rec = load_record(run.out_root, SC.STEP, SC.unit_name('coadd', job))
    if rec is None:
        return None, 'no coadd record (run the coadd stage first)'
    if rec.get('status') != 'done':
        return None, f"coadd record status {rec.get('status')}: {rec.get('reason')}"
    return rec, ''


def _map_paths(rec, job):
    out = {}
    for o in rec.get('outputs', []):
        out[os.path.basename(o['path'])] = o['path']
    return out


def run_render(args, run, jobs, ctx):
    """band_products.build in place per (job, domain) for the jobs in mosaic_render.SCOPES."""
    C = _cfg()
    rnd_dir = SC.layout(run.out_root, 'renders')
    hp_dir = SC.layout(run.out_root, 'healpix')
    logs = SC.layout(run.out_root, 'logs')
    state = {}
    todo, tasks, tmps, links = [], [], {}, {}
    for job in jobs:
        if job.scope not in C.mosaic_render.SCOPES:
            _say(f'mosaic render {job.name}: not rendered (scope not in mosaic_render.SCOPES; sub-band maps are not '
                 f'rendered today)')
            continue
        doms = [d for d in job.domains if d in C.mosaic_render.DOMAINS]
        if not doms:
            continue
        rec, why = _coadd_outputs(run, job)
        name = SC.unit_name('render', job)
        if rec is None:
            with run.unit(name, job=job.name) as u:
                u.refuse(f'render needs the coadd: {why}')
            state[job.name] = 'refused'
            continue
        files = _map_paths(rec, job)
        need = {}
        psf = files.get(SC.coadd_products(job, 'jysr')[1][0]) or os.path.join(hp_dir, SC.coadd_products(job, 'jysr')[1][0])
        for d in doms:
            mname = SC.coadd_products(job, d)[0][0]
            mp = files.get(mname)
            if not mp:
                mp = os.path.join(hp_dir, mname) if os.path.exists(os.path.join(hp_dir, mname)) else None
            need[d] = mp
        missing = [d for d, p in need.items() if not p or not os.path.exists(p)]
        if missing:
            with run.unit(name, job=job.name) as u:
                u.refuse(f'render: the {",".join(missing)} map(s) of {job.name} are not in the coadd record / on disk')
            state[job.name] = 'refused'
            continue
        inputs = [p for p in need.values()] + ([psf] if os.path.exists(psf) else [])
        if run.resume and not run.fresh:
            ok, why = run.resume_check(name, inputs=inputs, admitted=None)
            if ok:
                u = run.unit(name, job=job.name)
                u.status, u.reason = 'skipped-valid', why
                _say(f'mosaic render {job.name}: skipped (resume: {why})')
                state[job.name] = 'skipped-valid'
                continue
        todo.append((job, doms, need, psf, inputs))
        for d in doms:
            ldir = os.path.join(rnd_dir, f'.links_{job.name}_{d}_{os.getpid()}')
            tmp = os.path.join(rnd_dir, f'.tmp_{job.name}_{d}_{os.getpid()}')
            _rm(ldir)
            _rm(tmp)
            os.makedirs(ldir)
            # band_products reads by the production names (always nside2048 there, BP:41, :67)
            os.symlink(need[d], os.path.join(ldir, f'healpix_{job.stem}_{d}_nside2048.fits'))
            if os.path.exists(psf):
                os.symlink(psf, os.path.join(ldir, f'healpix_PSF_{job.stem}_nside2048.fits'))
            links[(job.name, d)], tmps[(job.name, d)] = ldir, tmp
            tasks.append({'kind': 'render', 'key': f'{job.name}_{d}', 'variant': job.variant, 'scope': job.scope,
                          'domain': d, 'hpx_dir': ldir, 'out_tmp': tmp,
                          'moll_W': C.mosaic_render.MOLLWEIDE_WIDTH_PX,
                          'car_cdelt_deg': C.mosaic_render.CAR_CDELT_ARCMIN / 60.0,     # BP:375 car_arcmin / 60
                          'beams': dict(C.mosaic_render.BEAM_FALLBACK_ARCMIN),
                          'log': os.path.join(logs, f'render_{job.variant}_{job.scope}_{d}.log')})
    if tasks:
        res = SC.run_tasks(tasks, ctx['lanes'](min(len(tasks), C.mosaic.RENDER_MAX_PARALLEL), 1), ctx['task_dir'],
                           'render')
    else:
        res = {}
    os.makedirs(rnd_dir, exist_ok=True)
    for job, doms, need, psf, inputs in todo:
        try:
            with run.unit(SC.unit_name('render', job), job=job.name, domains=doms) as u:
                for p in inputs:
                    u.add_input(p)
                planned = []
                for d in doms:
                    for fname, kind in SC.render_products(job, d):
                        planned.append((d, fname, kind))
                        u.plan(os.path.join(rnd_dir, fname), 'mosaic_render')
                bad, err = [], []
                for d in doms:
                    r = res.get(f'{job.name}_{d}') or {}
                    _child_code(u, r)
                    u.meta.setdefault('render_tasks', {})[d] = {k: r.get(k) for k in ('status', 'message', 'wall_s', 'log')}
                    if r.get('status') == 'refused':
                        bad.append(f'{d}: {r.get("message")}')
                    elif r.get('status') != 'ok':
                        err.append(f'{d}: {r.get("message")}')
                if err:
                    raise RuntimeError('; '.join(err))
                if bad:
                    u.refuse('; '.join(bad))
                for d, fname, kind in planned:
                    if not os.path.exists(os.path.join(tmps[(job.name, d)], fname)):
                        u.refuse(f'{d}: band_products did not write {fname}')
                for d, fname, kind in planned:
                    os.replace(os.path.join(tmps[(job.name, d)], fname), os.path.join(rnd_dir, fname))
                psf_used = psf if os.path.exists(psf) else None
                if psf_used is None:
                    u.note(f'render beam: the July fallback table (no PSF map {os.path.basename(psf)}; B-45)')
                for d, fname, kind in planned:
                    u.add_output(os.path.join(rnd_dir, fname), 'mosaic_render', source=need[d], psfmap=psf_used,
                                 kind=kind, scope=job.scope)
            state[job.name] = u.status
        except Exception as e:
            _say(f'mosaic render {job.name}: error {type(e).__name__}: {e}')
            state[job.name] = 'error'
        finally:
            for d in doms:
                _rm(links.get((job.name, d), ''))
                _rm(tmps.get((job.name, d), ''))
    return state
