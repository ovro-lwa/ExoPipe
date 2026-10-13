"""`exopipe3 mosaic` stage qa (run-wide unit 'qa/_all'), MEASURE ONLY (R-MOSAIC-V2-FLUX-SCALE-MEASURED-NOT-CORRECTED):

- calibrator aperture flux of every listed Jy/sr map against Scaife & Heald 2012 (exopipe_ops/mosaic_v2_flux_anchor_measure.py
  run in place on a directory of links to the maps the coadd records list - the script globs its directory, ANC:37);
- VLSSr astrometry of every listed Jy/sr map (lib/mosaic/verify_mosaic.astrometry_check in place; a NEW measure-only table).

Inputs: the jysr map and psfmap of every selected job whose coadd record is done (read from the record, never a glob of the
build tree). Outputs (class mosaic_qa_table): <out>/qa/flux_anchor_measured_<yyyymmdd>.csv, <out>/qa/vlssr_astrometry_<yyyymmdd>.csv.
"""
import os
import shutil
import sys

from exopipe.phase3.lib.mosaic import steps_common as SC


def _say(msg):
    print(msg, file=sys.stderr, flush=True)


def qa_maps(run, jobs):
    """[(job, jysr map, psfmap)] from the done coadd records of the jobs; [(job, reason)] for the others."""
    from exopipe.phase3.core.record import load_record
    have, miss = [], []
    for job in jobs:
        if 'jysr' not in job.domains:
            continue
        rec = load_record(run.out_root, SC.STEP, SC.unit_name('coadd', job))
        if rec is None or rec.get('status') != 'done':
            miss.append((job, 'no done coadd record' if rec is None else f"coadd record {rec.get('status')}"))
            continue
        files = {os.path.basename(o['path']): o['path'] for o in rec.get('outputs', [])}
        (mname, _), (pname, _) = SC.coadd_products(job, 'jysr')
        if mname in files and pname in files and os.path.exists(files[mname]) and os.path.exists(files[pname]):
            have.append((job, files[mname], files[pname]))
        else:
            miss.append((job, 'coadd record lists no jysr map + psfmap on disk'))
    return have, miss


def run_qa(args, run, jobs, ctx):
    from exopipe.phase3.core.record import NoInput, Refusal
    C = SC.cfg()
    if C.mosaic_qa.APPLY_FLUX_ANCHOR:              # also checked at start (G11)
        raise Refusal('G11: mosaic_qa.APPLY_FLUX_ANCHOR is True - the mosaics are never rescaled '
                      '(R-MOSAIC-V2-FLUX-SCALE-MEASURED-NOT-CORRECTED)')
    qa_dir = SC.layout(run.out_root, 'qa')
    date = SC.utc_date()
    anchor = os.path.join(qa_dir, C.mosaic_qa.ANCHOR_TABLE_NAME.format(yyyymmdd=date))
    vlssr = os.path.join(qa_dir, C.mosaic_qa.VLSSR_TABLE_NAME.format(yyyymmdd=date))
    have, miss = qa_maps(run, jobs)
    for job, why in miss:
        _say(f'mosaic qa: {job.name} left out ({why})')
    name = 'qa/_all'
    inputs = [p for _, m, s in have for p in (m, s)]
    if have and run.resume and not run.fresh:
        ok, why = run.resume_check(name, inputs=inputs, admitted=None)
        if ok:
            u = run.unit(name)
            u.status, u.reason = 'skipped-valid', why
            _say(f'mosaic qa: skipped (resume: {why})')
            return 'skipped-valid'
    anchor_script = SC.script_path(SC.ANCHOR_SCRIPT)
    ldir = os.path.join(qa_dir, f'.links_{os.getpid()}')
    tmp = os.path.join(qa_dir, f'.tmp_{os.getpid()}')
    try:
        with run.unit(name, jobs=[j.name for j, _, _ in have], left_out=[(j.name, w) for j, w in miss]) as u:
            if not have:
                raise NoInput('no selected job has a done coadd record with a jysr map and psfmap')
            for p in inputs:
                u.add_input(p)
            u.add_input(anchor_script)
            u.add_input(C.mosaic_qa.VLSSR_CATALOG)
            shutil.rmtree(ldir, ignore_errors=True)
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(ldir)
            os.makedirs(tmp)
            for job, m, p in have:
                os.symlink(m, os.path.join(ldir, os.path.basename(m)))
                os.symlink(p, os.path.join(ldir, os.path.basename(p)))
            u.plan(anchor, 'mosaic_qa_table')
            u.plan(vlssr, 'mosaic_qa_table')
            task = {'kind': 'qa', 'key': 'qa', 'anchor_script': anchor_script, 'link_dir': ldir,
                    'anchor_out': os.path.join(tmp, os.path.basename(anchor)),
                    'vlssr_out': os.path.join(tmp, os.path.basename(vlssr)),
                    'maps': [{'variant': j.variant, 'scope': j.scope, 'jysr': m} for j, m, _ in have],
                    'nsrc': C.mosaic_qa.VLSSR_NSRC, 'minflux': C.mosaic_qa.VLSSR_MIN_FLUX_JY,
                    'log': os.path.join(SC.layout(run.out_root, 'logs'), 'qa.log')}
            r = SC.run_tasks([task], ctx['lanes'](1, 1), ctx['task_dir'], 'qa')['qa']
            u.meta['qa_task'] = {k: r.get(k) for k in ('status', 'message', 'wall_s', 'log')}
            if r.get('legacy_code'):
                u.meta.setdefault('legacy_code', {}).update(r['legacy_code'])
            if r.get('status') == 'refused':
                u.refuse(f'qa: {r.get("message")}')
            if r.get('status') != 'ok':
                raise RuntimeError(f'qa task failed: {r.get("message")}')
            os.replace(task['anchor_out'], anchor)
            os.replace(task['vlssr_out'], vlssr)
            maps = [{'variant': j.variant, 'band': j.scope, 'jysr': m} for j, m, _ in have]
            u.add_output(anchor, 'mosaic_qa_table', kind='anchor', maps=maps)
            u.add_output(vlssr, 'mosaic_qa_table', kind='vlssr', maps=maps)
        return u.status
    except Exception as e:
        _say(f'mosaic qa: error {type(e).__name__}: {e}')
        return 'error'
    finally:
        shutil.rmtree(ldir, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)
