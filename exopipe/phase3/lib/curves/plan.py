"""The plan of one snapshot cell for `exopipe3 curves` (spec curves.md sections 4-5).

``plan_cell`` decides, per subband, whether its (subband, Stokes) pairs are PLANNED:
- not searched: no ``admission_<sb>.csv`` under paths.SNAPSHOT_FLEET_OUT/<cell>/ (PLANNED_SUBBANDS = 'admission'; BF:29-30, :47) -
  outside the search universe, never planned (G15 when no subband of the cell has one);
- excluded: ``core.exclusions.is_excluded(lst, sb, date)`` (G2; CLAUDE.md 2) - recorded, never extracted or published;
- flow-unregistered: ``runinfo_<sb>_I_median.json`` says ``flow_registered`` False (G3; ledger 1421, open) - recorded;
- planned otherwise (a night-refused subband keeps its admission table and is extracted as today, ledger 2302 / B-70).
``search_done`` is gate G1, ``stage_problems`` gate G4 on a given stage root, ``catalog_path`` the Master catalogue of
gate G6. ``inputs`` lists what the unit fingerprints (run record; --resume identity).
"""
import glob
import json
import os

STOKES_ALL = ('I', 'V')


def _paths():
    """config.paths: the shared roots SNAPSHOT_FLEET_OUT (search outputs), SNAPSHOT_FLOW_ROOT (flow fields), CLV15_ROOT
    (target samples) - refactor decision 3."""
    from exopipe.phase3.config import CFG
    return CFG.paths


def catalog_path(C, lst, sb):
    return os.path.join(C.MASTER_CATALOG_DIR, f'master_R075_{lst}_{sb}.gaul.csv')


def table_name(cell, sb, st):
    return f'{cell}_{sb}_{st}_lightcurves.fits'


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _flow_counts(C, cell):
    """{subband: number of flow_<sb>_<ts>.npy files} under paths.SNAPSHOT_FLOW_ROOT/<cell>/ (one directory listing)."""
    d = os.path.join(_paths().SNAPSHOT_FLOW_ROOT, cell)
    counts = {}
    try:
        names = os.listdir(d)
    except FileNotFoundError:
        return counts
    for n in names:
        if n.startswith('flow_') and n.endswith('.npy'):
            sb = n[5:].split('_', 1)[0]
            counts[sb] = counts.get(sb, 0) + 1
    return counts


def plan_cell(cell, sbs, stokes, ex, C, all_subbands):
    """The plan of one cell: dict with lst, date, cell, stokes, subbands {sb: info}, planned [sb], search_dir."""
    lst, date = cell.split('_', 1)
    sdir = os.path.join(_paths().SNAPSHOT_FLEET_OUT, cell)
    flows = _flow_counts(C, cell)
    scope = list(all_subbands) if sbs is None else list(sbs)
    info = {}
    for sb in scope:
        adm = os.path.join(sdir, f'admission_{sb}.csv')
        runinfo = {st: os.path.join(sdir, f'runinfo_{sb}_{st}_median.json') for st in STOKES_ALL}
        nightref = {st: os.path.join(sdir, f'nightrefused_{sb}_{st}_median.json') for st in STOKES_ALL}
        i = {'admission': adm, 'runinfo': {st: p for st, p in runinfo.items() if os.path.exists(p)},
             'nightrefused': {st: p for st, p in nightref.items() if os.path.exists(p)},
             'catalog': catalog_path(C, lst, sb), 'flow_files': flows.get(sb, 0), 'reason': ''}
        i['catalog_exists'] = os.path.exists(i['catalog'])
        if C.PLANNED_SUBBANDS != 'admission':
            raise ValueError(f"curves.PLANNED_SUBBANDS = {C.PLANNED_SUBBANDS!r}: only 'admission' is implemented")
        if not os.path.exists(adm):
            i['status'], i['reason'] = 'not-searched', f'no admission table {adm}'
        elif ex.is_excluded(lst, sb, date):
            i['status'], i['reason'] = 'excluded', f'data_decisions.csv excludes ({lst}, {sb}, {date})'
        else:
            ri = _read_json(runinfo['I']) if 'I' in i['runinfo'] else None
            if C.SKIP_FLOW_UNREGISTERED and ri is not None and ri.get('flow_registered') is False:
                i['status'], i['reason'] = 'flow-unregistered', f"{runinfo['I']}: flow_registered False (ledger 1421)"
            else:
                i['status'] = 'planned'
                if ri is not None:
                    i['n_frames_search'] = ri.get('n_frames')
        info[sb] = i
    planned = [sb for sb in scope if info[sb]['status'] == 'planned']
    # G1 looks at EVERY searched subband of the cell, whatever --scopes selects (the fleet's completeness rule is per cell)
    unaccounted = []
    for sb in all_subbands:
        if os.path.exists(os.path.join(sdir, f'admission_{sb}.csv')):
            for st in STOKES_ALL:
                if not (os.path.exists(os.path.join(sdir, f'runinfo_{sb}_{st}_median.json'))
                        or os.path.exists(os.path.join(sdir, f'nightrefused_{sb}_{st}_median.json'))):
                    unaccounted.append(f'{sb}_{st}')
    return {'cell': cell, 'lst': lst, 'date': date, 'stokes': tuple(stokes), 'subbands': info, 'planned': planned,
            'search_dir': sdir, 'searched': [sb for sb in scope if info[sb]['status'] != 'not-searched'],
            'unaccounted': unaccounted}


def search_done(plan, C):
    """Gate G1 (R-SNAP-LIGHTCURVE-BACKFILL: 'every shipped, fleet-DONE cell'): (ok, detail).

    Every searched subband of the cell (admission table present; all subbands, not only the selected ones) must have, for
    each Stokes, a runinfo or a night-refusal record (the fleet's completeness rule, v9:215-229). Window evidence ('fleet-claim-done'): <FLEET_CLAIMS_ROOT>/<cell>/done (v9:380)
    replaces the grep of the fleet logs (BF:31-33). 'search-record': <SEARCH_RECORD_ROOT>/_records/search/<cell>.json with
    status done."""
    missing = plan['unaccounted']
    if missing:
        return False, (f'search incomplete: {len(missing)} searched (subband, Stokes) without a runinfo or night-refusal '
                       f'record: {" ".join(missing[:8])}' + (' ...' if len(missing) > 8 else ''))
    cell = plan['cell']
    if C.SEARCH_DONE_EVIDENCE == 'fleet-claim-done':
        p = os.path.join(C.FLEET_CLAIMS_ROOT, cell, 'done')
        if not os.path.exists(p):
            return False, f'the fleet has not finished the cell ({p} absent)'
        return True, f'fleet claim done ({p}); every searched (subband, Stokes) has a runinfo or night-refusal record'
    if C.SEARCH_DONE_EVIDENCE == 'search-record':
        if not C.SEARCH_RECORD_ROOT:
            return False, "curves.SEARCH_DONE_EVIDENCE = 'search-record' needs curves.SEARCH_RECORD_ROOT"
        p = os.path.join(C.SEARCH_RECORD_ROOT, '_records', 'search', f'{cell}.json')
        rec = _read_json(p)
        if rec is None:
            return False, f'no search record {p}'
        if rec.get('status') != 'done':
            return False, f"search record {p} status {rec.get('status')!r}"
        return True, f'search record done ({p})'
    return False, f'unknown curves.SEARCH_DONE_EVIDENCE {C.SEARCH_DONE_EVIDENCE!r}'


def stage_problems(stage_root, plan):
    """Gate G4 on a staged cell: {subband: problem} for planned subbands whose staging is not complete: no directory, no
    decompressed frame, compressed or temporary leftovers (.fs/.fz/.tmp), or a staging list (.filelist, as the fleet and
    the backfill write it, v9:106-111) whose length differs from the staged frame count (v9:116-122, BF:59-64).
    ``*_flowed.fits`` siblings (written by the flow stage) are not frames."""
    out = {}
    for sb in plan['planned']:
        d = os.path.join(stage_root, sb)
        if not os.path.isdir(d):
            out[sb] = f'no staged directory {d}'
            continue
        names = os.listdir(d)
        frames = [n for n in names if n.endswith('-image.fits')]
        left = [n for n in names if n.endswith(('.fs', '.fz', '.tmp'))]
        if left:
            out[sb] = f'{len(left)} compressed/temporary leftovers in {d} (first {left[0]})'
        elif not frames:
            out[sb] = f'no decompressed *-image.fits frame in {d}'
        else:
            fl = os.path.join(d, '.filelist')
            if os.path.exists(fl):
                with open(fl) as f:
                    n_list = sum(1 for line in f if line.strip())
                if n_list != len(frames):
                    out[sb] = f'{len(frames)} staged frames != {n_list} listed in {fl}'
    return out


def phase1_dirs(C, plan, sb, phase1_root):
    """Phase-1 frame directories of one subband (every Run_*): the archive the stage was copied from."""
    out = []
    for rd in sorted(glob.glob(os.path.join(phase1_root, plan['lst'], plan['date'], 'Run_*'))):
        for sub in ('snapshots', 'snapshots_clean'):
            d = os.path.join(rd, sb, sub)
            if os.path.isdir(d):
                out.append(d)
    return out


def inputs(plan, C, phase1_root, code_files):
    """[(path, extra or None)] the unit fingerprints: per searched subband its admission table and search records; per
    planned subband its Master catalogue, flow-field directory and Phase-1 frame directories; the target samples, the PB
    library directory (its 90 files are a pinned 2026-07-04 product, identified by the directory), the fleet claim, and
    the worker code (the in-place clv15 modules and the shim, by sha1). An absent Master catalogue or sample file is
    identified by its directory (size + mtime: a file added later invalidates --resume); a missing input recorded as
    missing would make the unit never resumable (core/record.same_fingerprint)."""
    out = []
    for sb in plan['searched']:
        i = plan['subbands'][sb]
        out.append((i['admission'], None))
        for p in list(i['runinfo'].values()) + list(i['nightrefused'].values()):
            out.append((p, None))
    flowdir = os.path.join(_paths().SNAPSHOT_FLOW_ROOT, plan['cell'])
    for sb in plan['planned']:
        i = plan['subbands'][sb]
        if os.path.exists(i['catalog']):
            out.append((i['catalog'], None))
        else:                                   # G6: the directory identifies the absence (its mtime changes on a new file)
            out.append((C.MASTER_CATALOG_DIR, {'absent': os.path.basename(i['catalog'])}))
        for d in phase1_dirs(C, plan, sb, phase1_root):
            try:
                n = len(os.listdir(d))
            except OSError:
                n = None
            out.append((d, {'n_files': n}))
    if plan['planned']:
        out.append((flowdir, {'n_flow_files': {sb: plan['subbands'][sb]['flow_files'] for sb in plan['planned']}}))
        missing = []
        for _tag, fn in C.SAMPLES:
            p = os.path.join(_paths().CLV15_ROOT, fn)
            if os.path.exists(p):
                out.append((p, None))
            else:
                missing.append(fn)
        if missing:                             # skipped as today (B-58); the directory identifies the absence
            out.append((_paths().CLV15_ROOT, {'absent': missing}))
        out.append((C.PB_LIBRARY, None))
        if C.SEARCH_DONE_EVIDENCE == 'fleet-claim-done':
            out.append((os.path.join(C.FLEET_CLAIMS_ROOT, plan['cell'], 'done'), None))
        for p in code_files:
            out.append((p, None))
    return out


def admitted(plan):
    """The admitted set of this unit (resume identity): '<date>/<sb>' for every planned subband."""
    return [f"{plan['date']}/{sb}" for sb in plan['planned']]
