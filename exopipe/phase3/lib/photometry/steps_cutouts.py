"""`exopipe3 photometry --stages cutouts | cutmeasure`: the photometry CUTOUT product of every tier, written into a cutout
GENERATION root (CUTOVER_READINESS 'PHOTOMETRY CUTOUTS REGENERATED FROM THE FLOW CUBES, WITHOUT RE-SEARCHING' and 'EVERY TIER';
Gregg 2026-10-25 ~00:52 UTC "Forced photometry is also approved", ~02:15 UTC "I want the photometry done for all tiers btw,
including 1 hour (deep)").

Stage cutouts, one unit per item:
  snapshot  per (night cell, band): the search's own path to the cutouts on the consolidated flow cubes, then stop - in place,
            phase3_snapshot_search.py --cutouts-from-cubes (subbands, V then I) / phase3_snapshot_group_search.py
            --cutouts-from-cubes GROUP (widebands, per Stokes V then I), as snapshot_cutout_regen_20261025.sh :26-32 and
            snapshot_cutout_regen_wb_20261025.sh :27-41 ran them. Exit 3 (data_decisions) = excluded, an outcome. Units
            cutouts.<cell>.<band>.
  deep, sublst  per <LST>_<band>: photometry_tier_cutouts_20261025.py --tier deep|10min --cells <LST>_<band> in place (the
            tier residuals: deep Run_P3S5 iresid / vresid, 10-minute tenmin_root_x2 windows). Units cutouts.<deep|10min>.<cell>.
Stage cutmeasure (deep, sublst), one unit per (tier, LST, root): photometry_tier_forced_20261026.py --tier --lsts --samples in
  place (the snapshot chain's estimator on the tier cutouts; planets at each plane's own pixel with orbit controls). Its samples
  go to their served roots (photometry.TIER_FORCED_ROOTS) unless --tier-out names one. Units cutmeasure.<deep|10min>.<LST>.

Every unit writes through a STAGING directory of its own on the root's filesystem (<root>/_staging/<run id>/<unit>/, the
engine's output root), then:
  1. identity of each staged cutout with its source (unit checks): 10 s = the regeneration's content test against the cube
     (tests_cutouts.identity_subband / identity_wideband, the wideband rebuild bounded by the weights' stored precision x
     (A + |S|)); tiers = one plane per file re-cut from its source frame (tier_recut). A failure refuses the unit before
     anything reaches the root (the staging directory is kept and named);
  2. ONE FILE PER (sample, band, object, LST, date, Stokes) (CUTOVER_READINESS 'Photometry cutout writer, snapshot tier'): a
     file of the unit's cell and samples in the root that the unit did not write - another time stamp of a written key
     (stale) or an object / Stokes not written now (orphan) - refuses the unit (C-STALE); with --fresh they are MOVED to
     <root>/quarantine_superseded_cutouts_<run id>/ with a sha256 manifest, never deleted;
  3. placement by os.replace (atomic; a tier file of the same name is replaced in place), then the unit's manifest
     <root>/_manifest/<10s|deep|10min>/<key>_<band>[.<tags>].csv (file, sha256, bytes, sample, object, stokes): output class
     photometry_cutout_manifest (CM1 files + structure + sha256 sample, CM2 one file per key).
cutmeasure stages the tool's --out (the root's existing forced_manifest.csv copied in first, so a sample subset merges as the
tool merges it), places the npz and the manifest (classes photometry_tier_npz, photometry_tier_manifest).

Gates: C-ROOT (gate_write_root: a retired generation never; a production root while paths.PRODUCTION_WRITES is False), C-CELLS
(a regeneration names its items), C-RUN (the tier tool reads one deep run), PHOT-ADHOC (an ad-hoc target set into a served
root, or for cutmeasure / evaluate), PHOT-FULL (snapshot Full).
"""
import csv
import glob
import hashlib
import os
import re
import shutil

from exopipe.phase3.core import cells as C
from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.photometry import steps_common as K


def _cc():
    return K.cfg().photometry_cutouts


def is_wideband(band):
    return band in ('Red', 'Green', 'Blue')


def run_samples(targets, planets):
    """The sample directories a cutout run writes: the ad-hoc tags (+ the planets with --planets), else the measured samples
    and the planets (phase3_snapshot_cutouts._samples / _planets)."""
    ph = K.cfg().photometry
    if targets:
        return [t for t, _ in targets] + ([ph.PLANET_SAMPLE] if planets else [])
    return list(ph.SAMPLES_MEASURED) + [ph.PLANET_SAMPLE]


def tag_suffix(targets):
    return ('.' + '+'.join(t for t, _ in targets)) if targets else ''


def listing_sha1(paths):
    h = hashlib.sha1()
    for p in sorted(paths):
        try:
            st = os.stat(p)
            h.update(('%s\t%d\t%d\n' % (p, st.st_size, st.st_mtime_ns)).encode())
        except OSError:
            h.update(('%s\tmissing\n' % p).encode())
    return h.hexdigest()


def staging_dir(root, run, key):
    """<root>/<STAGING_SUBDIR>/<run id>/<key>/ - fresh, on the root's filesystem (placement is a rename)."""
    d = os.path.join(root, _cc().STAGING_SUBDIR, run.run_id, key)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    return d


def staged_cutouts(stage, samples):
    """(paths relative to the staging root of the cutouts an engine wrote there, <sample>/<band>/<object>/<file>.fits;
    every other file it wrote there, the engine's own output directory aside). A cutout of a sample the run does not cut
    counts among the others: the caller refuses them (nothing an engine wrote is dropped unseen)."""
    from exopipe.phase3.lib.photometry import tests_cutouts as TC
    ok, other = [], []
    for dp, dn, fn in os.walk(stage):
        if os.path.relpath(dp, stage).split(os.sep)[0] == '_engine_out':
            continue
        for f in fn:
            rel = os.path.relpath(os.path.join(dp, f), stage)
            q = TC.parse_cutout(os.path.join(stage, rel))
            (ok if q and rel.count(os.sep) == 3 and q[0] in samples else other).append(rel)
    return sorted(ok), sorted(other)


def remove_staging(stage):
    """Remove a unit's staging directory and the run's staging parents left empty."""
    shutil.rmtree(stage, ignore_errors=True)
    for d in (os.path.dirname(stage), os.path.dirname(os.path.dirname(stage))):
        try:
            os.rmdir(d)
        except OSError:
            break


def quarantine(run, root, rels, reason):
    """Move superseded cutouts out of the generation, never delete them (CLAUDE.md 7): to
    <root>/<QUARANTINE_PREFIX><run id>/<relative path>, each with a row of MANIFEST_sha256.csv (sha256,bytes,moved_from,
    moved_to,reason), its checksum verified after the move. Returns the moved paths."""
    from exopipe.phase3.lib.photometry import tests_cutouts as TC
    q = os.path.join(root, _cc().QUARANTINE_PREFIX + run.run_id)
    man = os.path.join(q, 'MANIFEST_sha256.csv')
    os.makedirs(q, exist_ok=True)
    new = not os.path.exists(man)
    moved = []
    with open(man, 'a') as mf:
        if new:
            mf.write('sha256,bytes,moved_from,moved_to,reason\n')
        for rel in sorted(rels):
            src, dst = os.path.join(root, rel), os.path.join(q, rel)
            h = TC.sha256_file(src)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(src, dst)
            if TC.sha256_file(dst) != h:
                raise Refusal('C-STALE: quarantine checksum of %s differs after the move' % dst)
            mf.write('%s,%d,%s,%s,%s\n' % (h, os.path.getsize(dst), src, dst, reason.replace(',', ';')))
            moved.append(dst)
    return moved


def place_unit(u, run, root, stage, rels, token, key, band, samples, manifest_path, ctx):
    """Steps 2 and 3 of the module docstring: one file per key (refuse, or quarantine with --fresh), sha256, os.replace into
    the root, the manifest, the class tests. Returns the manifest path."""
    from exopipe.phase3.lib.photometry import tests_cutouts as TC
    stale, orphan = TC.superseded(root, token, key, band, samples, rels)
    u.meta.update(n_stale=len(stale), n_orphan=len(orphan))
    if stale or orphan:
        what = '%d other time stamp(s) of a written key (%s) and %d file(s) of the unit\'s cell and samples it did not write ' \
               '(%s)' % (len(stale), TC._fmt(stale), len(orphan), TC._fmt(orphan))
        if not run.fresh:
            u.refuse('C-STALE: %s in %s - one file per (sample, band, object, LST, date, Stokes); rerun with --fresh to move '
                     'them to a quarantine directory (staged files kept: %s)' % (what, root, stage))
        u.meta['quarantined'] = quarantine(run, root, stale + orphan, 'superseded by %s (%s)' % (u.name, run.run_id))
        u.note('C-STALE --fresh: %s moved to quarantine' % what)
    rows = []
    for rel in rels:
        p = TC.parse_cutout(os.path.join(stage, rel))
        sp = os.path.join(stage, rel)
        rows.append(dict(file=rel, sha256=TC.sha256_file(sp), bytes=os.path.getsize(sp), sample=p[0], object=p[2],
                         stokes=p[3]['st']))
    u.plan(manifest_path, 'photometry_cutout_manifest')
    for rel in rels:
        dst = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(os.path.join(stage, rel), dst)
    TC.write_manifest(manifest_path, rows)
    u.add_output(manifest_path, 'photometry_cutout_manifest', root=root, samples=samples, **ctx)
    u.meta['n_cutouts'] = len(rows)
    return manifest_path


# ---------------------------------------------------------------------------------------------------------------------
# snapshot: the 10-s cutouts regenerated from the consolidated flow cubes
# ---------------------------------------------------------------------------------------------------------------------

def snapshot_universe():
    root = K.cfg().paths.SNAPSHOT_CUBES_ROOT
    out = []
    for n in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        try:
            lst, date, sb = C.parse_night_cell(n)
        except ValueError:
            continue
        if not sb:
            out.append(n)
    return out


def snapshot_items(args):
    """[(cell, band)] of --cells / --epochs night cells (<LST>_<date>[_<band>], patterns over the cube store), the bands of
    photometry.SNAPSHOT_BANDS (--scopes narrows), minus --exclude-cells. C-CELLS: a regeneration names its cells."""
    from exopipe.phase3.lib.photometry import steps_snapshot as KS
    spec = ','.join(x for x in (args.cells, getattr(args, 'epochs', None)) if x)
    if not spec:
        raise Refusal('C-CELLS: a cutout regeneration names its night cells (--cells / --epochs; patterns over %s, e.g. '
                      '--cells "18h_*")' % K.cfg().paths.SNAPSHOT_CUBES_ROOT)
    universe = snapshot_universe()
    out = []
    for it in C.expand(spec, universe=universe, kind='night cell'):
        lst, date, sb = C.parse_night_cell(it)
        for b in KS.bands_for(args, sb):
            out.append(('%s_%s' % (lst, date), b))
    if args.exclude_cells:
        ex = set(C.expand(args.exclude_cells, universe=universe, kind='night cell'))
        out = [(c, b) for c, b in out if c not in ex and '%s_%s' % (c, b) not in ex]
    return sorted(set(out))


def snapshot_cubes(cell, band):
    """{Stokes: [cube paths]} the unit reads: the band's own V and I cubes (subband) or every member's (wideband)."""
    from exopipe.phase3.lib.photometry import tests_cutouts as TC
    members = K.cfg().instrument.SUBBAND_BANDS.get(band, ()) if is_wideband(band) else (band,)
    return {S: [TC.cube_path(cell, sb, S) for sb in members if os.path.exists(TC.cube_path(cell, sb, S))] for S in 'VI'}


def stage_snapshot_cutouts(run, args):
    targets = K.adhoc_targets(args)
    root = K.cutout_root(args)
    planets = bool(getattr(args, 'planets', False))
    items = snapshot_items(args)
    samples = run_samples(targets, planets)
    K.plan(run, '[cutouts] %d (night cell, band) units -> %s (generation %s); samples %s%s; engines in place '
                '(phase3_snapshot_search.py / phase3_snapshot_group_search.py --cutouts-from-cubes, site PYTHON[%s])'
           % (len(items), root, K.cfg().photometry.GENERATION if root == K.cfg().photometry.CUTOUT_ROOT else 'not served',
              ','.join(samples), ' (ad hoc: %s)' % ', '.join('%s=%s' % t for t in targets) if targets else '',
              K.python_key('snapshot', 'cutouts_subband')))
    try:
        K.gate_write_root(root, '--cutout-root')
        K.gate_adhoc_roots(targets, {'cutout root': root})
    except Refusal as e:
        if run.dry_run:
            K.would_refuse(run, 'cutouts', str(e))
            return []
        raise
    units = []
    for cell, band in items:
        cubes = snapshot_cubes(cell, band)
        if run.dry_run:
            K.plan(run, '[cutouts]   unit cutouts.%s.%s%s: cubes V %d, I %d' % (cell, band, tag_suffix(targets),
                                                                             len(cubes['V']), len(cubes['I'])))
            continue
        units += snapshot_unit(run, args, cell, band, root, targets, planets, samples, cubes)
    return units


def cutout_admitted(cell, band):
    """The unit's resume identity under the exclusions (core.exclusions, read live): '<cell>:<band>' unless the regeneration
    refuses the (cell, band) - a subband by its own row (phase3_snapshot_search.main :1824-1827), a wideband by the deep rule
    (regen_group_cutouts :467-470) - plus, for a wideband, '<cell>:<member>' of every member subband it keeps (it drops the
    excluded ones, :472-473). A new exclusion changes the set, so --resume re-runs the unit."""
    from exopipe.phase3.core import exclusions
    ex = exclusions.load()
    lst, date = cell.split('_', 1)
    if ex.is_excluded(lst, band, date):
        return []
    out = ['%s:%s' % (cell, band)]
    if is_wideband(band):
        members = K.cfg().instrument.SUBBAND_BANDS.get(band, ())
        out += ['%s:%s' % (cell, sb) for sb in members if not ex.is_excluded(lst, sb, date)]
    return sorted(out)


def _regen_counts(line):
    """{'V': ..., 'I': ...} of a 'CUTOUTS-REGEN <lst> <date> <sb>: V <n> I <n>[; refused: ...]' line (regen_cutouts :1414)."""
    m = re.search(r': V (.*?) I (.*?)(?:; refused: (.*))?$', line.strip())
    if not m:
        return {}, ''
    return {'V': m.group(1), 'I': m.group(2)}, m.group(3) or ''


def snapshot_unit(run, args, cell, band, root, targets, planets, samples, cubes):
    from exopipe.phase3.lib.photometry import tests_cutouts as TC
    c = K.cfg()
    cc = c.photometry_cutouts
    lst, date = cell.split('_', 1)
    sfx = tag_suffix(targets)
    name = 'cutouts.%s.%s%s' % (cell, band, sfx)
    allcubes = cubes['V'] + cubes['I']
    inputs = sorted(allcubes) + [p for _, p in targets] + ([] if targets else list(K.sample_paths().values()))
    admitted = cutout_admitted(cell, band)
    if K.run_resume_ok(run, name, admitted, {'inputs_listing_sha1': listing_sha1(inputs), 'cutout_root': root}):
        return run.units[-1:]
    with run.unit(name, tier='snapshot', stage='cutouts', cell=cell, band=band, cutout_root=root, samples=samples,
                  targets=['%s=%s' % t for t in targets], planets=planets, inputs_listing_sha1=listing_sha1(inputs)) as u:
        if not cubes['V'] or not cubes['I']:
            raise NoInput('C-NOCUBE: no consolidated flow cube of %s %s (%s) under %s' % (
                cell, band, 'V' if not cubes['V'] else 'I', c.paths.SNAPSHOT_CUBES_ROOT))
        for p in inputs + [c.search_snapshot.SOLAR_AUDIT]:      # data_decisions: through the admitted set (exclusions read live)
            u.add_input(p)
        for S in 'VI':
            for sb in (c.instrument.SUBBAND_BANDS.get(band, ()) if is_wideband(band) else (band,)):
                nz = os.path.join(c.paths.SNAPSHOT_FLEET_OUT, cell, 'noise_%s_%s_median.csv' % (sb, S))
                if os.path.exists(nz):          # the frame list of a duplicate-run cube (one frame per integration)
                    u.add_input(nz)
        u.set_admitted(admitted)
        stage = staging_dir(root, run, '%s_%s%s' % (cell, band, sfx))
        u.meta['staging'] = stage
        eout = os.path.join(stage, '_engine_out')
        # the engine's own switches, set to the values the step's checks read (paths.SNAPSHOT_CUBES_ROOT = the engine's CUBE_ROOT
        # default, paths.SNAPSHOT_FLEET_OUT = its SEARCH_OUT_ROOT default - mirror-tested), so the cube the identity test reads
        # is the cube the engine cut; the output root is the unit's staging directory
        env = dict(K.adhoc_env(targets, planets), P3_SNAP_CUTOUT_ROOT=stage, P3_SNAP_CUBE_ROOT=c.paths.SNAPSHOT_CUBES_ROOT,
                   P3_SNAP_SEARCH_OUT_ROOT=c.paths.SNAPSHOT_FLEET_OUT)
        outcome = {}
        members = {}
        if not is_wideband(band):
            argv = ['--lst', lst, '--date', date, '--subband', band, '--stokes', 'VI', '--mode', 'median', '--out', eout,
                    '--cutouts-from-cubes']
            res, text = K.call(u, 'cutouts_subband', argv, 'snapshot', 'cutouts_subband', run, 'cutouts_%s_%s.log' % (cell, band),
                               env_extra=env, threads=args.threads)
            if res.rc == cc.REGEN_EXCLUDED_RC or '\nEXCLUDED:' in '\n' + text:
                remove_staging(stage)
                raise NoInput('C-EXCLUDED: %s %s carries verdict exclude in data_decisions (the engine refused it, exit %d)'
                              % (cell, band, res.rc))
            done = [ln for ln in text.splitlines() if cc.REGEN_DONE_SUBBAND in ln]
            if res.rc != 0 or not done:
                raise RuntimeError('phase3_snapshot_search.py --cutouts-from-cubes exit %d without a %r line (staging kept: '
                                   '%s): %s' % (res.rc, cc.REGEN_DONE_SUBBAND.strip(), stage, res.tail[-600:]))
            counts, refused = _regen_counts(done[-1])
            outcome = {S: counts.get(S) for S in 'VI'}
            if refused:
                u.note('the search refused (night rule): %s' % refused)
        else:
            for S in 'VI':                       # V then I: the I run reuses the member admissions the V run wrote
                argv = ['--lst', lst, '--date', date, '--stokes', S, '--out', eout, '--cutouts-from-cubes', band]
                res, text = K.call(u, 'cutouts_wideband', argv, 'snapshot', 'cutouts_wideband', run,
                                   'cutouts_%s_%s_%s.log' % (cell, band, S), env_extra=env, threads=args.threads)
                if res.rc == cc.REGEN_EXCLUDED_RC:
                    outcome[S] = 'excluded'
                    continue
                done = [ln for ln in text.splitlines() if cc.REGEN_DONE_WIDEBAND in ln]
                if res.rc != 0 or not done:
                    raise RuntimeError('phase3_snapshot_group_search.py --cutouts-from-cubes %s (%s) exit %d without a %r '
                                       'line (staging kept: %s): %s' % (band, S, res.rc, cc.REGEN_DONE_WIDEBAND.strip(), stage,
                                                                        res.tail[-600:]))
                m = re.search(r': (\S+) cutout cubes', done[-1])
                outcome[S] = m.group(1) if m else done[-1].split(': ', 1)[-1].strip()
                mp = os.path.join(eout, 'wideband_members_%s.csv' % S)
                if os.path.exists(mp):
                    members[S] = mp
            if all(outcome.get(S) == 'excluded' for S in 'VI'):
                remove_staging(stage)
                raise NoInput('C-EXCLUDED: %s %s is excluded under the deep rule (the group search refused it, exit 3)'
                              % (cell, band))
        u.meta['engine_outcome'] = outcome
        rels, other = staged_cutouts(stage, samples)
        if other:
            u.refuse('C-NAMES: the engine wrote files that are no cutout of samples %s: %s (staging kept: %s)' % (
                ','.join(samples), TC._fmt(other), stage))
        if not rels:
            remove_staging(stage)
            raise NoInput('C-NOCUT: the engine wrote no cutout for %s %s (outcome %s)' % (cell, band, outcome))
        bad = [r for r in rels if TC.parse_cutout(os.path.join(stage, r))[1] != band
               or TC.parse_cutout(os.path.join(stage, r))[3]['lst'] != lst]
        if bad:
            u.refuse('C-NAMES: staged cutouts outside %s %s: %s (staging kept: %s)' % (cell, band, TC._fmt(bad), stage))
        # 1. identity of every staged cutout with its cube(s), per Stokes (the regeneration's content test)
        for S in 'VI':
            files = [os.path.join(stage, r) for r in rels if r.endswith('_%s.fits' % S)]
            if not files:
                continue
            if is_wideband(band):
                if S not in members:
                    u.refuse('C-IDENTITY: no wideband_members_%s.csv from the group search for %s %s (staging kept: %s)'
                             % (S, cell, band, stage))
                res_ = TC.identity_wideband(files, cell, band, S, members[S])
            else:
                res_ = TC.identity_subband(files, cell, band, S)
            nbad = [(os.path.relpath(f, stage), d) for f, ok, _, d in res_ if not ok]
            worst = max([v for _, _, v, _ in res_ if v == v] or [float('nan')])
            u.check('photometry_cutout_manifest.I1_identity_%s' % S, not nbad, worst,
                    '%d of %d cutouts equal their %s within %g x rms (worst %.2e)%s' % (
                        len(files) - len(nbad), len(files), 'rebuilt wideband stack' if is_wideband(band) else 'cube',
                        cc.IDENTITY_TOL, worst, ('; FAIL %s' % TC._fmt(nbad)) if nbad else ''), cls='photometry_cutout_manifest')
        if u.failed_tests():
            u.refuse('C-IDENTITY: %s (staging kept: %s)' % ('; '.join(t['detail'] for t in u.failed_tests())[:600], stage))
        # 2-3. one file per key, placement, the manifest
        mdir = TC.manifest_dir(root, '10s')
        mpath = os.path.join(mdir, '%s_%s%s.csv' % (cell, band, sfx))
        place_unit(u, run, root, stage, rels, '10s', cell, band, samples, mpath, {'cell': cell, 'band': band})
        for S, mp in sorted(members.items()):      # beside the manifest: the member tables a wideband is rebuilt from
            dst = os.path.join(mdir, '%s_%s%s.members_%s.csv' % (cell, band, sfx, S))
            K.atomic_copy(mp, dst)
            u.meta.setdefault('members', {})[S] = dst
        remove_staging(stage)
        u.meta['staging'] = None
    return run.units[-1:]


# ---------------------------------------------------------------------------------------------------------------------
# deep and 10-minute tiers: photometry_tier_cutouts_20261025.py, photometry_tier_forced_20261026.py
# ---------------------------------------------------------------------------------------------------------------------

def tier_universe(tier):
    """The tool's own cell list (photometry_tier_cutouts_20261025.cells :233-237): deep = the run's iresid / vresid cell
    directories, 10min = the cells with a frames.csv under the 10-minute root."""
    cc = _cc()
    if tier == 'deep':
        return sorted({os.path.basename(d) for s in ('iresid', 'vresid')
                       for d in glob.glob(os.path.join(cc.TIER_DEEP_RUN, s, '*'))})
    return sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(cc.TIER_TENMIN_ROOT, '*', 'frames.csv')))


def gate_tier_run(args):
    """C-RUN: the tier tool reads the deep run it names by literal path (photometry_cutouts.TIER_DEEP_RUN); a --src-run /
    --run naming another run refuses (the tool cannot be pointed there in the window)."""
    from exopipe.phase3.lib.photometry import steps_deep as KD
    if not (getattr(args, 'src_run', None) or getattr(args, 'run_label', None)):
        return
    s = KD.src_run(args)
    if os.path.realpath(s) != os.path.realpath(_cc().TIER_DEEP_RUN):
        raise Refusal('C-RUN: --src-run / --run %s is not the deep run the tier tool reads (%s; '
                      'photometry_tier_cutouts_20261025.py:46 DEEP)' % (s, _cc().TIER_DEEP_RUN))


def tier_items(args, tier):
    spec = args.cells
    if not spec:
        raise Refusal('C-CELLS: a tier cutout run names its <LST>_<band> cells (--cells; patterns over the tool\'s cells, '
                      'e.g. --cells "08h_*")')
    universe = tier_universe(tier)
    cells = C.expand(spec, universe=universe, kind='cell')
    for cl in cells:
        C.parse_cell(cl)
    if args.scopes:
        sc = {s.strip() for s in args.scopes.split(',') if s.strip()}
        cells = [cl for cl in cells if cl.split('_', 1)[1] in sc]
    if args.exclude_cells:
        ex = set(C.expand(args.exclude_cells, universe=universe, kind='cell'))
        cells = [cl for cl in cells if cl not in ex]
    return sorted(cells)


def _master_path(lst, band, stokes):
    """The REFERENCE master the tool cuts (p3s2_frames.master_path :36-43 through the roots table cell_root :25-33,
    photometry_frames.ROOTS_CSV), written out; None when absent."""
    root = None
    for line in open(K.cfg().photometry_frames.ROOTS_CSV):
        q = line.strip().split(',')
        if len(q) >= 3 and q[0] == '%s_%s' % (lst, band):
            root = q[2]
    if not root:
        return None
    fn = 'I_deep_Taper_R075_%s.fits' % band if stokes == 'I' else 'V_deep_Taper_R0_%s.fits' % band
    p = '%s/%s_%s/coadd_final/%s' % (root, lst, band, fn)
    return p if os.path.exists(p) else None


def tier_inputs(tier, cell):
    """The files a tier cutout unit reads (the tool's planes_deep / planes_10min :75-96 + the REFERENCE masters + the deep
    refusal record :57-72): its resume identity."""
    cc = _cc()
    lst, band = cell.split('_', 1)
    out = []
    if tier == 'deep':
        for S, sub in (('I', 'iresid'), ('V', 'vresid')):
            out += glob.glob(os.path.join(cc.TIER_DEEP_RUN, sub, cell, '*_%s_redo_resid_%s.fits' % (S, band)))
        out += glob.glob(os.path.join(cc.TIER_DEEP_RUN, 'refusals', '%s_*_excluded.csv' % lst))
    else:
        fr = os.path.join(cc.TIER_TENMIN_ROOT, cell, 'frames.csv')
        if os.path.exists(fr):
            out.append(fr)
            with open(fr, newline='') as fh:
                out += [r['sub_path'] for r in csv.DictReader(fh) if os.path.exists(r['sub_path'])]
        for S in 'IV':
            vi = os.path.join(cc.TIER_TENMIN_SEARCH, '%s_%s_%s' % (lst, band, S), 'transients', 'Run_SLST%sB' % lst[:-1],
                              'v_deep_images.csv')
            if os.path.exists(vi):
                out.append(vi)
    out += [p for p in (_master_path(lst, band, S) for S in 'IV') if p]
    return sorted(set(out))


def stage_tier_cutouts(run, args):
    tier = args.tier
    token = _cc().TIER_TOKEN[tier]
    targets = K.adhoc_targets(args)
    root = K.cutout_root(args)
    planets = bool(getattr(args, 'planets', False))
    samples = run_samples(targets, planets)
    try:
        gate_tier_run(args)
        cells = tier_items(args, tier)
        K.gate_write_root(root, '--cutout-root')
        K.gate_adhoc_roots(targets, {'cutout root': root})
    except Refusal as e:
        if run.dry_run:
            K.would_refuse(run, 'cutouts', str(e))
            return []
        raise
    K.plan(run, '[cutouts] tier %s: %d <LST>_<band> units -> %s; samples %s%s (photometry_tier_cutouts_20261025.py in place, '
                'site PYTHON[%s])' % (token, len(cells), root, ','.join(samples),
                                      ' (ad hoc: %s)' % ', '.join('%s=%s' % t for t in targets) if targets else '',
                                      K.python_key(tier, 'cutouts_tier')))
    units = []
    for cell in cells:
        if run.dry_run:
            K.plan(run, '[cutouts]   unit cutouts.%s.%s%s' % (token, cell, tag_suffix(targets)))
            continue
        units += tier_unit(run, args, tier, cell, root, targets, planets, samples)
    return units


def tier_unit(run, args, tier, cell, root, targets, planets, samples):
    from exopipe.phase3.lib.photometry import tests_cutouts as TC
    token = _cc().TIER_TOKEN[tier]
    lst, band = cell.split('_', 1)
    sfx = tag_suffix(targets)
    name = 'cutouts.%s.%s%s' % (token, cell, sfx)
    inputs = tier_inputs(tier, cell) + [p for _, p in targets] + ([] if targets else list(K.sample_paths().values()))
    if K.run_resume_ok(run, name, ['%s:%s' % (token, cell)], {'inputs_listing_sha1': listing_sha1(inputs), 'cutout_root': root}):
        return run.units[-1:]
    with run.unit(name, tier=tier, stage='cutouts', cell=cell, cutout_root=root, samples=samples,
                  targets=['%s=%s' % t for t in targets], planets=planets, inputs_listing_sha1=listing_sha1(inputs)) as u:
        for p in inputs:
            if not p.endswith('.fits') or os.path.getsize(p) <= 16 * 2 ** 20:
                u.add_input(p)
        u.meta['n_input_frames'] = sum(1 for p in inputs if p.endswith('.fits'))
        u.set_admitted(['%s:%s' % (token, cell)])
        stage = staging_dir(root, run, '%s_%s%s' % (token, cell, sfx))
        u.meta['staging'] = stage
        argv = ['--root', stage, '--tier', token, '--cells', cell, '--workers', '1']
        res, text = K.call(u, 'cutouts_tier', argv, tier, 'cutouts_tier', run, 'cutouts_%s_%s.log' % (token, cell),
                           env_extra=K.adhoc_env(targets, planets), threads=args.threads)
        lines = [ln for ln in text.splitlines() if ln.startswith('%s %s ' % (token, cell))]
        if res.rc != 0 or 'PHOTOMETRY-TIER-DONE' not in text:
            raise RuntimeError('photometry_tier_cutouts_20261025.py exit %d without PHOTOMETRY-TIER-DONE (staging kept: %s): %s'
                               % (res.rc, stage, res.tail[-600:]))
        u.meta['engine_outcome'] = lines
        rels, other = staged_cutouts(stage, samples)
        if other:
            u.refuse('C-NAMES: the tool wrote files that are no cutout of samples %s: %s (staging kept: %s)' % (
                ','.join(samples), TC._fmt(other), stage))
        if not rels:
            remove_staging(stage)
            raise NoInput('C-NOCUT: no tier cutout for %s %s (%s)' % (token, cell, '; '.join(lines) or 'no line'))
        bad = [r for r in rels if TC.parse_cutout(os.path.join(stage, r))[1] != band
               or TC.parse_cutout(os.path.join(stage, r))[3]['lst'] != lst or TC.parse_cutout(os.path.join(stage, r))[4] != token]
        if bad:
            u.refuse('C-NAMES: staged cutouts outside %s %s: %s (staging kept: %s)' % (token, cell, TC._fmt(bad), stage))
        for S in 'IV':
            files = [os.path.join(stage, r) for r in rels if r.endswith('_%s.fits' % S)]
            if not files:
                continue
            res_ = TC.tier_recut(files, stage)
            nbad = [(os.path.relpath(f, stage), d) for f, ok, _, d in res_ if not ok]
            u.check('photometry_cutout_manifest.I1_recut_%s' % S, not nbad, len(files) - len(nbad),
                    '%d of %d cutouts: one plane each re-cut from its source frame equals the file%s' % (
                        len(files) - len(nbad), len(files), ('; FAIL %s' % TC._fmt(nbad)) if nbad else ''),
                    cls='photometry_cutout_manifest')
        if u.failed_tests():
            u.refuse('C-IDENTITY: %s (staging kept: %s)' % ('; '.join(t['detail'] for t in u.failed_tests())[:600], stage))
        mpath = os.path.join(TC.manifest_dir(root, token), '%s%s.csv' % (cell, sfx))
        place_unit(u, run, root, stage, rels, token, lst, band, samples, mpath, {'tier': token, 'band': band})
        remove_staging(stage)
        u.meta['staging'] = None
    return run.units[-1:]


def tier_out_roots(args, samples):
    """{root: [samples]} of the cutmeasure outputs: --tier-out for every sample, else photometry.TIER_FORCED_ROOTS."""
    ph = K.cfg().photometry
    out = {}
    for s in samples:
        r = os.path.abspath(args.tier_out) if getattr(args, 'tier_out', None) else ph.TIER_FORCED_ROOTS[s]
        out.setdefault(r, []).append(s)
    return out


def stage_tier_cutmeasure(run, args):
    tier = args.tier
    ph = K.cfg().photometry
    token = _cc().TIER_TOKEN[tier]
    targets = K.adhoc_targets(args)
    root = K.cutout_root(args)
    try:
        if targets and any(t not in ph.TIER_FORCED_SAMPLES for t, _ in targets):
            raise Refusal('PHOT-ADHOC: cutmeasure measures the samples %s only (photometry_tier_forced_20261026.py :54 '
                          'SAMPLES, its --samples refuses others); --targets names %s' % (
                              ','.join(ph.TIER_FORCED_SAMPLES), ','.join(t for t, _ in targets)))
        if targets:
            samples = [t for t, _ in targets] + ([ph.PLANET_SAMPLE] if getattr(args, 'planets', False) else [])
        else:
            samples = list(ph.TIER_FORCED_SAMPLES)
        roots = tier_out_roots(args, samples)
        for r in roots:
            K.gate_write_root(r, 'the tier forced root')
        K.gate_adhoc_roots(targets, {'tier forced root': r for r in roots})
        cells = tier_items(args, tier)
    except Refusal as e:
        if run.dry_run:
            K.would_refuse(run, 'cutmeasure', str(e))
            return []
        raise
    lsts = sorted({cl.split('_', 1)[0] for cl in cells})
    K.plan(run, '[cutmeasure] tier %s: LSTs %s on the cutouts of %s -> %s (photometry_tier_forced_20261026.py in place, '
                'site PYTHON[%s])' % (token, ','.join(lsts), root, '; '.join('%s: %s' % (','.join(s), r)
                                                                             for r, s in sorted(roots.items())),
                                      K.python_key(tier, 'cutmeasure_tier')))
    units = []
    for lst in lsts:
        for out_root, smp in sorted(roots.items()):
            if run.dry_run:
                K.plan(run, '[cutmeasure]   unit cutmeasure.%s.%s.%s -> %s' % (token, lst, '+'.join(smp), out_root))
                continue
            units += cutmeasure_unit(run, args, tier, lst, root, out_root, smp)
    return units


def cutmeasure_unit(run, args, tier, lst, root, out_root, samples):
    from exopipe.phase3.lib.photometry import tests_cutouts as TC
    ph = K.cfg().photometry
    token = _cc().TIER_TOKEN[tier]
    name = 'cutmeasure.%s.%s.%s' % (token, lst, '+'.join(samples))
    files = sorted(p for s in samples for p in glob.glob(os.path.join(root, s, '*', '*', '*_%s_%s_[IV].fits' % (lst, token))))
    lsha = listing_sha1(files)
    if K.run_resume_ok(run, name, ['%s:%s' % (token, lst)], {'cutout_listing_sha1': lsha, 'tier_out': out_root}):
        return run.units[-1:]
    with run.unit(name, tier=tier, stage='cutmeasure', lst=lst, cutout_root=root, tier_out=out_root, samples=samples,
                  cutout_listing_sha1=lsha, n_cutouts=len(files)) as u:
        if not files:
            raise NoInput('C-NOCUT: no %s cutout of %s at %s under %s' % (token, '/'.join(samples), lst, root))
        u.set_admitted(['%s:%s' % (token, lst)])
        mc = K.cfg().catalog_snapshot.MASTER_CATALOGS
        for b in sorted({os.path.abspath(p).split('/')[-3] for p in files}):
            gp = os.path.join(mc, 'master_R075_%s_%s.gaul.csv' % (lst, b))
            if os.path.exists(gp):
                u.add_input(gp)
        stage = staging_dir(out_root, run, '%s_%s_%s' % (token, lst, '+'.join(samples)))
        u.meta['staging'] = stage
        odir = os.path.join(out_root, token, lst)
        sdir = os.path.join(stage, token, lst)
        os.makedirs(sdir)
        old_man = os.path.join(odir, 'forced_manifest.csv')
        partial = set(samples) != set(ph.TIER_FORCED_SAMPLES)
        if partial and os.path.exists(old_man):     # the tool merges a sample subset with the manifest in --out (:186-192)
            shutil.copy2(old_man, os.path.join(sdir, 'forced_manifest.csv'))
        argv = ['--root', root, '--out', stage, '--tier', token, '--lsts', lst, '--workers', '1', '--samples', ','.join(samples)]
        res, text = K.call(u, 'cutmeasure_tier', argv, tier, 'cutmeasure_tier', run, 'cutmeasure_%s_%s.log' % (token, lst),
                           threads=args.threads)
        m = re.search(r'PHOTOMETRY-TIER-FORCED-DONE (\d+) cutouts measured, (\d+) refused', text)
        if res.rc != 0 or not m:
            raise RuntimeError('photometry_tier_forced_20261026.py exit %d without PHOTOMETRY-TIER-FORCED-DONE (staging kept: '
                               '%s): %s' % (res.rc, stage, res.tail[-600:]))
        u.meta.update(n_measured=int(m.group(1)), n_refused=int(m.group(2)))
        npz = sorted(glob.glob(os.path.join(sdir, 'forced_*.npz')))
        mine = [p for p in npz if TC._NPZ_RE.match(os.path.basename(p))
                and TC._NPZ_RE.match(os.path.basename(p))['smp'] in samples]
        if int(m.group(2)):
            u.refuse('CM-REFUSED: the tool refused %s cutouts of %s %s (staging kept: %s)' % (m.group(2), token, lst, stage))
        for p in mine:
            u.plan(os.path.join(odir, os.path.basename(p)), 'photometry_tier_npz')
        u.plan(old_man, 'photometry_tier_manifest')
        os.makedirs(odir, exist_ok=True)
        for p in mine:
            os.replace(p, os.path.join(odir, os.path.basename(p)))
        os.replace(os.path.join(sdir, 'forced_manifest.csv'), old_man)
        for p in mine:
            u.add_output(os.path.join(odir, os.path.basename(p)), 'photometry_tier_npz', cutout_root=root)
        u.add_output(old_man, 'photometry_tier_manifest')
        remove_staging(stage)
        u.meta['staging'] = None
    return run.units[-1:]


# ---------------------------------------------------------------------------------------------------------------------
# the sub-LST tier (cutouts and cutmeasure only)
# ---------------------------------------------------------------------------------------------------------------------

STAGE_FUNCS = {'cutouts': stage_tier_cutouts, 'cutmeasure': stage_tier_cutmeasure}


def run_tier(run, args, stages):
    args.photometry_stages = list(stages)
    for st in stages:
        units = STAGE_FUNCS[st](run, args)
        if run.dry_run:
            continue
        if units and not K.stage_ok(units):
            K.say('photometry %s: stage %s did not complete (%s) - later stages not run'
                  % (args.tier, st, ', '.join('%s %s' % (u.name, u.status) for u in units
                                              if u.status not in ('done', 'skipped-valid'))))
            break
    return None
