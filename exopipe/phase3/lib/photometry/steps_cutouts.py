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
     <root>/quarantine_superseded_cutouts_<run id>/ with a sha256 manifest, never deleted. A unit that writes nothing for its
     (cell, band) (C-EXCLUDED, C-NOCUT) treats every such file of the root the same way before it ends with no input;
  3. the unit's manifest _manifest/<10s|deep|10min>/<key>_<band>[.<tags>].csv (file, sha256, bytes, sample, object, stokes)
     is written in the staging directory and its class tests run on the STAGED files (photometry_cutout_manifest CM1 files +
     structure + sha256 sample, CM2 one file per key): a failure refuses before anything reaches the root (C-CONTENT);
  4. placement, one transaction: every file of the root the unit replaces (same name) - and with --fresh every stale /
     orphan file - moves to the quarantine directory first (sha256 manifest), then the staged files are renamed in (os.replace,
     the manifest last) and the class tests run again on the placed manifest. Any failure from the first move on - an
     exception, a failed test, Ctrl-C or SIGTERM (cli.Terminated) - moves the placed files back to the staging directory and
     the quarantined ones back in place: a refused or interrupted unit leaves the root's files as they were (review F2 and
     fix verification N2, 2026-11-03).
cutmeasure stages the tool's --out (the root's existing forced_manifest.csv copied in first, so a sample subset merges as the
tool merges it), runs TN1 / TN2 (photometry_tier_npz) and TM1 / TM2 (photometry_tier_manifest) on the staged npz and manifest,
then places them the same way (the npz and manifest it replaces quarantined under <root>/quarantine_superseded_tier_forced_
<run id>/).
An engine REFUSAL refuses its unit (C-ENGINE) and an error is recorded as that unit's: the stage goes on with the next unit
(review F3; production's drivers marked the job FAILED and went on).

Gates: C-ROOT (gate_write_root: a retired generation never; a production root while paths.PRODUCTION_WRITES is False), the
INJECTED-RUN GUARD on every root a stage writes (core/injected.check), C-INPLACE (the served roots only with --in-place, the
measurement chained, snapshot over whole cells; gate_in_place), C-CELLS (a regeneration names its items; a named tier cell
the tool does not know), C-HOST (the tier tool's inputs not readable on this host), C-RUN (the tier tool reads one deep run),
PHOT-ADHOC (an ad-hoc target set into or under a served root, or for cutmeasure / evaluate), PHOT-FULL (snapshot Full).
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


def quarantine(run, root, rels, reason, prefix=None, moved=None):
    """Move files out of the root, never delete them (CLAUDE.md 7): to <root>/<prefix><run id>/<relative path> (prefix
    photometry_cutouts.QUARANTINE_PREFIX unless given), each with a row of MANIFEST_sha256.csv (sha256,bytes,moved_from,
    moved_to,reason), its checksum verified after the move. Each move is appended to ``moved`` as (relative path, quarantine
    path, sha256, bytes) the moment it is made, so a caller that fails part-way can put back every file moved (restore).
    Returns ``moved``."""
    from exopipe.phase3.lib.photometry import tests_cutouts as TC
    moved = [] if moved is None else moved
    if not rels:
        return moved
    q = os.path.join(root, (prefix or _cc().QUARANTINE_PREFIX) + run.run_id)
    man = os.path.join(q, 'MANIFEST_sha256.csv')
    os.makedirs(q, exist_ok=True)
    new = not os.path.exists(man)
    with open(man, 'a') as mf:
        if new:
            mf.write('sha256,bytes,moved_from,moved_to,reason\n')
        for rel in sorted(rels):
            src, dst = os.path.join(root, rel), os.path.join(q, rel)
            h = TC.sha256_file(src)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(src, dst)
            moved.append((rel, dst, h, os.path.getsize(dst)))
            mf.write('%s,%d,%s,%s,%s\n' % (h, os.path.getsize(dst), src, dst, reason.replace(',', ';')))
            mf.flush()
            if TC.sha256_file(dst) != h:
                raise Refusal('C-STALE: quarantine checksum of %s differs after the move' % dst)
    return moved


def restore(run, root, moved, why, prefix=None):
    """Undo quarantine(): every (relative path, quarantine path, sha256, bytes) of ``moved`` back to <root>/<relative path>,
    the last moved first, each with a 'restored' row in the quarantine manifest. Returns [problem] (a file that could not go
    back - its place taken, or the move failed - named; the caller records them)."""
    problems, rows = [], []
    for rel, qp, h, n in reversed(moved):
        dst = os.path.join(root, rel)
        try:
            if os.path.lexists(dst):
                raise OSError('its place %s is taken' % dst)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(qp, dst)
            rows.append('%s,%d,%s,%s,%s\n' % (h, n, qp, dst, ('restored: ' + why).replace(',', ';').replace('\n', ' ')))
        except OSError as e:
            problems.append('%s not restored from %s: %s' % (rel, qp, e))
    if rows:
        man = os.path.join(root, (prefix or _cc().QUARANTINE_PREFIX) + run.run_id, 'MANIFEST_sha256.csv')
        with open(man, 'a') as mf:
            mf.writelines(rows)
    return problems


def _makedirs_tracked(d, created):
    """os.makedirs(d), appending each directory it creates (outermost first) to ``created`` (unplace removes them)."""
    miss, p = [], d
    while p and not os.path.isdir(p):
        miss.append(p)
        p = os.path.dirname(p)
    os.makedirs(d, exist_ok=True)
    created.extend(reversed(miss))


def place(root, pairs, placed, created):
    """Rename each (staged path, relative path) of ``pairs`` to <root>/<relative path>, appending it to ``placed`` the moment it
    is made (unplace undoes it). A destination that exists refuses: the caller moved every file it replaces to the quarantine
    first, so one here was written by someone else meanwhile."""
    for src, rel in pairs:
        dst = os.path.join(root, rel)
        _makedirs_tracked(os.path.dirname(dst), created)
        if os.path.lexists(dst):
            raise Refusal('C-PLACE: %s appeared in the root while the unit placed its files (another writer?) - not '
                          'replaced' % dst)
        os.replace(src, dst)
        placed.append((src, rel))


def unplace(root, placed, created):
    """Undo place(): every placed file back to its staged path (the last placed first), then the directories placement
    created, if empty. Returns [problem]."""
    problems = []
    for src, rel in reversed(placed):
        try:
            os.makedirs(os.path.dirname(src), exist_ok=True)
            os.replace(os.path.join(root, rel), src)
        except OSError as e:
            problems.append('%s not moved back to %s: %s' % (rel, src, e))
    for d in reversed(created):
        try:
            os.rmdir(d)
        except OSError:
            pass
    return problems


def placement(u, run, root, pairs, outputs, reason, prefix=None, aside=(), aside_reason=''):
    """Step 4 of the module docstring, one transaction: ``aside`` (relative paths: stale / orphan files, --fresh) and every
    file of ``pairs`` (staged path, relative path) that exists in the root move to the quarantine, the staged files are renamed
    in, then ``outputs`` (a callable) records the outputs and runs their class tests on the placed files. A failure from the
    first move on - an exception, a failed test, Ctrl-C or SIGTERM (cli.Terminated, a KeyboardInterrupt) - undoes every move
    (unplace, restore), so the root's files are as they were, and the unit is refused (C-PLACE; an exception or interrupt is
    re-raised). A SIGKILL cannot be undone. Returns the moves to the quarantine."""
    moved, placed, created = [], [], []
    n_out = len(u.outputs)
    try:
        quarantine(run, root, aside, aside_reason, prefix, moved)
        replaced = [rel for _, rel in pairs if os.path.lexists(os.path.join(root, rel))]
        quarantine(run, root, replaced, reason, prefix, moved)
        place(root, pairs, placed, created)
        outputs()
        bad = u.failed_tests()
        if bad:
            raise Refusal('C-PLACE: the placed files fail their content test: %s' % '; '.join(
                '%s (%s)' % (t['name'], str(t.get('detail') or t.get('value'))[:200]) for t in bad[:3]))
    except BaseException as e:
        probs = unplace(root, placed, created)
        probs += restore(run, root, moved, '%s did not complete (%s: %s)' % (u.name, type(e).__name__, str(e)[:200]), prefix)
        del u.outputs[n_out:]
        u.meta['placement_undone'] = dict(placed=len(placed), restored=len(moved), problems=probs)
        if probs:
            u.note('PLACEMENT NOT FULLY UNDONE (%d problem(s)): %s' % (len(probs), '; '.join(probs[:5])))
        if isinstance(e, Refusal):
            u.refuse('%s - every move undone (%d placed file(s) back in the staging directory, %d quarantined back in place%s'
                     ')' % (e, len(placed), len(moved), '; %d NOT undone, see the notes' % len(probs) if probs else ''))
        raise
    return moved


def staged_tests(u, cls, path, **ctx):
    """The class tests of ``cls`` on a STAGED product, recorded as unit checks '<cls>.<test>_staged' (review F2: they decide
    before anything reaches the root). Returns the failed ones."""
    from exopipe.phase3.core import validate
    bad = []
    for r in validate.run(cls, path, **ctx):
        u.check(r['name'] + '_staged', r['passed'], r['value'], r['detail'], cls=cls, path=path)
        if not r['passed']:
            bad.append('%s: %s' % (r['name'], str(r['detail'])[:300]))
    return bad


def place_unit(u, run, root, stage, rels, token, key, band, samples, mrel, ctx, extra=()):
    """Steps 2 to 4 of the module docstring for the cutouts of one unit (``rels``, relative to the staging directory and the
    root alike): one file per key (C-STALE: refuse, or quarantine with --fresh), the unit's manifest written in the staging
    directory (<stage>/<mrel>) and its class tests run on the staged files (C-CONTENT), then the placement transaction with the
    class tests on the placed manifest. ``extra``: more (staged path, relative path) pairs placed with the cutouts (the wideband
    member tables beside the manifest). Returns the placed manifest's path."""
    from exopipe.phase3.lib.photometry import tests_cutouts as TC
    cls = 'photometry_cutout_manifest'
    stale, orphan = TC.superseded(root, token, key, band, samples, rels)
    u.meta.update(n_stale=len(stale), n_orphan=len(orphan))
    what = ''
    if stale or orphan:
        what = '%d other time stamp(s) of a written key (%s) and %d file(s) of the unit\'s cell and samples it did not write ' \
               '(%s)' % (len(stale), TC._fmt(stale), len(orphan), TC._fmt(orphan))
        if not run.fresh:
            u.refuse('C-STALE: %s in %s - one file per (sample, band, object, LST, date, Stokes); rerun with --fresh to move '
                     'them to a quarantine directory (staged files kept: %s)' % (what, root, stage))
    rows = []
    for rel in rels:
        p = TC.parse_cutout(os.path.join(stage, rel))
        sp = os.path.join(stage, rel)
        rows.append(dict(file=rel, sha256=TC.sha256_file(sp), bytes=os.path.getsize(sp), sample=p[0], object=p[2],
                         stokes=p[3]['st']))
    staged_man = os.path.join(stage, mrel)
    TC.write_manifest(staged_man, rows)
    bad = staged_tests(u, cls, staged_man, root=stage, samples=samples, **ctx)
    if bad:
        u.refuse('C-CONTENT: the staged cutouts fail their content test before placement - nothing placed (staging kept: '
                 '%s): %s' % (stage, '; '.join(bad)[:900]))
    manifest_path = os.path.join(root, mrel)
    u.plan(manifest_path, cls)
    pairs = [(os.path.join(stage, r), r) for r in rels] + list(extra) + [(staged_man, mrel)]
    moved = placement(u, run, root, pairs, lambda: u.add_output(manifest_path, cls, root=root, samples=samples, **ctx),
                      'replaced by %s (%s)' % (u.name, run.run_id), aside=stale + orphan,
                      aside_reason='superseded by %s (%s)' % (u.name, run.run_id))
    aside = set(stale + orphan)
    u.meta['quarantined'] = [q for rel, q, _, _ in moved if rel in aside]
    u.meta['n_replaced_quarantined'] = sum(1 for rel, _, _, _ in moved if rel not in aside)
    if aside:
        u.note('C-STALE --fresh: %s moved to quarantine' % what)
    u.meta['n_cutouts'] = len(rows)
    return manifest_path


def unit_manifest_rels(token, key, sfx):
    """The paths, relative to the root, of a unit's manifest and its wideband member tables (_manifest/<token>/<key><sfx>.csv,
    .members_<S>.csv; key = <cell>_<band> for 10 s, <LST>_<band> for the tiers)."""
    d = os.path.join(_cc().MANIFEST_SUBDIR, token)
    return [os.path.join(d, '%s%s.csv' % (key, sfx))] + [os.path.join(d, '%s%s.members_%s.csv' % (key, sfx, S)) for S in 'VI']


def nothing_written(u, run, root, token, key, band, samples, why, mrels=()):
    """A unit that writes no cutout for its (cell, band) (C-EXCLUDED, C-NOCUT): the files of that (cell, band) and samples
    already in the root - and the unit's earlier manifest (``mrels``, relative paths), which lists them - are not the current
    regeneration's (review F8: an exclusion added since, a night rule that now refuses both Stokes). One file per key holds
    only with none left: refused (C-STALE), or with --fresh moved to the quarantine. Returns normally when none is left (the
    caller then ends the unit with no input)."""
    from exopipe.phase3.lib.photometry import tests_cutouts as TC
    stale, orphan = TC.superseded(root, token, key, band, samples, [])
    left = stale + orphan + [r for r in mrels if os.path.lexists(os.path.join(root, r))]
    u.meta.update(n_stale=0, n_orphan=len(stale + orphan))
    if not left:
        return
    if not run.fresh:
        u.refuse('C-STALE: %s, but %s holds %d file(s) of it (%s) - they are not this regeneration\'s (one file per key); '
                 'rerun with --fresh to move them to a quarantine directory' % (why, root, len(left), TC._fmt(left)))
    moved = []
    try:
        quarantine(run, root, left, 'superseded: %s writes nothing (%s)' % (u.name, why), moved=moved)
    except BaseException as e:          # part-way: every file moved goes back, the root as it was
        probs = restore(run, root, moved, '%s did not complete (%s: %s)' % (u.name, type(e).__name__, str(e)[:200]))
        if probs:
            u.note('QUARANTINE NOT FULLY UNDONE (%d problem(s)): %s' % (len(probs), '; '.join(probs[:5])))
        raise
    u.meta['quarantined'] = [q for _, q, _, _ in moved]
    u.note('C-STALE --fresh: %d file(s) of %s moved to quarantine - %s' % (len(left), u.name, why))


def run_unit(run, name, meta, fn, *a):
    """Run one unit (``fn(*a)``). An error it raises - an engine crash, a bug - is that unit's: recorded as error by the unit's
    own __exit__ (or by a unit opened here when it came before the unit started), and the stage goes on with the next unit, as
    production's drivers marked such a job FAILED and went on (review F3); the run exits 1 at its end. Returns the units the
    call added."""
    n0 = len(run.units)
    try:
        fn(*a)
    except Exception:           # noqa: BLE001 - recorded as the unit's error (status error, the traceback in its notes)
        if not any(u.name == name for u in run.units[n0:]):
            try:
                with run.unit(name, **meta):
                    raise
            except Exception:   # noqa: BLE001 - recorded (status error) by the unit's __exit__
                pass
    return run.units[n0:]


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
        gate_write_roots(run, args, 'cutouts', [root], targets, {'cutout root': root}, cells=sorted({cl for cl, _ in items}))
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
        units += run_unit(run, 'cutouts.%s.%s%s' % (cell, band, tag_suffix(targets)),
                          dict(tier='snapshot', stage='cutouts', cell=cell, band=band, cutout_root=root),
                          snapshot_unit, run, args, cell, band, root, targets, planets, samples, cubes)
    return units


def gate_write_roots(run, args, stage, roots, targets, adhoc_roots, cells=None):
    """Every gate on the roots a cutouts / cutmeasure stage writes, before its first unit: the INJECTED-RUN GUARD (core/
    injected.check: an injected run writes scratch only - review F6, whatever flag or default named the root), C-ROOT (a
    retired generation; a production root in the refactor window), PHOT-ADHOC (an ad-hoc set into or under a served root) and
    C-INPLACE (the served roots only with --in-place; gate_in_place). ``cells``: the night cells a snapshot cutouts stage
    writes (gate_in_place_whole_cells)."""
    from exopipe.phase3.core import injected
    what = {'cutouts': 'the cutout root (--cutout-root)', 'cutmeasure': 'the tier forced root (--tier-out)'}[stage]
    for r in roots:
        injected.check(r, 'photometry %s (%s)' % (stage, what))
    for r in roots:
        K.gate_write_root(r, what)
    K.gate_adhoc_roots(targets, adhoc_roots)
    gate_in_place(run, args, stage, roots, cells)


def gate_in_place(run, args, stage, roots, cells=None):
    """C-INPLACE (review F4; production's rule, phase3_snapshot_search.regen_cutouts: P3_SNAP_CUTOUT_ROOT is 'a new root -
    never over the existing product'; generation 5 was built beside generation 4 and the pages switched after). A cutouts /
    cutmeasure stage writes a SERVED root (photometry.CUTOUT_ROOT; photometry.TIER_FORCED_ROOTS) only with --in-place, and then
    the measurement of the new cutouts runs in the same invocation: snapshot cutouts -> the measure stage into the served
    stage 1 (--out = photometry.OUT_ROOT[snapshot]), over whole cells (gate_in_place_whole_cells); deep / sublst cutouts ->
    cutmeasure into the served tier forced roots (no --tier-out; it re-measures every band of each LST found in the root);
    cutmeasure into the served tier forced roots measures the served cutouts (no other --cutout-root). A root INSIDE a served
    root is no generation root at all (refused); --in-place with no served root to write refuses."""
    ph = K.cfg().photometry
    in_place = bool(getattr(args, 'in_place', False))
    stages = list(getattr(args, 'photometry_stages', ()) or ())
    real = os.path.realpath
    served = []
    for r in roots:
        hit = K.served_root_of(r)
        if not hit:
            continue
        what, sr = hit
        if real(r) != real(sr):
            raise Refusal('C-INPLACE: %s is inside the served generation\'s %s %s - a generation root is a directory of its '
                          'own, never inside a served one' % (r, what, sr))
        want = 'cutout root' if stage == 'cutouts' else 'tier forced root'
        if not what.startswith(want):
            raise Refusal('C-INPLACE: %s is the served generation\'s %s - the %s stage writes a %s' % (r, what, stage, want))
        served.append(r)
    if not served:
        if in_place:
            raise Refusal('C-INPLACE: --in-place regenerates the served generation, but the %s stage writes %s, no served '
                          'root (photometry.CUTOUT_ROOT / TIER_FORCED_ROOTS)' % (stage, ', '.join(roots)))
        return
    if not in_place:
        raise Refusal('C-INPLACE: %s is the SERVED generation\'s root - a regeneration writes a new root (%s), never over the '
                      'served product (production\'s rule: phase3_snapshot_search.regen_cutouts "a new root - never over the '
                      'existing product"; generation 5 was built beside generation 4, its measurement too, and the config '
                      'switched after); --in-place regenerates the served generation itself, its measurement chained'
                      % (', '.join(served), '--cutout-root NEW' if stage == 'cutouts' else '--tier-out NEW'))
    if stage == 'cutouts':
        if args.tier == 'snapshot':
            if 'measure' not in stages or real(run.out_root) != real(ph.OUT_ROOT['snapshot']):
                raise Refusal('C-INPLACE: --in-place cutouts replace the cutouts the served stage 1 measured: the measure stage '
                              'must run in this invocation (--stages cutouts,measure; selected %s) into the served stage 1 '
                              '(--out %s; given %s)' % (','.join(stages), ph.OUT_ROOT['snapshot'], run.out_root))
            gate_in_place_whole_cells(args, cells or [])
        elif 'cutmeasure' not in stages or getattr(args, 'tier_out', None):
            raise Refusal('C-INPLACE: --in-place tier cutouts replace the cutouts the served tier forced roots measured: '
                          'cutmeasure must run in this invocation (--stages cutouts,cutmeasure; selected %s) into the served '
                          'roots photometry.TIER_FORCED_ROOTS (no --tier-out; given %s)'
                          % (','.join(stages), getattr(args, 'tier_out', None)))
    else:
        if getattr(args, 'tier_out', None):
            raise Refusal('C-INPLACE: --in-place cutmeasure writes each sample\'s own served root (photometry.'
                          'TIER_FORCED_ROOTS); --tier-out %s puts every sample under one root' % args.tier_out)
        if real(K.cutout_root(args)) != real(ph.CUTOUT_ROOT):
            raise Refusal('C-INPLACE: the served tier forced roots measure the served cutouts (photometry.CUTOUT_ROOT %s); '
                          '--cutout-root %s is another generation' % (ph.CUTOUT_ROOT, K.cutout_root(args)))


def gate_in_place_whole_cells(args, cells):
    """C-INPLACE, snapshot (fix verification N1, 2026-11-03): an --in-place regeneration re-measures what it places in the
    same invocation, and the chained measure rewrites a cell's served stage 1 only as a whole - its full-band manifest
    (photometry_measure.MANIFEST_NAME, rewritten when the measured bands are every band of photometry.SNAPSHOT_BANDS;
    steps_snapshot.measure_cell). So --in-place regenerates WHOLE cells: the measure's own selection (steps_snapshot.
    select_cells / bands_for, what measure_cell will measure) must hold every cell the cutouts stage writes (``cells``; a
    --cells pattern is expanded over the cube store for the cutouts and over the search outputs for the measure), each item
    over every band. A band subset (<cell>_<band>, --scopes) - which the measure would refuse (SM-SUBSET) only AFTER the
    cutouts were placed in the served root - another --manifest-name, or an --index listing (the measure would read a listing,
    not the root as placed) refuses here, before anything is placed."""
    from exopipe.phase3.lib.photometry import steps_snapshot as KS
    c = K.cfg()
    full = list(c.photometry.SNAPSHOT_BANDS)
    man = c.photometry_measure.MANIFEST_NAME
    if getattr(args, 'manifest_name', None) and args.manifest_name != man:
        raise Refusal('C-INPLACE: --in-place re-measures the served stage 1 into its full-band manifest %s; --manifest-name %s '
                      'would leave that manifest naming the replaced cutouts' % (man, args.manifest_name))
    if getattr(args, 'index', None):
        raise Refusal('C-INPLACE: --in-place re-measures the cutouts as placed in the served root; --index %s would measure a '
                      'listing made before them' % args.index)
    try:
        sel, _ = KS.select_cells(args, c.paths.SNAPSHOT_FLEET_OUT)
    except ValueError as e:
        raise Refusal('C-INPLACE: the chained measure cannot select the cells of this regeneration (%s)' % e)
    subset = sorted('%s_%s' % (cl, sb) if sb else cl for cl, sb in sel if list(KS.bands_for(args, sb)) != full)
    if subset:
        raise Refusal('C-INPLACE: --in-place regenerates whole cells - the chained measure rewrites a cell\'s served stage 1 '
                      'only over every band (%d, photometry.SNAPSHOT_BANDS) and refuses a band subset (SM-SUBSET) after the '
                      'cutouts are placed; %s select%s a band subset (%s) - name whole cells, without --scopes'
                      % (len(full), ', '.join(subset[:5]), 's' if len(subset) == 1 else '',
                         '--scopes %s' % args.scopes if args.scopes else 'a <cell>_<band> item'))
    missing = sorted(set(cells) - {cl for cl, _ in sel})
    if missing:
        raise Refusal('C-INPLACE: the cutouts stage would place %d cell(s) the chained measure does not select (%s; it '
                      'expands --cells over the search outputs %s, the cutouts over the cube store) - their served stage 1 '
                      'would not be re-measured; name the cells' % (len(missing), ', '.join(missing[:5]),
                                                                    c.paths.SNAPSHOT_FLEET_OUT))


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
        mrels = unit_manifest_rels('10s', '%s_%s' % (cell, band), sfx)
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
                u.meta['staging'] = None
                why = 'C-EXCLUDED: %s %s carries verdict exclude in data_decisions (the engine refused it, exit %d)' % (
                    cell, band, res.rc)
                nothing_written(u, run, root, '10s', cell, band, samples, why, mrels)
                raise NoInput(why)
            if res.refused:          # the engine's own refusal (e.g. a cube repeating integrations without the search's frame
                u.refuse('C-ENGINE: %s (staging kept: %s)' % (res.reason, stage))   # list): this unit, not the run
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
                if res.refused:
                    u.refuse('C-ENGINE: %s (%s): %s (staging kept: %s)' % (band, S, res.reason, stage))
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
                u.meta['staging'] = None
                why = 'C-EXCLUDED: %s %s is excluded under the deep rule (the group search refused it, exit 3)' % (cell, band)
                nothing_written(u, run, root, '10s', cell, band, samples, why, mrels)
                raise NoInput(why)
        u.meta['engine_outcome'] = outcome
        rels, other = staged_cutouts(stage, samples)
        if other:
            u.refuse('C-NAMES: the engine wrote files that are no cutout of samples %s: %s (staging kept: %s)' % (
                ','.join(samples), TC._fmt(other), stage))
        if not rels:
            remove_staging(stage)
            u.meta['staging'] = None
            why = 'C-NOCUT: the engine wrote no cutout for %s %s (outcome %s)' % (cell, band, outcome)
            nothing_written(u, run, root, '10s', cell, band, samples, why, mrels)
            raise NoInput(why)
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
        # 2-4. one file per key, the manifest and its content tests on the staged files, the placement transaction
        extra = [(members[S], mrels[1 + 'VI'.index(S)]) for S in sorted(members)]   # beside the manifest: the member tables
        place_unit(u, run, root, stage, rels, '10s', cell, band, samples, mrels[0],   # a wideband is rebuilt from
                   {'cell': cell, 'band': band}, extra)
        for S, (_, rel) in zip(sorted(members), extra):
            u.meta.setdefault('members', {})[S] = os.path.join(root, rel)
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


def tier_input_roots(tier):
    """The input roots the tier tool reads by literal path (photometry_tier_cutouts_20261025.py :46-48): the deep run
    (TIER_DEEP_RUN, node-local to the deep host) or the 10-minute window root and its search (TIER_TENMIN_ROOT,
    TIER_TENMIN_SEARCH)."""
    cc = _cc()
    return [cc.TIER_DEEP_RUN] if tier == 'deep' else [cc.TIER_TENMIN_ROOT, cc.TIER_TENMIN_SEARCH]


def tier_inputs_unreadable(tier):
    """[root] of tier_input_roots that are not readable directories on this host."""
    return [r for r in tier_input_roots(tier) if not (os.path.isdir(r) and os.access(r, os.R_OK | os.X_OK))]


def gate_tier_host(tier):
    """C-HOST (review F5): an input root of the tier tool that this host cannot read refuses the tier cutouts (exit 2) - the
    tool would cut no plane and the unit would end 'no input' (exit 3), which a chain records as nothing to do. The deep run
    is on /fast of the deep host only."""
    bad = tier_inputs_unreadable(tier)
    if bad:
        raise Refusal('C-HOST: the %s tier cutouts read %s by literal path (photometry_cutouts.TIER_DEEP_RUN / '
                      'TIER_TENMIN_ROOT / TIER_TENMIN_SEARCH), not a readable directory on %s - run them on the host that '
                      'holds it' % (tier, ', '.join(bad), K.hostname()))


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
    """The <LST>_<band> cells of --cells over the tier tool's cell list (tier_universe). A named cell the tool does not know
    refuses (C-CELLS, review F5: it would cut nothing). Where the tool's inputs are not readable (cutmeasure, which reads only
    the cutout root, may run there) the cells are taken as named and a pattern refuses (C-HOST)."""
    spec = args.cells
    if not spec:
        raise Refusal('C-CELLS: a tier cutout run names its <LST>_<band> cells (--cells; patterns over the tool\'s cells, '
                      'e.g. --cells "08h_*")')
    unreadable = tier_inputs_unreadable(tier)
    universe = None if unreadable else tier_universe(tier)
    pats = [t for t in str(spec).split(',') if any(ch in t for ch in '*?[')]
    if universe is None and pats:
        raise Refusal('C-HOST: --cells %s: a pattern over the tier tool\'s cells, whose list (%s) is not readable on %s - '
                      'name the cells' % (','.join(pats), ', '.join(unreadable), K.hostname()))
    cells = C.expand(spec, universe=universe, kind='cell')
    for cl in cells:
        C.parse_cell(cl)
    if universe is not None:
        unknown = [cl for cl in cells if cl not in universe]
        if unknown:
            raise Refusal('C-CELLS: %s %s no cell of the %s tier tool\'s input (%s: %d cells) - it would cut nothing'
                          % (', '.join(unknown[:5]), 'is' if len(unknown) == 1 else 'are', tier,
                             ', '.join(tier_input_roots(tier)), len(universe)))
    if args.scopes:
        sc = {s.strip() for s in args.scopes.split(',') if s.strip()}
        cells = [cl for cl in cells if cl.split('_', 1)[1] in sc]
    if args.exclude_cells:
        ex = set(C.expand(args.exclude_cells, universe=universe if universe is not None else cells, kind='cell'))
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
        gate_tier_host(tier)
        cells = tier_items(args, tier)
        gate_write_roots(run, args, 'cutouts', [root], targets, {'cutout root': root})
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
        units += run_unit(run, 'cutouts.%s.%s%s' % (token, cell, tag_suffix(targets)),
                          dict(tier=tier, stage='cutouts', cell=cell, cutout_root=root),
                          tier_unit, run, args, tier, cell, root, targets, planets, samples)
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
        if res.refused:              # the tool's own refusal (REFUSING: injected run / the old root): this unit, not the run
            u.refuse('C-ENGINE: %s (staging kept: %s)' % (res.reason, stage))
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
            u.meta['staging'] = None
            why = 'C-NOCUT: no tier cutout for %s %s (%s)' % (token, cell, '; '.join(lines) or 'no line')
            nothing_written(u, run, root, token, lst, band, samples, why, unit_manifest_rels(token, cell, sfx))
            raise NoInput(why)
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
        place_unit(u, run, root, stage, rels, token, lst, band, samples, unit_manifest_rels(token, cell, sfx)[0],
                   {'tier': token, 'band': band})
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
        gate_write_roots(run, args, 'cutmeasure', sorted(roots), targets,
                         {'tier forced root %s' % r: r for r in roots})      # one entry per root (review F9)
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
            units += run_unit(run, 'cutmeasure.%s.%s.%s' % (token, lst, '+'.join(smp)),
                              dict(tier=tier, stage='cutmeasure', lst=lst, cutout_root=root, tier_out=out_root),
                              cutmeasure_unit, run, args, tier, lst, root, out_root, smp)
    return units


def cutmeasure_unit(run, args, tier, lst, root, out_root, samples):
    from exopipe.phase3.lib.photometry import tests_cutouts as TC
    ph = K.cfg().photometry
    cc = _cc()
    token = cc.TIER_TOKEN[tier]
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
        if res.refused:              # the tool's own refusal (REFUSING: --samples ...): this unit, not the run
            u.refuse('C-ENGINE: %s (staging kept: %s)' % (res.reason, stage))
        if res.rc != 0 or not m:
            raise RuntimeError('photometry_tier_forced_20261026.py exit %d without PHOTOMETRY-TIER-FORCED-DONE (staging kept: '
                               '%s): %s' % (res.rc, stage, res.tail[-600:]))
        u.meta.update(n_measured=int(m.group(1)), n_refused=int(m.group(2)))
        npz = sorted(glob.glob(os.path.join(sdir, 'forced_*.npz')))
        mine = [p for p in npz if TC._NPZ_RE.match(os.path.basename(p))
                and TC._NPZ_RE.match(os.path.basename(p))['smp'] in samples]
        if int(m.group(2)):
            u.refuse('CM-REFUSED: the tool refused %s cutouts of %s %s (staging kept: %s)' % (m.group(2), token, lst, stage))
        # the content tests on the STAGED npz and manifest decide before anything reaches the tier forced root (review F2):
        # TN1 / TN2 (each npz's targets re-measured from the cutout root) and TM1 / TM2 (the npz of the samples this unit does
        # not measure - a subset run merges the manifest - read where they are served)
        staged_man = os.path.join(sdir, 'forced_manifest.csv')
        bad = []
        for p in mine:
            bad += staged_tests(u, 'photometry_tier_npz', p, cutout_root=root)
        bad += staged_tests(u, 'photometry_tier_manifest', staged_man,
                            npz_dirs={s: odir for s in ph.TIER_FORCED_SAMPLES if s not in samples})
        if bad:
            u.refuse('CM-CONTENT: the staged npz / manifest fail their content test before placement - nothing placed '
                     '(staging kept: %s): %s' % (stage, '; '.join(bad)[:900]))
        for p in mine:
            u.plan(os.path.join(odir, os.path.basename(p)), 'photometry_tier_npz')
        u.plan(old_man, 'photometry_tier_manifest')
        pairs = [(p, os.path.join(token, lst, os.path.basename(p))) for p in mine] + \
            [(staged_man, os.path.join(token, lst, 'forced_manifest.csv'))]

        def outputs():
            for p in mine:
                u.add_output(os.path.join(odir, os.path.basename(p)), 'photometry_tier_npz', cutout_root=root)
            u.add_output(old_man, 'photometry_tier_manifest')
        # placement: the npz and the manifest this unit replaces move to <out root>/<FORCED_QUARANTINE_PREFIX><run id>/ first
        # (sha256 manifest); a failure undoes every move (placement)
        moved = placement(u, run, out_root, pairs, outputs, 'replaced by %s (%s)' % (u.name, run.run_id),
                          prefix=cc.FORCED_QUARANTINE_PREFIX)
        u.meta['n_replaced_quarantined'] = len(moved)
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
        if K.stop_after(args, st, units):
            K.say('photometry %s: stage %s did not complete (%s) - later stages not run'
                  % (args.tier, st, ', '.join('%s %s' % (u.name, u.status) for u in units
                                              if u.status not in ('done', 'skipped-valid'))))
            break
    return None
