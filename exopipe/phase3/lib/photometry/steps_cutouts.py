"""`exopipe3 photometry --stages cutouts | cutmeasure`: the photometry CUTOUT product of every tier, written into a cutout
GENERATION root (CUTOVER_READINESS 'PHOTOMETRY CUTOUTS REGENERATED FROM THE FLOW CUBES, WITHOUT RE-SEARCHING' and 'EVERY TIER';
Gregg 2026-10-25 ~00:52 UTC "Forced photometry is also approved", ~02:15 UTC "I want the photometry done for all tiers btw,
including 1 hour (deep)").

Stage cutouts, one unit per item:
  snapshot  per (night cell, band): the search's own path to the cutouts on the consolidated flow cubes, then stop - in place,
            phase3_snapshot_search.py --cutouts-from-cubes (subbands, V then I) / phase3_snapshot_group_search.py
            --cutouts-from-cubes GROUP (widebands, per Stokes V then I), as snapshot_cutout_regen_20261025.sh :26-32 and
            snapshot_cutout_regen_wb_20261025.sh :27-41 ran them. An excluded (cell, band) ends with no input BEFORE any
            engine starts (exclusion_verdict: core/exclusions decides; the engines' own readers, read by cutouts_probe.py
            once per stage, must read the same bytes and agree, else the unit is refused - B-38 c, PHOT-EXCL-READERS /
            PHOT-EXCL-FILE); the engines' exit 3
            (data_decisions) stays a second line. Then, before any engine starts, the FLUX-SCALE GATE (C-FLUXSCALE,
            fluxscale_gate; gen6_cube_gate_DRAFT.py): every cube the engine reads - a subband's V and I, a wideband's
            member cubes of the group search's own member list (read by cutouts_probe.py; it must equal core/exclusions',
            PHOT-EXCL-READERS) - carries FSAPPLD equal to its rescale record, V = I. After the cut (step 1 below) the cubes
            are checked again, unchanged, and a subband's flux-scale record is staged (fluxscale_recheck; placed with the
            cutouts, its test FS1 and the cubes' identity check inside the placement, fluxscale_in_place), then the
            generation checks (a) record = cube,
            (b) = generation 5 / FSAPPLD on its comparison sample, (c) the writer's target set, (w) wideband weights =
            1 / (FSAPPLD rms^2) refuse the unit on failure (C-GENCHECK, generation_checks). Units cutouts.<cell>.<band>.
  deep, sublst  per <LST>_<band>: photometry_tier_cutouts_20261025.py --tier deep|10min --cells <LST>_<band> in place (the
            tier residuals: deep Run_P3S5 iresid / vresid, 10-minute tenmin_root_x2 windows). Units cutouts.<deep|10min>.<cell>.
Stage cutmeasure (deep, sublst), one unit per (tier, LST, root): photometry_tier_forced_20261026.py --tier --lsts --samples in
  place (the snapshot chain's estimator on the tier cutouts; planets at each plane's own pixel with orbit controls). Its samples
  (photometry.TIER_FORCED_SAMPLES; LPT, which the tool does not name, as an ad-hoc tag: tier_engine_env) go to their served
  roots (photometry.TIER_FORCED_ROOTS) unless --tier-out names one. Units cutmeasure.<deep|10min>.<LST>.

Every unit writes through a STAGING directory of its own on the root's filesystem (<root>/_staging/<run id>/<unit>/, the
engine's output root), then:
  0. the write itself (FINDING-20261122-CUTOUT-WRITER-DISCARDED-PAGES-PASS-AS-DONE; settle_staged, check_counts): every staged
     FITS file fsynced, its on-disk size equal to the size its headers imply and holding no hole (C-WRITE), then - 10 s - the
     files per Stokes accounting for the engine's reported count (list entries sharing one file name counted as the writer
     counts the ones in its view), a positive count without files included (C-COUNT). A failure refuses the unit, nothing
     placed;
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
C-FLUXSCALE (snapshot: a cube the engine reads not rescaled, off its rescale record, V != I, without a rescale record, or a
cube changed between the gate and placement), C-FLUXSCALE-READ (snapshot, before the engine: the rescale records or a cube
header cannot be read - operational, retryable; never a campaign gap), C-GENCHECK (snapshot: a generation check
(a)(b)(c)(w) failed), PHOT-ADHOC (an ad-hoc target set into or under a served root, or for evaluate), PHOT-FULL (snapshot
Full).
"""
import csv
import glob
import hashlib
import json
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


def run_samples(targets, planets, tier):
    """The sample directories a cutout run writes: the ad-hoc tags (+ the planets with --planets), else the samples ``tier``'s
    cutters write (photometry.SAMPLES_CUT, every tier) and the planets (phase3_snapshot_cutouts._samples / _planets)."""
    ph = K.cfg().photometry
    if targets:
        return [t for t, _ in targets] + ([ph.PLANET_SAMPLE] if planets else [])
    return list(K.samples_cut(tier)) + [ph.PLANET_SAMPLE]


def tag_suffix(targets):
    return ('.' + '+'.join(t for t, _ in targets)) if targets else ''


def listing_sha1(paths, extra=''):
    """sha1 of (path, size, mtime_ns) of each path, then of ``extra`` (a text the identity adds: a snapshot unit's resolved
    rescale records, rescale_identity)."""
    h = hashlib.sha1()
    for p in sorted(paths):
        try:
            st = os.stat(p)
            h.update(('%s\t%d\t%d\n' % (p, st.st_size, st.st_mtime_ns)).encode())
        except OSError:
            h.update(('%s\tmissing\n' % p).encode())
    if extra:
        h.update(extra.encode())
    return h.hexdigest()


def rescale_identity(cubes):
    """({cube: resolved rescale record | None}, digest text) of the flow cubes a snapshot unit reads (fluxscale.resolved:
    each cube's record by the row it was resolved from - RESULTS file and line, utc, fsappld; or plan file and line, status,
    factor). The unit's resume identity for the flux-scale gate (review of PR 153, finding 3, follow-up): a newer row of one
    of its cubes, or a plan row of one changing status, re-runs the unit on --resume; a row appended for another cube does
    not (whole record files as inputs re-ran every unit of a generation on any rescale run). Records that cannot be read:
    ({}, 'unreadable: ...'), never a done unit's identity (the gate then refuses C-FLUXSCALE)."""
    from exopipe.phase3.lib.photometry import fluxscale as FS
    try:
        res = FS.resolved(sorted(cubes))
    except (FS.RecordsError, OSError) as e:
        return {}, 'rescale records unreadable: %s: %s' % (type(e).__name__, e)
    return res, json.dumps(res, sort_keys=True)


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


def place_unit(u, run, root, stage, rels, token, key, band, samples, mrel, ctx, extra=(), extra_outputs=(), in_place=()):
    """Steps 2 to 4 of the module docstring for the cutouts of one unit (``rels``, relative to the staging directory and the
    root alike): one file per key (C-STALE: refuse, or quarantine with --fresh), the unit's manifest written in the staging
    directory (<stage>/<mrel>) and its class tests run on the staged files (C-CONTENT), then the placement transaction with the
    class tests on the placed manifest. ``extra``: more (staged path, relative path) pairs placed with the cutouts (the wideband
    member tables beside the manifest, the flux-scale record); ``extra_outputs`` [(placed path, class)]: recorded with their
    class tests once placed, inside the transaction (the flux-scale record, FS1); ``in_place``: callables run after them,
    inside the transaction, each recording unit checks (a failure undoes the placement: the cubes unchanged,
    fluxscale_in_place). Returns the placed manifest's path."""
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
    for p, c_ in extra_outputs:
        u.plan(p, c_)
    pairs = [(os.path.join(stage, r), r) for r in rels] + list(extra) + [(staged_man, mrel)]

    def outputs():
        u.add_output(manifest_path, cls, root=root, samples=samples, **ctx)
        for p, c_ in extra_outputs:
            u.add_output(p, c_)
        for fn in in_place:
            fn()
    moved = placement(u, run, root, pairs, outputs,
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
    photometry.SNAPSHOT_BANDS (--scopes narrows), minus --exclude-cells. C-CELLS: a regeneration names its cells. A campaign
    (--campaign, started: args.photometry_campaign): the campaign's own unit list (campaign.enumerate_units), subbands before
    widebands within a cell."""
    from exopipe.phase3.lib.photometry import steps_snapshot as KS
    camp = getattr(args, 'photometry_campaign', None)
    if camp is not None:
        return sorted(camp.enum.units)
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
    from exopipe.phase3.lib.photometry import campaign as CP
    camp = CP.start(run, args) if getattr(args, 'campaign', False) else None   # the campaign's units, pin, space, I/O guard
    targets = K.adhoc_targets(args)
    root = K.cutout_root(args)
    planets = bool(getattr(args, 'planets', False))
    items = snapshot_items(args)
    samples = run_samples(targets, planets, 'snapshot')
    K.plan(run, '[cutouts] %d (night cell, band) units -> %s (generation %s); samples %s%s; engines %s, site PYTHON[%s]'
           % (len(items), root, K.cfg().photometry.GENERATIONS['snapshot'] if root == K.served_cutout_root('snapshot')
              else 'not served',
              ','.join(samples), ' (ad hoc: %s)' % ', '.join('%s=%s' % t for t in targets) if targets else '',
              ('subbands lib/photometry/stamp_engine.py (the stamp method, reference %s), widebands in place '
               '(phase3_snapshot_group_search.py --cutouts-from-cubes)' % K.cfg().photometry_cutouts.STAMP_REFERENCE_ROOT
               if targets else 'in place (phase3_snapshot_search.py / phase3_snapshot_group_search.py --cutouts-from-cubes)'),
              K.python_key('snapshot', 'cutouts_subband')))
    try:
        gate_write_roots(run, args, 'cutouts', [root], targets, {'cutout root': root}, cells=sorted({cl for cl, _ in items}))
    except Refusal as e:
        if run.dry_run:
            K.would_refuse(run, 'cutouts', str(e))
            return []
        raise
    units = []
    reads, gen5 = None, Gen5Samples()     # once per stage: the engines' reads, the generation-5 sample of each hour
    dry_ex, gate_refused, dry_nocube = None, 0, 0
    if run.dry_run and items:          # the flux-scale gate's verdicts, headers only (no engine runs in a dry run)
        from exopipe.phase3.core import exclusions
        try:
            dry_ex = exclusions.for_tier('snapshot', fresh=True)
        except Exception as e:                                 # noqa: BLE001 - reported per unit as the run would refuse
            dry_ex = '%s: %s' % (type(e).__name__, str(e)[:200])
    for cell, band in items:
        cubes = snapshot_cubes(cell, band)
        if not run.dry_run and reads is None:      # once per stage: the engines' target list and exclusion reads, every cell
            reads = EngineReads(run, [cl for cl, _ in items], targets, planets, sorted({b for _, b in items if is_wideband(b)}))
        if run.dry_run:            # in snapshot_unit's order: no cube ends the unit (C-NOCUBE) before the flux-scale gate
            if not cubes['V'] or not cubes['I']:
                verdict = 'no input C-NOCUBE (no consolidated flow cube, %s)' % ('V' if not cubes['V'] else 'I')
                dry_nocube += 1
            else:
                verdict = fluxscale_dry(cell, band, dry_ex)
                gate_refused += verdict.startswith('would refuse')
            K.plan(run, '[cutouts]   unit cutouts.%s.%s%s: cubes V %d, I %d; %s' % (cell, band, tag_suffix(targets),
                                                                                 len(cubes['V']), len(cubes['I']), verdict))
            continue
        done = run_unit(run, 'cutouts.%s.%s%s' % (cell, band, tag_suffix(targets)),
                        dict(tier='snapshot', stage='cutouts', cell=cell, band=band, cutout_root=root),
                        snapshot_unit, run, args, cell, band, root, targets, planets, samples, cubes, reads, gen5)
        if camp is not None:
            for x in done:              # the unit's engine children: their code joins the campaign's pin, or stops it (CP-PIN)
                camp.after(x)
        units += done
    if run.dry_run and items:
        K.plan(run, '[cutouts] flux-scale gate (headers only): %d of %d unit(s) would refuse; %d without a cube (C-NOCUBE)'
                    % (gate_refused, len(items), dry_nocube))
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
    cutmeasure stage writes a SERVED root (photometry.CUTOUT_ROOTS; photometry.TIER_FORCED_ROOTS) only with --in-place, and then
    the measurement of the new cutouts runs in the same invocation: snapshot cutouts -> the measure stage into the served
    stage 1 (--out = photometry.OUT_ROOT[snapshot]), over whole cells (gate_in_place_whole_cells); deep / sublst cutouts ->
    cutmeasure into the served tier forced roots (no --tier-out; it re-measures every band of each LST found in the root);
    cutmeasure into the served tier forced roots measures the served cutouts (no other --cutout-root). A root INSIDE a served
    root is no generation root at all (refused); --in-place with no served root to write refuses; a cutouts stage writes only
    its OWN tier's served cutout root (photometry.CUTOUT_ROOTS[tier]: the snapshot per-night cutouts are generation 6, the
    deep / sub-LST tier cutouts generation 5 - never one tier's cutouts into another tier's served root)."""
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
        if stage == 'cutouts' and real(sr) != real(K.served_cutout_root(args.tier)):
            raise Refusal('C-INPLACE: %s is the served %s, not tier %s\'s (photometry.CUTOUT_ROOTS[%s] = %s) - a tier\'s '
                          'cutouts never go into another tier\'s served root' % (r, what, args.tier, args.tier,
                                                                               K.served_cutout_root(args.tier)))
        served.append(r)
    if not served:
        if in_place:
            raise Refusal('C-INPLACE: --in-place regenerates the served generation, but the %s stage writes %s, no served '
                          'root (photometry.CUTOUT_ROOTS / TIER_FORCED_ROOTS)' % (stage, ', '.join(roots)))
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
        if real(K.cutout_root(args)) != real(K.served_cutout_root(args.tier)):
            raise Refusal('C-INPLACE: the served tier forced roots measure the served cutouts (photometry.CUTOUT_ROOTS[%s] %s); '
                          '--cutout-root %s is another generation' % (args.tier, K.served_cutout_root(args.tier),
                                                                      K.cutout_root(args)))


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


def cutout_admitted(cell, band, ex=None):
    """The unit's resume identity under the exclusions (core.exclusions; ``ex`` the unit's own fresh read, else the process
    read): '<cell>:<band>' unless the regeneration refuses the (cell, band) - a subband by its own row
    (phase3_snapshot_search.main :1824-1827), a wideband by the deep rule (regen_group_cutouts :467-470) - plus, for a
    wideband, '<cell>:<member>' of every member subband it keeps (it drops the excluded ones, :472-473). A new exclusion
    changes the set, so --resume re-runs the unit."""
    from exopipe.phase3.core import exclusions
    ex = ex or exclusions.for_tier('snapshot')
    lst, date = cell.split('_', 1)
    if ex.is_excluded(lst, band, date):
        return []
    out = ['%s:%s' % (cell, band)]
    if is_wideband(band):
        members = K.cfg().instrument.SUBBAND_BANDS.get(band, ())
        out += ['%s:%s' % (cell, sb) for sb in members if not ex.is_excluded(lst, sb, date)]
    return sorted(out)


def file_key(rel):
    """'<sample>/<object>' of a staged cutout '<sample>/<band>/<object>/<file>' (the writer's target name key)."""
    p = rel.split('/')
    return '%s/%s' % (p[0], p[2])


class EngineReads:
    """The 10-s cutout engines' own reads (FINDING-20261122-CUTOUT-WRITER-DISCARDED-PAGES-PASS-AS-DONE), taken by
    cutouts_probe.py in a child process under their interpreter and environment (the ad hoc set included): ONE call per stage
    ('run' mode, before any engine starts) for the writer's target list and the exclusion readers of every night cell - the
    subband engine's (excluded_subbands) and, when the stage has a wideband, the group search's deep-rule reader
    (coadd_full_pipeline, config photometry_cutouts.GROUP_EXCLUSION_ROOT); a cell is read again when the unit's own read of
    data_decisions.csv finds other bytes than the probe read (verdict). Per unit, after its engine, only when a staged cutout's
    name is shared by more than one list entry: 'inview' mode (in_view) - it needs the unit's staged TIMES and cube frame
    headers. Each unit takes the calls' code fingerprints as inputs (attach)."""

    def __init__(self, run, cells, targets, planets, widebands):
        self.run, self.targets, self.planets = run, targets, planets
        c = K.cfg()
        self.widebands = sorted(set(widebands))
        self.scopes = sorted({s for b in widebands for s in [b] + list(c.instrument.SUBBAND_BANDS.get(b, ()))})
        self.code_root = os.path.dirname(K.engine('cutouts_subband'))
        self.error = None
        self.multiple, self.target_meta, self.cells, self.calls, self.inputs = {}, {}, {}, [], set()
        if widebands and os.path.dirname(K.engine('cutouts_wideband')) != self.code_root:
            self.error = ('PHOT-EXCL-FILE: the subband and wideband engines live in different code roots (%s, %s); the probe '
                          'reads one' % (self.code_root, os.path.dirname(K.engine('cutouts_wideband'))))
            return
        self.error = self._probe(sorted(set(cells)))

    def _call(self, argv, log, u=None):
        res, text = K.call(u, 'cutouts_probe', argv, 'snapshot', 'cutouts_probe', self.run, log,
                           env_extra=K.adhoc_env(self.targets, self.planets))
        if u is None:
            self.inputs |= {f for f in (res.fingerprint or {}) if os.path.isfile(f)} | {K.engine('cutouts_probe')}
            self.calls.append({'argv': argv[:6] + (['...'] if len(argv) > 6 else []), 'rc': res.rc, 'log': res.log})
        return res, text

    def _probe(self, cells):
        """One 'run' call for ``cells``; their records into self.cells. Returns the failure (refusal text) or None."""
        argv = ['run', '--code-root', self.code_root]
        if self.scopes:
            argv += (['--group-reader-root', K.cfg().photometry_cutouts.GROUP_EXCLUSION_ROOT, '--scopes'] + self.scopes
                     + ['--groups'] + self.widebands)
        try:
            res, text = self._call(argv + ['--cells'] + cells, 'cutouts_probe_run.log')
        except Exception as e:     # noqa: BLE001 - the reads not taken: the units refuse (PHOT-EXCL-FILE), the stage goes on
            return ('PHOT-EXCL-FILE: the engines\' reads (cutouts_probe.py) could not be taken (%s: %s)'
                    % (type(e).__name__, e))
        tl = [json.loads(ln.split(TARGETS_MARK, 1)[1]) for ln in text.splitlines() if ln.startswith(TARGETS_MARK)]
        recs = [json.loads(ln.split(EXCL_MARK, 1)[1]) for ln in text.splitlines() if ln.startswith(EXCL_MARK)]
        if res.rc != 0 or len(tl) != 1 or sorted(r.get('cell') for r in recs) != cells:
            return ('PHOT-EXCL-FILE: the engines\' reads in %s (cutouts_probe.py: their writer\'s target list, their '
                    'exclusion readers) failed (exit %d, %d target line(s), %d of %d cells): %s'
                    % (self.code_root, res.rc, len(tl), len(recs), len(cells), res.tail[-600:]))
        self.multiple = dict(tl[0]['multiple'])
        self.target_meta = tl[0]
        self.cells.update({r['cell']: r for r in recs})
        return None

    def attach(self, u):
        for f in sorted(self.inputs):
            u.add_input(f)
        u.meta['engine_reads'] = {'calls': self.calls, 'targets': self.target_meta}

    def cell(self, cell, sha1):
        """The probe's record of ``cell``, read again once when its data_decisions bytes (sha1) are not ``sha1``."""
        rec = self.cells.get(cell)
        if rec is not None and 'error' not in rec and rec.get('sha1') != sha1:
            err = self._probe([cell])
            if err:
                self.cells[cell] = {'cell': cell, 'error': err}
            rec = self.cells.get(cell)
        return rec

    def in_view(self, u, cell, band, stage, rels, cubes):
        """{Stokes: {key: entries of that shared name write_cutouts' view test passes}} for the staged cutouts whose name more
        than one list entry shares ({} when none); refuses the unit (C-COUNT) when the read fails or is ambiguous."""
        shared = sorted({file_key(r) for r in rels if file_key(r) in self.multiple})
        if not shared:
            return {}
        lst, date = cell.split('_', 1)
        times = ['%s=%s' % (S, os.path.join(stage, sorted(r for r in rels if r.endswith('_%s.fits' % S))[0]))
                 for S in 'VI' if any(r.endswith('_%s.fits' % S) for r in rels)]
        argv = (['inview', '--code-root', self.code_root, '--lst', lst, '--date', date, '--times'] + times + ['--cubes']
                + ['%s=%s' % (S, p) for S in 'VI' for p in cubes.get(S, [])] + ['--keys'] + shared)
        res, text = self._call(argv, 'cutouts_inview_%s_%s.log' % (cell, band), u)
        got = [json.loads(ln.split(INVIEW_MARK, 1)[1]) for ln in text.splitlines() if ln.startswith(INVIEW_MARK)]
        if res.rc != 0 or len(got) != 1:
            u.refuse('C-COUNT: the writer\'s view of the target names %s shared by list entries could not be read '
                     '(cutouts_probe.py inview exit %d, %d result line(s); staging kept: %s): %s'
                     % (', '.join(shared), res.rc, len(got), stage, res.tail[-600:]))
        u.meta['shared_in_view'] = got[0]
        amb = ['%s %s' % (S, k) for S, v in sorted(got[0].items()) for k, n in sorted(v['in_view'].items()) if n is None]
        if amb:
            u.refuse('C-COUNT: the cube frames\' geometries disagree on how many entries of %s are in the writer\'s view (%s) - '
                     'no count the files can be held to (staging kept: %s)' % ('; '.join(amb), got[0], stage))
        return {S: v['in_view'] for S, v in got[0].items()}


TARGETS_MARK, EXCL_MARK, INVIEW_MARK = 'ENGINE-TARGETS ', 'ENGINE-EXCLUSIONS ', 'ENGINE-INVIEW '


def exclusion_verdict(u, reads, ex, cell, band):
    """The unit's exclusion verdict, BEFORE an engine starts (spec photometry.md B-38 decision c and PHOT-EXCL-FILE). ``ex``
    is the unit's one fresh read of data_decisions (core/exclusions, the snapshot tier's per-night read; a missing or
    unreadable file refuses, C-09), the same read as its resume identity (cutout_admitted); it decides by its rule - a
    subband by its own row; a wideband by any member subband but 18 MHz (exclusions.SUBBAND_ONLY) or [PERBAND] rows, or its
    own row. Every engine reader the unit's engine uses (EngineReads) must read the same file, the same bytes, and agree: the
    subband engine's excluded_subbands on the rows of the band and its member subbands, and for a wideband the group
    search's deep-rule reader on the band's verdict and its members' rows, and the group search's member list
    (regen_group_cutouts: GROUPS[band] without excluded_subbands) = core/exclusions' (instrument.SUBBAND_BANDS[band] not
    excluded) - the flux-scale gate checks the cubes of exactly that list (u.meta['exclusions']['engine_members']). A reader
    that fails or reads another file refuses (PHOT-EXCL-FILE), a disagreement refuses (PHOT-EXCL-READERS). True when the
    (cell, band) is excluded."""
    from exopipe.phase3.core import exclusions
    c = K.cfg()
    lst, date = cell.split('_', 1)
    own = ex.of(lst, date)
    if reads.error:
        u.refuse(reads.error + ' - the engine not started')
    reads.attach(u)
    got = reads.cell(cell, own.fingerprint()['sha1'])
    if got is None or 'error' in got:
        u.refuse('PHOT-EXCL-FILE: the engines\' exclusion readers failed for %s (%s) - the engine not started'
                 % (cell, (got or {}).get('error', 'no record of the cell')))
    if is_wideband(band) and ('group_reader' not in got or band not in (got.get('group_members') or {})):
        u.refuse('PHOT-EXCL-FILE: the group search\'s exclusion reader or its member list of %s was not read for %s - the '
                 'engine not started' % (band, cell))
    paths =[got['path']] + ([got['group_reader']['path']] if is_wideband(band) else [])
    other = [p for p in paths if os.path.realpath(p) != os.path.realpath(own.path)]
    if other:
        u.refuse('PHOT-EXCL-FILE: the engines read the exclusion list %s, core/exclusions %s (paths.DATA_DECISIONS) - one list '
                 'for every reader; the engine not started' % (', '.join(other), own.path))
    if got['sha1'] != own.fingerprint()['sha1']:
        u.refuse('PHOT-EXCL-FILE: %s changed between the engines\' read (sha1 %s) and core/exclusions\' (sha1 %s), read again '
                 '- the engine not started' % (own.path, got['sha1'], own.fingerprint()['sha1']))
    norm = exclusions._norm_scope
    members = list(c.instrument.SUBBAND_BANDS.get(band, ()))
    row = {s: (lst, norm(s), date) in own.rows for s in [band] + members}
    ss_rows = {norm(s) for s in got['excluded']}
    differ = ['%s (core %s, excluded_subbands %s)' % (s, row[s], norm(s) in ss_rows) for s in [band] + members
              if row[s] != (norm(s) in ss_rows)]
    verdict = ex.is_excluded(lst, band, date)
    if is_wideband(band):
        gv = got['group_reader']['verdicts']
        want = dict({s: row[s] for s in members}, **{band: verdict})      # the band by the rule, a member by its row
        differ += ['%s (core %s, the group search\'s is_excluded %s)' % (s, want[s], gv.get(s)) for s in [band] + members
                   if gv.get(s) != want[s]]
        em = got['group_members'][band]
        cm = [s for s in members if not ex.is_excluded(lst, s, date)]
        if em is None or sorted(norm(s) for s in em) != sorted(norm(s) for s in cm):
            differ.append('the members of %s (core/exclusions %s, the group search %s)' % (
                band, cm, 'has no group %s' % band if em is None else em))
    if differ:
        u.refuse('PHOT-EXCL-READERS: the readers of %s disagree on %s %s (B-38: the step refuses on disagreement) - excluded: '
                 '%s; the engine not started' % (own.path, lst, date, '; '.join(differ)))
    u.meta['exclusions'] = {'path': own.path, 'scope': own.scope, 'sha1': got['sha1'], 'engine_excluded': got['excluded'],
                            'group_reader': got.get('group_reader') if is_wideband(band) else None, 'excluded': verdict,
                            'engine_members': got['group_members'][band] if is_wideband(band) else None}
    return verdict


def _regen_counts(line):
    """{'V': ..., 'I': ...} of a 'CUTOUTS-REGEN <lst> <date> <sb>: V <n> I <n>[; refused: ...]' line (regen_cutouts :1414)."""
    m = re.search(r': V (.*?) I (.*?)(?:; refused: (.*))?$', line.strip())
    if not m:
        return {}, ''
    return {'V': m.group(1), 'I': m.group(2)}, m.group(3) or ''


def expected_files(n, wideband=False):
    """The number of cutout files an engine's count for one Stokes promises: an integer as written; 'None' of the SUBBAND
    engine (the night rule refused that Stokes), 'excluded' (the group search's exit 3), 'no-members' text ('no member cube -
    nothing to build') or a night-rule 'refused: ...' of the group search: 0. Anything else ('FAILED: ...', 'REFUSED: ...', a
    count missing from the done line, the group search's 'None': the group was not built) is None: no count the files can
    be held to."""
    s = str(n).strip()
    if re.fullmatch(r'\d+', s):
        return int(s)
    if (s == 'None' and not wideband) or s == 'excluded' or s.startswith('no member cube') or s.startswith('refused: '):
        return 0
    return None


def settle_staged(u, stage, rels, what, hdus=None):
    """W1, every staged cutout BEFORE its content tests (FINDING-20261122-CUTOUT-WRITER-DISCARDED-PAGES-PASS-AS-DONE: a Lustre
    eviction discarded the writer's dirty pages, the writer (no fsync, no size check) left cutouts cut short at 24 MiB and the
    unit was taken as done): per file, on one descriptor, an fsync (an error the writer never saw is reported here), its
    on-disk size equal to the size its headers imply (and ``hdus`` HDUs, when given), no hole, then its cached pages dropped,
    so the content tests read the disk (core/fitsio.settle_fits; photometry_cutouts.SETTLE_WORKERS threads). Any failure
    refuses the unit, nothing placed (C-WRITE)."""
    from concurrent.futures import ThreadPoolExecutor
    from exopipe.phase3.core import fitsio
    with ThreadPoolExecutor(max_workers=K.cfg().photometry_cutouts.SETTLE_WORKERS) as pool:
        got = list(pool.map(lambda r: fitsio.settle_fits(os.path.join(stage, r)), rels))
    bad = []
    for r, (why, nhdu) in zip(rels, got):
        if why is None and hdus is not None and nhdu != hdus:
            why = '%d HDU(s), %d expected' % (nhdu, hdus)
        if why:
            bad.append('%s: %s' % (r, why))
    u.check('photometry_cutout_manifest.W1_fsync_size', not bad, len(bad),
            '%d of %d staged cutouts fsynced, their on-disk size = the size their headers imply%s, no hole%s'
            % (len(rels) - len(bad), len(rels), ' (%d HDUs)' % hdus if hdus is not None else '',
               ('; FAIL %s' % '; '.join(bad[:5])) if bad else ''), cls='photometry_cutout_manifest')
    if bad:
        u.refuse('C-WRITE: %d of %d staged cutouts of %s not on disk whole - nothing placed (staging kept: %s): %s'
                 % (len(bad), len(rels), what, stage, '; '.join(bad[:5])))


def check_counts(u, stage, rels, band, outcome, in_view=None):
    """W2: the files staged per Stokes account for the count the engine's done line reported (expected_files), a positive
    count with no file included (a count the files cannot be held to refuses). The writer counts list entries in its view, not
    files: entries sharing one name within MERGE_ARCMIN are one object and one file, written once per entry in view (each
    counted), so a staged file of a shared name accounts for ``in_view[S][key]`` of the count (EngineReads.in_view: the
    entries that pass the writer's view test), any other file for 1. Else the unit is refused, nothing placed (C-COUNT)."""
    in_view = in_view or {}
    bad, got = [], {}
    for S in 'VI':
        mine = [r for r in rels if r.endswith('_%s.fits' % S)]
        n = len(mine)
        entries = sum(in_view.get(S, {}).get(file_key(r), 1) for r in mine)
        want = expected_files(outcome.get(S), is_wideband(band))
        got[S] = (outcome.get(S), n, entries)
        if want is None or want != entries:
            bad.append('%s: engine reported %r, %d file(s) staged%s' % (
                S, outcome.get(S), n, ' (%d list entries in view)' % entries if entries != n else ''))
    u.check('photometry_cutout_manifest.W2_count', not bad, len(bad),
            'staged cutouts per Stokes account for the engine\'s count (files/list entries in view): %s%s' % (
                ', '.join('%s %s/%d/%d' % (S, v, n, e) for S, (v, n, e) in sorted(got.items())),
                ('; FAIL %s' % '; '.join(bad)) if bad else ''), cls='photometry_cutout_manifest')
    if bad:
        u.refuse('C-COUNT: the staged cutouts of %s are not the engine\'s count - nothing placed (staging kept: %s): %s'
                 % (band, stage, '; '.join(bad)))


def _fs_brief(info, band):
    """The flux-scale gate's cube facts for the unit record: {Stokes: {fsappld, record}} ({member: {...}} of a wideband)."""
    one = lambda d: {S: {'fsappld': x.get('fsappld'), 'record': x.get('record')} for S, x in sorted(d.items())}   # noqa: E731
    return {sb: one(d) for sb, d in sorted(info.items())} if is_wideband(band) else one(info)


def fluxscale_gate(u, cell, band, ex, members=None):
    """C-FLUXSCALE, after the exclusion verdict and BEFORE any engine starts (gen6_cube_gate_DRAFT.py; the member gate of
    gen6_wb_regen_DRAFT.sh :42-45; lib/photometry/fluxscale.py): every flow cube the unit's engine reads is a rescaled cube -
    FSAPPLD present, equal to its rescale record within photometry_fluxscale.FSAPPLD_TOL (the newest RESULTS row, else the
    newest plan's 'done' row) and V = I. A subband: its own V and I cubes; a wideband: every cube of ``members``, the group
    search's own member list (exclusion_verdict: the engine's readers through cutouts_probe.py, refused unless it equals
    core/exclusions'), those with a cube (default: fluxscale.members_read, core/exclusions'). Headers only (positioned
    reads). A VERDICT on the cubes refuses C-FLUXSCALE (a campaign gap: photometry_campaign.GAP_REFUSED); rescale records
    that cannot be read (a malformed row: <file>:<line>; a file unreadable) or a cube header that cannot be read refuse
    C-FLUXSCALE-READ - operational, retryable, never a gap (READ_REFUSAL). Returns the check's cube identities (the re-checks
    after the cut and inside the placement compare them)."""
    from exopipe.phase3.lib.photometry import fluxscale as FS
    try:
        ok, why, info = (FS.check_wideband(cell, band, ex=ex, members=members) if is_wideband(band)
                         else FS.check(cell, band))
    except (FS.RecordsError, OSError) as e:                          # OSError: a rescale record file unreadable
        u.refuse(READ_REFUSAL % e)
    u.meta['fluxscale'] = _fs_brief(info, band)
    unread = [w for w in why if UNREADABLE_HEADER in w]
    if unread:
        u.refuse(READ_REFUSAL % ('%s %s: %s' % (cell, band, '; '.join(unread)[:900])))
    if not ok:
        u.refuse('C-FLUXSCALE: %s %s: %s - the engine not started' % (cell, band, '; '.join(why)[:900]))
    return info


# The flux-scale gate's operational refusal (records or a header unreadable): its own code, so a campaign never counts it as a
# gap (campaign.gap_kind keys on the code); the reason fluxscale.check_cube gives an unreadable header.
READ_REFUSAL = 'C-FLUXSCALE-READ: %s - the engine not started; operational (the records or a header unreadable), retryable, not a gap'
UNREADABLE_HEADER = ': header unreadable ('


def fluxscale_recheck(u, cell, band, root, stage, before, regen_line, ex, members=None):
    """C-FLUXSCALE, after the cut and BEFORE placement: the cubes checked again - the gate still passes and every cube's
    identity (fluxscale.IDENTITY_KEYS) equals ``before`` (the check before the engine; a wideband's ``members`` as then). A
    subband's flux-scale record (the stamp engine's sidecar <root>/_fluxscale/<subband>/<cell>_<subband>_fluxscale.json: the
    cubes as checked, the engine's result line ``regen_line``, the code) is then written into the staging directory
    (fluxscale.write_sidecar: re-check, fsync, rename) and placed with the cutouts; its content test (FS1: every listed cube
    is the cube now) and the cubes' identity check (fluxscale_in_place, subbands and widebands alike) run inside the
    placement transaction, so a cube changed during placement undoes it. A failure refuses the unit, nothing placed.
    Returns (extra (staged path, relative path) pairs, [(placed path, class)])."""
    from exopipe.phase3.lib.photometry import fluxscale as FS
    try:
        if is_wideband(band):
            ok, why, _ = FS.recheck(cell, band, before, ex=ex, members=members)
            if not ok:
                u.refuse('C-FLUXSCALE: the member cubes of %s %s fail the gate or changed since the check before the engine - '
                         'nothing placed (staging kept: %s): %s' % (cell, band, stage, '; '.join(why)[:900]))
            return [], []
        sp = FS.write_sidecar(stage, cell, band, before, regen_line, os.path.dirname(K.engine('cutouts_subband')))
    except (FS.SidecarRefused, FS.RecordsError, OSError) as e:      # OSError: a cube or a code module unreadable
        u.refuse('C-FLUXSCALE: %s - nothing placed (staging kept: %s)' % (e, stage))
    rel = os.path.relpath(sp, stage)
    return [(sp, rel)], [(os.path.join(root, rel), 'photometry_fluxscale_record')]


def fluxscale_in_place(u, cell, band, before, ex, members=None):
    """C-FLUXSCALE inside the placement transaction (place_unit ``in_place``): a callable that checks every cube of ``before``
    again once the files are placed - the gate passes and each cube's identity (fluxscale.IDENTITY_KEYS: path, FSAPPLD,
    size, mtime_ns, inode, ctime_ns) equals the check before the engine; a wideband: every member cube of ``members`` (a
    member cube that appeared or vanished is a change). A unit check photometry_cutout_manifest.GP_cubes_unchanged: a
    failure (or records that cannot be read) undoes the placement (C-PLACE) and refuses the unit."""
    from exopipe.phase3.lib.photometry import fluxscale as FS
    cls = 'photometry_cutout_manifest'

    def check():
        try:
            ok, why, _ = FS.recheck(cell, band, before, ex=ex, members=members)
        except (FS.RecordsError, OSError) as e:
            ok, why = False, ['%s: %s' % (type(e).__name__, e)]
        n = sum(len(v) for v in before.values()) if is_wideband(band) else len(before)
        u.check('%s.GP_cubes_unchanged' % cls, ok, len(why),
                ('%d cube(s) unchanged since the check before the engine and on their records' % n) if ok else
                'C-FLUXSCALE: a cube changed or fails the gate since the check before the engine: %s' % '; '.join(why)[:600],
                cls=cls)
    return check


class Gen5Samples:
    """The generation-5 comparison sample of check (b) (fluxscale.gen5_sample) of each LST hour, drawn once per stage: per
    unit it read GEN5_JOBS again and tested the hour's generation jobs again (stat calls and cube MANIFESTs). A failed read
    is not kept (the next unit of the hour reads again)."""

    def __init__(self):
        self.by_lst, self.calls = {}, 0

    def of(self, lst, ex):
        """{(cell, subband)} of hour ``lst``; raises OSError / fluxscale.RecordsError."""
        from exopipe.phase3.lib.photometry import fluxscale as FS
        if lst not in self.by_lst:
            self.calls += 1
            self.by_lst[lst] = set(FS.gen5_sample(ex=ex, lsts=[lst]))
        return self.by_lst[lst]


def generation_checks(u, cell, band, stage, rels, targets, planets, ex, before, frame_info, weights, gen5=None):
    """C-GENCHECK: the content checks of a regeneration from the rescaled cubes, on the STAGED cutouts after their identity
    test and the staged flux-scale record, before placement (gen6_cutout_regen_test_DRAFT.py (a)(b)(c), gen6_gen5_sample_
    DRAFT.py, gen6_wb_test_DRAFT.py (w); lib/photometry/tests_fluxscale.py). Each a unit check of class
    photometry_cutout_manifest; any failure refuses the unit, nothing placed.
      subband   (a) each Stokes: the staged flux-scale record = the cube (FSAPPLD, path, size, mtime);
                (b) a (cell, subband) of the generation-5 comparison sample (fluxscale.gen5_sample of its LST hour, drawn
                    once per stage: ``gen5``, the stage's Gen5Samples; the default samples only), each Stokes cut: every
                    file = its generation-5 file / FSAPPLD (median per-plane slope within 1 +- photometry_fluxscale.
                    GEN5_SLOPE_TOL), the same NaN pattern and TIMES, the same file set (samples new
                    since generation 5 aside);
                (c) each Stokes cut: one file per distinct target, and the targets = those the writer selects (elevation at
                    the cube's mid-time, inside the frame; the run's lists and planets) - ``frame_info`` from the identity test;
      wideband  (w) each Stokes cut: the member table's weights = 1 / (FSAPPLD rms^2) recomputed from the member cubes
                    without the engine (``weights``: the WeightCheck the identity rebuild fed)."""
    from exopipe.phase3.lib.photometry import fluxscale as FS
    from exopipe.phase3.lib.photometry import tests_cutouts as TC
    from exopipe.phase3.lib.photometry import tests_fluxscale as TF
    fx, cls = K.cfg().photometry_fluxscale, 'photometry_cutout_manifest'
    lst = cell.split('_', 1)[0]
    n0 = len(u.failed_tests())
    files = {S: [os.path.join(stage, r) for r in rels if r.endswith('_%s.fits' % S)] for S in 'VI'}
    if is_wideband(band):
        for S in 'VI':
            if files[S]:
                ok, worst, det = weights[S].result() if S in weights else (False, None, 'no weight check ran')
                u.check('%s.GW_weights_%s' % (cls, S), ok, worst, '(w) ' + det, cls=cls)
    else:
        sample5 = False
        if not targets:
            try:
                sample5 = (cell, band) in (gen5 or Gen5Samples()).of(lst, ex)
            except (OSError, FS.RecordsError) as e:
                u.refuse('C-GENCHECK: the generation-5 comparison sample of %s cannot be read (%s) - nothing placed (staging '
                         'kept: %s)' % (lst, e, stage))
        u.meta['generation5_sample'] = sample5
        lists = dict(targets) if targets else None
        pl = (K.cfg().photometry_cutouts.PLANETS if planets else ()) if targets else None
        for S in 'VI':
            ok, fs, det = TF.check_a(stage, cell, band, S, TC.cube_path(cell, band, S))
            u.check('%s.GA_fluxscale_record_%s' % (cls, S), ok, fs, '(a) ' + det, cls=cls)
            if not files[S]:
                continue
            if sample5:
                ok, med, det = TF.check_b(files[S], stage, fx.GEN5_ROOT, cell, band, S, (before.get(S) or {}).get('fsappld'))
                u.check('%s.GB_generation5_%s' % (cls, S), ok, med, '(b) ' + det, cls=cls)
            fi = frame_info.get(S) or {}
            if not all(k in fi for k in ('cts', 'h0', 'shape')):
                u.check('%s.GC_targets_%s' % (cls, S), False, None, '(c) the identity test did not read the cube\'s frames',
                        cls=cls)
                continue
            ok, nobj, det = TF.check_c(files[S], fi['cts'], fi['h0'], fi['shape'], lists=lists, planets=pl)
            u.check('%s.GC_targets_%s' % (cls, S), ok, nobj, '(c) ' + det, cls=cls)
    bad = u.failed_tests()[n0:]
    if bad:
        u.refuse('C-GENCHECK: %s - nothing placed (staging kept: %s)' % ('; '.join(
            '%s: %s' % (t['name'].split('.')[-1], str(t.get('detail'))[:300]) for t in bad)[:900], stage))


def fluxscale_dry(cell, band, ex):
    """The flux-scale gate's verdict of a unit in a dry run, headers only (gen6_cube_gate_DRAFT.py --dry-run): 'excluded (no
    engine)', 'flux-scale gate pass', 'would refuse C-FLUXSCALE: <reasons>' or 'would refuse C-FLUXSCALE-READ: ...'.
    ``ex``: the stage's exclusion read, or the text of its failure."""
    from exopipe.phase3.lib.photometry import fluxscale as FS
    lst, date = cell.split('_', 1)
    if isinstance(ex, str):
        return 'would refuse C-09: %s' % ex
    if ex.is_excluded(lst, band, date):
        return 'excluded (no engine)'
    try:
        ok, why, _ = FS.check_wideband(cell, band, ex=ex) if is_wideband(band) else FS.check(cell, band)
    except (FS.RecordsError, OSError) as e:
        return 'would refuse %s' % (READ_REFUSAL % e)
    unread = [w for w in why if UNREADABLE_HEADER in w]
    if unread:
        return 'would refuse %s' % (READ_REFUSAL % '; '.join(unread)[:400])
    return 'flux-scale gate pass' if ok else 'would refuse C-FLUXSCALE: %s' % '; '.join(why)[:400]


def stamp_record(cell, band):
    """The generation's flux-scale record of a (cell, subband) the stamp engine reads its reference from."""
    return os.path.join(K.cfg().photometry_cutouts.STAMP_REFERENCE_ROOT, '_fluxscale', band, '%s_%s_fluxscale.json' % (cell, band))


def snapshot_unit(run, args, cell, band, root, targets, planets, samples, cubes, reads=None, gen5=None):
    """One (night cell, band) of the snapshot cutouts stage. ``reads``: the stage's EngineReads (None: the unit's own, of its
    cell); ``gen5``: the stage's Gen5Samples (None: the unit's own)."""
    from exopipe.phase3.core import exclusions
    from exopipe.phase3.lib.photometry import fluxscale as FS
    from exopipe.phase3.lib.photometry import tests_cutouts as TC
    c = K.cfg()
    cc = c.photometry_cutouts
    lst, date = cell.split('_', 1)
    sfx = tag_suffix(targets)
    name = 'cutouts.%s.%s%s' % (cell, band, sfx)
    allcubes = cubes['V'] + cubes['I']
    stamp = bool(targets) and not is_wideband(band)        # ad hoc targets on a subband: the stamp method (STAMP_REFERENCE_ROOT)
    lists = [p for _, p in targets] or list(K.sample_paths('snapshot', cut=True).values())
    inputs = (sorted(allcubes) + lists
              + [r for r in [stamp_record(cell, band)] if stamp and os.path.isfile(r)])
    rescale, rescale_text = rescale_identity(allcubes)   # each cube's resolved rescale record (not the record files): a
    lsha = listing_sha1(inputs, rescale_text)            # change to one re-gates the unit on --resume; the inputs name it
    ex = exclusions.for_tier('snapshot', fresh=True)     # the unit's one read: its resume identity and its verdict
    admitted = cutout_admitted(cell, band, ex)
    camp = getattr(args, 'photometry_campaign', None)
    if camp is not None and run.resume and camp.stale(run, name):   # another run's record, or another engine version: cut again
        print('resume %s: not skipped (its record is not this campaign\'s, or its engine code is not the pin\'s)' % name,
              flush=True)
    elif K.run_resume_ok(run, name, admitted, {'inputs_listing_sha1': lsha, 'cutout_root': root}):
        return run.units[-1:]
    with run.unit(name, tier='snapshot', stage='cutouts', cell=cell, band=band, cutout_root=root, samples=samples,
                  targets=['%s=%s' % t for t in targets], planets=planets, inputs_listing_sha1=lsha) as u:
        if camp is not None:
            u.meta['campaign'] = camp.generation
            camp.gate(u)                # the recorders' I/O guard (a wait), the code pin and the space (CP-PIN, CP-SPACE)
        if not cubes['V'] or not cubes['I']:
            raise NoInput('C-NOCUBE: no consolidated flow cube of %s %s (%s) under %s' % (
                cell, band, 'V' if not cubes['V'] else 'I', c.paths.SNAPSHOT_CUBES_ROOT))
        for p in inputs + [c.search_snapshot.SOLAR_AUDIT]:      # data_decisions: through the admitted set (exclusions read live)
            u.add_input(p, {'rescale_record': rescale[p]} if p in rescale else None)
        for S in 'VI':
            for sb in (c.instrument.SUBBAND_BANDS.get(band, ()) if is_wideband(band) else (band,)):
                nz = os.path.join(c.paths.SNAPSHOT_FLEET_OUT, cell, 'noise_%s_%s_median.csv' % (sb, S))
                if os.path.exists(nz):          # the frame list of a duplicate-run cube (one frame per integration)
                    u.add_input(nz)
        u.set_admitted(admitted)
        mrels = unit_manifest_rels('10s', '%s_%s' % (cell, band), sfx)
        if not is_wideband(band):          # the unit's placed flux-scale record: superseded too when the unit writes nothing
            mrels.append(os.path.relpath(FS.sidecar_path(root, cell, band), root))
        if reads is None:
            reads = EngineReads(run, [cell], targets, planets, [band] if is_wideband(band) else [])
        if exclusion_verdict(u, reads, ex, cell, band):     # before any engine starts; the engines' exit 3 (below) is the
                                                             # second line
            why = ('C-EXCLUDED: %s %s is excluded in data_decisions (core/exclusions, the snapshot tier\'s read; the engines\' '
                   'reader reads the same rows) - no engine started' % (cell, band))
            nothing_written(u, run, root, '10s', cell, band, samples, why, mrels)
            raise NoInput(why)
        emembers = (u.meta.get('exclusions') or {}).get('engine_members')   # a wideband: the group search's members
        fsbefore = fluxscale_gate(u, cell, band, ex, emembers)  # C-FLUXSCALE: every cube the engine reads rescaled, on its record
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
        regen_line = None
        if not is_wideband(band):
            calls = []
            if stamp:       # the stamp engine re-cuts the reference and refuses unless it equals the generation's file
                calls.append(('cutouts_stamp', cc.STAMP_DONE, 'stamp_engine.py', 'cutouts_stamp_%s_%s.log' % (cell, band),
                              ['--code-root', os.path.dirname(K.engine('cutouts_subband')), '--ref-root',
                               cc.STAMP_REFERENCE_ROOT, '--cell', cell, '--band', band, '--out', stage]))
            calls.append(('cutouts_subband', cc.REGEN_DONE_SUBBAND, 'phase3_snapshot_search.py --cutouts-from-cubes',
                          'cutouts_%s_%s.log' % (cell, band),
                          ['--lst', lst, '--date', date, '--subband', band, '--stokes', 'VI', '--mode', 'median', '--out', eout,
                           '--cutouts-from-cubes']))
            for key, mark, what, logname, argv in calls:
                res, text = K.call(u, key, argv, 'snapshot', key, run, logname, env_extra=env, threads=args.threads)
                if key == 'cutouts_stamp' and res.rc == cc.STAMP_NO_REFERENCE_RC:
                    # no current generation-6 reference: the engine wrote nothing, the regeneration engine cuts the unit
                    why = [ln.strip() for ln in text.splitlines() if ln.startswith('NO-REFERENCE:')]
                    left = sorted(os.listdir(stage))
                    if not why or left:
                        raise RuntimeError('stamp_engine.py exit %d with %d NO-REFERENCE line(s) and %d file(s) in the staging '
                                           'directory %s (%s): %s' % (res.rc, len(why), len(left), stage, left[:5],
                                                                      res.tail[-600:]))
                    u.note('cut by the regeneration engine: the stamp engine had no current generation-6 reference (%s)'
                           % why[-1])
                    u.meta['stamp_fallback'] = why[-1]
                    continue
                break
            if res.rc == cc.REGEN_EXCLUDED_RC or '\nEXCLUDED:' in '\n' + text:
                remove_staging(stage)
                u.meta['staging'] = None
                why = 'C-EXCLUDED: %s %s carries verdict exclude in data_decisions (the engine refused it, exit %d)' % (
                    cell, band, res.rc)
                nothing_written(u, run, root, '10s', cell, band, samples, why, mrels)
                raise NoInput(why)
            if res.refused:          # the engine's own refusal (e.g. a cube repeating integrations without the search's frame
                u.refuse('C-ENGINE: %s (staging kept: %s)' % (res.reason, stage))   # list): this unit, not the run
            done = [ln for ln in text.splitlines() if mark in ln]
            if res.rc != 0 or not done:
                raise RuntimeError('%s exit %d without a %r line (staging kept: %s): %s' % (
                    what, res.rc, mark.strip(), stage, res.tail[-600:]))
            regen_line = done[-1]
            counts, refused = _regen_counts(regen_line)
            outcome = {S: counts.get(S) for S in 'VI'}
            if refused:
                u.note('the search refused (night rule): %s' % refused)
            if key == 'cutouts_stamp':     # the reference cutouts the engine reproduced: one per Stokes generation 6 cut
                parity = [ln.strip() for ln in text.splitlines() if ln.startswith('STAMP-PARITY ')]
                cut_stokes = [S for S in 'VI' if outcome.get(S) not in (None, 'None')]
                if len(parity) != len(cut_stokes) or not all(any(' %s %s: ' % (band, S) in ln for ln in parity)
                                                             for S in cut_stokes):
                    u.refuse('C-ENGINE: the stamp engine reported %d parity checks, %d expected (Stokes %s; staging kept: %s)'
                             % (len(parity), len(cut_stokes), ''.join(cut_stokes) or '-', stage))
                for ln in parity:
                    u.add_input(ln.rsplit('(', 1)[-1].rstrip(')'))
                u.meta['stamp_parity'] = parity
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
                if outcome[S] == 'None':     # run_group_search returned no result for the group: not built (too few frames)
                    u.refuse('C-ENGINE: the group search did not build %s of %s (%s): its done line reports None cutout cubes '
                             '(no result for the group, e.g. too few frames) - no count the files can be held to (staging kept: '
                             '%s)' % (band, cell, S, stage))
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
        settle_staged(u, stage, rels, '%s %s' % (cell, band), cc.CUTOUT_HDUS)   # W1: fsync, size = header-implied (C-WRITE)
        check_counts(u, stage, rels, band, outcome, reads.in_view(u, cell, band, stage, rels, cubes))   # W2 (C-COUNT), before
                                                                                                         # C-NOCUT
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
        # 1. identity of every staged cutout with its cube(s), per Stokes (the regeneration's content test); it hands check (c)
        #    the cube's frames and check (w) each member cube
        from exopipe.phase3.lib.photometry import tests_fluxscale as TF
        finfo, wchecks = {}, {}
        for S in 'VI':
            files = [os.path.join(stage, r) for r in rels if r.endswith('_%s.fits' % S)]
            if not files:               # the engine's count for S is 0 (check_counts above held the files to it)
                continue
            if is_wideband(band):
                if S not in members:
                    u.refuse('C-IDENTITY: no wideband_members_%s.csv from the group search for %s %s (staging kept: %s)'
                             % (S, cell, band, stage))
                with open(members[S], newline='') as fh:
                    wchecks[S] = TF.WeightCheck([r for r in csv.DictReader(fh) if r['group'] == band])
                res_ = TC.identity_wideband(files, cell, band, S, members[S], member_hook=wchecks[S])
            else:
                res_ = TC.identity_subband(files, cell, band, S, frame_info=finfo.setdefault(S, {}))
            nbad = [(os.path.relpath(f, stage), d) for f, ok, _, d in res_ if not ok]
            worst = max([v for _, _, v, _ in res_ if v == v] or [float('nan')])
            u.check('photometry_cutout_manifest.I1_identity_%s' % S, not nbad, worst,
                    '%d of %d cutouts equal their %s within %g x rms (worst %.2e)%s' % (
                        len(files) - len(nbad), len(files), 'rebuilt wideband stack' if is_wideband(band) else 'cube',
                        cc.IDENTITY_TOL, worst, ('; FAIL %s' % TC._fmt(nbad)) if nbad else ''), cls='photometry_cutout_manifest')
        if u.failed_tests():
            u.refuse('C-IDENTITY: %s (staging kept: %s)' % ('; '.join(t['detail'] for t in u.failed_tests())[:600], stage))
        # 1b. the cubes unchanged since the gate and, for a subband, the staged flux-scale record (C-FLUXSCALE); the generation
        #     checks (a)(b)(c)(w) on the staged files (C-GENCHECK)
        fs_pairs, fs_outputs = fluxscale_recheck(u, cell, band, root, stage, fsbefore, regen_line, ex, emembers)
        generation_checks(u, cell, band, stage, rels, targets, planets, ex, fsbefore, finfo, wchecks, gen5)
        # 2-4. one file per key, the manifest and its content tests on the staged files, the placement transaction (with the
        #      flux-scale record and its content test FS1)
        extra = [(members[S], mrels[1 + 'VI'.index(S)]) for S in sorted(members)]   # beside the manifest: the member tables
        place_unit(u, run, root, stage, rels, '10s', cell, band, samples, mrels[0],   # a wideband is rebuilt from
                   {'cell': cell, 'band': band}, extra + fs_pairs, fs_outputs,
                   [fluxscale_in_place(u, cell, band, fsbefore, ex, emembers)])
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
    from exopipe.phase3.lib.serve import versions
    if not (getattr(args, 'src_run', None) or getattr(args, 'run_label', None)):
        return
    s = KD.src_run(args)                 # a versioned deep run: its work directory (versions.reads), on both sides
    if os.path.realpath(s) != os.path.realpath(versions.reads(_cc().TIER_DEEP_RUN)):
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
    samples = run_samples(targets, planets, tier)
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
    inputs = tier_inputs(tier, cell) + ([p for _, p in targets] or list(K.sample_paths(tier, cut=True).values()))
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
        settle_staged(u, stage, rels, '%s %s' % (token, cell))   # W1: fsync, size = header-implied (C-WRITE)
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
    """{root: [samples]} of the cutmeasure outputs: --tier-out for every sample, else photometry.TIER_FORCED_ROOTS (an ad-hoc tag
    has no served root: PHOT-ADHOC refuses it without --tier-out)."""
    ph = K.cfg().photometry
    out = {}
    for s in samples:
        if not getattr(args, 'tier_out', None) and s not in ph.TIER_FORCED_ROOTS:
            raise Refusal('PHOT-ADHOC: the ad-hoc set %s has no served tier forced root - cutmeasure of --targets writes its own '
                          'root (give --tier-out)' % s)
        r = os.path.abspath(args.tier_out) if getattr(args, 'tier_out', None) else ph.TIER_FORCED_ROOTS[s]
        out.setdefault(r, []).append(s)
    return out


def tier_engine_env(samples):
    """The process environment under which the tier engine measures the samples its SAMPLES literal does not name
    (photometry.TIER_FORCED_SAMPLES beyond TIER_FORCED_ENGINE_SAMPLES: LPT): each as an ad-hoc tag, {ADHOC_ENV:
    'TAG=<paths.CLV15_ROOT>/<SAMPLE_FILES[TAG]>,...'} - the engine measures a tag as a fixed target exactly as HWJ / 10 pc, from
    its cutouts under <cutout root>/<tag> (photometry_tier_forced_20261026.py:56-60); {} when every sample is the engine's."""
    c = K.cfg()
    ph = c.photometry
    extra = [s for s in samples if s not in ph.TIER_FORCED_ENGINE_SAMPLES]
    if not extra:
        return {}
    return {ph.ADHOC_ENV: ','.join('%s=%s' % (s, os.path.join(c.paths.CLV15_ROOT, ph.SAMPLE_FILES[s])) for s in extra)}


def stage_tier_cutmeasure(run, args):
    tier = args.tier
    ph = K.cfg().photometry
    token = _cc().TIER_TOKEN[tier]
    targets = K.adhoc_targets(args)
    root = K.cutout_root(args)
    try:
        # an ad-hoc tag is measured too: the tool takes the tags of photometry.ADHOC_ENV (K.adhoc_env, passed by
        # cutmeasure_unit) as --samples names (photometry_tier_forced_20261026.py ADHOC, exopipe_ops 2026-11-20)
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
    # --targets: the ad-hoc sets; else each unit's own samples, those the engine does not name (LPT) as ad-hoc tags (a unit
    # without such a sample, e.g. SolarSystem on its own root, gets none: P3_SNAP_CUTOUT_SAMPLES replaces the tool's SAMPLES)
    adhoc_env = K.adhoc_env(targets, bool(getattr(args, 'planets', False))) if targets else {}
    unit_env = (lambda smp: adhoc_env) if targets else tier_engine_env
    K.plan(run, '[cutmeasure] tier %s: LSTs %s on the cutouts of %s -> %s (photometry_tier_forced_20261026.py in place, '
                'site PYTHON[%s])%s' % (token, ','.join(lsts), root, '; '.join('%s: %s' % (','.join(s), r)
                                                                               for r, s in sorted(roots.items())),
                                        K.python_key(tier, 'cutmeasure_tier'),
                                        '; ad hoc: %s' % ' '.join('%s=%s' % kv for kv in sorted(adhoc_env.items()))
                                        if adhoc_env else ''))
    units = []
    for lst in lsts:
        for out_root, smp in sorted(roots.items()):
            env = unit_env(smp)
            if run.dry_run:
                K.plan(run, '[cutmeasure]   unit cutmeasure.%s.%s.%s -> %s%s' % (
                    token, lst, '+'.join(smp), out_root,
                    ('; ad hoc: %s' % ' '.join('%s=%s' % kv for kv in sorted(env.items()))) if env and not targets else ''))
                continue
            units += run_unit(run, 'cutmeasure.%s.%s.%s' % (token, lst, '+'.join(smp)),
                              dict(tier=tier, stage='cutmeasure', lst=lst, cutout_root=root, tier_out=out_root),
                              cutmeasure_unit, run, args, tier, lst, root, out_root, smp, env)
    return units


def cutmeasure_unit(run, args, tier, lst, root, out_root, samples, adhoc_env):
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
        partial = set(samples) != set(ph.TIER_FORCED_ENGINE_SAMPLES)   # the tool's own test (:198), against its SAMPLES
        if partial and os.path.exists(old_man):     # the tool merges a sample subset with the manifest in --out (:186-192)
            shutil.copy2(old_man, os.path.join(sdir, 'forced_manifest.csv'))
        argv = ['--root', root, '--out', stage, '--tier', token, '--lsts', lst, '--workers', '1', '--samples', ','.join(samples)]
        res, text = K.call(u, 'cutmeasure_tier', argv, tier, 'cutmeasure_tier', run, 'cutmeasure_%s_%s.log' % (token, lst),
                           env_extra=adhoc_env,
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
        kept = set(ph.TIER_FORCED_SAMPLES)
        if os.path.exists(staged_man):    # the rows the tool kept: the named samples and any earlier ad-hoc tag in this root
            with open(staged_man, newline='', encoding='utf-8') as f:
                kept |= {q[0] for q in (TC.parse_cutout(r['file']) for r in csv.DictReader(f)) if q}
        bad += staged_tests(u, 'photometry_tier_manifest', staged_man,
                            npz_dirs={s: odir for s in kept if s not in samples})
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
