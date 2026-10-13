"""`exopipe3 photometry --tier snapshot`: target photometry of the HWJ and 10 pc samples on the search's photometry cutouts
(10-s frames and 1-minute blocks), then the evaluation of every measured cell.

Stages (photometry.STAGES['snapshot']): measure (one unit per night cell) -> evaluate (one unit per band kind, then the zone
flags). Engines in place (core/legacy.run, site PYTHON['py36']): snapshot_forced_phot.py (one child per band) and
snapshot_forced_eval.py. Shell logic ported here from snapshot_forced_fleet.sh (claims -> exopipe3 run; pinning -> --cores;
band list :19-21; one process per band, PAR at once, node-local work dir :15, :33-34; 'no HWJ/10pc cutouts' is not a failure
:36-37, decided BEFORE launching with the engine's own glob; manifest merge in band order :38-40; completion rule :41-42;
copy to the product root :43 -> atomic per-file writes; done/failed markers -> run records; failed work kept) and from
snapshot_forced_reserve_20261007.sh / snapshot_forced_run_20260928.sh (per-kind refused-cell lists :138-147; the cells to
evaluate = measured minus refused; two evaluations :151-162 with OMP_NUM_THREADS=1; the 8-row check :161; the grid, zone and
stage-1 content tests :164, :169-173).

Units (records <out>/_records/photometry/): measure.<cell> (measure.<cell>.<band+band> for a band subset),
evaluate.<kind>.<tag>, evaluate.zones.<tag>. --out is the snapshot forced root (production /lustre/gh/main/phase3/snapshot_forced).
"""
import glob
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from exopipe.phase3.core import cells as C
from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.photometry import steps_common as K


def cutout_root(args):
    c = K.cfg()
    return os.path.abspath(args.cutout_root) if args.cutout_root else os.path.join(c.paths.PHASE3_ARCHIVE_ROOT,
                                                                                    c.photometry_measure.CUTOUT_SUBDIR)


def kind_bands(kind):
    b = K.cfg().photometry.SNAPSHOT_BANDS
    return tuple(x for x in b if ('MHz' in x) == (kind == 'sb'))


def _universe(root):
    out = []
    for d in sorted(glob.glob(os.path.join(root, '*_*'))):
        n = os.path.basename(d)
        try:
            lst, date, sb = C.parse_night_cell(n)
        except ValueError:
            continue
        if not sb and os.path.isdir(d):
            out.append(n)
    return out


def select_cells(args, universe_root):
    """[(cell, band subset or None)] from --cells / --epochs (night cells <LST>_<date>[_<band>]), minus --exclude-cells."""
    universe = _universe(universe_root) if os.path.isdir(universe_root) else []
    spec = ','.join(x for x in (args.cells, getattr(args, 'epochs', None)) if x)
    if spec:
        items = C.expand(spec, universe=universe, kind='night cell')
    else:
        items = list(universe)
    out = []
    for it in items:
        lst, date, sb = C.parse_night_cell(it)
        out.append(('%s_%s' % (lst, date), sb))
    if args.exclude_cells:
        ex = set(C.expand(args.exclude_cells, universe=universe, kind='night cell'))
        out = [(c, sb) for c, sb in out if c not in ex and ('%s_%s' % (c, sb) if sb else c) not in ex]
    return out, bool(spec)


def bands_for(args, sb):
    c = K.cfg()
    bands = list(c.photometry.SNAPSHOT_BANDS)
    if args.scopes:
        want = [s.strip() for s in args.scopes.split(',') if s.strip()]
        bands = [b for b in bands if b in want]
    if sb:
        bands = [b for b in bands if b == sb]
    return bands


def gate_full(args):
    """PHOT-FULL: snapshot Full is not a product (INCIDENT-SNAP-FULL-CUTOUTS-EQUAL-RED-62-CELLS)."""
    sc = {s.strip() for s in (args.scopes or '').split(',')} | {
        x.rsplit('_', 1)[-1] for x in (args.cells or '').split(',') if x.count('_') == 2}
    if 'Full' in sc:
        raise Refusal('PHOT-FULL: snapshot Full is not measured or evaluated (INCIDENT-SNAP-FULL-CUTOUTS-EQUAL-RED-62-CELLS; '
                      'the only Full cutouts were Red copies)')
    bad = [s for s in sc if s and s not in K.cfg().photometry.SNAPSHOT_BANDS]
    if args.scopes and bad:
        raise ValueError('--scopes %s not in photometry.SNAPSHOT_BANDS' % bad)


def cell_files(cell, root, index, band):
    """The engine's own selection (snapshot_forced_phot.cell_files, :107-114) for one band: the cutouts of the measured
    samples for this cell-hour."""
    c = K.cfg()
    lst, date = cell.split('_', 1)
    pat = '_%s_%sT' % (lst, date.replace('-', ''))
    smp = tuple(c.photometry.SAMPLES_MEASURED)
    if index and os.path.exists(index):
        out = [os.path.join(root, l.strip()) for l in open(index) if pat in l and l.split('/')[0] in smp]
        out = [p for p in out if os.path.relpath(p, root).split('/')[1] == band]
    else:
        out = [p for s in smp for p in glob.glob(os.path.join(root, s, band, '*', '*%s*_[IV].fits' % pat))]
    return sorted(out)


def listing_sha1(files):
    h = hashlib.sha1()
    for p in sorted(files):
        st = os.stat(p)
        h.update(('%s\t%d\t%d\n' % (p, st.st_size, st.st_mtime_ns)).encode())
    return h.hexdigest()


def unit_name(cell, bands):
    full = list(K.cfg().photometry.SNAPSHOT_BANDS)
    return 'measure.%s' % cell if list(bands) == full else 'measure.%s.%s' % (cell, '+'.join(bands))


# ---------------------------------------------------------------------------------------------------------------------
# stage measure (snapshot_forced_phot.py per band)
# ---------------------------------------------------------------------------------------------------------------------

def stage_measure(run, args):
    c = K.cfg()
    m = c.photometry_measure
    root = cutout_root(args)
    items, _ = select_cells(args, c.paths.SNAPSHOT_FLEET_OUT)
    stage1 = os.path.join(run.out_root, 'stage1')
    man_name = args.manifest_name or m.MANIFEST_NAME
    K.plan(run, '[measure] %d night cells; cutouts %s%s; -> %s/<cell>/forced_<sample>_<band>_<S>.npz + %s; band processes '
                'at once %d (snapshot_forced_phot.py in place, site PYTHON[%s])'
           % (len(items), root, ' (index %s)' % args.index if args.index else '', stage1, man_name,
              args.band_procs or m.BAND_PROCS['snapshot'], K.python_key('snapshot')))
    if not items:
        raise NoInput('no snapshot cell selected (--cells / --epochs; universe = cells under %s)' % c.paths.SNAPSHOT_FLEET_OUT)
    units = []
    for cell, sb in items:
        bands = bands_for(args, sb)
        files = {b: cell_files(cell, root, args.index, b) for b in bands}
        withc = [b for b in bands if files[b]]
        name = unit_name(cell, bands)
        if run.dry_run:
            K.plan(run, '[measure]   unit %s: %d bands, %d with cutouts (%d files); no cutouts in %s'
                   % (name, len(bands), len(withc), sum(len(v) for v in files.values()),
                      [b for b in bands if not files[b]] or 'none'))
            mp = os.path.join(stage1, cell, man_name)
            if list(bands) != list(c.photometry.SNAPSHOT_BANDS) and man_name == m.MANIFEST_NAME and os.path.exists(mp):
                K.plan(run, '[measure]   WOULD REFUSE SM-SUBSET: %s exists (full-band manifest); give --manifest-name' % mp)
            continue
        units += measure_cell(run, args, cell, bands, files, root, stage1, man_name)
    return units


def measure_cell(run, args, cell, bands, files, root, stage1, man_name):
    from exopipe.phase3.core import fitsio
    from exopipe.phase3.lib.photometry import tests as T
    c = K.cfg()
    m = c.photometry_measure
    name = unit_name(cell, bands)
    lst = cell.split('_', 1)[0]
    allf = [p for b in bands for p in files[b]]
    lsha = listing_sha1(allf) if allf else ''
    admitted = sorted('%s:%s' % (b, cell.split('_', 1)[1]) for b in bands if files[b])
    if K.run_resume_ok(run, name, admitted, {'cutout_listing_sha1': lsha}):
        return run.units[-1:]
    dst = os.path.join(stage1, cell)
    with run.unit(name, tier='snapshot', stage='measure', cell=cell, bands=bands, cutout_root=root,
                  manifest=man_name, cutout_listing_sha1=lsha, n_cutouts=len(allf)) as u:
        withc = [b for b in bands if files[b]]
        u.meta['bands_no_cutout'] = [b for b in bands if not files[b]]
        if not withc:
            raise NoInput('SM-NOCUT: no HWJ/10pc cutout for %s in %d bands under %s' % (cell, len(bands), root))
        full = list(bands) == list(c.photometry.SNAPSHOT_BANDS)
        mpath = os.path.join(dst, man_name)
        if not full and man_name == m.MANIFEST_NAME and os.path.exists(mpath):
            u.refuse('SM-SUBSET: a band subset would overwrite the full-band manifest %s; give --manifest-name' % mpath)
        for p in K.sample_paths().values():
            u.add_input(p)
        for b in withc:
            gp = os.path.join(c.catalog_snapshot.MASTER_CATALOGS, 'master_R075_%s_%s.gaul.csv' % (lst, b))
            if os.path.exists(gp):
                u.add_input(gp)
        u.set_admitted(admitted)
        # report-only content test of the INPUT cutouts (B-30 unanswered; search runs it when it writes them)
        for b in withc:
            r = T.c1(files[b][0])
            u.note('photometry_cutout C1-C4 (report only) %s: %s %s' % (os.path.relpath(files[b][0], root),
                                                                      'PASS' if r[0] else 'FAIL', r[2]))
        work = fitsio.scratch_dir('photometry_snap_%s' % cell, args.work)
        u.meta['work_dir'] = work
        procs = args.band_procs or m.BAND_PROCS['snapshot']
        out = {}

        def one(b):
            argv = [cell, os.path.join(work, 'b_%s' % b), '--bands', b, '--root', root] + (
                ['--index', os.path.abspath(args.index)] if args.index else [])
            res, text = K.call(u, 'measure_snapshot', argv, 'snapshot', 'snapshot_measure', run,
                               'measure_%s_%s.log' % (cell, b), threads=args.threads)
            out[b] = (res, text)
        with ThreadPoolExecutor(max_workers=max(1, procs)) as ex:
            list(ex.map(one, withc))
        bad, crashed = [], []
        for b in withc:
            res, text = out[b]
            if res.rc == 0:
                continue
            if 'REFUSING: no HWJ/10pc cutouts' in text:          # the fleet's 'not a failure' (:36-37)
                continue
            (bad if res.refused else crashed).append('%s: %s' % (b, res.reason))
        # merge the band manifests in band order, byte for byte as the fleet does (:38-40)
        body = b''
        for b in c.photometry.SNAPSHOT_BANDS:
            mp = os.path.join(work, 'b_%s' % b, cell, 'forced_manifest.csv')
            if b in withc and os.path.exists(mp):
                with open(mp, 'rb') as fh:
                    fh.readline()
                    body += fh.read()
        merged = (m.MANIFEST_HEADER + '\n').encode() + body
        os.makedirs(os.path.join(work, 'cell'), exist_ok=True)
        with open(os.path.join(work, 'cell', man_name), 'wb') as fh:
            fh.write(merged)
        lines = merged.decode('utf-8', 'replace').splitlines()
        nref = sum(1 for ln in lines if ',REFUSED' in ln)
        nok = sum(1 for ln in lines if ',ok,' in ln)
        npz = sorted(p for b in withc for p in glob.glob(os.path.join(work, 'b_%s' % b, cell, 'forced_*.npz')))
        u.meta.update(n_ok=nok, n_refused_rows=nref, n_npz=len(npz), bad_bands=bad, crashed_bands=crashed)
        if crashed:
            raise RuntimeError('SM-DONE: band process crashed (work kept: %s): %s' % (work, '; '.join(crashed)))
        if bad or nref or not nok:
            u.refuse('SM-DONE: %s%s%s (work kept: %s)' % ('bands refused %s; ' % bad if bad else '',
                                                         '%d REFUSED manifest rows; ' % nref if nref else '',
                                                         'no ok row' if not nok else '', work))
        # stale npz of the measured bands that this measurement did not write (they would be evaluated)
        mine = {os.path.basename(p) for p in npz}
        stale = sorted(p for p in glob.glob(os.path.join(dst, 'forced_*.npz'))
                       if os.path.basename(p) not in mine and os.path.basename(p).rsplit('_', 2)[-2] in withc)
        if stale:
            if not run.fresh:
                u.refuse('SM-STALE: %s holds %d npz of the measured bands that this measurement did not write (e.g. %s); '
                         'rerun with --fresh to replace them' % (dst, len(stale), os.path.basename(stale[0])))
            for p in stale:
                os.remove(p)
            u.meta['removed_stale'] = [os.path.basename(p) for p in stale]
        for p in npz:
            u.plan(os.path.join(dst, os.path.basename(p)), 'photometry_snapshot_npz')
        u.plan(mpath, 'photometry_manifest')
        for p in npz:
            K.atomic_copy(p, os.path.join(dst, os.path.basename(p)))
        K.atomic_copy(os.path.join(work, 'cell', man_name), mpath)
        for p in npz:
            u.add_output(os.path.join(dst, os.path.basename(p)), 'photometry_snapshot_npz', cutout_root=root)
        u.add_output(mpath, 'photometry_manifest')
        u.meta['bands_done'] = sorted(set(bands))
        if not u.failed_tests():
            fitsio.cleanup(work)
            u.meta['work_dir'] = None
    return run.units[-1:]


# ---------------------------------------------------------------------------------------------------------------------
# stage evaluate (snapshot_forced_eval.py per kind; zone flags)
# ---------------------------------------------------------------------------------------------------------------------

def refused_list(kind):
    """[(cell, reason)] refused before evaluation for this kind (photometry_evaluate.REFUSED_CELLS 'cell,kinds,reason':
    'sb+wb' rows refuse both kinds, 'wb' rows the widebands only; reserve :138-141)."""
    from exopipe.phase3.lib.photometry.tests import read_csv
    p = K.cfg().photometry_evaluate.REFUSED_CELLS['snapshot']
    out = []
    for r in read_csv(p)[1]:
        ks = r['kinds'].split('+')
        if kind in ks:
            out.append((r['cell'], r['reason']))
    return out


def _records_root(stage1):
    return os.path.dirname(os.path.abspath(stage1.rstrip('/')))


def done_evidence(stage1, cell, kind):
    """(ok, how) - the cell's stage 1 is complete for this kind: done measure records of the cell (under the stage-1 root's
    own records) covering the kind's bands; a stage-1 root without any record (the production fleet's) is accepted when
    the cell's manifests pass M1 + M2 (the fleet's completion rule, snapshot_forced_fleet.sh:41-42)."""
    from exopipe.phase3.lib.photometry import tests as T
    recs = glob.glob(os.path.join(_records_root(stage1), '_records', 'photometry', 'measure.%s*.json' % cell))
    recs = [json.load(open(p)) for p in recs]
    recs = [r for r in recs if r.get('unit') == 'measure.%s' % cell or r.get('unit', '').startswith('measure.%s.' % cell)]
    if recs:
        done = set()
        bad = []
        for r in recs:
            if r.get('status') == 'done':
                done |= set((r.get('meta') or {}).get('bands_done') or [])
            else:
                bad.append('%s %s' % (r['unit'], r.get('status')))
        need = set(kind_bands(kind))
        if need <= done:
            return True, 'measure record(s) done for every %s band' % kind
        return False, 'measure records cover %s of the %s bands%s' % (sorted(done & need), kind,
                                                                      '; not done: %s' % bad if bad else '')
    mans = glob.glob(os.path.join(stage1, cell, 'forced_manifest*.csv'))
    if not mans:
        return False, 'no measure record and no stage-1 manifest'
    for mp in mans:
        for fn in (T.mm1, T.mm2):
            ok, _, det = fn(mp)
            if not ok:
                return False, '%s: %s' % (os.path.basename(mp), det)
    return True, 'manifest content test (stage-1 root without run records)'


def stage_evaluate(run, args):
    c = K.cfg()
    ev = c.photometry_evaluate
    stage1 = os.path.abspath(args.stage1) if args.stage1 else os.path.join(run.out_root, 'stage1')
    kinds = [k.strip() for k in (args.kinds or ','.join(ev.KINDS['snapshot'])).split(',') if k.strip()]
    badk = [k for k in kinds if k not in ev.KINDS['snapshot']]
    if badk:
        raise ValueError('--kinds %s not in %s' % (badk, ev.KINDS['snapshot']))
    tag = args.eval_tag or time.strftime('%Y%m%d', time.gmtime())
    items, explicit = select_cells(args, stage1)
    cells = sorted({cl for cl, _ in items})
    K.plan(run, '[evaluate] stage 1 %s: %d cells; kinds %s; tag %s -> %s (snapshot_forced_eval.py in place, --nproc %d, '
                '--survey-cells %d)' % (stage1, len(cells), kinds, tag,
                                        ', '.join(os.path.join(run.out_root, ev.EVAL_DIR[k].format(tag=tag)) for k in kinds),
                                        args.eval_procs or ev.EVAL_PROCS['snapshot'],
                                        args.survey_cells or ev.SURVEY_CELLS['snapshot']))
    if not cells:
        msg = 'SE-NONE: no stage-1 cell under %s' % stage1
        if run.dry_run:
            K.plan(run, '[evaluate]   WOULD REFUSE ' + msg)
            return []
        raise Refusal(msg)
    units = []
    done_kinds = []
    for kind in kinds:
        ref = dict(refused_list(kind))
        rr = _measure_refusals(stage1, cells)
        for cl, why in rr.items():
            ref.setdefault(cl, why)
        listed = sorted(set(cells) & set(ref))
        if explicit and listed:
            msg = 'SE-LISTED: cells both selected and refused for %s: %s' % (kind, listed)
            if run.dry_run:
                K.plan(run, '[evaluate]   WOULD REFUSE ' + msg)
                continue
            raise Refusal(msg)
        ecells = [cl for cl in cells if cl not in ref]
        evid = {cl: done_evidence(stage1, cl, kind) for cl in ecells}
        nodone = sorted(cl for cl, (ok, _) in evid.items() if not ok)
        evdir = os.path.join(run.out_root, ev.EVAL_DIR[kind].format(tag=tag))
        name = 'evaluate.%s.%s' % (kind, tag)
        K.plan(run, '[evaluate]   unit %s: %d cells (%d refused before evaluation: %s) -> %s'
               % (name, len(ecells), len(listed), listed[:4], evdir))
        if nodone:
            msg = 'SE-DONE: %d cells without a complete stage 1 for %s (e.g. %s: %s)' % (
                len(nodone), kind, nodone[0], evid[nodone[0]][1])
            if run.dry_run:
                K.plan(run, '[evaluate]   WOULD REFUSE ' + msg)
                continue
            raise Refusal(msg)
        if run.dry_run:
            continue
        admitted = _eval_admitted(ecells, kind)
        if K.run_resume_ok(run, name, admitted, {}):
            units += run.units[-1:]
            done_kinds.append((kind, evdir))
            continue
        with run.unit(name, tier='snapshot', stage='evaluate', kind=kind, tag=tag, stage1=stage1,
                      cells=len(ecells)) as u:
            if not ecells:
                raise NoInput('SE-NONE: every selected cell is refused for %s' % kind)
            rf = os.path.join(run.out_root, ev.REFUSED_FILE.format(kind=kind, tag=tag))
            K.write_text_atomic(rf, 'cell,reason\n' + ''.join('%s,%s\n' % (cl, ref[cl].replace(',', ';'))
                                                               for cl in sorted(ref)))
            u.add_input(rf)
            u.add_input(ev.REFUSED_CELLS['snapshot'])
            u.add_input(ev.DEEP_SEARCH_PIN['snapshot'])
            u.add_input(c.paths.DATA_DECISIONS)
            for cl in ecells:
                u.add_input(os.path.join(stage1, cl))
                for mp in sorted(glob.glob(os.path.join(stage1, cl, 'forced_manifest*.csv'))):
                    u.add_input(mp)
            u.set_admitted(admitted)
            u.meta['evidence'] = {cl: evid[cl][1] for cl in ecells[:5]}
            u.plan(evdir, 'photometry_eval')
            argv = [stage1, evdir, '--kind', kind, '--nproc', args.eval_procs or ev.EVAL_PROCS['snapshot'],
                    '--survey-cells', args.survey_cells or ev.SURVEY_CELLS['snapshot'], '--cells', ','.join(ecells),
                    '--refused-cells', rf]
            res, text = K.call(u, 'evaluate_snapshot', argv, 'snapshot', 'snapshot_evaluate', run,
                               'evaluate_%s_%s.log' % (kind, tag), threads=args.threads)
            if res.refused:
                u.refuse('snapshot_forced_eval refused (%s)' % res.reason)
            if res.rc != 0:
                raise RuntimeError('snapshot_forced_eval.py exit %d: %s' % (res.rc, res.tail[-800:]))
            u.add_output(evdir, 'photometry_eval', stage1=stage1)
        units += run.units[-1:]
        if units[-1].status == 'done':
            done_kinds.append((kind, evdir))
    if not run.dry_run and done_kinds and len(done_kinds) == len(kinds):
        units += zone_flags(run, args, stage1, tag, done_kinds)
    return units


def _measure_refusals(stage1, cells):
    """{cell: reason} of cells whose measure record (under the stage-1 root's own records) is refused (reserve :142-144)."""
    out = {}
    for cl in cells:
        p = os.path.join(_records_root(stage1), '_records', 'photometry', 'measure.%s.json' % cl)
        if os.path.exists(p):
            r = json.load(open(p))
            if r.get('status') == 'refused':
                out[cl] = 'stage 1 refused in this run (%s)' % (r.get('reason') or '')[:160]
    return out


def _eval_admitted(cells, kind):
    from exopipe.phase3.core import exclusions
    ex = exclusions.load()
    return sorted('%s:%s' % (cl, b) for cl in cells for b in kind_bands(kind)
                  if not ex.is_excluded(cl[:3], b, cl[4:]))


def zone_flags(run, args, stage1, tag, done_kinds):
    """The Z1 recomputation of the evaluated kinds written as the production flags file (reserve :159 --write); its rows are
    what Z1 compared, so a failing Z1 makes this unit refuse as well."""
    from exopipe.phase3.lib.photometry import tests as T
    ev = K.cfg().photometry_evaluate
    name = 'evaluate.zones.%s' % tag
    path = os.path.join(run.out_root, ev.ZONE_FLAGS_FILE.format(tag=tag))
    with run.unit(name, tier='snapshot', stage='evaluate', tag=tag) as u:
        rows = []
        for kind, evdir in done_kinds:
            p = os.path.join(evdir, 'targets_ge4.csv')
            if os.path.exists(p):
                u.add_input(p)
            r, _, _ = T.zone_rows(evdir, stage1, None, ev.CONTENT_TEST_PROCS)
            rows.extend(r)
        u.plan(path, 'photometry_zone_flags')
        T.write_zone_flags(path, rows)
        u.add_output(path, 'photometry_zone_flags')
    return run.units[-1:]


STAGE_FUNCS = {'measure': stage_measure, 'evaluate': stage_evaluate}


def run_tier(run, args, stages):
    gate_full(args)
    for st in stages:
        units = STAGE_FUNCS[st](run, args)
        if run.dry_run:
            continue
        if units and st != 'measure' and not K.stage_ok(units):
            break
    return None

