"""`exopipe3 photometry --tier snapshot`: target photometry of the snapshot samples (photometry.SAMPLES_MEASURED['snapshot']:
HWJ, 10 pc, LPT) on the photometry cutouts of the served generation (photometry.CUTOUT_ROOT; 10-s frames and 1-minute
blocks), then the evaluation of every measured cell.

Stages (photometry.STAGES['snapshot']): cutouts (only when named: the 10-s cutouts regenerated from the consolidated flow
cubes into a generation root, lib/photometry/steps_cutouts.py) -> measure (one unit per night cell; stage 1 written to
<out>/photometry.STAGE1_DIR) -> evaluate (one unit per band kind, then the zone flags). An ad-hoc target set (--targets
TAG=CSV) is cut and measured, never evaluated. Engines in place (core/legacy.run, site PYTHON['py36']):
snapshot_forced_phot.py (one child per band) and snapshot_forced_eval.py. Shell logic ported here from
snapshot_forced_fleet.sh (claims -> exopipe3 run; pinning -> --cores; band list :19-21; one process per band, PAR at once,
node-local work dir :15, :33-34; 'no HWJ/10pc cutouts' is not a failure :36-37, decided BEFORE launching with the engine's own
glob; manifest merge in band order :38-40; completion rule :41-42; copy to the product root :43 -> atomic per-file writes;
done/failed markers -> run records; failed work kept) and from snapshot_forced_reserve_20261007.sh /
snapshot_forced_run_20260928.sh (per-kind refused-cell lists :138-147; the cells to evaluate = measured minus refused; two
evaluations :151-162 with OMP_NUM_THREADS=1; the 8-row check :161, now one row per (sample, Stokes, tier) of the samples
measured (V2); the grid, zone and stage-1 content tests :164, :169-173). A campaign's stage 1 (lib/photometry/campaign.py) is
evaluated per band kind: a cell without the kind's cutouts or held by a gap is listed refused, a held one refuses SE-DONE.

Units (records <out>/_records/photometry/): cutouts.<cell>.<band>[.<tags>] (steps_cutouts), measure.<cell>
(measure.<cell>.<band+band> for a band subset), evaluate.<kind>.<tag>, evaluate.zones.<tag>. --out is the snapshot forced root (production /lustre/gh/main/phase3/snapshot_forced).
"""
import csv
import glob
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from exopipe.phase3.core import cells as C
from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.photometry import steps_common as K


def cutout_root(args):
    """--cutout-root, else the served generation's cutouts (photometry.CUTOUT_ROOT; generation 4's PHASE3_ARCHIVE_ROOT/
    photometry is retired)."""
    return K.cutout_root(args)


def stage1_root(run, args=None):
    """Stage 1 of this run: <out>/photometry.STAGE1_DIR (production <OUT_ROOT[snapshot]>/stage1_gen5 since generation 5); a
    campaign's: <out>/<its generation's directory> (photometry_campaign.STAGE1_DIR_FMT)."""
    if getattr(args, 'campaign', False):
        from exopipe.phase3.lib.photometry import campaign
        return os.path.join(run.out_root, campaign.stage1_dir())
    return os.path.join(run.out_root, K.cfg().photometry.STAGE1_DIR)


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


def cell_files(cell, root, index, band, samples=None):
    """The engine's own selection (snapshot_forced_phot.cell_files, :218-225) for one band: the cutouts of the measured
    samples (or an ad-hoc target set's tags) for this cell-hour."""
    lst, date = cell.split('_', 1)
    pat = '_%s_%sT' % (lst, date.replace('-', ''))
    smp = tuple(samples or K.samples_measured('snapshot'))
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


def quarantine(run, cell, paths, reason):
    """Move superseded stage-1 files of ``cell`` out of the product directory, never delete them (CLAUDE.md 7; the
    production re-serve did the same, snapshot_forced_reserve_20261007.sh:104-113): to
    <out>/quarantine_superseded_photometry_<run id>/<cell>/, each with a row of MANIFEST_sha256.csv
    (sha256,bytes,moved_from,moved_to,reason), the checksum verified after the move. Returns the moved paths."""
    q = os.path.join(run.out_root, 'quarantine_superseded_photometry_%s' % run.run_id)
    man = os.path.join(q, 'MANIFEST_sha256.csv')
    os.makedirs(os.path.join(q, cell), exist_ok=True)
    moved = []
    new = not os.path.exists(man)
    with open(man, 'a') as mf:
        if new:
            mf.write('sha256,bytes,moved_from,moved_to,reason\n')
        for p in sorted(paths):
            h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
            dst = os.path.join(q, cell, os.path.basename(p))
            os.replace(p, dst)
            if hashlib.sha256(open(dst, 'rb').read()).hexdigest() != h:
                raise Refusal('SM-STALE: quarantine checksum of %s differs after the move' % dst)
            mf.write('%s,%d,%s,%s,%s\n' % (h, os.path.getsize(dst), p, dst, reason.replace(',', ';')))
            moved.append(dst)
    return moved


def unit_name(cell, bands):
    full = list(K.cfg().photometry.SNAPSHOT_BANDS)
    return 'measure.%s' % cell if list(bands) == full else 'measure.%s.%s' % (cell, '+'.join(bands))


# ---------------------------------------------------------------------------------------------------------------------
# stage measure (snapshot_forced_phot.py per band)
# ---------------------------------------------------------------------------------------------------------------------

def engine_samples(path):
    """The SAMPLES default of the snapshot measure engine at ``path``, read from its source without running it (as
    tests/test_photometry.py test_mirrors reads it): the module-level SAMPLES literal, or the default branch of its ad-hoc
    switch (`<P3_SNAP_CUTOUT_SAMPLES tags> if os.environ.get(...) else <tuple>`, 2026-10-28). None without such a literal."""
    import ast
    try:
        body = ast.parse(open(path).read()).body
    except (OSError, SyntaxError, ValueError):
        return None
    val = None
    for node in body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SAMPLES' for t in node.targets):
            v = node.value.orelse if isinstance(node.value, ast.IfExp) else node.value
            try:
                val = ast.literal_eval(v)
            except ValueError:
                val = None
    return tuple(val) if isinstance(val, (tuple, list)) and val and all(isinstance(x, str) for x in val) else None


def gate_engine_samples():
    """PHOT-ENGINE-SAMPLES: the snapshot measure engine the run executes (K.engine('measure_snapshot'): the one hit among the
    site LEGACY_CODE_ROOTS) must know every sample the step names to it (--samples = photometry.SAMPLES_MEASURED['snapshot']).
    An older engine refuses the flag in every band (REFUSING: --samples ...), which the step would record as a measure failure
    of each cell; this refuses once, before any cell runs."""
    p = K.engine('measure_snapshot')
    got = engine_samples(p)
    want = K.samples_measured('snapshot')
    if got is None:
        raise Refusal('PHOT-ENGINE-SAMPLES: engine %s has no readable SAMPLES literal (a tuple, or the default of its '
                      'P3_SNAP_CUTOUT_SAMPLES switch)' % p)
    lack = [t for t in want if t not in got]
    if lack:
        raise Refusal("PHOT-ENGINE-SAMPLES: engine %s SAMPLES %s lacks %s (photometry.SAMPLES_MEASURED['snapshot'] = %s)"
                      % (p, ','.join(got), ','.join(lack), ','.join(want)))
    return got


def stage_measure_campaign(run, args):
    """A campaign's stage 1 (lib/photometry/campaign.py): per cell and band kind, measured once every cutouts unit of the kind
    is terminal; a held kind holds only its cell (SM-HELD), a kind without any cutout ends 'no <kind>' (SM-NOBAND)."""
    from exopipe.phase3.lib.photometry import campaign as CP
    camp = CP.start(run, args)
    root, stage1 = cutout_root(args), stage1_root(run, args)
    K.plan(run, '[measure] campaign %s: %d cells, per band kind (subbands -> <cell>/%s, widebands -> <cell>/%s) under %s; '
                'cutouts %s' % (camp.generation, len(camp.enum.cells), K.cfg().photometry_campaign.STAGE1_MANIFEST['sb'],
                                K.cfg().photometry_campaign.STAGE1_MANIFEST['wb'], stage1, root))
    units = []
    for cell in camp.enum.cells:
        for kind in ('sb', 'wb'):
            units += CP.stage1_cell(run, args, camp, cell, kind, root, stage1)
    return units


def manifest_bands(path):
    """The bands of a stage-1 manifest's rows (the directory above each cutout's object directory: <sample>/<band>/<object>/),
    read with the manifest tests' own reader (tests._manifest_rows, as MM1 reads it). None when the manifest cannot be parsed
    for its bands - unreadable, a header other than photometry_measure.MANIFEST_HEADER, a row whose file names no snapshot
    band (photometry.SNAPSHOT_BANDS) at that place: the caller refuses SM-SUBSET, never reads it as an empty set."""
    from exopipe.phase3.lib.photometry import tests as T
    c = K.cfg()
    try:
        cols, rows = T._manifest_rows(path)
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    if cols != c.photometry_measure.MANIFEST_HEADER.split(','):
        return None
    known, out = set(c.photometry.SNAPSHOT_BANDS), set()
    for r in rows:
        parts = (r.get('file') or '').split('/')
        if len(parts) < 4 or parts[-3] not in known:
            return None
        out.add(parts[-3])
    return out


def stage_measure(run, args):
    if getattr(args, 'campaign', False):
        return stage_measure_campaign(run, args)
    c = K.cfg()
    m = c.photometry_measure
    root = cutout_root(args)
    targets = K.adhoc_targets(args)
    samples = [t for t, _ in targets] or list(K.samples_measured('snapshot'))
    items, _ = select_cells(args, c.paths.SNAPSHOT_FLEET_OUT)
    stage1 = stage1_root(run)
    man_name = args.manifest_name or m.MANIFEST_NAME
    K.plan(run, '[measure] %d night cells; cutouts %s%s; samples %s%s; -> %s/<cell>/forced_<sample>_<band>_<S>.npz + %s; band '
                'processes at once %d (snapshot_forced_phot.py in place, site PYTHON[%s])'
           % (len(items), root, ' (index %s)' % args.index if args.index else '', ','.join(samples),
              ' (ad hoc: %s)' % ', '.join('%s=%s' % t for t in targets) if targets else '', stage1, man_name,
              args.band_procs or m.BAND_PROCS['snapshot'], K.python_key('snapshot')))
    try:
        K.gate_adhoc_roots(targets, {'stage-1 root': stage1})
        if not targets:             # an ad-hoc set: the engine takes its tags from P3_SNAP_CUTOUT_SAMPLES (K.adhoc_env)
            gate_engine_samples()
    except Refusal as e:
        if run.dry_run:
            K.would_refuse(run, 'measure', str(e))
            return []
        raise
    if not items:
        raise NoInput('no snapshot cell selected (--cells / --epochs; universe = cells under %s)' % c.paths.SNAPSHOT_FLEET_OUT)
    units = []
    for cell, sb in items:
        bands = bands_for(args, sb)
        files = {b: cell_files(cell, root, args.index, b, samples) for b in bands}
        withc = [b for b in bands if files[b]]
        name = unit_name(cell, bands)
        if run.dry_run:
            K.plan(run, '[measure]   unit %s: %d bands, %d with cutouts (%d files); no cutouts in %s'
                   % (name, len(bands), len(withc), sum(len(v) for v in files.values()),
                      [b for b in bands if not files[b]] or 'none'))
            mp = os.path.join(stage1, cell, man_name)
            if list(bands) != list(c.photometry.SNAPSHOT_BANDS) and man_name == m.MANIFEST_NAME and os.path.exists(mp):
                K.would_refuse(run, 'measure', 'SM-SUBSET: %s exists (full-band manifest); give --manifest-name' % mp)
            continue
        units += measure_cell(run, args, cell, bands, files, root, stage1, man_name, targets)
    return units


def measure_cell(run, args, cell, bands, files, root, stage1, man_name, targets=(), kind_manifest=False, gate=None):
    """One night cell's stage 1 for ``bands``. ``kind_manifest``: ``man_name`` is a campaign's manifest of one band kind
    (photometry_campaign.STAGE1_MANIFEST): rewriting it is no band subset over a full-band manifest unless it holds rows of
    another band (SM-SUBSET). ``gate``: called with the unit first (a campaign's pin, space and I/O guard)."""
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
    tspec = ['%s=%s' % t for t in targets]
    with run.unit(name, tier='snapshot', stage='measure', cell=cell, bands=bands, cutout_root=root,
                  manifest=man_name, cutout_listing_sha1=lsha, n_cutouts=len(allf), targets=tspec) as u:
        if gate is not None:
            gate(u)
        withc = [b for b in bands if files[b]]
        u.meta['bands_no_cutout'] = [b for b in bands if not files[b]]
        if not withc:
            raise NoInput('SM-NOCUT: no %s cutout for %s in %d bands under %s' % (
                '/'.join(t for t, _ in targets) or '/'.join(K.samples_measured('snapshot')), cell, len(bands), root))
        full = list(bands) == list(c.photometry.SNAPSHOT_BANDS)
        mpath = os.path.join(dst, man_name)
        if not full and man_name == m.MANIFEST_NAME and os.path.exists(mpath):
            mb = manifest_bands(mpath) if kind_manifest else set()
            if kind_manifest and mb is None:
                u.refuse('SM-SUBSET: the manifest %s cannot be parsed for its bands (unreadable, another header than '
                         'photometry_measure.MANIFEST_HEADER, or a row naming no snapshot band) - a band subset would '
                         'overwrite it; nothing measured' % mpath)
            if not (kind_manifest and mb <= set(bands)):
                u.refuse('SM-SUBSET: a band subset would overwrite the full-band manifest %s; give --manifest-name' % mpath)
        for p in ([p for _, p in targets] or K.sample_paths('snapshot').values()):
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
        work = fitsio.scratch_dir('photometry_snap_%s' % cell,
                                  os.path.join(fitsio.scratch_root(args.work), m.WORK_SUBDIR['snapshot']))
        u.meta['work_dir'] = work
        procs = args.band_procs or m.BAND_PROCS['snapshot']
        out = {}

        def one(b):
            # an ad-hoc target set: the engine reads it from P3_SNAP_CUTOUT_SAMPLES and measures its tags (--samples TAG,
            # forced_adhoc_20261028/stage1_when_done_20261028.sh:10-11); else the snapshot samples of config, named (the
            # engine's default since 2026-11-18, mirror-tested; an engine without one of them: PHOT-ENGINE-SAMPLES above)
            argv = [cell, os.path.join(work, 'b_%s' % b), '--bands', b, '--root', root] + (
                ['--index', os.path.abspath(args.index)] if args.index else []) + (
                ['--samples', ','.join(t for t, _ in targets) or ','.join(K.samples_measured('snapshot'))])
            res, text = K.call(u, 'measure_snapshot', argv, 'snapshot', 'snapshot_measure', run,
                               'measure_%s_%s.log' % (cell, b), env_extra=K.adhoc_env(targets), threads=args.threads)
            out[b] = (res, text)
        with ThreadPoolExecutor(max_workers=max(1, procs)) as ex:
            list(ex.map(one, withc))
        bad, crashed = [], []
        for b in withc:
            res, text = out[b]
            if res.rc == 0:
                continue
            if re.search(r'REFUSING: no \S+ cutouts for ', text):   # the fleet's 'not a failure' (:36-37)
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
                         'rerun with --fresh to move them to a quarantine directory' % (dst, len(stale),
                                                                                       os.path.basename(stale[0])))
            u.meta['quarantined_stale'] = quarantine(run, cell, stale, 'superseded: not rewritten by the stage-1 '
                                                     'measurement of %s (bands %s)' % (cell, '+'.join(withc)))
        for p in npz:
            u.plan(os.path.join(dst, os.path.basename(p)), 'photometry_snapshot_npz')
        u.plan(mpath, 'photometry_manifest')
        for p in npz:
            K.atomic_copy(p, os.path.join(dst, os.path.basename(p)))
        K.atomic_copy(os.path.join(work, 'cell', man_name), mpath)
        for p in npz:
            u.add_output(os.path.join(dst, os.path.basename(p)), 'photometry_snapshot_npz', cutout_root=root,
                         **({'targets': tspec} if tspec else {}))
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


def refused_given(args):
    """{kind: path} of --refused-cells KIND=FILE: the per-kind 'cell,reason' list of cells refused before evaluation that
    a production evaluation used (snapshot_forced_reserve_20261007.sh:138-147 wrote refused_cells_<kind>_<tag>.csv) - a
    replay passes it so the engine's refused_cells.csv and drops.csv are the production ones."""
    kinds = K.cfg().photometry_evaluate.KINDS['snapshot']
    out = {}
    for item in getattr(args, 'refused_cells', None) or []:
        k, sep, p = item.partition('=')
        if not sep or k.strip() not in kinds or not p.strip():
            raise ValueError('--refused-cells %r: expected KIND=FILE with KIND in %s' % (item, kinds))
        out[k.strip()] = os.path.abspath(p.strip())
    return out


def pin_generation_root():
    """The cutout tree whose corrupt files the pinned refused-cell list names: generation 4, the FIRST entry of
    photometry.RETIRED_CUTOUT_ROOTS (later retired generations are appended after it; tests/test_photometry.py checks the
    pinned cells' generation-4 manifests lie under it)."""
    return K.cfg().photometry.RETIRED_CUTOUT_ROOTS[0]


def _cutout_base(path):
    """<root> of a cutout <root>/<sample>/<band>/<object>/<file> (phase3_snapshot_cutouts.write_cutouts layout)."""
    for _ in range(4):
        path = os.path.dirname(path)
    return path


def pin_lifted(stage1, cells, pending=(), planned_root=None):
    """{cell} of the given pinned cells whose pin does not apply in this stage 1, decided per cell from its OWN stage-1
    manifests (every forced_manifest*.csv in <stage1>/<cell>, as the evaluation's done evidence reads them): lifted when at
    least one manifest exists and none of its files lies under pin_generation_root() (each file's cutout root compared after
    os.path.realpath, so a symlinked or non-normalized path to generation 4 still counts). A cell with no manifest keeps the
    pin, unless it is in ``pending`` (a dry run whose measure stage writes it) and ``planned_root`` (that measure's cutout
    root) is outside generation 4."""
    from exopipe.phase3.lib.photometry.tests import read_csv
    g4 = os.path.normpath(pin_generation_root())
    g4r = os.path.realpath(g4)
    real = {}

    def under_g4(path):
        base = os.path.normpath(_cutout_base(path))
        if base not in real:
            real[base] = os.path.realpath(base)
        return any(x == r or x.startswith(r + os.sep) for x in (base, real[base]) for r in (g4, g4r))

    out = set()
    for cl in cells:
        mans = sorted(glob.glob(os.path.join(stage1, cl, 'forced_manifest*.csv')))
        if not mans:
            if cl in pending and planned_root and not under_g4(os.path.join(planned_root, 's', 'b', 'o', 'f')):
                out.add(cl)
            continue
        files = [r.get('file') or '' for mp in mans for r in read_csv(mp)[1]]
        if files and all(f and not under_g4(f) for f in files):
            out.add(cl)
    return out


def refused_for(kind, given, stage1, cells, lifted=None):
    """(OrderedDict cell -> reason, path of a given list or None) of the cells refused before the evaluation of ``kind``.
    Default: the pinned file's rows for the kind in file order (reserve :140-141), then the cells whose stage-1 records refuse
    them (:142-144; a campaign's units of the kind as _measure_refusals lists them). A given list (--refused-cells) is read
    as written and must hold every pinned cell of the kind (the pinned cells are never re-admitted,
    DECISION-FORCED-CORRUPT-CUTOUT-CELLS-REFUSED); the measure refusals are added after its rows.
    The pinned file names corrupt GENERATION-4 archive cutouts (written 2026-09-18..22 under the retired root). Generation 5
    re-cut every cutout from the cubes, and production's generation-5 evaluation (snapshot_forced_gen5_stages_20261026.sh)
    reads no pinned file: it evaluated 04h 2025-01-06, 09h 2025-01-01 and 08h 2025-01-09. So a pinned cell whose own
    stage-1 manifests name no generation-4 file (pin_lifted; ``lifted`` = its precomputed result) is neither added nor
    required, and a given list must hold the remaining pinned cells only where they are among ``cells`` (an unselected cell
    is not evaluated, so it cannot be re-admitted)."""
    from collections import OrderedDict
    from exopipe.phase3.lib.photometry.tests import read_csv
    rows = refused_list(kind)
    if lifted is None:
        lifted = pin_lifted(stage1, sorted({cl for cl, _ in rows}))
    pinned = [(cl, why) for cl, why in rows if cl not in lifted]
    ref = OrderedDict()
    path = given.get(kind)
    if path:
        if not os.path.isfile(path):
            raise Refusal('SE-LISTED: --refused-cells %s=%s: no such file' % (kind, path))
        cols, rows = read_csv(path)
        if cols[:2] != ['cell', 'reason']:
            raise Refusal('SE-LISTED: --refused-cells %s=%s: header %s, expected cell,reason' % (kind, path, cols))
        for r in rows:
            ref.setdefault(r['cell'], r['reason'])
        sel = set(cells)
        miss = [cl for cl, _ in pinned if cl not in ref and cl in sel]
        if miss:
            raise Refusal('SE-LISTED: --refused-cells %s=%s omits the pinned refused cell(s) %s of %s '
                          '(DECISION-FORCED-CORRUPT-CUTOUT-CELLS-REFUSED: never re-admitted)'
                          % (kind, path, miss, K.cfg().photometry_evaluate.REFUSED_CELLS['snapshot']))
    else:
        for cl, why in pinned:
            ref.setdefault(cl, why)
    for cl, why in _measure_refusals(stage1, cells, kind).items():
        ref.setdefault(cl, why)
    return ref, (path if path and len(ref) == len(rows) else None)


def refused_text(ref):
    """The per-kind list the engine reads (--refused-cells), as the reserve script wrote it: 'cell,reason' + one
    '<cell>,<reason>' line per cell (:138-147; a comma in a reason would split it there, so it is written as ';')."""
    return 'cell,reason\n' + ''.join('%s,%s\n' % (cl, why.replace(',', ';')) for cl, why in ref.items())


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
    if getattr(args, 'targets', None):
        raise Refusal('PHOT-ADHOC: the evaluation is defined on the snapshot samples %s (photometry.SAMPLES_MEASURED; '
                      'thresholds from their controls, survey size); an ad-hoc target set is measured, not evaluated '
                      '(production: adhoc_report_20261028.py)' % ', '.join(K.samples_measured('snapshot')))
    stage1 = os.path.abspath(args.stage1) if args.stage1 else stage1_root(run)
    kinds = [k.strip() for k in (args.kinds or ','.join(ev.KINDS['snapshot'])).split(',') if k.strip()]
    badk = [k for k in kinds if k not in ev.KINDS['snapshot']]
    if badk:
        raise ValueError('--kinds %s not in %s' % (badk, ev.KINDS['snapshot']))
    tag = args.eval_tag or time.strftime('%Y%m%d', time.gmtime())
    items, explicit = select_cells(args, stage1)
    cells = sorted({cl for cl, _ in items})
    pending = set()      # dry run: cells whose stage 1 the measure stage of THIS invocation writes (nothing on disk yet)
    if run.dry_run and K.earlier_in_this_run(args, 'measure', 'evaluate') \
            and os.path.abspath(stage1) == os.path.abspath(stage1_root(run)):
        mitems, _ = select_cells(args, c.paths.SNAPSHOT_FLEET_OUT)
        pending = {cl for cl, _ in mitems}
        if not explicit:
            cells = sorted(set(cells) | pending)
    K.plan(run, '[evaluate] stage 1 %s: %d cells; kinds %s; tag %s -> %s (snapshot_forced_eval.py in place, --nproc %d, '
                '--survey-cells %d)' % (stage1, len(cells), kinds, tag,
                                        ', '.join(os.path.join(run.out_root, ev.EVAL_DIR[k].format(tag=tag)) for k in kinds),
                                        args.eval_procs or ev.EVAL_PROCS['snapshot'],
                                        args.survey_cells or ev.SURVEY_CELLS['snapshot']))
    if pending:
        K.plan(run, '[evaluate]   stage 1 of %d cells from the measure stage of this run (their measure records are read '
                    'after measure finishes; a refused cell is listed refused, an errored one refuses SE-DONE)'
               % len(pending & set(cells)))
    if not cells:
        msg = 'SE-NONE: no stage-1 cell under %s' % stage1
        if run.dry_run:
            K.would_refuse(run, 'evaluate', msg)
            return []
        raise Refusal(msg)
    units = []
    done_kinds = []
    given = refused_given(args)
    pin_cells = sorted({cl for k in ev.KINDS['snapshot'] for cl, _ in refused_list(k)})
    lifted = pin_lifted(stage1, pin_cells, pending, cutout_root(args) if pending else None)
    if lifted:
        K.plan(run, '[evaluate]   pinned refused cells measured here without generation-4 cutouts, evaluated: %s'
               % sorted(lifted))
    for kind in kinds:
        try:
            ref, ref_path = refused_for(kind, given, stage1, cells, lifted)
        except Refusal as e:
            if run.dry_run:
                K.would_refuse(run, 'evaluate', str(e))
                continue
            raise
        listed = sorted(set(cells) & set(ref))
        if explicit and listed:
            msg = 'SE-LISTED: cells both selected and refused for %s: %s' % (kind, listed)
            if run.dry_run:
                K.would_refuse(run, 'evaluate', msg)
                continue
            raise Refusal(msg)
        ecells = [cl for cl in cells if cl not in ref]
        evid = {cl: (True, 'measure stage of this run') if cl in pending else done_evidence(stage1, cl, kind)
                for cl in ecells}
        nodone = sorted(cl for cl, (ok, _) in evid.items() if not ok)
        evdir = os.path.join(run.out_root, ev.EVAL_DIR[kind].format(tag=tag))
        name = 'evaluate.%s.%s' % (kind, tag)
        K.plan(run, '[evaluate]   unit %s: %d cells (%d refused before evaluation: %s) -> %s'
               % (name, len(ecells), len(listed), listed[:4], evdir))
        if nodone:
            msg = 'SE-DONE: %d cells without a complete stage 1 for %s (e.g. %s: %s)' % (
                len(nodone), kind, nodone[0], evid[nodone[0]][1])
            if run.dry_run:
                K.would_refuse(run, 'evaluate', msg)
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
            if ref_path:            # a given list holding every refused cell: passed to the engine as written
                rf = ref_path
            else:
                rf = K.write_text_atomic(os.path.join(run.out_root, ev.REFUSED_FILE.format(kind=kind, tag=tag)),
                                         refused_text(ref))
            u.meta['refused_cells_file'] = rf
            u.add_input(rf)
            u.add_input(ev.REFUSED_CELLS['snapshot'])
            u.add_input(ev.DEEP_SEARCH_PIN['snapshot'])
            u.add_input(c.paths.DATA_DECISIONS)
            # the a-priori screen's inputs (snapshot_forced_eval.zone_flags): the curve file and the gen-4 Master
            # catalogs of the evaluated LSTs and bands (rebuilt 2026-10-13: an evaluation of the old set is not resumed)
            u.add_input(c.catalog_snapshot.CURVE_CONSTANTS)
            for lst in sorted({cl[:3] for cl in ecells}):
                for b in kind_bands(kind):
                    gp = os.path.join(c.catalog_snapshot.MASTER_CATALOGS, 'master_R075_%s_%s.gaul.csv' % (lst, b))
                    if os.path.exists(gp):
                        u.add_input(gp)
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


CAMPAIGN_REFUSED = ('SM-DONE',)      # a campaign's measure refusals that list the cell refused (its stage 1 failed); any other
                                     # refusal (SM-HELD 'held', CP-PIN, CP-SPACE, SM-SUBSET, SM-STALE) leaves it incomplete


def _measure_refusals(stage1, cells, kind=None):
    """{cell: reason} of the cells refused before the evaluation of ``kind`` by their stage-1 records (under the stage-1 root's
    own records). A refused measure.<cell> unit refuses both kinds (reserve :142-144). A campaign's unit of the kind
    (campaign.stage1_cell: measure.<cell>.<the kind's bands>) refuses its kind when it ended without a cutout of the kind
    (SM-NOBAND, outcome 'no subband' / 'no wideband'; generation 5 listed such cells 'no cutouts to measure'), not run for a
    gap (SM-HELD with outcome 'gap': a gate refused the inputs of its cutouts units; gen6_forced_stages_DRAFT.sh:115 lists
    them 'held: input gate refused'), or its measurement refused (CAMPAIGN_REFUSED: 'stage 1 failed', :116-117). Any other
    state of the campaign unit - held for its cutouts, stopped, an error - is not listed: done_evidence then refuses SE-DONE
    (an incomplete campaign is never evaluated)."""
    out = {}
    rdir = os.path.join(_records_root(stage1), '_records', 'photometry')
    for cl in cells:
        p = os.path.join(rdir, 'measure.%s.json' % cl)
        if os.path.exists(p):
            r = json.load(open(p))
            if r.get('status') == 'refused':
                out[cl] = 'stage 1 refused in this run (%s)' % (r.get('reason') or '')[:160]
                continue
        if kind is None:
            continue
        p = os.path.join(rdir, '%s.json' % unit_name(cl, kind_bands(kind)))
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        st, why, outcome = r.get('status'), (r.get('reason') or '').strip(), (r.get('meta') or {}).get('outcome')
        code = why.split(':', 1)[0].strip()
        if st == 'no-input' and code == 'SM-NOBAND':
            out[cl] = 'no cutouts to measure: %s in the campaign (%s)' % (outcome or 'no %s cutout' % kind, why[:160])
        elif st == 'refused' and code == 'SM-HELD' and outcome == 'gap':
            out[cl] = 'held: input gate refused (stage 1 not run for a gap: %s)' % why[:200]
        elif st == 'refused' and code in CAMPAIGN_REFUSED:
            out[cl] = 'stage 1 failed in the campaign (%s)' % why[:160]
    return out


def _eval_admitted(cells, kind):
    from exopipe.phase3.core import exclusions
    ex = exclusions.load(scope='snapshot')     # as snapshot_forced_eval.py (exopipe_ops 0fc3207), in place: flat 'snapshot'
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
            if not (os.path.exists(p) and T.read_csv(p)[1]):
                # test_snapshot_forced_reserve.cmd_zones :215-216: an evaluation without targets_ge4 rows refuses Z
                u.refuse('Z: no targets_ge4.csv rows in %s' % evdir)
            u.add_input(p)
            r, _, _ = T.zone_rows(evdir, stage1, None, ev.CONTENT_TEST_PROCS)
            rows.extend(r)
        diff = [k for k, a, b in rows if a != b]
        if diff:            # cmd_zones fails (:220-224) BEFORE it writes the flags file (:225-230)
            u.refuse('Z1: %d targets_ge4 rows whose recomputed screen differs from the evaluation (e.g. %s); flags file '
                     'not written' % (len(diff), ' '.join(diff[0])))
        u.plan(path, 'photometry_zone_flags')
        T.write_zone_flags(path, rows)
        u.add_output(path, 'photometry_zone_flags')
    return run.units[-1:]


def _stage_cutouts(run, args):
    from exopipe.phase3.lib.photometry import steps_cutouts as SC
    return SC.stage_snapshot_cutouts(run, args)


STAGE_FUNCS = {'cutouts': _stage_cutouts, 'measure': stage_measure, 'evaluate': stage_evaluate}


def run_tier(run, args, stages):
    gate_full(args)
    args.photometry_stages = list(stages)      # a dry run plans a stage from an earlier stage of this invocation
    if getattr(args, 'campaign', False):       # once, before any stage: generation, unit list, code pin, space
        from exopipe.phase3.lib.photometry import campaign as CP
        CP.start(run, args)
    for st in stages:
        units = STAGE_FUNCS[st](run, args)
        if run.dry_run:
            continue
        # a campaign's cutouts never stop its stage 1: readiness holds each cell whose units are not terminal (campaign.py)
        if st != 'measure' and not getattr(args, 'campaign', False) and K.stop_after(args, st, units):
            break
    return None

