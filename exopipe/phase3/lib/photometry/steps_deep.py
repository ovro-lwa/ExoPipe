"""`exopipe3 photometry --tier deep`: target photometry of the HWJ and 10 pc samples on the SEARCHED deep frames of a run.

Stages (photometry.STAGES['deep']): frames -> stage -> measure -> evaluate. Each stage reads ONLY the files the previous
stage's record lists (DESIGN 11.5). Engines in place (core/legacy.run): deep_forced_frames.py, deep_forced_phot.py (one
child per shard), deep_forced_eval.py. Shell logic of run_deep_forced_20260928.sh ported here: frame-list presence (M-CHAIN),
the /data07 path rewrite + stage list (:14-26), the copy (:27-29), the staging completeness check (:30-31, now per file),
32 shards one BLAS thread each (:33), success by exit code + content tests instead of grepping logs (:34), evaluation (:36).

Units (records <out>/_records/photometry/): frames, stage, measure.shard_NN, evaluate. --out is the RUN directory
(production /fast/gh/main/phase3/deep_forced/run1..run3).
"""
import csv
import glob
import hashlib
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

from exopipe.phase3.core import cells as C
from exopipe.phase3.core import record
from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.photometry import steps_common as K


# ---------------------------------------------------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------------------------------------------------

def roots_cells():
    """{cell: cell root} of photometry_frames.ROOTS_CSV (the table deep_forced_frames.py reads, :21)."""
    out = {}
    for line in open(K.cfg().photometry_frames.ROOTS_CSV):
        if line.strip():
            q = line.strip().split(',')
            out[q[0]] = q[2]
    return out


def epochs_of(args):
    """{lst: set(dates)} of --epochs (per-hour items <LST>_<date>), or {}."""
    out = {}
    for it in C.expand_epochs(getattr(args, 'epochs', None)):
        lst, date = C.parse_epoch(it)
        out.setdefault(lst, set()).add(date)
    return out


def select_cells(args, roots):
    universe = sorted(roots)
    eps = epochs_of(args)
    if args.cells:
        cells = C.expand(args.cells, universe=universe, kind='cell')
        for c in cells:
            C.parse_cell(c)
        unknown = [c for c in cells if c not in roots]
        if unknown:
            raise ValueError('cells not in %s: %s' % (K.cfg().photometry_frames.ROOTS_CSV, unknown))
    else:
        cells = list(universe)
    if eps:
        cells = [c for c in cells if c.split('_', 1)[0] in eps]
    if args.scopes:
        sc = {s.strip() for s in args.scopes.split(',') if s.strip()}
        cells = [c for c in cells if c.split('_', 1)[1] in sc]
    if args.exclude_cells:
        ex = set(C.expand(args.exclude_cells, universe=universe, kind='cell'))
        cells = [c for c in cells if c not in ex]
    return sorted(cells), eps


def src_run(args):
    c = K.cfg()
    if args.src_run:
        return os.path.abspath(args.src_run)
    if args.run_label:
        return os.path.join(c.paths.TRANSIENTS_ROOT, c.photometry_frames.SRC_RUN_DIR.format(label=args.run_label))
    return c.photometry_frames.SRC_RUN


def default_out(args):
    c = K.cfg()
    s = src_run(args)
    label = os.path.basename(s.rstrip('/')) if s else 'Run_UNSET'
    base = os.path.join(c.photometry.OUT_ROOT['deep'], label)
    eps = getattr(args, 'epochs', None)
    if eps:
        items = C.expand(eps, kind='epoch')
        return os.path.join(base, 'epoch_' + items[0] if len(items) == 1 else 'epochs_%d_%s' % (
            len(items), hashlib.sha1(','.join(items).encode()).hexdigest()[:8]))
    return base


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------

def read_rows(path):
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)


def write_rows(path, cols, rows):
    """csv.DictWriter with its default CRLF terminator - the engines' own writer (deep_forced_frames.py:57,
    run_deep_forced_20260928.sh:24-25)."""
    tmp = '%s.tmp.%d' % (path, os.getpid())
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    return path


def v_noise_bad(rows):
    """{(lst, band): dates} failing the Stokes V 3x noise gate over the FULL night list of each cell (deep_forced_phot.py
    :233-234: THERMNOI > NOISE_GATE_X x the cell median, with >= NOISE_GATE_MIN_NIGHTS values). Per-hour lists keep one
    night, where the engine would skip the gate; the step applies it over the full list instead."""
    import numpy as np
    m = K.cfg().photometry_measure
    by = {}
    for r in rows:
        if r['stokes'] == 'V' and r['thermnoi'] not in ('', '0', '0.0'):
            by.setdefault((r['lst'], r['band']), {})[r['date']] = float(r['thermnoi'])
    bad = {}
    for k, tn in by.items():
        if len(tn) >= m.NOISE_GATE_MIN_NIGHTS['deep']:
            med = float(np.median(list(tn.values())))
            bad[k] = {d for d, v in tn.items() if v > m.NOISE_GATE_X['deep'] * med}
    return bad


def under(p, roots):
    return any(p == r or p.startswith(r.rstrip('/') + '/') for r in roots)


def holder_of(path):
    """The host whose site NODE_LOCAL_ROOTS hold ``path`` (None when no host is named)."""
    for h, rr in (K.cfg().site.NODE_LOCAL_ROOTS or {}).items():
        if under(path, rr):
            return h
    return None


def needs_staging(rows):
    """Inputs under photometry_frames.STAGE_PREFIX_ROOTS that this host cannot read (the measuring host's view)."""
    f = K.cfg().photometry_frames
    return sorted({r[k] for r in rows for k in f.STAGE_COLUMNS
                   if r[k] and under(r[k], f.STAGE_PREFIX_ROOTS) and not os.path.exists(r[k])})


def stage_root(args):
    from exopipe.phase3.core import fitsio
    if args.stage_root:
        return os.path.abspath(args.stage_root)
    return os.path.join(fitsio.scratch_root(args.work), K.cfg().photometry_frames.STAGE_SUBDIR)


def unit_names(nsh, shards):
    return ['measure.shard_%02d' % s for s in shards]


def owned_shards(rows, nsh):
    from exopipe.phase3.lib.photometry.tests import shard_cells
    return [s for s in range(nsh) if shard_cells(rows, s, nsh)]


# ---------------------------------------------------------------------------------------------------------------------
# stage frames
# ---------------------------------------------------------------------------------------------------------------------

def stage_frames(run, args):
    c = K.cfg()
    f = c.photometry_frames
    out = run.out_root
    roots = roots_cells()
    cells, eps = select_cells(args, roots)
    src, par = src_run(args), args.parent_dir or f.PARENT_DIR
    coadd = c.paths.COADD_SB_ROOT
    fl = os.path.join(out, f.FRAMES_FILE)
    K.plan(run, '[frames] %d cells%s -> %s  (deep_forced_frames.py in place: DEEP_FORCED_SRC_RUN=%s P3_PARENT_DIR=%s '
                'P3_COADD_ROOT=%s)' % (len(cells), ' (LSTs of epochs %s)' % sorted('%s_%s' % (l, d) for l, ds in eps.items()
                                                                                   for d in ds) if eps else '',
                                       fl, src, par, coadd))
    if not cells:
        raise NoInput('no deep cell selected (roots table %s, --cells/--epochs/--scopes/--exclude-cells)' % f.ROOTS_CSV)
    if run.dry_run:
        for what, p in (('searched run (F-SRCRUN)', src), ('parent catalogues (F-PARENTDIR)', par),
                        ('roots table', f.ROOTS_CSV)):
            K.plan(run, '[frames]   %s: %s %s' % (what, p, 'found' if p and os.path.exists(p) else 'MISSING - would refuse'))
        rr = sorted({roots[x] for x in cells})
        K.plan(run, '[frames]   cell roots %s readable here (%s): %s' % (rr, K.hostname(), all(os.path.isdir(r) for r in rr)))
        npar = sum(1 for x in cells if par and os.path.exists(os.path.join(par, f.PARENT_NAME.format(
            lst=x.split('_', 1)[0], band=x.split('_', 1)[1]))))
        K.plan(run, '[frames]   parent catalogues present for %d of %d cells (F-PARENTS refuses the others)' % (npar, len(cells)))
        if args.stokes:
            K.plan(run, '[frames]   --stokes %s: rows of the other Stokes dropped (changes every shard\'s blank-sky sequence)'
                   % args.stokes)
        K.plan(run, '[frames]   unit frames -> output %s (photometry_frame_list)' % fl)
        return []
    if record_resumed(run, 'frames'):
        return run.units[-1:]
    with run.unit('frames', tier='deep', stage='frames', src_run=src, parent_dir=par, cells=len(cells)) as u:
        if not src or not os.path.isdir(src):
            u.refuse('F-SRCRUN: the searched deep run is required (--src-run DIR or --run LABEL; no default, D7): %s'
                     % (src or 'not given'))
        if not par or not os.path.isdir(par):
            u.refuse('F-PARENTDIR: the parent catalogue directory the deep table pass used is required (--parent-dir; '
                     'gate_inputs.py would fall back to the stale Run_P3S4 catalogues): %s' % (par or 'not given'))
        for r_ in sorted({roots[x] for x in cells}):
            if not os.path.isdir(r_):
                u.refuse('cell root %s is not readable on %s: run the frames stage on the host that holds it (site '
                         'NODE_LOCAL_ROOTS: %s)' % (r_, K.hostname(), holder_of(r_) or 'unknown'))
        u.add_input(f.ROOTS_CSV)
        for x in cells:
            lst, band = x.split('_', 1)
            p = os.path.join(par, f.PARENT_NAME.format(lst=lst, band=band))
            for q in (p, p + '.gen.json', os.path.join(roots[x], x, 'coadd_final', f.MASTER_NAME.format(band=band)),
                      os.path.join(src, f.FRAME_SUBDIR['I'], x), os.path.join(src, f.FRAME_SUBDIR['V'], x)):
                if os.path.exists(q):
                    u.add_input(q)
        os.makedirs(out, exist_ok=True)
        u.plan(fl, 'photometry_frame_list')
        tmp = os.path.join(out, '.%s.engine.%d' % (f.FRAMES_FILE, os.getpid()))
        res, text = K.call(u, 'frames', [tmp] + cells, 'deep', 'deep_frames', run, 'frames.log',
                           env_extra={'DEEP_FORCED_SRC_RUN': src, 'P3_PARENT_DIR': par, 'P3_COADD_ROOT': coadd},
                           threads=args.threads)
        refused = [(m.group(1), m.group(2)) for m in (re.match(f.REFUSED_RE, ln) for ln in text.splitlines()) if m]
        u.meta['refused_cells'] = refused
        if res.rc != 0:
            if os.path.exists(tmp):
                os.remove(tmp)
            if 'IndexError' in text:
                raise NoInput('F-NOFRAMES: no frame over the %d selected cells (deep_forced_frames.py:57 IndexError)'
                              % len(cells))
            if 'DEEP_FORCED_SRC_RUN' in text:
                u.refuse('F-SRCRUN: deep_forced_frames refused (%s)' % res.reason)
            raise RuntimeError('deep_forced_frames.py exit %d: %s' % (res.rc, res.tail[-800:]))
        cols, rows = read_rows(tmp)
        n_engine = len(rows)
        keep = rows
        if eps:
            bad = v_noise_bad(rows)
            keep = []
            for r in rows:
                if r['date'] not in eps.get(r['lst'], ()):
                    continue
                if r['stokes'] == 'V' and r['date'] in bad.get((r['lst'], r['band']), ()):
                    u.refuse_item(r['path'], 'M-NOISE: Stokes V THERMNOI over %g x the cell median of the full night list '
                                  '(deep_forced_phot.py:233-234; applied here because the per-hour list holds one night)'
                                  % c.photometry_measure.NOISE_GATE_X['deep'], gate='M-NOISE')
                    continue
                keep.append(r)
        if args.stokes:
            ss = {s.strip() for s in args.stokes.split(',')}
            keep = [r for r in keep if r['stokes'] in ss]
        if keep is rows:
            os.replace(tmp, fl)
        else:
            os.remove(tmp)
            if not keep:
                raise NoInput('F-NOFRAMES: no frame of epochs %s / Stokes %s in the frame list (%d engine rows)'
                              % (sorted(eps), args.stokes, n_engine))
            write_rows(fl, cols, keep)
        for cell, why in refused:
            u.refuse_item('%s#cell=%s' % (fl, cell), why, gate=_gate_of(why))
        listed = {'%s_%s' % (r['lst'], r['band']) for r in keep}
        silent = [x for x in cells if x not in listed and x not in {cl for cl, _ in refused}
                  and not (eps or args.stokes)]
        u.check('frames.cells_listed_or_refused', not silent, len(silent),
                'every selected cell listed or refused' if not silent else 'neither listed nor refused: %s' % silent[:10])
        u.meta.update(frame_list_sha1=K.sha1_file(fl), n_rows=len(keep), n_engine_rows=n_engine)
        u.add_output(fl, 'photometry_frame_list', refused_cells=[cl for cl, _ in refused])
    return run.units[-1:]


def _gate_of(why):
    if why.startswith('parents'):
        return 'F-PARENTS'
    if why.startswith('no PB'):
        return 'F-PB'
    if why.startswith('no search frames'):
        return 'F-NOFRAMES'
    return 'deep_forced_frames'


def record_resumed(run, name, admitted=None):
    return K.resumed(run, name, admitted=admitted, tier='deep')


# ---------------------------------------------------------------------------------------------------------------------
# stage stage (run_deep_forced_20260928.sh:14-31)
# ---------------------------------------------------------------------------------------------------------------------

def frame_list_for(run, args, stage):
    """(path, sha1 expected or None, where it came from) of the frame list a stage reads."""
    f = K.cfg().photometry_frames
    if args.frames:
        return os.path.abspath(args.frames), None, '--frames'
    if stage == 'measure':
        rec = record.load_record(run.out_root, run.step, 'stage')
        if rec and rec.get('status') == 'done':
            p = K.outputs_of(rec, 'photometry_frame_list')
            return p[0], (rec.get('meta') or {}).get('frame_list_sha1'), 'stage record'
    if run.dry_run and not os.path.exists(record.record_path(run.out_root, run.step, 'frames')):
        return os.path.join(run.out_root, f.FRAMES_FILE), None, 'frames stage (not run yet)'
    rec = K.prev_record(run, 'frames', 'frames')
    p = K.outputs_of(rec, 'photometry_frame_list')
    if len(p) != 1:
        raise Refusal('M-CHAIN: the frames record lists %d frame lists' % len(p))
    return p[0], (rec.get('meta') or {}).get('frame_list_sha1'), 'frames record'


def check_list(path, sha1):
    if not os.path.exists(path):
        raise Refusal('M-CHAIN: frame list %s missing' % path)
    if sha1 and K.sha1_file(path) != sha1:
        raise Refusal('M-CHAIN: frame list %s changed since its record was written (sha1)' % path)


def stage_stage(run, args):
    c = K.cfg()
    f = c.photometry_frames
    src, sha, how = frame_list_for(run, args, 'stage')
    if run.dry_run and not os.path.exists(src):
        K.plan(run, '[stage] frame list %s from the %s: staging decided when it exists (inputs under %s this host cannot '
                    'read are copied)' % (src, how, f.STAGE_PREFIX_ROOTS))
        return []
    check_list(src, sha)
    cols, rows = read_rows(src)
    need = needs_staging(rows)
    if not need:
        K.plan(run, '[stage] not needed on %s: every input of %s is readable here (no unit)' % (K.hostname(), src))
        return []
    stg = stage_root(args)
    alln = sorted({r[k] for r in rows for k in f.STAGE_COLUMNS if r[k] and under(r[k], f.STAGE_PREFIX_ROOTS)})
    hosts = {}          # every prefix path is rewritten (the heredoc's rule), so every one is copied: from here when readable
    for p in alln:
        hosts.setdefault(K.hostname() if os.path.exists(p) else holder_of(p), []).append(p)
    K.plan(run, '[stage] %d inputs under %s not readable on %s -> %s (from %s); outputs %s, %s'
           % (len(need), f.STAGE_PREFIX_ROOTS, K.hostname(), stg, sorted(map(str, hosts)),
              f.STAGED_FRAMES_FILE, f.STAGE_LIST_FILE))
    if run.dry_run:
        return []
    if record_resumed(run, 'stage'):
        return run.units[-1:]
    out09 = os.path.join(run.out_root, f.STAGED_FRAMES_FILE)
    sl = os.path.join(run.out_root, f.STAGE_LIST_FILE)
    with run.unit('stage', tier='deep', stage='stage', stage_root=stg, frame_list=src) as u:
        if None in hosts:
            u.refuse('S-STAGE: no host in the site NODE_LOCAL_ROOTS holds %s' % hosts[None][:3])
        u.add_input(src)
        u.plan(out09, 'photometry_frame_list')
        u.plan(sl, 'photometry_stage_list')
        for r in rows:          # the heredoc of run_deep_forced_20260928.sh:14-26: every prefix path rewritten
            for k in f.STAGE_COLUMNS:
                if r[k] and under(r[k], f.STAGE_PREFIX_ROOTS):
                    r[k] = stg + r[k]
        K.write_text_atomic(sl, '\n'.join(alln) + '\n')
        os.makedirs(stg, exist_ok=True)
        for h, ps in sorted(hosts.items()):
            lst = os.path.join(K.logs_dir(run), 'stage_files_%s.txt' % h)
            os.makedirs(os.path.dirname(lst), exist_ok=True)
            K.write_text_atomic(lst, '\n'.join(sorted(ps)) + '\n')
            copy_from(h, lst, stg, u, run)
        miss = [p for p in alln if not os.path.exists(stg + p)]
        if miss:
            u.refuse('S-STAGE: %d of %d listed inputs not staged under %s (e.g. %s)' % (len(miss), len(alln), stg, miss[:3]))
        write_rows(out09, cols, rows)
        u.meta.update(frame_list_sha1=K.sha1_file(out09), n_staged=len(alln), source_list_sha1=K.sha1_file(src))
        u.add_output(sl, 'photometry_stage_list')
        u.add_output(out09, 'photometry_frame_list')
    return run.units[-1:]


def copy_from(host, listfile, dst, u, run):
    """rsync -aL --files-from=<list> <host>:/ <dst>/ (the copy of run_deep_forced_20260928.sh:27, pulled by the measuring
    host instead of pushed); the ssh environment from runner.SSH_PREFIX_ENV."""
    c = K.cfg()
    if host == K.hostname():
        argv = list(c.photometry_frames.RSYNC) + ['--files-from=%s' % listfile, '/', dst + '/']
    else:
        argv = list(c.photometry_frames.RSYNC) + ['--files-from=%s' % listfile, '%s:/' % host, dst + '/']
    env = dict(os.environ)
    env.update(c.runner.SSH_PREFIX_ENV)
    log = os.path.join(K.logs_dir(run), 'stage.log')
    with open(log, 'a') as lf:
        p = subprocess.run(argv, stdout=lf, stderr=subprocess.STDOUT, env=env)
    u.meta.setdefault('copies', []).append({'argv': argv, 'rc': p.returncode})
    if p.returncode != 0:
        raise Refusal('S-STAGE: copy from %s failed (rsync exit %d; %s)' % (host, p.returncode, log))


# ---------------------------------------------------------------------------------------------------------------------
# stage measure (deep_forced_phot.py, one child per shard)
# ---------------------------------------------------------------------------------------------------------------------

def stage_measure(run, args):
    c = K.cfg()
    m = c.photometry_measure
    path, sha, how = frame_list_for(run, args, 'measure')
    nsh = args.shards or m.SHARDS['deep']
    outd = os.path.join(run.out_root, 'out')
    if run.dry_run and not os.path.exists(path):
        K.plan(run, '[measure] frame list %s from the %s; %d shards (deep_forced_phot.py in place, one BLAS thread each) '
                    '-> %s/forced_NNN.csv.gz, blank_NNN.csv, frames_NNN.csv' % (path, how, nsh, outd))
        return []
    check_list(path, sha)
    _, rows = read_rows(path)
    if not rows:
        raise NoInput('M-CHAIN: frame list %s has no row' % path)
    need = needs_staging(rows)
    from exopipe.phase3.lib.photometry.tests import shard_cells
    shards = owned_shards(rows, nsh)
    K.plan(run, '[measure] frame list %s (%s; %d frames, %d cells); %d of %d shards own cells; out %s'
           % (path, how, len(rows), len({(r['lst'], r['band']) for r in rows}), len(shards), nsh, outd))
    if need:
        msg = 'M-CHAIN: %d inputs of %s are not readable on %s (e.g. %s): run --stages stage first' % (
            len(need), path, K.hostname(), need[:2])
        if run.dry_run:
            K.plan(run, '[measure]   WOULD REFUSE ' + msg)
            return []
        raise Refusal(msg)
    # every per-cell input and an evenly spaced sample of frames must be readable here (the engine would crash on a
    # missing one; a stat of every frame costs minutes on a loaded Lustre metadata server)
    percell = sorted({r[k] for r in rows for k in ('pb_path', 'weights_path', 'parent_path') if r[k]})
    step = max(1, len(rows) // 50)
    miss = [p for p in percell + [r['path'] for r in rows[::step]] if not os.path.exists(p)]
    if miss:
        msg = 'M-INPUTS: %d inputs of %s missing on %s (e.g. %s)' % (len(miss), path, K.hostname(), miss[:2])
        if run.dry_run:
            K.plan(run, '[measure]   WOULD REFUSE ' + msg)
            return []
        raise Refusal(msg)
    if run.dry_run:
        for s in shards[:4]:
            K.plan(run, '[measure]   unit measure.shard_%02d: cells %s' % (s, ['%s_%s' % x for x in shard_cells(rows, s, nsh)]))
        if len(shards) > 4:
            K.plan(run, '[measure]   ... %d more shard units' % (len(shards) - 4))
        return []
    os.makedirs(outd, exist_ok=True)
    samples = list(K.sample_paths().values())
    pinned = [m.PARENT_TABLES['deep'], m.SOLAR_AUDIT['deep'], c.paths.DATA_DECISIONS]

    def one(s):
        name = 'measure.shard_%02d' % s
        mine = set(shard_cells(rows, s, nsh))
        my = [r for r in rows if (r['lst'], r['band']) in mine]
        adm = K.admitted_epochs(my)
        if record_resumed(run, name, admitted=adm):
            return
        try:
            with run.unit(name, tier='deep', stage='measure', shard=s, nshards=nsh, frame_list=path,
                          frame_list_sha1=K.sha1_file(path), cells=sorted('%s_%s' % x for x in mine)) as u:
                for p in [path] + samples + pinned:
                    u.add_input(p)
                for p in sorted({r[k] for r in my for k in ('path', 'pb_path', 'weights_path', 'parent_path') if r[k]}):
                    u.add_input(p)
                u.set_admitted(adm)
                fo = os.path.join(outd, 'forced_%03d.csv.gz' % s)
                bo = os.path.join(outd, 'blank_%03d.csv' % s)
                ao = os.path.join(outd, 'frames_%03d.csv' % s)
                for p, cls in ((fo, 'photometry_table'), (bo, 'photometry_blank_null'), (ao, 'photometry_admission')):
                    u.plan(p, cls)
                res, text = K.call(u, 'measure_deep', [path, s, nsh, outd], 'deep', 'deep_measure', run,
                                   'measure_shard_%02d.log' % s, threads=args.threads)
                if res.refused:
                    u.refuse('deep_forced_phot shard %d: %s' % (s, res.reason))
                if res.rc != 0:
                    raise RuntimeError('deep_forced_phot.py shard %d exit %d: %s' % (s, res.rc, res.tail[-800:]))
                ctx = dict(frames_csv=path, shard=s, nshards=nsh)
                u.add_output(ao, 'photometry_admission', **ctx)
                u.add_output(bo, 'photometry_blank_null', **ctx)
                u.add_output(fo, 'photometry_table', **ctx)
        except Exception:       # noqa: BLE001 - recorded as the unit's error
            pass
    workers = args.workers or min(len(shards), max(1, len(os.sched_getaffinity(0))))
    n0 = len(run.units)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(one, shards))
    return run.units[n0:]


# ---------------------------------------------------------------------------------------------------------------------
# stage evaluate (deep_forced_eval.py)
# ---------------------------------------------------------------------------------------------------------------------

def shard_records(run):
    out = []
    for p in sorted(glob.glob(os.path.join(run.out_root, '_records', run.step, 'measure.shard_*.json'))):
        with open(p) as f:
            out.append(json.load(f))
    return out


def stage_evaluate(run, args):
    outd = os.path.join(run.out_root, 'out')
    ev = os.path.join(outd, 'eval')
    recs = shard_records(run)
    K.plan(run, '[evaluate] %d shard records under %s; deep_forced_eval.py in place -> %s (photometry_eval)'
           % (len(recs), run.out_root, ev))
    if run.dry_run:
        bad = [r['unit'] for r in recs if r.get('status') != 'done']
        if not recs or bad:
            K.plan(run, '[evaluate]   WOULD REFUSE M-CHAIN: %s' % ('no shard record' if not recs else 'not done: %s' % bad))
        return []
    if not recs:
        raise Refusal('M-CHAIN: no measure record under %s (run --stages measure first)' % run.out_root)
    lists = {((r.get('meta') or {}).get('frame_list'), (r.get('meta') or {}).get('frame_list_sha1'),
              (r.get('meta') or {}).get('nshards')) for r in recs}
    if len(lists) != 1:
        raise Refusal('M-CHAIN: the shard records were measured from different frame lists / shard counts: %s' % lists)
    path, sha, nsh = next(iter(lists))
    check_list(path, sha)
    _, rows = read_rows(path)
    want = set(owned_shards(rows, nsh))
    have = {(r.get('meta') or {}).get('shard'): r for r in recs}
    miss = sorted(want - set(have))
    notdone = sorted(s for s, r in have.items() if r.get('status') != 'done')
    if miss or notdone:
        raise Refusal('M-CHAIN: shards without a done record: missing %s, not done %s' % (miss, notdone))
    listed = {p for r in recs for p in K.outputs_of(r, 'photometry_table')}
    present = set(glob.glob(os.path.join(outd, 'forced_*.csv.gz')))
    stray = sorted(present - listed)
    if stray:
        raise Refusal('M-CHAIN: forced files in %s that no shard record lists (the evaluation globs forced_*.csv.gz): %s'
                      % (outd, [os.path.basename(p) for p in stray[:5]]))
    adm = sorted({a for r in recs for a in (r.get('admitted') or [])})
    if record_resumed(run, 'evaluate', admitted=adm):
        return run.units[-1:]
    with run.unit('evaluate', tier='deep', stage='evaluate', frame_list=path, nshards=nsh) as u:
        for p in sorted(listed) + sorted(glob.glob(os.path.join(outd, 'blank_*.csv'))):
            u.add_input(p)
        u.set_admitted(adm)
        u.plan(ev, 'photometry_eval')
        res, text = K.call(u, 'evaluate_deep', [outd], 'deep', 'deep_evaluate', run, 'evaluate.log', threads=args.threads)
        if res.refused:
            u.refuse('E-ORDER/E-POSCOLS/E-RADIO: deep_forced_eval refused (%s)' % res.reason)
        if res.rc != 0:
            raise RuntimeError('deep_forced_eval.py exit %d: %s' % (res.rc, res.tail[-800:]))
        u.add_output(ev, 'photometry_eval')
    return run.units[-1:]


STAGE_FUNCS = {'frames': stage_frames, 'stage': stage_stage, 'measure': stage_measure, 'evaluate': stage_evaluate}


def run_tier(run, args, stages):
    for st in stages:
        units = STAGE_FUNCS[st](run, args)
        if run.dry_run:
            continue
        if units and not K.stage_ok(units):
            K.say('photometry deep: stage %s did not complete (%s) - later stages not run'
                  % (st, ', '.join('%s %s' % (u.name, u.status) for u in units if u.status not in ('done', 'skipped-valid'))))
            break
    return None
