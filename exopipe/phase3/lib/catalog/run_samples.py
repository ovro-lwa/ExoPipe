"""THE SAMPLE CATALOGS A SNAPSHOT CHAIN RUN READ - recorded in the run, re-read from the run (cutover row 70; clv15 fix_ledger
FIX-20261125-SNAPCOMB-REJUDGE-PER-RUN-SAMPLE-CATALOGS).

The snapshot chain (clv15 phase3_snapshot_chain.py, run in place by lib/catalog/tier_snapshot.py) matches served sources against
the sample catalogs - its SAMPLES (Hot_Warm_Jupiters.csv, 10pc.csv, 25pc.csv, LPTs.csv, flare_ucd.csv, sydney_stars.csv,
pulsars_atnf.csv), read by load_samples from --samples-dir - in its catalog route ('E:<sample>:<name>'). The re-judging of the
combined 10-s / 1-minute view (lib/catalog/rejudge.py) reconstructs that route and refuses unless it reproduces the chain, so it
needs the catalogs the chain read, not today's: clv15 LPTs.csv gained 8 rows on 2026-11-18, after which Run_SNAPB4G3_20261104 no
longer reproduced on the live catalogs (2 served sources) and production passed a per-run directory by hand (exopipe_ops 352ef13,
SNAPCOMB_REJUDGE_SAMPLES, a CSV run,samples_dir).

Here every run carries its own set:
  <run>/sample_catalogs/<file>   a verified copy of each catalog file (core/placement.copy: the copy's size and md5 equal the bytes
                                 read from the source, else nothing is placed)
  <run>/sample_catalogs.json     the record: format, how the set was written ('chain' = by the chain stage before the chain read it;
                                 'backfill' = written once afterwards from a named directory, with the operator's reason), when,
                                 the source directory, and per file its name, md5, size in bytes, the path it was copied from and
                                 that file's modification time (UTC)
The chain stage copies the catalogs BEFORE the chain runs and passes --samples-dir <run>/sample_catalogs, so the record names
exactly the bytes the chain read. The re-judging reads the record itself (verified): it refuses when the record is missing or
malformed, a recorded copy is missing, a copy's md5 or size differs from the record, the copy directory holds a file the record
does not list, or the record's file names are not the chain's SAMPLES. It never falls back to the live catalogs.

A run chained before the record existed has none. `exopipe3 catalog --tier snapshot --stages samples-backfill` (backfill_main
below) writes it once for a named run from a named directory - refused when the run already has a record - then re-judges the run
with no removals on the recorded set (rejudge.py --reproduce: the re-judging's own reproduction check) and, unless that
reproduces the chain, moves the record and its copies aside and refuses.

The re-judging child (Python 3.6, the standard library and clv15) loads this file by its path: only the standard library is
imported at module level; the writers import core/placement when called (in the step's own interpreter).
"""
import collections
import hashlib
import json
import os
import time

RECORD = 'sample_catalogs.json'          # the record, in the run directory
COPY_DIR = 'sample_catalogs'             # the copies, in the run directory (not 'source_catalogs*': the run's own catalogues)
FORMAT = 1
HOWS = ('chain', 'backfill')
STAGE = 'samples-backfill'


class SamplesRefusal(Exception):
    """A run's sample-catalog record is missing, malformed or does not match its copies; or a write is refused."""


def record_path(run):
    return os.path.join(run, RECORD)


def copy_dir(run):
    return os.path.join(run, COPY_DIR)


def md5_file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 22), b''):
            h.update(blk)
    return h.hexdigest()


def _utc(t):
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t))


def chain_names(c=None):
    """The file names the chain's load_samples reads (catalog_snapshot.SAMPLES = clv15 phase3_snapshot_chain.SAMPLES, bound by
    tier_snapshot's binding check), in its order."""
    if c is None:
        from exopipe.phase3.config import CFG
        c = CFG.catalog_snapshot
    return [fn for _tag, fn in c.SAMPLES]


def verified(run, names=None):
    """(copy directory, record) of ``run``'s sample catalogs, every recorded copy checked against the record (md5 and size).
    ``names``: the file names the reader will read (the chain's SAMPLES) - the record must list exactly these. SamplesRefusal
    when the record is missing or malformed, a copy is missing or differs, the copy directory holds a file the record does not
    list, or the names differ. Never a fallback to another directory."""
    tag = os.path.basename(os.path.normpath(run))
    rp = record_path(run)
    if not os.path.isfile(rp):
        raise SamplesRefusal('%s has no %s: the sample catalogs its chain read are not recorded, so its catalog route cannot be '
                             'reconstructed (a run chained before the record existed needs the one-time backfill, '
                             '`exopipe3 catalog --tier snapshot --stages %s`)' % (tag, RECORD, STAGE))
    try:
        with open(rp) as f:
            rec = json.load(f, object_pairs_hook=collections.OrderedDict)
    except (OSError, ValueError) as e:
        raise SamplesRefusal('%s %s cannot be read: %s' % (tag, RECORD, e))
    files = rec.get('files') if isinstance(rec, dict) else None
    if not isinstance(rec, dict) or rec.get('format') != FORMAT or rec.get('how') not in HOWS or not isinstance(files, list) \
            or not files or not all(isinstance(x, dict) and x.get('name') and x.get('md5') and isinstance(x.get('bytes'), int)
                                    for x in files):
        raise SamplesRefusal('%s %s is malformed (format %s, how %s): it needs format %d, how in %s and a non-empty list of '
                             'files with name, md5 and bytes' % (tag, RECORD, rec.get('format') if isinstance(rec, dict) else None,
                                                                rec.get('how') if isinstance(rec, dict) else None, FORMAT,
                                                                '/'.join(HOWS)))
    listed = [x['name'] for x in files]
    if len(set(listed)) != len(listed) or any(n != os.path.basename(n) or n.startswith('.') for n in listed):
        raise SamplesRefusal('%s %s lists a file twice or a name that is not a plain file name: %s' % (tag, RECORD, listed))
    if names is not None and sorted(listed) != sorted(names):
        raise SamplesRefusal('%s %s lists %s; the chain reads %s (missing %s, extra %s)' % (
            tag, RECORD, sorted(listed), sorted(names), sorted(set(names) - set(listed)), sorted(set(listed) - set(names))))
    d = copy_dir(run)
    if not os.path.isdir(d):
        raise SamplesRefusal('%s has %s but no %s/ (the recorded copies)' % (tag, RECORD, COPY_DIR))
    extra = sorted(set(os.listdir(d)) - set(listed))
    if extra:
        raise SamplesRefusal('%s %s/ holds files its record does not list: %s' % (tag, COPY_DIR, extra))
    for x in files:
        p = os.path.join(d, x['name'])
        if not os.path.isfile(p):
            raise SamplesRefusal('%s recorded sample catalog %s/%s is missing' % (tag, COPY_DIR, x['name']))
        n, m = os.path.getsize(p), md5_file(p)
        if (n, m) != (x['bytes'], x['md5']):
            raise SamplesRefusal('%s recorded sample catalog %s/%s differs from its record: %d bytes, md5 %s; recorded %d bytes, '
                                 'md5 %s' % (tag, COPY_DIR, x['name'], n, m, x['bytes'], x['md5']))
    return d, rec


def md5_map(rec):
    """{file name: md5} of a record, in its order."""
    return collections.OrderedDict((x['name'], x['md5']) for x in rec['files'])


def write(run, src_dir, names, how, why=None, ledger=None):
    """Copy ``names`` from ``src_dir`` into <run>/sample_catalogs/ (core/placement.copy, verified) and write <run>/
    sample_catalogs.json, then read it back with verified(). ``how``: 'chain' (the chain stage, before the chain reads the copies)
    or 'backfill' (needs ``why``; ``ledger`` optional). SamplesRefusal when the run already has a record or a copy directory (a
    record is written once), a named file is missing from ``src_dir`` (the chain would skip it silently), or a copy cannot be
    verified. Returns the record."""
    from exopipe.phase3.core import placement
    tag = os.path.basename(os.path.normpath(run))
    if how not in HOWS:
        raise ValueError('how %r: one of %s' % (how, HOWS))
    if how == 'backfill' and not (why or '').strip():
        raise SamplesRefusal('a backfill records why the named directory holds the catalogs the chain read: give the reason')
    if not os.path.isdir(run):
        raise SamplesRefusal('%s is not a directory' % run)
    for p in (record_path(run), copy_dir(run)):
        if os.path.lexists(p):
            raise SamplesRefusal('%s already has %s: a run\'s sample-catalog record is written once (nothing changed)'
                                 % (tag, os.path.basename(p)))
    if not names:
        raise SamplesRefusal('no sample catalog names given')
    missing = [n for n in names if not os.path.isfile(os.path.join(src_dir, n))]
    if missing:
        raise SamplesRefusal('%s lacks %s: the chain would skip a missing catalog silently - refused, nothing written'
                             % (src_dir, missing))
    placement.make_dirs(copy_dir(run))
    files = []
    try:
        for n in names:
            src = os.path.join(src_dir, n)
            dst = os.path.join(copy_dir(run), n)
            m = placement.copy(src, dst, times=True)
            st = os.stat(dst)
            files.append(collections.OrderedDict([('name', n), ('md5', m), ('bytes', st.st_size),
                                                  ('source', os.path.abspath(src)), ('source_mtime_utc', _utc(st.st_mtime))]))
    except placement.PlacementRefusal as e:                 # this call's partial copies go: no half-written set remains
        import shutil
        shutil.rmtree(copy_dir(run), ignore_errors=True)
        raise SamplesRefusal('%s: a sample catalog copy was not placed (the partial copies removed): %s' % (tag, e))
    rec = collections.OrderedDict([
        ('format', FORMAT),
        ('what', 'the sample catalogs the snapshot chain read (phase3_snapshot_chain.load_samples, --samples-dir %s/)' % COPY_DIR),
        ('how', how), ('written_utc', _utc(time.time())), ('source_dir', os.path.abspath(src_dir)), ('copy_dir', COPY_DIR),
        ('files', files)])
    if how == 'backfill':
        rec['why'] = why.strip()
        if ledger:
            rec['ledger'] = ledger
    with placement.atomic(record_path(run)) as tmp:
        with open(tmp, 'w') as f:
            json.dump(rec, f, indent=1)
            f.write('\n')
    verified(run, names)
    return rec


def set_aside(run, dest):
    """Move the run's record and copies into ``dest`` (a new directory): a backfill whose set did not reproduce the chain leaves
    no record behind. Returns the paths moved."""
    os.makedirs(dest)
    moved = []
    for p in (record_path(run), copy_dir(run)):
        if os.path.lexists(p):
            os.rename(p, os.path.join(dest, os.path.basename(p)))
            moved.append(p)
    return moved


# =====================================================================================================================
# the one-time backfill: exopipe3 catalog --tier snapshot --stages samples-backfill
# =====================================================================================================================

def add_arguments(p):
    p.add_argument('--samples-run', metavar='DIR',
                   help='--stages %s: the chain run directory to give its sample-catalog record (written once; refused when '
                        'it has one)' % STAGE)
    p.add_argument('--samples-from', metavar='DIR',
                   help='--stages %s: the directory holding the sample catalogs that run\'s chain read (each of '
                        'catalog_snapshot.SAMPLES; copied and recorded with their md5)' % STAGE)
    p.add_argument('--samples-why', metavar='TEXT',
                   help='--stages %s: why that directory holds the catalogs the chain read (recorded; required)' % STAGE)
    p.add_argument('--samples-ledger', metavar='ID', help='--stages %s: the ledger row of the backfill (recorded)' % STAGE)


def wants(args):
    """True when the catalog step was asked for this stage (--stages samples-backfill, alone)."""
    st = getattr(args, 'stages', None)
    return bool(st) and [s.strip() for s in str(st).split(',') if s.strip()] == [STAGE]


def _run_arg(args):
    v = getattr(args, 'samples_run', None)
    if not v:
        raise SystemExit('error: catalog --stages %s needs --samples-run DIR (the chain run), --samples-from DIR and '
                         '--samples-why TEXT' % STAGE)
    return os.path.abspath(v)


def default_out(args):
    """The run directory itself (the record is written there; a production root is refused by the CLI in the refactor
    window)."""
    return _run_arg(args)


def backfill_main(args, run):
    """`exopipe3 catalog --tier snapshot --stages samples-backfill --samples-run RUN --samples-from DIR --samples-why TEXT
    [--samples-ledger ID] --out RUN` - one unit '<run>.samples-backfill': write the record (write(how='backfill')), then the
    reproduction check (rejudge.py --reproduce under catalog_omission.REJUDGE_PYTHON); a refusal of the check moves the record
    and copies to <RUN>/_records/catalog/<run>/samples_refused_<UTC>/ and refuses the unit (exit 2)."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import legacy
    from exopipe.phase3.core.record import Refusal
    if getattr(args, 'tier', None) != 'snapshot':
        raise SystemExit('error: --stages %s is a stage of the snapshot tier (--tier snapshot)' % STAGE)
    rd = _run_arg(args)
    src = getattr(args, 'samples_from', None)
    why = (getattr(args, 'samples_why', None) or '').strip()
    if not src or not why:
        raise SystemExit('error: catalog --stages %s needs --samples-from DIR and --samples-why TEXT' % STAGE)
    src = os.path.abspath(src)
    if os.path.abspath(run.out_root) != rd:
        raise SystemExit('error: --out %s: the backfill writes into the run itself - give --out %s (or leave it out)'
                         % (run.out_root, rd))
    tag = os.path.basename(rd)
    names = chain_names(CFG.catalog_snapshot)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rejudge.py')
    clv = legacy_clv15()
    if run.dry_run:
        print('[%s.%s] copy %s from %s into %s/%s/ and write %s/%s; then %s %s --reproduce %s %s (no removals)'
              % (tag, STAGE, ', '.join(names), src, rd, COPY_DIR, rd, RECORD, CFG.catalog_omission.REJUDGE_PYTHON, script, rd, clv))
        return 0
    with run.unit('%s.%s' % (tag, STAGE), tier='snapshot', stage=STAGE) as u:
        if not os.path.isfile(os.path.join(rd, 'chain_summary.json')):
            raise Refusal('SB1: %s has no chain_summary.json - not a snapshot chain run' % rd)
        try:
            rec = write(rd, src, names, 'backfill', why=why, ledger=getattr(args, 'samples_ledger', None))
        except SamplesRefusal as e:
            raise Refusal('SB2: %s' % e)
        for x in rec['files']:
            u.add_input(x['source'])
        log = os.path.join(rd, '_records', 'catalog', tag, 'samples_reproduce_%s.log' % time.strftime('%Y%m%dT%H%M%SZ',
                                                                                                        time.gmtime()))
        os.makedirs(os.path.dirname(log), exist_ok=True)
        lr = legacy.run(script, ['--reproduce', rd, clv], python=CFG.catalog_omission.REJUDGE_PYTHON, log=log, unit=u,
                        timeout=CFG.catalog_omission.REJUDGE_TIMEOUT_S, refusal_codes=(2,))
        line = ''
        if os.path.exists(log):
            with open(log) as f:
                line = ([ln.strip() for ln in f if ln.startswith(('REPRODUCE-OK', 'REFUSING'))] or [''])[-1]
        u.check('SB3.reproduces-chain', bool(lr.ok) and line.startswith('REPRODUCE-OK'), detail=line[:500])
        if not (lr.ok and line.startswith('REPRODUCE-OK')):
            dest = os.path.join(rd, '_records', 'catalog', tag, 'samples_refused_%s' % time.strftime('%Y%m%dT%H%M%SZ',
                                                                                                     time.gmtime()))
            set_aside(rd, dest)
            raise Refusal('SB3: the catalogs of %s do not reproduce the chain of %s (%s; log %s) - the record and its copies '
                          'moved to %s' % (src, tag, line or lr.reason, log, dest))
        u.note('%s: %d sample catalogs recorded from %s (%s); %s' % (
            tag, len(rec['files']), src, ', '.join('%s %s' % kv for kv in md5_map(rec).items()), line))
        u.meta['samples'] = dict(md5_map(rec))
        u.add_output(record_path(rd), 'sample_catalogs', tier='snapshot')
    return None


def legacy_clv15():
    """The clv15 root the re-judging child imports the chain from (core.env.legacy_root)."""
    from exopipe.phase3.core import env
    return env.legacy_root('clv15')
