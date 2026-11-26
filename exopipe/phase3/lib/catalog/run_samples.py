"""THE SAMPLE CATALOGS A SNAPSHOT CHAIN RUN READ - recorded in the run, re-read from the run (cutover row 70; clv15 fix_ledger
FIX-20261125-SNAPCOMB-REJUDGE-PER-RUN-SAMPLE-CATALOGS).

The snapshot chain (clv15 phase3_snapshot_chain.py, run in place by lib/catalog/tier_snapshot.py) matches served sources against
the sample catalogs - its SAMPLES (Hot_Warm_Jupiters.csv, 10pc.csv, 25pc.csv, LPTs.csv, flare_ucd.csv, sydney_stars.csv,
pulsars_atnf.csv), read by load_samples from --samples-dir - in its catalog route ('E:<sample>:<name>'). The re-judging of the
combined 10-s / 1-minute view (lib/catalog/rejudge.py) reconstructs that route and refuses unless it reproduces the chain, so it
needs the catalogs the chain read, not today's: clv15 LPTs.csv gained 8 rows on 2026-11-18, after which Run_SNAPB4G3_20261104 no
longer reproduced on the live catalogs (2 served sources) and production passed a per-run directory by hand (exopipe_ops 352ef13,
SNAPCOMB_REJUDGE_SAMPLES, a CSV run,samples_dir).

Here every run carries its own set, in one directory:
  <run>/sample_catalogs/<file>                a verified copy of each catalog file (core/placement.copy: the copy's size and md5
                                              equal the bytes read from the source, else nothing is placed)
  <run>/sample_catalogs/sample_catalogs.json  the record: format, how the set was written ('chain' = by the chain stage before the
                                              chain read it; 'backfill' = written once afterwards from a named directory, with the
                                              operator's reason), when, the source directory, and per file its name, md5, size in
                                              bytes, the path it was copied from and that file's modification time (UTC)
A set is written under a staging name in the run (<run>/sample_catalogs.staging.<UTC>.<pid>.<8 hex digits>/: the copies, then the
record, then read back with verified()) and renamed into place in one rename, so the record appears with its copies or not at all.
A write that fails in any way removes the staging it made (nothing else) and re-raises; a staging left by a killed writer is removed
by the next write into that run, by its name alone. Only <run>/sample_catalogs/sample_catalogs.json is a record (verified(), the
re-judging's gate G9b, the content test sample_catalogs); a staging never is.

The chain stage writes the set BEFORE the chain runs and passes --samples-dir <run>/sample_catalogs, so the record names exactly the
bytes the chain read. The re-judging reads the record itself (verified): it refuses when the record is missing or malformed, a
recorded copy is missing, a copy's md5 or size differs from the record, the directory holds a file the record does not list (a
temporary file of placement's own naming, core/tempnames.py, is not counted), or the record's file names are not the chain's
SAMPLES. It never falls back to the live catalogs.

A run chained before the record existed has none. The one-time backfill (backfill / backfill_main: `exopipe3 catalog --tier snapshot
--stages samples-backfill`) stages the set for a named run from a named directory - refused when the run has a set already - then
re-judges the run with no removals on the STAGED set (rejudge.py --reproduce RUN CLV15 STAGING: the re-judging's own reproduction
check) and renames it into place only when that reproduces the chain; otherwise the staging is removed and the stage refuses. No
record exists before its reproduction passed.

The runs that need it: `exopipe3 verify sample_catalogs --glob '<root>/Run_SNAP*/' --emit-failing <list>` (read-only: verify
writes no records) names every run directory under <root> without a valid record (the content test takes a run directory as well
as the record).

The re-judging child (Python 3.6, the standard library and clv15) loads this file by its path: only the standard library is
imported at module level, core/tempnames.py is loaded by its path too, and the writers import core/placement when called (in the
step's own interpreter).
"""
import collections
import hashlib
import json
import os
import re
import shutil
import sys
import time

RECORD = 'sample_catalogs.json'          # the record, inside the set's directory
COPY_DIR = 'sample_catalogs'             # the set (copies + record), in the run directory (not 'source_catalogs*': the run's own
STAGING = COPY_DIR + '.staging'          # catalogues); <run>/sample_catalogs.staging.<UTC>.<pid>.<8 hex digits>/: a set being written
_STAGING_RE = re.compile(r'^%s\.\d{8}T\d{6}Z\.\d+\.[0-9a-f]{8}$' % re.escape(STAGING))
FORMAT = 1
HOWS = ('chain', 'backfill')
STAGE = 'samples-backfill'


class SamplesRefusal(Exception):
    """A run's sample-catalog record is missing, malformed or does not match its copies; or a write is refused."""


class NotReproduced(SamplesRefusal):
    """The backfill's staged set does not reproduce the run's chain (the staging removed, no record written)."""


def copy_dir(run):
    return os.path.join(run, COPY_DIR)


def record_path(run):
    """The run's record - the only name read as one."""
    return os.path.join(copy_dir(run), RECORD)


def is_staging(name):
    return bool(_STAGING_RE.match(name))


def stagings(run):
    """The staging names in ``run``, sorted (none when it cannot be listed)."""
    try:
        return sorted(n for n in os.listdir(run) if is_staging(n))
    except OSError:
        return []


def md5_file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 22), b''):
            h.update(blk)
    return h.hexdigest()


def _utc(t):
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t))


def _tempnames():
    """core/tempnames.py (placement's temporary names), loaded from its file: the re-judging child cannot import the package."""
    name = 'exopipe3_tempnames_run_samples'
    if name not in sys.modules:
        import importlib.util
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'core', 'tempnames.py')
        spec = importlib.util.spec_from_file_location(name, p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        sys.modules[name] = m
    return sys.modules[name]


def chain_names(c=None):
    """The file names the chain's load_samples reads (catalog_snapshot.SAMPLES = clv15 phase3_snapshot_chain.SAMPLES, bound by
    tier_snapshot's binding check), in its order."""
    if c is None:
        from exopipe.phase3.config import CFG
        c = CFG.catalog_snapshot
    return [fn for _tag, fn in c.SAMPLES]


def verified(run, names=None, at=None):
    """(directory, record) of ``run``'s sample catalogs, every recorded copy checked against the record (md5 and size).
    ``names``: the file names the reader will read (the chain's SAMPLES) - the record must list exactly these. ``at``: a staging
    of ``run`` (the backfill's reproduction reads the staged set) - nothing else; default the run's own set. SamplesRefusal when
    the record is missing or malformed, a copy is missing or differs, the directory holds a file the record does not list (a
    temporary file of placement's naming for one of its files excepted), or the names differ. Never a fallback to another
    directory."""
    tag = os.path.basename(os.path.normpath(run))
    if at is None:
        d = copy_dir(run)
    else:
        d = os.path.abspath(at)
        if os.path.dirname(d) != os.path.abspath(run) or not is_staging(os.path.basename(d)):
            raise SamplesRefusal('%s is not a staging of %s (%s.<UTC>.<pid>.<hex> in the run): no other directory is read'
                                 % (at, tag, STAGING))
    where = os.path.basename(d)
    rp = os.path.join(d, RECORD)
    if not os.path.isfile(rp):
        if at is None:
            raise SamplesRefusal('%s has no %s/%s: the sample catalogs its chain read are not recorded, so its catalog route '
                                 'cannot be reconstructed (a run chained before the record existed needs the one-time backfill, '
                                 '`exopipe3 catalog --tier snapshot --stages %s`)' % (tag, COPY_DIR, RECORD, STAGE))
        raise SamplesRefusal('%s staging %s has no %s' % (tag, where, RECORD))
    try:
        with open(rp) as f:
            rec = json.load(f, object_pairs_hook=collections.OrderedDict)
    except (OSError, ValueError) as e:
        raise SamplesRefusal('%s %s/%s cannot be read: %s' % (tag, where, RECORD, e))
    files = rec.get('files') if isinstance(rec, dict) else None
    if not isinstance(rec, dict) or rec.get('format') != FORMAT or rec.get('how') not in HOWS or not isinstance(files, list) \
            or not files or not all(isinstance(x, dict) and x.get('name') and x.get('md5') and isinstance(x.get('bytes'), int)
                                    for x in files):
        raise SamplesRefusal('%s %s/%s is malformed (format %s, how %s): it needs format %d, how in %s and a non-empty list of '
                             'files with name, md5 and bytes' % (tag, where, RECORD,
                                                                rec.get('format') if isinstance(rec, dict) else None,
                                                                rec.get('how') if isinstance(rec, dict) else None, FORMAT,
                                                                '/'.join(HOWS)))
    listed = [x['name'] for x in files]
    if len(set(listed)) != len(listed) or any(n != os.path.basename(n) or n.startswith('.') or n == RECORD for n in listed):
        raise SamplesRefusal('%s %s/%s lists a file twice or a name that is not a plain catalog file name: %s'
                             % (tag, where, RECORD, listed))
    if names is not None and sorted(listed) != sorted(names):
        raise SamplesRefusal('%s %s/%s lists %s; the chain reads %s (missing %s, extra %s)' % (
            tag, where, RECORD, sorted(listed), sorted(names), sorted(set(names) - set(listed)), sorted(set(listed) - set(names))))
    own = set(listed) | {RECORD}
    T = _tempnames()
    extra = sorted(n for n in os.listdir(d) if n not in own and T.product_of(n) not in own)
    if extra:
        raise SamplesRefusal('%s %s/ holds files its record does not list: %s' % (tag, where, extra))
    for x in files:
        p = os.path.join(d, x['name'])
        if not os.path.isfile(p):
            raise SamplesRefusal('%s recorded sample catalog %s/%s is missing' % (tag, where, x['name']))
        n, m = os.path.getsize(p), md5_file(p)
        if (n, m) != (x['bytes'], x['md5']):
            raise SamplesRefusal('%s recorded sample catalog %s/%s differs from its record: %d bytes, md5 %s; recorded %d bytes, '
                                 'md5 %s' % (tag, where, x['name'], n, m, x['bytes'], x['md5']))
    return d, rec


def md5_map(rec):
    """{file name: md5} of a record, in its order."""
    return collections.OrderedDict((x['name'], x['md5']) for x in rec['files'])


def stage(run, src_dir, names, how, why=None, ledger=None):
    """Copy ``names`` from ``src_dir`` into a new staging of ``run`` (core/placement.copy, verified), write its record and read it
    back (verified(at=staging)). ``how``: 'chain' or 'backfill' (needs ``why``; ``ledger`` optional). Refused (SamplesRefusal,
    nothing changed) when the run has a set already (written once), a named file is missing from ``src_dir`` (the chain would skip
    it silently) or no names are given. A staging left in ``run`` by a killed writer is removed first, by its name alone. On ANY
    exception afterwards the staging this call made is removed (nothing else) and the exception re-raised (a placement refusal as
    SamplesRefusal). Returns (staging directory, record)."""
    from exopipe.phase3.core import placement
    tag = os.path.basename(os.path.normpath(run))
    if how not in HOWS:
        raise ValueError('how %r: one of %s' % (how, HOWS))
    if how == 'backfill' and not (why or '').strip():
        raise SamplesRefusal('a backfill records why the named directory holds the catalogs the chain read: give the reason')
    if not os.path.isdir(run):
        raise SamplesRefusal('%s is not a directory' % run)
    if os.path.lexists(copy_dir(run)):
        raise SamplesRefusal('%s already has %s/: a run\'s sample-catalog record is written once (nothing changed)'
                             % (tag, COPY_DIR))
    if not names:
        raise SamplesRefusal('no sample catalog names given')
    missing = [n for n in names if not os.path.isfile(os.path.join(src_dir, n))]
    if missing:
        raise SamplesRefusal('%s lacks %s: the chain would skip a missing catalog silently - refused, nothing written'
                             % (src_dir, missing))
    for n in stagings(run):                     # a killed writer's leftover (the name alone decides; no other name is touched)
        p = os.path.join(run, n)
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p)
        else:
            os.remove(p)
    sd = os.path.join(run, '%s.%s.%d.%s' % (STAGING, time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()), os.getpid(),
                                            os.urandom(4).hex()))
    if os.path.lexists(sd):
        raise SamplesRefusal('%s exists already (nothing changed)' % sd)
    try:
        placement.make_dirs(sd)
        files = []
        for n in names:
            src = os.path.join(src_dir, n)
            dst = os.path.join(sd, n)
            m = placement.copy(src, dst, times=True)
            st = os.stat(dst)
            files.append(collections.OrderedDict([('name', n), ('md5', m), ('bytes', st.st_size),
                                                  ('source', os.path.abspath(src)), ('source_mtime_utc', _utc(st.st_mtime))]))
        rec = collections.OrderedDict([
            ('format', FORMAT),
            ('what', 'the sample catalogs the snapshot chain read (phase3_snapshot_chain.load_samples, --samples-dir %s/)'
             % COPY_DIR),
            ('how', how), ('written_utc', _utc(time.time())), ('source_dir', os.path.abspath(src_dir)), ('copy_dir', COPY_DIR),
            ('files', files)])
        if how == 'backfill':
            rec['why'] = why.strip()
            if ledger:
                rec['ledger'] = ledger
        with placement.atomic(os.path.join(sd, RECORD)) as tmp:
            with open(tmp, 'w') as f:
                json.dump(rec, f, indent=1)
                f.write('\n')
        verified(run, names, at=sd)
    except BaseException as e:
        shutil.rmtree(sd, ignore_errors=True)   # this call's staging (a new name: nothing else is in it)
        if isinstance(e, placement.PlacementRefusal):
            raise SamplesRefusal('%s: a sample catalog copy was not placed (the staging removed): %s' % (tag, e)) from e
        raise
    return sd, rec


def commit(run, sd, names):
    """Rename the staging ``sd`` into place as <run>/sample_catalogs/ (one rename: the record appears with its copies) and read it
    back (verified). Refused when the run has a set already. A read-back refusal removes the set this call placed. Returns the
    record."""
    final = copy_dir(run)
    if os.path.lexists(final):
        raise SamplesRefusal('%s already has %s/: a run\'s sample-catalog record is written once'
                             % (os.path.basename(os.path.normpath(run)), COPY_DIR))
    os.rename(sd, final)
    try:
        return verified(run, names)[1]
    except BaseException:
        shutil.rmtree(final, ignore_errors=True)
        raise


def write(run, src_dir, names, how, why=None, ledger=None):
    """stage() then commit(): the run's set, written once. Any exception leaves nothing of this call behind. Returns the
    record."""
    sd, _rec = stage(run, src_dir, names, how, why=why, ledger=ledger)
    try:
        return commit(run, sd, names)
    except BaseException:
        shutil.rmtree(sd, ignore_errors=True)   # still staged when the rename did not happen
        raise


def backfill(run, src_dir, names, why, reproduce, ledger=None):
    """The one-time backfill of ``run``: stage(how='backfill'), then ``reproduce(staging)`` -> (ok, line) - the re-judging with
    no removals on the staged set - and commit() only when ok. Not ok: NotReproduced, the staging removed, no record. Any
    exception: the staging removed, re-raised. Returns (record, line)."""
    sd, _rec = stage(run, src_dir, names, 'backfill', why=why, ledger=ledger)
    try:
        ok, line = reproduce(sd)
        if not ok:
            raise NotReproduced('the catalogs of %s do not reproduce the chain of %s (%s) - the staged set removed, no record '
                                'written' % (src_dir, os.path.basename(os.path.normpath(run)), line))
        rec = commit(run, sd, names)
    except BaseException:
        shutil.rmtree(sd, ignore_errors=True)
        raise
    return rec, line


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
    [--samples-ledger ID] --out RUN` - one unit '<run>.samples-backfill': backfill() with the reproduction check rejudge.py
    --reproduce RUN CLV15 STAGING under catalog_omission.REJUDGE_PYTHON (log <RUN>/_records/catalog/<run>/
    samples_reproduce_<UTC>.log). SB1 not a chain run; SB2 the set cannot be written (a set exists, a catalog is missing, a copy
    is not verified); SB3 the staged set does not reproduce the chain (the staging removed, no record)."""
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
        print('[%s.%s] copy %s from %s into a staging %s/%s.<UTC>.<pid>.<hex>/ with its %s; then %s %s --reproduce %s %s '
              '<staging> (no removals); renamed to %s/%s/ only when that reproduces the chain'
              % (tag, STAGE, ', '.join(names), src, rd, STAGING, RECORD, CFG.catalog_omission.REJUDGE_PYTHON, script, rd, clv,
                 rd, COPY_DIR))
        return 0
    with run.unit('%s.%s' % (tag, STAGE), tier='snapshot', stage=STAGE) as u:
        if not os.path.isfile(os.path.join(rd, 'chain_summary.json')):
            raise Refusal('SB1: %s has no chain_summary.json - not a snapshot chain run' % rd)
        log = os.path.join(rd, '_records', 'catalog', tag, 'samples_reproduce_%s.log' % time.strftime('%Y%m%dT%H%M%SZ',
                                                                                                        time.gmtime()))

        def reproduce(sd):
            os.makedirs(os.path.dirname(log), exist_ok=True)
            lr = legacy.run(script, ['--reproduce', rd, clv, sd], python=CFG.catalog_omission.REJUDGE_PYTHON, log=log, unit=u,
                            timeout=CFG.catalog_omission.REJUDGE_TIMEOUT_S, refusal_codes=(2,))
            line = ''
            if os.path.exists(log):
                with open(log) as f:
                    line = ([ln.strip() for ln in f if ln.startswith(('REPRODUCE-OK', 'REFUSING'))] or [''])[-1]
            ok = bool(lr.ok) and line.startswith('REPRODUCE-OK')
            u.check('SB3.reproduces-chain', ok, detail=(line or lr.reason or '')[:500])
            return ok, line or lr.reason
        try:
            rec, line = backfill(rd, src, names, why, reproduce, ledger=getattr(args, 'samples_ledger', None))
        except NotReproduced as e:
            raise Refusal('SB3: %s (log %s)' % (e, log))
        except SamplesRefusal as e:
            raise Refusal('SB2: %s' % e)
        for x in rec['files']:
            u.add_input(x['source'])
        u.note('%s: %d sample catalogs recorded from %s (%s); %s' % (
            tag, len(rec['files']), src, ', '.join('%s %s' % kv for kv in md5_map(rec).items()), line))
        u.meta['samples'] = dict(md5_map(rec))
        u.add_output(record_path(rd), 'sample_catalogs', tier='snapshot')
    return None


def legacy_clv15():
    """The clv15 root the re-judging child imports the chain from (core.env.legacy_root)."""
    from exopipe.phase3.core import env
    return env.legacy_root('clv15')
