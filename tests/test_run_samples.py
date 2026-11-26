#!/usr/bin/env python3
"""Data-free tests of the sample-catalog record of a snapshot chain run (cutover row 70; lib/catalog/run_samples.py) and of the
readers that depend on it: the re-judging (lib/catalog/rejudge.recorded_samples), the omit stage's gate G9b
(deep_omission.samples_gate), the combined view's G-V8, the chain stage's arguments (tier_snapshot.chain_args --samples-dir) and
the light-curve extractor with a chain run (lib/curves/workers.samples_dir, steps/curves G19).

The record must name exactly the bytes the chain read, so a later change of the live catalogs cannot change a re-judging: a run is
re-judged on its own copies or refused - a missing record, a missing copy, a copy whose md5 or size differs, a file the record does
not list, a record whose names are not the chain's SAMPLES. Nothing falls back to another directory. A write leaves nothing behind
when it fails; the backfill's record appears only after its reproduction check passed, and a killed backfill leaves no record.

Run: python tests/test_run_samples.py [test names]   (exit 0 = all passed, 1 = a failure, 2 = imports missing)
"""
import hashlib
import inspect
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

_IMPORT_ERROR = None
try:
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import placement, validate
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.catalog import rejudge as RJ
    from exopipe.phase3.lib.catalog import run_samples as RS
except Exception as e:  # noqa: BLE001
    _IMPORT_ERROR = e
    if __name__ != '__main__':
        raise

NAMES = ['Hot_Warm_Jupiters.csv', '10pc.csv', '25pc.csv', 'LPTs.csv', 'flare_ucd.csv', 'sydney_stars.csv', 'pulsars_atnf.csv']


def _md5(p):
    with open(p, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


class World(object):
    """A samples directory (each catalog a small CSV with its own content) and an empty chain run directory."""

    def __init__(self):
        self.root = tempfile.mkdtemp(prefix='run_samples_')
        self.src = os.path.join(self.root, 'clv15')
        self.run = os.path.join(self.root, 'Run_SNAPTEST_20261126')
        os.makedirs(self.src)
        os.makedirs(self.run)
        for i, n in enumerate(NAMES):
            with open(os.path.join(self.src, n), 'w') as f:
                f.write('name,ra,dec\n%s_a,%d.0,10.0\n' % (n[:-4], i))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        shutil.rmtree(self.root, ignore_errors=True)

    def write(self, how='chain', **kw):
        return RS.write(self.run, self.src, NAMES, how, **kw)

    def copy(self, n):
        return os.path.join(RS.copy_dir(self.run), n)


def refused(fn, *a, **kw):
    """The SamplesRefusal text of fn(*a, **kw); AssertionError when it does not refuse."""
    try:
        fn(*a, **kw)
    except RS.SamplesRefusal as e:
        return str(e)
    raise AssertionError('%s did not refuse' % fn.__name__)


# ---------------------------------------------------------------------------------------------------------------------

def test_config_samples_are_the_chain_list():
    """catalog_snapshot.SAMPLES (bound to phase3_snapshot_chain.SAMPLES by tier_snapshot.LIT) names the seven catalogs in the
    chain's order, and the binding is declared."""
    from exopipe.phase3.lib.catalog import tier_snapshot as P
    assert RS.chain_names(CFG.catalog_snapshot) == NAMES, RS.chain_names(CFG.catalog_snapshot)
    assert P.LIT['catalog_snapshot.SAMPLES'] == ('clv15', P.CH, 'SAMPLES')
    assert 'SAMPLES' in CFG.catalog_snapshot.PROTECTED_KEYS
    assert not hasattr(CFG.catalog_omission, 'REJUDGE_SAMPLES_DIR'), 'the live-directory fallback key is gone'


def test_write_copies_and_records_md5():
    """The copies are byte-identical to the sources; the record lists each with the source's md5, size and path; verified()
    returns the copy directory and the record."""
    with World() as w:
        rec = w.write()
        assert [x['name'] for x in rec['files']] == NAMES
        for x in rec['files']:
            src = os.path.join(w.src, x['name'])
            assert x['md5'] == _md5(src) == _md5(w.copy(x['name'])), x
            assert x['bytes'] == os.path.getsize(src)
            assert x['source'] == os.path.abspath(src)
        assert rec['how'] == 'chain' and rec['format'] == RS.FORMAT and rec['copy_dir'] == RS.COPY_DIR
        d, back = RS.verified(w.run, NAMES)
        assert d == RS.copy_dir(w.run) and back == json.load(open(RS.record_path(w.run)))
        assert RS.md5_map(back) == {n: _md5(os.path.join(w.src, n)) for n in NAMES}
        assert os.listdir(w.run) == [RS.COPY_DIR], 'one entry: the set (no staging left)'
        assert sorted(os.listdir(RS.copy_dir(w.run))) == sorted(NAMES + [RS.RECORD])


def test_live_change_does_not_change_the_record():
    """A later change of the live catalog (the 2026-11-18 LPTs.csv case) leaves the run's copies and record as they were."""
    with World() as w:
        rec = w.write()
        before = RS.md5_map(rec)['LPTs.csv']
        with open(os.path.join(w.src, 'LPTs.csv'), 'a') as f:
            f.write('J0410+4640,62.5,46.7\n')
        _d, back = RS.verified(w.run, NAMES)
        assert RS.md5_map(back)['LPTs.csv'] == before != _md5(os.path.join(w.src, 'LPTs.csv'))


def test_refuses_missing_record():
    with World() as w:
        assert 'has no sample_catalogs/sample_catalogs.json' in refused(RS.verified, w.run, NAMES)
        w.write()
        os.remove(RS.record_path(w.run))
        assert 'has no sample_catalogs/sample_catalogs.json' in refused(RS.verified, w.run, NAMES)


def test_refuses_missing_copy():
    with World() as w:
        w.write()
        os.remove(w.copy('LPTs.csv'))
        assert 'LPTs.csv is missing' in refused(RS.verified, w.run, NAMES)


def test_refuses_changed_copy():
    """A copy substituted by another version (another size) or edited in place (same size, other bytes) refuses on md5."""
    with World() as w:
        w.write()
        os.chmod(w.copy('LPTs.csv'), 0o644)
        with open(w.copy('LPTs.csv'), 'a') as f:
            f.write('J1101+5534,165.3,55.6\n')
        assert 'differs from its record' in refused(RS.verified, w.run, NAMES)
    with World() as w:
        w.write()
        p = w.copy('10pc.csv')
        os.chmod(p, 0o644)
        b = open(p, 'rb').read()
        open(p, 'wb').write(b.replace(b'10.0', b'11.0'))
        assert os.path.getsize(p) == len(b)
        assert 'differs from its record' in refused(RS.verified, w.run, NAMES)


def test_refuses_unlisted_file_and_name_mismatch():
    with World() as w:
        w.write()
        open(w.copy('extra.csv'), 'w').write('x\n')
        assert "does not list: ['extra.csv']" in refused(RS.verified, w.run, NAMES)
    with World() as w:
        w.write()
        assert 'missing' in refused(RS.verified, w.run, NAMES + ['new.csv'])


def test_refuses_malformed_record():
    with World() as w:
        w.write()
        rp = RS.record_path(w.run)
        rec = json.load(open(rp))
        for bad in (dict(rec, format=99), dict(rec, how='hand'), dict(rec, files=[])):
            os.chmod(rp, 0o644)
            json.dump(bad, open(rp, 'w'))
            assert 'malformed' in refused(RS.verified, w.run, NAMES)
        open(rp, 'w').write('{not json')
        assert 'cannot be read' in refused(RS.verified, w.run, NAMES)


def test_write_once_and_complete():
    """A record is written once; a missing source catalog refuses before anything is written; a backfill needs its reason."""
    with World() as w:
        w.write()
        assert 'already has' in refused(w.write)
    with World() as w:
        os.remove(os.path.join(w.src, 'flare_ucd.csv'))
        assert 'flare_ucd.csv' in refused(w.write)
        assert not os.path.lexists(RS.record_path(w.run)) and not os.path.lexists(RS.copy_dir(w.run))
    with World() as w:
        assert 'give the reason' in refused(w.write, 'backfill')
        rec = w.write('backfill', why='the directory the chain read before 2026-11-18', ledger='FIX-TEST')
        assert rec['how'] == 'backfill' and rec['why'].startswith('the directory') and rec['ledger'] == 'FIX-TEST'


def _foreign(w):
    """Entries of the run that no write may touch: a chain output and an unrelated directory."""
    open(os.path.join(w.run, 'chain_summary.json'), 'w').write('{}\n')
    os.makedirs(os.path.join(w.run, '_records', 'x'))
    return sorted(os.listdir(w.run))


def test_write_failure_leaves_nothing_then_retry():
    """Any exception inside write() - an OSError in the middle of the copies, a placement refusal of the directory, a failed
    record write, the read-back refusing after the rename - removes only what that call made and re-raises; a retry then
    succeeds."""
    with World() as w:
        before = _foreign(w)
        real_copy, calls = placement.copy, []

        def copy_fails_third(src, dst, **kw):
            calls.append(dst)
            if len(calls) == 3:
                raise OSError(5, 'Input/output error (injected)', dst)
            return real_copy(src, dst, **kw)
        placement.copy = copy_fails_third
        try:
            try:
                w.write()
                raise AssertionError('the OSError was not re-raised')
            except OSError as e:
                assert 'injected' in str(e) and not isinstance(e, RS.SamplesRefusal), e
        finally:
            placement.copy = real_copy
        assert len(calls) == 3 and sorted(os.listdir(w.run)) == before, os.listdir(w.run)
        rec = w.write()                                            # the retry
        assert RS.md5_map(RS.verified(w.run, NAMES)[1]) == RS.md5_map(rec)
    with World() as w:
        before = _foreign(w)
        real = placement.make_dirs

        def make_dirs_refuses(path):
            real(path)
            raise placement.PlacementRefusal('set-group-ID bit lost (injected)')
        placement.make_dirs = make_dirs_refuses
        try:
            assert 'injected' in refused(w.write)
        finally:
            placement.make_dirs = real
        assert sorted(os.listdir(w.run)) == before
        real_dump = RS.json.dump

        def dump_fails(*a, **kw):
            raise OSError(28, 'No space left on device (injected)')
        RS.json.dump = dump_fails
        try:
            try:
                w.write()
                raise AssertionError('the record-write OSError was not re-raised')
            except OSError as e:
                assert 'injected' in str(e)
        finally:
            RS.json.dump = real_dump
        assert sorted(os.listdir(w.run)) == before
        real_verified = RS.verified

        def final_read_back_refuses(run, names=None, at=None):
            if at is None:
                raise RS.SamplesRefusal('read-back refused (injected)')
            return real_verified(run, names, at=at)
        RS.verified = final_read_back_refuses
        try:
            assert 'injected' in refused(w.write)
        finally:
            RS.verified = real_verified
        assert sorted(os.listdir(w.run)) == before, 'the set placed by the failed call is removed'
        w.write()
        assert sorted(os.listdir(w.run)) == sorted(before + [RS.COPY_DIR])


def test_staging_is_never_a_record():
    """A staged set is read only when named as the run's staging (the backfill's reproduction); verified() of the run, G9b and
    the content test see no record until the rename. No other directory is accepted as a set."""
    with World() as w:
        open(os.path.join(w.run, 'chain_summary.json'), 'w').write('{}\n')
        sd, rec = RS.stage(w.run, w.src, NAMES, 'chain')
        assert RS.is_staging(os.path.basename(sd)) and RS.stagings(w.run) == [os.path.basename(sd)]
        assert 'has no sample_catalogs/sample_catalogs.json' in refused(RS.verified, w.run, NAMES)
        d, back = RS.verified(w.run, NAMES, at=sd)
        assert d == sd and RS.md5_map(back) == RS.md5_map(rec)
        res = validate.run('sample_catalogs', os.path.join(sd, RS.RECORD))
        assert not res[0]['passed'] and 'a staging is never a record' in res[0]['detail'], res
        assert not validate.run('sample_catalogs', w.run)[0]['passed']
        assert 'not a staging of' in refused(RS.verified, w.run, NAMES, at=w.src)
        other = os.path.join(w.root, os.path.basename(sd))
        shutil.copytree(sd, other)
        assert 'not a staging of' in refused(RS.verified, w.run, NAMES, at=other)
        assert RS.commit(w.run, sd, NAMES)['files'] == rec['files']
        assert RS.stagings(w.run) == [] and RS.verified(w.run, NAMES)[0] == RS.copy_dir(w.run)


_KILLED = """
import os, signal, sys
sys.path.insert(0, %r)
from exopipe.phase3.lib.catalog import run_samples as RS
RS.backfill(%r, %r, %r, 'interrupt test', lambda sd: os.kill(os.getpid(), signal.SIGKILL))
"""


def test_backfill_killed_between_write_and_check_then_retry():
    """A backfill killed after its set was written and before its reproduction check finished leaves no record (only its
    staging); the retry removes that staging by its name, passes its check and places the record."""
    with World() as w:
        before = _foreign(w)
        p = subprocess.run([sys.executable, '-c', _KILLED % (REPO, w.run, w.src, NAMES)], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True, timeout=300)
        assert p.returncode == -signal.SIGKILL, (p.returncode, p.stderr[-1500:])
        left = RS.stagings(w.run)
        assert len(left) == 1 and not os.path.lexists(RS.copy_dir(w.run)), os.listdir(w.run)
        assert 'has no sample_catalogs' in refused(RS.verified, w.run, NAMES)
        assert not validate.run('sample_catalogs', w.run)[0]['passed']
        seen = []

        def reproduce(sd):
            seen.append(sd)
            assert not os.path.lexists(RS.copy_dir(w.run)), 'no record before the check passed'
            RS.verified(w.run, NAMES, at=sd)
            return True, 'REPRODUCE-OK (test)'
        rec, line = RS.backfill(w.run, w.src, NAMES, 'retry after a kill', reproduce, ledger='FIX-TEST')
        assert line == 'REPRODUCE-OK (test)' and os.path.basename(seen[0]) != left[0]
        assert sorted(os.listdir(w.run)) == sorted(before + [RS.COPY_DIR]), os.listdir(w.run)
        assert rec['how'] == 'backfill' and rec['ledger'] == 'FIX-TEST'
        assert RS.md5_map(RS.verified(w.run, NAMES)[1]) == RS.md5_map(rec)


def test_backfill_not_reproduced_writes_no_record():
    """A staged set that does not reproduce the chain (the live catalogs after 2026-11-18) is removed; no record; a reproduction
    check that raises removes it too."""
    with World() as w:
        before = _foreign(w)
        try:
            RS.backfill(w.run, w.src, NAMES, 'live catalogs', lambda sd: (False, 'REFUSING: 2 served sources differ'))
            raise AssertionError('not refused')
        except RS.NotReproduced as e:
            assert 'REFUSING: 2 served sources differ' in str(e) and 'no record' in str(e)
        assert sorted(os.listdir(w.run)) == before

        def boom(sd):
            raise RuntimeError('child crashed (injected)')
        try:
            RS.backfill(w.run, w.src, NAMES, 'live catalogs', boom)
            raise AssertionError('not raised')
        except RuntimeError:
            pass
        assert sorted(os.listdir(w.run)) == before
        assert 'give the reason' in refused(RS.backfill, w.run, w.src, NAMES, ' ', boom)


def test_leftover_temporary_file_does_not_break_a_valid_record():
    """verified() does not count a temporary file of placement's naming (core/tempnames.py) of one of the set's own files; any
    other unlisted name still refuses."""
    with World() as w:
        w.write()
        d = RS.copy_dir(w.run)
        for n in ('LPTs.csv.4242.0a1b2c3d.tmp', RS.RECORD + '.4242.deadbeef.tmp'):
            open(os.path.join(d, n), 'w').write('partial')
            assert RS.verified(w.run, NAMES)[0] == d
        for n in ('other.csv.4242.0a1b2c3d.tmp', 'LPTs.csv.tmp'):
            open(os.path.join(d, n), 'w').write('partial')
            assert n in refused(RS.verified, w.run, NAMES)
            os.remove(os.path.join(d, n))


def test_content_test_class():
    """Class sample_catalogs: passes a verified record (value = the number of catalogs) and fails a changed copy."""
    with World() as w:
        w.write()
        prev = CFG.catalog_snapshot.SAMPLES
        res = validate.run('sample_catalogs', RS.record_path(w.run))
        assert res and all(r['passed'] for r in res) and res[0]['value'] == len(NAMES), res
        assert RS.chain_names() == NAMES and CFG.catalog_snapshot.SAMPLES == prev
        os.remove(w.copy('25pc.csv'))
        res = validate.run('sample_catalogs', RS.record_path(w.run))
        assert not res[0]['passed'] and '25pc.csv is missing' in res[0]['detail'], res
    with World() as w:                       # a run directory: the backfill worklist (`exopipe3 verify ... --glob`)
        res = validate.run('sample_catalogs', w.run)
        assert not res[0]['passed'] and 'not a snapshot chain run' in res[0]['detail'], res
        open(os.path.join(w.run, 'chain_summary.json'), 'w').write('{}\n')
        res = validate.run('sample_catalogs', w.run + '/')
        assert not res[0]['passed'] and 'has no sample_catalogs/sample_catalogs.json' in res[0]['detail'], res
        w.write()
        res = validate.run('sample_catalogs', w.run + '/')
        assert res[0]['passed'] and res[0]['value'] == len(NAMES), res


def _fake_chain(log):
    """A stand-in for phase3_snapshot_chain: SAMPLES and load_samples (which skips a missing file, as the chain does)."""
    C = types.ModuleType('fake_chain')
    C.SAMPLES = [(n[:-4], n) for n in NAMES]

    def load_samples(d):
        log.append(d)
        out = {}
        for tag, fn in C.SAMPLES:
            p = os.path.join(d, fn)
            if os.path.exists(p):
                out[tag] = open(p).read()
        return out
    C.load_samples = load_samples
    return C


def test_rejudge_reads_the_recorded_copies():
    """rejudge.recorded_samples reads the directory of the set verified() returned (the run's own, or its staging) - never the
    source directory - hashes it once after the read, and refuses with RejudgeRefusal on a copy changed meanwhile or a directory
    that is not the run's."""
    with World() as w:
        log = []
        C = _fake_chain(log)
        w.write()
        samples = RS.verified(w.run, NAMES)
        RSc = RJ.run_samples_module()              # the child's copy of run_samples (loaded from its file)
        n_hash, real_md5 = [], RSc.md5_file

        def counting_md5(p):
            n_hash.append(p)
            return real_md5(p)
        RSc.md5_file = counting_md5
        try:
            cats = RJ.recorded_samples(w.run, C, samples)
        finally:
            RSc.md5_file = real_md5
        assert log == [RS.copy_dir(w.run)] and len(cats) == len(NAMES)
        assert len(n_hash) == len(NAMES), 'one post-read pass: %d files hashed' % len(n_hash)
        os.chmod(w.copy('LPTs.csv'), 0o644)
        shutil.copyfile(os.path.join(w.src, '10pc.csv'), w.copy('LPTs.csv'))
        try:
            RJ.recorded_samples(w.run, C, samples)
            raise AssertionError('changed copy: not refused')
        except RJ.RejudgeRefusal as e:
            assert 'LPTs.csv differs from its record' in str(e)
        try:
            RJ.recorded_samples(w.run, C, (w.src, samples[1]))
            raise AssertionError('a directory passed by hand: not refused')
        except RJ.RejudgeRefusal as e:
            assert 'not a staging of' in str(e)
    with World() as w:
        log = []
        sd, _rec = RS.stage(w.run, w.src, NAMES, 'backfill', why='test')
        RJ.recorded_samples(w.run, _fake_chain(log), RS.verified(w.run, NAMES, at=sd))
        assert log == [sd]


def test_rejudge_takes_the_verified_set():
    """rejudge() takes the (directory, record) pair verified() returned - no samples-directory parameter (production's run_inputs
    samples_dir; 352ef13's per-run map)."""
    assert list(inspect.signature(RJ.rejudge).parameters) == ['run', 'removed', 'samples', 'log']
    assert list(inspect.signature(RJ.reproduce_main).parameters) == ['run', 'clv15', 'staging']


def test_samples_gate_first_in_the_omit_stage():
    """G9b at the start of the omit stage: a run without its record refuses before the served identities, Card 2 or the
    decisions are built; a binding mismatch of catalog_snapshot.SAMPLES refuses once, before any child; with records the gate
    returns each run's md5s (the parent's one hashing pass)."""
    from exopipe.phase3.lib.catalog import deep_omission as DO, tier_sublst as TSL
    with World() as w:
        real_sid, real_rg, real_bc = DO.served_identities, DO._record_gate, TSL.binding_check

        def built(*a, **kw):
            raise AssertionError('served identities built before G9b')
        DO.served_identities, DO._record_gate = built, (lambda rdir: None)
        TSL.binding_check = lambda table: [(k, 'ok', None, None, 'test') for k in table]   # (no clv15 in CI)
        try:
            try:
                DO.run_snapshot([w.run], os.path.join(w.root, 'omit_out'), rdir=w.root, proto=w.root)
                raise AssertionError('not refused')
            except Refusal as e:
                assert str(e).startswith('G9b: %s: ' % DO.batch_tag(w.run)) and 'has no sample_catalogs' in str(e), e
            assert not os.path.exists(os.path.join(w.root, 'omit_out.rejudge')), 'nothing built'
            w.write()
            got = DO.samples_gate([w.run])
            assert got == {w.run: RS.md5_map(RS.verified(w.run, NAMES)[1])}, got
            TSL.binding_check = lambda table: [(k, 'MISMATCH', 'cfg', 'mod', 'clv15/phase3_snapshot_chain.py:SAMPLES')
                                               for k in table]
            try:
                DO.samples_gate([w.run])
                raise AssertionError('binding mismatch not refused')
            except Refusal as e:
                assert str(e).startswith('G9b: catalog_snapshot.SAMPLES MISMATCH'), e
        finally:
            DO.served_identities, DO._record_gate, TSL.binding_check = real_sid, real_rg, real_bc
        key = DO.samples_record_key()
        assert key == os.path.join(RS.COPY_DIR, RS.RECORD)
        inp = DO.rejudge_inputs([w.run])[DO.batch_tag(w.run)]
        assert inp[key] is not None and all(inp[os.path.join(RS.COPY_DIR, n)] is not None for n in NAMES)


def test_view_refuses_an_omit_product_without_records():
    """G-V8: an omit summary written before the records existed refuses by that name (no migration: v2 built no view yet)."""
    from exopipe.phase3.lib.serve import snapshot_view as SV
    from exopipe.phase3.lib.catalog import deep_omission as DO

    class U(object):
        def refuse(self, why):
            raise Refusal(why)
    plan = types.SimpleNamespace(omit_dir='/scratch/omit', runs=[])
    key = DO.samples_record_key()
    old_inputs = {'Run_SNAPB4G3_20261104': {'detection_record.csv.gz': [1, 2]}}
    for summ in ({}, {'rejudge_inputs': old_inputs}, {'rejudge_samples': {}, 'rejudge_inputs': old_inputs}):
        try:
            SV.check_decisions_current(U(), plan, summ, '/scratch/recs')
            raise AssertionError('not refused')
        except Refusal as e:
            assert str(e).startswith('G-V8 omit product predates sample-catalog records; backfill the member runs and '
                                     're-run the omit stage'), e
    summ = {'rejudge_samples': {}, 'rejudge_inputs': {'Run_X': {key: [1, 2]}}, 'refusals': '/elsewhere', 'proto': '/x'}
    try:
        SV.check_decisions_current(U(), plan, summ, '/scratch/recs')
    except Refusal as e:
        assert 'predates' not in str(e) and 'was decided from the refusal record' in str(e), e


def test_curves_with_a_chain_run_reads_its_catalogs():
    """A reader of a snapshot run after its record exists reads the run's copies: the light-curve extractor with
    curves.CHAIN_RUN_DIR takes --samples-dir <run>/sample_catalogs and G19 refuses a run without a valid record; without it
    (production: no run context) paths.CLV15_ROOT as before."""
    from exopipe.phase3.lib.curves import workers as W
    from exopipe.phase3.steps import curves as S
    C = CFG.curves
    assert C.CHAIN_RUN_DIR is None and W.samples_dir(C, CFG.paths) == CFG.paths.CLV15_ROOT
    with World() as w:
        fake = types.SimpleNamespace(**{k: getattr(C, k) for k in ('MASTER_CATALOG_DIR', 'SAMPLES')}, CHAIN_RUN_DIR=w.run)
        assert W.samples_dir(fake, CFG.paths) == RS.copy_dir(w.run)
        a = W.argv(fake, CFG.paths, '/clv15/phase3_snapshot_lightcurves.py', '05h_2024-12-29', '46MHz', 'I', None, '/o')
        assert a[a.index('--samples-dir') + 1] == RS.copy_dir(w.run) and a[a.index('--run-dir') + 1] == w.run, a
        out = os.path.join(w.root, 'curves_out')
        try:
            CFG.authorize([], out=out)
            CFG.set('curves.CHAIN_RUN_DIR', w.run)
            probs = S._config_gates(CFG, out)
            assert any(p.startswith('G19: ') and 'has no sample_catalogs' in p for p in probs), probs
            w.write()
            assert not any(p.startswith('G19') for p in S._config_gates(CFG, out))
        finally:
            CFG.reset()


def test_chain_args_samples_dir():
    """The chain stage passes --samples-dir <run>/sample_catalogs; the prefit and tie passes keep the chain's default."""
    from exopipe.phase3.lib.catalog import tier_snapshot as P
    c = CFG.catalog_snapshot
    defaults = {'--samples-dir': CFG.paths.CLV15_ROOT}
    a = P.chain_args(CFG, c, '/x/store.csv', '/x/chain.py', defaults=defaults)
    assert '--samples-dir' not in a, a
    a = P.chain_args(CFG, c, '/x/store.csv', '/x/chain.py', defaults=defaults, samples_dir='/r/Run_X/sample_catalogs')
    i = a.index('--samples-dir')
    assert a[i + 1] == '/r/Run_X/sample_catalogs'


def test_backfill_stage_selection():
    """Only `--stages samples-backfill` alone selects the backfill."""
    ns = types.SimpleNamespace
    assert RS.wants(ns(stages='samples-backfill'))
    assert not RS.wants(ns(stages='omit'))
    assert not RS.wants(ns(stages='samples-backfill,omit'))
    assert not RS.wants(ns(stages=None))


def main(argv):
    if _IMPORT_ERROR is not None:
        print(f'imports missing: {type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}')
        return 2
    tests = [(n, f) for n, f in sorted(globals().items(), key=lambda kv: inspect.getsourcelines(kv[1])[1]
                                       if callable(kv[1]) and kv[0].startswith('test_') else 0)
             if callable(f) and n.startswith('test_') and (not argv or n in argv)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f'PASS {n}')
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f'FAIL {n}: {type(e).__name__}: {str(e)[:600]}')
    print(f'{len(tests) - failed} of {len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
