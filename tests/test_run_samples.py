#!/usr/bin/env python3
"""Data-free tests of the sample-catalog record of a snapshot chain run (cutover row 70; lib/catalog/run_samples.py) and of the
readers that depend on it: the re-judging (lib/catalog/rejudge.recorded_samples) and the chain stage's arguments
(tier_snapshot.chain_args --samples-dir).

The record must name exactly the bytes the chain read, so a later change of the live catalogs cannot change a re-judging: a run is
re-judged on its own copies or refused - a missing record, a missing copy, a copy whose md5 or size differs, a file the record does
not list, a record whose names are not the chain's SAMPLES. Nothing falls back to another directory.

Run: python tests/test_run_samples.py [test names]   (exit 0 = all passed, 1 = a failure, 2 = imports missing)
"""
import hashlib
import inspect
import json
import os
import shutil
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

_IMPORT_ERROR = None
try:
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import validate
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
        assert 'has no sample_catalogs.json' in refused(RS.verified, w.run, NAMES)
        w.write()
        os.remove(RS.record_path(w.run))
        assert 'has no sample_catalogs.json' in refused(RS.verified, w.run, NAMES)


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


def test_set_aside_leaves_no_record():
    with World() as w:
        w.write()
        dest = os.path.join(w.run, '_records', 'aside')
        os.makedirs(os.path.dirname(dest))
        moved = RS.set_aside(w.run, dest)
        assert len(moved) == 2 and sorted(os.listdir(dest)) == [RS.COPY_DIR, RS.RECORD]
        refused(RS.verified, w.run, NAMES)


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
    """rejudge.recorded_samples reads the run's copy directory - never the source directory - and refuses with
    RejudgeRefusal on a missing record or a changed copy."""
    with World() as w:
        log = []
        C = _fake_chain(log)
        try:
            RJ.recorded_samples(w.run, C)
            raise AssertionError('no record: not refused')
        except RJ.RejudgeRefusal as e:
            assert 'has no sample_catalogs.json' in str(e)
        assert log == [], 'no catalog read without a record'
        w.write()
        cats = RJ.recorded_samples(w.run, C)
        assert log == [RS.copy_dir(w.run)] and len(cats) == len(NAMES)
        os.chmod(w.copy('LPTs.csv'), 0o644)
        shutil.copyfile(os.path.join(w.src, '10pc.csv'), w.copy('LPTs.csv'))
        try:
            RJ.recorded_samples(w.run, C)
            raise AssertionError('changed copy: not refused')
        except RJ.RejudgeRefusal as e:
            assert 'LPTs.csv differs from its record' in str(e)


def test_rejudge_takes_no_samples_directory():
    """rejudge() has no samples-directory parameter (production's run_inputs samples_dir; 352ef13's per-run map)."""
    assert list(inspect.signature(RJ.rejudge).parameters) == ['run', 'removed', 'log']


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
