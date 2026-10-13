#!/usr/bin/env python3
"""STANDING TEST - the DESIGN v2 infrastructure items (refactor_20261012/DESIGN.md section 11), on synthetic data.

  item 1  config: PROTECTED keys, RULED keys (--ruling-for checked in the fix ledger, or a non-production --out),
          one-entry override grammar, refusal of a whole-dict override that omits entries
  item 2  per-step CONFIG declaration: resume hash over the declared + shared sections; `exopipe3 config --step`
  item 3  record: FITS identity (DATASUM + processing cards), dirty-tree records never resumed, planned outputs and item
          refusals, exit codes (DESIGN 11.5)
  item 4  runner: --code COMMIT read-only worktrees, --set/--config forwarded to the step, chains
  item 5  parity write guard (in-process and, through the sitecustomize, in subprocesses)
  item 6  `exopipe3 verify compare` (FITS, CSV, JSON; permitted-difference signatures; product map)

Every test makes its own scratch directory and removes it. Scratch root: $EXOPIPE3_TEST_SCRATCH, else
/fast/gh/refactor_scratch/infra when it exists and is writable, else the system temp directory. Nothing outside the
scratch directory is written (the fix ledger is read, never written).

Run standalone:  python tests/test_core_v2.py [test_name ...]
Under pytest:    pytest tests/test_core_v2.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
PY = sys.executable
INFRA_SCRATCH = '/fast/gh/refactor_scratch/infra'


def _scratch_root():
    r = os.environ.get('EXOPIPE3_TEST_SCRATCH')
    if r:
        os.makedirs(r, exist_ok=True)
        return r
    if os.path.isdir(INFRA_SCRATCH) and os.access(INFRA_SCRATCH, os.W_OK):
        return INFRA_SCRATCH
    return tempfile.gettempdir()


class Scratch:
    """A scratch directory removed on exit (read-only files made writable first)."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix=f'test_core_v2_{self.name}_', dir=_scratch_root())
        return self.path

    def __exit__(self, *exc):
        for dp, dn, fn in os.walk(self.path):
            for x in dn + fn:
                p = os.path.join(dp, x)
                if not os.path.islink(p):
                    try:
                        os.chmod(p, 0o755 if os.path.isdir(p) else 0o644)
                    except OSError:
                        pass
        try:
            os.chmod(self.path, 0o755)
        except OSError:
            pass
        shutil.rmtree(self.path, ignore_errors=True)


def cli(*args, env=None, cwd=None, timeout=120):
    """Run `python -m exopipe.phase3 ARGS` in a fresh interpreter; returns the CompletedProcess."""
    e = dict(os.environ)
    e['PYTHONPATH'] = REPO + (os.pathsep + e['PYTHONPATH'] if e.get('PYTHONPATH') else '')
    e.update(env or {})
    return subprocess.run([PY, '-m', 'exopipe.phase3'] + [str(a) for a in args], cwd=cwd or REPO, env=e,
                          capture_output=True, text=True, timeout=timeout)


class cfg_restored:
    """Undo every override and authorisation made inside the block (CFG is process-global)."""

    def __enter__(self):
        from exopipe.phase3.config import CFG
        self.cfg = CFG
        self.auth = dict(CFG._auth)
        return CFG

    def __exit__(self, *exc):
        self.cfg.reset()
        self.cfg._auth = self.auth


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError(f'{fn.__name__}{a} did not raise {exc.__name__}')


# =====================================================================================================================
# item 1 - config: protected keys, ruled keys, one-entry grammar
# =====================================================================================================================

KNOWN_ID = 'R-ASSOCIATION-RADIUS-QUADRATURE'          # a fix_id row of clv15/fix_ledger.csv (read only)


def test_item1_protected_keys_refuse():
    from exopipe.phase3.config import ConfigRefusal
    with cfg_restored() as CFG:
        CFG.authorize([], out='/tmp/x')
        for k, v in [('paths.DATA_DECISIONS', '/tmp/dd.csv'), ('paths.PRODUCTION_WRITES', 'True'),
                     ('paths.PRODUCTION_ROOTS', '()'), ('paths.FIX_LEDGER', '/tmp/l.csv'),
                     ('paths.PINNED_INPUTS', '()'), ('exclusions.SUBBAND_ONLY', '()'),
                     ('exclusions.PERBAND_TAG', 'x')]:
            e = raises(ConfigRefusal, CFG.set, k, v)
            assert 'PROTECTED' in str(e) and k in str(e), e
        assert CFG.paths.PRODUCTION_WRITES is False and CFG.overrides == []
        # a key listed in paths.PINNED_INPUTS is protected too
        CFG.paths.PINNED_INPUTS = ('runner.POLL_S',)
        try:
            raises(ConfigRefusal, CFG.set, 'runner.POLL_S', '5')
        finally:
            CFG.paths.PINNED_INPUTS = ()
        CFG.set('runner.POLL_S', '5')
        assert CFG.runner.POLL_S == 5
        # RULED itself is not a tunable (it cannot be overridden away)
        raises(KeyError, CFG.set, 'fitting.RULED', '{}')
    # through the CLI: a protected override is refused with exit 2 before anything runs
    r = cli('verify', '--list', '--cores', 'none', '--set', 'paths.PRODUCTION_WRITES=True')
    assert r.returncode == 2 and 'PROTECTED' in r.stderr, (r.returncode, r.stderr)
    r = cli('config', '--set', 'exclusions.SUBBAND_ONLY=()')
    assert r.returncode == 2 and 'PROTECTED' in r.stderr, (r.returncode, r.stderr)


def test_item1_ruled_keys_need_ruling_or_nonproduction_out():
    from exopipe.phase3.config import ConfigRefusal
    from exopipe.phase3.core import record
    with cfg_restored() as CFG:
        prod = os.path.join(CFG.paths.PRODUCTION_ROOTS[0], 'some_run')
        # production out, no ruling: refused, nothing changed
        CFG.authorize([], out=prod)
        e = raises(ConfigRefusal, CFG.set, 'astrometry.K_ASSOC', '3.5')
        assert KNOWN_ID in str(e) and '--ruling-for' in str(e)
        assert CFG.astrometry.K_ASSOC == 3.034
        # an unruled key is unaffected
        CFG.set('runner.POLL_S', '7')
        # an unknown ledger ID is refused at authorisation
        e = raises(ConfigRefusal, CFG.authorize, ['astrometry.K_ASSOC=NO-SUCH-LEDGER-ID-XYZ'], out=prod)
        assert 'NO-SUCH-LEDGER-ID-XYZ' in str(e)
        # a real ledger ID authorises it into production; the authority is recorded
        CFG.authorize([f'astrometry.K_ASSOC={KNOWN_ID}'], out=prod)
        CFG.set('astrometry.K_ASSOC', '3.5')
        assert CFG.astrometry.K_ASSOC == 3.5
        assert CFG.overrides[-1][4] == f'ruling:{KNOWN_ID}'
        # a non-production out needs no ruling (recorded as such) ...
        with Scratch('item1') as d:
            CFG.authorize([], out=d)
            CFG.set('fitting.GAUSS_SNR', '6')
            assert CFG.overrides[-1][4].startswith('non-production out:')
            # ... and the run record carries every override with its authority
            run = record.Run('teststep', ['exopipe3 teststep'], d, CFG)
            h = run.header()
            keys = {o[0]: o[4] for o in h['config_overrides']}
            assert keys['astrometry.K_ASSOC'] == f'ruling:{KNOWN_ID}' and keys['runner.POLL_S'] is None
            ro = {o['key']: o for o in h['ruled_overrides']}
            assert set(ro) == {'astrometry.K_ASSOC', 'fitting.GAUSS_SNR'}
            assert ro['fitting.GAUSS_SNR']['ruled_by'] == ['RULING-TEMPLATE-FIT-BELOW-7-SIGMA']
            assert [o['key'] for o in record.unruled_overrides(h)] == ['fitting.GAUSS_SNR']
            # a final --out under production re-checks the overrides allowed only for a non-production out
            raises(ConfigRefusal, CFG.check_out, prod)
            CFG.check_out(d)
    # CLI: a ruled key into a production --out refuses (exit 2); with --ruling-for it passes the config stage
    prod = '/lustre/pipeline/exopipe/phase3/x'
    r = cli('verify', '--list', '--cores', 'none', '--out', prod, '--set', 'fitting.GAUSS_SNR=6')
    assert r.returncode == 2 and 'carries a ruling' in r.stderr, (r.returncode, r.stderr)
    r = cli('verify', '--list', '--cores', 'none', '--out', prod, '--set', 'fitting.GAUSS_SNR=6',
            '--ruling-for', 'fitting.GAUSS_SNR=RULING-TEMPLATE-FIT-BELOW-7-SIGMA')
    assert r.returncode == 0, (r.returncode, r.stderr)


def test_item1_one_entry_grammar_and_whole_dict_refusal():
    from exopipe.phase3.config import ConfigRefusal
    with cfg_restored() as CFG:
        CFG.authorize([], out='/tmp/x')          # non-production: ruled keys allowed, recorded
        before = dict(CFG.fitting.K_TEMPLATE)
        CFG.set('fitting.K_TEMPLATE[sublst,V]', '(0.6, 0.5)')
        k = CFG.fitting.K_TEMPLATE
        assert k[('sublst', 'V')] == (0.6, 0.5) and len(k) == 4
        assert all(k[x] == before[x] for x in before if x != ('sublst', 'V'))
        assert CFG.overrides[-1][:3] == ('fitting.K_TEMPLATE[sublst,V]', (0.658, 0.499), (0.6, 0.5))
        assert CFG.get('fitting.K_TEMPLATE[sublst,V]') == (0.6, 0.5)
        # ints become floats inside the tuple; quotes around entry names are accepted
        CFG.set("fitting.K_TEMPLATE['deep','I']", '(1, 1)')
        assert CFG.fitting.K_TEMPLATE[('deep', 'I')] == (1.0, 1.0)
        assert isinstance(CFG.fitting.K_TEMPLATE[('deep', 'I')][0], float)
        # nested dict with int-tuple keys
        CFG.set('astrometry.SYS_ARCMIN[75,90][Red]', '2')
        assert CFG.astrometry.SYS_ARCMIN[(75, 90)]['Red'] == 2.0
        assert CFG.astrometry.SYS_ARCMIN[(75, 90)]['Green'] == 1.3
        # an entry that does not exist is refused (no silent new key)
        e = raises(KeyError, CFG.set, 'fitting.K_TEMPLATE[sublts,V]', '(1,1)')
        assert 'no entry [sublts,V]' in str(e) and '[sublst,V]' in str(e)
        # a whole-dict override that omits entries is refused; one that gives every entry is accepted
        e = raises(ConfigRefusal, CFG.set, 'fitting.K_TEMPLATE', "{('deep','I'): (0.5, 0.5)}")
        assert 'omits 3 of its 4 entries' in str(e)
        e = raises(ConfigRefusal, CFG.set, 'astrometry.SYS_ARCMIN[0,45]', "{'Red': 3.0}")
        assert 'omits 2' in str(e)
        full = {k: (0.5, 0.5) for k in CFG.fitting.K_TEMPLATE}
        CFG.set('fitting.K_TEMPLATE', repr(full))
        assert CFG.fitting.K_TEMPLATE == full
        # [entry] on a non-dict
        raises(KeyError, CFG.set, 'fitting.GAUSS_SNR[x]', '1')
        # override file with the same grammar
        with Scratch('item1f') as d:
            f = os.path.join(d, 'ov.py')
            open(f, 'w').write('fitting.K_TEMPLATE[sublst,I] = (0.4, 0.4)   # campaign\nrunner.POLL_S = 3\n')
            CFG.apply_override_file(f)
            assert CFG.fitting.K_TEMPLATE[('sublst', 'I')] == (0.4, 0.4) and CFG.runner.POLL_S == 3
    from exopipe.phase3.config import CFG
    assert CFG.fitting.K_TEMPLATE[('sublst', 'V')] == (0.658, 0.499) and CFG.overrides == []   # reset() restored


def test_item1_ledger_lookup_is_tolerant():
    from exopipe.phase3.core import ledger
    with Scratch('ledger') as d:
        p = os.path.join(d, 'fix_ledger.csv')
        rows = ['﻿date,fix_id,files,description,authorized_by,status',
                '2026-08-11,U-1c,"a.py, b.py",desc with, commas,Gregg,applied',     # more fields than the header
                '2026-09-08,R-SHORT-ROW,file,desc',                                # fewer fields
                'R-SHIFTED-ROW-XYZ,2026-09-20,implemented',                        # ID in the date column
                '2026-10-13T00:10Z,R-GREGG-TIMESTAMPED,x,"multi\r\nline",Gregg,ok']
        with open(p, 'w', newline='') as f:
            f.write('\r\n'.join(rows) + '\r\n')
        for good in ('U-1c', 'R-SHORT-ROW', 'R-SHIFTED-ROW-XYZ', 'R-GREGG-TIMESTAMPED', ' U-1c '):
            assert ledger.has_id(good, p), good
        for bad in ('', '2026-08-11', 'desc with', 'NOPE', 'implemented x'):
            assert not ledger.has_id(bad, p), bad
    # the real ledger (read only): the IDs config.py declares in RULED are rows of it
    from exopipe.phase3.config import CFG
    for sec in CFG.sections().values():
        for ids in (getattr(sec, 'RULED', None) or {}).values():
            for lid in (ids if isinstance(ids, (tuple, list)) else (ids,)):
                assert ledger.has_id(lid, CFG.paths.FIX_LEDGER), lid


# =====================================================================================================================

def _all_tests():
    return [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_') and callable(f)]


def main(argv):
    names = argv or [n for n, _ in _all_tests()]
    fails = 0
    for n, f in _all_tests():
        if n not in names:
            continue
        t0 = time.time()
        try:
            f()
            print(f'PASS {n} ({time.time() - t0:.1f} s)')
        except Exception as e:                     # noqa: BLE001 - report every failure
            fails += 1
            import traceback
            traceback.print_exc()
            print(f'FAIL {n}: {e}')
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
