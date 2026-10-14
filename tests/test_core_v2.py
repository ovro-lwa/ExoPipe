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
  item 7  B-30 fallback sites (core/fallback.py): refuse or count, sticky refusal through broad excepts, children's
          policies and counts through core.legacy.run, the key registry (every key ruled, in the parity file, used)

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
        assert CFG.overrides[-1][:3] == ('fitting.K_TEMPLATE[sublst,V]', before[('sublst', 'V')], (0.6, 0.5))
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
    assert CFG.fitting.K_TEMPLATE[('sublst', 'V')] == before[('sublst', 'V')] and CFG.overrides == []   # reset() restored


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
# item 2 - per-step CONFIG declaration
# =====================================================================================================================

def _fake_step(name, **attrs):
    """Register a synthetic step module exopipe.phase3.steps.<name> (removed by the caller)."""
    import types
    mod = types.ModuleType(f'exopipe.phase3.steps.{name}')
    mod.__doc__ = f'synthetic step {name}'
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[mod.__name__] = mod
    return mod


def _stdout_of(fn, *a):
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*a)
    return rc, buf.getvalue()


def test_item2_step_config_declaration():
    from exopipe.phase3 import cli, config
    from exopipe.phase3.core import record
    names = ['_t2_nocfg', '_t2_empty', '_t2_runner', '_t2_bad']
    try:
        m_no = _fake_step('_t2_nocfg')
        m_empty = _fake_step('_t2_empty', CONFIG=())
        m_runner = _fake_step('_t2_runner', CONFIG=('runner',))
        m_bad = _fake_step('_t2_bad', CONFIG=('no_such_section',))
        shared = config.SHARED_SECTIONS
        assert config.step_sections(m_no) is None                      # backward compatible: every section
        assert config.step_sections(m_empty) == shared
        assert config.step_sections(m_runner) == ('runner',) + shared
        assert 'no_such_section' in str(raises(KeyError, config.step_sections, m_bad))
        # the real steps import and declare
        import exopipe.phase3.steps.run as srun
        import exopipe.phase3.steps.verify as sver
        assert config.declared_sections(srun) == ('runner',) and config.declared_sections(sver) == ()
        # `exopipe3 config --step`: 'not declared' for a step without CONFIG; the sections for one with it
        rc, out = _stdout_of(cli._config_cmd, ['--step', '_t2_nocfg'])
        assert rc == 0 and 'not declared' in out, out
        rc, out = _stdout_of(cli._config_cmd, ['--step', '_t2_runner'])
        assert rc == 0 and 'class runner(_Section)' in out and 'FREE_CORES_CMD' in out, out
        assert 'class fitting' not in out and 'shared sections' in out
        rc, _ = _stdout_of(cli._config_cmd, ['--step', '_t2_bad'])
        assert rc == 1
        # the run record's config_hash covers only the declared + shared sections
        with cfg_restored() as CFG, Scratch('item2') as d:
            CFG.authorize([], out=d)
            runs = {n: record.Run('t2', ['x'], d, CFG, config_sections=config.step_sections(m))
                    for n, m in (('no', m_no), ('empty', m_empty), ('runner', m_runner))}
            for r in runs.values():
                r.commit = 'abc1234'                       # a clean-tree commit (item 3 tests the dirty case)
            h0 = {n: r.header() for n, r in runs.items()}
            assert h0['empty']['config_hash'] == CFG.hash(shared) and h0['empty']['config_sections'] == sorted(shared)
            assert h0['no']['config_hash'] == CFG.hash() and h0['no']['config_sections'] is None
            CFG.set('runner.POLL_S', '11')                 # a section the 'empty' step does not read
            h1 = {n: r.header() for n, r in runs.items()}
            assert h1['empty']['config_hash'] == h0['empty']['config_hash']
            assert h1['runner']['config_hash'] != h0['runner']['config_hash']
            assert h1['no']['config_hash'] != h0['no']['config_hash']
            # resume: a done unit of the 'empty' step survives the runner override; a shared-section change does not
            src = os.path.join(d, 'in.csv')
            dst = os.path.join(d, 'out.csv')
            open(src, 'w').write('a,b\n1,2\n')
            open(dst, 'w').write('a,b\n1,2\n')
            r1 = runs['empty']
            with r1.unit('u1') as u:
                u.add_input(src)
                u.add_output(dst, 'csv_table')
            assert u.status == 'done', (u.status, u.reason)
            ok, why = r1.resume_check('u1', inputs=[src], admitted=None)
            CFG.set('runner.POLL_S', '12')
            r2 = record.Run('t2', ['x'], d, CFG, config_sections=config.step_sections(m_empty))
            r2.commit = 'abc1234'
            ok2, why2 = r2.resume_check('u1', inputs=[src], admitted=None)
            CFG.set('astrometry.SYS_SCALE', "{('V', 'Red'): 1.6}")
            ok3, why3 = r2.resume_check('u1', inputs=[src], admitted=None)
            assert ok and ok2, (why, why2)
            assert ok3 is False and why3 == 'config changed', (ok3, why3)
    finally:
        for n in names:
            sys.modules.pop(f'exopipe.phase3.steps.{n}', None)


# =====================================================================================================================
# item 3 - record: FITS identity, dirty trees, planned outputs, exit codes
# =====================================================================================================================

def _write_fits(path, data, checksum=True, **cards):
    import numpy as np
    from astropy.io import fits
    h = fits.PrimaryHDU(np.asarray(data, dtype=np.float32))
    for k, v in cards.items():
        h.header[k] = v
    h.header['HISTORY'] = 'synthetic test product'
    h.writeto(path, overwrite=True, checksum=checksum)


def test_item3_fits_identity():
    import numpy as np
    from astropy.io import fits
    from exopipe.phase3.core import record
    with Scratch('item3fits') as d:
        p = os.path.join(d, 'm.fits')
        data = np.arange(64, dtype=np.float32).reshape(8, 8)
        _write_fits(p, data, COADDOP='inverse-variance', FLUXBAKE='scale_v4', FLUXVER=4, MDWARP=True,
                    CONSTDIG='abc123', NCOADD=17, OBJECT="it's")
        hdr = fits.getheader(p)
        cards = record.fits_primary_cards(p)
        for k in ('DATASUM', 'COADDOP', 'FLUXBAKE', 'FLUXVER', 'MDWARP', 'CONSTDIG', 'NCOADD', 'OBJECT', 'NAXIS1'):
            assert cards[k] == hdr[k], (k, cards[k], hdr[k])
        fp = record.fingerprint(p, hash_limit_bytes=0)        # as for a large FITS: no sha1 of the bytes
        assert 'sha1' not in fp and fp['datasum'] == hdr['DATASUM']
        assert fp['cards'] == {'COADDOP': 'inverse-variance', 'FLUXBAKE': 'scale_v4', 'FLUXVER': 4, 'MDWARP': True,
                               'CONSTDIG': 'abc123', 'NCOADD': 17}
        assert 'datasum' in record.fingerprint(p) and 'sha1' in record.fingerprint(p)   # small files keep sha1 too
        # mtime is not identity when a digest exists
        st = os.stat(p)
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 10))
        assert record.same_fingerprint(fp, record.fingerprint(p, hash_limit_bytes=0))
        # same size, same mtime, different data -> different DATASUM -> different
        data2 = data.copy()
        data2[3, 3] += 1
        _write_fits(p, data2, COADDOP='inverse-variance', FLUXBAKE='scale_v4', FLUXVER=4, MDWARP=True,
                    CONSTDIG='abc123', NCOADD=17, OBJECT="it's")
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
        fp2 = record.fingerprint(p, hash_limit_bytes=0)
        assert fp2['size'] == fp['size'] and fp2['mtime_ns'] == fp['mtime_ns']
        assert not record.same_fingerprint(fp, fp2)
        # same data, new processing card (a re-bake) -> different
        _write_fits(p, data2, COADDOP='inverse-variance', FLUXBAKE='scale_v4', FLUXVER=5, MDWARP=True,
                    CONSTDIG='abc123', NCOADD=17, OBJECT="it's")
        fp3 = record.fingerprint(p, hash_limit_bytes=0)
        assert fp3['datasum'] == fp2['datasum'] and not record.same_fingerprint(fp2, fp3)
        # one side with a digest, the other without -> different; neither -> size + mtime
        q = os.path.join(d, 'nosum.fits')
        _write_fits(q, data, checksum=False)
        fq = record.fingerprint(q, hash_limit_bytes=0)
        assert 'datasum' not in fq and fq['cards'] == {}
        no_digest = {k: v for k, v in fp.items() if k != 'datasum'}
        assert not record.same_fingerprint(fp, no_digest) and not record.same_fingerprint(no_digest, fp)
        assert record.same_fingerprint(fq, record.fingerprint(q, hash_limit_bytes=0))
        # not a FITS file: no FITS identity, no error
        t = os.path.join(d, 'x.fits')
        open(t, 'w').write('not fits')
        assert 'datasum' not in record.fingerprint(t) and record.fits_primary_cards(t) == {}
        # resume through the digest: a large-FITS input recorded without sha1 survives a touch, not a data change
        with cfg_restored() as CFG:
            CFG.authorize([], out=d)
            run = record.Run('t3', ['x'], d, CFG, config_sections=())
            run.commit = 'abc1234'
            out = os.path.join(d, 'out.csv')
            open(out, 'w').write('a\n1\n')
            with run.unit('u') as u:
                u.inputs[os.path.abspath(p)] = record.fingerprint(p, hash_limit_bytes=0)
                u.add_output(out, 'csv_table')
            st = os.stat(p)
            os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5 * 10 ** 9))
            ok, why = run.resume_check('u', inputs=[p], admitted=None)
            assert ok, why
            _write_fits(p, data, COADDOP='inverse-variance', FLUXBAKE='scale_v4', FLUXVER=5, MDWARP=True,
                        CONSTDIG='abc123', NCOADD=17, OBJECT="it's")
            ok, why = run.resume_check('u', inputs=[p], admitted=None)
            assert not ok and why.startswith('input changed'), why


def test_item3_dirty_tree_never_resumed():
    from exopipe.phase3.core import record
    with cfg_restored() as CFG, Scratch('item3dirty') as d:
        CFG.authorize([], out=d)
        out = os.path.join(d, 'o.csv')
        open(out, 'w').write('a\n1\n')
        for commit, want in (('abc1234', True), ('abc1234-dirty', False)):
            run = record.Run('t3d', ['x'], d, CFG, config_sections=())
            run.commit = commit
            with run.unit('u') as u:
                u.add_output(out, 'csv_table')
            assert u.status == 'done'
            ok, why = run.resume_check('u', inputs=[], admitted=None)
            assert ok is want, (commit, why)
            if not want:
                assert 'dirty tree' in why and 'abc1234-dirty' in why


def test_item3_planned_outputs_item_refusals_exit_codes():
    from exopipe.phase3.core import record
    with cfg_restored() as CFG, Scratch('item3plan') as d:
        CFG.authorize([], out=d)

        def csvf(name, text='a\n1\n'):
            p = os.path.join(d, name)
            open(p, 'w').write(text)
            return p

        def unit(run, name, plan=(), write=(), refuse=(), raise_=None, **ctx):
            with run.unit(name) as u:
                for p in plan:
                    u.plan(p, 'csv_table')
                for p in write:
                    u.add_output(p, 'csv_table', **ctx)
                for p in refuse:
                    u.refuse_item(p, 'night rule: sun above -12 deg', gate='night_rule')
                if raise_:
                    raise raise_
            return u

        a, b, c = csvf('a.csv'), csvf('b.csv'), csvf('c.csv')
        run = record.Run('t3p', ['x'], d, CFG, config_sections=())
        # every planned output written or refused -> done (exit 0), the refusal recorded
        u = unit(run, 'complete', plan=(a, b, c), write=(a, b), refuse=(c,))
        assert u.status == 'done', (u.status, u.reason)
        rec = record.load_record(d, 't3p', 'complete')
        assert rec['refused_items'][0]['gate'] == 'night_rule' and len(rec['planned']) == 3
        assert rec['record_version'] == 2
        # a planned output neither written nor refused -> error (the unit dropped an item silently)
        u = unit(run, 'short', plan=(a, b, c), write=(a, b))
        assert u.status == 'error' and '1 of 3 planned outputs' in u.reason and c in u.reason, u.reason
        # every planned item refused -> the unit is refused
        u = unit(run, 'allref', plan=(a, b), refuse=(a, b))
        assert u.status == 'refused' and 'every planned item refused' in u.reason
        # a written output that fails its test refuses the unit ...
        u = unit(run, 'badtest', plan=(a, b), write=(a, b), required_columns=('zz',))
        assert u.status == 'refused' and 'content test failed' in u.reason
        # ... unless that item is itself recorded as refused
        with run.unit('badref') as u:
            u.plan(a, 'csv_table')
            u.plan(b, 'csv_table')
            u.add_output(a, 'csv_table')
            u.add_output(b, 'csv_table', required_columns=('zz',))
            u.refuse_item(b, 'size gate: 2.9 beams > 2.5')
        assert u.status == 'done', (u.status, u.reason)
        # refuse_item needs a reason
        raises(ValueError, record.UnitRecord(run, 'x', {}).refuse_item, a, '')
        # without a plan: refusals only -> refused; nothing -> no-input
        assert unit(run, 'onlyref', refuse=(a,)).status == 'refused'
        assert unit(run, 'nothing').status == 'no-input'
        assert unit(run, 'whole', raise_=record.Refusal('beam gate')).status == 'refused'
        assert unit(run, 'noin', raise_=record.NoInput('no frames')).status == 'no-input'

        def code(*statuses):
            r = record.Run('t3e', ['x'], d, CFG, dry_run=True)
            for i, s in enumerate(statuses):
                uu = record.UnitRecord(r, f'u{i}', {})
                uu.status = s
                r.units.append(uu)
            return r.exit_code()

        assert code() == 3
        assert code('no-input', 'no-input') == 3
        assert code('done', 'no-input') == 0                  # was 3 before DESIGN 11.5
        assert code('done', 'skipped-valid') == 0
        assert code('done', 'refused', 'no-input') == 2
        assert code('refused', 'error') == 1
        assert run.exit_code() == 1                           # the 'short' unit errored
        summ = json.load(open(run.write_summary()))
        assert {x['unit']: x['n_refused_items'] for x in summ['units']}['complete'] == 1


# =====================================================================================================================
# item 4 - runner: --code worktrees, forwarded overrides, chains
# =====================================================================================================================

class runner_cfg:
    """Set CFG.runner attributes for the block (restored after)."""

    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        from exopipe.phase3.config import CFG
        self.old = {k: getattr(CFG.runner, k) for k in self.kw}
        for k, v in self.kw.items():
            setattr(CFG.runner, k, v)
        return CFG

    def __exit__(self, *exc):
        from exopipe.phase3.config import CFG
        for k, v in self.old.items():
            setattr(CFG.runner, k, v)


def _git(repo, *args):
    r = subprocess.run(['git', '-C', repo, '-c', 'user.name=test', '-c', 'user.email=test@example.invalid']
                       + list(args), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_item4_code_worktree_runs_pinned_steps():
    import socket
    from exopipe.phase3.core import runner
    with Scratch('item4code') as d:
        # a synthetic repository whose exopipe.phase3 entry point reports the file it runs from
        repo = os.path.join(d, 'repo')
        os.makedirs(os.path.join(repo, 'exopipe', 'phase3'))
        open(os.path.join(repo, 'exopipe', '__init__.py'), 'w').write('')
        open(os.path.join(repo, 'exopipe', 'phase3', '__init__.py'), 'w').write('')
        open(os.path.join(repo, 'exopipe', 'phase3', '__main__.py'), 'w').write(
            'import os, sys\nprint("PINNED-CODE", os.path.abspath(__file__), " ".join(sys.argv[1:]))\n'
            'sys.exit(0)\n')
        subprocess.run(['git', 'init', '-q', repo], check=True, capture_output=True)
        _git(repo, 'add', '-A')
        _git(repo, 'commit', '-q', '-m', 'pinned')
        sha = _git(repo, 'rev-parse', 'HEAD')
        code_root = os.path.join(d, '_code')
        jobs = os.path.join(d, 'jobs')
        with runner_cfg(JOBS_ROOT=jobs, CODE_REPO=repo, CODE_WORKTREES_ROOT=code_root, POLL_S=1):
            # once per commit, reused
            s1, path, created = runner.ensure_code_worktree(sha[:9])
            assert s1 == sha and created and path == os.path.join(code_root, sha)
            s2, path2, created2 = runner.ensure_code_worktree(sha)
            assert path2 == path and not created2
            main = os.path.join(path, 'exopipe', 'phase3', '__main__.py')
            assert os.path.isfile(main) and not os.access(main, os.W_OK) and not os.access(path, os.W_OK)
            assert _git(path, 'rev-parse', 'HEAD') == sha
            assert os.path.isdir(os.path.join(path, 'exopipe', 'phase3', '__pycache__'))   # compiled before chmod
            raises(SystemExit, runner.ensure_code_worktree, 'no-such-commit-xyz')
            # a job pinned to the commit: the step process runs from the worktree; the commit is recorded
            job = runner.submit('t4code', ['search', '--cells', '{item}'], ['04h_Red'],
                                [(socket.gethostname(), 1)], start=False, code=sha[:9],
                                forward=['--set', 'runner.POLL_S=7'])
            j = runner.load_job(job)
            assert j['code_commit'] == sha and j['code_root'] == path and j['code_repo'] == repo
            runner.worker(job)
            st = runner.item_state(job, '04h_Red')
            assert st['status'] == 'done' and st['code_commit'] == sha, st
            log = open(st['log']).read()
            assert f'PINNED-CODE {main} search --cells 04h_Red --set runner.POLL_S=7' in log, log
            # list_jobs skips the code-worktree root and anything without a job.json
            os.makedirs(os.path.join(jobs, '_code'))
            assert runner.list_jobs() == [job]


def test_item4_chain_steps_per_item():
    import socket
    from exopipe.phase3.core import runner
    with Scratch('item4chain') as d:
        good = os.path.join(d, 'good.csv')
        late = os.path.join(d, 'late.csv')           # missing at first: step 2 refuses, step 3 never runs
        open(good, 'w').write('a,b\n1,2\n')
        # a template without {item} would get '--cells <item>' appended (verify has no --cells): every step names it
        chains = {'t4': ['verify --list --glob {item} --cores none', 'verify csv_table {item} --cores none',
                         ['verify', '--list', '--glob', '{item}', '--cores', 'none']]}
        with runner_cfg(JOBS_ROOT=os.path.join(d, 'jobs'), CHAINS=chains, POLL_S=1):
            raises(SystemExit, runner.submit, None, [], [good], [('localhost', 1)], start=False, chain='nope')
            raises(SystemExit, runner.submit, None, ['verify'], [good], [('localhost', 1)], start=False, chain='t4')
            job = runner.submit(None, [], [good, late], [(socket.gethostname(), 1)], start=False, chain='t4',
                                forward=['--set', 'runner.POLL_S=7'])
            j = runner.load_job(job)
            assert job.endswith('_t4') and j['chain'] == 't4' and len(j['chain_steps']) == 3
            runner.worker(job)
            g, l_ = runner.item_state(job, good), runner.item_state(job, late)
            assert g['status'] == 'done' and [s['status'] for s in g['steps']] == ['done'] * 3, g
            assert all(s['argv'][-2:] == ['--set', 'runner.POLL_S=7'] for s in g['steps'])
            assert l_['status'] == 'refused' and [s['status'] for s in l_['steps']] == ['done', 'refused'], l_
            assert l_['steps'][1]['rc'] == 2 and l_['step'] == 1
            assert runner.job_state(job) == ('finished', {'done': 1, 'refused': 1})
            # `run status` shows the step each item reached (the existing columns unchanged)
            r = cli('run', 'status', job, '--cores', 'none', '--set', f'runner.JOBS_ROOT={os.path.join(d, "jobs")}')
            assert r.returncode == 0, r.stderr
            assert 'steps 3/3' in r.stdout and 'steps 1/3 at step 2 verify (refused)' in r.stdout, r.stdout
            # a retried chain item resumes at the step that stopped it
            open(late, 'w').write('a\n1\n')
            assert runner.retry(job, ('refused',), start=False) == 1
            first_start = l_['steps'][0]['started']
            runner.worker(job)
            l2 = runner.item_state(job, late)
            assert l2['status'] == 'done' and [s['status'] for s in l2['steps']] == ['done'] * 3, l2
            assert l2['steps'][0]['started'] == first_start            # step 1 was not run again
            assert open(l2['log']).read().count('step 1/3') == 1


def test_item4_submit_forwards_set_and_config():
    import socket
    from exopipe.phase3.core import runner
    with Scratch('item4fwd') as d:
        jobs = os.path.join(d, 'jobs')
        ov = os.path.join(d, 'campaign.py')
        open(ov, 'w').write('runner.FREE_CORES_CMD = "echo 0-3"\n')
        # a protected key is refused at submit; no job is created
        r = cli('run', 'submit', '--no-start', '--nodes', 'localhost:1', '--items', 'runner', '--cores', 'none',
                '--set', f'runner.JOBS_ROOT={jobs}', '--set', 'paths.DATA_DECISIONS=/tmp/x', '--',
                'config', '--section', '{item}')
        assert r.returncode == 2 and 'PROTECTED' in r.stderr and not os.path.isdir(jobs), (r.returncode, r.stderr)
        # --set and --config before '--' reach the step command and the job record
        r = cli('run', 'submit', '--no-start', '--name', 'fw', '--nodes', f'{socket.gethostname()}:1',
                '--items', 'runner', '--cores', 'none', '--set', f'runner.JOBS_ROOT={jobs}',
                '--set', 'runner.POLL_S=7', '--config', ov, '--', 'config', '--section', '{item}')
        assert r.returncode == 0, r.stderr
        job = r.stdout.split()[1].rstrip(':')
        with runner_cfg(JOBS_ROOT=jobs):
            j = runner.load_job(job)
            assert j['forward_argv'] == ['--set', f'runner.JOBS_ROOT={jobs}', '--set', 'runner.POLL_S=7',
                                         '--config', ov], j['forward_argv']
            assert j['overrides']['set'] == [f'runner.JOBS_ROOT={jobs}', 'runner.POLL_S=7']
            assert j['overrides']['config'] == ov and len(j['overrides']['config_sha1']) == 40
            assert not os.path.exists(os.path.join(runner.job_dir(job), 'supervisor.log'))   # --no-start
            runner.worker(job)
            st = runner.item_state(job, 'runner')
            assert st['status'] == 'done', st
            log = open(st['log']).read()
            assert 'override runner.POLL_S: 30 -> 7' in log and "override runner.FREE_CORES_CMD" in log, log


# =====================================================================================================================
# item 5 - parity write guard
# =====================================================================================================================

def _under_default_allowed(path):
    """True when ``path`` lies under a root the guard always allows (/tmp, $TMPDIR): denial cannot be shown there."""
    r = os.path.realpath(path) + '/'
    roots = ['/tmp'] + ([os.environ['TMPDIR']] if os.environ.get('TMPDIR') else [])
    return any(r.startswith(os.path.realpath(x).rstrip('/') + '/') for x in roots)


def test_item5_guard_in_process():
    import io
    import pathlib
    import tempfile as tf
    import numpy as np
    from astropy.io import fits
    from exopipe.phase3.core import guard
    with Scratch('item5') as d:
        if _under_default_allowed(d):
            print(f'SKIP denial checks: scratch {d} is under /tmp or $TMPDIR, which the guard always allows')
            return
        out, other = os.path.join(d, 'out'), os.path.join(d, 'other')
        os.makedirs(other)
        victim = os.path.join(other, 'victim.txt')
        open(victim, 'w').write('production bytes')
        # default roots kept: astropy probes mmap support with a temporary file in /tmp on every write
        guard.install([out], log=os.path.join(d, 'refusals.log'))
        try:
            assert guard.active()
            # allowed: under --out (created, with the parents it needs), /dev/null, an existing directory, reads
            os.makedirs(os.path.join(out, 'a', 'b'))
            f = os.path.join(out, 'a', 'f.txt')
            with open(f, 'w') as fh:
                fh.write('ok')
            open('/dev/null', 'w').close()
            os.makedirs(other, exist_ok=True)
            assert open(victim).read() == 'production bytes'
            fits.writeto(os.path.join(out, 'ok.fits'), np.zeros((2, 2), np.float32))
            os.symlink(other, os.path.join(out, 'escape'))          # a link inside --out may be made ...
            # ... and links to outside files removed, renamed, replaced and swept by rmtree: an operation on a directory
            # ENTRY touches the link, never its target (2026-10-14: the catalogue replay's p3s3_merge.py re-points
            # refusals/<file> links to the attempts' files and was refused at the unlink)
            ln = os.path.join(out, 'ln_victim')
            os.symlink(victim, ln)
            os.unlink(ln)
            os.symlink(victim, ln)
            os.rename(ln, ln + '2')
            os.symlink(victim, ln)
            os.replace(ln, ln + '2')
            pathlib.Path(ln + '2').unlink()
            os.makedirs(os.path.join(out, 'linked'))
            os.symlink(victim, os.path.join(out, 'linked', 'l'))
            shutil.rmtree(os.path.join(out, 'linked'))
            os.symlink(victim, ln)
            shutil.move(ln, os.path.join(out, 'moved_ln'))
            assert open(victim).read() == 'production bytes' and os.path.islink(os.path.join(out, 'moved_ln'))
            # a hard link of an outside file is refused; the callers' copy fallback then makes an independent file
            hl = os.path.join(out, 'hl.txt')
            try:
                os.link(victim, hl)
            except OSError:
                shutil.copy2(victim, hl)
            assert not os.path.samefile(victim, hl)
            open(hl, 'w').write('parity bytes')
            assert open(victim).read() == 'production bytes'
            os.link(f, os.path.join(out, 'hl_inside.txt'))          # a hard link inside the roots is fine
            p = pathlib.Path(other)
            denied = {
                'open w': lambda: open(os.path.join(other, 'x'), 'w'),
                'open a': lambda: open(victim, 'a'),
                'open r+': lambda: open(victim, 'r+'),
                'io.open wb': lambda: io.open(os.path.join(other, 'x'), 'wb'),
                'os.open': lambda: os.open(os.path.join(other, 'x'), os.O_WRONLY | os.O_CREAT),
                'os.mkdir': lambda: os.mkdir(os.path.join(other, 'sub')),
                'os.makedirs': lambda: os.makedirs(os.path.join(other, 's1', 's2')),
                'os.remove': lambda: os.remove(victim),
                'os.unlink': lambda: os.unlink(victim),
                'os.rename out': lambda: os.rename(victim, os.path.join(out, 'v')),
                'os.replace in': lambda: os.replace(f, os.path.join(other, 'y')),
                'shutil.copy': lambda: shutil.copy(f, other),
                'shutil.copyfile': lambda: shutil.copyfile(f, os.path.join(other, 'y')),
                'shutil.copy2': lambda: shutil.copy2(f, os.path.join(other, 'y')),
                'shutil.move': lambda: shutil.move(victim, os.path.join(out, 'm')),
                'shutil.copytree': lambda: shutil.copytree(os.path.join(out, 'a'), os.path.join(other, 'tree')),
                'shutil.rmtree': lambda: shutil.rmtree(other),
                'os.symlink': lambda: os.symlink(f, os.path.join(other, 'ln')),
                'pathlib write_text': lambda: (p / 'pw.txt').write_text('x'),
                'pathlib mkdir': lambda: (p / 'pd').mkdir(),
                'pathlib unlink': lambda: pathlib.Path(victim).unlink(),
                'numpy save': lambda: np.save(os.path.join(other, 'x.npy'), np.zeros(2)),
                'tempfile': lambda: tf.mkstemp(dir=other),
                'fits.writeto': lambda: fits.writeto(os.path.join(other, 'x.fits'), np.zeros((2, 2))),
                'HDUList.writeto': lambda: fits.HDUList([fits.PrimaryHDU()]).writeto(os.path.join(other, 'h.fits')),
                'PrimaryHDU.writeto': lambda: fits.PrimaryHDU().writeto(os.path.join(other, 'p.fits')),
                'symlink escape': lambda: open(os.path.join(out, 'escape', 'z'), 'w'),   # ... but not followed out
                'entry under a linked dir': lambda: os.unlink(os.path.join(out, 'escape', 'victim.txt')),
                'rename under a linked dir': lambda: os.rename(os.path.join(out, 'escape', 'victim.txt'),
                                                               os.path.join(out, 'stolen')),
                'symlink name under a linked dir': lambda: os.symlink(f, os.path.join(out, 'escape', 'ln2')),
                # '..' after a link is applied where the kernel applies it (out/escape/.. = the scratch dir, not out)
                'dotdot after a link': lambda: open(os.path.join(out, 'escape', '..', 'dd.txt'), 'w'),
                'dotdot entry after a link': lambda: os.symlink(f, os.path.join(out, 'escape', '..', 'dd_ln')),
                'os.link of an outside file': lambda: os.link(victim, os.path.join(out, 'hl2.txt')),
                'Path.link_to of an outside file': lambda: pathlib.Path(victim).link_to(os.path.join(out, 'hl3.txt')),
            }
            for name, fn in denied.items():
                e = raises(PermissionError, fn)
                assert isinstance(e, guard.ParityWriteRefused) and e.filename, (name, e)
            assert sorted(os.listdir(other)) == ['victim.txt'], os.listdir(other)
            assert open(victim).read() == 'production bytes'
            assert not os.path.exists(os.path.join(d, 'dd.txt')) and not os.path.lexists(os.path.join(d, 'dd_ln'))
            assert guard._STATE['refused'] >= len(denied)
        finally:
            guard.uninstall()
        assert not guard.active()
        open(os.path.join(other, 'after.txt'), 'w').write('free again')          # restored
        log = open(os.path.join(d, 'refusals.log')).read()
        assert log.count('PARITY GUARD') >= len(denied) and victim in log


def test_item5_guard_reaches_subprocesses():
    from exopipe.phase3.core import guard
    with Scratch('item5sub') as d:
        if _under_default_allowed(d):
            print(f'SKIP denial checks: scratch {d} is under /tmp or $TMPDIR, which the guard always allows')
            return
        out, other = os.path.join(d, 'out'), os.path.join(d, 'other')
        os.makedirs(out)
        os.makedirs(other)
        script = os.path.join(d, 'legacy.py')
        gcode = f'open({os.path.join(other, "g.txt")!r}, "w")'        # a grandchild Python process
        open(script, 'w').write(f'''
import json, os, subprocess, sys
import numpy as np
from astropy.io import fits            # imported AFTER the guard (the sitecustomize ran first): the import hook patches it
out, other = {out!r}, {other!r}
res = {{}}
def attempt(name, fn):
    try:
        fn()
        res[name] = 'ok'
    except PermissionError as e:
        res[name] = 'refused'
attempt('open_out', lambda: open(os.path.join(out, 'a.txt'), 'w').write('x'))
attempt('open_other', lambda: open(os.path.join(other, 'b.txt'), 'w').write('x'))
attempt('fits_out', lambda: fits.writeto(os.path.join(out, 'e.fits'), np.zeros((2, 2))))
attempt('fits_other', lambda: fits.writeto(os.path.join(other, 'c.fits'), np.zeros((2, 2))))
attempt('hdul_other', lambda: fits.HDUList([fits.PrimaryHDU()]).writeto(os.path.join(other, 'd.fits')))
attempt('fits_patched', lambda: None if 'guard' in fits.HDUList.writeto.__module__ else 1 / 0)
g = subprocess.run([sys.executable, '-c', {gcode!r}], capture_output=True, text=True)
res['grandchild'] = 'refused' if g.returncode != 0 and 'ParityWriteRefused' in g.stderr else 'ok'
print(json.dumps(res))
''')
        env = guard.export_env([out], env=dict(os.environ))
        assert env['PYTHONPATH'].split(os.pathsep)[0] == guard.SITECUSTOMIZE_DIR
        r = subprocess.run([PY, script], env=env, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr
        res = json.loads(r.stdout.strip().splitlines()[-1])
        assert res == {'open_out': 'ok', 'open_other': 'refused', 'fits_out': 'ok', 'fits_other': 'refused',
                       'hdul_other': 'refused', 'fits_patched': 'ok', 'grandchild': 'refused'}, (res, r.stderr)
        assert os.listdir(other) == [] and 'PARITY GUARD' in r.stderr
        # without the variable the sitecustomize does nothing
        env2 = dict(env)
        env2.pop(guard.ENV_ROOTS)
        r = subprocess.run([PY, '-c', f'open({os.path.join(other, "free.txt")!r}, "w")'], env=env2,
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0 and os.listdir(other) == ['free.txt'], r.stderr


def test_item5_cli_parity_flag():
    with Scratch('item5cli') as d:
        if _under_default_allowed(d):
            print(f'SKIP: scratch {d} is under /tmp or $TMPDIR, which the guard always allows')
            return
        out, other = os.path.join(d, 'out'), os.path.join(d, 'other')
        os.makedirs(other)
        good = os.path.join(d, 'good.csv')
        open(good, 'w').write('a\n1\n')
        r = cli('verify', 'csv_table', good, '--cores', 'none', '--parity')
        assert r.returncode == 2 and '--parity needs --out' in r.stderr, r.stderr
        r = cli('verify', 'csv_table', good, '--cores', 'none', '--parity', '--out', out,
                '--emit-failing', os.path.join(other, 'failing.txt'))
        assert r.returncode == 1 and 'ParityWriteRefused' in r.stderr and 'PARITY GUARD' in r.stderr, r.stderr
        assert os.listdir(other) == []
        assert 'PARITY GUARD' in open(os.path.join(out, '_records', '_parity_guard_refusals.log')).read()
        r = cli('verify', 'csv_table', good, '--cores', 'none', '--parity', '--out', out,
                '--emit-failing', os.path.join(out, 'failing.txt'))
        assert r.returncode == 0 and os.path.isfile(os.path.join(out, 'failing.txt')), r.stderr


# =====================================================================================================================
# item 6 - verify compare
# =====================================================================================================================

def _parity_side(root, side, data, gittag, utc, bmaj=0.25, rows=None, rec_status='done'):
    """One side of a synthetic parity pair: 3 FITS (one renamed on side B), a CSV, a JSON, a run record."""
    import numpy as np
    from astropy.io import fits
    cell = os.path.join(root, '04h_Red')
    os.makedirs(cell, exist_ok=True)
    stamp = utc.replace('-', '').replace(':', '')[:15] + 'Z'

    def fits_file(name, arr):
        h = fits.PrimaryHDU(np.asarray(arr, np.float32))
        h.header['BMAJ'] = bmaj
        h.header['BUNIT'] = 'Jy/beam'
        h.header['DATE'] = utc
        h.header['GITTAG'] = gittag
        h.header['HISTORY'] = f'built {utc} by masters_{stamp}_lwacalim09_4242 in {root}/04h_Red'
        h.header['HISTORY'] = 'constituents: 2024-12-20 2024-12-21'
        h.writeto(os.path.join(cell, name), overwrite=True, checksum=True)

    fits_file('I_master.fits', data)
    fits_file('unchanged.fits', np.ones((4, 4)))
    fits_file('old_name.fits' if side == 'A' else 'new_name.fits', np.full((3, 3), 2.0))
    with open(os.path.join(cell, 'detections.csv'), 'w') as f:
        f.write('id,ra,flux,path\n' + ''.join(f'{r[0]},{r[1]},{r[2]},{root}/04h_Red/I_master.fits\n'
                                              for r in (rows or [(1, '10.5', '1.25'), (2, '11.0', '0.5')])))
    with open(os.path.join(cell, 'unit.json'), 'w') as f:
        json.dump({'run_id': f'search_{side}', 'started': utc, 'status': rec_status, 'n': [1, 2, 3],
                   'out': f'{root}/04h_Red'}, f)
    os.makedirs(os.path.join(root, '_records', 'masters'), exist_ok=True)
    with open(os.path.join(root, '_records', 'masters', 'x.json'), 'w') as f:
        json.dump({'side': side}, f)                       # compared only with --include-records


def test_item6_verify_compare():
    import csv as _csv
    import numpy as np
    from astropy.io import fits
    from exopipe.phase3.core import compare as cmp
    with Scratch('item6') as d:
        data = np.arange(64, dtype=np.float32).reshape(8, 8)
        data[0, 0] = np.nan
        A, B = os.path.join(d, 'A'), os.path.join(d, 'B')
        _parity_side(A, 'A', data, 'exopipe@d4889d3', '2026-10-12T22:00:00')
        _parity_side(B, 'B', data, 'exopipe@1fd3460', '2026-10-13T03:10:00')
        open(os.path.join(A, 'only_in_old.csv'), 'w').write('a\n1\n')
        mp = os.path.join(d, 'map.csv')
        open(mp, 'w').write('old,new\n04h_Red/old_name.fits,04h_Red/new_name.fits\n')
        pd_file = os.path.join(d, 'permitted.csv')

        def permitted(*rows):
            with open(pd_file, 'w') as f:
                f.write('# signatures for this test\nid,ledger_id,class,path,kind,key,tolerance,note\n')
                f.write(''.join(','.join(r) + '\n' for r in rows))
            return pd_file

        def diffs(rep):
            return [r for r in rep.rows if r['level'] == 'DIFF']

        drop = ('PD-DROP', 'FIX-X', '*', 'only_in_old.csv', 'missing', '', '', 'retired product')
        # only provenance differs (DATE ignored, HISTORY timestamps/run id/root normalised, GITTAG permitted as
        # PD-GITTAG) - and one file is missing in B
        rep = cmp.compare_roots(A, B, 'auto', mapping=mp)
        assert rep.pairs == 5 and rep.verdict == 'FAIL', rep.summary()
        assert [(r['kind'], r['a']) for r in diffs(rep)] == [('missing', 'only_in_old.csv')], diffs(rep)
        perm = [r for r in rep.rows if r['level'] == 'PERM']
        assert len(perm) == 3 and {(r['key'], r['permitted_by']) for r in perm} == {('GITTAG', 'PD-GITTAG')}
        # the missing file permitted by a named signature -> PASS
        rep = cmp.compare_roots(A, B, 'auto', mapping=mp, permitted=permitted(drop))
        assert rep.verdict == 'PASS' and rep.rules_used == {'PD-DROP': 1, 'PD-GITTAG': 3}, rep.summary()
        # without the product map the renamed product is missing + extra
        rep = cmp.compare_roots(A, B, 'fits')
        assert sorted((r['kind'], r['a'] or r['b']) for r in diffs(rep)) == \
            [('extra', '04h_Red/new_name.fits'), ('missing', '04h_Red/old_name.fits')]
        # run records are skipped unless --include-records
        assert cmp.compare_roots(A, B, 'json').verdict == 'PASS'
        rep = cmp.compare_roots(A, B, 'json', opts=cmp.Options(include_records=True))
        assert [r['key'] for r in diffs(rep)] == ['side']

        # FITS: one pixel off by 1e-3, one NaN more, a science card changed (BMAJ, in every file of B2)
        B2 = os.path.join(d, 'B2')
        d2 = data.copy()
        d2[4, 4] += 1e-3
        d2[1, 1] = np.nan
        _parity_side(B2, 'B', d2, 'exopipe@1fd3460', '2026-10-13T03:10:00', bmaj=0.26)
        rep = cmp.compare_roots(A, B2, 'fits', mapping=mp)
        got = {(r['kind'], r['key']) for r in diffs(rep) if r['a'] == '04h_Red/I_master.fits'}
        assert got == {('data', ''), ('nanmask', ''), ('header', 'BMAJ')}, got
        drow = next(r for r in diffs(rep) if r['kind'] == 'data')
        assert abs(drow['value'] - 1e-3) < 1e-5 and 'maxabs/rms' in drow['detail'], drow
        assert next(r for r in diffs(rep) if r['kind'] == 'nanmask')['value'] == 1
        # the signature's tolerance decides: 1e-2 permits the data difference, 1e-4 does not
        rules = [('PD-NAN', 'L1', 'fits', '*I_master.fits', 'nanmask', '', '', ''),
                 ('PD-BEAM', 'L2', 'fits', '*', 'header', 'BMAJ', '', '')]
        rep = cmp.compare_roots(A, B2, 'fits', mapping=mp,
                                permitted=permitted(*rules, ('PD-DATA', 'L3', 'fits', '*', 'data', '', '1e-2', '')))
        assert rep.verdict == 'PASS', diffs(rep)
        assert rep.rules_used == {'PD-NAN': 1, 'PD-BEAM': 3, 'PD-DATA': 1, 'PD-GITTAG': 3}, rep.rules_used
        rep = cmp.compare_roots(A, B2, 'fits', mapping=mp,
                                permitted=permitted(*rules, ('PD-DATA', 'L3', 'fits', '*', 'data', '', '1e-4', '')))
        assert rep.verdict == 'FAIL' and [r['kind'] for r in diffs(rep)] == ['data']
        # the comparison tolerance (--fits-atol) makes the pixels equal
        rep = cmp.compare_roots(A, B2, 'fits', mapping=mp, permitted=permitted(*rules),
                                opts=cmp.Options(fits_atol=1e-2))
        assert rep.verdict == 'PASS', diffs(rep)
        # a real HISTORY change is a difference
        B3 = os.path.join(d, 'B3')
        _parity_side(B3, 'B', data, 'exopipe@1fd3460', '2026-10-13T03:10:00')
        with fits.open(os.path.join(B3, '04h_Red', 'unchanged.fits'), mode='update') as h:
            h[0].header['HISTORY'] = 'an extra processing step'
        rep = cmp.compare_roots(A, B3, 'fits', mapping=mp)
        assert [r['kind'] for r in diffs(rep)] == ['history'], diffs(rep)

        # CSV: exact by default (the path column differs only by root); rtol; row order ignored; keyed rows
        B4 = os.path.join(d, 'B4')
        _parity_side(B4, 'B', data, 'x', '2026-10-13T03:10:00', rows=[(2, '11.0', '0.5'), (1, '10.5', '1.2500000001')])
        rep = cmp.compare_roots(A, B4, 'csv')
        assert [(r['kind'], r['key']) for r in diffs(rep) if r['kind'] != 'missing'] == [('cell', 'flux')]
        assert cmp.compare_roots(A, B4, 'csv', permitted=permitted(drop),
                                 opts=cmp.Options(rtol=1e-6)).verdict == 'PASS'
        B5 = os.path.join(d, 'B5')
        _parity_side(B5, 'B', data, 'x', '2026-10-13T03:10:00',
                     rows=[(1, '10.5', '1.25'), (2, '11.0', '0.5'), (3, '12.0', '0.1')])
        rep = cmp.compare_roots(A, B5, 'csv', opts=cmp.Options(csv_keys=['id']))
        got = sorted((r['kind'], r['detail']) for r in diffs(rep) if r['a'] == '04h_Red/detections.csv')
        assert got == [('rows', 'row count A=2 B=3'), ('rows', "row in B only: ('3',)")], got

        # JSON: ignore keys and roots; a real value change
        B6 = os.path.join(d, 'B6')
        _parity_side(B6, 'B', data, 'x', '2026-10-13T03:10:00', rec_status='refused')
        rep = cmp.compare_roots(A, B6, 'json')
        assert [(r['key'], r['detail']) for r in diffs(rep)] == [('status', "'done' vs 'refused'")], diffs(rep)

        # the command: report files; exit 0 only when every difference is permitted
        rdir = os.path.join(d, 'report')
        r = cli('verify', 'compare', A, B, '--class', 'auto', '--map', mp, '--permitted-diffs', permitted(drop),
                '--report', rdir, '--cores', 'none')
        assert r.returncode == 0 and 'verdict: PASS' in r.stdout, (r.returncode, r.stdout, r.stderr)
        s = json.load(open(os.path.join(rdir, 'parity_summary.json')))
        assert s['verdict'] == 'PASS' and s['pairs_compared'] == 5 and s['rules_source_sha256'] == cmp.SOURCE_SHA256
        rows = list(_csv.DictReader(open(os.path.join(rdir, 'parity_report.csv'))))
        assert {x['level'] for x in rows} == {'OK', 'PERM'} and os.path.isfile(os.path.join(rdir, 'parity_summary.txt'))
        r = cli('verify', 'compare', A, B2, '--class', 'fits', '--map', mp, '--report', rdir, '--cores', 'none')
        assert r.returncode == 2 and 'verdict: FAIL' in r.stdout and 'DIFF  nanmask' in r.stdout, r.stdout
        r = cli('verify', 'compare', A, B4, '--class', 'csv', '--rtol', '1e-6', '--permitted-diffs', permitted(drop),
                '--report', rdir, '--cores', 'none')
        assert r.returncode == 0, r.stdout
        empty_a, empty_b = os.path.join(d, 'e1'), os.path.join(d, 'e2')
        os.makedirs(empty_a)
        os.makedirs(empty_b)
        assert cli('verify', 'compare', empty_a, empty_b, '--report', rdir, '--cores', 'none').returncode == 3
        assert cli('verify', 'compare', A, '--report', rdir, '--cores', 'none').returncode == 1
        bad = os.path.join(d, 'bad_pd.csv')
        open(bad, 'w').write('just,some,columns\n1,2,3\n')
        r = cli('verify', 'compare', A, B, '--permitted-diffs', bad, '--report', rdir, '--cores', 'none')
        assert r.returncode == 1 and 'permitted-differences file' in r.stderr, r.stderr


# =====================================================================================================================

# =====================================================================================================================
# item 7 - B-30 fallback sites (core/fallback.py): refuse / count, sticky refusal, children (core.legacy.run)
# =====================================================================================================================

KEY = 'fallbacks.IMAGE_TIME_FALLBACK'


def test_item7_fallback_policy_count_refuse():
    from exopipe.phase3.core import fallback
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.masters import coadd
    from astropy.io import fits
    with cfg_restored() as CFG:
        assert fallback.policy(KEY) == 'refuse' and CFG.ruled(KEY), 'the ruled default is refuse, with its ruling'
        raises(KeyError, fallback.policy, 'fallbacks.IMAGE_TIME')                 # a fallback key ends in _FALLBACK
        raises(KeyError, fallback.policy, 'catalog.SEARCH_QUARANTINE_FALLBACK')    # ... and lives in section fallbacks
        assert KEY in fallback.all_keys() and fallback.all_keys()[KEY] == 'refuse'
        h = fits.Header()
        n0 = len(fallback.REFUSALS)
        e = raises(fallback.FallbackRefused, coadd.get_midpoint_time, h, '/x/no_time_here.fits')
        assert KEY in str(e) and 'no_time_here.fits' in str(e) and isinstance(e, Refusal)
        assert fallback.REFUSALS[n0:] and fallback.REFUSALS[-1][0] == KEY
        # a header time or a name timestamp is not a fallback: no site reached
        m = fallback.mark()
        h2 = fits.Header()
        h2['DATE-OBS'] = '2025-08-16T09:36:31'
        assert coadd.get_midpoint_time(h2).isot.startswith('2025-08-16T09:36:31')
        assert coadd.get_midpoint_time(h, '/x/23MHz-I-20250816_093631-image.fits').isot.startswith('2025-08-16T09:36:31')
        assert fallback.since(m) == ([], {})
        # production value: the default, counted
        CFG.authorize([], out='/tmp/x')
        CFG.set(KEY, 'fallback')
        m = fallback.mark()
        assert coadd.get_midpoint_time(h, '/x/a.fits').isot.startswith('2025-01-01T00:00:00')
        coadd.get_midpoint_time(h)
        assert fallback.since(m) == ([], {KEY: 2})
        CFG.set(KEY, 'sometimes')
        raises(Refusal, fallback.policy, KEY)                                       # one of refuse / fallback


def test_item7_fallback_sticky_unit_and_record():
    from exopipe.phase3.core import fallback, record
    from exopipe.phase3.lib.masters import coadd
    from astropy.io import fits
    with cfg_restored() as CFG, Scratch('item7') as d:
        CFG.authorize([], out=d)
        out = os.path.join(d, 'a.csv')
        open(out, 'w').write('a\n1\n')
        run = record.Run('t7', ['x'], d, CFG, config_sections=())
        # a refusal caught by the code between the site and the unit (a broad except, as the moved modules hold):
        # the unit still refuses, naming the site
        with run.unit('swallowed') as u:
            try:
                coadd.get_midpoint_time(fits.Header(), '/x/b.fits')
            except Exception:                    # noqa: BLE001 - what production code does
                pass
            u.add_output(out, 'csv_table')
        assert u.status == 'refused' and KEY in u.reason and 'caught before the unit' in u.reason, (u.status, u.reason)
        rec = record.load_record(d, 't7', 'swallowed')
        assert rec['fallback_refusals'][0]['key'] == KEY
        # the next unit starts clean
        with run.unit('clean') as u2:
            u2.add_output(out, 'csv_table')
        assert u2.status == 'done' and 'fallback_refusals' not in record.load_record(d, 't7', 'clean')
        # production value: done, the counts in the record
        CFG.set(KEY, 'fallback')
        with run.unit('counted') as u3:
            coadd.get_midpoint_time(fits.Header())
            coadd.get_midpoint_time(fits.Header())
            u3.add_output(out, 'csv_table')
        rec = record.load_record(d, 't7', 'counted')
        assert u3.status == 'done' and rec['fallbacks'] == {KEY: 2}, (u3.status, rec.get('fallbacks'))


CHILD = """
import sys
sys.path.insert(0, %r)
from astropy.io import fits
from exopipe.phase3.lib.masters import coadd
for _ in range(int(sys.argv[1])):
    try:
        coadd.get_midpoint_time(fits.Header(), '/x/child.fits')
    except Exception:
        pass                       # swallowed: the child exits 0
print('child done')
"""


def test_item7_fallback_children():
    from exopipe.phase3.core import fallback, legacy, record
    with cfg_restored() as CFG, Scratch('item7c') as d:
        CFG.authorize([], out=d)
        script = os.path.join(d, 'child.py')
        open(script, 'w').write(CHILD % REPO)
        out = os.path.join(d, 'a.csv')
        open(out, 'w').write('a\n1\n')
        run = record.Run('t7c', ['x'], d, CFG, config_sections=())
        # ruled: the child refuses at the site and swallows it; the result and the parent unit are refused all the same
        with run.unit('child_refuses') as u:
            res = legacy.run(script, ['1'], log=os.path.join(d, 'c1.log'), unit=u)
            u.add_output(out, 'csv_table')
        assert res.rc == 0 and res.refused and KEY in res.reason and res.fallback_refusals, (res.rc, res.reason)
        assert u.status == 'refused' and KEY in u.reason, (u.status, u.reason)
        # production value, set by an override in the PARENT: the child counts (the policy reaches it as JSON)
        CFG.set(KEY, 'fallback')
        with run.unit('child_counts') as u2:
            res = legacy.run(script, ['3'], log=os.path.join(d, 'c2.log'))          # no unit=: the process state carries it
            u2.add_output(out, 'csv_table')
        assert res.ok and res.fallbacks == {KEY: 3}, (res.rc, res.reason, res.fallbacks)
        rec = record.load_record(d, 't7c', 'child_counts')
        assert u2.status == 'done' and rec['fallbacks'] == {KEY: 3}, (u2.status, rec.get('fallbacks'))


def _fork_worker(i):
    """A fork-pool worker that reaches a site and swallows it, as coadd._run_final_worker does."""
    from astropy.io import fits
    from exopipe.phase3.lib.masters import coadd
    try:
        coadd.get_midpoint_time(fits.Header(), f'/x/w{i}.fits')
    except Exception:                                # noqa: BLE001
        pass
    return i


def test_item7_fallback_fork_workers():
    """Sites reached inside FORKED pool workers (their memory dies with them, and they skip atexit): the events reach the
    parent's unit through the spool - a swallowed refusal still refuses the unit, a count is still recorded."""
    import multiprocessing as mp
    from exopipe.phase3.core import record
    with cfg_restored() as CFG, Scratch('item7f') as d:
        CFG.authorize([], out=d)
        out = os.path.join(d, 'a.csv')
        open(out, 'w').write('a\n1\n')
        run = record.Run('t7f', ['x'], d, CFG, config_sections=())
        with run.unit('fork_refuse') as u:
            with mp.get_context('fork').Pool(2) as pool:
                assert pool.map(_fork_worker, range(3)) == [0, 1, 2]
            u.add_output(out, 'csv_table')
        assert u.status == 'refused' and KEY in u.reason, (u.status, u.reason)
        CFG.set(KEY, 'fallback')
        with run.unit('fork_count') as u2:
            with mp.get_context('fork').Pool(2) as pool:
                pool.map(_fork_worker, range(5))
            u2.add_output(out, 'csv_table')
        rec = record.load_record(d, 't7f', 'fork_count')
        assert u2.status == 'done' and rec['fallbacks'] == {KEY: 5}, (u2.status, rec.get('fallbacks'))


def test_item7_fallback_threads_and_outside_units():
    """Units in threads each take their own thread's events; events outside every unit go to the run record, and a
    refusal there refuses the run."""
    import threading
    from astropy.io import fits
    from exopipe.phase3.core import record
    from exopipe.phase3.lib.masters import coadd
    with cfg_restored() as CFG, Scratch('item7t') as d:
        CFG.authorize([], out=d)
        CFG.set(KEY, 'fallback')
        out = os.path.join(d, 'a.csv')
        open(out, 'w').write('a\n1\n')
        run = record.Run('t7t', ['x'], d, CFG, config_sections=())
        go = threading.Barrier(2)

        def work(name, n):
            with run.unit(name) as u:
                go.wait()
                for _ in range(n):
                    coadd.get_midpoint_time(fits.Header())
                u.add_output(out, 'csv_table')
        ts = [threading.Thread(target=work, args=(f'th{n}', n)) for n in (2, 7)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        got = {n: record.load_record(d, 't7t', f'th{n}').get('fallbacks') for n in (2, 7)}
        assert got == {2: {KEY: 2}, 7: {KEY: 7}}, got
        coadd.get_midpoint_time(fits.Header())                      # outside every unit: the run's record holds it
        assert run.fallback_outside() == ([], {KEY: 1}) and run.exit_code() == 0, run.fallback_outside()
        CFG.set(KEY, 'refuse')
        try:
            coadd.get_midpoint_time(fits.Header(), '/x/outside.fits')
        except Exception:                            # noqa: BLE001 - caught outside a unit (mosaic's task runner)
            pass
        assert run.exit_code() == 2, 'a refusal outside every unit refuses the run'
        summ = json.load(open(run.write_summary()))
        assert summ['fallbacks_outside_units'] == {KEY: 1} and summ['fallback_refusals_outside_units'][0]['key'] == KEY


def test_item7_fallback_sticky_outcomes():
    """After a caught refusal: a later error keeps its traceback in the notes (the unit refuses, the site named); a
    KeyboardInterrupt keeps its own outcome ('error: interrupted') and propagates."""
    from astropy.io import fits
    from exopipe.phase3.core import record
    from exopipe.phase3.lib.masters import coadd
    with cfg_restored() as CFG, Scratch('item7s') as d:
        run = record.Run('t7s', ['x'], d, CFG, config_sections=())
        with run.unit('later_error') as u:
            try:
                coadd.get_midpoint_time(fits.Header(), '/x/a.fits')
            except Exception:                        # noqa: BLE001
                pass
            {}['missing']
        assert u.status == 'refused' and KEY in u.reason and any('Traceback' in n and 'KeyError' in n for n in u.notes)
        try:
            with run.unit('interrupted') as u2:
                try:
                    coadd.get_midpoint_time(fits.Header(), '/x/b.fits')
                except Exception:                    # noqa: BLE001
                    pass
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError('KeyboardInterrupt swallowed')
        assert u2.status == 'error' and u2.reason == 'interrupted', (u2.status, u2.reason)


PROD_MOD = """def f(h, p):
    try:
        x = read(p)
    except Exception:
        x = None
    if h.get('A'):
        y = 1
    if not x:
        return 0
    return x + y
"""
BRANCH_MOD = """def f(h, p):
    try:
        x = read(p)
    except Exception:
        x = None
        from exopipe.phase3.core import fallback      # WAVE2 B-30
        fallback.site('fallbacks.IMAGE_TIME_FALLBACK', f'{p}: unreadable')
    if h.get('A'):
        y = 1
    else:
        fallback.site('fallbacks.IMAGE_TIME_FALLBACK', 'no A')
    if 'B' not in h:
        from exopipe.phase3.core import fallback      # WAVE2 B-30
        fallback.site('fallbacks.IMAGE_TIME_FALLBACK',
                      'no B')
    if not x:
        return 0
    return x + y
"""


def test_item7_site_lines_pure_insertion():
    """A B-30 site is a pure insertion of whole lines: removing them gives the production text byte for byte (dewarp G-D0b
    compares that way); a comment edit, a changed condition or an `if` that does more than a site is still a difference."""
    from exopipe.phase3.core import fallback
    assert fallback.strip_site_lines(BRANCH_MOD) == PROD_MOD, fallback.strip_site_lines(BRANCH_MOD)
    assert fallback.same_code_modulo_sites(PROD_MOD, BRANCH_MOD)
    assert not fallback.same_code_modulo_sites(PROD_MOD + '# local edit\n', BRANCH_MOD)
    assert not fallback.same_code_modulo_sites(PROD_MOD, BRANCH_MOD.replace("if h.get('A'):", "if h.get('a'):"))
    more = BRANCH_MOD.replace("    if 'B' not in h:\n", "    if 'B' not in h:\n        z = 2\n")
    assert not fallback.same_code_modulo_sites(PROD_MOD, more), 'an if with real code is not a site'


GRANDPARENT = """
import subprocess, sys
sys.path.insert(0, %r)
from astropy.io import fits
from exopipe.phase3.lib.masters import coadd
try:
    coadd.get_midpoint_time(fits.Header(), '/x/parent.fits')
except Exception:
    pass
subprocess.run([sys.executable, %r, '3'], check=True)     # a plain subprocess grandchild inheriting EXOPIPE3_FALLBACK_OUT
print('parent done')
"""


def test_item7_fallback_plain_subprocess_and_grandchild():
    """A descendant NOT started by core.legacy.run (a plain subprocess, as vclean_masters starts search_v.py) uses the step
    process's policies (exported by cli.main: export_env) and spools its events to it - the running unit records them;
    a grandchild of a legacy.run child writes its own dump file (it would overwrite its parent's)."""
    import subprocess
    from exopipe.phase3.core import fallback, legacy, record
    keep = {k: os.environ.get(k) for k in ('EXOPIPE3_FALLBACK_POLICY', 'EXOPIPE3_FALLBACK_POLICY_PID',
                                           'EXOPIPE3_FALLBACK_SPOOL_OWNER')}
    try:
        with cfg_restored() as CFG, Scratch('item7p') as d:
            CFG.authorize([], out=d)
            script = os.path.join(d, 'child.py')
            open(script, 'w').write(CHILD % REPO)
            out = os.path.join(d, 'a.csv')
            open(out, 'w').write('a\n1\n')
            run = record.Run('t7p', ['x'], d, CFG, config_sections=())
            CFG.set(KEY, 'fallback')
            fallback.export_env()                        # what cli.main does after its overrides
            assert fallback.policy(KEY) == 'fallback'    # the exporting process keeps reading config
            with run.unit('plain_counts') as u:
                subprocess.run([PY, script, '2'], check=True, capture_output=True)
                u.add_output(out, 'csv_table')
            rec = record.load_record(d, 't7p', 'plain_counts')
            assert u.status == 'done' and rec['fallbacks'] == {KEY: 2}, (u.status, rec.get('fallbacks'))
            CFG.set(KEY, 'refuse')
            fallback.export_env()
            with run.unit('plain_refuses') as u2:
                subprocess.run([PY, script, '1'], check=True, capture_output=True)   # exits 0: it swallowed the refusal
                u2.add_output(out, 'csv_table')
            assert u2.status == 'refused' and KEY in u2.reason, (u2.status, u2.reason)
            CFG.set(KEY, 'fallback')
            parent = os.path.join(d, 'parent.py')
            open(parent, 'w').write(GRANDPARENT % (REPO, script))
            res = legacy.run(parent, [], log=os.path.join(d, 'gp.log'))
            assert res.ok and res.fallbacks == {KEY: 4}, (res.rc, res.reason, res.fallbacks)
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        fallback._ENV = None


def test_item7_fallback_registry():
    """Every fallback key is RULED, is set to 'fallback' in the production parity file, and is named by a
    fallback.site(<key>) call; every such call names its key as a string literal that is a config key (an AST scan: no
    quoting style or computed key escapes it - a computed key would fail at run time under every policy)."""
    import ast
    import re
    from exopipe.phase3.core import fallback
    from exopipe.phase3.config import CFG
    keys = fallback.all_keys()
    assert keys, 'no fallback keys'
    calls, bad = set(), []
    for dp, dn, fn in os.walk(os.path.join(REPO, 'exopipe')):
        dn[:] = [x for x in dn if x != '__pycache__']
        for f in fn:
            if not f.endswith('.py') or f == 'fallback.py':
                continue
            p = os.path.join(dp, f)
            for n in ast.walk(ast.parse(open(p).read())):
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'site'
                        and isinstance(n.func.value, ast.Name) and n.func.value.id == 'fallback'):
                    a0 = n.args[0] if n.args else None
                    if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                        calls.add(a0.value)
                    else:
                        bad.append(f'{os.path.relpath(p, REPO)}:{n.lineno}')
    assert not bad, f'fallback.site calls without a literal key: {bad}'
    assert calls == set(keys), ('unkeyed sites', sorted(calls - set(keys)), 'dead keys', sorted(set(keys) - calls))
    assert all(CFG.ruled(k) for k in keys), [k for k in keys if not CFG.ruled(k)]
    par = open(os.path.join(REPO, 'exopipe', 'phase3', 'parity', 'production_20261013.cfg')).read()
    missing = [k for k in keys if not re.search(rf"^{re.escape(k)} = 'fallback'$", par, re.M)]
    assert not missing, f'not set to fallback in the parity file: {missing}'


def test_item3_child_code_fingerprint_per_process():
    """core.legacy.run records the code of every Python process of a child: each writes <path>.<pid> (parity sitecustomize)
    and the parent merges them - a grandchild's own module is in the record (one shared file kept only the last writer's),
    and a multiprocessing forkserver, which exits AFTER the child and runs atexit, neither erases the child's record nor
    leaves a stray file (it imported no recorded code: nothing written). 2026-10-14: phase3_snapshot_refastro's forkserver."""
    import glob as _glob
    from exopipe.phase3.core import legacy
    with Scratch('fp') as d:
        for name in ('exopipe_fp_main', 'exopipe_fp_grand'):
            open(os.path.join(d, f'{name}.py'), 'w').write('X = 1\n')
        grand = os.path.join(d, 'grand.py')
        open(grand, 'w').write(f'import sys\nsys.path.insert(0, {d!r})\nimport exopipe_fp_grand\n')
        child = os.path.join(d, 'child.py')
        open(child, 'w').write(
            'import multiprocessing as mp, os, subprocess, sys\n'
            'def sq(x):\n    return x * x\n'
            "if __name__ == '__main__':\n"
            f'    sys.path.insert(0, {d!r})\n'
            '    import exopipe_fp_main\n'
            f'    subprocess.run([sys.executable, {grand!r}], check=True)\n'
            "    with mp.get_context('forkserver').Pool(2) as p:\n"
            "        print('POOL', sum(p.map(sq, range(10))))\n")
        before = set(_glob.glob(os.path.join(tempfile.gettempdir(), 'exopipe3_fp_*')))
        r = legacy.run(child, [], log=os.path.join(d, 'c.log'))
        assert r.ok and 'POOL 285' in r.tail, r.tail[-300:]
        names = {os.path.basename(f) for f in r.fingerprint}
        assert {'exopipe_fp_main.py', 'exopipe_fp_grand.py', 'child.py'} <= names, sorted(names)
        time.sleep(1.0)                         # the forkserver exits a moment after the child
        stray = set(_glob.glob(os.path.join(tempfile.gettempdir(), 'exopipe3_fp_*'))) - before
        assert not stray, f'stray fingerprint files: {sorted(stray)}'


def test_item7_parity_file_keys():
    """The production parity file (parity/production_20261013.cfg): every line names a config key; outside section fallbacks
    (where a count-only site's ruled default IS production's 'fallback') each line differs from the config default - a line
    equal to the default would mean the default is not the ruled value, or the line is dead; and each such key carries the
    2026-10-13 ruling, or one of Gregg's 2026-10-14 refactor rulings, in RULED unless it is a PROPOSED value awaiting Gregg
    (listed here; none today)."""
    import ast
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import fallback
    B = 'R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32'
    LATER = {'R-GREGG-20261014-REFACTOR-A2B-C8B-C8C', 'R-GREGG-20261014-REFACTOR-A2-C8D',
             'R-GREGG-20261014-REFACTOR-DECISIONS-LIST-RECOMMENDATIONS'}     # clv15 fix_ledger.csv
    PROPOSED = set()       # 2026-10-14: MASTER_SPAN_SOURCE and EPOCH_GATE_MAX_PASSES ruled (DECISIONS-LIST-RECOMMENDATIONS)
    path = os.path.join(REPO, 'exopipe', 'phase3', 'parity', 'production_20261013.cfg')
    bad, n = [], 0
    for ln in open(path):
        ln = ln.split('#', 1)[0].strip()
        if not ln:
            continue
        k, v = (x.strip() for x in ln.split('=', 1))
        n += 1
        try:
            default = CFG.get(k)
        except Exception as e:                     # noqa: BLE001 - an unknown key is the failure reported
            bad.append(f'{k}: not a config key ({e})')
            continue
        if k.startswith(fallback.SECTION + '.'):
            continue
        if ast.literal_eval(v) == default:
            bad.append(f'{k}: the parity value {v} equals the config default')
        ids = CFG.ruled(k) or ()
        ids = (ids,) if isinstance(ids, str) else tuple(ids)
        if k not in PROPOSED and not ({B} | LATER) & set(ids):
            bad.append(f'{k}: no {B} (or a 2026-10-14 refactor ruling) in RULED ({ids})')
    assert n and not bad, bad


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
