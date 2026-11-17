#!/usr/bin/env python3
"""STANDING TEST: no memory maps (Gregg 2026-11-17, R-GREGG-20261117-NO-MMAP-OF-LUSTRE-FILES-READ-OR-WRITE: "never memory-map a
Lustre file, for reading OR writing"; the 11-09 and 11-16 lwacalim07 Lustre client crashes were on mapped pages).

1. Importing exopipe turns astropy's FITS memory map off (fits.conf.use_memmap False), so default fits.open/getdata reads.
2. No source file under exopipe/, tools/ or tests/ asks for a map: memmap=True, mmap_mode=, np.memmap / numpy.memmap,
   mmap.mmap(, or use_memmap = True. Read slices with ordinary reads (ImageHDU.section, f.seek + f.read) instead.
3. An entry point that calls astropy.io.fits (any file under tools/, or a module under exopipe/ with an
   `if __name__ == '__main__':` block, since it can be run by path, often in another conda env, where exopipe/__init__.py
   never runs) imports exopipe or sets fits.conf.use_memmap = False at module level (code review of PR 81). A library
   module imported as exopipe.<...> is covered by the package's __init__.
4. IN-PLACE CHILDREN (the production scripts core.legacy.run starts, serve's shell-launched Python) import no exopipe:
   core/parity_sitecustomize/sitecustomize.py, first on their PYTHONPATH, sets fits.conf.use_memmap False right after
   astropy.io.fits loads. A child that opens a FITS file with the defaults has the switch off and no map of the file in
   /proc/self/maps; a child that never imports astropy does not import it; the parity guard's astropy hook and the sub-LST
   frozen-code guard still compose with it. FALSIFIER: the same child without the sitecustomize has the switch on and maps a
   local (non-Lustre) file - so the map check sees a map. Every child reads an empty astropy configuration directory of the
   test's own (XDG_CONFIG_HOME), so its switch starts at astropy's default whatever the running user's astropy.cfg says. A
   child never opens the Lustre file unless its switch is off. On the cluster (the site file present) the children run
   through core.legacy.run under py38 ('default') and py36 and also read a small Lustre FITS; elsewhere (CI) the same checks
   run with this interpreter on a local file.

Run standalone: python tests/test_no_mmap.py    Under pytest: pytest tests/test_no_mmap.py
"""
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
SCAN = ('exopipe', 'tools', 'tests')
PATTERN = re.compile(r"memmap\s*=\s*True|mmap_mode\s*=|\bnp\.memmap\b|\bnumpy\.memmap\b|\bmmap\.mmap\s*\(|use_memmap\s*=\s*True")


def offending_lines():
    out = []
    for top in SCAN:
        for d, _sub, files in os.walk(os.path.join(REPO, top)):
            for fn in files:
                p = os.path.join(d, fn)
                if not fn.endswith('.py') or fn.endswith('test_no_mmap.py'):     # this file names the forms it refuses
                    continue
                with open(p, encoding='utf-8', errors='replace') as f:
                    for i, line in enumerate(f, 1):
                        code = line.split('#', 1)[0]
                        if PATTERN.search(code):
                            out.append(f'{os.path.relpath(p, REPO)}:{i}: {line.strip()}')
    return out


FITS_CALLS = {'open', 'getdata', 'getheader', 'getval', 'setval', 'update', 'append', 'writeto'}


def _top_level(body):
    """Statements run at import: the module body and the bodies of top-level try/if blocks."""
    for n in body:
        yield n
        for attr in ('body', 'orelse', 'finalbody'):
            if isinstance(n, (ast.Try, ast.If)):
                yield from _top_level(getattr(n, attr, []))
        if isinstance(n, ast.Try):
            for hnd in n.handlers:
                yield from _top_level(hnd.body)


def _is_main_guard(n):
    return (isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
            and n.test.left.id == '__name__')


def entry_points_without_switch():
    """Entry points (tools/*, or exopipe/ modules with a __main__ block) that call fits.<open|getdata|...> but neither
    import exopipe nor set use_memmap = False at module level."""
    out = []
    for top in ('exopipe', 'tools'):
        for d, _sub, files in os.walk(os.path.join(REPO, top)):
            for fn in files:
                if not fn.endswith('.py'):
                    continue
                p = os.path.join(d, fn)
                with open(p, encoding='utf-8', errors='replace') as f:
                    t = ast.parse(f.read(), p)
                calls = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in FITS_CALLS
                            and isinstance(n.func.value, ast.Name) and n.func.value.id in ('fits', 'pyfits')
                            for n in ast.walk(t))
                if not calls or (top == 'exopipe' and not any(_is_main_guard(n) for n in t.body)):
                    continue
                ok = False
                for n in _top_level(t.body):
                    if isinstance(n, ast.Import) and any(a.name.split('.')[0] == 'exopipe' for a in n.names):
                        ok = True
                    elif isinstance(n, ast.ImportFrom) and (n.module or '').split('.')[0] == 'exopipe' and not n.level:
                        ok = True
                    elif (isinstance(n, ast.Assign) and any(isinstance(x, ast.Attribute) and x.attr == 'use_memmap'
                                                            for x in n.targets)
                          and isinstance(n.value, ast.Constant) and n.value.value is False):
                        ok = True
                if not ok:
                    out.append(os.path.relpath(p, REPO))
    return sorted(out)


def test_import_turns_memmap_off():
    import exopipe  # noqa: F401
    from astropy.io import fits
    assert fits.conf.use_memmap is False


def test_no_memory_map_requested_in_source():
    bad = offending_lines()
    assert not bad, 'memory maps requested (refused by R-GREGG-20261117-NO-MMAP-OF-LUSTRE-FILES-READ-OR-WRITE):\n' + '\n'.join(bad)


def test_every_fits_entry_point_turns_memmap_off():
    bad = entry_points_without_switch()
    assert not bad, ('modules calling astropy.io.fits without importing exopipe or setting fits.conf.use_memmap = False at '
                     'module level (R-GREGG-20261117-NO-MMAP-OF-LUSTRE-FILES-READ-OR-WRITE):\n' + '\n'.join(bad))


def test_pattern_catches_the_forms():
    for s in ("fits.open(p, memmap=True)", "np.load(p, mmap_mode='r')", "np.memmap(p)", "mmap.mmap(fd, 0)",
              "fits.conf.use_memmap = True"):
        assert PATTERN.search(s), s
    for s in ("fits.open(p, memmap=False)", "np.load(p)", "fits.conf.use_memmap = False"):
        assert not PATTERN.search(s), s


# ---- 4. in-place children ------------------------------------------------------------------------------------------------
BOOT = os.path.join(REPO, 'exopipe', 'phase3', 'core', 'parity_sitecustomize')
SITECUSTOMIZE = os.path.join(BOOT, 'sitecustomize.py')
SITE_FILE = '/lustre/gh/main/exopipe_ops/exopipe3_site.py'
# a 23 kB image (65 x 65 float32), git-tracked in clv15 since its pre-refactor snapshot (766b766); read only, and only by a
# child whose switch is off
LUSTRE_FITS = '/lustre/gh/main/clv15/tauboo_best_snap/hwj046_tau_Boo_18MHz-Pilot-Snapshot-20250407_082949-image.fits'
FROZEN_GUARD = '/lustre/gh/main/exopipe_ops/sublst_frozen_guard'      # its sitecustomize.py (the sub-LST frozen-code guard)

# The child (Python 3.6+): argv = result file, mode, FITS paths. 'fits': open each file with the defaults and read its data
# (fits.open and fits.getdata), recording the switch and the file mappings seen meanwhile; 'plain': import nothing; 'late':
# astropy.io.fits first, then the sitecustomize executed by path (a module loaded before it) - and once more (refused).
CHILD = r'''
import json, os, sys
res, mode, paths = sys.argv[1], sys.argv[2], sys.argv[3:]
out = {'python': sys.version.split()[0], 'astropy_at_start': sorted(m for m in sys.modules if m.split('.')[0] == 'astropy')}


def maps():
    rows = []
    with open('/proc/self/maps') as f:
        for line in f:
            p = line.rstrip('\n').split(None, 5)
            if len(p) == 6:
                rows.append(p[5])
    return rows


if mode in ('fits', 'late'):
    import builtins
    from astropy.io import fits
    if mode == 'late':
        out['before'] = fits.conf.use_memmap
        sc = os.environ['NOMMAP_TEST_SITECUSTOMIZE']
        code = compile(open(sc).read(), sc, 'exec')
        exec(code, {'__name__': 'sitecustomize', '__file__': sc})
        try:
            exec(code, {'__name__': 'sitecustomize', '__file__': sc})
            out['second_exec'] = 'ran'
        except ImportError:
            out['second_exec'] = 'ImportError'
    out['use_memmap'] = fits.conf.use_memmap
    out['finders'] = [type(f).__name__ for f in sys.meta_path]
    out['opened'], out['mapped'], out['lustre_maps'] = [], [], []
    for p in paths:
        if p.startswith('/lustre/') and fits.conf.use_memmap is not False:
            continue                       # never map a Lustre file, even when the switch is missing
        seen = []
        with fits.open(p) as h:
            out.setdefault('sums', []).append(float(h[0].data.astype('f8').sum()))
            seen += maps()
        d = fits.getdata(p)
        out['sums'].append(float(d.astype('f8').sum()))
        seen += maps()
        del d
        out['opened'].append(p)
        out['mapped'] += sorted(set(m for m in seen if m == p))
        out['lustre_maps'] += sorted(set(m for m in seen if m.startswith('/lustre/')))
    st = getattr(builtins, '__exopipe3_parity_guard__', {})
    out['guard_writeto_patches'] = sum(1 for (_o, _a, k) in st.get('patched', []) if k == 'HDUList.writeto')
    out['writeto'] = getattr(fits.writeto, '__name__', '')
    refused = os.environ.get('NOMMAP_TEST_REFUSED')
    if refused:
        import numpy as np
        try:
            fits.writeto(refused, np.zeros((2, 2), dtype=np.float32))
            out['refused'] = 'written'
        except PermissionError as e:
            out['refused'] = type(e).__name__
out['astropy_at_end'] = sorted(m for m in sys.modules if m.split('.')[0] == 'astropy')
with open(res, 'w') as f:
    json.dump(out, f)
'''


class _Scratch(object):
    """A node-local scratch directory (TMPDIR) with the child script and a small local FITS image."""

    def __init__(self):
        import numpy as np
        from astropy.io import fits
        self.d = tempfile.mkdtemp(prefix='exopipe3_nommap_')
        self.child = os.path.join(self.d, 'nommap_child.py')
        with open(self.child, 'w') as f:
            f.write(CHILD)
        self.local = os.path.join(self.d, 'local_8x8.fits')
        fits.writeto(self.local, np.arange(64, dtype=np.float32).reshape(8, 8))
        # every child reads an astropy configuration directory of its own (empty): the switch starts at astropy's default,
        # whatever the running user's astropy.cfg says (a use_memmap line there would decide the falsifier)
        self.xdg = os.path.join(self.d, 'xdg')
        os.makedirs(os.path.join(self.xdg, 'astropy'))
        self.n = 0

    def result_path(self):
        self.n += 1
        return os.path.join(self.d, 'result_%d.json' % self.n)

    def close(self):
        shutil.rmtree(self.d, ignore_errors=True)


def _env(pythonpath=(), **extra):
    """os.environ without PYTHONPATH and the exopipe3 child variables (EXOPIPE3_SITE kept), then ``extra``."""
    env = {k: v for k, v in os.environ.items()
           if k != 'PYTHONPATH' and not (k.startswith('EXOPIPE3_') and k != 'EXOPIPE3_SITE') and not k.startswith('SUBLST_')}
    env['NUMPY_MADVISE_HUGEPAGE'] = '0'
    if pythonpath:
        env['PYTHONPATH'] = os.pathsep.join(pythonpath)
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _direct(py, sc, mode, paths=(), pythonpath=(), **extra):
    """Run the child with interpreter ``py`` (no core.legacy.run); returns its result."""
    res = sc.result_path()
    p = subprocess.run([py, sc.child, res, mode] + list(paths), env=_env(pythonpath, XDG_CONFIG_HOME=sc.xdg, **extra),
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=600)
    assert p.returncode == 0 and os.path.isfile(res), (py, mode, p.returncode, p.stdout.decode('utf-8', 'replace')[-2000:])
    with open(res) as f:
        return json.load(f)


def _check_off(r, paths, what):
    assert r['use_memmap'] is False, (what, r)
    assert r['astropy_at_start'] == [], (what, 'astropy imported at start-up', r['astropy_at_start'])
    assert r['opened'] == list(paths), (what, r['opened'])
    assert r['mapped'] == [] and r['lustre_maps'] == [], (what, 'mapped', r['mapped'], r['lustre_maps'])


def _check_falsifier(r, local, what):
    assert r['use_memmap'] is True, (what, 'the child without the sitecustomize has the switch off - falsifier void', r)
    assert r['opened'] == [local], (what, 'only the local file may be opened by a child whose switch is on', r['opened'])
    assert r['mapped'] == [local], (what, 'a mapped file was not seen in /proc/self/maps - the map check is blind', r)
    assert r['lustre_maps'] == [], (what, r['lustre_maps'])


def _check_plain(r, what):
    assert r['astropy_at_start'] == [] and r['astropy_at_end'] == [], (what, 'astropy imported by a child that never uses it',
                                                                        r)


def test_inplace_child_switch_off_lazily():
    """Section 4 with this interpreter and a local file (data-free; CI): the switch, no map, lazy, the falsifier, the parity
    guard's astropy hook composed (its writeto patched once, a write outside the roots still refused), a module loaded before
    the sitecustomize, a second execution refused."""
    sc = _Scratch()
    try:
        py = sys.executable
        _check_off(_direct(py, sc, 'fits', [sc.local], [BOOT]), [sc.local], 'sitecustomize')
        _check_plain(_direct(py, sc, 'plain', (), [BOOT]), 'sitecustomize, plain child')
        _check_falsifier(_direct(py, sc, 'fits', [sc.local]), sc.local, 'no sitecustomize')
        proot = os.path.join(sc.d, 'parity_root')
        os.makedirs(proot)
        outside = os.path.join(REPO, 'tests', '_nommap_write_refused.fits')
        r = _direct(py, sc, 'fits', [sc.local], [BOOT], EXOPIPE3_PARITY_ROOT=proot, EXOPIPE3_WRAPS='{}',
                    EXOPIPE3_FALLBACK_POLICY=json.dumps({'fallbacks.GITTAG_FALLBACK': 'fallback'}), NOMMAP_TEST_REFUSED=outside)
        refused_written = os.path.exists(outside)
        if refused_written:
            os.remove(outside)
        _check_off(r, [sc.local], 'sitecustomize + parity guard')
        assert r['guard_writeto_patches'] == 1 and r['writeto'] == '_w_fits_writeto', ('parity guard astropy hook', r)
        assert r['refused'] == 'ParityWriteRefused' and not refused_written, ('parity guard', r['refused'])
        r = _direct(py, sc, 'late', [sc.local], NOMMAP_TEST_SITECUSTOMIZE=SITECUSTOMIZE)
        assert r['before'] is True and r['second_exec'] == 'ImportError', ('late', r)
        assert r['finders'].count('_FitsNoMmapFinder') == 1, ('late: one finder', r['finders'])
        assert r['use_memmap'] is False and r['mapped'] == [], ('late: a module loaded before the sitecustomize', r)
        print('in-place child, Python %s: switch off, local file not mapped, astropy not imported by a plain child; falsifier '
              '(no sitecustomize): switch on, local file mapped; parity guard: writeto patched once, outside write refused; '
              'module loaded first: switched off, second execution refused' % r['python'])
    finally:
        sc.close()


def test_inplace_children_through_core_legacy_run():
    """Section 4 on the cluster: children of core.legacy.run under py38 ('default') and py36 read a small Lustre FITS with
    the defaults - no map; the same interpreter without the sitecustomize is the falsifier (the Lustre file is not opened
    there); the parity guard (legacy.run parity_root) and the sub-LST frozen-code guard (either first on the path) compose."""
    site = os.environ.get('EXOPIPE3_SITE') or SITE_FILE
    if not os.path.isfile(site):
        print('      (skipped: no site file %s on this node - the core.legacy.run children run on the cluster)' % site)
        return
    os.environ.setdefault('EXOPIPE3_SITE', site)
    assert os.path.isfile(LUSTRE_FITS), ('the small Lustre FITS of this test is gone - name another', LUSTRE_FITS)
    from exopipe.phase3.core import legacy
    sc = _Scratch()
    try:
        for key in ('default', 'py36'):
            py = legacy.interpreter(key)
            paths = [sc.local, LUSTRE_FITS]
            for kw, what in (({}, 'core.legacy.run'), ({'parity_root': os.path.join(sc.d, 'parity_' + key)},
                                                       'core.legacy.run --parity')):
                if kw:
                    os.makedirs(kw['parity_root'])
                res = sc.result_path()
                lr = legacy.run(sc.child, [res, 'fits'] + paths, python=key, env={'XDG_CONFIG_HOME': sc.xdg}, **kw)
                assert lr.ok and os.path.isfile(res), (key, what, lr.rc, lr.reason, lr.tail[-2000:])
                with open(res) as f:
                    r = json.load(f)
                _check_off(r, paths, '%s %s' % (key, what))
                assert r['python'].startswith('3.6' if key == 'py36' else '3.8'), (key, r['python'])
                if kw:
                    assert r['guard_writeto_patches'] == 1 and r['writeto'] == '_w_fits_writeto', (key, what, r)
            res = sc.result_path()
            lr = legacy.run(sc.child, [res, 'plain'], python=key, env={'XDG_CONFIG_HOME': sc.xdg})
            assert lr.ok, (key, lr.reason, lr.tail[-2000:])
            with open(res) as f:
                _check_plain(json.load(f), '%s core.legacy.run plain child' % key)
            _check_falsifier(_direct(py, sc, 'fits', paths), sc.local, '%s without the sitecustomize' % key)
            frozen = 'not composed (no guard file)'
            if os.path.isfile(os.path.join(FROZEN_GUARD, 'sitecustomize.py')):
                froot = os.path.join(sc.d, 'frozen_' + key)
                os.makedirs(froot)
                open(os.path.join(froot, 'MANIFEST.sha256'), 'w').close()
                for order in ((FROZEN_GUARD, BOOT), (BOOT, FROZEN_GUARD)):
                    r = _direct(py, sc, 'fits', paths, order, SUBLST_CODE_ROOT=froot)
                    _check_off(r, paths, '%s frozen-code guard, path %s' % (key, order))
                    assert 'FrozenCodeGuard' in r['finders'], (key, order, r['finders'])
                frozen = 'switch off with it first or second on the path'
            else:
                print('      (skipped: no sub-LST frozen-code guard at %s)' % FROZEN_GUARD)
            print('core.legacy.run %s (Python %s): switch off, %s and the local file read with the defaults, no mapping of '
                  'either and no /lustre mapping, with and without --parity (guard writeto patched once); plain child '
                  'imports no astropy; falsifier (no sitecustomize): switch on, local file mapped, Lustre file not opened; '
                  'frozen-code guard: %s' % (key, r['python'], LUSTRE_FITS, frozen))
    finally:
        sc.close()


if __name__ == '__main__':
    test_pattern_catches_the_forms()
    test_import_turns_memmap_off()
    test_no_memory_map_requested_in_source()
    test_every_fits_entry_point_turns_memmap_off()
    test_inplace_child_switch_off_lazily()
    test_inplace_children_through_core_legacy_run()
    print('PASS: no memory maps')
