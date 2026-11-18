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
5. THE NO-LUSTRE-MAP GATE: the same sitecustomize replaces mmap.mmap - the type every map made in Python goes through
   (astropy's FITS file layer, numpy.memmap, np.load(mmap_mode=...), numpy.lib.format.open_memmap) - by a subclass that
   refuses a descriptor of a Lustre file with NoLustreMapError (a PermissionError, errno EACCES, text 'NO-LUSTRE-MAP: ...'
   naming the file and the ruling) and, before raising, writes one line, 'NO-LUSTRE-MAP: <real path> pid <pid> <ruling>', to
   stderr and to the file $EXOPIPE3_MAP_REFUSAL_LOG names (so a caller that swallows the exception cannot hide the refusal).
   NO TEST ATTEMPTS A MAP OF A REAL LUSTRE FILE, not even one expected to be refused (a gate
   failure would then be the forbidden act). The whole chain is proven on a LOCAL STAND-IN: the child points the gate's
   protected path prefix, then its protected device, at a scratch directory; every map call on a file there is refused and
   leaves no map in /proc/self/maps, the same calls on a file outside it map (seen in /proc/self/maps), anonymous maps work,
   a FITS header read that asks for a map works (it maps nothing), astropy and numpy are not imported at start-up. The
   child's refusal lines, on stderr and in the log, are exactly one per refused map in the order of the refusals - the one
   a caller swallows (`except Exception: return None` around a FITS read, the production pattern) included - and none for a
   map that is made; an unwritable log path changes no refusal and adds a 'NO-LUSTRE-MAP-LOG:' line per refusal. On the
   real Lustre file only the gate's decision is asked (refused, by its path and by its device; no refusal line) and the
   default and memmap=False reads read it with no /lustre mapping. FALSIFIER: the stand-in child with the gate taken out
   (mmap.mmap back to the builtin type) maps every stand-in file, writes no refusal line and fails the stand-in check.
   The gate appends only to a log that exists (its collector made it): a collected (removed) log is never recreated, and
   the refusal then adds a 'NO-LUSTRE-MAP-LOG:' line. A descriptor that cannot be judged (neither its path nor its mount
   readable in /proc) is refused; one judged by either test alone is decided by it.
   Through core.legacy.run the child writes its lines to a file of that call: any line refuses the unit even when the child
   caught every refusal and exited 0 (res.refused, res.map_refusals - exactly the child's lines, in order - the reason naming
   the first refused map). On the cluster the children run through core.legacy.run under py38 and py36 with and without
   --parity, and directly under flow (3.10) and the TESS interpreters (3.12); elsewhere (CI) with this interpreter,
   stand-in only.
6. A REFUSED MAP IS STICKY (core/fallback.py MAP_KEY): every collector records each line as a refusal of the process's
   fallback state, so the unit that was running refuses whatever the caller does with the result - core.legacy.run (its
   call's file, and the gate's lines in the call's own output: a descendant that dropped the variable still wrote one to
   stderr; with and without a log), serve's run_shell, and the step's own collector for plain subprocesses (export_env);
   a refusal outside every unit refuses the run. The unit record lists the lines in 'map_refusals', the run summary in
   'map_refusals_outside_units'. core.legacy.run leaves no file of its call behind, also when the call raises; an earlier
   refusal reason is kept. Long-lived servers (serve's server_env, jupiter serve) get no collector. The callers that judge a
   result themselves (sub-LST judge_own_lines, the curves classifier) keep a refused map refused. FALSIFIERS in the test:
   without the output scan a dropped variable's refusal is lost; a plain subprocess without the step's collector leaves
   the unit done; without map_refusals the two callers turn the call into an error / a done table.

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
        if os.path.realpath(p).startswith('/lustre/') and fits.conf.use_memmap is not False:
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
        # the falsifier maps the local file on purpose: refuse a scratch directory whose REAL path is on Lustre (a TMPDIR
        # that reaches Lustre through a link or a bind mount included)
        if os.path.realpath(self.d).startswith('/lustre/'):
            shutil.rmtree(self.d, ignore_errors=True)
            raise RuntimeError('test_no_mmap: scratch %s is on Lustre (real path %s): set TMPDIR to node-local disk'
                               % (self.d, os.path.realpath(self.d)))
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
    # outside every root the guard allows (the parity root, /tmp, $TMPDIR - the scratch is under $TMPDIR): the repository's
    # tests directory, under a name of this run only, removed whatever happens
    outside = os.path.join(REPO, 'tests', '_nommap_write_refused_%s.fits' % os.path.basename(sc.d))
    try:
        py = sys.executable
        _check_off(_direct(py, sc, 'fits', [sc.local], [BOOT]), [sc.local], 'sitecustomize')
        _check_plain(_direct(py, sc, 'plain', (), [BOOT]), 'sitecustomize, plain child')
        _check_falsifier(_direct(py, sc, 'fits', [sc.local]), sc.local, 'no sitecustomize')
        proot = os.path.join(sc.d, 'parity_root')
        os.makedirs(proot)
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
        if os.path.exists(outside):
            os.remove(outside)


def test_inplace_children_through_core_legacy_run():
    """Section 4 on the cluster: children of core.legacy.run under py38 ('default') and py36 read a small Lustre FITS with
    the defaults - no map; the same interpreter without the sitecustomize is the falsifier (the Lustre file is not opened
    there); the parity guard (legacy.run parity_root) and the sub-LST frozen-code guard (either first on the path) compose."""
    site = os.environ.get('EXOPIPE3_SITE') or SITE_FILE
    if not os.path.isfile(site):
        print('      (skipped: no site file %s on this node - the core.legacy.run children run on the cluster)' % site)
        return
    assert os.path.isfile(LUSTRE_FITS), ('the small Lustre FITS of this test is gone - name another', LUSTRE_FITS)
    from exopipe.phase3.core import legacy
    # core.legacy.run hands the shell's environment on: the children run without the shell's PYTHONPATH, EXOPIPE3_*
    # (EXOPIPE3_SITE kept) and SUBLST_* variables, as _env does, so an inherited EXOPIPE3_PARITY_ROOT cannot turn the case
    # without --parity into a parity run; all of it is restored at the end
    saved = {k: os.environ.pop(k) for k in list(os.environ)
             if k == 'PYTHONPATH' or (k.startswith('EXOPIPE3_') and k != 'EXOPIPE3_SITE') or k.startswith('SUBLST_')}
    site_set = 'EXOPIPE3_SITE' in os.environ
    os.environ.setdefault('EXOPIPE3_SITE', site)
    sc = None
    try:
        sc = _Scratch()
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
                else:
                    assert r['guard_writeto_patches'] == 0, (key, what, 'a parity guard without --parity', r)
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
        if sc is not None:
            sc.close()
        if not site_set:
            os.environ.pop('EXOPIPE3_SITE', None)
        os.environ.update(saved)


# ---- 5. the no-Lustre-map gate --------------------------------------------------------------------------------------------
RULING = 'R-GREGG-20261117-NO-MMAP-OF-LUSTRE-FILES-READ-OR-WRITE'
TESS_FAST = '/fast/gh/main/envs/tess/bin/python'        # the TESS env of lwacalim07 (the site's 'tess' is its NFS copy)
# every way of mapping the child tries; all but fits_header make a map (a header read that asks for a map maps nothing)
GATE_CASES = ('fits_header', 'fits_data', 'fits_getdata', 'fits_update', 'np_memmap', 'np_memmap_write', 'np_load',
              'open_memmap', 'mmap_read', 'mmap_keywords', 'mmap_write')
DEVICE_CASES = ('fits_data', 'np_memmap', 'mmap_read')

# The gate child (Python 3.6+; _GateScratch writes CASES and DEVICE_CASES above it): argv = result file, mode, ... 'standin'
# (standin dir, outside dir, /dev/shm dir or '-', gate '1' / '0'): the gate's prefix, then its device, pointed at the
# stand-in; every case on the files there and outside; '0' takes the gate out first (the falsifier). 'real' (Lustre FITS,
# local FITS): the gate's decision on descriptors only, then the Lustre file read with the defaults and with memmap=False -
# never a map of it.
GATE_CHILD = r'''
import json, os, sys
res, mode = sys.argv[1], sys.argv[2]
out = {'python': sys.version.split()[0],
       'heavy_at_start': sorted(m for m in sys.modules if m.split('.')[0] in ('astropy', 'numpy'))}
import mmap
G = mmap.mmap
out['gate'] = bool(hasattr(G, 'lustre_refusal') and G is not G.__mro__[1] and G.__mro__[1].__module__ == 'mmap')
out['pid'] = os.getpid()


def maps():
    rows = set()
    with open('/proc/self/maps') as f:
        for line in f:
            p = line.rstrip('\n').split(None, 5)
            if len(p) == 6:
                rows.add(p[5])
    return rows


def refusal(e):
    return {'type': type(e).__name__, 'oserror': isinstance(e, OSError), 'errno': getattr(e, 'errno', None), 'text': str(e)}


def case(name, d):
    import numpy as np
    from astropy.io import fits
    from numpy.lib import format as npformat
    fp, npy, raw = (os.path.realpath(os.path.join(d, b)) for b in ('img.fits', 'arr.npy', 'raw.bin'))
    target = {'np_memmap': raw, 'np_memmap_write': raw, 'np_load': npy, 'open_memmap': npy, 'mmap_read': raw,
              'mmap_keywords': raw, 'mmap_write': raw}.get(name, fp)
    r = {'ok': False, 'mapped_during': False, 'refusal': None}
    try:
        if name == 'fits_header':
            with fits.open(fp, memmap=True) as h:
                r['value'] = len(h[0].header)
                seen = maps()
        elif name in ('fits_data', 'fits_update'):
            with fits.open(fp, memmap=True, mode='update' if name == 'fits_update' else 'readonly') as h:
                a = h[0].data
                seen = maps()
                r['value'] = float(a.sum())
                del a
        elif name == 'fits_getdata':
            a = fits.getdata(fp, memmap=True)
            seen = maps()
            r['value'] = float(a.sum())
            del a
        elif name in ('np_memmap', 'np_memmap_write'):
            a = np.memmap(raw, dtype=np.uint8, mode='r' if name == 'np_memmap' else 'r+')
            seen = maps()
            r['value'] = int(a.sum())
            del a
        elif name in ('np_load', 'open_memmap'):
            a = np.load(npy, mmap_mode='r') if name == 'np_load' else npformat.open_memmap(npy, mode='r')
            seen = maps()
            r['value'] = float(a.sum())
            del a
        else:
            with open(raw, 'r+b' if name == 'mmap_write' else 'rb') as f:
                if name == 'mmap_keywords':
                    m = mmap.mmap(fileno=f.fileno(), length=0, access=mmap.ACCESS_READ)
                elif name == 'mmap_read':
                    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                else:
                    m = mmap.mmap(f.fileno(), 0)
                seen = maps()
                r['value'] = len(m)
                m.close()
        r['ok'] = True
        r['mapped_during'] = target in seen
    except Exception as e:
        r['refusal'] = refusal(e)
    r['mapped_after'] = target in maps()
    r['target'] = target
    return r


def master_context(p):
    """The production pattern (exopipe_ops source_localization.master_context): a FITS read whose failure becomes an empty
    field."""
    from astropy.io import fits
    try:
        with fits.open(p, memmap=True) as h:
            return float(h[0].data.sum())
    except Exception:
        return None


if mode == 'standin':
    standin, outside, shm, on = sys.argv[3:7]
    if on == '0' and out['gate']:
        mmap.mmap = G = G.__mro__[1]                     # the falsifier: the gate taken out
    out['gate_in_use'] = hasattr(mmap.mmap, 'lustre_refusal')
    if out['gate_in_use']:
        G.lustre_prefixes, G.lustre_devices = (os.path.realpath(standin) + os.sep,), frozenset()
    out['standin'] = dict((c, case(c, standin)) for c in CASES)
    out['outside'] = dict((c, case(c, outside)) for c in CASES)
    ctx = os.path.realpath(os.path.join(standin, 'ctx.fits'))
    out['swallowed'] = {'target': ctx, 'standin': master_context(ctx),
                        'outside': master_context(os.path.join(outside, 'ctx.fits')), 'mapped_after': ctx in maps()}
    if shm != '-':
        if out['gate_in_use']:
            G.lustre_prefixes, G.lustre_devices = (), frozenset([os.stat(shm).st_dev])
        out['device'] = dict((c, case(c, shm)) for c in DEVICE_CASES)
        out['device_outside'] = dict((c, case(c, outside)) for c in DEVICE_CASES)
    if out['gate_in_use']:
        G.lustre_prefixes, G.lustre_devices = ('/lustre/',), None
    m = mmap.mmap(-1, 1 << 16)
    m[:5] = b'abcde'
    out['anonymous'] = {'ok': m[:5] == b'abcde', 'isinstance': isinstance(m, mmap.mmap), 'type': type(m) is mmap.mmap}
    m.close()
elif mode == 'real':
    lustre, local = sys.argv[3:5]
    dec = {}
    fd = os.open(lustre, os.O_RDONLY)                    # a descriptor only: nothing here maps it
    try:
        for key, prefixes, devices in (('default', ('/lustre/',), None), ('by_path', ('/lustre/',), frozenset()),
                                       ('by_device', (), None)):
            G.lustre_prefixes, G.lustre_devices = prefixes, devices
            e = G.lustre_refusal(fd)
            dec[key] = refusal(e) if e is not None else None
        G.lustre_prefixes, G.lustre_devices = ('/lustre/',), None
    finally:
        os.close(fd)
    fd = os.open(local, os.O_RDONLY)
    try:
        e = G.lustre_refusal(fd)
        dec['local'] = refusal(e) if e is not None else None
    finally:
        os.close(fd)
    e = G.lustre_refusal(-1)
    dec['anonymous'] = refusal(e) if e is not None else None
    out['decision'] = dec
    out['lustre_path'] = os.path.realpath(lustre)
    from astropy.io import fits
    out['use_memmap'] = fits.conf.use_memmap
    out['sums'], out['lustre_maps'] = [], []
    if fits.conf.use_memmap is False:                  # never open the Lustre file with the switch on
        with fits.open(lustre) as h:
            out['sums'].append(float(h[0].data.astype('f8').sum()))
            out['lustre_maps'] += sorted(x for x in maps() if x.startswith('/lustre/'))
        with fits.open(lustre, memmap=False) as h:
            out['sums'].append(float(h[0].data.astype('f8').sum()))
            out['lustre_maps'] += sorted(x for x in maps() if x.startswith('/lustre/'))
        a = fits.getdata(lustre)
        out['sums'].append(float(a.astype('f8').sum()))
        out['lustre_maps'] += sorted(x for x in maps() if x.startswith('/lustre/'))
        del a
    out['lustre_maps'] += sorted(x for x in maps() if x.startswith('/lustre/'))
with open(res, 'w') as f:
    json.dump(out, f)
'''


class _GateScratch(_Scratch):
    """_Scratch plus the gate child and, in three places, a FITS image, a .npy array and raw bytes: the stand-in directory,
    a directory outside it (same disk) and a directory on /dev/shm (another device: the device stand-in)."""

    def __init__(self):
        import numpy as np
        from astropy.io import fits
        super(_GateScratch, self).__init__()
        self.gate_child = os.path.join(self.d, 'gate_child.py')
        with open(self.gate_child, 'w') as f:
            f.write('CASES, DEVICE_CASES = %r, %r\n' % (GATE_CASES, DEVICE_CASES) + GATE_CHILD)
        self.standin, self.outside = os.path.join(self.d, 'standin'), os.path.join(self.d, 'outside')
        self.shm = None
        if os.path.isdir('/dev/shm') and os.stat('/dev/shm').st_dev != os.stat(self.d).st_dev:
            self.shm = tempfile.mkdtemp(prefix='exopipe3_lmg_', dir='/dev/shm')
        for d in [self.standin, self.outside] + ([self.shm] if self.shm else []):
            if not os.path.isdir(d):
                os.makedirs(d)
            for b in ('img.fits', 'ctx.fits'):
                fits.writeto(os.path.join(d, b), np.arange(64, dtype=np.float32).reshape(8, 8))
            np.save(os.path.join(d, 'arr.npy'), np.arange(256, dtype=np.float32))
            with open(os.path.join(d, 'raw.bin'), 'wb') as f:
                f.write(bytes(bytearray(range(256))) * 32)

    def close(self):
        if self.shm:
            shutil.rmtree(self.shm, ignore_errors=True)
        super(_GateScratch, self).close()


def _gate_args(sc, mode, on='1'):
    if mode == 'standin':
        return [mode, sc.standin, sc.outside, sc.shm or '-', on]
    return [mode, LUSTRE_FITS, sc.local]


def _gate_direct(py, sc, mode, on='1', pythonpath=(BOOT,), **extra):
    res = sc.result_path()
    env = {'XDG_CONFIG_HOME': sc.xdg, 'EXOPIPE3_MAP_REFUSAL_LOG': res + '.refusals'}
    open(res + '.refusals', 'w').close()                 # the collector makes the log; the gate never creates it
    env.update(extra)
    p = subprocess.run([py, sc.gate_child, res] + _gate_args(sc, mode, on), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       stdin=subprocess.DEVNULL, env=_env(pythonpath, **env), timeout=600)
    assert p.returncode == 0 and os.path.isfile(res), (py, mode, p.returncode, p.stdout.decode('utf-8', 'replace')[-2000:])
    return _gate_result(res, p.stdout.decode('utf-8', 'replace'), env['EXOPIPE3_MAP_REFUSAL_LOG'])


def _gate_result(res, output, refusal_log):
    """The child's result, with its output (stdout and stderr) and the lines of its refusal log."""
    with open(res) as f:
        r = json.load(f)
    r['_output'], r['_refusal_log'] = output, []
    if os.path.isfile(refusal_log):
        with open(refusal_log) as f:
            r['_refusal_log'] = f.read().splitlines()
    return r


def _refusal_lines(text, prefix='NO-LUSTRE-MAP: '):
    return [line for line in text.splitlines() if line.startswith(prefix)]


def _check_refused(r, what):
    f = r['refusal']
    assert r['ok'] is False and f, (what, 'not refused', r)
    assert f['type'] == 'NoLustreMapError' and f['oserror'] and f['errno'] == 13, (what, 'not the gate refusal', f)
    assert f['text'].startswith('NO-LUSTRE-MAP:') and r['target'] in f['text'] and RULING in f['text'], (what, f['text'])
    assert not r['mapped_during'] and not r['mapped_after'], (what, 'a refused file is mapped', r)


def _check_mapped(r, what):
    assert r['ok'] is True and r['refusal'] is None, (what, 'a map outside the stand-in failed', r)
    assert r['mapped_during'] is True, (what, 'a map outside the stand-in was not seen in /proc/self/maps: the check is blind',
                                        r)


def _check_gate_cases(r, sc, what):
    for c in GATE_CASES:
        s, o = r['standin'][c], r['outside'][c]
        if c == 'fits_header':                           # exact: a header read maps nothing, so it is not refused
            assert s['ok'] and o['ok'] and not s['mapped_during'] and not o['mapped_during'], (what, c, s, o)
            continue
        _check_refused(s, (what, 'stand-in', c))
        _check_mapped(o, (what, 'outside', c))
    if sc.shm:
        for c in DEVICE_CASES:
            _check_refused(r['device'][c], (what, 'device stand-in', c))
            _check_mapped(r['device_outside'][c], (what, 'outside the device stand-in', c))


def _check_gate_lines(r, sc, what, logged=True):
    """One refusal line per refused map, in the order of the refusals - the one the caller swallowed included - on stderr
    and in $EXOPIPE3_MAP_REFUSAL_LOG; none for a map that is made. logged=False: the log path cannot be written, so no log
    and a NO-LUSTRE-MAP-LOG line per refusal. Returns the lines."""
    targets = ([r['standin'][c]['target'] for c in GATE_CASES if c != 'fits_header'] + [r['swallowed']['target']]
               + ([r['device'][c]['target'] for c in DEVICE_CASES] if sc.shm else []))
    want = ['NO-LUSTRE-MAP: %s pid %d %s' % (t, r['pid'], RULING) for t in targets]
    got = _refusal_lines(r['_output'])
    assert got == want, (what, 'the refusal lines on stderr', got, want)
    if logged:
        assert r['_refusal_log'] == want, (what, 'the refusal log', r['_refusal_log'], want)
    else:
        bad = _refusal_lines(r['_output'], 'NO-LUSTRE-MAP-LOG: ')
        assert r['_refusal_log'] == [] and len(bad) == len(want), (what, 'the unwritable refusal log', r['_refusal_log'], bad)
    return want


def _check_gate_standin(r, sc, what, logged=True):
    assert r['gate'] is True and r['gate_in_use'] is True, (what, 'mmap.mmap is not the gate', r['gate'])
    assert r['heavy_at_start'] == [], (what, 'astropy or numpy imported at start-up', r['heavy_at_start'])
    _check_gate_cases(r, sc, what)
    assert r['anonymous'] == {'ok': True, 'isinstance': True, 'type': True}, (what, 'anonymous map', r['anonymous'])
    s = r['swallowed']
    assert s['standin'] is None and s['outside'] == 2016.0 and not s['mapped_after'], (what, 'the swallowed refusal', s)
    return _check_gate_lines(r, sc, what, logged)


def _check_gate_falsifier(r, sc, what):
    """The stand-in child with the gate taken out maps every stand-in file, and the stand-in cases check fails on it.
    Returns the check's failure."""
    assert r['gate'] is True and r['gate_in_use'] is False, (what, 'the gate was not taken out', r)
    runs = [r['standin'][c] for c in GATE_CASES if c != 'fits_header'] + [r['device'][c] for c in DEVICE_CASES if sc.shm]
    assert all(x['ok'] is True and x['mapped_during'] is True for x in runs), (what, 'a stand-in file not mapped without '
                                                                                'the gate', runs)
    assert r['swallowed']['standin'] == 2016.0, (what, 'the swallowing reader got no value without the gate', r['swallowed'])
    assert _refusal_lines(r['_output'], 'NO-LUSTRE-MAP') == [] and r['_refusal_log'] == [], (
        what, 'a refusal line without the gate', r['_output'][-1000:], r['_refusal_log'])
    try:
        _check_gate_cases(r, sc, what)
    except AssertionError as e:
        return '%d of %d stand-in maps made; the check: AssertionError%s' % (len(runs), len(runs), str(e)[:220])
    raise AssertionError((what, 'the stand-in check passed without the gate - falsifier void'))


def _check_gate_real(r, what):
    assert r['gate'] is True and r['heavy_at_start'] == [], (what, r['gate'], r['heavy_at_start'])
    for key in ('default', 'by_path', 'by_device'):
        f = r['decision'][key]
        assert f and f['type'] == 'NoLustreMapError' and f['oserror'] and f['errno'] == 13, (what, key, f)
        assert f['text'].startswith('NO-LUSTRE-MAP:') and r['lustre_path'] in f['text'] and RULING in f['text'], (what, key, f)
    assert r['decision']['local'] is None and r['decision']['anonymous'] is None, (what, 'a local or anonymous map refused', r)
    assert r['use_memmap'] is False and len(r['sums']) == 3 and len(set(r['sums'])) == 1, (what, r['use_memmap'], r['sums'])
    assert r['lustre_maps'] == [], (what, 'a /lustre mapping', r['lustre_maps'])
    assert _refusal_lines(r['_output'], 'NO-LUSTRE-MAP') == [] and r['_refusal_log'] == [], (
        what, 'a refusal line where no map was attempted', r['_output'][-1000:], r['_refusal_log'])


def test_lustre_map_gate_standin():
    """Section 5 with this interpreter (data-free; CI): the stand-in chain and its falsifier."""
    sc = _GateScratch()
    try:
        r = _gate_direct(sys.executable, sc, 'standin')
        lines = _check_gate_standin(r, sc, 'gate, Python %s' % r['python'])
        bad = os.path.join(sc.d, 'no_such_dir', 'refusals.log')
        _check_gate_standin(_gate_direct(sys.executable, sc, 'standin', EXOPIPE3_MAP_REFUSAL_LOG=bad), sc,
                            'gate, unwritable refusal log', logged=False)
        gone = os.path.join(sc.d, 'collected.log')       # a log its collector already read and removed
        _check_gate_standin(_gate_direct(sys.executable, sc, 'standin', EXOPIPE3_MAP_REFUSAL_LOG=gone), sc,
                            'gate, collected refusal log', logged=False)
        assert not os.path.exists(gone), ('the gate recreated a collected refusal log', gone)
        f = _check_gate_falsifier(_gate_direct(sys.executable, sc, 'standin', on='0'), sc, 'no gate')
        print('no-Lustre-map gate, Python %s: on the stand-in %d map calls refused (NoLustreMapError, errno 13, '
              'NO-LUSTRE-MAP text) and none mapped, the same calls outside it mapped, %s; header read with a map asked for '
              'works; anonymous map works; falsifier (gate taken out): %s'
              % (r['python'], len(GATE_CASES) - 1, 'device stand-in refused' if sc.shm else 'no device stand-in (no '
                 '/dev/shm on another device)', f))
        print('no-Lustre-map gate, Python %s: %d refusal lines on stderr and in EXOPIPE3_MAP_REFUSAL_LOG, one per refused map '
              'in order, including the refusal a reader swallowed (it returned None, no map); none for the maps made '
              'outside the stand-in, none without the gate; unwritable log path, and a log already collected (removed; '
              'not recreated): the same refusals and lines plus one NO-LUSTRE-MAP-LOG line each; first line: %s'
              % (r['python'], len(lines), lines[0]))
    finally:
        sc.close()


def test_lustre_map_gate_children():
    """Section 5 on the cluster: core.legacy.run children under py38 ('default') and py36, with and without --parity, and
    direct children under flow and the TESS interpreters: the stand-in chain, the decision on the real Lustre file (no map
    attempted), the default reads of it with no /lustre mapping; the falsifier per interpreter."""
    site = os.environ.get('EXOPIPE3_SITE') or SITE_FILE
    if not os.path.isfile(site):
        print('      (skipped: no site file %s on this node - the gate children run on the cluster)' % site)
        return
    assert os.path.isfile(LUSTRE_FITS), ('the small Lustre FITS of this test is gone - name another', LUSTRE_FITS)
    from exopipe.phase3.core import fallback, legacy
    saved = {k: os.environ.pop(k) for k in list(os.environ)
             if k == 'PYTHONPATH' or (k.startswith('EXOPIPE3_') and k != 'EXOPIPE3_SITE') or k.startswith('SUBLST_')}
    site_set = 'EXOPIPE3_SITE' in os.environ
    os.environ.setdefault('EXOPIPE3_SITE', site)
    sc = None
    try:
        sc = _GateScratch()
        for key in ('default', 'py36'):
            for kw, what in (({}, 'core.legacy.run'), ({'parity_root': os.path.join(sc.d, 'gate_parity_' + key)},
                                                       'core.legacy.run --parity')):
                if kw:
                    os.makedirs(kw['parity_root'])
                for mode in ('standin', 'real'):
                    res = sc.result_path()
                    m0 = fallback.mark()
                    lr = legacy.run(sc.gate_child, [res] + _gate_args(sc, mode), python=key,
                                    env={'XDG_CONFIG_HOME': sc.xdg}, log=res + '.log', **kw)
                    taken = fallback.take(m0)[0]                  # what this process's running unit would take
                    assert lr.rc == 0 and os.path.isfile(res), (key, what, mode, lr.rc, lr.reason, lr.tail[-2000:])
                    with open(res + '.log') as f:
                        r = _gate_result(res, f.read(), res + '.no_such_log')
                    r['_refusal_log'] = list(lr.map_refusals)    # the lines core.legacy.run collected
                    assert r['python'].startswith('3.6' if key == 'py36' else '3.8'), (key, r['python'])
                    assert taken == [(fallback.MAP_KEY, x) for x in lr.map_refusals], (key, what, mode, 'state', taken)
                    if mode == 'standin':
                        nlines = len(_check_gate_standin(r, sc, (key, what)))   # the call collected every line, in order
                        # the child caught every refusal and exited 0: core.legacy.run refuses the unit all the same
                        assert (lr.refused and not lr.ok and len(lr.map_refusals) == nlines
                                and all(m.startswith('NO-LUSTRE-MAP: ') for m in lr.map_refusals)
                                and lr.reason.startswith(os.path.basename(sc.gate_child) + ': NO-LUSTRE-MAP: ')
                                and lr.reason.endswith(' - REFUSED')), (key, what, 'refused at exit 0', lr.refused,
                                                                        lr.reason, len(getattr(lr, 'map_refusals', ())), nlines)
                    else:
                        _check_gate_real(r, (key, what))
                        assert lr.ok and not lr.refused and lr.map_refusals == [], (key, what, 'real file', lr.reason)
            f = _check_gate_falsifier(_gate_direct(legacy.interpreter(key), sc, 'standin', on='0'), sc, key)
            frozen = 'not composed (no guard file)'
            if os.path.isfile(os.path.join(FROZEN_GUARD, 'sitecustomize.py')):
                froot = os.path.join(sc.d, 'gate_frozen_' + key)
                os.makedirs(froot)
                open(os.path.join(froot, 'MANIFEST.sha256'), 'w').close()
                for order in ((FROZEN_GUARD, BOOT), (BOOT, FROZEN_GUARD)):
                    _check_gate_standin(_gate_direct(legacy.interpreter(key), sc, 'standin', pythonpath=order,
                                                     SUBLST_CODE_ROOT=froot), sc, (key, 'frozen-code guard', order))
                frozen = 'the gate in place with it first or second on the path'
            print('no-Lustre-map gate, core.legacy.run %s (Python %s), with and without --parity: stand-in maps refused, '
                  'outside mapped, %d refusal lines in the child log and collected by the call (res.map_refusals and this '
                  'process\'s state, in order), the swallowed refusal included, the unit refused at exit 0; %s refused by '
                  'path and by device without a map, read with the defaults and '
                  'memmap=False, no /lustre mapping, no refusal line; frozen-code guard: %s; falsifier: %s'
                  % (key, r['python'], nlines, LUSTRE_FITS, frozen, f[:120]))
        pys = [('flow', legacy.interpreter('flow')), ('tess', legacy.interpreter('tess')), ('tess /fast', TESS_FAST)]
        for name, py in pys:
            if not os.path.isfile(py):
                print('      (skipped: no %s interpreter %s on this node)' % (name, py))
                continue
            r = _gate_direct(py, sc, 'standin')
            nlines = len(_check_gate_standin(r, sc, name))
            r = _gate_direct(py, sc, 'real')
            _check_gate_real(r, name)
            f = _check_gate_falsifier(_gate_direct(py, sc, 'standin', on='0'), sc, name)
            print('no-Lustre-map gate, %s %s (Python %s): stand-in maps refused, outside mapped, %d refusal lines on stderr '
                  'and in EXOPIPE3_MAP_REFUSAL_LOG, the swallowed refusal included; the Lustre file refused without a map and '
                  'read with no /lustre mapping, no refusal line; falsifier: %s' % (name, py, r['python'], nlines, f[:120]))
    finally:
        if sc is not None:
            sc.close()
        if not site_set:
            os.environ.pop('EXOPIPE3_SITE', None)
        os.environ.update(saved)


# ---- 6. a refused map is sticky ------------------------------------------------------------------------------------------
# The mini child (Python 3.6+; neither numpy nor astropy): argv = stand-in dir, action. It points the gate's prefix at the
# stand-in and maps raw.bin there once, swallowing the refusal ('one'); or a grandchild does that with the collector's
# variable dropped ('drop') or naming a path that cannot be written ('badlog'); or one refused map, then the script refuses
# on its own (a REFUSED line, exit 2: 'refused2') or fails (exit 1: 'error1').
MINI_CHILD = r'''
import mmap, os, subprocess, sys
standin, action = sys.argv[1], sys.argv[2]
if hasattr(mmap.mmap, 'lustre_refusal'):
    mmap.mmap.lustre_prefixes, mmap.mmap.lustre_devices = (os.path.realpath(standin) + os.sep,), frozenset()


def refuse_one():
    with open(os.path.join(standin, 'raw.bin'), 'rb') as f:
        try:
            mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ).close()
        except Exception:
            pass                                         # swallowed: the production pattern


def grandchild(log):
    env = dict(os.environ)
    if log is None:
        env.pop('EXOPIPE3_MAP_REFUSAL_LOG', None)
    else:
        env['EXOPIPE3_MAP_REFUSAL_LOG'] = log
    subprocess.check_call([sys.executable, os.path.abspath(__file__), standin, 'one'], env=env)


if action == 'one':
    refuse_one()
elif action == 'drop':
    grandchild(None)
elif action == 'badlog':
    grandchild(os.path.join(standin, 'no_such_dir', 'refusals.log'))
elif action == 'refused2':
    refuse_one()
    print('REFUSED: the script refuses on its own')
    sys.exit(2)
elif action == 'error1':
    refuse_one()
    sys.exit(1)
print('mini child done')
'''

# A step process (fresh interpreter): argv = repo, run dir, mini child, stand-in, BOOT, 'step' (export_env first, as
# cli.main) or 'plain' (the falsifier: no step collector), result file. A plain subprocess refuses one map inside a unit and
# one outside every unit, each swallowed (exit 0).
STEP_CHILD = r'''
import json, os, subprocess, sys
repo, d, mini, standin, boot, mode, res = sys.argv[1:8]
sys.path.insert(0, repo)
from exopipe.phase3.config import CFG
from exopipe.phase3.core import fallback, record
if mode == 'step':
    fallback.export_env()
CFG.authorize([], out=d)
out = os.path.join(d, 'a.csv')
with open(out, 'w') as f:
    f.write('a\n1\n')
run = record.Run('t6s', ['x'], d, CFG, config_sections=())
env = dict(os.environ, PYTHONPATH=boot)
quiet = dict(env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
with run.unit('plain') as u:
    subprocess.check_call([sys.executable, mini, standin, 'one'], **quiet)
    u.add_output(out, 'csv_table')
rec = record.load_record(d, 't6s', 'plain')
subprocess.check_call([sys.executable, mini, standin, 'one'], **quiet)
code = run.exit_code()
with open(run.write_summary()) as f:
    summ = json.load(f)
with open(res, 'w') as f:
    json.dump({'status': u.status, 'reason': u.reason, 'map_refusals': rec.get('map_refusals'),
               'fallback_refusals': rec.get('fallback_refusals'), 'exit_code': code,
               'outside': summ.get('map_refusals_outside_units'), 'step_file': fallback._MAP_STEP.get('path'),
               'env': os.environ.get(fallback.MAP_ENV)}, f)
'''

# The gate's decision when /proc cannot answer (argv = result file, local file, refusal log): os.readlink (the path) and the
# gate's _lustre_mount (the mounts; None = unreadable) replaced in turn; a map of the local file each time.
UNDECIDED_CHILD = r'''
import json, mmap, os, sys
res, local = sys.argv[1], sys.argv[2]
mods = [m for m in list(sys.modules.values()) if getattr(m, '_NoLustreMap', None) is mmap.mmap]
out = {'gate': len(mods) == 1}
if out['gate']:
    S = mods[0]
    real_readlink, real_mount = os.readlink, S._lustre_mount

    def no_readlink(*a, **k):
        raise OSError(13, 'Permission denied')

    for name, rl, lm in (('neither', no_readlink, lambda dev: None), ('mount_only', no_readlink, real_mount),
                         ('path_only', real_readlink, lambda dev: None), ('both', real_readlink, real_mount)):
        os.readlink, S._lustre_mount = rl, lm
        try:
            with open(local, 'rb') as f:
                try:
                    mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ).close()
                    out[name] = None
                except Exception as e:
                    out[name] = {'type': type(e).__name__, 'errno': getattr(e, 'errno', None), 'text': str(e),
                                 'fd': f.fileno(), 'pid': os.getpid()}
        finally:
            os.readlink, S._lustre_mount = real_readlink, real_mount
with open(res, 'w') as f:
    json.dump(out, f)
'''


class _Section6(object):
    """Section 6 in this process: os.environ without PYTHONPATH and the exopipe3 child variables (EXOPIPE3_SITE kept), as
    _env; core.legacy.run's interpreter = this one (CI has no site file); tempfile's directory = a private one, where each
    call makes its own files (no_leftovers counts them); CFG authorized for the scratch; all undone at the end."""

    def __enter__(self):
        from exopipe.phase3.config import CFG
        from exopipe.phase3.core import legacy
        self.saved = {k: os.environ.pop(k) for k in list(os.environ)
                      if k == 'PYTHONPATH' or (k.startswith('EXOPIPE3_') and k != 'EXOPIPE3_SITE') or k.startswith('SUBLST_')}
        self.legacy, self.interpreter, self.tempdir = legacy, legacy.interpreter, tempfile.tempdir
        self.cfg, self.auth = CFG, dict(CFG._auth)
        self.sc = _Scratch()
        self.d = self.sc.d
        self.tmp, self.standin = os.path.join(self.d, 'calls_tmp'), os.path.join(self.d, 'standin6')
        for x in (self.tmp, self.standin):
            os.makedirs(x)
        self.raw = os.path.realpath(os.path.join(self.standin, 'raw.bin'))
        with open(self.raw, 'wb') as f:
            f.write(bytes(bytearray(range(256))))
        self.mini = os.path.join(self.d, 'mini_child.py')
        with open(self.mini, 'w') as f:
            f.write(MINI_CHILD)
        self.out = os.path.join(self.d, 'a.csv')
        with open(self.out, 'w') as f:
            f.write('a\n1\n')
        legacy.interpreter = lambda name='default': sys.executable
        tempfile.tempdir = self.tmp
        CFG.authorize([], out=self.d)
        self.want = 'NO-LUSTRE-MAP: %s pid ' % self.raw            # the start of the gate's line for raw.bin
        return self

    def line_ok(self, ln):
        return ln.startswith(self.want) and ln.endswith(' ' + RULING)

    def no_leftovers(self, what):
        left = [x for x in os.listdir(self.tmp) if x.startswith(('exopipe3_maprefusal_', 'exopipe3_fp_', 'exopipe3_fb_'))]
        assert left == [], (what, 'files of a call left behind', left)

    def __exit__(self, *exc):
        self.legacy.interpreter, tempfile.tempdir = self.interpreter, self.tempdir
        self.cfg.reset()
        self.cfg._auth = self.auth
        os.environ.update(self.saved)
        self.sc.close()


def _collect_quiet(path):
    """map_collect without the sticky record (the code before PR 103's review: the lines came back in the result only)."""
    try:
        with open(path) as f:
            return [ln for ln in f.read().splitlines() if ln.strip()]
    finally:
        os.remove(path)


def test_map_collect_counts_each_refusal_once():
    """fallback.map_collect: the call file's lines, then the gate's lines found only in the call's output and not among
    `seen`, each refused map once; recorded under MAP_KEY in this thread's state; the file removed; other text ignored."""
    from exopipe.phase3.core import fallback
    sc = _Scratch()
    try:
        a, b, c = ('NO-LUSTRE-MAP: /x/%s pid 7 %s' % (n, RULING) for n in ('a.fits', 'b.fits', 'c.fits'))
        p = os.path.join(sc.d, 'call.log')
        with open(p, 'w') as f:
            f.write(a + '\n' + a + '\n')                 # one process refused the same file twice: two refusals
        output = ['other output', a, 'NO-LUSTRE-MAP-LOG: not appended', b, a, a + ' trailing', c + '\r\n',
                  'NO-LUSTRE-MAP: /x/d.fits pid x ' + RULING]
        m = fallback.mark()
        got = fallback.map_collect(p, output, seen=[c])
        assert got == [a, a, b], got
        assert fallback.since(m)[0] == [(fallback.MAP_KEY, x) for x in (a, a, b)], fallback.since(m)
        assert not os.path.exists(p), 'the call file was not removed'
        m = fallback.mark()
        assert fallback.map_collect(os.path.join(sc.d, 'never_made.log'), [b]) == [b], 'output only'
        assert fallback.since(m)[0] == [(fallback.MAP_KEY, b)], fallback.since(m)
        assert fallback.refusal_text(fallback.MAP_KEY, a) == a
        assert fallback.refusal_text('fallbacks.X', 'y') == 'B-30 fallback site fallbacks.X: y'
        print('map_collect: file lines + output-only gate lines, each refused map once (a line in both once, a seen line '
              'not again, non-gate text ignored), recorded under %s, file removed' % fallback.MAP_KEY)
    finally:
        sc.close()


def test_refused_map_is_sticky_through_core_legacy_run():
    """Section 6 through core.legacy.run (data-free; CI): the unit refuses when the caller drops the result; a descendant
    that dropped the variable or could not write the log is still counted (output scan); the reasons; no file of a call left
    behind (also when the call raises); a refusal outside every unit refuses the run. FALSIFIERS: no sticky record -> the
    unit is done; no output scan -> the dropped variable's refusal is lost."""
    from exopipe.phase3.core import fallback, legacy, record
    with _Section6() as s6:
        base = os.path.basename(s6.mini)
        run = record.Run('t6', ['x'], s6.d, s6.cfg, config_sections=())
        with run.unit('dropped') as u:                   # the caller drops the result (no unit=, no check)
            legacy.run(s6.mini, [s6.standin, 'one'], log=os.path.join(s6.d, 'c1.log'))
            u.add_output(s6.out, 'csv_table')
        rec = record.load_record(s6.d, 't6', 'dropped')
        assert u.status == 'refused' and s6.line_ok(u.reason.split(' - REFUSED')[0]), (u.status, u.reason)
        assert u.reason.endswith(' - REFUSED (caught before the unit; 1 refusal(s) in the unit)'), u.reason
        assert len(rec.get('map_refusals') or []) == 1 and s6.line_ok(rec['map_refusals'][0]), rec
        assert 'fallback_refusals' not in rec, rec
        s6.no_leftovers('dropped')
        real = fallback.map_collect
        fallback.map_collect = lambda path, output=(), seen=(): _collect_quiet(path)
        try:
            with run.unit('dropped_unrecorded') as u0:
                lr0 = legacy.run(s6.mini, [s6.standin, 'one'], log=os.path.join(s6.d, 'c0.log'))
                u0.add_output(s6.out, 'csv_table')
        finally:
            fallback.map_collect = real
        assert lr0.refused and u0.status == 'done', ('falsifier void: the unit refused without the sticky record',
                                                     lr0.refused, u0.status)
        for action, log in (('drop', 'c2.log'), ('badlog', 'c3.log'), ('drop', None)):
            with run.unit('%s_%s' % (action, log)) as u:
                lr = legacy.run(s6.mini, [s6.standin, action], log=os.path.join(s6.d, log) if log else None)
                u.add_output(s6.out, 'csv_table')
            assert lr.rc == 0 and lr.refused and len(lr.map_refusals) == 1 and s6.line_ok(lr.map_refusals[0]), (
                action, log, lr.rc, lr.refused, lr.map_refusals, lr.tail[-1500:])
            assert u.status == 'refused', (action, log, u.status, u.reason)
        s6.no_leftovers('drop / badlog')
        real_scan = legacy._map_lines_from
        legacy._map_lines_from = lambda src, start: []
        try:
            lrf = legacy.run(s6.mini, [s6.standin, 'drop'], log=os.path.join(s6.d, 'c4.log'))
        finally:
            legacy._map_lines_from = real_scan
        assert lrf.ok and not lrf.refused and lrf.map_refusals == [], ('falsifier void: counted without the output scan',
                                                                      lrf.reason)
        lr = legacy.run(s6.mini, [s6.standin, 'one'])
        assert lr.rc == 0 and lr.reason == '%s: %s - REFUSED' % (base, lr.map_refusals[0]), lr.reason
        lr = legacy.run(s6.mini, [s6.standin, 'refused2'])
        assert (lr.rc == 2 and lr.refused and lr.reason.startswith('%s exit 2 (REFUSED: ' % base)
                and lr.reason.endswith('; %s: %s - REFUSED' % (base, lr.map_refusals[0]))), (lr.rc, lr.reason)
        lr = legacy.run(s6.mini, [s6.standin, 'error1'])
        assert lr.rc == 1 and lr.refused and lr.reason == '%s exit 1; %s: %s - REFUSED' % (
            base, base, lr.map_refusals[0]), (lr.rc, lr.reason)
        s6.no_leftovers('reasons')
        try:
            legacy.run(s6.mini, [s6.standin, 'one'], cwd=os.path.join(s6.d, 'no_such_dir'))
        except OSError:
            pass
        else:
            raise AssertionError('a call with a working directory that does not exist did not raise')
        s6.no_leftovers('a call that raised')
        run_o = record.Run('t6o', ['x'], s6.d, s6.cfg, config_sections=())
        assert run_o.exit_code() != 2, run_o.fallback_outside()
        legacy.run(s6.mini, [s6.standin, 'one'])                     # outside every unit
        assert run_o.exit_code() == 2, ('a refused map outside every unit does not refuse the run', run_o.fallback_outside())
        with open(run_o.write_summary()) as f:
            summ = json.load(f)
        assert (len(summ.get('map_refusals_outside_units') or []) == 1 and s6.line_ok(summ['map_refusals_outside_units'][0])
                and 'fallback_refusals_outside_units' not in summ), summ
        print('core.legacy.run: a refused map refuses the unit when the caller drops the result (record map_refusals); a '
              'descendant without the variable or with an unwritable log still counted (with and without a log file); '
              'reasons at exit 0 / exit 2 + REFUSED / exit 1 name the map; no call file left (also when the call raises); '
              'outside every unit: the run refused, map_refusals_outside_units; falsifiers: unit done without the sticky '
              'record, refusal lost without the output scan')


def test_refused_map_is_sticky_in_a_step():
    """Section 6, the step's own collector (export_env, as cli.main): a plain subprocess's refused map refuses the unit, and
    one outside every unit refuses the run; the step's file is removed at exit. FALSIFIER: the same step without export_env
    leaves the unit done and the run unrefused."""
    sc = _Scratch()
    try:
        standin = os.path.join(sc.d, 'standin6')
        os.makedirs(standin)
        with open(os.path.join(standin, 'raw.bin'), 'wb') as f:
            f.write(bytes(bytearray(range(256))))
        mini, child = os.path.join(sc.d, 'mini_child.py'), os.path.join(sc.d, 'step_child.py')
        for p, text in ((mini, MINI_CHILD), (child, STEP_CHILD)):
            with open(p, 'w') as f:
                f.write(text)
        want = 'NO-LUSTRE-MAP: %s pid ' % os.path.realpath(os.path.join(standin, 'raw.bin'))
        got = {}
        for mode in ('step', 'plain'):
            res, d = sc.result_path(), os.path.join(sc.d, 'run_' + mode)
            os.makedirs(d)
            p = subprocess.run([sys.executable, child, REPO, d, mini, standin, BOOT, mode, res], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=_env(XDG_CONFIG_HOME=sc.xdg),
                               timeout=600)
            assert p.returncode == 0 and os.path.isfile(res), (mode, p.returncode, p.stdout.decode('utf-8', 'replace')[-3000:])
            with open(res) as f:
                got[mode] = json.load(f)
        r = got['step']
        assert r['status'] == 'refused' and r['reason'].startswith(want), r
        assert len(r['map_refusals'] or []) == 1 and r['map_refusals'][0].startswith(want) and not r['fallback_refusals'], r
        assert r['exit_code'] == 2 and len(r['outside'] or []) == 1 and r['outside'][0].startswith(want), r
        assert r['env'] == r['step_file'] and r['step_file'] and not os.path.exists(r['step_file']), (
            'the step file is not this process\'s variable, or was left behind', r)
        f = got['plain']
        assert f['status'] == 'done' and not f['map_refusals'] and f['exit_code'] == 0 and not f['outside'], (
            'falsifier void: refused without the step collector', f)
        print('step collector (export_env): a plain subprocess\'s refused map refuses the unit and, outside every unit, the '
              'run (exit 2); the step file removed at exit; falsifier (no export_env): unit done, run exit 0')
    finally:
        sc.close()


def test_refused_map_is_sticky_through_run_shell_and_servers_get_no_collector():
    """Section 6, serve: run_shell refuses the running unit when a script's Python child refused a map (result dropped), its
    call file removed; server_env and jupiter serve's start() give a long-lived server no collector variable. FALSIFIERS:
    run_shell without the sticky record leaves the unit done; the probe child reads the variable when it is passed."""
    import types
    from exopipe.phase3.core import fallback, record
    from exopipe.phase3.lib.serve import scripts
    from exopipe.phase3.lib.jupiter import serve as jserve
    with _Section6() as s6:
        sh = os.path.join(s6.d, 'mini.sh')
        with open(sh, 'w') as f:
            f.write('#!/bin/bash\n"$1" "$2" "$3" one\necho "shell done"\n')
        saved = scripts.path, scripts.ops_root
        scripts.path, scripts.ops_root = (lambda name: sh), (lambda: s6.d)
        try:
            run = record.Run('t6r', ['x'], s6.d, s6.cfg, config_sections=())
            with run.unit('shell') as u:
                rc, out = scripts.run_shell('snapshot_localize', [sys.executable, s6.mini, s6.standin])   # result dropped
                u.add_output(s6.out, 'csv_table')
            assert rc == 0 and 'shell done' in out and any(s6.line_ok(x) for x in out.splitlines()), (rc, out[-1500:])
            rec = record.load_record(s6.d, 't6r', 'shell')
            assert u.status == 'refused' and len(rec.get('map_refusals') or []) == 1, (u.status, u.reason, rec)
            s6.no_leftovers('run_shell')
            real = fallback.map_collect
            fallback.map_collect = lambda path, output=(), seen=(): _collect_quiet(path)
            try:
                with run.unit('shell_unrecorded') as u0:
                    scripts.run_shell('snapshot_localize', [sys.executable, s6.mini, s6.standin])
                    u0.add_output(s6.out, 'csv_table')
            finally:
                fallback.map_collect = real
            assert u0.status == 'done', ('falsifier void: run_shell refused the unit without the sticky record', u0.reason)
        finally:
            scripts.path, scripts.ops_root = saved
        outer = os.path.join(s6.d, 'outer.log')
        os.environ[fallback.MAP_ENV] = outer
        try:
            assert fallback.MAP_ENV not in scripts.server_env({}), 'server_env hands the collector variable to a server'
            probe = ('import os, sys\nwith open(sys.argv[1], "w") as f:\n    f.write(os.environ.get(sys.argv[2], "<unset>"))\n')
            seen = os.path.join(s6.d, 'probe_server.txt')
            spec = {'argv': [sys.executable, '-c', probe, seen, fallback.MAP_ENV], 'cwd': s6.d,
                    'log': os.path.join(s6.d, 'jupiter_logs', 'probe.log')}
            p = jserve.start(spec, types.SimpleNamespace(guard=()))
            assert p.wait(timeout=120) == 0, 'the probe server did not exit 0'
            with open(seen) as f:
                assert f.read() == '<unset>', 'jupiter serve start() hands the collector variable to a server'
            seen2 = os.path.join(s6.d, 'probe_plain.txt')
            subprocess.check_call([sys.executable, '-c', probe, seen2, fallback.MAP_ENV], env=dict(os.environ))
            with open(seen2) as f:
                assert f.read() == outer, 'falsifier void: the probe does not see the variable when it is passed'
        finally:
            os.environ.pop(fallback.MAP_ENV, None)
        print('run_shell: a refused map in the script\'s Python child refuses the unit (result dropped), call file removed; '
              'falsifier: unit done without the sticky record. server_env and jupiter serve start(): no collector variable '
              '(a probe started by start() reads it unset; the same probe run plainly reads it)')


def test_callers_keep_a_refused_map_refused():
    """The callers that judge a result themselves: sub-LST judge_own_lines leaves a refused map refused (FALSIFIER: without
    map_refusals it re-judges the call an error); the curves classifier makes it 'refused' (gate NO-LUSTRE-MAP) although the
    table exists at exit 0 (FALSIFIER: without map_refusals it is 'done')."""
    import types
    from exopipe.phase3.lib.frames import tier_sublst
    from exopipe.phase3.lib.curves import workers
    line = 'NO-LUSTRE-MAP: /x/a.fits pid 7 ' + RULING

    def res(maps, rc):
        return types.SimpleNamespace(rc=rc, refused=bool(maps) or rc == 2, reason='x.py exit %d; x.py: %s - REFUSED' % (rc, line),
                                     map_refusals=list(maps), fallback_refusals=[], tail='', wall_s=0.1, log=None)
    own = ['===== 2026-11-18T00:00:00Z py x.py', 'some output']
    r = res([line], 1)
    tier_sublst.judge_own_lines(r, own, 'x.py', (2,))
    assert r.refused is True and r.reason.endswith(line + ' - REFUSED'), (r.refused, r.reason)
    f = res([], 1)
    f.refused = True
    tier_sublst.judge_own_lines(f, own, 'x.py', (2,))
    assert f.refused is False and f.reason == 'x.py exit 1 - error', ('falsifier void', f.refused, f.reason)
    sc = _Scratch()
    try:
        table = os.path.join(sc.d, 'curve.csv')
        with open(table, 'w') as fh:
            fh.write('a\n1\n')
        c = workers.classify(res([line], 0), table, 'cell', 'sb', 'I')
        assert c['status'] == 'refused' and c['gate'] == 'NO-LUSTRE-MAP' and line in c['reason'], c
        c0 = workers.classify(res([], 0), table, 'cell', 'sb', 'I')
        assert c0['status'] == 'done', ('falsifier void', c0)
    finally:
        sc.close()
    print('callers: judge_own_lines keeps a refused map refused (without map_refusals: re-judged "exit 1 - error"); curves '
          'classify -> refused, gate NO-LUSTRE-MAP, at exit 0 with the table written (without: done)')


def test_undecidable_descriptor_is_refused():
    """The gate when /proc cannot answer: neither the path nor the mounts readable -> refused (a line naming the
    descriptor); either one readable -> decided by it (a local file maps)."""
    sc = _Scratch()
    try:
        child, res, log = os.path.join(sc.d, 'undecided_child.py'), sc.result_path(), os.path.join(sc.d, 'undecided.log')
        with open(child, 'w') as f:
            f.write(UNDECIDED_CHILD)
        open(log, 'w').close()
        p = subprocess.run([sys.executable, child, res, sc.local], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, env=_env((BOOT,), XDG_CONFIG_HOME=sc.xdg, EXOPIPE3_MAP_REFUSAL_LOG=log),
                           timeout=600)
        assert p.returncode == 0 and os.path.isfile(res), (p.returncode, p.stdout.decode('utf-8', 'replace')[-2000:])
        with open(res) as f:
            r = json.load(f)
        assert r['gate'] is True, r
        n = r['neither']
        assert n and n['type'] == 'NoLustreMapError' and n['errno'] == 13 and 'cannot be judged' in n['text'], r
        want = 'NO-LUSTRE-MAP: descriptor %d pid %d %s' % (n['fd'], n['pid'], RULING)
        with open(log) as f:
            assert f.read().splitlines() == [want], ('the refusal line', want)
        assert r['mount_only'] is None and r['path_only'] is None and r['both'] is None, r
        from exopipe.phase3.core import fallback
        assert fallback.map_lines([want]) == [want], 'the collectors do not recognize the line of an undecidable descriptor'
        print('undecidable descriptor (no path, no mounts): refused, line "%s"; path or mounts readable: decided by it (a '
              'local file maps)' % want[:60])
    finally:
        sc.close()


def test_map_ruling_ids_agree():
    """The collectors' ruling ID (core/fallback.py MAP_RULING_ID) is the gate's (sitecustomize _LUSTRE_RULING_ID)."""
    from exopipe.phase3.core import fallback
    with open(SITECUSTOMIZE) as f:
        tree = ast.parse(f.read())
    ids = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
           and any(getattr(t, 'id', None) == '_LUSTRE_RULING_ID' for t in n.targets)]
    assert ids == [fallback.MAP_RULING_ID] == [RULING], (ids, fallback.MAP_RULING_ID)


if __name__ == '__main__':
    test_pattern_catches_the_forms()
    test_import_turns_memmap_off()
    test_no_memory_map_requested_in_source()
    test_every_fits_entry_point_turns_memmap_off()
    test_inplace_child_switch_off_lazily()
    test_inplace_children_through_core_legacy_run()
    test_lustre_map_gate_standin()
    test_lustre_map_gate_children()
    test_map_collect_counts_each_refusal_once()
    test_refused_map_is_sticky_through_core_legacy_run()
    test_refused_map_is_sticky_in_a_step()
    test_refused_map_is_sticky_through_run_shell_and_servers_get_no_collector()
    test_callers_keep_a_refused_map_refused()
    test_undecidable_descriptor_is_refused()
    test_map_ruling_ids_agree()
    print('PASS: no memory maps')
