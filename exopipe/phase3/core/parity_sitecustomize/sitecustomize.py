# exopipe3 PARITY WRITE GUARD for every Python process of a parity run (DESIGN 11.1; modelled on the sub-LST frozen-code
# guard, exopipe_ops/sublst_frozen_guard/sitecustomize.py). `exopipe3 <step> --parity` puts this directory first on
# PYTHONPATH and sets EXOPIPE3_PARITY_ROOT (the allowed roots, os.pathsep-separated), so every Python subprocess it starts -
# legacy scripts included - imports this file at start-up and installs core/guard.py: a write outside the parity roots
# raises a PermissionError naming the path instead of touching a production tree. An INJECTED run (core/injected.py) exports
# the same variable over its scratch roots; with EXOPIPE3_INJECTED set the guard is labelled INJECTED-RUN GUARD and raises
# InjectedWriteRefused. With EXOPIPE3_PARITY_ROOT unset no write guard is installed; the blocks below (import wraps, GITTAG
# policy, the FITS no-memory-map default and the refusal of a memory map of a Lustre file, in every process) still act, and
# the file chains to the next sitecustomize on the path. Python 3.6+.
import os
import sys

# ONCE PER PROCESS. Another sitecustomize that chains "to any other sitecustomize on the path" finds THIS file again and
# executes it again, and this file chains back to it, without end: the sub-LST frozen-code guard does so (2026-11-14,
# `pages --tier sublst` under a frozen tree: both hooks were installed dozens of times and every later import ended in a
# RecursionError). The second execution stops here; the chaining file catches the exception and goes on.
if getattr(sys, '_exopipe3_sitecustomize_loaded', False):
    raise ImportError('exopipe3 sitecustomize: already loaded in this process (chained back from another sitecustomize)')
sys._exopipe3_sitecustomize_loaded = True

# SAFE COPIES (as exopipe/__init__.py installs in every process that imports exopipe; the in-place children import none):
# shutil copies with ordinary reads and writes (its sendfile copies silently truncated Lustre-to-Lustre copies on lwacalim07,
# 2026-11-20) and a copy shorter than its source is refused - exopipe/_safe_copy.py, the one source, loaded by path. First in
# this file, so no later block's failure can skip it; its own failure is reported on stderr and the blocks below still run.
try:
    import importlib.util as _ilu_sc
    _sc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '_safe_copy.py')
    _sc_spec = _ilu_sc.spec_from_file_location('exopipe3_safe_copy', _sc_path)
    _sc = _ilu_sc.module_from_spec(_sc_spec)
    _sc_spec.loader.exec_module(_sc)
    _sc.install()
    del _ilu_sc, _sc_path, _sc_spec, _sc
except Exception as _sc_exc:                                     # noqa: BLE001 - reported, the guard below must still install
    sys.stderr.write(f'SAFE-COPY: not installed in this process: {_sc_exc!r}\n')

_ROOTS = os.environ.get('EXOPIPE3_PARITY_ROOT', '')
_HERE = os.path.dirname(os.path.abspath(__file__))

# CODE FINGERPRINT of a legacy child (core/legacy.run): with EXOPIPE3_FINGERPRINT_OUT set, write at exit a JSON
# {module file: sha1[:12]} of every imported source file under the legacy roots (EXOPIPE3_FINGERPRINT_ROOTS,
# os.pathsep-separated) or in an 'exopipe' package, so the parent's run record names the code that produced its products.
# One file PER PROCESS (<path>.<pid>; legacy.run merges them) and none when this process imported no such file: a
# descendant that outlives the child - a multiprocessing forkserver exits after its parent and runs atexit - wrote '{}' over
# the child's fingerprint (a race legacy.run usually won; 2026-10-14, phase3_snapshot_refastro's forkserver pool).
# Each module's source is hashed when it is loaded to run, not at exit (2026-10-15 reviews: a file edited while the child ran
# was recorded as the code that ran): Python 3.8+ by an audit hook on the 'exec' of every module body (its import, a reload),
# Python 3.6 / 3.7 by wrapping the source loader's get_code (imports and `python -m` both call it); the main script before it
# runs (sys.argv, or /proc/self/cmdline where sys.argv is not set yet). A file executed in more than one version is recorded
# 'mixed:<h1>,<h2>' - a value no file matches, so --resume re-runs the unit. This block runs first: everything loaded after
# it - the parity guard below included - is hashed when loaded.
_FP_OUT = os.environ.get('EXOPIPE3_FINGERPRINT_OUT', '')
_FP_RAN = {}
if _FP_OUT:
    import atexit as _atexit
    import hashlib as _hashlib

    def _fp_sha(f):
        with open(f, 'rb') as fo:
            return _hashlib.sha1(fo.read()).hexdigest()[:12]

    def _fp_note(f):
        try:
            seen = _FP_RAN.setdefault(os.path.abspath(f), [])
            h = _fp_sha(f)
            if h not in seen:
                seen.append(h)
        except Exception:                      # never break the load it observes
            pass

    if hasattr(sys, 'addaudithook'):
        def _fp_audit(event, args):
            if event != 'exec':
                return
            try:
                co = args[0]
                f = getattr(co, 'co_filename', None)
                if f and f.endswith('.py') and getattr(co, 'co_name', None) == '<module>':
                    _fp_note(f)
            except Exception:                  # an audit hook must never break the event it observes
                pass
        sys.addaudithook(_fp_audit)
    else:
        import importlib.machinery as _imach_fp
        _fp_get_code = _imach_fp.SourceFileLoader.get_code

        def _fp_get_code_hashed(self, fullname, _orig=_fp_get_code):
            p = getattr(self, 'path', None)
            if p and str(p).endswith('.py'):
                _fp_note(p)
            return _orig(self, fullname)
        _imach_fp.SourceFileLoader.get_code = _fp_get_code_hashed

    def _fp_main_script():
        argv = getattr(sys, 'argv', None)
        if argv:
            return argv[0] if argv[0] not in ('-c', '-m') else None
        try:                                   # sys.argv is not set yet while site runs on Python 3.6 / 3.7
            a = [x.decode('utf-8', 'replace') for x in open('/proc/self/cmdline', 'rb').read().split(b'\0') if x]
        except OSError:
            return None
        i = 1
        while i < len(a):
            if a[i] in ('-c', '-m'):
                return None
            if a[i] in ('-W', '-X', '-Q'):
                i += 2
                continue
            if a[i].startswith('-'):
                i += 1
                continue
            return a[i]
        return None
    try:
        _m0 = _fp_main_script()
        if _m0 and os.path.isfile(_m0):
            _fp_note(_m0)
    except Exception:
        pass

    def _fp_value(f):
        hs = _FP_RAN.get(f)
        if not hs:
            return _fp_sha(f)                  # loaded before this hook (none in practice: site runs first)
        return hs[0] if len(hs) == 1 else 'mixed:' + ','.join(hs)

    def _dump_fp():
        import json
        roots = tuple(r.rstrip('/') + '/' for r in os.environ.get('EXOPIPE3_FINGERPRINT_ROOTS', '').split(os.pathsep) if r)
        out = {}
        for name, mod in list(sys.modules.items()):
            f = getattr(mod, '__file__', None)
            if not f or not f.endswith('.py'):
                continue
            f = os.path.abspath(f)
            if name.startswith('exopipe') or (roots and f.startswith(roots)):
                try:
                    out[f] = _fp_value(f)
                except OSError:
                    pass
        main = getattr(sys.modules.get('__main__'), '__file__', None) or _fp_main_script()
        if main and os.path.isfile(main):
            try:
                out[os.path.abspath(main)] = _fp_value(os.path.abspath(main))
            except OSError:
                pass
        if not out:
            return
        mine = f'{_FP_OUT}.{os.getpid()}'
        try:
            with open(mine + '.tmp', 'w') as fo:
                json.dump(out, fo, indent=0, sort_keys=True)
            os.replace(mine + '.tmp', mine)
        except OSError:
            pass
    _atexit.register(_dump_fp)

if _ROOTS:
    import importlib.util as _ilu
    _gpath = os.path.join(os.path.dirname(_HERE), 'guard.py')
    _spec = _ilu.spec_from_file_location('exopipe.phase3.core.guard', _gpath)
    _guard = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_guard)
    sys.modules.setdefault('exopipe.phase3.core.guard', _guard)
    _guard.install([r for r in _ROOTS.split(os.pathsep) if r], log=os.environ.get('EXOPIPE3_PARITY_LOG') or None)

# NO PATCH PATH in a child either (B-01 b, Gregg 2026-10-13; core/legacy.PATCH_NAME, kept equal by a test): importing a
# Run_P3S4 -> Run_P3S5 patch module (p3s5_patch_cell, p3s5_repatch_epochgate, any p3s5_*patch*) raises ImportError. No
# production module imports one (2026-10-15); the patch scripts are refused by name before any child starts.
import re as _re_patch
_PATCH_NAME = _re_patch.compile(r'^p3s5_(re)?patch')


class _PatchImportGuard(object):
    def find_spec(self, name, path=None, target=None):
        if _PATCH_NAME.match(name.rsplit('.', 1)[-1]):
            raise ImportError(f'{name} is a Run_P3S4 -> Run_P3S5 patch path - the new code has none (B-01 b)')
        return None


sys.meta_path.insert(0, _PatchImportGuard())

# B-30 FALLBACK STATE of a legacy child (core.legacy.run, core/fallback.py): with EXOPIPE3_FALLBACK_OUT set, write at exit
# the counts and refusals of the fallback sites the child reached - only when it imported core/fallback (a branch module;
# production modules never do), so the parent's unit records them.
_FB_OUT = os.environ.get('EXOPIPE3_FALLBACK_OUT', '')
if _FB_OUT:
    import atexit as _atexit_fb

    def _dump_fb():
        m = sys.modules.get('exopipe.phase3.core.fallback')
        if m is not None:
            try:
                m.dump(_FB_OUT)
            except OSError:
                pass
    _atexit_fb.register(_dump_fb)

# IMPORT WRAPS (core/wrap.py; B-32 first): with EXOPIPE3_WRAPS naming modules ('{}' = none), each named PRODUCTION module
# loads as always, then its branch patch file loads BY PATH (it imports nothing from the branch: a child's `exopipe` can be the
# production package), is registered as sys.modules['exopipe3_wrap.<module>'] (the code fingerprint above records it), checks
# its pin against the production source and applies with the config values the parent passed.
def _fallback_module():
    """core/fallback.py, loaded by path as exopipe.phase3.core.fallback (the child's `exopipe` may be production's) - ONE
    module for the import wraps and the GITTAG check below, so the exit dump holds every site event of the child."""
    m = sys.modules.get('exopipe.phase3.core.fallback')
    if m is None:
        import importlib.util as _iu
        _fp = os.path.join(os.path.dirname(_HERE), 'fallback.py')
        _s = _iu.spec_from_file_location('exopipe.phase3.core.fallback', _fp)
        m = _iu.module_from_spec(_s)
        sys.modules['exopipe.phase3.core.fallback'] = m
        _s.loader.exec_module(m)
    return m


_WRAPS = os.environ.get('EXOPIPE3_WRAPS', '')
_WR = {}
if _WRAPS:
    import json as _json_w
    _WR = _json_w.loads(_WRAPS)
if _WR:
    import importlib.abc as _iabc_w
    import importlib.machinery as _imach_w
    import importlib.util as _iutil_w

    class _WrapFinder(_iabc_w.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in _WR:
                return None
            spec = _imach_w.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None:
                return None
            _orig = spec.loader.exec_module

            def _exec(module, _orig=_orig, _name=name):
                _orig(module)
                w = _WR[_name]
                ps = _iutil_w.spec_from_file_location(f'exopipe3_wrap.{_name}', w['patch'])
                pm = _iutil_w.module_from_spec(ps)
                ps.loader.exec_module(pm)
                sys.modules[f'exopipe3_wrap.{_name}'] = pm      # named in the child's code fingerprint (review MINOR 5)
                pm.apply(module, w['cfg'], _fallback_module())
            spec.loader.exec_module = _exec
            return spec
    sys.meta_path.insert(0, _WrapFinder())

# B-30 GITTAG_FALLBACK IN A CHILD (core/record.check_code_identity checks the step's own process at its start). Every process
# that imports exopipe.common.phase2_config - a legacy script, a spawned worker - resolves PIPELINE_GITTAG again with
# `git rev-parse` (10 s timeout) and keeps production's static 'clv15' on ANY failure, silently: git refusing a tree another
# user owns ('dubious ownership'), a stall under Lustre load. With the parent's fallback policies in the environment (every
# core.legacy.run child), the module is checked right after it loads, with the rule of record.code_identity_problems: the
# value is a commit (7-40 hex digits) AND the tree it was resolved in (the directory holding exopipe/common/) is the top of
# its own checkout - a copy inside another repository would stamp that repository's commit. Anything else reaches the site
# fallbacks.GITTAG_FALLBACK under the parent's policy: 'refuse' (the ruled value) raises FallbackRefused at that import,
# recorded for the parent (EXOPIPE3_FALLBACK_OUT: the unit refuses even when the child swallows the error); 'fallback'
# (production's behavior, the parity file) counts it - once per process that loads the module - and the module keeps its
# value. The resolution itself is production's phase2_config (byte-identical in the branch; it works under Python 3.6
# since production exopipe 4c7098c).
_GT_MOD = 'exopipe.common.phase2_config'
if os.environ.get('EXOPIPE3_FALLBACK_POLICY'):
    import importlib.abc as _iabc_g
    import importlib.util as _iutil_g
    import re as _re_g
    import subprocess as _sp_g

    def _gt_problem(module):
        """'' when the loaded module's PIPELINE_GITTAG is its own code tree's commit, else why not."""
        g, pid = str(getattr(module, 'PIPELINE_GITTAG', '')), os.getpid()
        if not _re_g.match(r'^[0-9a-f]{7,40}$', g):
            return (f'GITTAG would be {g!r} in child process {pid} (git rev-parse of its code tree failed or timed '
                    f'out)')
        f = getattr(module, '__file__', None)
        if not f:
            return f'GITTAG {g} in child process {pid} not checked: the module has no file, so no code tree'
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(f))))
        try:                                    # Python 3.6 too: no capture_output / text
            r = _sp_g.run(['git', '-C', root, 'rev-parse', '--show-toplevel'], stdout=_sp_g.PIPE, stderr=_sp_g.PIPE,
                          universal_newlines=True, timeout=10)
        except (OSError, _sp_g.SubprocessError) as e:
            return f'GITTAG {g} in child process {pid} not checked against its code tree {root}: git failed ({e})'
        top = r.stdout.strip()
        if r.returncode != 0 or not top:
            return (f'GITTAG {g} in child process {pid} not checked against its code tree {root}: git rev-parse '
                    f'--show-toplevel exit {r.returncode}')
        if os.path.realpath(top) != os.path.realpath(root):
            return (f'GITTAG {g} in child process {pid} is the commit of the checkout {top}, not of its code tree {root} '
                    f'(a copy inside another repository)')
        return ''

    def _gt_check(module):
        why = _gt_problem(module)
        if why:
            _fallback_module().site('fallbacks.GITTAG_FALLBACK', f'{getattr(module, "__file__", _GT_MOD)}: {why}, so its '
                                    f'products would carry no commit of their own code')

    class _GittagFinder(_iabc_g.MetaPathFinder):
        _busy = False

        def find_spec(self, name, path, target=None):
            # re-entered while asking the finders after it: a finder that asks EVERY finder (the sub-LST frozen-code guard:
            # when its sitecustomize is first on the path it chains here, and this finder then sits AHEAD of it) calls back
            # here while this finder asks it - without this guard the two call each other without end (RecursionError)
            if name != _GT_MOD or self._busy:
                return None
            later = sys.meta_path[sys.meta_path.index(self) + 1:] if self in sys.meta_path else []
            spec = None
            self._busy = True
            try:
                for f in later:                 # the spec the import would get without this finder (an import wrap's too)
                    fs = getattr(f, 'find_spec', None)
                    if fs is not None:
                        spec = fs(name, path, target)
                    elif hasattr(f, 'find_module'):     # a finder without find_spec: the spec the import system builds
                        ld = f.find_module(name, path)
                        spec = _iutil_g.spec_from_loader(name, ld) if ld is not None else None
                    if spec is not None:
                        break
            finally:
                self._busy = False
            if spec is None or spec.loader is None:
                return spec
            ld = spec.loader
            # the check runs only for this module even when a finder hands out one loader for many modules
            if hasattr(ld, 'exec_module'):
                _orig = ld.exec_module

                def _exec(module, _orig=_orig):
                    _orig(module)
                    if getattr(module, '__name__', None) == _GT_MOD:
                        _gt_check(module)
                ld.exec_module = _exec
            elif hasattr(ld, 'load_module'):    # a loader without exec_module
                _orig_l = ld.load_module

                def _load(fullname, _orig_l=_orig_l):
                    m = _orig_l(fullname)
                    if fullname == _GT_MOD:
                        _gt_check(m)
                    return m
                ld.load_module = _load
            return spec
    sys.meta_path.insert(0, _GittagFinder())

# NO FITS MEMORY MAP BY DEFAULT (Gregg 2026-11-17, R-GREGG-20261117-NO-MMAP-OF-LUSTRE-FILES-READ-OR-WRITE: never memory-map a
# Lustre file, for reading or writing; the 11-09 and 11-16 lwacalim07 Lustre client crashes were on mapped pages). astropy's
# fits.open / getdata map the file unless fits.conf.use_memmap is False. exopipe/__init__.py turns it off in every process
# that imports exopipe; the production scripts run in place (core.legacy.run, serve's shell scripts) do not import it, so
# here, in every process that loads this file, the switch is set right after astropy.io.fits finishes loading - by a
# meta-path finder that wraps the module's loader. astropy is never imported here (a process that reads no FITS file pays one
# name comparison per import). Every astropy of the site interpreters reads the switch the same way (4.1 under py36, 5.2
# under default = py38, 6.1 under flow, 8.0 under tess: fitsopen takes memmap=None to mean the switch). An explicit
# memmap=True still maps (a map of a Lustre file is refused: the gate below); its owner converts it (tests/test_no_mmap.py).
# The finder asks only the finders AFTER it, with a re-entry flag (as _GittagFinder above): a finder that asks EVERY finder
# (the parity guard's astropy hook in core/guard.py, the sub-LST frozen-code guard) gets None from it while it is busy, or
# its wrapped spec; the loader the import uses is wrapped once. Idempotent: installed once per process; a module loaded
# before this file gets the switch at once.
_FITS_MOD = 'astropy.io.fits'


def _fits_no_mmap(module):
    module.conf.use_memmap = False


class _FitsNoMmapFinder(object):
    _busy = False

    def find_spec(self, name, path=None, target=None):
        if name != _FITS_MOD or self._busy:
            return None
        later = sys.meta_path[sys.meta_path.index(self) + 1:] if self in sys.meta_path else []
        spec = None
        self._busy = True
        try:
            for f in later:                     # the spec the import would get without this finder
                fs = getattr(f, 'find_spec', None)
                if fs is not None:
                    spec = fs(name, path, target)
                elif hasattr(f, 'find_module') and sys.version_info < (3, 12):  # a finder without find_spec, as the
                    import importlib.util as _iutil_m                             # import system uses it before 3.12
                    ld = f.find_module(name, path)
                    spec = _iutil_m.spec_from_loader(name, ld) if ld is not None else None
                if spec is not None:
                    break
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return spec
        ld = spec.loader
        # the switch is set only for this module even when a finder hands out one loader for many modules
        if hasattr(ld, 'exec_module'):
            _orig = ld.exec_module

            def _exec(module, _orig=_orig):
                _orig(module)
                if getattr(module, '__name__', None) == _FITS_MOD:
                    _fits_no_mmap(module)
            ld.exec_module = _exec
        elif hasattr(ld, 'load_module'):        # a loader without exec_module
            _orig_l = ld.load_module

            def _load(fullname, _orig_l=_orig_l):
                m = _orig_l(fullname)
                if fullname == _FITS_MOD:
                    _fits_no_mmap(m)
                return m
            ld.load_module = _load
        return spec


sys.meta_path.insert(0, _FitsNoMmapFinder())      # once: this file runs once per process (the check at its top)
if _FITS_MOD in sys.modules:
    _fits_no_mmap(sys.modules[_FITS_MOD])

# NO MEMORY MAP OF A LUSTRE FILE, refused (the same ruling; gates refuse, they do not warn). The switch above is a default
# only: an explicit memmap=True, a numpy memory map or a direct map still maps. Every map made in Python is made by the
# type mmap.mmap, looked up when the map is made - astropy's FITS file layer (4.1, 5.2, 6.1, 8.0), numpy.memmap behind
# np.load(mmap_mode=...) and numpy.lib.format.open_memmap (numpy 1.19, 1.24, 2.2, 2.5), pandas, PIL, scipy.io's netcdf (the
# installed sources, 2026-11-17) - so here mmap.mmap becomes a subclass (isinstance and type checks hold) whose constructor
# refuses a descriptor of a Lustre file: one on a mount of type lustre (/proc/self/mountinfo) or whose path
# (/proc/self/fd/<fd>) is under /lustre/. The refusal is NoLustreMapError, a PermissionError (errno EACCES; never ENOMEM,
# which astropy answers with a second map) whose text starts 'NO-LUSTRE-MAP:' and names the file and the ruling; it is also
# mmap.NoLustreMapError, so a pool worker's refusal unpickles in its parent. The decision reads /proc, never Lustre.
# Before it raises, each refusal writes one line, 'NO-LUSTRE-MAP: <real path> pid <pid> <ruling>', to stderr (descriptor 2:
# a redirect of sys.stderr does not hide it) and appends it to the file $EXOPIPE3_MAP_REFUSAL_LOG names, when set and only
# when that file exists (its collector made it; a descendant that outlives the collection never recreates it) - a failed
# append adds a 'NO-LUSTRE-MAP-LOG:' line on stderr: a caller that swallows the exception (`except Exception: return None`
# around a FITS read) cannot hide the refusal. The collectors (core.legacy.run, serve's run_shell, the step itself:
# core/fallback.py) record each line as a sticky refusal: the unit that was running refuses.
# A descriptor that cannot be judged - neither its path nor its mount readable in /proc - is refused too (never assumed
# local). Unchanged: anonymous maps (fileno -1), maps of node-local files (/fast, /dataNN, /dev/shm, /tmp), a FITS header
# read with a map asked for (it maps nothing). astropy and numpy are not imported here. NOT covered: a map made in C
# (casacore maps tables only when its tsm option says so; its default reads them - py36, py38 and flow, 2026-11-17), a
# process that does not load this file (a program not in Python, python -I or -S, a PYTHONPATH without this directory).
# KNOWN FALSE REFUSAL: a library that TRIES a map and reads without one when it fails still gets the line, and its unit
# refuses although the data was read without a map - PIL's ImageFile.load on a raw single-tile image opened by file name
# (BMP, PPM, uncompressed TIFF) on Lustre; PNG (the pipeline's image format) never tries a map. Open such an image through
# a file object (Image.open(open(path, 'rb'))), which PIL never maps.
import errno as _errno
import mmap as _mmap

_MMAP = _mmap.mmap
_LUSTRE_RULING_ID = 'R-GREGG-20261117-NO-MMAP-OF-LUSTRE-FILES-READ-OR-WRITE'
_LUSTRE_RULING = _LUSTRE_RULING_ID + ', Gregg 2026-11-17: "Never memory-map a Lustre file, for reading OR writing"'


class NoLustreMapError(PermissionError):
    """A memory map of a Lustre file, refused (errno EACCES); str() is the refusal text, 'NO-LUSTRE-MAP: ...'."""

    def __str__(self):
        return str(self.strerror)


def _lustre_mount(dev, _seen={}):              # _seen: the answer per device, kept for the life of the process
    """Is device ``dev`` a mount of type lustre in /proc/self/mountinfo? None when that cannot be read."""
    hit = _seen.get(dev)
    if hit is None:
        try:
            with open('/proc/self/mountinfo') as f:
                rows = [line.partition(' - ') for line in f]
        except OSError:
            return None                         # not kept: asked again at the next map
        hit = False
        for head, sep, tail in rows:
            fields = head.split()
            if sep and len(fields) > 2 and tail.split()[:1] == ['lustre']:
                major, _c, minor = fields[2].partition(':')
                if major.isdigit() and minor.isdigit() and os.makedev(int(major), int(minor)) == dev:
                    hit = True
        _seen[dev] = hit
    return hit


def _report_refusal(where):
    """The line of one refused map, on stderr and appended to $EXOPIPE3_MAP_REFUSAL_LOG (when set), written before the
    refusal is raised."""
    line = 'NO-LUSTRE-MAP: %s pid %d %s\n' % (where, os.getpid(), _LUSTRE_RULING_ID)
    try:
        data = os.fsencode(line)                # the path's own bytes (os.readlink decoded them)
    except UnicodeError:
        data = line.encode('ascii', 'backslashreplace')
    try:
        sys.stderr.flush()                      # what the process wrote to stderr before stays before the line
    except Exception:
        pass
    try:
        os.write(2, data)
    except OSError:
        pass
    log = os.environ.get('EXOPIPE3_MAP_REFUSAL_LOG')
    if log:
        try:
            fd = os.open(log, os.O_WRONLY | os.O_APPEND)    # never O_CREAT: the collector made it, and may have read it
            try:
                os.write(fd, data)
            finally:
                os.close(fd)
        except OSError as e:
            try:
                os.write(2, ('NO-LUSTRE-MAP-LOG: the line above was not appended to EXOPIPE3_MAP_REFUSAL_LOG=%s: %s\n'
                             % (log, e)).encode('ascii', 'backslashreplace'))
            except OSError:
                pass


class _NoLustreMap(_MMAP):
    """mmap.mmap in a process that loaded core/parity_sitecustomize/sitecustomize.py: a map of a Lustre file is refused with
    NoLustreMapError; every other map is made as by the builtin type."""
    __slots__ = ()
    lustre_prefixes = ('/lustre/',)             # a file is on Lustre by its path ...
    lustre_devices = None                       # ... or by its device (None: the mounts of type lustre). The tests point
                                                # both at a local stand-in.

    @staticmethod
    def lustre_refusal(fileno):
        """The NoLustreMapError a map of descriptor ``fileno`` gets, or None (anonymous, node-local, not a descriptor: the
        builtin type decides)."""
        try:
            fd = fileno.__index__()
        except Exception:
            return None
        if fd < 0:
            return None
        gate = _NoLustreMap
        try:
            path = os.readlink('/proc/self/fd/%d' % fd)
        except OSError:
            path = None
        hit = path is not None and path.startswith(tuple(gate.lustre_prefixes))
        undecided = False
        if not hit:
            try:
                dev = os.fstat(fd).st_dev
            except OSError:
                return None                     # not an open descriptor: the builtin type refuses it (EBADF)
            hit = dev in gate.lustre_devices if gate.lustre_devices is not None else _lustre_mount(dev)
            if hit is None:                     # the mounts cannot be read: judged by the path alone, refused without one
                hit = undecided = path is None
        if not hit:
            return None
        where = path or 'descriptor %d' % fd
        what = ('a memory map of %s, which cannot be judged (neither its path nor its mount can be read in /proc),' % where
                if undecided else 'a memory map of the Lustre file %s' % where)
        return NoLustreMapError(_errno.EACCES, 'NO-LUSTRE-MAP: %s is refused (%s); read it without a map (astropy '
                                'memmap=False, numpy without mmap_mode)' % (what, _LUSTRE_RULING), where)

    def __new__(cls, *args, **kwargs):
        refused = _NoLustreMap.lustre_refusal(args[0] if args else kwargs.get('fileno', -1))
        if refused is not None:
            _report_refusal(refused.filename)
            raise refused
        return _MMAP.__new__(cls, *args, **kwargs)


_NoLustreMap.__name__ = _NoLustreMap.__qualname__ = 'mmap'
_NoLustreMap.__module__ = NoLustreMapError.__module__ = 'mmap'
_mmap.mmap = _NoLustreMap                         # once: this file runs once per process (the check at its top)
_mmap.NoLustreMapError = NoLustreMapError

# chain to any other sitecustomize on the path (the environment's own, if it has one) - never back to the sitecustomize
# Python imported at start-up when that is another file which chained here (the sub-LST frozen-code guard first on the
# path): running it again would install its import hook a second time, and two such hooks call each other without end
try:
    import importlib.machinery as _m
    _skip = {_HERE}
    _top = getattr(sys.modules.get('sitecustomize'), '__file__', None)
    if _top and os.path.abspath(_top) != os.path.abspath(__file__):
        _skip.add(os.path.dirname(os.path.abspath(_top)))
    _spec2 = _m.PathFinder.find_spec('sitecustomize', [p for p in sys.path if p and os.path.abspath(p) not in _skip])
    if _spec2 is not None and _spec2.origin and os.path.abspath(_spec2.origin) != os.path.abspath(__file__):
        with open(_spec2.origin) as _f:
            _code = compile(_f.read(), _spec2.origin, 'exec')
        exec(_code, {'__name__': 'sitecustomize', '__file__': _spec2.origin})
except SystemExit:
    raise
except Exception:
    pass
