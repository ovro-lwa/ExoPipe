#!/usr/bin/env python3
"""Speed-ups of Phase 2 (exopipe/phase2: extractor_pb_75.py, cascade_dewarp.py, phase2_controller.py): each must change no
product (REVIEW.md item 8). What full LST hours prove on data is in exopipe_ops/refactor_20261012/plan/STAGE_MAP_20261111.md
(sections 17-19); these data-free tests guard the rule each speed-up rests on. The three modules need the cluster's packages
(PyBDSF, h5py) to import, so the functions are taken from their SOURCE and run here with numpy, pandas and astropy only:

  1. the VLSSr reference catalog parsed in one call (`parse_vlssr_text`) returns, bit for bit, what the original parser (one
     coordinate object per line, kept as `_parse_vlssr_text_per_line`) returns: positions in degrees, the stored coordinates in
     radians and as Cartesian vectors, fluxes and major axes; lines that are not sources are skipped as before; a line that
     astropy accepts outside the plain sexagesimal form, or a refusal of the single call, sends the WHOLE file through the
     original parser;
  2. the 10-minute stage reuses the cascade's catalog of a dewarped hour image only for exactly that file (path, size,
     modification time) under exactly the extraction settings of the first run; `_catalog_key` holds EVERY setting
     `extract_sources` and its helpers read (if a new setting appears there and not in the key, this test fails); what is kept
     and what is handed back are copies; an empty extraction is never kept; P2_REUSE_CASCADE_CATALOGS=0 switches the reuse off;
  3. the copy-in (`_copy_files`) copies every listed file with the same bytes whatever P2_COPY_WORKERS is, creates the
     destination files in the listed order before any parallel copy starts, runs the plain serial loop when the setting is
     unset, and refuses (after one retry) a copy whose size differs from its source's; a copy that does not complete leaves
     no destination file behind, and `copy_to_nvme` refuses a subband directory that holds an empty image (the trace of a
     killed parallel copy) instead of taking it as copied;
  4. the 10-minute interval pool keeps its size (8, or the CPU count minus 2) unless P2_TENMIN_WORKERS is set; the runtime
     settings are checked at the start of the run: a value that is not a whole number of at least 1 is refused, and so are
     more than 6 copy streams (the recorders share Lustre);
  5. the controller switches numpy's huge-page advice off before anything can import numpy, and a value already in the
     environment wins.
  6. the cascade's first extractions made in a pool up front, and its UP arm run in a child process beside the DOWN arm,
     leave the cascade the one it was: every subband is dewarped against the same reference with the same first catalog,
     the results keep their order (anchor, DOWN arm, UP arm), the same subbands reach the 10-minute stage and the same
     dewarped catalogs are kept for it; P2_CASCADE_WORKERS=1 and P2_CASCADE_ARMS_SIDE_BY_SIDE=0 give the code path as it
     was (no pool is made); an up-front catalog is used only for exactly that file under exactly those settings, an empty
     or failed one is never kept, and the warning lines of a kept one reach the hour's log; an up-front pool that stops,
     or a child that cannot be started, leaves the cascade to do that work itself, in order, as before, and the
     elevation-cut copies that stopped workers left are removed; a child that fails or dies after it started ends the
     cascade with that error (the arm is not run a second time over what the child left), and an error in the DOWN arm
     waits for the child before it goes on; the helper processes are forked and end with their parent. (Most of these
     tests replace the pools by an in-process stand-in; two run real processes; the products are proven on a full LST
     hour.)
  7. a rerun in the same work directory after a killed run: an elevation-cut copy a killed extraction left beside an image
     (<image>.elcut<N>[e<M>].fits) is never taken for an image by any image finder (every finder that excludes PyBDSF
     products excludes them too), and every <image>_dewarped.fits is written through a temporary name and renamed, so a
     write that stops leaves nothing under the final name for the rerun to skip as done.
Data-free. Run: python tests/test_phase2_speedups.py   (or under pytest)
"""
import ast
import contextlib
import glob
import io
import itertools
import os
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
import types
import warnings

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P2 = os.path.join(REPO, 'exopipe', 'phase2')
EXTRACTOR = os.path.join(P2, 'extractor_pb_75.py')
CASCADE = os.path.join(P2, 'cascade_dewarp.py')
CONTROLLER = os.path.join(P2, 'phase2_controller.py')


def _source(path):
    with open(path) as f:
        src = f.read()
    return src, ast.parse(src)


def _pick(tree, names, path, ns):
    """Run the module-level definitions (functions and plain assignments) called `names` in the namespace ns."""
    body, found = [], set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            body.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in names:
            body.append(node)
            found.add(node.targets[0].id)
    assert found == set(names), f'{os.path.basename(path)}: not found at module level: {sorted(set(names) - found)}'
    exec(compile(ast.Module(body=body, type_ignores=[]), path, 'exec'), ns)
    return ns


def _function(tree, name):
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(fns) == 1, f'{name}: {len(fns)} definitions'
    return fns[0]


# ------------------------------------------------------------------------------------------------------------------------
# 1. the VLSSr catalog parser
# ------------------------------------------------------------------------------------------------------------------------
HEADER = '''
VLA Low-frequency Sky Survey (VLSSr) Catalog search, 1.15
Catalog made on 2013-05-30 (yyyy-mm-dd) with   92965 entries
Selection box: 24.00 hr x180.00 deg with center 12  0  0.000  00  0  0.00

   RA(2000)  Dec(2000) Dist(") Flux  Major Minor  PA  Res Field    X_pix  Y_pix
 h  m    s    d  m   s   Ori      Jy   "     "     deg
'''
SOURCES = '''00 00  0.12 -04 02 40.6 14549   0.92 <143. <82.9 -62.8    0000-050 2039.87 2269.30
       0.60         7.0  -180   0.15
00 00  0.16 +68 10 23.7 99999   1.36 <52.5 <52.5 -38.1    0000+750 2039.94  405.48
       0.82         4.5     0   0.19
00 22  8.94 -00 00 28.1 42973   1.23 <112. <56.7  -6.6    0040-150 3091.40 3062.57
       0.71         5.9   -12   0.21
05 34 31.97 +22 00 52.1 12345 1256.3  338.  221.  -48.2 PS 0520+200 1011.02 1999.98
       0.01         0.1    -3  12.50
12 30 49.42 +12 23 28.0 30010 <566.9 <75.1 <75.1   0.0    1220+150 1500.00 1500.00
23 59 59.99 -89 59 59.9   100   0.50  80.1  75.0  12.0    2340-900  100.00  200.00
17 45 40.04 -29 00 28.1 77777   4.50 <200.
Page 3 of the list made on day 12 of 30 entries follow below this line
   RA(2000)  Dec(2000) Dist(") Flux  Major Minor  PA  Res Field    X_pix  Y_pix
09 60 00.00 +10 00 60.0 55555   2.00  90.0  80.0  10.0    0940+100  300.00  400.00
'''


def _parsers():
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.utils.exceptions import AstropyWarning
    _, tree = _source(EXTRACTOR)
    ns = {'re': re, 'warnings': warnings, 'np': np, 'u': u, 'SkyCoord': SkyCoord, 'AstropyWarning': AstropyWarning}
    return _pick(tree, ('_VLSSR_RA', '_VLSSR_DEC', 'parse_vlssr_text', '_parse_vlssr_text_per_line'), EXTRACTOR, ns)


def _write(text):
    fd, path = tempfile.mkstemp(suffix='.text')
    with os.fdopen(fd, 'w') as f:
        f.write(text)
    return path


def _same_catalog(new, old):
    assert new is not None and old is not None, 'one parser returned nothing'
    assert new['type'] == old['type'] == 'VLSS'
    for key in ('ra', 'dec', 'flux', 'maj'):
        assert new[key].dtype == old[key].dtype and new[key].tobytes() == old[key].tobytes(), f'{key} differs'
    for attr in ('ra', 'dec'):
        a, b = getattr(new['coords'], attr), getattr(old['coords'], attr)
        assert a.unit == b.unit, f'coords.{attr}: unit {a.unit} against {b.unit}'
        assert np.asarray(a.value).tobytes() == np.asarray(b.value).tobytes(), f'coords.{attr} differs'
        assert np.asarray(a.rad).tobytes() == np.asarray(b.rad).tobytes(), f'coords.{attr} in radians differs'
    assert new['coords'].cartesian.xyz.value.tobytes() == old['coords'].cartesian.xyz.value.tobytes(), 'Cartesian vectors differ'


def _count_per_line_calls(ns):
    calls, original = [], ns['_parse_vlssr_text_per_line']

    def counted(lines):
        calls.append(len(lines))
        return original(lines)
    ns['_parse_vlssr_text_per_line'] = counted
    return calls, original


def test_vlssr_one_call_parser_equals_the_per_line_parser():
    ns = _parsers()
    calls, per_line = _count_per_line_calls(ns)
    path = _write(HEADER + SOURCES)
    try:
        new = ns['parse_vlssr_text'](path)
        with open(path) as f:
            old = per_line(f.readlines())
    finally:
        os.remove(path)
    assert not calls, 'the per-line fallback ran on a catalog of plain lines'
    # 8 sources: 7 plain lines (one of them with only 8 columns: no major axis) and the line with 60 minutes / 60 seconds,
    # which astropy accepts in both parsers; headers, uncertainty lines and the "Page 3" line (its eighth word is a number)
    # are not sources
    assert len(new['ra']) == len(old['ra']) == 8, f"{len(new['ra'])} and {len(old['ra'])} sources, expected 8"
    _same_catalog(new, old)
    assert new['dec'][2] < 0, 'a declination of -00 degrees lost its sign'
    # the size the parser keeps is the catalog's tenth column, arc seconds turned into degrees; a line too short to hold it gets 0
    assert new['maj'][6] == 0.0 and new['flux'][4] == 566.9 and abs(new['maj'][0] - 82.9 / 3600.0) < 1e-15


def test_vlssr_parser_falls_back_to_the_per_line_parser():
    # (a) a line astropy accepts although it is not in the plain form (a signed right ascension): the whole file goes
    #     through the original parser, and the line is a source as it always was
    # (b) a plain line the single call refuses (declination of 95 degrees): the same, and the line is skipped as it always was
    for extra, n_expected in (('+12 30 45.00 +10 00 00.0 11111   3.00  60.0  50.0  10.0    1220+100  500.00  600.00\n', 9),
                              ('12 30 45.00 +95 00 00.0 11111   3.00  60.0  50.0  10.0    1220+100  500.00  600.00\n', 8)):
        ns = _parsers()
        calls, per_line = _count_per_line_calls(ns)
        path = _write(HEADER + SOURCES + extra)
        said = io.StringIO()
        try:
            with contextlib.redirect_stdout(said):
                new = ns['parse_vlssr_text'](path)
            with open(path) as f:
                lines = f.readlines()
        finally:
            os.remove(path)
        assert calls == [len(lines)], f'the per-line parser ran {len(calls)} times on {calls} lines, expected once on the file'
        assert 'using the per-line parser for the whole catalog' in said.getvalue(), 'the fallback was taken without a word'
        old = per_line(lines)
        assert len(new['ra']) == n_expected, f"{len(new['ra'])} sources, expected {n_expected}"
        _same_catalog(new, old)
    # nothing parses: None, as before
    ns = _parsers()
    path = _write(HEADER)
    try:
        assert ns['parse_vlssr_text'](path) is None
    finally:
        os.remove(path)


# ------------------------------------------------------------------------------------------------------------------------
# 2. the kept catalogs of the dewarped hour images
# ------------------------------------------------------------------------------------------------------------------------
KEY_FUNCTIONS = ('_DEWARPED_CATALOGS', '_catalog_key', '_remember_catalog', '_recall_catalog')


def _catalog_namespace():
    import pandas as pd
    _, tree = _source(CASCADE)
    ns = {'os': os, 'pd': pd, 'cfg': types.SimpleNamespace(BDSF_THRESH_PIX=5.0, BDSF_THRESH_ISL=3.0),
          'ANCHOR_EL_CUT': None, 'ANCHOR_EAST_EL_CUT': None, 'ANCHOR_MAXPIX_BEAM_AREAS': None}
    return _pick(tree, KEY_FUNCTIONS, CASCADE, ns), pd


def test_kept_catalog_is_reused_only_for_the_same_file_and_settings():
    ns, pd = _catalog_namespace()
    remember, recall = ns['_remember_catalog'], ns['_recall_catalog']
    saved = {k: os.environ.pop(k, None) for k in ('P2_REUSE_CASCADE_CATALOGS', 'BDSF_NCORES')}
    tmp = tempfile.mkdtemp()
    try:
        image = os.path.join(tmp, 'hour_dewarped.fits')
        with open(image, 'wb') as f:
            f.write(b'x' * 2880)
        df = pd.DataFrame({'ra': [10.0, 20.0], 'dec': [1.0, -2.0], 'flux_peak': [3.0, 4.0], 'maj': [0.1, 0.2], 'min': [0.1, 0.1]})
        assert recall(image) is None, 'a catalog came back before any was kept'
        remember(image, df)
        df.loc[0, 'ra'] = 999.0                                   # the caller's frame changes after the catalog was kept
        got = recall(image)
        assert got is not None and list(got['ra']) == [10.0, 20.0], 'what is kept is not a copy of the extraction'
        got.loc[1, 'dec'] = 999.0                                 # the user of a recalled catalog changes it
        again = recall(image)
        assert list(again['dec']) == [1.0, -2.0] and list(again.columns) == ['ra', 'dec', 'flux_peak', 'maj', 'min'], \
            'what is handed back is not a copy'
        assert recall(os.path.join(tmp, '.', 'hour_dewarped.fits')) is not None, 'the same file under another spelling of its path'

        # every extraction setting: a changed value means a fresh extraction
        for name, value in (('ANCHOR_EL_CUT', 20.0), ('ANCHOR_EAST_EL_CUT', 35.0), ('ANCHOR_MAXPIX_BEAM_AREAS', 400.0)):
            ns[name] = value
            assert recall(image) is None, f'the kept catalog was reused although {name} changed'
            ns[name] = None
        for name, value in (('BDSF_THRESH_PIX', 7.0), ('BDSF_THRESH_ISL', 4.0)):
            old = getattr(ns['cfg'], name)
            setattr(ns['cfg'], name, value)
            assert recall(image) is None, f'the kept catalog was reused although cfg.{name} changed'
            setattr(ns['cfg'], name, old)
        os.environ['BDSF_NCORES'] = '1'
        assert recall(image) is None, 'the kept catalog was reused although BDSF_NCORES changed'
        del os.environ['BDSF_NCORES']
        assert recall(image) is not None

        # the switch
        os.environ['P2_REUSE_CASCADE_CATALOGS'] = '0'
        assert recall(image) is None, 'P2_REUSE_CASCADE_CATALOGS=0 did not switch the reuse off'
        del os.environ['P2_REUSE_CASCADE_CATALOGS']

        # the file itself: another modification time, then another size
        st = os.stat(image)
        os.utime(image, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
        assert recall(image) is None, 'the kept catalog was reused for a file with another modification time'
        os.utime(image, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert recall(image) is not None
        with open(image, 'ab') as f:
            f.write(b'y')
        os.utime(image, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert recall(image) is None, 'the kept catalog was reused for a file of another size'

        # nothing is kept for an empty or failed extraction, and a missing file raises nothing
        other = os.path.join(tmp, 'other_dewarped.fits')
        with open(other, 'wb') as f:
            f.write(b'z' * 10)
        remember(other, pd.DataFrame())
        remember(other, None)
        assert recall(other) is None, 'an empty extraction was kept'
        gone = os.path.join(tmp, 'gone.fits')
        remember(gone, df)
        assert recall(gone) is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def _settings_read(fn):
    """What a function reads besides its arguments: cfg attributes, module-level names in capitals, os.environ keys."""
    cfg_attrs, capitals, env = set(), set(), set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'cfg':
            cfg_attrs.add(node.attr)
        elif isinstance(node, ast.Name) and node.id.isupper() and len(node.id) > 3:
            capitals.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.isupper() and '_' in node.value:
            env.add(node.value)
    return cfg_attrs, capitals, env


def test_the_catalog_key_holds_every_setting_the_extraction_reads():
    src, tree = _source(CASCADE)
    extract = _function(tree, 'extract_sources')
    assert [a.arg for a in extract.args.args] == ['image_path', 'logger'] and not extract.args.kwonlyargs \
        and extract.args.vararg is None and extract.args.kwarg is None, \
        'extract_sources takes more than the image and the logger: a new argument must enter _catalog_key'
    read = [set(), set(), set()]
    for name in ('extract_sources', '_anchor_maxpix', '_elevation_masked_copy'):
        for have, more in zip(read, _settings_read(_function(tree, name))):
            have |= more
    # the helpers extract_sources calls are the two read above: a new one must be added to this test
    called = {n.func.id for n in ast.walk(extract) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    module_functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert called & module_functions == {'_anchor_maxpix', '_elevation_masked_copy'}, \
        f'extract_sources calls other functions of the module: {sorted(called & module_functions)}'
    key_cfg, key_capitals, key_env = _settings_read(_function(tree, '_catalog_key'))
    assert read[0] and read[0] <= key_cfg, f'cfg settings read by the extraction, missing in the key: {sorted(read[0] - key_cfg)}'
    assert read[1] and read[1] <= key_capitals, f'module settings missing in the key: {sorted(read[1] - key_capitals)}'
    assert read[2] == {'BDSF_NCORES'} and read[2] <= key_env, f'environment settings missing in the key: {sorted(read[2] - key_env)}'

    # the two call sites: the cascade keeps the catalog of the file it has just extracted; the 10-minute stage asks for it
    # and extracts as before when there is none
    subband = ast.get_source_segment(src, _function(tree, 'dewarp_subband'))
    assert re.search(r'dewarped_catalog = extract_sources\(dewarped_pbcorr, logger\)\s+_remember_catalog\(dewarped_pbcorr, dewarped_catalog\)',
                     subband), 'dewarp_subband no longer keeps the catalog of the file it has just extracted'
    tenmin = ast.get_source_segment(src, _function(tree, 'dewarp_10min_intervals'))
    assert re.search(r'ref_catalog = _recall_catalog\(deep_dewarped\)\s+if ref_catalog is not None:.*?else:\s+'
                     r'ref_catalog = extract_sources\(deep_dewarped, logger\)', tenmin, re.S), \
        'the 10-minute stage no longer extracts the reference itself when no catalog was kept'
    assert src.count('_recall_catalog(') == 2 and src.count('_remember_catalog(') == 2, \
        'the kept catalogs have a new user: re-examine the reuse before trusting it'


# ------------------------------------------------------------------------------------------------------------------------
# 3. the copy-in
# ------------------------------------------------------------------------------------------------------------------------
class _Log:
    def __init__(self):
        self.warnings, self.infos = [], []

    def warning(self, msg):
        self.warnings.append(msg)

    def info(self, msg):
        self.infos.append(msg)


COPY_NAMES = ('_WHOLE_NUMBER_SETTINGS', 'MAX_COPY_WORKERS', '_whole_number_setting', '_copy_workers',
              'check_runtime_settings', '_copy_checked', '_copy_files', 'copy_to_nvme')
SETTINGS = ('P2_COPY_WORKERS', 'P2_TENMIN_WORKERS', 'FULLBAND_WORKERS', 'BDSF_NCORES', 'P2_CASCADE_WORKERS')


def _copy_namespace(copy2=None):
    _, tree = _source(CONTROLLER)
    log = _Log()
    sh = types.SimpleNamespace(copy2=copy2 or shutil.copy2)
    return _pick(tree, COPY_NAMES, CONTROLLER, {'os': os, 'sys': sys, 'glob': glob, 'shutil': sh, 'logger': log}), log


def test_copy_in_copies_the_same_bytes_in_the_listed_order():
    saved = os.environ.pop('P2_COPY_WORKERS', None)
    tmp = tempfile.mkdtemp()
    try:
        src_dir = os.path.join(tmp, 'src')
        os.makedirs(src_dir)
        rng = np.random.default_rng(1)
        names = [f'img_{i:03d}.fits' for i in rng.permutation(40)]          # a listed order that is not the sorted order
        for i, n in enumerate(names):
            with open(os.path.join(src_dir, n), 'wb') as f:
                f.write(rng.bytes(1000 + 37 * i))
        for setting in (None, '1', '6'):
            dst_dir = os.path.join(tmp, f'dst_{setting}')
            os.makedirs(dst_dir)
            pairs = [(os.path.join(src_dir, n), os.path.join(dst_dir, n)) for n in names]
            ns, log = _copy_namespace()
            created, copy_threads = [], set()
            real_open, real_copy2 = open, ns['shutil'].copy2

            def recording_open(path, mode='r', *a, **k):
                if 'w' in mode:
                    created.append((path, threading.current_thread() is threading.main_thread()))
                return real_open(path, mode, *a, **k)

            def recording_copy2(s, d):
                copy_threads.add(threading.current_thread().name)
                return real_copy2(s, d)
            ns['open'] = recording_open
            ns['shutil'].copy2 = recording_copy2
            if setting is None:
                os.environ.pop('P2_COPY_WORKERS', None)
            else:
                os.environ['P2_COPY_WORKERS'] = setting
            assert ns['_copy_files'](pairs) == len(pairs), 'the count of copied files is wrong'
            assert ns['_copy_files']([]) == 0
            for s, d in pairs:
                with open(s, 'rb') as fs, open(d, 'rb') as fd:
                    assert fs.read() == fd.read(), f'{os.path.basename(d)} differs from its source (P2_COPY_WORKERS={setting})'
            assert sorted(os.listdir(dst_dir)) == sorted(names) and not log.warnings
            if setting == '6':
                assert created == [(d, True) for _, d in pairs], \
                    'the destination files were not all created first, in the listed order, by the calling thread'
                assert len(copy_threads) > 1 and threading.main_thread().name not in copy_threads
            else:
                assert not created and copy_threads == {threading.main_thread().name}, \
                    'unset or 1 must run the plain serial loop (no file created up front, no thread)'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        os.environ.pop('P2_COPY_WORKERS', None)
        if saved is not None:
            os.environ['P2_COPY_WORKERS'] = saved


def test_copy_in_refuses_a_short_copy():
    saved = os.environ.pop('P2_COPY_WORKERS', None)
    tmp = tempfile.mkdtemp()
    try:
        src = os.path.join(tmp, 'a.fits')
        with open(src, 'wb') as f:
            f.write(b'0123456789' * 100)
        for workers in (None, '4'):
            if workers:
                os.environ['P2_COPY_WORKERS'] = workers
            for n_short, must_raise in ((1, False), (2, True)):
                dst = os.path.join(tmp, f'out_{workers}_{n_short}.fits')
                attempts = {}                                     # per destination: each destination has one thread

                def short_copy2(s, d, _n=n_short, _attempts=attempts, _short=dst):
                    _attempts[d] = _attempts.get(d, 0) + 1
                    with open(s, 'rb') as fs, open(d, 'wb') as fd:
                        data = fs.read()
                        fd.write(data[:500] if d == _short and _attempts[d] <= _n else data)
                ns, log = _copy_namespace(short_copy2)
                pairs = [(src, dst), (src, dst + '.second')] if workers else [(src, dst)]
                try:
                    n = ns['_copy_files'](pairs)
                    raised = False
                except IOError as e:
                    raised = True
                    assert 'short copy after two attempts' in str(e)
                assert raised == must_raise, f'{n_short} short attempt(s), P2_COPY_WORKERS={workers}: raised={raised}'
                if must_raise:
                    left = [d for _, d in pairs if os.path.exists(d)]
                    assert not left, f'a refused copy left {len(left)} destination file(s) behind (P2_COPY_WORKERS={workers})'
                if not must_raise:
                    assert n == len(pairs) and os.path.getsize(dst) == 1000 and len(log.warnings) == 1 \
                        and '500 of 1000 bytes' in log.warnings[0], 'one short attempt must be retried, logged and then accepted'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        os.environ.pop('P2_COPY_WORKERS', None)
        if saved is not None:
            os.environ['P2_COPY_WORKERS'] = saved


def test_an_interrupted_copy_is_never_taken_as_copied():
    saved = {k: os.environ.pop(k, None) for k in SETTINGS}
    tmp = tempfile.mkdtemp()
    try:
        src = os.path.join(tmp, 'phase1', '73MHz')
        names = {('I', 'deep'): ['a-I-image.fits', 'a-I-image_pbcorr.fits', 'a-I-residual.fits', 'a-I-image.pybdsf.fits'],
                 ('I', '10min'): ['t%04d-I-image.fits' % i for i in range(6)],
                 ('V', '10min'): ['t%04d-V-image.fits' % i for i in range(6)]}
        for (pol, cat), files in names.items():
            os.makedirs(os.path.join(src, pol, cat))
            for i, n in enumerate(files):
                with open(os.path.join(src, pol, cat, n), 'wb') as f:
                    f.write(b'F' * (2880 * (i + 1)))
        wanted = sorted(os.path.join(pol, cat, n) for (pol, cat), files in names.items() for n in files
                        if '-image' in n and '.pybdsf' not in n)

        def images(work):
            return sorted(os.path.relpath(p, os.path.join(work, '73MHz'))
                          for p in glob.glob(os.path.join(work, '73MHz', '*', '*', '*.fits')))

        for workers in (None, '4'):
            if workers:
                os.environ['P2_COPY_WORKERS'] = workers
            # a copy interrupted at its fifth file: the error goes on, and no image is left in the run directory
            order = itertools.count(1)                            # next() is one step: no two threads get the same number

            def interrupted_copy2(s, d, _order=order):
                if next(_order) == 5:
                    raise KeyboardInterrupt()
                shutil.copy2(s, d)
            ns, log = _copy_namespace(interrupted_copy2)
            work = os.path.join(tmp, f'work_{workers}')
            try:
                ns['copy_to_nvme']({'73MHz': src}, work)
                raise AssertionError('the interrupt did not go on')
            except KeyboardInterrupt:
                pass
            assert images(work) == [], f'an interrupted copy left images behind (P2_COPY_WORKERS={workers}): {images(work)[:3]}'
            # the rerun copies the subband; a third run finds it copied and copies nothing
            ns, log = _copy_namespace()
            ns['copy_to_nvme']({'73MHz': src}, work)
            assert images(work) == wanted, 'the rerun did not copy exactly the image files'
            for rel in wanted:
                assert os.path.getsize(os.path.join(work, '73MHz', rel)) == os.path.getsize(os.path.join(src, rel))
            copies = []
            ns, log = _copy_namespace(lambda s, d: copies.append(d))
            ns['copy_to_nvme']({'73MHz': src}, work)
            assert not copies and any('already on NVMe' in m for m in log.infos), 'a copied subband was copied again'
            # the trace of a KILLED parallel copy (no clean-up could run): an empty image. The directory is refused.
            open(os.path.join(work, '73MHz', wanted[0]), 'wb').close()
            try:
                ns['copy_to_nvme']({'73MHz': src}, work)
                raise AssertionError('a subband directory holding an empty image was taken as copied')
            except IOError as e:
                assert 'empty image files' in str(e) and 'Remove that directory' in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        for k in SETTINGS:
            os.environ.pop(k, None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


# ------------------------------------------------------------------------------------------------------------------------
# 4. the runtime settings and the 10-minute interval pool, 5. the huge-page advice
# ------------------------------------------------------------------------------------------------------------------------
def test_unusable_runtime_settings_are_refused_at_the_start():
    saved = {k: os.environ.pop(k, None) for k in SETTINGS}
    try:
        ns, _ = _copy_namespace()
        check = ns['check_runtime_settings']
        check()                                                           # nothing set
        for name in SETTINGS:
            for bad in ('auto', '0', '-2', '2.5', 'six'):
                os.environ[name] = bad
                try:
                    check()
                    raise AssertionError(f'{name}={bad!r} was accepted')
                except SystemExit as e:
                    assert 'REFUSING' in str(e) and name in str(e)
            for good in ('1', '6', ' 4 '):
                os.environ[name] = good
                check()
            os.environ[name] = ''                                         # empty = unset
            check()
            del os.environ[name]
        for name, value, refused in (('P2_COPY_WORKERS', '7', True), ('P2_COPY_WORKERS', '32', True),
                                     ('P2_TENMIN_WORKERS', '32', False), ('FULLBAND_WORKERS', '8', False)):
            os.environ[name] = value
            try:
                check()
                was_refused = False
            except SystemExit as e:
                was_refused = True
                assert 'at most 6' in str(e)
            assert was_refused == refused, f'{name}={value}: refused={was_refused}'
            if name == 'P2_COPY_WORKERS':
                try:                                                      # the copy itself refuses too, whoever calls it
                    ns['_copy_files']([])
                    raise AssertionError('_copy_files ran with more than 6 copy streams')
                except SystemExit:
                    pass
            del os.environ[name]
    finally:
        for k in SETTINGS:
            os.environ.pop(k, None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    # the check runs right after the arguments are read, before the run starts
    src, tree = _source(CONTROLLER)
    body = _function(tree, 'main').body
    at = [i for i, n in enumerate(body) if ast.get_source_segment(src, n) == 'args = parser.parse_args()']
    assert len(at) == 1 and ast.get_source_segment(src, body[at[0] + 1]) == 'check_runtime_settings()', \
        'main no longer checks the runtime settings right after reading its arguments'
    # every setting the three modules read as a whole number is in the checked list
    read = set()
    for path in (CONTROLLER, CASCADE, os.path.join(P2, 'phase2_science_agg.py')):
        read |= set(re.findall(r"int\(os\.environ(?:\.get\(|\[)'([A-Z0-9_]+)'", _source(path)[0]))
    assert read and read <= set(ns['_WHOLE_NUMBER_SETTINGS']), \
        f"read as whole numbers but not checked at the start: {sorted(read - set(ns['_WHOLE_NUMBER_SETTINGS']))}"


def test_interval_pool_size_is_unchanged_unless_set():
    src, tree = _source(CASCADE)
    tenmin = _function(tree, 'dewarp_10min_intervals')
    assigns = [n for n in ast.walk(tenmin) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
               and n.targets[0].id == 'n_workers']
    assert len(assigns) == 2, f'n_workers is assigned {len(assigns)} times in the 10-minute stage, expected the default and the setting'
    assert ast.get_source_segment(src, assigns[0]) == 'n_workers = min(8, max(1, multiprocessing.cpu_count() - 2))', \
        'the default size of the 10-minute interval pool changed'
    guards = [n for n in ast.walk(tenmin) if isinstance(n, ast.If) and assigns[1] in n.body]
    assert len(guards) == 1 and ast.get_source_segment(src, guards[0].test) == "os.environ.get('P2_TENMIN_WORKERS')", \
        'the pool size is overridden by something other than a set P2_TENMIN_WORKERS'


def test_huge_page_advice_is_off_before_numpy_can_load():
    src, tree = _source(CONTROLLER)
    line = "os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')"
    at = [n.lineno for n in tree.body if ast.get_source_segment(src, n) == line]
    assert len(at) == 1, 'the controller no longer sets NUMPY_MADVISE_HUGEPAGE by default (setdefault: a set value wins)'
    standard = {'argparse', 'os', '__future__'}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node.lineno < at[0]:
            names = {a.name.split('.')[0] for a in node.names} if isinstance(node, ast.Import) else {(node.module or '').split('.')[0]}
            assert names <= standard, f'line {node.lineno} imports {sorted(names)} before the huge-page setting'


# ------------------------------------------------------------------------------------------------------------------------
# 5. the cascade's first extractions up front, and its two arms side by side
# ------------------------------------------------------------------------------------------------------------------------
UPFRONT_FUNCTIONS = KEY_FUNCTIONS + ('_UPFRONT_CATALOGS', 'CASCADE_WORKERS_DEFAULT', '_cascade_workers', '_end_with_parent',
                                     '_helper_pool', '_extract_up_front_one', '_remove_left_cut_copies',
                                     '_extract_up_front', '_recall_up_front', '_cascade_arm_in_child',
                                     'run_cascade_dewarping')
BANDS = ['55MHz', '59MHz', '64MHz', '69MHz', '73MHz', '78MHz', '82MHz']


class _InlinePool:
    """Stands in for concurrent.futures.ProcessPoolExecutor: the same calls, run in this process."""
    made = []
    fail_submit = fail_result = fail_map_after = None

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        self.max_workers, self.mp_context, self.initializer, self.initargs = max_workers, mp_context, initializer, initargs
        _InlinePool.made.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def shutdown(self, wait=True):
        self.was_shut_down = True
        self.waited = wait

    def map(self, fn, items):
        for i, item in enumerate(items):
            if _InlinePool.fail_map_after is not None and i >= _InlinePool.fail_map_after:
                raise RuntimeError('a worker of the pool died')
            yield fn(item)

    def submit(self, fn, arg):
        if _InlinePool.fail_submit:
            raise OSError('no process could be started')
        pool = self

        class _Future:
            def result(self_inner):
                if _InlinePool.fail_result:
                    raise RuntimeError('the child process died')
                if not hasattr(pool, 'value'):
                    pool.value = fn(arg)                      # run when asked for: after the DOWN arm, like a slow child
                return pool.value
        return _Future()


@contextlib.contextmanager
def _inline_pools():
    import concurrent.futures as cf
    real = cf.ProcessPoolExecutor
    cf.ProcessPoolExecutor = _InlinePool
    _InlinePool.made, _InlinePool.fail_submit, _InlinePool.fail_result, _InlinePool.fail_map_after = [], None, None, None
    try:
        yield _InlinePool
    finally:
        cf.ProcessPoolExecutor = real


class _CascadeLog:
    name = 'test_cascade'

    def __init__(self):
        self.lines = []

    def info(self, m):
        self.lines.append(m)
    warning = error = info


def _cascade_namespace(tmp, bands=BANDS, empty=(), extraction_fails=(), warns=(), down_fails=False, module=None,
                       killed_in_helper=()):
    """The cascade driver with its engines replaced: every extraction and every subband dewarp is recorded, no image is
    read. Hour images are small files under tmp/<band>/. What a helper PROCESS does is not seen in ``calls``; every
    extraction and dewarp also writes a line (its process id first) to tmp/extracted.txt or tmp/dewarped.txt.
    ``warns``: bands whose extraction says a warning; ``down_fails``: the DOWN arm raises; ``module``: run the
    definitions in this module object (real pools pickle their functions by module and name); ``killed_in_helper``:
    'extract:<band>' or 'dewarp:<band>' - the helper process doing that work kills itself (never this process)."""
    import pandas as pd
    _, tree = _source(CASCADE)
    calls = dict(extract=[], dewarp=[], arm=[], tenmin=[])
    main_pid = os.getpid()

    def note(name, *words):
        with open(os.path.join(tmp, name), 'a') as f:
            f.write(' '.join(str(w) for w in (os.getpid(),) + words) + '\n')

    def killed(what, band):
        if f'{what}:{band}' in killed_in_helper and os.getpid() != main_pid:
            os.kill(os.getpid(), signal.SIGKILL)
    for b in bands:
        os.makedirs(os.path.join(tmp, b), exist_ok=True)
        with open(os.path.join(tmp, b, f'hour_{b}.fits'), 'wb') as f:
            f.write(b.encode() * 100)

    def catalog(tag):
        return pd.DataFrame({'ra': [1.0, 2.0], 'dec': [3.0, 4.0], 'flux_peak': [5.0, 6.0], 'tag': [tag, tag]})

    def extract_sources(image_path, logger):
        calls['extract'].append(os.path.basename(image_path))
        note('extracted.txt', os.path.basename(image_path))
        b = os.path.basename(os.path.dirname(image_path))
        if f'extract:{b}' in killed_in_helper and os.getpid() != main_pid:
            with open(image_path.replace('.fits', '.elcut20e35.fits'), 'wb') as f:      # the copy a killed worker leaves
                f.write(b'cut')
        killed('extract', b)
        if b in extraction_fails:
            raise RuntimeError('extraction engine broke')
        if b in warns:
            logger.warning('  [BDSF] elevation mask failed (stand-in) - using full image')
        return pd.DataFrame() if b in empty else catalog('first:' + os.path.basename(image_path))

    def find_deep_pb_image(work_dir, subband, logger):
        p = os.path.join(work_dir, subband, f'hour_{subband}.fits')
        return p if os.path.exists(p) else None

    def dewarp_subband(work_dir, subband, ref_catalog, ref_label, is_anchor, diag_dir, logger, **kw):
        img = find_deep_pb_image(work_dir, subband, logger)
        first = ns['_recall_up_front'](img, logger)
        if first is None:
            first = extract_sources(img, logger)
        killed('dewarp', subband)
        dewarped = img.replace('.fits', '_dewarped.fits')
        with open(dewarped, 'wb') as f:
            f.write(b'd' * 50)
        out = catalog('dewarped:' + subband)
        ns['_remember_catalog'](dewarped, out)
        calls['dewarp'].append((subband, ref_label, os.getpid(), list(first['tag'])[0] if len(first) else None))
        note('dewarped.txt', subband, ref_label, list(first['tag'])[0] if len(first) else None)
        return out, True, None

    def _run_cascade_arm(work_dir, subband_list, start_catalog, start_subband, diag_dir, results, logger, **kw):
        calls['arm'].append((tuple(subband_list), start_subband, tuple(sorted(kw))))
        if down_fails and subband_list[0] == '69MHz':
            raise RuntimeError('the DOWN arm broke')
        ref = start_subband
        for sb in subband_list:
            _, ok, _ = dewarp_subband(work_dir, sb, start_catalog, f'{ref}_dewarped', False, diag_dir, logger, **kw)
            results[sb] = ok
            ref = sb

    def refuse(*a, **k):
        raise AssertionError('a flow-method or snapshot engine ran in a source-based test')

    ns = {} if module is None else module.__dict__
    ns.update({'os': os, 'glob': glob, 'pd': pd, 'warnings': warnings,
          'cfg': types.SimpleNamespace(BDSF_THRESH_PIX=5.0, BDSF_THRESH_ISL=3.0, ANCHOR_METHOD='rbf', CASCADE_METHOD='rbf',
                                       WARP_RBF_KERNEL='thin_plate_spline', WARP_RBF_SMOOTHING=1.5, WARP_RBF_EPSILON=None,
                                       ALL_SUBBANDS=list(bands), ANCHOR_SUBBAND='73MHz', VLSSR_CATALOG='vlssr'),
          'ANCHOR_EL_CUT': None, 'ANCHOR_EAST_EL_CUT': None, 'ANCHOR_MAXPIX_BEAM_AREAS': None,
          'ANCHOR_FALLBACK': ['73MHz', '78MHz', '69MHz'],
          'extract_sources': extract_sources, 'find_deep_pb_image': find_deep_pb_image, 'dewarp_subband': dewarp_subband,
          '_run_cascade_arm': _run_cascade_arm, 'load_vlssr_catalog': lambda *a: {'ra': []},
          'dewarp_10min_intervals': lambda work_dir, subbands, *a, **k: calls['tenmin'].append(list(subbands)),
          '_run_cascade_arm_flow': refuse, '_run_flow_anchor': refuse, '_dewarp_10min_flow': refuse,
          '_dewarp_remaining_snapshots': refuse, 'find_deep_pb_image_dewarped': refuse})
    return _pick(tree, UPFRONT_FUNCTIONS, CASCADE, ns), calls


def _run_cascade(tmp, **kw):
    ns, calls = _cascade_namespace(tmp, **kw)
    log = _CascadeLog()
    results = ns['run_cascade_dewarping'](tmp, log)
    return ns, calls, log, results


def _clean_cascade_env():
    return {k: os.environ.pop(k, None) for k in ('P2_CASCADE_WORKERS', 'P2_CASCADE_ARMS_SIDE_BY_SIDE', 'BDSF_NCORES',
                                                 'P2_REUSE_CASCADE_CATALOGS', 'P2_FORCE_ANCHOR_SUBBAND')}


def _restore_env(saved):
    for k, v in saved.items():
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = v


def _what_the_cascade_did(ns, calls, results):
    """Everything a product depends on: per subband its reference and its first catalog; the order of the results; the
    subbands handed to the 10-minute stage; the dewarped catalogs kept for it."""
    return (sorted((sb, ref, first) for sb, ref, _, first in calls['dewarp']), list(results.items()), calls['tenmin'],
            sorted((k[0], list(v['tag'])) for k, v in ns['_DEWARPED_CATALOGS'].items()))


def test_the_cascade_is_the_same_cascade_with_the_pool_and_the_child_arm():
    saved = _clean_cascade_env()
    tmps = []
    try:
        def run(env, **kw):
            _restore_env({k: None for k in saved})
            os.environ.update(env)
            tmp = tempfile.mkdtemp()
            tmps.append(tmp)
            with _inline_pools() as pools:
                ns, calls, log, results = _run_cascade(tmp, **kw)
                made = list(pools.made)
            return ns, calls, log, _what_the_cascade_did(ns, calls, results), made, tmp

        def strip(did, tmp):
            return (did[0], did[1], did[2], [(k.replace(tmp, ''), v) for k, v in did[3]])

        # as before: no pool, the UP arm after the DOWN arm
        ns0, calls0, log0, did0, made0, tmp0 = run({'P2_CASCADE_WORKERS': '1', 'P2_CASCADE_ARMS_SIDE_BY_SIDE': '0'})
        assert made0 == [], 'a pool was made although both settings were off'
        assert [a[0] for a in calls0['arm']] == [('69MHz', '64MHz', '59MHz', '55MHz'), ('78MHz', '82MHz')], calls0['arm']
        assert calls0['extract'] == [f'hour_{b}.fits' for b in ('73MHz', '69MHz', '64MHz', '59MHz', '55MHz', '78MHz', '82MHz')], \
            'with the pool off the first extractions are not made one by one in cascade order'
        assert list(did0[1]) == [(b, True) for b in ('73MHz', '69MHz', '64MHz', '59MHz', '55MHz', '78MHz', '82MHz')]
        assert not any('up-front' in m or 'child process' in m for m in log0.lines)

        # the defaults: every first extraction once, in the pool; the UP arm through the child function
        ns1, calls1, log1, did1, made1, tmp1 = run({})
        assert [p.max_workers for p in made1] == [4, 1], f'pools made: {[p.max_workers for p in made1]}'
        for p in made1:
            assert p.mp_context is not None and p.mp_context.get_start_method() == 'fork', 'a helper pool is not forked'
            assert p.initializer is ns1['_end_with_parent'] and p.initargs == (os.getpid(),), \
                'a helper pool does not ask to end with this process'
        assert sorted(calls1['extract']) == sorted(f'hour_{b}.fits' for b in BANDS), \
            f'first extractions with the pool on: {calls1["extract"]}'
        assert [a[0] for a in calls1['arm']] == [('69MHz', '64MHz', '59MHz', '55MHz'), ('78MHz', '82MHz')]
        assert strip(did1, tmp1) == strip(did0, tmp0), 'the cascade with the pool and the child arm is not the cascade as before'
        # the anchor and the four subbands of the DOWN arm say so here; the UP arm's two say so through the logger of the
        # same name in the child
        assert sum('catalog of the up-front extraction' in m for m in log1.lines) == 5
        assert all(first == f'first:hour_{sb}.fits' for sb, _, first in did1[0])
        assert any('7 of 7 catalogs kept' in m for m in log1.lines) and any('UP arm (child process) finished: 2 of 2' in m
                                                                             for m in log1.lines)
        assert getattr(made1[1], 'was_shut_down', False), 'the child pool was not shut down'
        assert calls1['arm'][0][2] == calls1['arm'][1][2] == calls0['arm'][1][2], 'the arms were given different settings'

        # each half alone
        for env in ({'P2_CASCADE_WORKERS': '1'}, {'P2_CASCADE_ARMS_SIDE_BY_SIDE': '0'}, {'P2_CASCADE_WORKERS': '3'}):
            nsx, callsx, logx, didx, madex, tmpx = run(env)
            assert strip(didx, tmpx) == strip(did0, tmp0), f'{env}: not the cascade as before'
            want = ([] if env.get('P2_CASCADE_WORKERS') == '1' else [int(env.get('P2_CASCADE_WORKERS', 4))]) + \
                   ([] if env.get('P2_CASCADE_ARMS_SIDE_BY_SIDE') == '0' else [1])
            assert [p.max_workers for p in madex] == want, f'{env}: pools {[p.max_workers for p in madex]}, expected {want}'

        # an empty or failed up-front extraction is not kept: the cascade extracts that image itself, as before
        for kw in (dict(empty=('64MHz',)), dict(extraction_fails=('64MHz',))):
            tmp = tempfile.mkdtemp()
            tmps.append(tmp)
            _restore_env({k: None for k in saved})
            with _inline_pools():
                ns, calls = _cascade_namespace(tmp, **kw)
                log = _CascadeLog()
                try:
                    ns['run_cascade_dewarping'](tmp, log)
                except RuntimeError:
                    assert 'extraction_fails' in kw                # the stand-in engine raises again at step 3: as before
            assert calls['extract'].count('hour_64MHz.fits') == 2, f'{kw}: {calls["extract"]}'
            if 'empty' in kw:
                assert any('6 of 7 catalogs kept' in m for m in log.lines)
            else:
                assert any('Up-front extraction stopped' in m for m in log.lines)

        # the warning of an up-front extraction whose catalog is kept reaches the hour's log, named by its image; the
        # warning of one that is not kept is said by step 3, which extracts that image again
        for kw, want in ((dict(warns=('59MHz',)), 1), (dict(warns=('59MHz',), empty=('59MHz',)), 0)):
            tmp = tempfile.mkdtemp()
            tmps.append(tmp)
            _restore_env({k: None for k in saved})
            with _inline_pools():
                ns, calls, log, results = _run_cascade(tmp, **kw)
            said = [m for m in log.lines if 'up-front extraction of hour_59MHz.fits: [BDSF] elevation mask failed' in m]
            assert len(said) == want, f'{kw}: {said}'
            if not want:
                assert sum('elevation mask failed (stand-in)' in m for m in log.lines) == 1, 'step 3 did not say the warning'

        # the setting, read by any caller of the cascade: unset or empty = 4; not a whole number of at least 1 = refused
        ns, _ = _cascade_namespace(tmps[0])
        for raw, want in ((None, 4), ('', 4), ('1', 1), ('7', 7)):
            _restore_env({'P2_CASCADE_WORKERS': raw})
            assert ns['_cascade_workers']() == want, (raw, ns['_cascade_workers']())
        for raw in ('0', '-2', 'four', '2.5'):
            os.environ['P2_CASCADE_WORKERS'] = raw
            try:
                ns['_cascade_workers']()
            except ValueError as e:
                assert 'REFUSING: P2_CASCADE_WORKERS' in str(e), e
            else:
                raise AssertionError(f'P2_CASCADE_WORKERS={raw!r} was taken')
    finally:
        _restore_env(saved)
        for t in tmps:
            shutil.rmtree(t, ignore_errors=True)


def test_a_failed_pool_or_child_leaves_the_cascade_as_before():
    saved = _clean_cascade_env()
    tmps = []
    try:
        def run(setup, left=()):
            tmp = tempfile.mkdtemp()
            tmps.append(tmp)
            for name in left:
                os.makedirs(os.path.dirname(os.path.join(tmp, name)), exist_ok=True)
                open(os.path.join(tmp, name), 'wb').close()
            with _inline_pools() as pools:
                setup(pools)
                ns, calls, log, results = _run_cascade(tmp)
            did = _what_the_cascade_did(ns, calls, results)
            run.tmp = tmp
            return calls, log, (did[0], did[1], did[2], [(k.replace(tmp, ''), v) for k, v in did[3]])

        def stops(setup, **kw):
            """A cascade that must end with an error: (the error, calls, log, the pools made)."""
            tmp = tempfile.mkdtemp()
            tmps.append(tmp)
            with _inline_pools() as pools:
                setup(pools)
                ns, calls = _cascade_namespace(tmp, **kw)
                log = _CascadeLog()
                try:
                    ns['run_cascade_dewarping'](tmp, log)
                except RuntimeError as e:
                    return e, calls, log, list(pools.made)
            raise AssertionError('the cascade did not stop')

        os.environ.update({'P2_CASCADE_WORKERS': '1', 'P2_CASCADE_ARMS_SIDE_BY_SIDE': '0'})
        _, _, before = run(lambda pools: None)
        del os.environ['P2_CASCADE_WORKERS'], os.environ['P2_CASCADE_ARMS_SIDE_BY_SIDE']

        # the up-front pool dies after three images: three catalogs kept, the other four extracted at step 3; the
        # elevation-cut copies its stopped workers left beside the hour images are removed, nothing else is
        left = ('64MHz/hour_64MHz.elcut20.fits', '59MHz/hour_59MHz.elcut20e35.fits')
        stay = ('64MHz/hour_64MHz.elcut20.txt', '64MHz/other_64MHz.elcut20.fits', '64MHz/hour_64MHz_dewarped_old.fits')
        calls, log, did = run(lambda pools: setattr(pools, 'fail_map_after', 3), left=left + stay)
        assert did == before
        assert len(calls['extract']) == 3 + 4 and sorted(set(calls['extract'])) == sorted(f'hour_{b}.fits' for b in BANDS)
        assert any('Up-front extraction stopped' in m for m in log.lines) and any('3 of 7 catalogs kept' in m for m in log.lines)
        assert not any(os.path.exists(os.path.join(run.tmp, n)) for n in left), 'a left elevation-cut copy is still there'
        assert all(os.path.exists(os.path.join(run.tmp, n)) for n in stay), 'a file that is no elevation-cut copy was removed'
        assert any('Removed 2 elevation-cut copies' in m for m in log.lines), log.lines
        # a pool that does not stop removes nothing (its workers remove their own copies)
        calls, log, did = run(lambda pools: None, left=left)
        assert did == before and all(os.path.exists(os.path.join(run.tmp, n)) for n in left)

        # the child cannot be started: the UP arm runs after the DOWN arm in this process
        calls, log, did = run(lambda pools: setattr(pools, 'fail_submit', True))
        assert did == before and [a[0] for a in calls['arm']] == [('69MHz', '64MHz', '59MHz', '55MHz'), ('78MHz', '82MHz')]
        assert any('could not be started beside the DOWN arm' in m for m in log.lines)


        # the child fails after it started: the cascade ends with the child's error; the arm is NOT run a second time in
        # this process, the 10-minute stage is not reached, the pool is shut down
        e, calls, log, made = stops(lambda pools: setattr(pools, 'fail_result', True))
        assert str(e) == 'the child process died', e
        assert [a[0] for a in calls['arm']] == [('69MHz', '64MHz', '59MHz', '55MHz')], calls['arm']
        assert calls['tenmin'] == [] and getattr(made[1], 'was_shut_down', False) and made[1].waited
        assert any("UP arm's child process failed" in m and 'not run a second time' in m for m in log.lines)

        # the DOWN arm fails: its error goes on only after the child was waited for
        e, calls, log, made = stops(lambda pools: None, down_fails=True)
        assert str(e) == 'the DOWN arm broke', e
        assert getattr(made[1], 'was_shut_down', False) and made[1].waited, 'the child was not waited for'
        assert any("waiting for the UP arm's child process to end" in m for m in log.lines) and calls['tenmin'] == []
    finally:
        _restore_env(saved)
        for t in tmps:
            shutil.rmtree(t, ignore_errors=True)


def _process_state(pid):
    """The kernel's state letter of a process, or None when it is gone."""
    try:
        with open(f'/proc/{pid}/stat') as f:
            return f.read().rsplit(')', 1)[1].split()[0]
    except OSError:
        return None


def test_the_pool_and_the_child_are_real_processes():
    """The same cascade through real forked processes: the catalogs and results cross the process boundary, the UP arm's
    products are written by the child, a killed up-front worker is absorbed, a killed child ends the cascade."""
    if not sys.platform.startswith('linux'):
        return
    saved = _clean_cascade_env()
    tmps = []
    try:
        def run(env, **kw):
            _restore_env({k: None for k in saved})
            os.environ.update(env)
            tmp = tempfile.mkdtemp()
            tmps.append(tmp)
            module = types.ModuleType(f'cascade_in_real_processes_{len(tmps)}')
            sys.modules[module.__name__] = module
            ns, calls = _cascade_namespace(tmp, module=module, **kw)
            log = _CascadeLog()
            try:
                results, error = ns['run_cascade_dewarping'](tmp, log), None
            except RuntimeError as e:
                results, error = {}, e
            finally:
                del sys.modules[module.__name__]

            def lines(name):
                with open(os.path.join(tmp, name)) as f:
                    return [ln.split() for ln in f.read().splitlines()]
            kept = sorted((k[0].replace(tmp, ''), list(v['tag'])) for k, v in ns['_DEWARPED_CATALOGS'].items())
            return dict(tmp=tmp, log=log, calls=calls, error=error, results=list(results.items()), kept=kept,
                        extracted=lines('extracted.txt'), dewarped=lines('dewarped.txt'))

        me = str(os.getpid())
        before = run({'P2_CASCADE_WORKERS': '1', 'P2_CASCADE_ARMS_SIDE_BY_SIDE': '0'})
        assert before['error'] is None and {ln[0] for ln in before['extracted'] + before['dewarped']} == {me}

        on = run({})
        assert on['error'] is None and on['results'] == before['results'] and on['kept'] == before['kept'], \
            'through real processes the cascade is not the cascade as before'
        assert sorted(ln[1:] for ln in on['dewarped']) == sorted(ln[1:] for ln in before['dewarped']), \
            'a subband was dewarped against another reference or from another first catalog'
        assert sorted(ln[1] for ln in on['extracted']) == sorted(f'hour_{b}.fits' for b in BANDS), on['extracted']
        workers = {ln[0] for ln in on['extracted']}
        assert me not in workers and 1 <= len(workers) <= 4, f'the first extractions ran in {workers} (this process: {me})'
        by = {ln[1]: ln[0] for ln in on['dewarped']}
        assert all(by[b] == me for b in ('73MHz', '69MHz', '64MHz', '59MHz', '55MHz')), by
        assert by['78MHz'] == by['82MHz'] and by['78MHz'] not in workers | {me}, f'the UP arm ran in {by}'
        assert all(os.path.exists(os.path.join(on['tmp'], b, f'hour_{b}_dewarped.fits')) for b in BANDS)

        # an up-front worker is killed in its extraction (it leaves its elevation-cut copy): absorbed, the copy removed
        got = run({}, killed_in_helper=('extract:64MHz',))
        assert got['error'] is None and got['results'] == before['results'] and got['kept'] == before['kept']
        assert any('Up-front extraction stopped (BrokenProcessPool' in m for m in got['log'].lines), got['log'].lines
        assert glob.glob(os.path.join(got['tmp'], '*', '*.elcut*')) == [], 'a killed worker\'s copy is still there'
        assert [ln[0] for ln in got['dewarped'] if ln[1] == '64MHz'] == [me]

        # the child is killed in its second subband: the cascade ends; this process does not dewarp the UP arm again
        got = run({}, killed_in_helper=('dewarp:82MHz',))
        assert isinstance(got['error'], RuntimeError) and 'ended from outside' in str(got['error']), got['error']
        up = [ln for ln in got['dewarped'] if ln[1] in ('78MHz', '82MHz')]
        assert [ln[1] for ln in up] == ['78MHz'] and up[0][0] != me, f'the UP arm after its child was killed: {up}'
        assert got['calls']['tenmin'] == [], 'the 10-minute stage ran after the cascade had stopped'
    finally:
        _restore_env(saved)
        for t in tmps:
            shutil.rmtree(t, ignore_errors=True)


def test_a_helper_process_ends_with_its_parent():
    """A process that starts a helper pool and is then killed leaves no helper behind (Linux: the parent-death signal)."""
    if not sys.platform.startswith('linux'):
        return
    tmp = tempfile.mkdtemp()
    module = types.ModuleType('cascade_helper_ends_with_parent')
    sys.modules[module.__name__] = module
    helper = None
    try:
        ns, _ = _cascade_namespace(tmp, module=module)
        r, w = os.pipe()
        sys.stdout.flush()                                 # the fork below copies what is not yet written
        sys.stderr.flush()
        middle = os.fork()
        if middle == 0:                                    # the stand-in controller: starts a helper, then is killed
            try:
                pool = ns['_helper_pool'](1)
                os.write(w, str(pool.submit(os.getpid).result()).encode())
                pool.submit(time.sleep, 300)
                time.sleep(0.5)
            finally:
                os.kill(os.getpid(), signal.SIGKILL)
        os.close(w)
        os.waitpid(middle, 0)
        helper = int(os.read(r, 32).decode() or 0)
        os.close(r)
        assert helper > 0, 'the stand-in controller started no helper'
        t0 = time.time()
        while _process_state(helper) not in (None, 'Z', 'X') and time.time() - t0 < 20:
            time.sleep(0.1)
        state = _process_state(helper)
        assert state in (None, 'Z', 'X'), f'helper {helper} is still at work (state {state}) 20 s after its parent was killed'
        helper = None
    finally:
        if helper:
            try:
                os.kill(helper, signal.SIGKILL)
            except OSError:
                pass
        del sys.modules[module.__name__]
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_up_front_catalog_is_for_the_same_file_and_settings_only():
    saved = _clean_cascade_env()
    tmp = tempfile.mkdtemp()
    try:
        ns, calls = _cascade_namespace(tmp)
        log = _CascadeLog()
        with _inline_pools():
            ns['_extract_up_front'](tmp, BANDS, log)
        image = os.path.join(tmp, '64MHz', 'hour_64MHz.fits')
        got = ns['_recall_up_front'](image, log)
        assert got is not None and list(got['tag']) == ['first:hour_64MHz.fits'] * 2
        got.loc[0, 'ra'] = 999.0
        assert list(ns['_recall_up_front'](image, log)['ra']) == [1.0, 2.0], 'what is handed back is not a copy'
        for name, value in (('ANCHOR_EL_CUT', 20.0), ('ANCHOR_EAST_EL_CUT', 35.0), ('ANCHOR_MAXPIX_BEAM_AREAS', 400.0)):
            ns[name] = value
            assert ns['_recall_up_front'](image, log) is None, f'the up-front catalog was used although {name} changed'
            ns[name] = None
        os.environ['BDSF_NCORES'] = '2'
        assert ns['_recall_up_front'](image, log) is None, 'the up-front catalog was used although BDSF_NCORES changed'
        del os.environ['BDSF_NCORES']
        st = os.stat(image)
        os.utime(image, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
        assert ns['_recall_up_front'](image, log) is None, 'the up-front catalog was used for a file written again'
        assert ns['_recall_up_front'](os.path.join(tmp, 'gone.fits'), log) is None
        # the dewarped catalogs of the 10-minute stage are another memo: an up-front catalog is never taken for one
        assert ns['_recall_catalog'](os.path.join(tmp, '55MHz', 'hour_55MHz.fits')) is None
        # the worker runs under the settings it is handed, and says which image its catalog is of
        path, df, said = ns['_extract_up_front_one']((os.path.join(tmp, '55MHz', 'hour_55MHz.fits'), (11.0, 12.0, 13.0)))
        assert (ns['ANCHOR_EL_CUT'], ns['ANCHOR_EAST_EL_CUT'], ns['ANCHOR_MAXPIX_BEAM_AREAS']) == (11.0, 12.0, 13.0)
        assert path.endswith('hour_55MHz.fits') and list(df['tag']) == ['first:hour_55MHz.fits'] * 2 and said == []
        # a new cascade starts from an empty memo
        ns['_extract_up_front'](tmp, [], log)
        assert ns['_UPFRONT_CATALOGS'] == {}
    finally:
        _restore_env(saved)
        shutil.rmtree(tmp, ignore_errors=True)


def test_step_3_extracts_itself_when_no_up_front_catalog_is_there():
    src, tree = _source(CASCADE)
    body = _function(tree, 'dewarp_subband').body
    at = [i for i, n in enumerate(body) if ast.get_source_segment(src, n) == 'df_lwa = _recall_up_front(calc_img, logger)']
    assert len(at) == 1, 'step 3 no longer asks for the up-front catalog of its extraction image'
    assert ast.get_source_segment(src, body[at[0] + 1]) == 'if df_lwa is None:\n        df_lwa = extract_sources(calc_img, logger)', \
        'step 3 no longer extracts the image itself when no up-front catalog is there'
    calls = [ast.get_source_segment(src, n) for n in ast.walk(_function(tree, 'dewarp_subband'))
             if isinstance(n, ast.Call) and getattr(n.func, 'id', '') == 'extract_sources']
    assert sorted(calls) == ['extract_sources(calc_img, logger)', 'extract_sources(dewarped_pbcorr, logger)'], calls


# ------------------------------------------------------------------------------------------------------------------------
# 7. a rerun after a killed run: left elevation-cut copies and cut-off dewarped files
# ------------------------------------------------------------------------------------------------------------------------
FINDERS = ('find_deep_pb_image', 'find_deep_raw_image', 'find_all_images_to_warp', 'find_deep_raw_image_dewarped',
           'find_deep_pb_image_dewarped')


def test_a_left_elevation_cut_copy_is_never_taken_for_an_image():
    _, tree = _source(CASCADE)
    ns = _pick(tree, FINDERS, CASCADE, {'os': os, 'glob': glob})
    log = types.SimpleNamespace(warning=lambda *a, **k: None, info=lambda *a, **k: None)
    tmp = tempfile.mkdtemp(prefix='p2_elcut_')
    try:
        d = os.path.join(tmp, '73MHz', 'I', 'deep')
        os.makedirs(d)
        pb = os.path.join(d, '73MHz-I-deep-Taper-Robust-0.75-image-pbcorr.fits')
        raw = os.path.join(d, '73MHz-I-deep-Taper-Robust-0.75-image.fits')
        left = [x.replace('.fits', t + '.fits') for x in (pb, raw) for t in ('.elcut20', '.elcut20e30')]
        left += [x.replace('.fits', '_dewarped.fits') for x in left]           # a left copy that a rerun once dewarped
        for f in [pb, raw, pb.replace('.fits', '_dewarped.fits'), raw.replace('.fits', '_dewarped.fits')] + left:
            open(f, 'w').close()
        assert sorted(left)[0] < pb, 'the test no longer has a left copy that sorts before the image'
        assert ns['find_deep_pb_image'](tmp, '73MHz', log) == pb
        assert ns['find_deep_raw_image'](tmp, '73MHz', log) == raw
        assert ns['find_deep_pb_image_dewarped'](tmp, '73MHz', log) == pb.replace('.fits', '_dewarped.fits')
        assert ns['find_deep_raw_image_dewarped'](tmp, '73MHz', log) == raw.replace('.fits', '_dewarped.fits')
        assert sorted(ns['find_all_images_to_warp'](tmp, '73MHz', ['deep'])) == sorted([pb, raw])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    src, _ = _source(CASCADE)
    assert src.count('".pybdsf" not in os.path.basename(f)') == src.count('".elcut" not in os.path.basename(f)'), \
        'an image finder excludes PyBDSF products but not the elevation-cut copies'
    assert src.count('".pybdsf" in bn') == src.count('".elcut" in bn'), \
        'a file loop skips PyBDSF products but not the elevation-cut copies'


def test_a_dewarped_file_is_written_whole_or_not_at_all():
    src, tree = _source(CASCADE)
    assert 'fits.writeto(out_path, warped' not in src, 'a _dewarped.fits is written without the temporary name'
    assert src.count('_writeto_atomic(out_path, warped, ') == 3, 'the three dewarp writers no longer use _writeto_atomic'
    from astropy.io import fits as real

    def stops(path, data, header, overwrite=False):
        with open(path, 'wb') as f:
            f.write(b'SIMPLE  =                    T')           # a cut-off file, then the run is killed
        raise KeyboardInterrupt('killed')
    tmp = tempfile.mkdtemp(prefix='p2_atomic_')
    try:
        out = os.path.join(tmp, 'x-image_dewarped.fits')
        ns = _pick(tree, ('_writeto_atomic',), CASCADE, {'os': os, 'fits': types.SimpleNamespace(writeto=stops)})
        try:
            ns['_writeto_atomic'](out, np.zeros((4, 4), np.float32), real.Header())
        except KeyboardInterrupt:
            pass
        assert not os.path.exists(out), 'a stopped write left a file under the final name'
        assert not glob.glob(os.path.join(tmp, '*.fits')), 'a stopped write left a .fits name a finder could take'
        ns = _pick(tree, ('_writeto_atomic',), CASCADE, {'os': os, 'fits': real})
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        ns['_writeto_atomic'](out, data, real.Header())
        ns['_writeto_atomic'](out, data + 1, real.Header())                   # over an existing file, as overwrite=True did
        assert np.array_equal(real.getdata(out), data + 1)
        assert sorted(os.listdir(tmp)) == ['x-image_dewarped.fits'], sorted(os.listdir(tmp))   # the stopped write's .part too
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


TESTS = [test_vlssr_one_call_parser_equals_the_per_line_parser, test_vlssr_parser_falls_back_to_the_per_line_parser,
         test_kept_catalog_is_reused_only_for_the_same_file_and_settings,
         test_the_catalog_key_holds_every_setting_the_extraction_reads,
         test_copy_in_copies_the_same_bytes_in_the_listed_order, test_copy_in_refuses_a_short_copy,
         test_an_interrupted_copy_is_never_taken_as_copied, test_unusable_runtime_settings_are_refused_at_the_start,
         test_interval_pool_size_is_unchanged_unless_set, test_huge_page_advice_is_off_before_numpy_can_load,
         test_the_cascade_is_the_same_cascade_with_the_pool_and_the_child_arm,
         test_a_failed_pool_or_child_leaves_the_cascade_as_before,
         test_the_pool_and_the_child_are_real_processes, test_a_helper_process_ends_with_its_parent,
         test_the_up_front_catalog_is_for_the_same_file_and_settings_only,
         test_step_3_extracts_itself_when_no_up_front_catalog_is_there,
         test_a_left_elevation_cut_copy_is_never_taken_for_an_image, test_a_dewarped_file_is_written_whole_or_not_at_all]


def main():
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f'PASS  {t.__name__}')
        except Exception as e:
            failed += 1
            print(f'FAIL  {t.__name__}: {e.__class__.__name__}: {str(e)[:400]}')
    print(f'{len(TESTS) - failed}/{len(TESTS)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
