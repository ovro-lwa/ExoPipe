#!/usr/bin/env python3
"""Speed-ups of the ported deep / 10-minute search engine (exopipe/phase3/lib/search/deep_search.py): each must change no
product (REVIEW.md item 8). What a real job proves on data is in exopipe_ops/refactor_20261012/plan/STAGE_MAP_20261111.md;
these data-free tests guard the rule each speed-up rests on, from the engine's SOURCE (the module itself needs the cluster's
packages and site file to import):

  1. the x2 FFT-upsampled copies (master_up2.fits, ep_<window>_up2.fits) are made only when the x2 flow refinement is on:
     `_x2_copy` writes nothing with P3_REFINE_OFF=1, writes once otherwise, never rewrites an existing copy;
  2. nothing but the refinement reads those copies: in the engine, `fft_upsample2` and `write_up_fits` are called only by
     `_x2_copy` and (the upsampling itself, behind the same switch) by `_residual`; the `*_up2.fits` names are only handed to
     `_batch_flows`, which runs only with the refinement on. If a new reader appears, this test fails and the speed-up must
     be re-examined before it is trusted;
  3. the threaded per-frame reads (`_in_order`) return their results in the order of the items whatever order the threads
     finish in, and with one thread run nothing but the plain loop; the reader threads of the first read of every window
     only read pixels (`_read`): the zap of corrupt pixels and its record (`_zap`) stay in the caller, and a read error is
     re-raised at its serial position;
  4. the look-ahead source extraction (`_IdentAhead`): the detection loop gets, for every window, exactly what the serial
     call returns, and what the call printed appears at the place of the call, in the serial order; a result is used ONLY
     when it was computed from the very arguments the loop passes - different arguments, an error in the worker, or a call
     that changed its arguments make the loop call itself (so an error is raised where it was); with one process there is
     no pool at all.
Data-free (numpy only). Run: python tests/test_search_speedups.py   (or under pytest)
"""
import ast
import os
import sys
import tempfile
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'search', 'deep_search.py')


def _tree():
    with open(ENGINE) as f:
        src = f.read()
    return src, ast.parse(src)


def _function(tree, name):
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(fns) == 1, f'{name}: {len(fns)} definitions in the engine'
    return fns[0]


def _calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name]


def test_x2_copies_are_made_only_for_the_refinement():
    src, tree = _tree()
    fn = _function(tree, '_x2_copy')
    code = compile(ast.Module(body=[fn], type_ignores=[]), ENGINE, 'exec')
    for refine_off in (True, False):
        written, upsampled, existing = [], [], set()
        ns = {'_REFINE_OFF': refine_off, 'os': type('os', (), {'path': type('p', (), {'exists': staticmethod(lambda p: p in existing)})}),
              'fft_upsample2': lambda a: upsampled.append(a) or ('x2', a),
              'write_up_fits': lambda data, hdr0, path: written.append((data, hdr0, path)) or existing.add(path)}
        exec(code, ns)
        r1 = ns['_x2_copy']('frame', 'hdr', '/w/ep_A_up2.fits')
        r2 = ns['_x2_copy']('frame', 'hdr', '/w/ep_A_up2.fits')
        if refine_off:
            assert (r1, r2) == (False, False) and not written and not upsampled, 'a x2 copy was made with the refinement off'
        else:
            assert (r1, r2) == (True, False), 'with the refinement on the copy must be written once and never rewritten'
            assert written == [(('x2', 'frame'), 'hdr', '/w/ep_A_up2.fits')] and upsampled == ['frame']


def test_only_the_refinement_reads_the_x2_copies():
    src, tree = _tree()
    owners = {}
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        inner = {id(c) for sub in ast.walk(fn) if isinstance(sub, ast.FunctionDef) and sub is not fn for c in ast.walk(sub)}
        for name in ('fft_upsample2', 'write_up_fits', '_batch_flows', '_x2_copy'):
            for c in _calls(fn, name):
                if id(c) not in inner:                         # attributed to the innermost function only
                    owners.setdefault(name, []).append(fn.name)
    assert sorted(set(owners.get('write_up_fits', []))) == ['_x2_copy'], owners.get('write_up_fits')
    assert sorted(set(owners.get('fft_upsample2', []))) == ['_residual', '_x2_copy'], owners.get('fft_upsample2')
    # _residual upsamples only with the refinement on (the same switch)
    res = ast.get_source_segment(src, _function(tree, '_residual'))
    assert 'epoch if _REFINE_OFF else fft_decimate2(apply_flow_field(fft_upsample2(epoch), flow))' in res, \
        '_residual no longer skips the x2 path with the refinement off'
    # the copies' names are built in run_deep_cell and handed to _batch_flows only, which runs only with the refinement on
    run = ast.get_source_segment(src, _function(tree, 'run_deep_cell'))
    assert run.count('_up2.fits') == 2 and run.count('_batch_flows(') == 1, (run.count('_up2.fits'), run.count('_batch_flows('))
    i_off, i_else, i_flows = run.index('if _REFINE_OFF:'), run.index('else:', run.index('if _REFINE_OFF:')), run.index('_batch_flows(')
    assert i_off < i_else < i_flows and 'flows = {d_: None for d_, _p in items}' in run[i_off:i_else], \
        '_batch_flows is no longer confined to the branch with the refinement on'
    others = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name not in ('run_deep_cell', '_x2_copy', '_batch_flows', '_batch_flows_unlocked')
              and '_up2' in (ast.get_source_segment(src, n) or '')]
    assert not others, f'another function of the engine names the x2 copies: {others}'


def _namespace(names, **stubs):
    """The named top-level definitions of the engine, executed with stand-ins for what they call (the module itself needs
    the cluster's packages to import)."""
    src, tree = _tree()
    body = [n for n in tree.body if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)
            or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
    found = {getattr(n, 'name', None) or n.targets[0].id for n in body}
    assert found == set(names), f'not found in the engine: {set(names) - found}'
    ns = dict(os=os, sys=sys)
    ns.update(stubs)
    exec(compile(ast.Module(body=body, type_ignores=[]), ENGINE, 'exec'), ns)
    return ns


def test_threaded_reads_keep_the_serial_order():
    ns = _namespace(['_in_order'])
    seen = []

    def slow_first(x):                                   # the first items finish last
        time.sleep(0.02 * (10 - x) if x < 10 else 0)
        seen.append(x)
        return x * x
    assert ns['_in_order'](slow_first, range(12), 6) == [x * x for x in range(12)]
    assert seen != sorted(seen), 'the stand-in did not finish out of order: the test proves nothing'
    main = threading.current_thread()
    assert ns['_in_order'](lambda x: threading.current_thread() is main, range(5), 1) == [True] * 5, 'one thread must be the plain loop'
    src, tree = _tree()
    run = ast.get_source_segment(src, _function(tree, 'run_deep_cell'))
    # the reader threads of the first read only READ: the zap and its fallback record are applied by the loop, in order
    assert 'return _read(f' in run and "ei = _zap(_sub, f'{ebase}/subtracted/{d}_I_sub_{scope}.fits') + _recon_master(d)" in run
    assert 'if isinstance(_sub, Exception):\n            raise _sub' in run, 'a read error is no longer re-raised at its serial position'
    read = ast.get_source_segment(src, _function(tree, '_read'))
    assert 'fallback' not in read and 'print' not in read
    load = ast.get_source_segment(src, _function(tree, '_load'))
    assert 'return _zap(_read(p), p)' in load, '_load is no longer the read followed by the zap'


class _Hdr(dict):
    def tostring(self):
        return repr(sorted(self.items()))


def _ahead_namespace(calls, fail=(), mutate=()):
    import numpy as np

    def ident_i(resid, master, hdr, beam, sstack, refuse_on_extraction_failure=False):
        d = hdr['WINDOW']
        calls.append(('I', d, os.getpid()))
        print(f'  Running BDSF on difference image {d}', flush=True)
        os.write(2, f'stty: noise of {d}\n'.encode())            # C-level output of the call, straight to the descriptor
        if d in fail:
            raise RuntimeError(f'extraction failed for {d}')
        if d in mutate:
            resid[0, 0] += 1.0
        print(f'  BDSF: {int(resid.sum())} positive peaks', flush=True)
        return [{'window': d, 'flux': float(resid.sum()), 'status': 'PASS', 'beam': beam, 'stack': None if sstack is None else float(sstack.sum())}]

    def ident_v(resid, master, hdr, beam, sstack, sigma_thresh=None, refuse_on_extraction_failure=False, zenith_time=None, return_all=False):
        calls.append(('V', hdr['WINDOW'], os.getpid()))
        print(f'  V search {hdr["WINDOW"]} zenith {zenith_time}', flush=True)
        return [{'window': hdr['WINDOW'], 'flux': -float(resid.sum()), 'zt': zenith_time, 'all': return_all, 'sigma': sigma_thresh}]

    def ident_args(ctx, kind, d):
        return ctx['resid'][kind][d].copy(), _Hdr(WINDOW=d, DURATION=600.0), (ctx['sstack'] if kind == 'I' else None), (None if kind == 'I' else 'T' + d)
    return _namespace(['IDENT_PROCS', 'IDENT_WAIT_S', '_ident_digest', '_ident_call', '_ident_worker', '_IdentAhead'], time=time,
                      np=np, run_identification_i=ident_i, run_identification_v=ident_v, _ident_args=ident_args), np


def _loop(ns, np, nproc, windows, resid, sstack, tamper=None):
    """A stand-in for the detection loop: the same calls in the same order; returns (results, everything written to fd 1/2)."""
    ctx = dict(resid=resid, sstack=sstack, master_I='M', beam='B', xrefuse=True)
    ahead = ns['_IdentAhead'](ctx, nproc)
    ahead.start([(k, d) for d in windows for k in ('I', 'V')])
    out, errors = [], []
    sys.stdout.flush(); sys.stderr.flush()
    tf = tempfile.TemporaryFile()
    keep = os.dup(1), os.dup(2)
    os.dup2(tf.fileno(), 1); os.dup2(tf.fileno(), 2)
    try:
        for d in windows:
            for k in ('I', 'V'):
                r = resid[k][d].copy()
                if tamper and (k, d) in tamper:
                    r[1, 1] += 5.0                                 # the loop passes something else than the worker assumed
                print(f'window {d} Stokes {k}', flush=True)
                try:
                    out.append(ahead.take(k, d, r, _Hdr(WINDOW=d, DURATION=600.0), sstack if k == 'I' else None, None if k == 'I' else 'T' + d))
                except RuntimeError as e:
                    errors.append((k, d, str(e)))
                    print(f'REFUSED {d}: {e}', flush=True)
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(keep[0], 1); os.dup2(keep[1], 2); os.close(keep[0]); os.close(keep[1])
    rec = os.path.join(tempfile.mkdtemp(prefix='ident_ahead_'), 'ident_ahead.json')
    ahead.close(rec)
    tf.seek(0)
    text = tf.read().decode()
    tf.close()
    return out, errors, text, ahead, rec


def test_look_ahead_identification_equals_the_serial_loop():
    import json
    windows = ['2025-05-08T%02d' % i for i in range(6)]
    for case, kw in (('plain', {}), ('a failing window', dict(fail=('2025-05-08T02',))), ('a call that changes its arguments', dict(mutate=('2025-05-08T03',))),
                     ('the loop passes other pixels', dict(tamper={('I', '2025-05-08T04'), ('V', '2025-05-08T01')}))):
        tamper = kw.pop('tamper', None)
        calls_s, calls_p = [], []
        ns_s, np = _ahead_namespace(calls_s, **kw)
        ns_p, _ = _ahead_namespace(calls_p, **kw)
        rng = np.random.RandomState(7)
        resid = {k: {d: rng.normal(size=(8, 8)) for d in windows} for k in ('I', 'V')}
        sstack = np.ones((8, 8))
        out_s, err_s, text_s, ah_s, _ = _loop(ns_s, np, 1, windows, resid, sstack, tamper)
        out_p, err_p, text_p, ah_p, rec = _loop(ns_p, np, 4, windows, resid, sstack, tamper)
        assert ah_s.dir is None and not ah_s.pid and ah_s.used == 0 and len(calls_s) == 12, f'{case}: one process must mean no look-ahead at all'
        assert out_p == out_s and err_p == err_s, f'{case}: the look-ahead changed what the loop received'
        assert text_p == text_s, f'{case}: the log differs from the serial loop\n--- serial\n{text_s}\n--- look-ahead\n{text_p}'
        here = os.getpid()
        in_loop = [(k, d) for k, d, pid in calls_p if pid == here]
        r = json.load(open(rec))
        expect = sorted({('I', d) for d in kw.get('fail', ())} | {('I', d) for d in kw.get('mutate', ())} | set(tamper or ()))
        assert sorted(in_loop) == expect, f'{case}: the loop itself made the calls {in_loop}, expected {expect}'
        assert r['used'] == 12 - len(expect) and len(r['called_in_loop']) == len(expect) and r['jobs'] == 12, (case, r)
        assert not ah_p.alive and ah_p.dir is None, f'{case}: a look-ahead process or its directory was left behind'
        if 'fail' in kw:
            assert err_p == [('I', '2025-05-08T02', 'extraction failed for 2025-05-08T02')], 'the error was not raised by the loop\'s own call'


TESTS = [test_x2_copies_are_made_only_for_the_refinement, test_only_the_refinement_reads_the_x2_copies,
         test_threaded_reads_keep_the_serial_order, test_look_ahead_identification_equals_the_serial_loop]


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
