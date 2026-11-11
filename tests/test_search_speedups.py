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
     be re-examined before it is trusted.
Data-free; standard library only. Run: python tests/test_search_speedups.py   (or under pytest)
"""
import ast
import os
import sys

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


TESTS = [test_x2_copies_are_made_only_for_the_refinement, test_only_the_refinement_reads_the_x2_copies]


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
