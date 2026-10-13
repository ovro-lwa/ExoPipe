"""Child process of `exopipe3 frames --tier deep` (and of the optional sub-LST searchframes stage): ONE engine call.

    python -m exopipe.phase3.lib.frames.deep_child SPEC.json

Started by lib/frames/deep_engine.run_child through core.legacy.run (cwd = the worktree, environment built from config,
inherited P3_* cleared; spec plan/specs/frames.md 7.2). SPEC.json holds the UnitSpec fields and ``result_path``. The
engine (exopipe.phase3.lib.search.deep_search.run_deep_cell) is imported only here, after sys.argv is reset as the
production fleet does (p3s5_search_fleet.sh:193; the module captures sys.argv at import, inventory E29), and is called
with frames_only=True: a frame-only rebuild (refactor decision 2026-10-13: `search` detect is the deep frame producer in
the window). Without the keyword (the search owner's commit, lib/frames/deep_seam.py) the child refuses (gate
'engine-seam') before any work.

Writes ONE result file (JSON, atomic): {'status': 'ok'|'refused'|'error', 'gate', 'reason', 'return', 'engine_file'} and
exits 0 (ok), 2 (refused) or 1 (error). The parent maps it (deep_engine.run_child).
"""
import json
import os
import sys
import traceback


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    return repr(v)


def _write(path, obj):
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 1
    with open(argv[1]) as f:
        spec = json.load(f)
    sys.argv = ['exopipe3-frames']            # as the fleets do before the engine import (p3s5_search_fleet.sh:193)
    res = {'status': None, 'unit': f"{spec['lst']}_{spec['scope']}_{spec['stokes']}"}
    rc = 0
    try:
        import inspect
        from exopipe.phase3.lib.search import deep_search as ds
        res['engine_file'] = ds.__file__
        if 'frames_only' not in inspect.signature(ds.run_deep_cell).parameters:
            res.update(status='refused', gate='engine-seam',
                       reason=f'{ds.__file__}: run_deep_cell has no frames_only keyword yet (search owner\'s commit; '
                              f'lib/frames/deep_seam.py)')
            _write(spec['result_path'], res)
            print('FRAMES-ENGINE-END refused engine-seam', flush=True)
            return 2
        kw = dict(label=spec['label'], included_dates=(spec.get('included_dates') or None), stokes=spec['stokes'],
                  cadence=spec['cadence'], frames_only=True)
        print(f"FRAMES-ENGINE-START {res['unit']} cadence={spec['cadence']} included={kw['included_dates']}", flush=True)
        ret = ds.run_deep_cell(spec['lst'], spec['scope'], **kw)
        res.update(status='ok', **{'return': _jsonable(ret)})
    except RuntimeError as e:
        msg = str(e)
        if 'REFUS' in msg:
            res.update(status='refused', gate='engine', reason=msg)
            rc = 2
        else:
            res.update(status='error', reason=f'RuntimeError: {msg}', traceback=traceback.format_exc()[-4000:])
            rc = 1
    except BaseException as e:            # noqa: B902 - a child must always leave its result
        res.update(status='error', reason=f'{type(e).__name__}: {e}', traceback=traceback.format_exc()[-4000:])
        rc = 1
    _write(spec['result_path'], res)
    print(f"FRAMES-ENGINE-END {res['status']}", flush=True)
    if res.get('traceback'):
        print(res['traceback'], flush=True)
    return rc


if __name__ == '__main__':
    sys.exit(main(sys.argv))
