#!/usr/bin/env python
"""The 10-s cutout engines' own reads (steps_cutouts.EngineReads), taken in a child process under the engines' interpreter and
environment (an ad hoc target set included). It imports the production modules of --code-root (the engines' directory, as
production runs them: `cd $OPS`) and prints result lines; it writes nothing. Two modes:

run (once per stage, BEFORE any engine starts; --cells: every night cell of the stage):
  'ENGINE-TARGETS ' + JSON {module, n_targets, multiple}: the writer's fixed targets (phase3_snapshot_cutouts._targets, the
      lists or the ad hoc set) and, per '<sample>/<file object name>' that more than one entry shares, their number. Entries
      of one name within MERGE_ARCMIN are one object and one file, but write_cutouts writes and counts each entry in view.
  'ENGINE-EXCLUSIONS ' + JSON per cell {cell, path, sha1, excluded[, group_reader]} or {cell, error}: the file the subband
      engine reads (phase3_snapshot_search.DATA_DECISIONS, its sha1) and excluded_subbands(lst, date) as written - it refuses
      a subband in phase3_snapshot_search.main (exit 3) and drops the excluded member subbands of a wideband in
      regen_group_cutouts; with --group-reader-root, the group search's deep-rule reader as regen_group_cutouts calls it
      (:488-490: that root first on sys.path, coadd_full_pipeline.load_exclusions(scope=_excl_scope(lst, date)) then
      is_excluded) for every name of --scopes: {path, module, scope, verdicts}. A reader that fails reports the cell's error;
      the step refuses its units (PHOT-EXCL-FILE, ruling C-09: both readers).

inview (once per unit after its engine, when a staged cutout's name is shared by more than one list entry; --keys):
  'ENGINE-INVIEW ' + JSON {Stokes: {in_view: {key: n or null}, geometries, grids, frames}}: per shared name, how many of
      its entries pass write_cutouts' own view test (below MIN_EL_DEG at the mid-time of the frames it cut: skipped; a pixel
      position outside the cube's grid: skipped) - the mid-time from the TIMES table of a staged cutout of that Stokes (the
      writer writes its frames' ts there), the celestial WCS from every frame header of the unit's cubes (FRAME_TABLE 'header',
      the headers load_cube_from_flowcube takes the writer's from) and the grid from their FRAMES axes. null when the
      frames' geometries disagree on the count."""
import argparse
import hashlib
import json
import os
import sys

from astropy.io import fits as _fits_nommap   # noqa: E402
_fits_nommap.conf.use_memmap = False   # never memory-map a Lustre file (R-GREGG-20261117-NO-MMAP-OF-LUSTRE-FILES-READ-OR-WRITE)

TARGETS = 'ENGINE-TARGETS '
EXCLUSIONS = 'ENGINE-EXCLUSIONS '
INVIEW = 'ENGINE-INVIEW '


def _sha1(path):
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()


def shared_names(CUT):
    """{'<sample>/<file object name>': [(ra, dec), ...]} of the writer's fixed targets, every name (the writer's file name)."""
    out = {}
    for tag, nm, ra, dec in CUT._targets():
        out.setdefault('%s/%s' % (tag, CUT.safe_name(nm)), []).append((ra, dec))
    return out


def run_mode(a):
    import phase3_snapshot_search as SS
    import phase3_snapshot_cutouts as CUT
    names = shared_names(CUT)
    print(TARGETS + json.dumps({'module': os.path.abspath(CUT.__file__), 'n_targets': sum(len(v) for v in names.values()),
                                'multiple': {k: len(v) for k, v in sorted(names.items()) if len(v) > 1}},
                               sort_keys=True), flush=True)
    recs = {}
    try:
        before = _sha1(SS.DATA_DECISIONS)
    except Exception as e:
        before = None
        for cell in a.cells:
            recs[cell] = {'cell': cell, 'error': 'the subband engine\'s list %s unreadable (%s: %s)' % (
                SS.DATA_DECISIONS, type(e).__name__, e)}
    if before is not None:
        for cell in a.cells:
            lst, date = cell.split('_', 1)
            try:
                recs[cell] = {'cell': cell, 'path': SS.DATA_DECISIONS, 'module': os.path.abspath(SS.__file__),
                              'excluded': sorted(SS.excluded_subbands(lst, date))}
            except Exception as e:
                recs[cell] = {'cell': cell, 'error': 'excluded_subbands: %s: %s' % (type(e).__name__, e)}
        if a.group_reader_root:
            try:
                import phase3_snapshot_group_search as GS
                sys.path.insert(0, a.group_reader_root)     # as regen_group_cutouts :488
                from exopipe.phase3 import coadd_full_pipeline as CF
                scopes = {}
                for cell in a.cells:
                    lst, date = cell.split('_', 1)
                    scopes.setdefault(GS._excl_scope(lst, date), []).append(cell)
                for scope, cells in sorted(scopes.items()):
                    ex = CF.load_exclusions(scope=scope)    # sets the module's [PERBAND] set to this scope's, as the engine
                    for cell in cells:
                        lst, date = cell.split('_', 1)
                        if 'error' not in recs[cell]:
                            recs[cell]['group_reader'] = {
                                'path': CF.DECISIONS_CSV, 'module': os.path.abspath(CF.__file__), 'scope': scope,
                                'verdicts': {s: bool(CF.is_excluded(lst, s, date, ex)) for s in a.scopes}}
            except Exception as e:
                for cell in a.cells:
                    recs[cell] = {'cell': cell, 'error': 'the group search\'s reader (%s): %s: %s' % (
                        a.group_reader_root, type(e).__name__, e)}
        after = _sha1(SS.DATA_DECISIONS)
        for cell in a.cells:
            if 'error' in recs[cell]:
                continue
            if after != before:
                recs[cell] = {'cell': cell, 'error': '%s changed while it was read (sha1 %s, then %s)' % (
                    SS.DATA_DECISIONS, before, after)}
            else:
                recs[cell]['sha1'] = before
    for cell in a.cells:
        print(EXCLUSIONS + json.dumps(recs[cell], sort_keys=True), flush=True)


def inview_mode(a):
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    from phase3_snapshot_search import lst_deg_at, ts_to_epoch_seconds     # as write_cutouts imports them (first, as run_mode)
    import phase3_snapshot_cutouts as CUT
    names = shared_names(CUT)
    keys = sorted(set(a.keys))
    times = dict(x.split('=', 1) for x in a.times)
    cubes = {}
    for x in a.cubes:
        S, p = x.split('=', 1)
        cubes.setdefault(S, []).append(p)
    out = {}
    for S in sorted(times):
        with fits.open(times[S], memmap=False) as h:
            ts = [str(x) for x in h['TIMES'].data['ts']]
        # write_cutouts: times = [ts_to_epoch_seconds(m["ts"]) for m in meta]; lst_mid = lst_deg_at(0.5 * (min + max))
        t = [ts_to_epoch_seconds(x) for x in ts]
        lst_mid = lst_deg_at(0.5 * (min(t) + max(t)))
        geoms, grids = {}, set()
        for p in cubes.get(S, []):
            with fits.open(p, memmap=False) as h:
                grids.add((int(h['FRAMES'].header['NAXIS1']), int(h['FRAMES'].header['NAXIS2'])))   # cube [T, Y, X]
                for x in h['FRAME_TABLE'].data['header']:
                    w = WCS(fits.Header.fromstring(str(x), sep='\n')).celestial       # write_cutouts: WCS(hdr).celestial
                    c = w.wcs
                    g = (tuple(c.ctype), tuple(c.crval), tuple(c.crpix), tuple(c.cdelt), tuple(c.get_pc().ravel()),
                         c.lonpole, c.latpole, tuple(tuple(v) for v in c.get_pv()), w.sip is not None)
                    geoms.setdefault(g, w)
        n = {}
        for k in keys:
            seen = set()
            for w in geoms.values():
                for nx, ny in grids:
                    m = 0
                    for ra, dec in names.get(k, []):
                        if CUT._elevation(ra, dec, lst_mid) < CUT.MIN_EL_DEG:
                            continue
                        px, py = w.all_world2pix(ra, dec, 0)
                        if not (np.isfinite(px) and np.isfinite(py)) or not (0 <= px < nx and 0 <= py < ny):
                            continue
                        m += 1
                    seen.add(m)
            n[k] = seen.pop() if len(seen) == 1 else None
        out[S] = {'in_view': n, 'geometries': len(geoms), 'grids': sorted(grids), 'frames': len(ts)}
    print(INVIEW + json.dumps(out, sort_keys=True), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('mode', choices=('run', 'inview'))
    ap.add_argument('--code-root', required=True)
    ap.add_argument('--cells', nargs='*', default=[])
    ap.add_argument('--group-reader-root')
    ap.add_argument('--scopes', nargs='*', default=[])
    ap.add_argument('--lst')
    ap.add_argument('--date')
    ap.add_argument('--times', nargs='*', default=[], help='S=<staged cutout> per Stokes')
    ap.add_argument('--cubes', nargs='*', default=[], help='S=<flow cube> per Stokes and cube')
    ap.add_argument('--keys', nargs='*', default=[])
    a = ap.parse_args(argv)
    # the production modules of --code-root, never this package's directory (as stamp_engine.py)
    here = os.path.dirname(os.path.realpath(__file__))
    sys.path[:] = [a.code_root] + [p for p in sys.path if os.path.realpath(p or '.') != here]
    os.chdir(a.code_root)
    (run_mode if a.mode == 'run' else inview_mode)(a)
    return 0


if __name__ == '__main__':
    sys.exit(main())
