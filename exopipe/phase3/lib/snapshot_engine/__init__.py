"""The 10-s / 1-minute snapshot ENGINE, ported from production clv15 (Gregg 2026-11-11: the faster 10-s / 1-minute code lands
in the refactor; production clv15 and its running fleet stay untouched).

What this directory is
  The engine scripts that the snapshot fleet runs per cell (exopipe_ops/snapshot_fleet_node_v10.sh), copied from clv15 at ONE
  pinned commit and then sped up here, each speed-up proven byte-identical against the pinned production files on real hours
  (plan/STAGE_MAP_20261111.md in exopipe_ops/refactor_20261012; REVIEW.md item 8). The files keep production's names and run
  as SCRIPTS under production's interpreters, exactly as clv15's do, so a difference with production is `diff -u` of two files
  and the snapshot chat can adopt a file back unchanged:

    phase3_snapshot_search.py        subband search, 10-s and 1-minute tiers (py36: site PYTHON['py36'])
    phase3_snapshot_group_search.py  wideband Red / Green / Blue search (py36)
    phase3_snapshot_cutouts.py       photometry cutout cubes written by the search (py36)
    phase3_snapshot_inject.py        injection stamps of --inject runs (py36)
    phase3_snapshot_lightcurves.py   master-parent light curves per (subband, Stokes) (py36)
    phase3_snapshot_regrid.py        pilot-era frames onto the Master grid (py36)
    phase3_snapshot_flow.py          per-frame flow to the Master (flow interpreter, GPU)
    phase3_snapshot_cube.py          flowed-frame cubes: write (flow interpreter), verify (py36), explode
    phase3_snapshot_flow_movie.py    flow movie + series per subband (flow interpreter)
    phase3_snapshot_frame_movie.py   the 19 frame movies of a cell (flow interpreter)

The per-cell driver (refactor code, not from clv15)
    schedule.py                      tasks with prerequisites and core / memory / GPU budgets, run side by side
    cell.py                          one cell, the fleet's products, pipelined: a port of the cell body of
                                     exopipe_ops/snapshot_fleet_node_v10.sh onto the scheduler - two flow processes per GPU, and
                                     each subband's search, light curves, cubes and flow movie started when its flow ends.
                                     Writes only under its --out root; the fleet remains production's driver until cutover.

The pin (PIN_COMMIT / PIN_BLOBS below)
  clv15 e731a4a (2026-11-12) = be63765 (2026-11-11, the commit that searched the Dec 18-27 2024 cells used for the parity
  runs) plus production's two later commits to phase3_snapshot_cube.py (c868cb1: write --fluxscal divides the pixels by the
  hour-band factor and records FSAPPLD, default off; e731a4a: rescale rewrites an archived cube in Master units), ported
  2026-11-12; no other pinned file changed between the two commits. The cell driver passes no --fluxscal: where the factor
  comes from in v2 is the flux-scale ruling's implementation, not part of this port.
  clv15 12b9a71 (2026-11-13) = e731a4a plus production's _excl_scope change in phase3_snapshot_group_search.py (a non-GPU
  cell reads exclusion scope 'snapshot', which honors the [SNAPSHOT-TIERS-ONLY rows; was None, the default), ported
  2026-11-13; no other pinned file changed between the two commits. PIN_BLOBS holds the
  git blob id of every ported file at that commit. Production keeps moving: tests/test_snapshot_engine.py reports every pinned
  file production has changed since (one explicit failure naming what to port), as tests/production_pin.py does for the ops
  ports. Moving the pin is a port round.

Still run or imported IN PLACE (not ported; resolved through the legacy clv15 root placed LAST on sys.path, or by an absolute path
in the engine itself):
  phase3_snapshot_refastro.py, phase3_snapshot_chain.py   read by the light curves (Master catalog reader, names, samples,
                                                          flow-field lookup)
  /lustre/gh/main/exopipe exopipe.phase3.coadd_full_pipeline   the exclusion reader of the wideband search (load_exclusions,
                                                          is_excluded)
  image_plane_correction (flow interpreter package)       the flow solver

This module itself is import-safe and holds only the pin; nothing here is imported by the engine scripts.
"""

CLV15_REPO = '/lustre/gh/main/clv15'
PIN_COMMIT = '12b9a71d67c687fe6f4d1b03e97f01a822f312f8'
PIN_BLOBS = {
    'phase3_snapshot_search.py': 'db49397021a1c7596170dc6c0c3de48d4c8bf75c',
    'phase3_snapshot_group_search.py': '860b42e94cc6073596583997d416a3d4a83ffc24',
    'phase3_snapshot_cutouts.py': '5c6fa63521b581e27d31524c6aec819fc779de9d',
    'phase3_snapshot_lightcurves.py': '8f669f5504115267a7b9551e06b0a120bf36e8c4',
    'phase3_snapshot_cube.py': '912501b17d139c56b165c725687cb211dcd7ba62',
    'phase3_snapshot_frame_movie.py': '709a9448ddb2ba218754ee88347696e32ecef965',
    'phase3_snapshot_flow_movie.py': '0407eb749f1e2fd34329d9f41a4ad8a40873572c',
    'phase3_snapshot_flow.py': 'ff7c35eb50ce120a71a68cf4f94f4ca19b0e21be',
    'phase3_snapshot_regrid.py': '9a0ef18f1d2568ed9d57ed8dc227217e8dfa3069',
    'phase3_snapshot_inject.py': '57e92129cf4c9e26ffc0b2e754e12c589990adeb',
}
