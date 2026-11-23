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
  2026-11-13; no other pinned file changed between the two commits.
  clv15 b3bedae (2026-11-13) = 12b9a71 plus production's two later commits to pinned files, ported 2026-11-14: 715bd0d
  (phase3_snapshot_cube.py: production installed this directory's dither-seed and memory changes, adding the opt-in write
  --stamp of the standard header and rescale's acceptance check - the ported file is production's, byte for byte; it differed
  from this directory's earlier copy in five comment lines) and b3bedae (phase3_snapshot_search.py: the 1-minute tier bins a
  block at its minute's middle, clipped to the hour's bins, so the first block no longer takes the elevation mask of the
  hour's last bin; applied to this directory's copy, its speed-ups kept). The cell driver passes no --stamp.
  clv15 5142b5c (2026-11-15) = b3bedae plus production's cube rescale gate (phase3_snapshot_cube.py: exact re-quantization
  bound of half the tile step, step no coarser than archived / F, lossless tiles bit-exact; replaces the fixed 0.05
  worst-pixel limit), ported 2026-11-17. This directory's copy is production's file at 5142b5c except for four no-memory-map
  lines (R-GREGG-20261117-NO-MMAP-OF-LUSTRE-FILES-READ-OR-WRITE: the module-level switch, memmap=False twice, Cube.flow
  reading one frame through .section); no other pinned file changed between the two commits.
  clv15 1985fc8 (2026-11-18) = 5142b5c plus production's three commits of 2026-11-18 to pinned files, ported 2026-11-23:
  6e4febe (phase3_snapshot_cutouts.py: SAMPLES gains ('LPT', 'LPTs.csv'); Gregg 2026-11-18 "Approve gen6 with the LPTs
  included"), dfb75ea (no memory map: memmap=False written out on four reads of phase3_snapshot_search.py and two of
  phase3_snapshot_frame_movie.py; production's phase3_snapshot_cube.py lines were already here, and that copy is now
  production's file at 1985fc8 plus the two module-switch lines) and 1985fc8 (phase3_snapshot_group_search.py: wideband
  weights on rescaled members, R-GREGG-20261118-WIDEBAND-WEIGHTS-ON-RESCALED-CUBES-KEEP-RULING: a member is weighted
  1/(F rms^2), F = its frame header's FSAPPLD, which _FlowCubeFrames copies from the cube's primary; a factor that is not
  positive and finite refuses; here in accumulate()'s _one, which production's loop and the per-instant speed-up share, so the
  speed-up is kept). No other pinned file changed between the two commits, and clv15 HEAD (84139b6 on 2026-11-23) holds the
  same blobs. PROVEN 2026-11-23 on real data: production's wideband cutout regeneration (clv15 HEAD,
  phase3_snapshot_group_search.py --cutouts-from-cubes Red, 00h 2025-09-06 Stokes I, member cubes rescaled with FSAPPLD
  0.956-1.049) and this directory's copy (search threads 6) wrote the same 12 output files - 11 byte-identical, the outcome
  record differing only in its 'written' time - and the same 162 cutout files (HWJ 38, 10 pc 119, LPT 2, planets 3),
  byte-identical after blanking the one DATE-CRE card; the pre-port copy (v2 b9ad9bf) on the same job differed (member weights
  in the ratio 1/FSAPPLD per subband within 6.6e-6, no LPT files, the data of every common cutout). The cutout target lists
  equal production's _targets() (389 entries: 77 HWJ, 292 10 pc, 20 LPT). The memory-map lines change no read: Cube.flow
  (.section) equals an ordinary seek-and-read of the FLOW plane, and a PB library file and a warp screen read with
  memmap=False equal the earlier default read of a node-local copy. tests/test_snapshot_engine.py adds the weight form, its
  refusal and the member-cube reader's factor.
  PIN_BLOBS holds the git blob id of every ported file at the pin commit. Production keeps moving: tests/test_snapshot_engine.py
  reports every pinned file production has changed since (one explicit failure naming what to port), as tests/production_pin.py
  does for the ops ports. Moving the pin is a port round.

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
PIN_COMMIT = '1985fc80d3b2b04b20105bf93471c51c44af3add'
PIN_BLOBS = {
    'phase3_snapshot_search.py': 'aec1a4b61cb3541c488ce5ef8e9170f0c8f313d3',
    'phase3_snapshot_group_search.py': '51675633dd061a13a10b6eb2ecfdbcc5fb7a4b48',
    'phase3_snapshot_cutouts.py': 'c6eba30d0f783670744abeb79b8daa8d5361de86',
    'phase3_snapshot_lightcurves.py': '8f669f5504115267a7b9551e06b0a120bf36e8c4',
    'phase3_snapshot_cube.py': '719edc7b41fd2a7c36c9ab832ce51e1c29375c15',
    'phase3_snapshot_frame_movie.py': '7e38901d7bbb5801576518e9c4925a23ae681c58',
    'phase3_snapshot_flow_movie.py': '0407eb749f1e2fd34329d9f41a4ad8a40873572c',
    'phase3_snapshot_flow.py': 'ff7c35eb50ce120a71a68cf4f94f4ca19b0e21be',
    'phase3_snapshot_regrid.py': '9a0ef18f1d2568ed9d57ed8dc227217e8dfa3069',
    'phase3_snapshot_inject.py': '57e92129cf4c9e26ffc0b2e754e12c589990adeb',
}
