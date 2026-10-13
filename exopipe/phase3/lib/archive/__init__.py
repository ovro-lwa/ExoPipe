"""lib/archive - the code behind `exopipe3 archive` (steps/archive.py): manifests, checksum transfers, content audits,
quarantine, retirement of excluded nights, rollback.

Modules
  common      Ctx (roots of one run), checksums, the quarantine move and the verified copy (the only two ways this step changes
              a file: a same-filesystem rename with inode/size check, or a '.part' copy verified by checksum then renamed),
              gates G2 (no production write in the window), G3/G4/G5/G6, G10 (ruling), G11 (apply = approved manifest), G12
              (concurrent writer), G19 (free space), manifest CSV helpers.
  quarantine  pre-search quarantine of a deep (cell, Stokes)'s earlier archive search frames (ops/p3s5_quarantine_frames.py
              run in place; plan evaluated here). Public API used by `search`: plan_frame_quarantine, quarantine_for_search.
  sync        archive frames = the published attempt's frames (ops/p3s5_archive_sync.py imported in place), retire-refused.
  retire      retire-excluded: scan with core.exclusions, plan, apply, retirement audit; sub-LST / snapshot report only.
  push        push --class mosaic (ops/push_mosaic_v2_20260929.py procedure) and push --class epoch (per-hour push of one
              epoch's products, bulk pushes of many epochs); audit of an earlier push manifest.
  snapshot    supersede (ops/snapshot_frametime_ship.sh in place), quarantine-rows (ops/snapshot_quarantine_wideband_fits.py
              in place).
  rollback    put quarantined files back from their manifest (restore_refused_frames.py generalised).
  tests       content tests of the product classes this step writes (discovered by core/validate).
"""
