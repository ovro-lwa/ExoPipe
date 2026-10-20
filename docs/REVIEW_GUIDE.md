# Reviewer guide - ExoPipe v2 (Phase 3 refactor)

This branch (`v2`) is the refactored pipeline, published for review while the refactor finishes. The `main` branch is the
earlier v1 release of Phases 1 and 2 and is unchanged. Feedback: open an Issue on this repository.

## What changed
Phase 3 (co-addition, transient searches in four independent tiers - deep 1-hour, 10-minute, 1-minute and 10-second snapshots -
catalogs, review pages, forced photometry, follow-up) grew as hundreds of dated scripts. v2 rebuilds it as one command with
one step per stage:

    exopipe3 <step> --help        # python -m exopipe.phase3.cli <step> --help

with the steps masters, flow, dewarp, install, frames, search, catalog, forced, pages, serve, archive, mosaic, photometry,
followup, jupiter, status, calibrate, curves, verify and run.

| directory | contents |
|---|---|
| exopipe/phase3/steps/ | one module per command: arguments, refusals, units of work, records |
| exopipe/phase3/lib/<domain>/ | the science and plumbing each step calls (masters, search, catalog, serve, ...) |
| exopipe/phase3/core/ | shared machinery: inputs resolver, run records, exclusions, environment |
| exopipe/phase3/config/ | every operating value in one place, each with the ruling it carries |
| tests/ | unit, content and parity tests (parity = byte-for-byte against the production outputs) |

Site paths (cluster roots) come from a site file named by EXOPIPE3_SITE; many modules still name cluster paths directly, and these
are being moved to the site file.

## Status (2026-10-20)
Each stage replaces production only after its parity run: same inputs, same products, byte for byte, except where a recorded
ruling changes a result.
- Parity passed: deep masters (admission, co-add, Stokes V clean, flux re-bake, primary-beam maps), alignment, deep search,
  catalog stages (merge, conform, associate, select), the two-port review server, archive (sync, retire-excluded, audit,
  rollback).
- Parity owed: dewarp, install, parent catalogs, forced photometry, mosaics and photometry, and the 10-minute and 10-second
  tiers, which follow the production runs now in progress.
- Known open items: the library is split into more modules than needed and will be consolidated after this review; some
  production fixes are still being folded in.

## Not in v2 yet
The calibration pipeline (its own repository today) and Phases 1 and 2 (v1 on `main`) will join v2 after the Phase 3
refactor, together with ORCA-based orchestration of all phases.
