# ExoPipe

Pipeline for coadded, per-epoch and mosaiced image products from the continuous all-sky snapshot imaging of the Owens Valley
Radio Observatory Long Wavelength Array (OVRO-LWA), together with untargeted transient searches and forced photometry on populations of interest (exoplanets, stars, Long Period Transients (LPTs), solar system planets, etc.). ExoPipe processes data into sidereally co-added images and multi-cadence transient searches in Stokes I and
Stokes V, with a follow-up harness for candidate validation. This repository is pre-release. The v2 tree is undergoing golden-cell validation against the production outputs of the v1 codebase; interfaces and layout may change until that validation is signed off.

## Pipeline overview

Processing is organized into three phases:

- **Phase 1 — calibration and imaging.** Per-subband calibration, peeling
  of bright A-team sources, and snapshot imaging of the interferometric
  data, orchestrated on the OVRO-LWA cluster (see the
  [distributed-pipeline](https://github.com/ovro-lwa/distributed-pipeline)
  framework).
- **Phase 2 — image conditioning.** Ionospheric dewarping and astrometric
  alignment of snapshot images, wideband stacking of subbands, primary-beam
  handling, and assembly of science-ready image sets with standardized
  FITS headers and time semantics.
- **Phase 3 — co-addition and transient search.** Quality assessment and
  exclusion gating; two-pass inverse-variance co-addition into per-LST
  averaged images and combined all-sky mosaics; coadd subtraction and striation removal; source
  detection in the residuals with a gated veto chain; catalog, cutout, and
  dossier products; astrometric association; and follow-up capabilities
  (forced photometry, snapshot re-imaging, periodicity analysis).

Searches run at four tiers, each in both Stokes I and Stokes V:

| Tier | Integration | Images searched |
|---|---|---|
| Deep | 1 hour (one image per LST hour per night) | wideband and subband |
| 10-minute | 10 minutes (six per hour) | wideband and subband |
| 1-minute | 1 minute | wideband and subband |
| 10-second (snapshot) | one 10-second integration | wideband and subband |

Every tier is an untargeted transient search. Forced photometry on populations of interest runs alongside as follow-up.

## Repository layout

```
exopipe/
├── common/        shared contracts: config, products/naming/headers,
│                  astrometry (two-tier association), exclusions, validate
├── phase1/        calibration + imaging drivers
├── phase2/        dewarp, alignment, stacking, primary beam, controllers
└── phase3/        co-add + search capabilities and follow-up subpackage
docs/              architecture and design documents (curated)
reference/         small reference tables
tools/             one-off repair/audit utilities (non-pipeline)
tests/             golden-cell validation + standing import-hygiene test
```

Phase 3 is organized as capability modules (aggregators over their
constituent implementations):

- `phase3.coadd` — two-pass QA'd co-addition into master images
- `phase3.residuals` — master subtraction, striation removal, PSF matching
- `phase3.identify` — source detection and identification in residuals
- `phase3.gates` — detection gate chain and gate ledgers
- `phase3.search_deep` / `phase3.tenmin` — deep and 10-minute search tiers
- `phase3.c50` — completeness characterization
- `phase3.photometry` — cutouts and photometric measurement
- `phase3.assemble` — catalog and product assembly
- `phase3.mosaic` — all-sky mosaic products
- `phase3.qa_status` — QA dashboards and processing status
- `phase3.followup` — candidate follow-up harness (astrometry, forced
  photometry, snapshot imaging, dossiers)

Shared contracts live in `exopipe.common`: every data product is named and
headed through `products`, every position/association flows through the
two-tier `astrometry` module, and exclusion decisions are applied through
`exclusions` — uniformly, in every consumer.

## Configuration

Site- and deployment-specific settings (data roots, cluster resources,
notification settings) are centralized in the configuration modules under
`exopipe/common/`. No science threshold or path is hardcoded in capability
code paths; operational job files and processing ledgers live outside this
repository by design.

## Testing

`tests/test_import_side_effects.py` is a standing repo-wide test that every
module is side-effect-free at import (AST deny-list layer, plus an opt-in
strace verification layer via `EXOPIPE_IMPORT_AUDIT=1`). Golden-cell
end-to-end validation harnesses accompany the v1→v2 migration.

## Attribution

ExoPipe is developed for the OVRO-LWA project at the Owens Valley Radio
Observatory, California Institute of Technology.

Principal Investigator: Gregg Hallinan (Caltech); co-PIs: Andrea Isella (Rice), Bin Chen (NJIT), Judd Bowman (ASU) and Andres Romero-Wolf (JPL). Project Scientist: Marin Anderson (JPL). OVRO-LWA has been supported by the National Science Foundation (NSF), NASA, by OVRO-LWA partner institutions (NJIT, Caltech, Rice, ASU, JPL) and through funding from the Mt. Cuba Astronomical Foundation and the Wilf Family Discovery Fund in Space and Planetary Science. The OVRO-LWA is a pathfinder for radio camera technology developed as part of the DSA project. See the [ovro-lwa](https://github.com/ovro-lwa)
organization for related software.

## License

ExoPipe is licensed under the GNU General Public License v3 (GPLv3); the full text is in `LICENSE`.
