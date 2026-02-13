# ExoPipe

**Extrasolar Space Weather Pipeline for the OVRO-LWA**

ExoPipe is an automated radio imaging and science extraction pipeline developed at Caltech for the [Owens Valley Radio Observatory Long Wavelength Array](https://www.ovro.caltech.edu/main/lwa.html) (OVRO-LWA). It processes calibrated visibility data across 15 subbands (1882 MHz), producing science-ready images, targeted photometry, blind transient searches, and solar system body tracking. This pipeline will be ported to a Celery-based implementation (Orca) as the primary science data reduction framework. This Slurm version is primarily used for development of the pipeline.

---

## Architecture

```
+-------------------------------------------------------------------------+
¦                        pipeline_controller.py                          ¦
¦         LST scheduling · Pre-flight audit · Slurm orchestration        ¦
¦                                                                        ¦
¦  Input: --range "10h-11h" --date "2024-12-20" --targets *.csv          ¦
¦  Output: 15 Pipe_ worker jobs + 1 Science_Agg dependency job           ¦
+-------------------------------------------------------------------------+
           ¦ Phase 1 (parallel × 15 subbands)         ¦ Phase 2 (after all workers)
           ?                                          ?
+--------------------------+             +------------------------------+
¦   process_subband.py     ¦             ¦   post_process_science.py    ¦
¦                          ¦             ¦                              ¦
¦  Per-subband worker:     ¦             ¦  Science Aggregation:        ¦
¦  Stage 1  Setup         ¦             ¦  1. Flux scale check (QA/)   ¦
¦    Copy, flag, calibrate ¦             ¦  2. Wideband stacking        ¦
¦  Stage 2  Peel          ¦             ¦  3. 3-color PNGs             ¦
¦    TTCal source removal  ¦             ¦  4. Wideband transient search¦
¦  Stage 3  Image         ¦             ¦  5. Wideband solar system    ¦
¦    Concat, QA, wsclean,  ¦             ¦  6. Gather detections        ¦
¦    PB corr, dewarp,      ¦             ¦  7. Email report             ¦
¦    photometry, transient ¦             ¦                              ¦
¦    search, movies,       ¦             +------------------------------+
¦    archive               ¦
+--------------------------+

Supporting modules:
+------------------------------------------------------------------------+
¦  hot_baseline_worker ¦  cutout.py            ¦  transient_search.py    ¦
¦  UV + heatmap        ¦  Target photometry    ¦  Stokes I subtraction   ¦
¦  analysis & flagging ¦  + confusing source   ¦  + Stokes V blind       ¦
¦                      ¦  masking              ¦  search with multi-tier ¦
¦                      ¦  + 10min differenced  ¦  bright source masking  ¦
¦                      ¦  I cutouts            ¦                         ¦
+----------------------+-----------------------+-------------------------¦
¦  solar_system_cutout ¦  flux_check_cutout.py ¦  make_wideband_3color   ¦
¦  Ephemeris-driven    ¦  Scaife & Heald 2012  ¦  Standalone 3-color     ¦
¦  Moon + planet       ¦  flux scale check     ¦  RGB image generator    ¦
¦  tracking & cutouts  ¦  via CASA imfit       ¦                         ¦
+----------------------+-----------------------+-------------------------¦
¦  pipeline_config.py  ¦  pipeline_utils.py    ¦  extractor_pb_75.py     ¦
¦  All tunable params, ¦  Shared helpers       ¦  Source extraction &    ¦
¦  hardware map,       ¦                       ¦  ionospheric warp       ¦
¦  imaging configs     ¦                       ¦  screen generation      ¦
+------------------------------------------------------------------------+
```

### Execution Model

The pipeline runs on the OVRO-LWA computing cluster where each of 11 `lwacalim` nodes stores visibility data for 12 of the 15 subbands (1882 MHz, 4.6 MHz channel width). `pipeline_controller.py` converts an LST range and date into UTC boundaries, divides the observation into LST-hour segments, then submits a Slurm job per subband. Resource allocation (CPUs, memory, wsclean threads) is automatically adjusted for nodes hosting two subbands so both jobs can run concurrently.

Each subband worker runs in three Slurm stages chained by `afterok` dependencies: Setup ? Peel (array job, one task per MS file) ? Image. All computation happens on NVMe (`/fast/`). After all 15 workers complete, a final Science Aggregation job triggers to stack subbands into wideband images and compile results.

### Data Flow

```
Visibilities (MS)          NVMe working dir              Lustre archive
/lustre/.../raw/    --?    /fast/gh/main/{lst}/    --?   /lustre/gh/main/
  per node                   {date}/{run}/{sb}/            +-- {lst}/{date}/{run}/{sb}/
  15 subbands                +-- I/deep/                   ¦   +-- I/, V/, QA/, Movies/
                             +-- V/deep/                   ¦   +-- Dewarp_Diagnostics/
                             +-- I/10min/                  +-- samples/
                             +-- V/10min/                  ¦   +-- {sample}/{target}/{sb}/
                             +-- samples/                  +-- detections/
                             +-- detections/                   +-- transients/{I,V}/{J-name}/{sb}/
                             +-- QA/                           +-- SolarSystem/{body}/{sb}/
```

---

## Pipeline Steps  Phase 1 (Per-Subband)

Each of the 15 subbands is processed independently on its host node. Processing is split across three Slurm stages. All work happens on NVMe; Lustre is only touched at the final archive step.

### Stage 1  Setup (`run_stage_setup`)

**1.1  Data Discovery and Copy.** The pipeline locates archive measurement sets (MS) for the target subband and UTC time range on Lustre, then copies them to the NVMe working directory `/fast/gh/main/{lst_label}/{date}/{run_label}/{subband}/`. The 13 MHz subband is known to be unusable and is skipped automatically.

**1.2  Antenna Flagging.** Each MS file is checked against the MNC (Monitor aNd Control) antenna health database. Antennas with known hardware faults are flagged. This step runs in a separate conda environment (`development`) that has access to the MNC Python API.

**1.3  Bandpass and XY-Phase Calibration.** Pre-computed bandpass and XY-phase calibration tables are applied to each MS file via `pipeline_utils.apply_calibration()`. The tables are generated offline from dedicated calibrator observations and passed via `--bp_table` and `--xy_table`. MS files that fail calibration (e.g., due to corrupted data or missing spectral windows) are removed from the processing list.

**1.4  State Persistence.** The pipeline saves its full argument state to a pickle file in the working directory. This enables later stages (Peel, Image) and resume modes to reconstruct the original configuration without re-parsing command-line arguments.

**1.5  Submit Peel + Image.** The Setup stage submits a Slurm array job for peeling (one array task per MS file, throttled to 4 concurrent tasks) and a single Image job with an `afterok` dependency on peeling completion. Both are pinned to the same node. The Image job ID is written to a shared dependency file on Lustre so the Science Aggregation job can wait on all 15 subbands.

### Stage 2  Peel (`run_stage_peel`)

**2.1  Sky Source Peeling (optional, `--peel_sky`).** Bright sky sources (Cas A, Cyg A, Tau A, Vir A, and others defined in `sources.json`) are subtracted from the visibilities using TTCal, a Julia-based direction-dependent calibration and subtraction tool. TTCal runs in its own conda environment (`julia060`) because it depends on a specific Julia runtime. The peeling uses a constant beam model with a minimum UV distance of 5 wavelengths and iterates up to 5 times to a tolerance of 10?4.

**2.2  RFI Source Peeling (optional, `--peel_rfi`).** Known RFI sources (satellites, terrestrial transmitters) are subtracted using the same TTCal framework with a separate source model (`rfi_43.2_ver20251101.json`). This runs in a different conda environment (`ttcal_dev`) to accommodate version differences.

Each MS file is peeled independently as a Slurm array task. Peeling is the most time-consuming pre-processing step.

### Stage 3  Image + Science (`run_stage_image`)

This is the main processing stage. It concatenates the peeled MS files, performs quality analysis, images the data, then runs all science extraction. In `--resume_science` mode, steps 3.13.7 are skipped entirely and the pipeline jumps straight to the science phase using images from a prior run.

**3.1  Concatenation.** All peeled MS files for the subband are concatenated into a single measurement set using `pipeline_utils.concatenate_with_auto_heal()`. This function handles edge cases like mismatched spectral windows or missing columns by attempting automatic repair. After successful concatenation, the individual peeled MS files are deleted to reclaim NVMe space (unless `--skip-cleanup` is set).

**3.2  Phase Center and RFI Flagging.** The phase center of the concatenated MS is set to the LST-hour meridian at the OVRO latitude (e.g., `10h30m00s +37d12m57s` for the 10h LST block) using `chgcentre`. Then AOFlagger runs with a custom OVRO-LWA-optimized Lua strategy (`LWA_opt_GH1.lua`) to flag remaining RFI.

**3.3  Pilot Snapshot QA.** A fast dirty image (no CLEAN, no deconvolution, `-niter 0`) is made for every 10-second integration in both Stokes I and V at 4096×4096 pixels with 1.875' resolution. These pilot snapshots serve as a quality gate: the Stokes V snapshots are analyzed with `pipeline_utils.analyze_snapshot_quality()`, which computes the RMS of each frame and flags integrations where the RMS deviates by more than 5s from the running median. A diagnostic plot of RMS-vs-time with flagged integrations highlighted is saved to `QA/`. Flagged integrations are then removed from the concatenated MS via `pipeline_utils.flag_bad_integrations()`.

**3.4  Hot Baseline Removal (optional, `--hot_baselines`).** A two-pass analysis identifies and flags baselines or entire antennas with anomalous correlations:

*Pass 1  Amplitude vs. UV distance:* The median visibility amplitude is computed in a rolling window (default 100 baselines, configurable via `uv_window_size`) as a function of UV distance. Baselines whose amplitude exceeds 7s above this local envelope are flagged. The window size is kept small to track the natural taper of the visibility function rather than averaging over it.

*Pass 2  Cross-polarization heatmap:* An antenna-by-antenna matrix of median cross-polarization (XY, YX) amplitudes is constructed. Baselines exceeding 5s are flagged. If more than 25% of an antenna's baselines are flagged, the entire antenna is flagged. The heatmap is annotated with hardware provenance (ARX board, SNAP2 board, correlator number) from the full `SYSTEM_CONFIG` wiring table to aid fault diagnosis.

Both passes produce diagnostic plots saved to `QA/`. All flagging operates on the `CORRECTED_DATA` column and is applied to the MS in-place with `casacore`.

**3.5  Science Imaging (wsclean).** Seven imaging configurations are run, producing a total of 19 images per subband (7 deep + 12 ten-minute intervals):

| Image | Stokes | Duration | Weighting | Taper | CLEAN | Purpose |
|-------|--------|----------|-----------|-------|-------|---------|
| I-Deep-Taper-Robust-0.75 | I | 1 hr | Briggs -0.75 | Inner Tukey 30 | 500k iter, multiscale | Primary science (max resolution) |
| I-Deep-Taper-Robust-0 | I | 1 hr | Briggs 0 | Inner Tukey 30 | 500k iter, multiscale | Transient reference |
| I-Deep-NoTaper-Robust-0.75 | I | 1 hr | Briggs -0.75 | None | 150k iter, multiscale | Wideband 3-color (Red band) |
| I-Deep-NoTaper-Robust-0 | I | 1 hr | Briggs 0 | None | 150k iter, multiscale | Wideband 3-color (Green/Blue) |
| I-Taper-10min | I | 6 × 10 min | Briggs 0 | Inner Tukey 30 | 50k iter, multiscale | Transient search intervals |
| V-Taper-Deep | V | 1 hr | Briggs 0 | Inner Tukey 30 | Dirty (0 iter) | Circular polarization |
| V-Taper-10min | V | 6 × 10 min | Briggs 0 | Inner Tukey 30 | Dirty (0 iter) | V interval search |

All images are 4096×4096 pixels at 1.875'/pixel (0.03125°), imaging horizon to horizon. Stokes I uses multiscale CLEAN with a scale bias of 0.8, auto-threshold of 0.5s, auto-mask at 3s, and local RMS estimation. Stokes V images are dirty (no deconvolution) because astrophysical circular polarization is extremely rare and faint  deconvolution would just fit noise. The inner Tukey taper suppresses short-spacing sidelobes by down-weighting baselines shorter than ~30 wavelengths. The 10min images use `-intervals-out 6` to produce 6 intervals spanning the 1-hour observation. Timestamps from the MS are written into the FITS headers of all output images.

**3.6  Primary Beam Correction.** The OVRO-LWA primary beam model developed by Nivedita Mahesh (`OVRO-LWA_MROsoil_updatedheight.h5`, a frequency-dependent electromagnetic simulation assuming MRO soil properties) is applied to every science image to produce `*pbcorr.fits` files. The beam model is evaluated at the image frequency and interpolated onto the image pixel grid. PB-corrected images are used for photometry (target and solar system cutouts) because they give correct flux densities across the field of view.

**3.7  Ionospheric Dewarping.** The ionosphere introduces position-dependent shifts at low frequencies that smear sources and corrupt astrometry. The dewarping procedure:

1. **Source extraction:** Point sources are extracted from the deep Stokes I tapered image using `extractor_pb_75.py`.
2. **Cross-match against VLSSr:** Extracted sources are matched against the VLSSr catalog (74 MHz; Cohen et al. 2007) within a 5' match radius. Only sources with a signal-to-noise ratio above 5 are used.
3. **Warp screen computation:** The RA and Dec offsets between extracted and catalog positions are used to fit a 2D interpolated warp screen across the image. The screen captures the large-scale ionospheric refraction pattern.
4. **Application:** The warp screen is applied to *all* images in the working directory  both PB-corrected (for photometry) and raw (for transient search). Each image is resampled using the inverse warp to produce `*_dewarped.fits`.

Diagnostic plots showing the offset vectors, warp screen, and residuals are saved to `Dewarp_Diagnostics/`. If dewarping fails (e.g., too few matched sources), processing continues with undewarped images.

**3.8  Target Photometry.** For each target CSV file specified via `--targets`, the pipeline loads the target list and extracts Stokes I and V cutouts from the dewarped, PB-corrected images. The cutout pipeline (`cutout.py`) processes each target as follows:

*Image selection priority:* Dewarped PB-corrected images are preferred. If unavailable, raw PB-corrected images are used. If neither exists in the science output directory, the pipeline falls back to the original working directory.

*Deep images (1 hour):*
- **Stokes I:** A 2°×2° cutout is extracted. If the target has confusing sources specified in the CSV (known background sources contaminating the beam), pixels within 1 beam FWHM of each confusing source position are masked to NaN before flux measurement. The flux is measured at the catalog position (source flux), the peak within one beam + ionospheric padding (peak flux), and the local RMS in an annulus. If the detection exceeds the fit threshold (5s), a 2D Gaussian fit is attempted via CASA `imfit`.
- **Stokes V:** Measured directly from the deep V image. No confusing source masking is needed because unpolarized background sources do not appear in Stokes V.

*10-minute intervals (6 per observation):*
- **Stokes I:** The deep I reference is subtracted from each 10min interval image to produce a difference image. This removes the static sky, eliminating confusion noise and enabling detection of variable or transient emission well below the traditional confusion limit. Flux is measured on the difference image  no confusing source masking is needed because the subtraction handles it. If the 10min and deep images have mismatched shapes (should not happen for same-subband data), the pipeline falls back to unsubtracted measurement with a warning.
- **Stokes V:** Measured directly on each 10min V image.

*Output per target:* FITS cutouts (I and V), a 2-panel diagnostic PNG (I left, V right) with measured fluxes, peak position, RMS, confusing source markers (red circles, deep I only), and a `{target}_photometry.csv` with one row per image (1 deep + 6×10min = 7 rows per subband). Detection is evaluated per the target's `detection_stokes` mode: `I` (Stokes I only), `V` (Stokes V only), or `both` (either triggers). Targets below 30° elevation at the observation midpoint are skipped.

**3.9  Solar System Photometry.** Ephemerides are computed for the Moon, Mercury, Venus, Mars, Jupiter, Saturn, Uranus, and Neptune at the observation midpoint using Astropy's built-in JPL ephemeris. Bodies above 30° elevation get I+V cutouts extracted from the dewarped PB-corrected images, with FITS headers augmented with geocentric distance, apparent angular diameter, and ephemeris position. The same flux measurement pipeline (peak, source, RMS, optional Gaussian fit) is applied.

**3.10  Transient Search.** The blind transient search operates on tapered, dewarped, non-PB-corrected images (to avoid PB-correction noise amplification at the field edges). Two search modes run independently:

*Stokes V  Blind search (no subtraction):*

Each V image (deep + all 6 10min intervals) is searched directly. The deep Stokes I image is loaded only for bright source masking  sidelobes of very bright unpolarized sources can leak into Stokes V through instrumental polarization. Multi-tier masking of the I image suppresses these: sources above 500s in I are masked to a radius of 2 beams, and sources above 100s to 1 beam. A-team sources (Cas A, Cyg A, Tau A, Vir A, Hydra A) are additionally masked to fixed radii (3°5°). The RMS is computed locally in 32-pixel boxes. Candidates above 5s trigger a search for a bi-lobe partner (the PSF response of a point source produces positive and negative lobes): a second peak above 3s within 30' with a flux ratio between 0.2× and 5×. Both positive and negative detections are reported (corresponding to LCP and RCP emission). Each detection is cross-matched against the input catalog within 2'.

*Stokes I  Subtraction search:*

The deep Stokes I Robust-0 image is subtracted from each 10min Robust-0 interval to produce a difference image. Only the Robust-0 weighting is used for both reference and interval to avoid subtraction artifacts from mismatched PSFs. The difference image is then searched for positive-only transients (negative transients in Stokes I are not physical). Multi-tier masking is applied to the reference I image: sources above 100s masked to 2 beams, above 30s to 1 beam. A-team exclusion zones are larger (10°15°). Detection threshold is 7s with a 4s partner requirement. Cross-matching against the catalog follows the same procedure.

*Outputs per detection:* A 2°×2° FITS cutout, diagnostic PNG showing the detection and its context (reference + subtracted for I; I + V for V), with reticle at the detection position and measured flux annotated. Filenames use truncated J-names (`J0553+31` format) for directory grouping, with the full J-name in the filename. A maximum of 5 detections per subband per Stokes prevents runaway cutout generation from bad data. If more than 10 total candidates are found across both Stokes, a quality warning is logged.

**3.11  Movies.** MP4 animations are generated from the pilot images (the dirty per-10-second-integration images from step 3.3). For Stokes I, two movies are produced: a raw movie (percentile-stretched) and a median-subtracted filtered movie that highlights time-variable emission (scaled to -3s to +5s). For Stokes V, only the raw movie is produced (scaled to ±5s). Movies are limited to 150 frames and encoded at 10 fps with ffmpeg.

**3.12  Archive to Lustre.** The final step copies products from NVMe to permanent storage on Lustre. The archive is split into three tiers:

*Run-local products*  Images (I/deep, V/deep, I/10min, V/10min, pilot images), QA diagnostics, movies, and dewarping diagnostics are copied to the per-run archive: `/lustre/gh/main/{lst}/{date}/{run_label}/{subband}/`. This preserves the complete imaging record for each observation.

*Centralized samples*  Target photometry products (cutout FITS, diagnostic PNGs, photometry CSVs) are copied to a content-organized archive: `/lustre/gh/main/samples/{sample_name}/{target_name}/{subband}/`. This enables cross-run analysis of individual targets without navigating the per-observation directory tree.

*Centralized detections*  Transient detections go to `/lustre/gh/main/detections/transients/{I,V}/{J-name}/{subband}/`. Solar system detections go to `/lustre/gh/main/detections/SolarSystem/{body}/{subband}/`. Target-triggered detections (from the photometry pipeline's detection flag) are copied to `/lustre/gh/main/detections/{sample_name}/{target_name}/{subband}/`.

---

## Pipeline Steps  Phase 2 (Science Aggregation)

`post_process_science.py` runs once after all 15 subband workers complete, triggered by a Slurm `afterok` dependency on all Image job IDs. It operates entirely on the Lustre archive.

### Step 1  Flux Scale Check

For each calibrator in the Scaife & Heald 2012 catalog (3C48, 3C123, 3C147, 3C196, 3C286, 3C295, 3C380), the pipeline checks whether the source is above 20° elevation at the observation midpoint. For visible calibrators at each subband, a 2° cutout is extracted from the deep tapered PB-corrected Stokes I image and a 2D Gaussian is fitted using CASA `imfit`. The fitted integrated flux is compared to the model prediction using the polynomial log-flux-density coefficients (Perley & Butler 2017 coefficients are used for 3C123).

Results are saved to `QA/flux_check_hybrid.csv` with columns for each calibrator: fitted flux, model flux, ratio, fit uncertainty. A 2-panel diagnostic PNG shows: (1) flux ratio vs. frequency for all calibrators (highlighting the expected ratio of 1.0), and (2) measured vs. model flux as a scatter plot. Systematic flux scale errors (e.g., from bandpass drift) appear as coherent deviations across frequency.

### Step 2  Wideband Stacking

Subband images are co-added into wideband images using inverse-variance weighting (weights derived from the thermal noise measured in each subband image). Three color bands are defined:

| Band | Frequency Range | Subbands |
|------|----------------|----------|
| Red | 1841 MHz | Up to 5 |
| Green | 4164 MHz | Up to 5 |
| Blue | 6485 MHz | Up to 5 |

*Transient search stacks:* Tapered, dewarped, Briggs-0, non-PB-corrected images are stacked for Stokes I and V. Deep images produce one stack per band. The 6 10min intervals are stacked separately (images are grouped by their interval tag `t0001``t0006`), producing 6 stacks per band per Stokes. This preserves time resolution for the wideband transient search. Output filenames follow the pattern `Wideband_{Band}_{Pol}_{cat}_Taper_{Robust}_{interval}.fits`.

*Science stacks:* NoTaper, dewarped (not PB-corrected) images are stacked for Stokes I deep only, in two robust weightings: Briggs 0 and Briggs -0.75. These are used for the 3-color PNGs.

A `thermal_noise.csv` file records the measured noise per subband per image type, used both as stacking weights and as a diagnostic of data quality across the band.

### Step 3  3-Color PNGs

RGB composite images are generated from the three wideband science stacks. Each channel is independently normalized with a percentile stretch [0.5, 99.8]. Three versions are produced:

| PNG | Red (1841 MHz) | Green (4164 MHz) | Blue (6485 MHz) |
|-----|-----------------|-------------------|-------------------|
| Mixed (recommended) | Robust -0.75 | Robust 0 | Robust 0 |
| Uniform Robust 0 | Robust 0 | Robust 0 | Robust 0 |
| Uniform Robust -0.75 | Robust -0.75 | Robust -0.75 | Robust -0.75 |

The mixed version uses higher resolution at the lowest frequencies (where the synthesized beam is largest) and higher sensitivity at higher frequencies, giving a more uniform effective resolution across the composite. The standalone `make_wideband_3color.py` script can regenerate these from any directory of wideband FITS files.

### Step 4  Wideband Transient Search

The same transient search algorithm from Phase 1 is applied to the wideband stacks. Each 10min interval is searched independently using the wideband deep image as reference for subtraction (Stokes I) or masking (Stokes V). Detections at the wideband level have higher sensitivity (vN improvement from stacking ~5 subbands) and are less susceptible to single-subband artifacts.

### Step 5  Wideband Solar System Photometry

Cutouts for solar system bodies are extracted from the wideband stacks, providing broadband flux measurements.

### Step 6  Gather Detections

Three detection paths are scanned and compiled into a summary:

1. **Target photometry detections:** All `*_photometry.csv` files from per-subband and wideband outputs are scanned for rows where `Detection == True`.
2. **Transient candidates:** Files in `detections/transients/{I,V}/` are counted and the first diagnostic PNG from each Stokes is collected.
3. **Solar system detections:** Files in `detections/SolarSystem/` are counted.

### Step 7  Email Report

A summary email is sent with the thermal noise profile, detection list, and diagnostic PNGs attached (flux check diagnostic, 3-color composites, first transient detection PNGs).

---

## Target CSV Format

Target files are standard CSV (or whitespace-separated) with the following columns:

| Column | Required | Description |
|--------|----------|-------------|
| `common_name` | Yes | Target identifier (also accepts `name`, `source`, `id`) |
| `ra_current` | Yes | Right ascension in degrees or sexagesimal (also `ra_deg`, `ra`) |
| `dec_current` | Yes | Declination in degrees or sexagesimal (also `dec_deg`, `dec`) |
| `detection_stokes` | No | What triggers a detection: `I`, `V`, or `both`. Default: `both` |
| `confusing_sources` | No | Known background sources to mask in deep Stokes I. Semicolon-separated `ra,dec` pairs in degrees |

Additional columns (e.g., `distance`, `spectral_type`) are preserved in the photometry CSV output but do not affect processing.

The `confusing_sources` column is relevant only for deep (1-hour) Stokes I images, where the static sky limits sensitivity through confusion noise. For 10min Stokes I, the deep reference is subtracted instead, which removes confusing sources along with the rest of the static sky. Stokes V is inherently free of unpolarized background contamination.

Example:
```csv
common_name,ra_current,dec_current,distance,detection_stokes,confusing_sources
Barnard's Star,269.4462683,4.768204473,1.828,V,
Wolf 359,164.0923699,6.995228998,2.408,V,164.088,7.001;164.092,6.998
UV Ceti,24.768,-17.950,2.680,V,
AD Leo,159.862,20.169,4.966,both,
```

---

## Lustre Archive Structure

```
/lustre/gh/main/
¦
+-- {lst_label}/{date}/{run_label}/{subband}/
¦   +-- I/deep/            # All Stokes I deep images (4 weightings, raw + pbcorr + dewarped)
¦   +-- I/10min/           # 6 × 10min intervals (tapered Robust-0, raw + pbcorr + dewarped)
¦   +-- V/deep/            # Stokes V deep (tapered Robust-0)
¦   +-- V/10min/           # 6 × 10min V intervals
¦   +-- snapshots/         # Pilot dirty images (per-10s-integration I+V)
¦   +-- QA/                # Hot baseline heatmaps, amp-vs-UV plots, integration RMS diagnostics
¦   +-- Movies/            # {freq}_{I,V}_Raw.mp4, {freq}_I_Filtered.mp4
¦   +-- Dewarp_Diagnostics/  # Offset vectors, warp screens, residuals
¦
+-- samples/
¦   +-- {sample_name}/{target_name}/{subband}/
¦       +-- {target}_{freq}MHz_deep_{timestamp}_I.fits       # Deep I cutout (with masking)
¦       +-- {target}_{freq}MHz_deep_{timestamp}_V.fits       # Deep V cutout
¦       +-- {target}_{freq}MHz_10min_t0001_{timestamp}_I.fits # Differenced I cutout
¦       +-- {target}_{freq}MHz_10min_t0001_{timestamp}_V.fits # Direct V cutout
¦       +-- ...through t0006...
¦       +-- {target}_{freq}MHz_*_Diagnostic.png              # 2-panel flux measurement
¦       +-- {target}_photometry.csv
¦
+-- detections/
¦   +-- transients/
¦   ¦   +-- I/{J-name}/{subband}/     # J-name truncated: J0553+31
¦   ¦   +-- V/{J-name}/{subband}/
¦   +-- SolarSystem/{body}/{subband}/
¦   +-- {sample_name}/{target_name}/{subband}/
¦
+-- Wideband/
    +-- Wideband_{Band}_{Pol}_{cat}_Taper_{Robust}.fits
    +-- Wideband_{Band}_{Pol}_10min_Taper_{Robust}_{interval}.fits
    +-- Wideband_I_deep_NoTaper_{Robust}.fits
    +-- Wideband_I_deep_NoTaper_mixed_3color.png
    +-- Wideband_I_deep_NoTaper_Robust-0_3color.png
    +-- Wideband_I_deep_NoTaper_Robust-0.75_3color.png
    +-- thermal_noise.csv
    +-- QA/
        +-- flux_check_hybrid.csv
        +-- flux_check_diagnostic.png
```

---

## Configuration

All tunable parameters live in `pipeline_config.py`:

| Section | Key Parameters |
|---------|---------------|
| **Observatory** | `OVRO_LOC`  EarthLocation (37.240°N, 118.282°W, 1222 m) |
| **Paths** | `PARENT_OUTPUT_DIR` (`/fast/gh/main/`), `LUSTRE_ARCHIVE_DIR` (`/lustre/gh/main/`) |
| **Slurm resources** | Auto-computed from node sharing; `get_image_resources(subband)` returns (cpus, mem_gb, wsclean_threads) |
| **Peeling** | `PEELING_PARAMS`  TTCal env names, source models, iteration args |
| **RFI flagging** | `AOFLAGGER_STRATEGY`  path to OVRO-LWA-optimized Lua strategy |
| **Hot baselines** | `HOT_BASELINE_PARAMS`  s thresholds (UV: 7, heatmap: 5), UV window size (100), antenna threshold (25%) |
| **Calibrators** | `CALIB_DATA`  Scaife & Heald 2012 / Perley & Butler 2017 polynomial coefficients |
| **Imaging** | `SNAPSHOT_PARAMS` (pilot dirty), `IMAGING_STEPS` (7 science imaging configs) |
| **Beam model** | `BEAM_MODEL_H5`  OVRO-LWA EM beam simulation (Nivedita Mahesh) |
| **Reference catalogs** | `VLSSR_CATALOG`  VLSSr catalog for ionospheric dewarping |
| **Hardware** | `SYSTEM_CONFIG`  full 352-antenna ? ARX ? SNAP ? correlator mapping (verified Feb 2026) |

---

## Usage

### Full Pipeline Run

```bash
python pipeline_controller.py \
    --range "10h-11h" \
    --date "2024-12-20" \
    --bp_table /lustre/gh/calibration/bandpass/2024-12-20_bp.ms \
    --xy_table /lustre/gh/calibration/xyphase/2024-12-20_xy.ms \
    --targets 10pc_sample.csv OVRO_LWA_Hot_Warm_Jupiters_2026.csv \
    --catalog OVRO_LWA_Local_Volume_Targets.csv \
    --hot_baselines \
    --peel_sky \
    --peel_rfi
```

### Resume Science-Only (reprocess targets without re-imaging)

```bash
python pipeline_controller.py \
    --range "10h-11h" \
    --date "2024-12-20" \
    --run_label "Run_20260209_171717" \
    --targets 10pc_sample.csv OVRO_LWA_Hot_Warm_Jupiters_2026.csv \
    --catalog OVRO_LWA_Local_Volume_Targets.csv \
    --resume_science
```

### Standalone 3-Color RGB

```bash
python make_wideband_3color.py /lustre/gh/main/.../Wideband/
python make_wideband_3color.py /lustre/gh/main/.../Wideband/ --output /tmp/pretty/
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| Python 3.8+ | Runtime (conda: `py38_orca_nkosogor`) |
| python-casacore | Measurement set I/O (`casacore.tables`) |
| casatasks | Gaussian fitting (`imfit`, `imstat`) |
| wsclean | Radio interferometric imaging |
| AOFlagger | RFI flagging |
| TTCal (Julia) | Direction-dependent source peeling |
| astropy | FITS, WCS, coordinates, ephemerides, sky catalogs |
| numpy, scipy, pandas | Numerical, signal processing, tabular data |
| matplotlib | Diagnostics, cutout PNGs, 3-color composites |
| ffmpeg | Movie encoding |
| Slurm | Job scheduling and dependency management |

---

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for full terms.

GPLv3 ensures that ExoPipe and any derivative works remain open source  modifications and extensions must also be distributed under GPLv3. This is appropriate for publicly funded scientific software where openness and reproducibility are paramount.

---

## Acknowledgments

ExoPipe is developed at Caltech for the OVRO-LWA. The OVRO-LWA is a project of the Caltech Owens Valley Radio Observatory. This work is supported by the National Science Foundation under grant No. AST-1828784, the Simons Foundation (668346, JPG), the Wilf Family Foundation and Mt. Cuba Astronomical Foundation.
