# The forced targets' wide-area burst test and the night-activity table

Two inputs of the astrometry figure v2 (exopipe_ops `source_astrometry_panel_v2.py`, `astrometry_assoc.py`; live on
2026-11-03 for the forced runs of every tier and for the deep tier) were made by analysis scripts, not by pipeline steps.
This spec covers the two `exopipe3 forced --tier snapshot` families that now make them:

| Family | Stages | Product the figure reads | Ported from (`phase3/refactor_analysis/review_items_20261031/`) |
|---|---|---|---|
| W | `wide_jobs`, `wide_search`, `wide_test`, `wide_publish` | `<forced run>/forced_wide_v3.csv` | `forced_wide/forced_wide_jobs.py`, `forced_wide_search.py`, `forced_wide_analyze.py` (v2 centering), `v3/jobs_ext.py`, `v3/forced_wide_v3.py`, `v3/publish_v3_to_runs.py`; the ambiguous-target list `forced_ambiguous.py` (2026-11-01) |
| N | `activity` | `night_activity_I10s.csv`, `night_activity_nights_I10s.csv` | `figure_v2/night_activity_build.py` |

Code: `exopipe/phase3/lib/forced/wide_stages.py` (stages, gates, records), `wide_jobs.py`, `wide_search.py`,
`wide_v3.py`, `night_activity.py`, `tests_wide.py` (content tests). Config: sections `forced_wide`,
`forced_wide_search`, `forced_wide_test` and `night_activity` in `exopipe/phase3/config/steps/forced.py`.
Tests: `tests/test_forced_wide.py`.

Both families are campaign stages: they run over whole runs, have no per-hour form, run only when named
(`--stages`), and one family runs per invocation.

## Approval

Gregg approved the v3 method on 2026-11-03: "v3 is not perfect, but it is approved. Proceed." The method is the one in
the `forced_wide_v3.py` docstring. Its values are the config defaults.

No `fix_ledger` row names this approval yet, so no key is marked `RULED`. `tests/test_core_v2.py` requires every RULED
ID to be a ledger row. When a row is added, its ID can be placed in the sections' `RULED`.

## Family W: which cataloged source, if any, makes a forced target's events?

Forced photometry picks frames by the flux at the target. A background source that scintillates or wanders onto the
target can therefore pull the forced events, and the combined position, toward the target. The position test alone
cannot see this. The wide-area test looks at every burst in the target's 5-degree cutouts. It then asks which
cataloged source the bursts at the target belong to.

### Inputs

- **The four forced runs** of the served generation (`forced_wide.RUNS`, named `forced_wide.RUN_NAME`):
  `Run_SNAPFORCED_{I,V}{10s,1min}_gen5` under `paths.SNAPSHOT_REVIEW_ROOT`. These are on lwacalim07's /data07, so
  `--wide-runs DIR` can read a copy instead. The test reads each run's `queue_<S>.txt`, `forced_assoc_events.csv`,
  and `candidates/*/*/detections.csv`.
- **The cutouts** the forced photometry measured: `photometry.CUTOUT_ROOT`, which is gen5. `--wide-cutout-root`
  overrides it, but a retired generation is refused.
- **In-place production modules, called in the step process:**
  - `snapshot_forced_eval`: the evaluation's domain gate `gate_mask`, the a-priori screen `zone_flags`, and the noise
    medians;
  - `phase3_snapshot_search`: time, Sun and night rules;
  - `phase3_tier_rules.deep_search_refused`: tier rule 3;
  - `snapshot_forced_assoc`: the VLSSr, TGSS and GLEAM catalogues at `forced_assoc.RADIO_CATDIR`, and `build_targets`;
  - `phase3_snapshot_astro.beam_fwhm_arcmin`: the snapshot frame beam.
- **Exclusions** come from `core.exclusions`.

### `wide_jobs` → `jobs.json` (`forced_wide_jobs`), `events.csv` (`forced_wide_events`)

- **Ambiguous targets.** Each listed identity of a run is placed at its current list position. Its nearest cataloged
  entry ≥ `AMB_MIN_S74_JY` (0.5 Jy at 74 MHz) is searched for within `AMB_SEARCH_BEAMS` (3) Master beams. The beam is
  taken at the median frequency of the target's forced events.
- **Base jobs.** These are ambiguous targets whose entry lies within `BASE_MAX_BEAMS` (1) beam. The background S is the
  nearest *physical* source ≥ 0.5 Jy within 1.01 beams. Entries within `GROUP_ARCMIN` (1.5′) count as one physical
  source, with VLSSr's flux where VLSSr lists it. The job's bands are the subbands with forced events of that Stokes.
- **Extension jobs** (v3, after Gregg's KELT-24 note). These are the other targets with events that have a physical
  source ≥ 3 Jy within 2 × the FWHM of their lowest event band.
- **Events.** One row per (identity, frame id) of the runs' subband detections that have a frame sigma. Each row holds
  the S/N and frame sigma as written. The analysis computes value = S/N × sigma.

`--wide-jobs FILE[,FILE]` adopts job lists that the analysis wrote, in place of selecting them. This is a replay, used
for parity.

### `wide_search` → `bursts/<S>_<identity>_<band>.npz` (`forced_wide_bursts`)

For each (Stokes, target, band), every cutout of the target is searched. A cutout is one per LST-hour cell: a 300′
residual cube (frame minus the hour median) on the cell's fixed grid.

- **Frames.**
  - 10 s: the frames that were admitted and that pass the evaluation's domain gate at the target. The gate covers
    elevation ≥ 30°, the A-team, the night rule, and the Sun and Jupiter.
  - 1 min: gen5's blocks. These are admitted frames of one minute, at least `BLOCK_MIN_FRAMES` (3), averaged and gated
    at the block time. `MINUTE_RULE = 'filename'` keys them by the file-name minute, which is what gen5's stage 1
    used.
  - A band excluded for that (LST, night), or refused by tier rule 3, is skipped.
  - The target's a-priori screen (zone or clean) is recorded per file and tier.
- **Peaks.** Local extrema of |value| over 3 × 3 pixels, of either sign. Candidates are kept at ≥ 4 × the plane's
  robust rms. Then S/N = value / the forced definition's local rms. That rms is 1.4826 × the MAD of the 41 × 41 stamp,
  with the central 5 × 5 excluded. A peak is kept at |S/N| ≥ `SNR_MIN` (7).
- **Positions.** A 2-D quadratic in log|value| over the 3 × 3 pixels, then the cutout's WCS.

The npz holds the analysis' members (rows of peaks, a per-file table, and `n_files`). It drops the analysis'
wall-clock `seconds`, which goes into the unit record instead. A re-run is therefore byte-identical.

A cutout that raises refuses its band (`FILE_ERRORS = 'refuse'`, W-FILE). Production dropped the file and recorded the
error (`'drop'`).

Processes are capped at `MAX_WORKERS` (6). Before each cutout read, a worker waits while the recorders' I/O guard
(`IO_GUARD_LOG`) shows a node above 4 s. It resumes after two rounds at or below 4 s. This is the rule of
`yield_forced_wide.sh`, applied cooperatively.

### `wide_test` → `results_v3.csv` (`forced_wide_results`), `candidates_v3.csv` (`forced_wide_candidates`)

The v3 method runs per (Stokes, tier, target). It uses the clean bursts: those from files whose status is ok, whose
zone is not 1, and that have gated planes. For Stokes I it uses positive peaks only; for Stokes V, either sign.

1. **Candidates.** Every physical source ≥ 0.25 Jy within `R_CAND` (2) FWHM of the target, in a band with events. A
   source is *capable* in a band when it can make at least half of the events there. That requires
   v ≤ `MU` × F, with F = S74 (74/ν)^0.7 and `MU` set as follows:

   | Stokes | 10 s | 1 min |
   |---|---|---|
   | I | 4.0 | 2.5 |
   | V | 2.5 | 2.0 |

2. **Taper test** (a capable candidate at ≥ 0.6 FWHM).
   - Bursts in a ±0.35 FWHM strip along the line from the candidate through the target are binned every 0.1 FWHM.
   - The near window (0.35–0.65 FWHM before the target) and the far window (0.35–0.65 FWHM beyond it) are joined
     log-linearly and integrated over the target's 0.35 FWHM disc. This gives n_pred.
   - The prediction is calibrated on **control lines**. Each control line uses a cataloged source of 0.5–2 × the
     candidate's flux (≥ 3 Jy) in any job of the same Stokes. Imaginary targets are placed at the candidate's distance
     in 8 directions, up to 900 lines in all. The lines must have comparable predictions.
   - From these: κ = Σn_obs / Σn_pred, λ = κ n_pred, ratio = (n_obs + ½)/(λ + ½), and
     p = (lines with a ratio ≥ the target's + 1)/(lines + 1).
   - The taper **accounts** for the count when p ≥ 0.05 and ratio ≤ 2. It **falls short** when p ≤ 0.05 and
     ratio ≥ 3.
3. **Centering** (a capable candidate within 0.6 FWHM): v2's test. It finds where the bursts in the pair's region
   center on the candidate → target line, compared with pooled flux-matched controls.
4. **Centroid** of the bursts in the target's disc. The error is √(stat² + max(0.028, 0.11/√nights)²), with stat from
   a bootstrap over nights. z = offset/error.
5. **Verdict.**

   | Centroid z | Verdict |
   |---|---|
   | ≥ 3 | BACKGROUND SOURCE (named, with evidence) or NOT CENTERED ON THE TARGET (source not identified) |
   | < 2 | UNDECIDED (centered on the target and on a cataloged source alike), NOT EXPLAINED BY A CATALOGED SOURCE (centered on the target), or UNDECIDED |
   | between | BACKGROUND SOURCE, CONSISTENT WITH A BACKGROUND SOURCE, or UNDECIDED |

   - Fewer than 3 bursts at the target gives UNDECIDED.
   - With no capable candidate, the verdict is NOT CENTERED or NOT EXPLAINED "(no cataloged source can make the
     events)".
   - Evidence that a candidate makes the bursts can be: the bursts center on it; v2's centering; night by night
     (Spearman ρ > 0, p < 0.01); or its taper accounts for the count.

Each job has its own numpy generator (seed `SEED` = 20261102). The results therefore do not depend on the number of
processes.

### `wide_publish` → `<review root>/<run>/forced_wide_v3.csv` (`forced_wide_published`)

Each run gets the results rows of its (Stokes, cadence). A `named_radec` column holds `ra,dec` (5 decimals) of each
named source, `;`-joined. A named source is found by name among the physical sources within 200′ of the target.

The file is written as `.part`, content-tested, and then renamed over the served file. A partial test (`--wide-only`) is
never published (W-PARTIAL).

### What the figure consumes

`source_astrometry_panel_v2.load(U)` reads `U/forced_wide_v3.csv` into `{identity: row}`. Then
`astrometry_assoc.forced_v3_verdict` uses these columns: `verdict`, `named`, `why` (its `[evidence: ...]` part),
`centroid_arcmin`, `centroid_z`, `centroid_nights` and `bursts_at_target`. Two verdicts map through `V3_MAP`:

- BACKGROUND SOURCE → CLEARLY A BACKGROUND SOURCE;
- NOT CENTERED ON THE TARGET (source not identified) → NEITHER THE CROSS-MATCH NOR A KNOWN BACKGROUND SOURCE.

Any other verdict keeps the position verdict, and the burst test's result is added to the reason. `named_radec` marks
the named sources.

### Deliberate differences from the analysis, and production's values

These are PROPOSED (2026-11-03). Production's values are in `exopipe/phase3/parity/production_20261013.cfg` (`# WAVE2`).

| Key | Default | Production | Why |
|---|---|---|---|
| `forced_wide.EVENT_ORDER` | `sorted` | `directory` | The order the events are read is the order a target's bands are listed and summed in. The analysis read `os.listdir` order of its Lustre copy (`v3/events_copy/`). Lustre lists a directory in a fixed hash order of its names, so a copy on Lustre lists the same way; /data07 does not. |
| `forced_wide.TARGET_MEMBER` | `identity` | `first` | A merged identity (26 in the four gen5 runs) takes the list position of one member. Production took the first member in `build_targets`' glob order on /data07; a copy lists in another order. The searched cutouts are the identity's own, centered on the member it is named after (their TARGRA/TARGDEC). On gen5, `first` put 6 of 205 jobs 0.1–0.4′ off that point, and swapped two identities' positions. `identity` centers each job on its cutouts. Through the pooled controls this changes one verdict: Struve 2398 A, I 1 min, where v2's centering power − z goes from 3.12 to 2.89 against the threshold of 3, so BACKGROUND SOURCE becomes NOT CENTERED ON THE TARGET (source not identified). |
| `forced_wide_search.FILE_ERRORS` | `refuse` | `drop` | Gates refuse; production dropped a raising cutout. No gen5 cutout raised. |

The night-activity bands table lists the nights of each (LST, band) in date order. The script iterated a set there, so
its rows came in string-hash order, which differs in every process. No value depends on this order.

### Not ported

- The analysis' figures (`figs_v3/`) and its review page (`forced_wide_page_v3.py`): nothing in the pipeline or the
  figure reads them.
- The magnification-bound measurement (`v3/mag_bound.py`, `mag_bound_V.py`): its result is the approved constant `MU`.
  `mag_bound_V.json` is empty; the V values came from the measurement the v3 docstring names.
- v1's same-cutout control pairs of the base jobs (`controls`, `mid`, `r_max_arcmin`, `bg_not_isolated` in
  `jobs.json`): no stage reads them.
- The diagnostics `kelt24_evidence.py`, `lag_test*.py` and `taper_quick.py`.

## Family N: the night-activity table

### `activity` → `night_activity_I10s.csv` (`night_activity_bands`), `night_activity_nights_I10s.csv` (`night_activity_nights`)

**Input.** One row per detection of every *listed* source of the served snapshot 10-s Stokes I view. This is the flood
rule's input, `flood_nights_20261031/flood_collect.py`, which builds the page as the two-port server does
(`Run_SNAPCOMB_20261026`, page p8987). The default is `night_activity.DETECTIONS`, and `--na-detections` overrides it.
Rows of another Stokes or tier are refused (N-INPUT).

**Method.** Detections are de-duplicated by (frame id, the position's first 9 characters).

- Per (LST, night, band): the count, divided by the median count of that band's *other searched* nights. A band with
  no detection that night is not searched there; it is never treated as quiet.
- Per (LST, night): the activity is the median over its searched bands, and its rank among the LST's nights
  (1 = most active).

**Output.** The output is written to `night_activity.OUT_ROOT`, which is where the figure reads it
(`astrometry_assoc.NIGHT_ACT`, with the nights table beside it). Rebuild it from each new combined view:

```
exopipe3 forced --tier snapshot --stages activity --na-detections <its table>
```

**Not ported:** the collect, `flood_collect.py`. It belongs to the snapshot hides (the flood rule), and it loads the
served page through the two-port server.

## Running it

**Family W, production.**

1. `wide_jobs` and `wide_publish` run on the review host, lwacalim07, where /data07 holds the runs:

   ```
   exopipe3 forced --tier snapshot --stages wide_jobs
   ```

2. `wide_search` and `wide_test` run on any node that reads Lustre:

   ```
   exopipe3 forced --tier snapshot --stages wide_search,wide_test --workers 6
   ```

3. Publish on the review host:

   ```
   exopipe3 forced --tier snapshot --stages wide_publish
   ```

**Family W, parity** (scratch roots only):

```
exopipe3 forced --tier snapshot --stages wide_jobs --wide-runs <Lustre copy> \
    --wide-jobs <analysis>/jobs.json,<analysis>/v3/jobs_ext.json \
    --out <scratch> --parity --config exopipe/phase3/parity/production_20261013.cfg
```

Then run `wide_search --wide-only I:KELT-24,...`, `wide_test --wide-bursts <analysis>/bursts_gen5` and `wide_publish`
into `<scratch>/review/<run>`, which must exist. The comparison tools and results are under
`phase3/refactor_analysis/forcedwide_parity_20261103/`.

**Family N:**

```
exopipe3 forced --tier snapshot --stages activity [--na-detections FILE]
```
