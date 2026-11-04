# The forced targets' wide-area burst test and the night-activity table

Two inputs of the astrometry figure v2 (exopipe_ops `source_astrometry_panel.py`, v2 since 7e6904a on 2026-11-04, with
`source_astrometry_panel_v2.py` as its alias; and `astrometry_assoc.py`; live on
2026-11-03 for the forced runs of every tier and for the deep tier) were made by analysis scripts, not by pipeline steps.
This spec covers the two `exopipe3 forced --tier snapshot` families that now make them:

| Family | Stages | Product the figure reads | Ported from (`phase3/refactor_analysis/review_items_20261031/`) |
|---|---|---|---|
| W | `wide_jobs`, `wide_search`, `wide_test`, `wide_publish` | `forced_wide_v3.csv` in each of the 4 tested forced runs (`Run_SNAPFORCED_{I,V}{10s,1min}_gen5`) | `forced_wide/forced_wide_jobs.py`, `forced_wide_search.py`, `forced_wide_analyze.py` (v2 centering), `v3/jobs_ext.py`, `v3/forced_wide_v3.py`, `v3/publish_v3_to_runs.py`; the ambiguous-target list `forced_ambiguous.py` (2026-11-01) |
| N | `activity` | `night_activity_I10s.csv`, `night_activity_nights_I10s.csv` | `figure_v2/night_activity_build.py` |

Code: `exopipe/phase3/lib/forced/wide_stages.py` (stages, gates, records), `wide_jobs.py`, `wide_search.py`,
`wide_v3.py`, `night_activity.py`, `tests_wide.py` (content tests). Config: sections `forced_wide`,
`forced_wide_search`, `forced_wide_test` and `night_activity` in `exopipe/phase3/config/steps/forced.py`.
Tests: `tests/test_forced_wide.py`.

Both families are campaign stages: they run over whole runs, have no per-hour form, run only when named
(`--stages`), and one family runs per invocation.

## Approval

Gregg approved the v3 method on 2026-11-03: "v3 is not perfect, but it is approved. Proceed." The method is the one in
the `forced_wide_v3.py` docstring. Its values are the config defaults, and the method's keys are RULED by the ledger row
`R-GREGG-20261103-FORCED-WIDE-V3-APPROVED`. That covers the candidates, the taper test and its control lines, the
centroid test, the evidence rules, v2's centering, the burst definition and the job thresholds.

Implementation choices inside the approved method are the refactor chat's, made under Gregg's delegation
(`R-GREGG-20261016-REFACTOR-DECIDES-BY-GOALS`). Two are RULED by `R-REFACTOR-20261104-FORCED-WIDE-CONTROLS-ORDER-INDEPENDENT`:

- `V2_CTL_DRAW`: every eligible v2 control is used;
- `CONTROL_ORDER`: the controls are order-independent.

Its amendment `R-REFACTOR-20261104-FORCED-WIDE-CONTROLS-ORDER-INDEPENDENT-AMENDED-PRECISION-KNIFE-EDGE` ("precision and
honesty, not a choice of draw") RULES the Monte Carlo keys:

- `TAPER_CTL_MAX`: every eligible taper control line (no 900 cap);
- `BOOT_DRAW`, `BOOT_Z_SE`, `BOOT_MAX`: the centroid's bootstrap per target, until its z is precise to 0.02;
- `KNIFE_EDGE_SE`: the knife-edge flag;
- `KNIFE_EDGE_DRAWS`, `KNIFE_EDGE_DRAWS_MAX`, `KNIFE_EDGE_DECIDE`: a flagged row is decided only by Monte Carlo draws of the
  method (20 at a time, up to 100 while undecided; Wilson 95%).

The other three defaults that differ from production are PROPOSED (see the deliberate differences below).

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

- **Ambiguous targets.** Each listed identity of the eight forced runs (`AMBIGUOUS_RUNS`: the four tested runs, then the
  10-minute and 1-hour runs, the order `forced_ambiguous.py` read them in) is placed at its current list position. Its
  nearest cataloged entry ≥ `AMB_MIN_S74_JY` (0.5 Jy at 74 MHz) is searched for within `AMB_SEARCH_BEAMS` (3) Master
  beams. The beam is taken at the median frequency of the target's forced events in that run.
- **Base jobs.** These are ambiguous targets of the four tested runs whose entry lies within `BASE_MAX_BEAMS` (1) beam. The background S is the
  nearest *physical* source ≥ 0.5 Jy within 1.01 beams. Entries within `GROUP_ARCMIN` (1.5′) count as one physical
  source, with VLSSr's flux where VLSSr lists it. The job's bands are the subbands with forced events of that Stokes.
- **Extension jobs** (v3, after Gregg's KELT-24 note). These are the other targets with 10-s or 1-min events that have a
  physical source ≥ 3 Jy within 2 × the FWHM of their lowest event band. A (Stokes, identity) takes its first row of all
  eight runs, as `jobs_ext.py` did, so a target with events that only a 10-minute or 1-hour run lists is found too. On
  gen5 the four tier runs add no job: 55 (Stokes, identity) appear only in their rows, and none has 10-s or 1-min events
  (review F9).
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
     in 8 directions. Every eligible line is used (`TAPER_CTL_MAX` None; production kept at most 900). The lines must
     have comparable predictions. The lines are **order-independent** (`CONTROL_ORDER = 'canonical'`):
     - The pool holds one source per 1/20° cell. Its host is the job with the best score (bands searched, nearest the
       cutout center); a tie goes to the smallest (Stokes, identity), then the source's identity.
     - The sources are taken in the order of a stable hash (sha256) of (seed, the target's identity, the source's
       identity). This is the seeded selection a `TAPER_CTL_MAX` cap applies; without a cap it only fixes the order of
       the sums.
     - Line i of a source lies at (i + uᵢ) × 45°, where uᵢ is a stable hash of (seed, target, source, i).
     - So a moved or added job changes another target's lines only where that job adds lines to its pool: a new source,
       a source it hosts better, or, under a cap, the lines it displaces at the end of the order.
     - Production (`'pool'`, the parity file) shuffled a pool-ordered list with the job's generator and drew one
       rotation per source from it.
   - From these: κ = Σn_obs / Σn_pred, λ = κ n_pred, ratio = (n_obs + ½)/(λ + ½), and
     p = (lines with a ratio ≥ the target's + 1)/(lines + 1).
   - The taper **accounts** for the count when p ≥ 0.05 and ratio ≤ 2. It **falls short** when p ≤ 0.05 and
     ratio ≥ 3.
3. **Centering** (a capable candidate within 0.6 FWHM): v2's test. It finds where the bursts in the pair's region
   center on the candidate → target line, compared with pooled flux-matched controls. By default it uses **every**
   eligible control pair (`V2_CTL_DRAW` None), with host ties broken canonically (`CONTROL_ORDER`).
4. **Centroid** of the bursts in the target's disc. The error is √(stat² + max(0.028, 0.11/√nights)²), with stat from
   a bootstrap over nights (≥ 5 nights; else 0.12/√n). z = offset/error. The bootstrap draws from its own generator per
   (target, tier), seeded by a stable hash of (seed, the target's (Stokes, identity), tier) (`BOOT_DRAW = 'target'`). It
   draws 1000 resamples, then more in blocks of 1000 until the Monte Carlo error of z is at most 0.02 (`BOOT_Z_SE`),
   and of each candidate's z_c, counted at most at 3, or `BOOT_MAX` (10⁶) are drawn. Production drew exactly 1000 from
   the job's generator.
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

No generator is shared between targets. The control lines are stable hashes of identities (`CONTROL_ORDER
'canonical'`), and each (target, tier) bootstrap has its own seed (`BOOT_DRAW 'target'`), so no other job's presence or
position moves a target's results. Production used one numpy generator per job (seed `SEED` = 20261102) for both. The
results also do not depend on the number of processes. `wide_test` refuses more than `MAX_WORKERS` (6) processes (W-WORKERS): its workers read the Lustre burst
files. It pools burst files of one search only: every `W_search` record must have read the current `jobs.json` and the
selected cutout generation (W-POOL).

### Monte Carlo errors, the knife-edge flag (`KNIFE_EDGE_SE` = 4) and the decision by draws (`KNIFE_EDGE_DRAWS` = 20)

Two parts of the method are random draws: the control lines' directions and the centroid's bootstrap. A verdict can
therefore sit on a threshold within the method's own Monte Carlo noise. Each row's `knife_edge` column names every
statistic that **decides** the row and lies within 4 Monte Carlo errors of one of its thresholds. A statistic decides the
row when moving it alone to just below and just above that threshold (the others as they are) changes the verdict or the
set of named sources (`knife_edges` re-runs `verdict_row`). The text reads, for example,
`knife-edge: centroid z 2.97 vs 3, Monte Carlo error 0.02`. A statistic that decides at both of its thresholds is named
once (`... 9.10 vs 2 and 3, ...`).

| Statistic | Thresholds | Monte Carlo error |
|---|---|---|
| centroid z | `Z_ON` 2, `Z_OFF` 3 | z × SE(stat²)/(2 err²); SE(stat²) = sd(q_b)/√B over the resamples, q_b = ((x_b − x̄)² + (y_b − ȳ)²)/2 |
| `centroid z to <candidate>` (z_c) | `Z_C_ON` 2, `Z_C_NEAR` 3 | the same relative error as z |
| `taper p of <candidate>` | `P_CONSISTENT` = `P_EXCESS` = 0.05 | the binomial within each control source: √(Σⱼ kⱼ(mⱼ − kⱼ)/(mⱼ − 1))/n over the n matched lines (source j: mⱼ lines, kⱼ at or above the target's ratio) |
| `taper ratio of <candidate>` | `RATIO_CONSISTENT` 2, `R_EXCESS` 3 | ratio × λ/(λ + ½) × SE(κ)/κ; SE(κ) from the matched lines' residuals n_obs − κ n_pred, their spread within each control source |
| `matched control lines (within 3x / 10x) of <candidate>` | `MATCH_MIN` 15 | the binomial within each control source. Flipped, `matched()` takes another window, calibrated on its lines, or none |

The errors are within control sources because a salt re-draws a line's direction, never its control source: the spread
between sources is no Monte Carlo noise. A within-source estimate of zero (no control source with lines on both sides,
for example no line at or above the target's ratio) is degenerate, not a zero error; the binomial over all lines is used
then. On real rows (`se_validate.py`: 30 independent global salts, every line re-rotated and every bootstrap re-drawn, 26
calibrated candidates, 56 matched-line counts and 3 bootstrapped centroids of 5 rows) the spread over the salts divided by
the error has a median of 0.96 for the centroid z (3 of 3 within a factor of 1.5), 0.89 for p (23 of 26), 1.00 for the
ratio (25 of 26) and 0.95 for the counts (53 of 56). The binomial over all lines alike gives 0.77 (20 of 26) for p and
0.84 (50 of 56) for the counts; its misses all overestimate, by up to 2×. For the centroid alone (`zrow_boot.py`, 150-400
salts per row) the ratio is 0.996-1.026 at 1000 resamples, and the realized error at the adaptive count is 0.016-0.018.

**The flag's limit.** It tests one realization against its own estimated error. A statistic whose distribution straddles a
threshold is flagged in most runs, not in all. On the gen5 run (2026-11-04, every job) the flag marked 5 of the 7 rows that
differ from the served files. Re-run with 20 independent salts each (`flip_check.py`), all 7 flip between draws: the
served outcome comes out in 6-13 of 20, the new one in 7-14. With `KNIFE_EDGE_SE` = 4 the same draws flag all 7, and every
row the salts showed flipping, for 17 flags in 312 rows; with 2 they give 13. Hence the default 4 (the refactor chat,
2026-11-04).

**A flagged row is DECIDED only when the method's own draws decide it** (`KNIFE_EDGE_DRAWS` 20, `KNIFE_EDGE_DRAWS_MAX` 100,
`KNIFE_EDGE_DECIDE` 0.95; the refactor chat's decisions, 2026-11-04). One draw cannot decide such a row, so `analyze_all`
re-runs the approved method itself (8 lines per source, `CTL_DIRS`) for it in Monte Carlo draws.

- **The salts.** Draw k has its own salt in place of `SEED`: a stable hash (sha256) of (`SEED`, the target's (Stokes,
  identity), k). The salts are fixed and recorded (the unit's `draw_salt`), so the result is reproducible and independent of
  the job list's order.
- **Sequential draws.** A row gets 20 draws first. While its verdict class or one of its sources stays undecided, it gets 20
  more, up to 100 (k = 0 .. 99, the same salt for draw k in any batch). "20 draws are too few near 75%."
- **The verdict class** (the verdict string) is published only when the Wilson 95% interval of its fraction of the row's K
  draws lies wholly above one half: 15 of 20, 27 of 40, 38 of 60, 49 of 80, 60 of 100.
- **Otherwise the row is published as** "UNDECIDED (knife-edge: the method's random draws split - <class> in a of K,
  <class> in b of K)". It names no source, keeps the default draw's numbers (`mc_row` = `split`), and leaves `mc_draws`
  empty, because the verdict holds the split.
- **A source is named** only when it is named in a decided fraction of the K draws, in the same sense.
  - One named in an undecided fraction goes to `named_knife_edge` ("knife-edge: VLSSr J... named in 9 of 20 draws"), never
    into `named` or `named_radec`.
  - One at or below the lower line (5 of 20, 13 of 40, 22 of 60, 31 of 80, 40 of 100) is decidedly not named and appears in
    `mc_outcomes` only.
- **The settled case** is a naming class decided with no source decidedly named. It publishes the class, with no named source
  and the sources' split: "<class> (source split: A in a of K, B in b of K draws)" (the refactor chat: "It's informative,
  and it reads straight"). It does not occur on gen5.
- **A decided row's other columns** come from the default draw when its verdict and named set are exactly the decided ones
  (`mc_row` = `default`). Otherwise they come from the first draw that has them (`draw k`), with that draw's candidate rows,
  so each row stays self-consistent. `mc_draws` gives the fractions: "CONSISTENT WITH A BACKGROUND SOURCE in 20 of 20 Monte
  Carlo draws; named: VLSSr J105425.8+715204 in 20 of 20, VLSSr J105246.5+715924 in 19 of 20".
- **`knife_edge`** is always the default draw's: it is what flagged the row.
- **`mc_n`** is the row's number of draws K.
- **`mc_outcomes`** counts every outcome (verdict and named set), most frequent first. Equal counts are ordered by name, so
  the decision depends on the counts alone.
- **Unflagged rows** keep their single draw.

With the flag, `results_v3.csv`, and so `forced_wide_v3.csv`, gain `bootstrap_n`, `centroid_z_se`, `knife_edge`,
`named_knife_edge`, `mc_draws`, `mc_n`, `mc_outcomes` and `mc_row`. `candidates_v3.csv` gains `p_emp_se`, `ratio_se`, `n_ctl_se` and `centroid_to_candidate_se`. With
`KNIFE_EDGE_SE` None (production) none of these columns is written. `KNIFE_EDGE_DRAWS` None (production's value) keeps
every row a single draw.

### `wide_publish` → `<review root>/<run>/forced_wide_v3.csv` in the 4 tested runs (`forced_wide_published`)

It writes **4 runs**, `forced_wide.RUNS`: `Run_SNAPFORCED_{I,V}{10s,1min}_gen5`. The 10-minute and deep forced runs
neither hold nor read the file; their queues only feed the job list. Each run gets the results rows of its (Stokes,
cadence). A `named_radec` column holds `ra,dec` (5 decimals) of each named source, `;`-joined. A named source is found by
name among the physical sources within 200′ of the target.

The file is written as `.part` and content-tested. The file it replaces is first copied beside it as
`<file>.bak_<UTC>` and read back. The `.part` is then renamed over the served file in one atomic replace. The unit's
meta records `backup`, `backup_sha1` (the replaced file's sha1) and `published_sha1`. A partial test (`--wide-only`) is
never published (W-PARTIAL). Each unit runs W-ENV and W-CATDIR first: `named_radec` reads the radio catalogues.

`--dry-run` plans and counts every gate the real publication would hit, and exits 2 if any: W-HOST (the review root on
another host), W-ROOT (a production review root while `paths.PRODUCTION_WRITES` is False; that holds for the served runs
today, so the install is a recorded copy of the scratch-published files, the coordinator's) and W-RUNDIR, for each of the
4 runs.

### What the figure consumes

`source_astrometry_panel.load(U)` reads `U/forced_wide_v3.csv` into `{identity: row}`. Then
`astrometry_assoc.forced_v3_verdict` uses these columns: `verdict`, `named`, `why` (its `[evidence: ...]` part),
`centroid_arcmin`, `centroid_z`, `centroid_nights` and `bursts_at_target`. Two verdicts map through `V3_MAP`:

- BACKGROUND SOURCE → CLEARLY A BACKGROUND SOURCE;
- NOT CENTERED ON THE TARGET (source not identified) → NEITHER THE CROSS-MATCH NOR A KNOWN BACKGROUND SOURCE.

Any other verdict keeps the position verdict, and the burst test's result is added to the reason. `named_radec` marks
the named sources.

The figure does not read the knife-edge columns yet. `docs/pending_production_patch_forced_wide_knife_edge.diff` adds a
flagged row's `mc_draws`, `named_knife_edge` and `knife_edge` text to the burst test's part of the reason. It is a patch to exopipe_ops'
`astrometry_assoc.forced_v3_verdict`, for the owning chat to apply. A row without them reads exactly as before (every
served row and every unflagged row checked).

### Deliberate differences from the analysis, and production's values

`EVENT_ORDER`, `TARGET_MEMBER` and `FILE_ERRORS` are PROPOSED (2026-11-03). `V2_CTL_DRAW`, `CONTROL_ORDER` and the Monte
Carlo keys are decided (2026-11-04). Production's values of all nine are in
`exopipe/phase3/parity/production_20261013.cfg` (`# WAVE2`).

| Key | Default | Production | Why |
|---|---|---|---|
| `forced_wide.EVENT_ORDER` | `sorted` | `directory` | The order the events are read is the order a target's bands are listed and summed in. The analysis read `os.listdir` order of its Lustre copy (`v3/events_copy/`). Lustre lists a directory in a fixed hash order of its names, so a copy on Lustre lists the same way; /data07 does not. |
| `forced_wide.TARGET_MEMBER` | `identity` | `first` | A merged identity (26 rows, 21 distinct (Stokes, identity) in the four gen5 runs) takes the list position of one member. Production took the first member in `build_targets`' glob order on /data07; a copy lists in another order. The searched cutouts are the identity's own, centered on the member it is named after (their TARGRA/TARGDEC, in every cutout). On gen5, `first` put 6 of 205 jobs 7.30–32.19″ (0.12–0.54′) off that point, at most 0.05 of the frame FWHM, and swapped two identities' positions. `identity` centers each job exactly on its cutouts. The member choice leaves the tests' statistics unchanged. Under production's capped control draw it did move one verdict (Struve 2398 A, I 1 min), but only because one control was re-indexed and the fixed-seed draw then picked 44 different controls; see `V2_CTL_DRAW`. |
| `forced_wide_test.V2_CTL_DRAW` | `None` (every eligible control pair) | `(300, 20261102)` | v2's centering kept at most 300 eligible control pairs, drawn at random with a fixed seed over a list in pool order, so any change to the job pool re-draws them. Struve 2398 A, I 1 min: 476 eligible pairs, 475 identical under both member rules; production's draw gives power − z 3.120 (BACKGROUND SOURCE), the draw after the member change 2.890 (UNDECIDED), all 476 give 3.003 (BACKGROUND SOURCE) under both rules, and 55% of 2,000 random draws of 300 read BACKGROUND SOURCE (review F2). Decided (RULED): `R-REFACTOR-20261104-FORCED-WIDE-CONTROLS-ORDER-INDEPENDENT` (1), "The cap was a speed convenience, and it makes knife-edge verdicts depend on sampling noise." |
| `forced_wide_test.CONTROL_ORDER` | `canonical` | `pool` | The taper test's control lines came from a pool-ordered list, shuffled by the job's generator, with one rotation per source drawn from the same generator; that generator then drew the centroid's bootstrap. v2's host ties went to the first job in pool order. So a moved or added job re-drew unrelated targets' calibration: the 900 cap binds in 484 of 1,267 gen5 calls (38%). Decided (RULED): `R-REFACTOR-20261104-FORCED-WIDE-CONTROLS-ORDER-INDEPENDENT` (2), "in a pipeline fed new data daily, adding a target must not re-draw an unrelated target's calibration." |
| `forced_wide_test.TAPER_CTL_MAX` | `None` (every eligible line) | `900` | v3 kept at most 900 control lines per candidate. The cap shortened 486 of 1,267 gen5 calls (38%, all Stokes I) and left the calibration to the draw of lines it kept. Without it the test evaluates 1.43 × the lines (`cap_cost_census.py`). Decided (RULED): the amendment (1). |
| `forced_wide_test.BOOT_DRAW` | `target` | `job` | The bootstrap drew from the job's generator, after the control lines in production's order. Its own stable seed per (target, tier) keeps it free of every other draw. Decided (RULED): the amendment (2), "seeded per target". |
| `forced_wide_test.BOOT_Z_SE` | `0.02` | `None` (exactly 1000 resamples) | At 1000 resamples the bootstrap alone moved a gen5 row's centroid z from 3.016 (the served draw) to 2.968 (round 3's draw), across the threshold 3 (Gaia DR3 3316364602541746048, I 1 min). Decided (RULED): the amendment (2). |
| `forced_wide_test.KNIFE_EDGE_SE` | `4.0` | `None` (no Monte Carlo columns) | Decided (RULED): the amendment (3); 4 from the 20-salt flip check of the gen5 run (2026-11-04). |
| `forced_wide_test.KNIFE_EDGE_DRAWS` | `20` | `None` (every row a single draw) | A flagged row is re-run in Monte Carlo draws, 20 at a time up to `KNIFE_EDGE_DRAWS_MAX` (100) while undecided, and decided only by them (`KNIFE_EDGE_DECIDE`, a Wilson 95% interval above one half). Production has no equivalent. Decided (RULED): the amendment, extended 2026-11-04. |
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

v3's own 900-line cap bound in 484 of 1,267 gen5 control-line calls (38%), in 199 of the 280 (Stokes, tier, target)
with a taper test, all Stokes I (2026-11-04 census). It is now off by default (`TAPER_CTL_MAX` None); a cap, if set, is
applied by the canonical order's seeded selection, keyed on stable identities.

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

2. `wide_search` and `wide_test` run on any node that reads Lustre, at most 6 processes each. Each process runs one BLAS /
   OpenMP thread unless `--threads` says otherwise:

   ```
   exopipe3 forced --tier snapshot --stages wide_search,wide_test --workers 6
   ```

   A `W_search` unit with a refused band (W-FILE, W-NOCUT) ends `done`, so `--resume` keeps it, and `wide_test` refuses
   W-POOL until `wide_search` runs again **without** `--resume` (with `FILE_ERRORS = 'refuse'` a transient read error
   needs that re-run). A changed `jobs.json` (a re-run of `wide_jobs` that changes any job) or another cutout generation
   needs a fresh `wide_search` of every job before `wide_test` pools them.

3. Publish on the review host, into the 4 tested runs:

   ```
   exopipe3 forced --tier snapshot --stages wide_publish
   ```

   While `paths.PRODUCTION_WRITES` is False this refuses W-ROOT for the served runs, and its dry run says so with exit 2.
   Until the cutover the files are published into a scratch review root and installed by a recorded copy.

**Family W, parity** (scratch roots only; the parity file holds production's nine values: the capped v2 draw, the
pool-ordered controls, the 900-line cap and the job generator's 1000 resamples included):

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
