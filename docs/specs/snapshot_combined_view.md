# The combined snapshot view: the snapshot tier's cumulative step

Spec of `exopipe3 serve --tier snapshot --family comb` (lib/serve/snapshot_view.py, its content tests
lib/serve/tests_snapshot_view.py, its standing test tests/test_snapshot_view.py). Refactor chat, 2026-11-03.

## Why

Production serves the 10-second and 1-minute snapshot pages (p8986-p8989) from ONE combined view of every served batch run:
36 batch runs since 2026-10-26, rebuilt whole by exopipe_ops `snapshot_combined_view.py` (1af2dab 2026-10-21 .. 5b8b8ff
2026-10-31; dde1760, dbc40a6 2026-11-04) through dated drivers, the latest `snapshot_batch/comb_build_20261104.sh` (the same
steps as comb_build_20261102.sh; the restore-only wave's comb_build_wave_20261104.sh adds per-batch tests, below). Until this stage the refactor had no stage,
spec or plan for it: its snapshot serve (lib/serve/snapshot_copy.py) serves single chain runs only, and the deep-omission
stage's 10-s / 1-minute drop (lib/catalog/deep_omission.py, the kept product `deep_omission_kept.csv`) had no consumer.

The decision (refactor chat, 2026-11-03): port the view as the snapshot tier's CUMULATIVE step, the counterpart of the 10-minute
cumulative (lib/serve/sublst_build.py). The reason: the real-time pipeline appends hours, so a full all-cell re-chain per night
does not fit; the view merges the identities the batch chains already served. The full all-cell chain with re-routing stays what
the view's own docstring names it: the end-of-run replacement or the major redo.

Rulings behind the view (clv15 fix_ledger.csv):

- R-REFACTOR-20261020-BUILD-COMBINED-SNAPSHOT-VIEW: the refactor chat, citing Gregg (a running combined run unless very expensive):
  build it as a VIEW, its limits stated on the pages. Gregg 2026-10-21: "Start the combined build"
  (IMPLEMENTED-COMBINED-SNAPSHOT-VIEW-BUILDER-FIRST-BUILD-20261021).
- R-GREGG-20261026-5-LST-REBUILD-AFTER-EVERYTHING-DONE-COMBINED-VIEW-ONE-PAGE-PER-TIER-POLARIZATION: "No! I just mean combine the
  batches for each tier/polarization into one page." One page per tier and Stokes, every batch combined into it.
- SNAP-PAGES-ONE-TIER-PER-PAGE (Gregg 2026-10-03: "Having the same tiers on a single page is not conformant"; "Match today's
  rule"), applied to the view's page directories after DEFECT-SNAPCOMB-PAGE-DIRS-HOLD-BOTH-TIERS-20261031.
- The deep omission of the short tiers: R-GREGG-20261024-DEEP-OMISSION-APPLIES-TO-SHORT-TIERS-WITH-FRAME-EXCEPTION and
  R-GREGG-20261024-DEEP-OMISSION-DESIGN-APPROVED-AND-10MIN-HOLD-RELEASED; the rebuild from the merged refusal record
  R-GREGG-20261031-SNAPSHOT-RESEARCH-LC-DEEPOMIT-GPU-RELEASE.

The VIEW LIMITS, written into every queue header as production writes them: (1) routing is not re-run on merged identities - a
source served by no batch alone but servable from two batches' detections combined is absent; (2) the merge is two-level (batch
identities, then across batches) and can differ from one all-cell chain.

## Command and stages

    exopipe3 serve --tier snapshot --family comb --run Run_SNAPCOMB_<YYYYMMDD> --view-runs RUN,RUN|@FILE \
        [--view-omit DIR] [--stages omit,build,filtercat,localize,ready] [--out OUT] [--dry-run]

`--run` names the view (the snapshot prefix `Run_SNAP`; production names `Run_SNAPCOMB_<date>`). `--view-runs` lists the served
batch runs in the view's order (a comma list, `@FILE` with `#` comments, or bare names under `paths.SNAPSHOT_REVIEW_ROOT`): the
order decides the batch tags of the headers, the order of the record and the ties of the names. A production `--out` is
`paths.SNAPSHOT_REVIEW_ROOT` (the view `<root>/<view>`, refused while `paths.PRODUCTION_WRITES` is False); any other `--out` is a
parity run: the view is `<out>/snapshot/<view>` (plan.make's layout). Each stage runs only after the previous one of the
invocation ended done; a stage run alone requires the build's record done and current (G28). When the view or its `.part`
exists, an invocation with the omit or the build stage refuses BEFORE any stage runs (G-V2): the omit product of an existing view
is its product of record and is never rewritten by a re-run of its name (V_combview review F2; filtercat, localize and ready
may still run alone on an existing view).

| Stage | Unit | What | Production source |
|---|---|---|---|
| omit | snapshot_\<view\>_omit | the deep-omission stage on the view's runs: decisions, record, the kept product, into `serve.SNAPVIEW_OMIT_DIR` (`<run root>/deep_omission/<view>`) | snapshot_combined_view.py step 0 (:71-144) |
| build | snapshot_\<view\>_build | the view, built aside as `<view>.part` and renamed after every test | snapshot_combined_view.py steps 1-4, T1-T6; driver step 1 |
| filtercat | snapshot_\<view\>_filtercat | snapshot_filter_catalog.py over the member batches' fit stores (SNAPCAT_PROCS = `SNAPVIEW_FILTERCAT_PROCS`), the card2 union, FILTER-INPUTS | driver step 2 (:50-80) |
| localize | snapshot_\<view\>_localize | snapshot_localize_tiers.sh on the view (the batch serve's own stage, `SNAPVIEW_LOC_ENV`) | driver step 3 (:81-84) |
| ready | snapshot_\<view\>_ready | the serve-time exclusion gate, every view test on the final files, `VIEW-READY-FOR-HIDES` | driver step 4 (:85) |

`--view-omit DIR` applies an omit product made earlier (by `catalog --tier snapshot --stages omit --omit-runs` on exactly these
runs, in this order); the omit stage is then not run, and asking for both refuses.

## Inputs

- The served batch runs (read only): `source_catalogs/*_sources_survivors.csv` (the general catalogue: one row per identity and
  band class, `review_dir` = the identity's directory, `ra_common`, `dec_common`, `pos_err_common_arcsec` = its served common-frame
  anchor); each `review_dir/detections.csv`; `source_catalogs_<S>_<T>/<band>_sources_survivors.csv` and `_source_members.csv`;
  every per-tier page directory (`detections.csv` and its panels); `queue_<S>_<T>.txt` (the first batch's header lines);
  `detection_ts.csv` when present; `card2_filter.csv`; `served_tiers.csv` (the chain's per-tier serving table: G4b / G-V7).
- The omit product (`deep_omission.csv`, `deep_omission_dropped.csv`, `deep_omission_served.csv`, `deep_omission_kept.csv`,
  `deep_omission_summary.json`) and the deep refusal record `catalog_omission.REFUSALS` and the fleet records under
  `paths.SNAPSHOT_FLEET_OUT` - read by the omit stage, and read AGAIN by the build, which re-decides from them (G-V8).
- clv15 `phase3_snapshot_chain.py` in place (`merge_identities_or_refuse`, its content test `verify_identity_merge`, the frame
  models it reads), exopipe_ops `snapshot_filter_catalog.py` and `snapshot_localize_tiers.sh` in place.
- The member batches' fit stores `serve.SNAPVIEW_FIT_STORE` (`<catalog_snapshot.BATCH_DIR>/rechain_<run>/fits_store.csv`).
- `data_decisions.csv` (the serve-time exclusion gate at ready; its fingerprint in the build record).

## The build

The child entry point of lib/serve/snapshot_view.py runs under the production interpreter `serve.SNAPVIEW_PYTHON` ('py36', the
interpreter production's driver runs the builder with), imports only the standard library, clv15 and deep_omission.py (loaded
from its file), and sets production's search path at the chain's import (exopipe_ops, then clv15 before it). Steps, in
production's order and code:

1. Clusters: each served batch identity (a `source_id` of the general catalogue; its detections = the rows of its rows'
   `review_dir/detections.csv`, one per detection key `dkey` = cand_id + peak position) is ONE cluster, anchored at its served
   common-frame position and error.
0. THE KEPT PRODUCT APPLIED, in place of production's step 0 (which called snapshot_deep_omission.decide in the build): an
   identity keeps exactly its `detection kept` rows; every other served detection must sit on an OMIT key of the decisions beside
   it; an identity left with none is dropped. Before anything is applied, the served set as read now must equal the one the omit
   stage decided on (the `served` digest of its summary, Q fix verification N2). The omission record rows are written in
   production's format and order.
2. The chain's `merge_identities_or_refuse` across all batches, with its content test (T1).
3. Names: a combined identity takes the J-name of its member identity of best |S/N|; a name already taken gets `~<batch>`.
4. The view under `<view>.part`: the general catalogue (`review_dir` = `candidates/_identity/<name>` holding every member
   detection, `batch_members`); per tier and Stokes (V first, 10 s first) the per-tier views (`<band>_sources_survivors.csv`: one
   row per combined source - the member row of largest |best S/N|, identity fields replaced, `n_det` and `n_det_identity`
   recomputed, ranks by |best S/N|; `<band>_source_members.csv`: every member row, `source_id` re-named); ONE page directory per
   tier, `candidates/<S>/<name>_<band>` (10 s) and `..._1min` (1 minute), holding HARD LINKS of the member batch sources' panels
   (never `detections.csv` or `astrometry.png`; a name clash gets the batch tag) and the merged `detections.csv`; the queues
   `queue_<S>_<T>.txt` (by best |S/N|); the record `deep_omission.csv` (the omit stage's table, byte for byte),
   `deep_omission_dropped.csv` and `deep_omission_summary.txt` (production's formats), `detection_ts.csv`, the empty
   `judgements/`, `combined_view_summary.json`.

Per tier, every omission decision is taken FROM THE KEPT PRODUCT and re-derived with production's rule from the same OMIT keys; a
disagreement refuses (G-K): a batch source dropped (its identity dropped, or every detection omitted), a source re-pointed, a
member row dropped, a panel left unlinked. The page directory's detections are the source's detections off the OMIT keys
(production's reading; the kept product lists no per-source detections). At the end every kept-product row must have been
applied (G-K).

Before the child runs, the step side re-decides (G-V8): the decisions made NOW from the refusal record and the fleet records the
build reads, with the served counts the omit product recorded (`deep_omission_served.csv`), must equal the omit product's
`deep_omission.csv` byte for byte - production decides inside its build from the current inputs, so an omit product made before
the deep search refused more nights, or before the snapshot search rewrote a noise or member table, refuses instead of being
applied (V_combview review F5). The child then checks those served counts against the served set it reads (G-V4) and each batch
run's per-tier catalogues against its `served_tiers.csv` (G-V7).

A build that refuses or fails after it made `<view>.part` moves the `.part` aside to `<view>.refused_<UTC>`; a built view that
fails its read-back tests (V1-V7 at build) is moved aside the same way; nothing is deleted, the view name is left free for a retry,
and the refusal names the new place (V_combview review F7). A `.part` the build did not make (another build's, G-V2) is never
touched.

## Outputs (product classes)

| Path | Class | Content test |
|---|---|---|
| the omit product (`SNAPVIEW_OMIT_DIR`) | omission_decisions, omission_record, omission_served, omission_kept | lib/catalog/tests_deep_omission.py (N2's digests, K5 the served counts) and the per-instant test |
| `<view>/` | snapshot_view | V1-V7 below |
| `<view>/master_catalog.csv` | filter_catalog (per_detection) | the current writer's contract: one row per detection, every detection once (FILTER-INPUTS) |
| `<view>/card2_filter.csv` | snapshot_card2_union | the union of the member batches' files, recomputed byte for byte |
| `<view>/source_catalogs_loc_<T>/localizations.csv` | csv_table (the batch serve's) | rows present; LOCALIZE-TIERS-OK |

Records: `<out>/_records/serve/<unit>.json`; the build's job, log and result under `<out>/_records/serve/snapshot_view/`.

## Gates (each refuses, exit 2)

- G-V1: no view name or no snapshot prefix; no run, a run listed twice, a run without `source_catalogs`; an identity without
  detections; a detection in two batch identities (the batches are not cell-disjoint).
- G-V2: the view or its `.part` exists (a view is never overwritten) - checked before ANY stage of an invocation that asks for
  omit or build (F2), again in the build stage and in the child.
- G-V3: the omit product misses a file, fails its content tests now, or was made for other runs or another order.
- G-V4: the served batch runs changed since the omit stage (served digest), or the served counts the decisions were made with
  (`deep_omission_served.csv`) are not the served set's.
- G-V7 (V_combview review F4): a batch run without `served_tiers.csv`, or whose per-tier survivors do not hold exactly the
  identities its `served_tiers.csv` serves in each tier (chain_outcome 'served', qualifies 1); a kept per-tier source whose page
  holds no detection (`detections.csv` missing or empty). An incomplete input (an interrupted copy, a re-serve that stopped
  halfway) refuses; production would serve the view without those sources, or record them as "every detection omitted" by the
  deep rule. The omit stage refuses the same inputs first (deep_omission G4b, G4c). The served_tiers relation holds exactly on
  all 36 batches of Run_SNAPCOMB_20261102 (2026-11-04, both tiers). Not seen by it: a per-tier file lost for an identity that
  keeps another per-tier source in that tier (the table is per identity and tier).
- G-V8 (V_combview review F5): the omit product's decisions are not the ones re-made now from the refusal record and the fleet
  records the build reads (a product decided from another record or fleet, a refusal added, a noise or member table rewritten).
- G-V5: a batch run on another file system than a PRODUCTION view, or a panel that cannot be hard-linked into it (the panels are
  hard links; production failed there with an error, and a copy of ~3 million panels would fill the review disk). A scratch or
  parity view copies such a panel instead - same bytes, counted (`panels_copied`) - because the parity guard (core/guard.py)
  refuses a hard link of a file outside the parity roots, as it does for every caller that links to save a copy.
- G-V6: an ordinary run reading a run or an omit product under an injected run's start record (also the CLI's source-flag guard
  for `--view-runs` / `--view-omit`).
- G-K: the kept product does not apply exactly (see the build).
- T1-T6, production's tests inside the build, before the rename: T1 the merge's content test; T2 every kept detection in exactly
  one combined identity, none added; T3 every per-tier source row in exactly one combined source, the panels conserved; T4 every
  combined identity with a source in a tier view in that view's queue exactly once; T5 every member row's detection in its page
  directory's `detections.csv`; T6 every page directory of one tier (rows and panel tokens).
- F-G1 a member batch without its fit store; F-G2 a `card2_filter.csv` missing or of another header; F-G3 the filter catalogue
  script refused or failed; F-G4 a reader count (`SNAPVIEW_FILTERCAT_PROCS`) that is not an integer >= 1.
- G21 the localization (LOCALIZE-TIERS-OK); G17 the serve-time exclusion gate on the view (ready); G28 an earlier stage's record
  missing or stale.

## Content tests of the view (class snapshot_view, read back from disk)

- V1 files: the summary's tests line, name and batch tags; every omission file, per-tier view and queue; no `.part`; at ready
  the filter catalogue, the card2 union, `coadd_root.txt` and both localizations.
- V2 queues (T4): the header lines (name, run, VIEW LIMITS, DEEP OMISSION with the summary's counts) and the names = the tier
  view's identities, once each, by best |S/N|.
- V3 pages (T5, T6): every combined source's page directory has `n_det` rows, all of its tier and on no OMIT key, panels of its
  tier and none of an omitted instant - each panel's own key (LST, band, night) parsed by the test itself, not with the shared
  `deep_omission.SNAP_PANEL` the kept product and the build both use (F1) - every member row's detection (all directories, or a
  seeded sample `serve.SNAPVIEW_TEST_SAMPLE_DIRS` for a full view).
- V7 panels (F1): every page directory's files are exactly its member batch sources' files (the general catalogue's
  `batch_members`, each member identity's per-tier survivors row of the page's band, less the sources the kept product drops,
  their page directories listed now) less the kept product's 'panel dropped' rows; a name clash (`<batch>_<file>`) counts as the
  file. A panel linked that the omission removed, or a kept panel missing, refuses - production's T3 counts links in the loop that
  decides them and cannot see a wrong decision. The reviewer's mutation M1 (the build links an omitted panel it records as
  dropped) is refused by V3 and V7 (tests/test_snapshot_view.py test_m1_mutation). Cost: one listing of every page directory
  and of every member batch source's directory (on cold Lustre about 0.08 s each; a production view on the review disk lists
  far faster); `serve.SNAPVIEW_TEST_SAMPLE_DIRS` samples V3 and V7 alike.
- V4 identities (T2 against the kept product): the detections under `candidates/_identity/` are exactly the kept product's kept
  detections; the summary's counts; every general-catalogue row's directory exists.
- V5 record: `deep_omission.csv` equals the omit stage's byte for byte; the record's removals equal the omit stage's (detections,
  dropped identities; per-tier rows kind for kind); the page line; the per-instant test.
- V6 at ready: the filter catalogue's per-detection contract; the localized identities are view identities; THE PAGE FIGURES:
  every identity localized with more than one detection in a tier has its figure, each of its page directories of that tier holds
  astrometry.png of the figure's size, and a page of an identity without a figure holds none (production's localizer left 7 of
  11,245 page copies truncated on 2026-11-04, the V2 follow-up's parity; sizes, not bytes, so a full view's re-test stays cheap).

## Configuration (section serve)

| Key | Value | Source |
|---|---|---|
| SNAPVIEW_TIERS | ('10s', '1min') | snapshot_combined_view.py:57; protected, ruled (one page per tier) |
| SNAPVIEW_STOKES | ('V', 'I') | :57 (the pass order, the record's order); protected |
| SNAPVIEW_PAGE_SUFFIX | {'10s': '', '1min': '_1min'} | :204 TSFX; protected, ruled (SNAP-PAGES-ONE-TIER-PER-PAGE) |
| SNAPVIEW_OMIT_DIR | '{root}/deep_omission/{name}' | the omit product's place (production had none: it decided inside the build) |
| SNAPVIEW_FIT_STORE | '{batch_dir}/rechain_{run}/fits_store.csv' | comb_build_20261102.sh:51 |
| SNAPVIEW_PYTHON | 'py36' | comb_build_20261102.sh:18, :48, :52 |
| SNAPVIEW_LOC_ENV | {'P3_LOC_SHARDS': '12'} | comb_build_20261102.sh:82 (speed only) |
| SNAPVIEW_BUILD_TIMEOUT_S | 172800 | the 36-batch view took 33,492 s; production sets no limit |
| SNAPVIEW_FILTERCAT_PROCS | 6 | SNAPCAT_PROCS of the filter catalogue (exopipe_ops 1b9475c: chunks of the glob order read in a process pool, merged in that order; fewer than 1000 files are read serially). Production's default is 16; 6 is the Lustre rule (at most 6 readers). Speed only: the catalogue is byte-identical for every value. Always passed, so no inherited value reaches the child (`SNAPCAT_` is not a cleared switch family). The snapshot pages' own filter catalogue takes `pages_snapshot.FILTERCAT_PROCS` (6) |
| SNAPSHOT_FIGURE_HASHSEED | '0' | PYTHONHASHSEED of `snapshot_localize_tiers.sh` (every snapshot localize). Production sets none: its figure pool starts from a random string-hash seed each run, and 64 of 4,130 figure v2 PNGs differed between two runs on byte-identical tables (V_combview review F10). A reproducibility setting of the refactor, not a science value and not a ruling: no parity line (production's "value" is a random seed); '' = unset, as production |
| SNAPVIEW_TEST_SAMPLE_DIRS | 0 | the re-read's sample (0 = every page directory; measured about 0.08 s per directory on cold Lustre, 10,562 directories in 880 s - set a sample for a full view if that is too slow; the build's T5 and T6 always cover every directory) |

Every default but SNAPSHOT_FIGURE_HASHSEED is production's value; no ruled value differs from production's, so the parity file
(parity/production_20261013.cfg) has no line for the view. The serve step's CONFIG now declares `catalog_snapshot` (BATCH_DIR).

## Parity target

Production's own `snapshot_combined_view.py` (and the driver's steps 2-3) on the same batch runs, the same refusal record and the
same fleet records, run under its interpreter, against this stage, into scratch:

- byte for byte: the general catalogue, every per-tier view (survivors with production's duplicate `batch_members` column, and
  members), every page directory's `detections.csv` and its set of linked panel names, every identity directory's
  `detections.csv`, the queues, `deep_omission.csv`, `deep_omission_dropped.csv`, `deep_omission_summary.txt`,
  `detection_ts.csv`, `master_catalog.csv`, `coadd_root.txt`, `card2_filter.csv`, the localizations;
- `combined_view_summary.json` equal but for `seconds` (the run time);
- panels: the same names, each the same inode (both views link the same input copy) or, in a `--parity` run of the stage
  (copies under the parity guard), the same bytes;
- production is deterministic given the directory listings it globs (the per-tier catalogues are read in `glob` order, unsorted:
  the record's row order and the ties of equal |S/N| follow it - kept, marked PRODUCTION ORDER). A parity run therefore reads one
  input copy for both builds.

The standing test (tests/test_snapshot_view.py) runs production's builder and the stage on a synthetic world and compares the two
trees. tests/parity_snapshot_view.py runs both sides on real data and compares them (prod, port, compare).

THE PRODUCTION PIN (V_combview review F3; tests/production_pin.py). The tests compare the port with the production code it
ports, read at ONE pinned exopipe_ops commit (since the V2 follow-up round fa90670, 2026-11-04 15:38: the builder and the snapshot
decisions of dde1760 / dbc40a6, snapshot_rejudge.py of dbc40a6, the 10-minute decisions, drop and cumulative of dde1760 / edf3263
/ 35aad1a, the drivers comb_build_20261102.sh and comb_build_20261104.sh; before it c7ca600), each file checked against its git
blob sha and materialized into the test's
scratch: the stage's own tests (gates, copy fallback, content tests, ready) build the stage's view alone and never wait on
production's builder; production's side runs only in test_drift, test_parity and test_repoint_pinned, from the pinned files.
`test_production_pin` (in test_snapshot_view and test_deep_omission) is the explicit drift test: it fails when production
changed a pinned file since the pin, naming the files and the commits to port. Production code the stage runs IN PLACE (the
chain's merge, snapshot_filter_catalog.py, snapshot_localize_tiers.sh, the localizer and the figure) is live on both sides.
Moving the pin is a port: the round that ports production's later commits moves PIN_COMMIT / PIN_BLOBS with it. The V2
follow-up round (below) moved it to fa90670; the drift test passes again (no pinned file changed after it, 8ff3fc8 included).

Result on real data (2026-11-03/04; refactor_reviews_20261103/V_combview_report.md):

- **Inputs.** Run_SNAPB6H2_20261101, Run_SNAPB6I2_20261101 and Run_SNAPB6L_20261025, with the refusal record and the fleet records
  frozen.
- **The subset.** 43,553 served detections, 6,730 omitted (165 OMIT keys), 182 identities dropped. 4,136 combined identities, 196
  of them merged across batches and 1 renamed.
- **The stage.** 78201c1: every stage done, every content test passed.
- **The comparison.** Each view holds 53,620 files in 14,717 directories, the same sets. Equal byte for byte: every catalogue,
  per-tier view, page and identity detections.csv, queue, omission file, filter file, localization table and localization input.
  The 24,911 panels have the same bytes, and the summary is equal but for `seconds`.
- **The 751 differences, all explained.**
  - 2 are localization build.log files, each naming its own output path.
  - 122 of the 4,130 figures, and their 627 copies as page astrometry.png, differ in the stacking order of overlapping detection
    markers (3-456 pixels, inside the scatter panel). Production's figure code before exopipe_ops 7e6904a, which ran in place and
    was the same on both sides, draws bytes that depend on the drawing process, not only on its inputs. Redrawn from the same tables,
    one process drawing in sequence reproduces production's figures (122 of 122), and production's pool layout reproduces the
    stage's (121 of 122).
  - CORRECTED (V_combview review F10): the figure code that replaced it on 2026-11-04 (figure v2) is NOT byte-reproducible across
    runs. In the reviewer's comparison (figure v2 on both sides, byte-identical tables) 64 of 4,130 figures and their 325 page
    copies differ; the figure pool's string-hash seed decides the stacking of overlapping markers (PYTHONHASHSEED 4 reproduces all
    64 of one side, 5 all 64 of the other). The earlier check (seeds 0 and 1 on the 122 v1 identities) happened to agree. The
    stage now runs the localization with PYTHONHASHSEED = `serve.SNAPSHOT_FIGURE_HASHSEED` ('0'); a parity run gives production's
    side the same seed (tests/parity_snapshot_view.py), and the figures then compare byte for byte (see the fix round's
    demonstration below).

The fix round (2026-11-04, refactor_reviews_20261103/V_combview_report.md section 10):

- **The 64 figures.** Both sides' localization tables from the reviewer's comparison were copied and redrawn THROUGH THE STAGE'S
  localize stage (figures only, `SNAP_LOC_FIGURES_ONLY=1`; the seed from the config, PYTHONHASHSEED=0 seen in every figure
  process). Production's tables were drawn by a pool of 2 and the stage's by a pool of 3. All 4,130 figures were identical (3,361
  at 10 s, 769 at 1 minute), the 64 that had differed among them.
- **The new checks on real data, read only.**
  - V3 with F1 and V7 pass on both the stage's and production's parity views: 10,562 page directories, 24,911 panels, all
    parsed as panels.
  - G-V7 (served_tiers.csv) passes on the three batches.
  - G-V8 re-makes 869 decisions from the frozen record and fleet, equal to the omit product byte for byte; K5 holds.
- **The stage at 2027bce on the same inputs.** It ran omit, build and filtercat as an ordinary run, hard links, SNAPCAT_PROCS=6.
  - Every unit ended done and every content test passed, G-V8 included.
  - The decisions equal production's view's byte for byte.
  - Against production's view, every file is identical but the localization's, which was not re-run: the general catalogue,
    per-tier views, queues, 10,562 page and 4,136 identity detections.csv, 24,911 panels (the same inodes), the omission
    files, the filter files, and the summary but `seconds`.

## The V2 follow-up round (2026-11-04): production's 11-04 commits

Ported on branch wt-20261104-V2-follow (from 3833971); each is production's behavior byte for byte on the synthetic world
(tests/test_snapshot_view.py test_parity, tests/test_deep_omission.py) and on real data (below):

- exopipe_ops dde1760, the omission PER IMAGE: a wideband detection is omitted only when its own frame (10 s, `ts`) or block (1
  minute, clv15 `integration_minute_key(date_obs)` + '00') holds the omitted subband (the cell's `wideband_members_<S>.csv`, per
  image); an image absent from the table is omitted (production, fail closed) and refused here as a key mismatch (G7, the review's
  T2); a cell without a member table (searched before 2026-10-07) takes every image of the night to hold every admitted subband
  (`admission_<sb>.csv`; whole-night rows, fail closed); gate names, values and thresholds paired in one order (`sorted(gates)`);
  the queue's DEEP OMISSION line and the summary count the per-image wideband omissions separately. The light-curve change of
  dde1760 was already in the branch (lib/curves/tests.py reads the admission table; an absent ts counts as admitted).
- exopipe_ops dbc40a6: CARD 2 per frame in the 10-s tier (each batch's `card2_vleak.csv`, self-checked; record kinds `card2
  detection`, `card2 member row`, `card2 panel`; the served counts exclude them); THE RE-JUDGING of every batch source per tier on
  the detections it keeps (snapshot_rejudge.rejudge, ported as lib/catalog/rejudge.py and run by the OMIT stage in py36 children,
  `catalog_omission.REJUDGE_PROCS` 6, never inside the build: the chain's functions cannot run there; its inputs - the batch's
  detection_record.csv.gz, chain_summary.json, served_tiers.csv, card2_vleak.csv - are fingerprinted (size, mtime) and re-checked by
  G-V8 at the build; refused unless the reconstruction reproduces the chain on every served source, G9); a (batch identity,
  tier) that no longer qualifies loses its per-tier source (`source re-judged`, `member row re-judged`, rejudged.csv); THE
  RE-POINT FIX (the review's defect RP, production's fix): a partial source's best is its own (ts, subband) detection, re-pointed
  to the largest-|S/N| detection it keeps (`best_cand_id` `<identity>_<lst>_<date>_<band>`, `fit_verdict`, `rejudged_routes`);
  THE RATE POPULATION (R-REFACTOR-20261104-OMISSION-RATE-POPULATION-SEARCHED-NIGHTS): the rate test counts the nights the view's
  own batches searched (`chain_summary.json` cells; G2b a batch without them), a decided night outside them has no served rate
  and passes; T7 (no Card 2 detection left, every re-judged (batch source, tier) met, every combined row ranked by a detection it
  keeps), the queue line `# CARD 2 AND RE-JUDGING ...`, the summary keys `card2` / `rejudge`, the tests line `T1 ... T7 PASS`.
  The omit summary records each batch's Card 2 vintage (`card2_vintage`: 'chain' for a batch whose chain ran Card 2 itself, clv15
  1279bd4, before the tie and the 3-D gate; 'view' for one chained before - the view then applies Card 2 after its chain; the
  review's T7: a view mixing both applies Card 2 in two orders - recorded, not a gate).
  The ledger's correction (CORRECTION-R-REFACTOR-20261104-OMISSION-RATE-POPULATION-SEARCHED-NIGHTS-FULL-VIEW-CHANGES, measured by
  the snapshot chat): on the full 36-batch view the rate population changes 2 decision rows (08h 2025-01-07 23 MHz I 1-minute OMIT
  -> KEEP, restoring 728 detections, and its Red I 1-minute member row).
- exopipe_ops edf3263 and 35aad1a, the 10-minute side: the merge drop's re-judging (sublst_deep_omission_drop.py, IN PLACE) is
  predicted by the omit stage's 10-minute record (`source re-judged` rows; G10 a survival the drop does not re-derive; G11 a
  re-judged-out source without Gregg's go `catalog_omission.TENMIN_REJUDGE_APPROVED`, which the drop itself also requires).
- exopipe_ops 35aad1a and 8ff3fc8, the serve side (lib/serve/snapshot_copy.py, G27 before every stage of a snapshot serve):
  a run whose combined_view_summary.json holds "card2" or "rejudge" is served only when `serve.SNAPSHOT_CARD2_REJUDGE_APPROVED`
  names it; a combined view's member runs (one per tag of its `batches`, beside it) that ran the restore-only 3-D gate need
  their test against the run they replaced passed and `serve.SNAPSHOT_RESTORE_ONLY_APPROVED` naming the view. A name counts only
  as a whole token (production's grep: a substring - a go naming Run_SNAPCOMB_20261104B also passes Run_SNAPCOMB_20261104).
- exopipe_ops 1b9475c, the parallel filter catalogue: `serve.SNAPVIEW_FILTERCAT_PROCS` (above).
- exopipe_ops 7119fad, the 10-minute leak veto's Master floor on the deep flux scale, restore-only (R-REFACTOR-20261104-LEAKVETO-
  FLOOR-RESTORE-ONLY): runs in place; the content test leak_veto_table now reads each row's `master_floor_jy` (= max(0.5, 0.5 x
  `window_flux_factor`), checked) instead of the fixed 0.5 Jy. exopipe_ops 1907941 (the forced-reserve radius back to linear in
  ln S): no change, the branch's lib/photometry/tests.py `_radius_bw` was already linear.
- clv15 1279bd4 + 6baaee5 (Card 2 in the chain before the tie and the 3-D gate; the radius lookup unchanged) and the restore-only
  3-D gate (clv15 b8d1989, 827cb7d, bf7dce8; exopipe_ops fa90670), in the snapshot chain stage (lib/catalog/tier_snapshot.py): the
  chain runs in place; its approval marker's `CURVE_CONSTANTS_RESTORE=<table>` line passes `--curve-constants-restore` to every
  chain call (P-G21 a named table missing); the run content test (class snapshot_run) reads Card 2 (card2_removed.csv against the
  summary and the record) and the restore-only gate (restored_by against the record's two passes, the summary and every served
  detections.csv) back; the chain's literals CARD2_RATIO / CARD2_TIERS, RESTORED_BY, RESTORE_TOL_BEAMS are bound config keys.

THE ROUND'S PARITY ON REAL DATA (2026-11-04; tests/parity_snapshot_view.py: production at the pin fa90670 - dbc40a6's builder
and re-judging, SNAPCOMB_REJUDGE_PROCS=6, its filter catalogue SERIAL - against the stage at 008ec96 as a --parity run, omit ..
ready; the frozen B6H2 + B6I2 + B6L copies with their chain records, the refusal record, the fleet records frozen 2026-11-04):

- **The subset under the 11-04 code.** 43,553 served detections; 886 decisions, 144 OMIT keys + 23 wideband ones per image; 232
  detections omitted (6,730 under the 11-03 code: in a subset view the nights of the batches left out counted as zero-rate nights -
  the rate population, review T1), 5 removed by Card 2, 10 identities dropped; the re-judging reproduced 4,524 served sources, 69
  (source, tier) touched, 48 still qualify, 3 re-judged out, 18 lose every member; 4,289 combined identities (211 from two or more
  batches, 1 renamed), 12,108 page directories. Production's T1-T7 pass on both sides; every stage of the stage ended done with
  every content test passed (G-V8, V1-V7), stale_best 0, no image absent from a member table (T2), the 5 served Card 2 rows all
  matched (T3), every batch's Card 2 vintage 'view' (T7).
- **The comparison.** 61,522 files in 16,416 directories on each side, the same sets. Identical: the general catalogue, the 138
  per-tier view files, 12,108 page and 4,289 identity detections.csv, the 4 queues, the omission files, rejudged.csv, the filter
  files (master_catalog.csv read by the stage's 6 processes = production's serial pass, 43,317 lines), the 29,052 panels (bytes:
  the stage copies under the parity guard), the 140 localization inputs, both localization tables and offsets, all 4,529 figures,
  and the summary but `seconds` (2,850 s, 6,200 s).
- **The 9 differences, explained.** 2 localization build.log files, each naming its own output path. 7 page astrometry.png files
  in PRODUCTION's view are truncated copies of their identical figures - 0, 65,536 or 131,072 bytes, each a byte prefix of the
  figure - written by the localizer's figure pool (snapshot_localize_tiers.sh, shutil.copyfile) without an error; the stage's
  copies, made by the same in-place script, are complete. A production defect, not a port difference; V6 now refuses a page copy
  of another size than its figure.

Not ported in this round (listed for the next): the restore-only wave's per-batch test against the run each new run replaces and
its totals (exopipe_ops snapshot_restore_wave_diff.py, 8ff3fc8) and the wave's dated view driver comb_build_wave_20261104.sh (per
batch tests as runs land, a GO file, one build): the branch has no place yet that knows which run a new run replaces (production:
the wave's schedule CSV); until it does, G27 refuses a view holding a restore-only run whose test file is missing (fail closed).
The snapshot serve's SERVE_HOLD file (exopipe_ops aa100bf, 2026-10-17: snapshot_batch/SERVE_HOLD refuses every snapshot serve) was
never ported to lib/serve/snapshot_copy.py - found in this round, listed.

## Production behavior carried for parity, pending Gregg

None of these is ruled; each is production's today and changes only with a ruling (or production's fix first):

1. (RESOLVED 2026-11-04: production fixed the re-point in dbc40a6 and the port follows it - the V2 follow-up round above. It was
   PRODUCTION BUG RP: the re-point fired only when a per-tier row's `best_cand_id`, which names a member source, was an omitted
   detection's cand_id - never.)
2. (RESOLVED 2026-11-04: partial sources are re-judged on the detections they keep - dbc40a6's re-judging, ported above; Gregg's
   answers R-GREGG-20261104-DEEP-OMISSION-SHORT-TIERS-ANSWERS.)
3. The name-clash rule (`<J>~<batch>`, directories with `+` -> `p`, `~` -> `_`), the combined source's row (its member row of
   largest |best S/N|), the queue order (best |S/N|, then name): production's choices, stated in its docstring, not ruled.
4. PRODUCTION FORMAT: the per-tier survivors files and the general catalogue name `batch_members` twice in their header
   (snapshot_combined_view.py:198, :299: `hdr_union` already ends with it); every row writes it twice. Kept for byte parity.

## Not ported, and why

- REPAIR MODE (`SNAPCOMB_REPAIR_OF`, 2026-10-28): it recomputes a served view built by older code and installs only the
  `detections.csv` files that differ, in place. It existed to repair the 10-28 defect in served views without a rebuild; the
  refactor builds a new view directory each time (aside, renamed, never overwritten), and repairing a served product in place is
  a write the refactor does not make. A view built before 2026-10-31 is not repairable by production's code either.
- The test-build switch `SNAPCOMB_DEEP_OMISSION=0`: the served view always applies the rule; the stage always applies a kept
  product (a view without the rule would be indistinguishable on the pages).
- The registration of the view's pages (p8986-p8989, `snapshot_serve.sh` with `SNAP_REGISTER=1`, `CARRY_FROM` the served view,
  `SNAP_LOCALIZED_ON_NODE=1`, the page context `page_context_snapshot.py`, the hub's `snapshot-comb` entries): serve.PAGES holds
  no combined-view page and the refactor has no page-context step; the snapshot serve's stages (gate, carry, test, start, hub,
  verify) apply once the four pages are configured. Until then production registers the views.
- The two snapshot hides and the cross-tier pair files the refactor chat writes between VIEW-READY and the registration (FOLD_IN
  10-31 16:40-17:38, 22:00): there is no FP-filter step yet (CUTOVER_READINESS "FP-study page filters must exist BEFORE the pages
  register").
- An incremental append (one new batch into an existing view): production rebuilds the view whole; a cumulative step that
  merges only the new batch's identities into the existing combined identities is a design change (the merge's order is by
  |S/N| over all clusters), owed to the real-time pipeline, not a port.
- The driver's wait for the replacement runs' BUILT lines and its run-list resolution from the previous view
  (comb_build_20261102.sh:20-46): dated scheduling; the stage takes its runs explicitly and refuses missing ones.

## Open questions

- (The re-point: done - production fixed it, the port follows, test_repoint.)
- Whether the view should also refuse a per-tier page detection that is not among its identity's kept detections (the build
  counts them; zero would make it a gate).

Listed from the V_combview review, not fixed in its fix round (2026-11-04; low, none seen on real data):

- F6: a KEPT panel moved into another source's page directory of the same batch after the omit stage is served on that page
  (production links whatever the directory lists). V7 compares a page with its member batch sources' directories as they are
  now, so it does not see the move; the review's cheap fix is a check that a linked panel's frame is a detection of its page.
  (A kept panel copied onto another page of the VIEW is refused by V7: test_content.)
- F8: a three-way panel-name clash (three member sources of one page directory with one panel name, two from one batch): the
  second clash name `<TAG>_<fn>` exists too - a production view refuses with the wrong G-V5 cause, a scratch / parity view's copy
  overwrites one panel silently (both counted). Production raises FileExistsError. Fix: refuse a second clash explicitly in both
  modes.
- F9: the view's `--dry-run` does not run the injected-run guard on the refusal record (`--set catalog_omission.REFUSALS=<an
  injected run's record> --dry-run` exits 0; the real run refuses at the omit stage, G1, nothing written; `catalog --stages omit
  --dry-run` refuses). Fix: `DO.injected_record_gate` before `describe()`.
