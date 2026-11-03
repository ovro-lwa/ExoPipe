# The combined snapshot view: the snapshot tier's cumulative step

Spec of `exopipe3 serve --tier snapshot --family comb` (lib/serve/snapshot_view.py, its content tests
lib/serve/tests_snapshot_view.py, its standing test tests/test_snapshot_view.py). Refactor chat, 2026-11-03.

## Why

Production serves the 10-second and 1-minute snapshot pages (p8986-p8989) from ONE combined view of every served batch run:
36 batch runs since 2026-10-26, rebuilt whole by exopipe_ops `snapshot_combined_view.py` (1af2dab 2026-10-21 .. 5b8b8ff
2026-10-31) through dated drivers, the latest `snapshot_batch/comb_build_20261102.sh`. Until this stage the refactor had no stage,
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
invocation ended done; a stage run alone requires the build's record done and current (G28).

| Stage | Unit | What | Production source |
|---|---|---|---|
| omit | snapshot_\<view\>_omit | the deep-omission stage on the view's runs: decisions, record, the kept product, into `serve.SNAPVIEW_OMIT_DIR` (`<run root>/deep_omission/<view>`) | snapshot_combined_view.py step 0 (:71-144) |
| build | snapshot_\<view\>_build | the view, built aside as `<view>.part` and renamed after every test | snapshot_combined_view.py steps 1-4, T1-T6; driver step 1 |
| filtercat | snapshot_\<view\>_filtercat | snapshot_filter_catalog.py over the member batches' fit stores, the card2 union, FILTER-INPUTS | driver step 2 (:50-80) |
| localize | snapshot_\<view\>_localize | snapshot_localize_tiers.sh on the view (the batch serve's own stage, `SNAPVIEW_LOC_ENV`) | driver step 3 (:81-84) |
| ready | snapshot_\<view\>_ready | the serve-time exclusion gate, every view test on the final files, `VIEW-READY-FOR-HIDES` | driver step 4 (:85) |

`--view-omit DIR` applies an omit product made earlier (by `catalog --tier snapshot --stages omit --omit-runs` on exactly these
runs, in this order); the omit stage is then not run, and asking for both refuses.

## Inputs

- The served batch runs (read only): `source_catalogs/*_sources_survivors.csv` (the general catalogue: one row per identity and
  band class, `review_dir` = the identity's directory, `ra_common`, `dec_common`, `pos_err_common_arcsec` = its served common-frame
  anchor); each `review_dir/detections.csv`; `source_catalogs_<S>_<T>/<band>_sources_survivors.csv` and `_source_members.csv`;
  every per-tier page directory (`detections.csv` and its panels); `queue_<S>_<T>.txt` (the first batch's header lines);
  `detection_ts.csv` when present; `card2_filter.csv`.
- The omit product (`deep_omission.csv`, `deep_omission_dropped.csv`, `deep_omission_kept.csv`, `deep_omission_summary.json`)
  and, through the omit stage, the deep refusal record `catalog_omission.REFUSALS` and the fleet records under
  `paths.SNAPSHOT_FLEET_OUT`.
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

## Outputs (product classes)

| Path | Class | Content test |
|---|---|---|
| the omit product (`SNAPVIEW_OMIT_DIR`) | omission_decisions, omission_record, omission_kept | lib/catalog/tests_deep_omission.py (N2's digests) and the per-instant test |
| `<view>/` | snapshot_view | V1-V6 below |
| `<view>/master_catalog.csv` | filter_catalog (per_detection) | the current writer's contract: one row per detection, every detection once (FILTER-INPUTS) |
| `<view>/card2_filter.csv` | snapshot_card2_union | the union of the member batches' files, recomputed byte for byte |
| `<view>/source_catalogs_loc_<T>/localizations.csv` | csv_table (the batch serve's) | rows present; LOCALIZE-TIERS-OK |

Records: `<out>/_records/serve/<unit>.json`; the build's job, log and result under `<out>/_records/serve/snapshot_view/`.

## Gates (each refuses, exit 2)

- G-V1: no view name or no snapshot prefix; no run, a run listed twice, a run without `source_catalogs`; an identity without
  detections; a detection in two batch identities (the batches are not cell-disjoint).
- G-V2: the view or its `.part` exists (a view is never overwritten).
- G-V3: the omit product misses a file, fails its content tests now, or was made for other runs or another order.
- G-V4: the served batch runs changed since the omit stage (served digest).
- G-V5: a batch run on another file system than the view (the panels are hard links; production failed there with an error).
- G-V6: an ordinary run reading a run or an omit product under an injected run's start record (also the CLI's source-flag guard
  for `--view-runs` / `--view-omit`).
- G-K: the kept product does not apply exactly (see the build).
- T1-T6, production's tests inside the build, before the rename: T1 the merge's content test; T2 every kept detection in exactly
  one combined identity, none added; T3 every per-tier source row in exactly one combined source, the panels conserved; T4 every
  combined identity with a source in a tier view in that view's queue exactly once; T5 every member row's detection in its page
  directory's `detections.csv`; T6 every page directory of one tier (rows and panel tokens).
- F-G1 a member batch without its fit store; F-G2 a `card2_filter.csv` missing or of another header; F-G3 the filter catalogue
  script refused or failed.
- G21 the localization (LOCALIZE-TIERS-OK); G17 the serve-time exclusion gate on the view (ready); G28 an earlier stage's record
  missing or stale.

## Content tests of the view (class snapshot_view, read back from disk)

- V1 files: the summary's tests line, name and batch tags; every omission file, per-tier view and queue; no `.part`; at ready
  the filter catalogue, the card2 union, `coadd_root.txt` and both localizations.
- V2 queues (T4): the header lines (name, run, VIEW LIMITS, DEEP OMISSION with the summary's counts) and the names = the tier
  view's identities, once each, by best |S/N|.
- V3 pages (T5, T6): every combined source's page directory has `n_det` rows, all of its tier and on no OMIT key, panels of its
  tier, every member row's detection (all directories, or a seeded sample `serve.SNAPVIEW_TEST_SAMPLE_DIRS` for a full view).
- V4 identities (T2 against the kept product): the detections under `candidates/_identity/` are exactly the kept product's kept
  detections; the summary's counts; every general-catalogue row's directory exists.
- V5 record: `deep_omission.csv` equals the omit stage's byte for byte; the record's removals equal the omit stage's (detections,
  dropped identities; per-tier rows kind for kind); the page line; the per-instant test.
- V6 at ready: the filter catalogue's per-detection contract; the localized identities are view identities.

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
| SNAPVIEW_TEST_SAMPLE_DIRS | 0 | the re-read's sample (0 = every page directory) |

Every default is production's value; no ruled value differs from production's, so the parity file
(parity/production_20261013.cfg) has no line for the view. The serve step's CONFIG now declares `catalog_snapshot` (BATCH_DIR).

## Parity target

Production's own `snapshot_combined_view.py` (and the driver's steps 2-3) on the same batch runs, the same refusal record and the
same fleet records, run under its interpreter, against this stage, into scratch:

- byte for byte: the general catalogue, every per-tier view (survivors with production's duplicate `batch_members` column, and
  members), every page directory's `detections.csv` and its set of linked panel names, every identity directory's
  `detections.csv`, the queues, `deep_omission.csv`, `deep_omission_dropped.csv`, `deep_omission_summary.txt`,
  `detection_ts.csv`, `master_catalog.csv`, `coadd_root.txt`, `card2_filter.csv`, the localizations;
- `combined_view_summary.json` equal but for `seconds` (the run time);
- panels: the same names hard-linked to the same inodes (both views link the same input copy);
- production is deterministic given the directory listings it globs (the per-tier catalogues are read in `glob` order, unsorted:
  the record's row order and the ties of equal |S/N| follow it - kept, marked PRODUCTION ORDER). A parity run therefore reads one
  input copy for both builds.

The standing test (tests/test_snapshot_view.py) runs production's builder and the stage on a synthetic world and compares the two
trees; the real-data parity of 2026-11-03 is in refactor_reviews_20261103/V_combview_report.md.

## Production behavior carried for parity, pending Gregg

None of these is ruled; each is production's today and changes only with a ruling (or production's fix first):

1. PRODUCTION BUG RP - the re-point of a partial source never fires on real data. Production re-points a per-tier source whose
   best detection was omitted only when its `best_cand_id` is one of the omitted detections' cand_ids
   (snapshot_combined_view.py:254). A per-tier survivors row's `best_cand_id` names its best MEMBER SOURCE
   (`<identity>_<cell>_<band>`), never a detection, so the condition is never true: such a source keeps the omitted detection's
   `best_snr`, `ts` and `peak_jy_best`, and the queue ranks it by them. The kept product (deep_omission.per_tier_snapshot) carries
   the same rule. The build counts these sources (record `stale_best`) and the tests pin the behavior. For production's owner
   first, then Gregg.
2. Partial sources keep their survivor gates (deep_omission.py, OWED NOT DONE; DECISIONS_PENDING "Deep omission: the
   survivor-gate re-run of a partial source").
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

- The re-point (1 above): production's fix first, then the port follows; the pinned test changes on purpose.
- Whether the view should also refuse a per-tier page detection that is not among its identity's kept detections (the build
  counts them; zero would make it a gate).
