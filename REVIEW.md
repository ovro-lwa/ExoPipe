# REVIEW.md - what matters when reviewing ExoPipe

Adopted 2026-11-08 (Gregg Hallinan: "all approved"). It follows the practices in
https://claude.dev/blog/solving-code-review-at-anthropic/: review effort matched to risk, findings that are verified before
they are posted, and a person accountable for every change.

## Risk tiers (set by what the change does, not by who wrote it)

| tier | examples | review before merge |
|---|---|---|
| Low | docs, comments, page wording or layout, test-only changes | self-review; CI green |
| Medium | a production code path, a new tool, a writer of products, serving code | one `/code-review` (medium); CI green; the cluster suites of the touched stages |
| High | anything that removes or restores detections, rewrites or deletes archived files, changes a science result (fluxes, positions, masters, thresholds) | multi-agent review with verified findings; Gregg's named approval (ledger id in the PR); a parity or content test on real data |

## Important: a finding of this kind blocks the merge
1. **A detection removed silently.** Every removal names its rule, which must be ruled by Gregg for that tier. Rules are
   recorded per detection, never as a bare count. A rule extended to a new tier is a NEW removal and needs his approval.
2. **A veto across time.** Vetoes act per instant, across frequency only: never on other times or other tiers.
3. **An exclusion that leaks.** data_decisions rows exclude in every tier, absolutely. The one exception is an excluded 18 MHz
   in Red / Full: it never kills the wideband, and an existing master keeps the night until its next rebuild.
4. **A product without its standard header.** That covers: the beam, HISTORY, the frequency block, its OWN time block
   (DATE-OBS / DATE-END / MJD-OBS / MJD-END / MJD-MID / DURATION / INTTIME, never its master's), noise, provenance and GITTAG.
   Its name must round-trip in `core/products.py`.
5. **An archived file overwritten or deleted in place.** A replacement goes to a quarantine Run with a checksum manifest,
   after a dry run. Archived deletions are Gregg's.
6. **Output left half-written.** Write to staging, re-read and verify (Lustre has made silent zero-byte copies), then rename
   into place. A failure leaves nothing behind.
7. **A refusal turned into a silent zero.** A stage that cannot do its job refuses loudly and records why. It never yields an
   empty result that reads as "searched, nothing found".
8. **A speed-up that changes results.** A speed change must be byte-identical on real cells. Parity is not correctness: a
   behaviour change is judged against the written goals.

## Also check
- **Lustre load:** at most 6 Lustre-heavy workers, and long jobs pause on the I/O guard (the recorders share Lustre).
- **Processes:** never `pkill -f` / `pgrep -f`; select by PID or exact name.
- **Language:** American English in text and messages.
- **Configuration:** values live in `config/steps/*.py` with the ruling that set them (`RULED`). A protected key changes only
  with a ruling.

## Severity in a review
- **Important:** any item of the list above, or a crash, wrong data or data loss.
- **Nit:** style or naming; never blocks.
- **Pre-existing:** found but not introduced by the change; becomes a ledger row, never a blocker.
Findings are verified before they are posted: a failure scenario with concrete inputs, or it is dropped.

## Skip
- `plan/` and other generated or recorded documents.
- Parity captures.
- Large data tables.

## Every pull request states (see .github/pull_request_template.md)
- what changed and why (the ruling or finding);
- the risk tier;
- the testing done: CI, the cluster suites with their pass counts, parity or content tests on real data;
- how to revert;
- the ledger rows.
