# PUBLISH_CHECKLIST.md — ExoPipe public release preparation

Status 2026-08-12: PREPARATION ONLY. No remote configured, nothing pushed.
Nothing outward-facing happens without Gregg's explicit sign-off on each
DECISION item below. /lustre/gh/main/Public (stale v1.0) is superseded by
this release and is not touched by it.

## DECISION 1 — License (Gregg's call)

| Option | What it implies |
|---|---|
| **BSD-3-Clause** | Permissive; the astropy/radio-astronomy community default; allows commercial/derivative use with attribution; "no endorsement" clause protects Caltech/OVRO-LWA names. Simplest fit with likely dependencies. |
| **MIT** | Maximally permissive and shortest; equivalent in practice to BSD-3 minus the no-endorsement clause. |
| **GPLv3** | Copyleft: derivatives must remain open under GPL. Strongest guarantee the code stays open; can deter institutional reuse and complicates linking with permissively-licensed forks. |

Note: Caltech institutional policy may have a say — worth a one-line check
with OTL before tagging. No license file ships until ruled; README states
"all rights reserved" in the interim.

## DECISION 2 — Repo name and organization (Gregg's call)

- The plan of record (PHASE3_ARCHITECTURE.md §12.1) already names the
  target: **ovro-lwa/ExoPipe** ("Publish: fresh v2.0 history to
  ovro-lwa/ExoPipe"). The github.com/ovro-lwa organization exists and hosts
  the related distributed-pipeline repo.
- Alternatives if Gregg wants them: `ovro-lwa/exopipe` (lowercase matches
  the package name) or a personal-org home first, transferred later.
  Recommendation: follow the plan of record — ovro-lwa/ExoPipe.

## DECISION 3 — History: squashed v2.0 vs full history (recommendation: fresh/squashed)

Facts (verified 2026-08-12 on the 5-commit repo at d4889d3):
- Commit MESSAGES/bodies are clean: zero occurrences of /lustre, /fast,
  /data07, lwacalim, or clv15 in any message.
- Commit CONTENT is not: `git log -p` carries 261 internal-path lines
  (module docstrings citing /lustre/gh/main/clv15 origins, path constants
  since edited, plus the working tree's current 226 path references — see
  hygiene table below). Full history also encodes the internal migration
  sequence (disposition table names, golden-cell references).
- The plan of record already ratifies the answer: §12.1 "fresh v2.0
  history". **Recommend: initialize a fresh public history (single v2.0
  root commit, or squash to it) once the hygiene items below are cleared;
  keep the internal repo as the private full-history record.** Author
  identity on the public root commit: Gregg's choice of name/email.

## DECISION 4 — CI

- Add GitHub Actions workflow running
  `tests/test_import_side_effects.py` (AST layer; the strace layer stays
  opt-in/off in CI) on push + PR, over the supported Python versions
  (tree currently exercises 3.8 and 3.12 bytecode locally).
- Draft workflow file NOT added yet (kept out of this commit so the repo
  gains no CI behavior before Gregg rules on it).

## Hygiene sweep results (2026-08-12, tree at d4889d3)

**(a) Secrets/credentials: CLEAN.** No tokens, keys, key-like files
(.pem/.netrc/id_rsa/credentials), or embedded passwords. The only hits are
the email-notification code reading an EXTERNAL secrets file
(`cfg.SECRETS_FILE`, expanduser'd, outside the repo) in
phase1/post_process_science.py and phase2/phase2_science_agg.py — correct
pattern, nothing embedded.

**(b) Hardcoded absolute paths / hostnames (tracked .py):**

| Pattern | Total refs |
|---|---|
| /lustre | 73 |
| /fast | 89 |
| /data07 | 63 |
| /opt/devel | 1 |
| lwacalim | 23 |

Classification:
- CONFIG (acceptable if these remain the single source): the three path/
  config modules concentrate 35 path refs + 15 of 23 lwacalim refs
  (common/pipeline_config.py 18+11, common/phase3_paths.py 6+2,
  phase3/followup/config.py 11+2). These should become env/config-file
  driven before publish, but they are the sanctioned location.
- VIOLATIONS of CLAUDE.md rule 8 (hostnames/paths in production modules),
  worst offenders: phase3/mosaic_conformance_pass.py (12 path refs),
  phase3/phase3_pbcorr_residuals.py (10), phase3/phase3_i_deep.py (9),
  phase3/phase3_final_csv.py (9), phase1/process_subband.py (9); lwacalim
  in 6 production modules (8 refs: generate_dashboard,
  generate_qa_dashboard, phase3_hwj_snap, phase3_sb_assemble ×2 — one is a
  Run_SB1_lwacalim0* run-label glob, phase_status, phase3_tenmin_assemble
  ×2). ~190 path refs sit outside the config modules overall.
- ACTION before publish: sweep the ~190 non-config path refs into
  exopipe.common config (defaults overridable by env), and the 8
  production-module lwacalim refs to config/runtime lookup. This is a
  ratifiable code-change task for Gregg — not done silently here.

**(c) Data files / large binaries: CLEAN.** No files >1 MB outside .git;
only .py + three .md/.gitignore tracked; reference/ and tools/ are empty
(their content arrives at retirement stage R2). __pycache__ untracked and
gitignored. Re-check at R2 when tools/reference populate.

**(d) Emails/PII:** three addresses in code —
`gh@astro.caltech.edu`, `nkosogor@caltech.edu`
(EMAIL_RECIPIENT constants in common/phase2_config.py:207 and
phase1/post_process_science.py:50) and `pipeline@astro.caltech.edu`
(default From). ACTION: move recipient lists into the external secrets/
config file before publish; institutional From default is arguably fine —
Gregg's call. No other PII found.

## Pre-push gate (all boxes, in order; each needs Gregg's sign-off where marked)

- [ ] License chosen (DECISION 1 — Gregg) and LICENSE file added
- [ ] Repo name/org confirmed (DECISION 2 — Gregg)
- [ ] History mode ruled (DECISION 3 — Gregg); if fresh: build the v2.0
      root commit and verify `git log -p` of the PUBLIC repo is
      internal-path-clean
- [ ] Path/hostname sweep to config completed and golden-revalidated
- [ ] EMAIL_RECIPIENT constants externalized
- [ ] Golden cells 4/4 PASS + GC-PRE-1..4 ruled (repo content is not final
      until GC-PRE-1's V-core re-absorb is resolved — the tree currently
      carries the stale 952-line dev copy of search_v_transients_v2)
- [ ] docs/ populated per retirement stage R3 (architecture doc curated for
      publish: strip internal paths/run numbers)
- [ ] tools/ + reference/ populated (retirement R2) and re-swept ((c) above)
- [ ] CI workflow added (DECISION 4)
- [ ] Final full-tree secrets/path re-scan on the exact tree to be pushed
- [ ] Gregg's named sign-off to push — NOTHING OUTWARD WITHOUT IT
