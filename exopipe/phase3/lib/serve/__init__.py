"""lib/serve - the ported orchestration of `exopipe3 serve` (steps/serve.py is the entry point).

Modules (each ports named production launcher logic; the production file:line is cited at every port):
  common.py         small shared helpers: UTC strings, the run's incremental.log, CSV reading exactly as the launchers read
  scripts.py        where the in-place exopipe_ops scripts are; every Python one runs through core.legacy.run (decision 1)
  procs.py          /proc scans, port listeners, exact-pid stop, server start, readiness, index source links
  plan.py           the Plan: tier, served run directory (production or parity mirror), pages, ports, argv, environment
  mirror.py         the parity mirror of a served run (files serve rewrites copied, everything else symlinked)
  queues.py         queue sectioning and the multiples / multi-integration queues (ports of the launchers' heredocs)
  views.py          the per-Stokes catalogue views and unions (ports of the launchers' heredocs; D9 residual option)
  exclusion_gate.py the serve-time exclusion gate for every tier (tier row keyers + core.exclusions; in-place snapshot gate)
  journals.py       judgement carry (in place), HOLD rule (in process), sub-LST journal copy / swap / append
  hub.py            review hub registry (locked; in place) and inspection hub
  servers.py        replace / twin / restart / stop of review servers and the served_page records
  flow.py           the stages shared by the tiers (test, start, twins, hub, verify), the record chain, the dry-run header
  deep.py, sublst_build.py, snapshot_copy.py   the review pages' stage sequences per tier
  forced.py         the snapshot forced-photometry pages (port of snapshot_forced_serve_20261007.sh)
  jupiter.py        the Jupiter program's pages 8800 / 8801
  app_probe.py      run as a SCRIPT (no exopipe import): the server's own loader + App on one page -> expected sources
  tests.py          content tests: served_queue, served_view, judgement_journal, hub_page, served_page, stopped_page,
                    sublst_cumulative, review_copy, exclusion_check
"""
