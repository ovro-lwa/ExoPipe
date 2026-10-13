"""The shared substrate of Phase 3 (PHASE3_ARCHITECTURE.md section 2): the ONLY place each concern lives.

record (run records, resume), validate (content-test registry), cells (item grammar), runner (multi-node jobs),
exclusions (the only reader of data_decisions.csv), products (naming + standard headers), astrometry (the two-tier
contract), fitsio (atomic writes, node-local scratch), qa (percentile QA), env (process environment defaults).
"""
