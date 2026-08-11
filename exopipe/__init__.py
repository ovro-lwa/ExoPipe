"""OVRO-LWA ExoPipe v2 — package root (PHASE3_ARCHITECTURE.md §12.0).

Layout: common/ (config, products, astrometry, exclusions, validate,
residuals, qa substrate — §2), phase1/, phase2/ (relocated, light cleanup),
phase3/ (six capabilities + followup/ subpackage — §1, §9).
The git tree here IS the GitHub tree; operational files live in
/lustre/gh/main/exopipe_ops/ and structurally cannot leak (§12.0).
"""
