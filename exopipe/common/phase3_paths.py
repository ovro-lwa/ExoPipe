#!/usr/bin/env python3
"""
phase3_paths.py — canonical Phase 3 directory layout (PHASE3_STRUCTURE.md).

Staging on /fast MIRRORS the final /lustre tree so promotion is a plain copy.
One builder for every write site; nothing constructs paths ad hoc.

    {root}/{LST}/{date}/P3_{label}/{scope}/{Stokes}/{cadence}/    per-epoch
    {root}/{LST}/coadd/Run_{label}_{lo}-{hi}/{scope}/{Stokes}/{cadence}/   masters
    {root}/transients/Run_{label}/...                            survey-wide blind search
    {root}/samples/{sample}/{target}/                            targeted forced photometry

scope   ∈ {18MHz…82MHz, Red, Green, Blue, Full}
Stokes  ∈ {I, V}
cadence ∈ {deep, 10min}
"""
import os

STAGING_ROOT = os.environ.get('P3_STAGING_ROOT',
                              '/data07/gh/phase3')  # conformant FINAL products (lwacalim07;
                                                # /fast too small). Flat working staging
                                                # (WORKDIR_BASE) stays on /fast for I/O.
                                                # P3_STAGING_ROOT: staged-node override
                                                # (lwacalim01/02 have no /data07).
ARCHIVE_ROOT = '/lustre/pipeline/exopipe/phase3'  # promotion target


def epoch_dir(root, lst, date, label, scope, stokes, cadence, make=False):
    """Per-epoch product dir: aligned+scaled variants + final subtracted images."""
    p = os.path.join(root, lst, date, f'P3_{label}', scope, stokes, cadence)
    if make:
        os.makedirs(p, exist_ok=True)
    return p


def coadd_dir(root, lst, label, date_lo, date_hi, scope, stokes, cadence,
              make=False):
    """Co-add Run dir (sibling of date dirs; spans dates; date range in name)."""
    run = f'Run_{label}_{date_lo}-{date_hi}'
    p = os.path.join(root, lst, 'coadd', run, scope, stokes, cadence)
    if make:
        os.makedirs(p, exist_ok=True)
    return p


def coadd_run_root(root, lst, label, date_lo, date_hi, make=False):
    """The co-add Run root (holds QA csvs spanning scopes/Stokes)."""
    p = os.path.join(root, lst, 'coadd', f'Run_{label}_{date_lo}-{date_hi}')
    if make:
        os.makedirs(p, exist_ok=True)
    return p


def transients_run(root, label, make=False):
    """Survey-wide blind-search Run (spans ALL LSTs+dates; de-siloed)."""
    p = os.path.join(root, 'transients', f'Run_{label}')
    if make:
        os.makedirs(p, exist_ok=True)
    return p


def candidate_dir(root, label, cand_id, make=False):
    p = os.path.join(transients_run(root, label), 'candidates', cand_id)
    if make:
        os.makedirs(p, exist_ok=True)
    return p


def sample_dir(root, sample, target, make=False):
    """Targeted forced-photometry per (sample, target). sample e.g.
    solar_system, 10pc, Hot_Warm_Jupiters, Local_Volume."""
    safe = target.replace(' ', '_')
    p = os.path.join(root, 'samples', sample, safe)
    if make:
        os.makedirs(p, exist_ok=True)
    return p
