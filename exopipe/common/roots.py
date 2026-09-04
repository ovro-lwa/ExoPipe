"""Canonical data roots — REQUIRED from the environment, no defaults.

Gregg 2026-09-04: the superseded sigma-clipped co-add tree
(/fast/gh/main/phase3/coadd) was deleted and every module default that
pointed at it was removed ("remove the defaults"). A run that does not
state its co-add root explicitly must fail loudly here, never fall
through to a stale generation.
"""
import os


class RootUnset(RuntimeError):
    pass


def required_root(var, purpose):
    v = os.environ.get(var)
    if not v:
        raise RootUnset(
            f"{var} is not set ({purpose}). No default exists: the retired "
            f"/fast/gh/main/phase3/coadd tree was deleted 2026-09-04 and module "
            f"defaults were removed on Gregg's order. Export {var} to the "
            f"current-generation cell root (wideband: rebuild_w2/rebuild_v2 "
            f"on the node's /fast; subband: /data07/gh/phase3/coadd_sb).")
    return v


def coadd_root():
    """Root holding <lst>_<scope>/coadd_final/ and <lst>_<scope>/subtracted/."""
    return required_root('P3_COADD_ROOT', 'co-add cell root: masters + subtracted epochs')
