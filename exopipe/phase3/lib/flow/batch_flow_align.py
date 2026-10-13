#!/usr/bin/env python3
"""Batch optical flow alignment — runs in py310_orca to amortize JIT compilation.

Takes a JSON manifest file listing (target, reference, out_dir) entries.
Processes all pairs in one JAX session so the ~43s JIT compilation happens once.

Usage:
  conda run -n py310_orca python batch_flow_align.py --manifest manifest.json

Location (GC-PRE-2, Gregg ruling 2026-08-12): exopipe/common/ — flow is
cross-phase CORE INFRASTRUCTURE (Phase 2: dewarp alternative to RBF;
Phase 3: alignment for sky subtraction), not a tools/ utility and not attic.
Byte-identical to the frozen clv15 original (md5 a835e442dc7e3dbd6cabd4f6838b4810).

Subprocess call sites (this script is always SHELLED into py310_orca, never
imported — JAX lives only in that env):
  - exopipe/phase3/coadd_full_pipeline.py  BATCH_FLOW_SCRIPT (module const)
    -> subprocess.run(["conda","run","-n","py310_orca",...,"--manifest",...])
    in the no-stage batch-flow phase (GPU-locked).
  - exopipe/phase3/phase3_pipeline.py      BATCH_FLOW_SCRIPT (module const)
    -> subprocess.run(...) with a per-run manifest (10-min flow batch).
Both constants resolve <repo>/exopipe/common/batch_flow_align.py.
"""
import argparse
import json
import os
import sys
import time
import numpy as np

os.environ.setdefault("JAX_COMPILATION_CACHE_DIR",
                       os.path.expanduser("~/.cache/jax"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True,
                        help='JSON manifest: list of {target, reference, out_dir}')
    parser.add_argument('--alpha', type=float, default=1.3)
    parser.add_argument('--gamma', type=float, default=150.0)
    parser.add_argument('--scale_factor', type=float, default=0.7)
    parser.add_argument('--preprocess_weight', type=float, default=1.5)
    args = parser.parse_args()

    with open(args.manifest) as f:
        pairs = json.load(f)

    n = len(pairs)
    if n == 0:
        print("batch_flow_align: empty manifest, nothing to do")
        return

    print(f"batch_flow_align: {n} pairs to process")

    try:
        from image_plane_correction.flow import calcflow
    except ImportError as e:
        print(f"ERROR: Cannot import image_plane_correction: {e}")
        print("This script must run in the py310_orca environment.")
        sys.exit(1)

    t_total = time.time()
    success = 0
    skipped = 0

    for i, entry in enumerate(pairs):
        target = entry['target']
        reference = entry['reference']
        out_dir = entry['out_dir']
        os.makedirs(out_dir, exist_ok=True)

        tgt_base = os.path.splitext(os.path.basename(target))[0]
        dewarp_path = os.path.join(out_dir, f"{tgt_base}_dewarp.fits")
        flow_path = os.path.join(out_dir, f"{tgt_base}_flow.npy")

        if os.path.exists(dewarp_path) and os.path.exists(flow_path):
            print(f"  [{i+1}/{n}] SKIP: {tgt_base}")
            skipped += 1
            success += 1
            continue

        if os.path.exists(dewarp_path):
            os.remove(dewarp_path)

        t0 = time.time()
        try:
            result = calcflow(
                target,
                reference_sky_fn=reference,
                qa=False,
                write=True,
                outroot=out_dir,
                preprocess_weight=args.preprocess_weight,
                alpha=args.alpha,
                gamma=args.gamma,
                scale_factor=args.scale_factor,
            )

            flow = result[2]
            offsets = np.asarray(flow.offsets)
            np.save(flow_path, offsets)

            mag = np.sqrt(offsets[:, :, 0]**2 + offsets[:, :, 1]**2)
            h, w = mag.shape
            cy, cx = h // 2, w // 2
            yy, xx = np.ogrid[:h, :w]
            circle = (xx - cx)**2 + (yy - cy)**2 < (min(h, w) * 0.4)**2
            med_shift = float(np.median(mag[circle]))

            dt = time.time() - t0
            print(f"  [{i+1}/{n}] {tgt_base}: "
                  f"med_shift={med_shift:.3f}px  {dt:.1f}s")
            success += 1

        except Exception as e:
            dt = time.time() - t0
            print(f"  [{i+1}/{n}] FAILED {tgt_base}: {e}  ({dt:.1f}s)")

    elapsed = time.time() - t_total
    print(f"\nbatch_flow_align: {success}/{n} succeeded "
          f"({skipped} skipped) in {elapsed:.0f}s")


if __name__ == '__main__':
    main()
