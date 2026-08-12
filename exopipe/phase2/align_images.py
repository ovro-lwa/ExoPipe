#!/usr/bin/env python3
"""
align_images.py  Wrapper for image-plane-correction optical flow

Thin script meant to be called via subprocess from py38_orca_nkosogor.
Runs in py310_orca environment which has JAX and the flow code.

Aligns a target image to a reference image using Brox optical flow,
writes the dewarped target and optionally the flow field.

Usage (single pair):
  conda run -n py310_orca python align_images.py \
      --target target.fits \
      --reference reference.fits \
      --out_dir QA/aligned/

Usage (batch — amortizes JAX JIT compilation, ~43s/shape, across all
pairs of the same image shape in ONE process; actual flow ~0.2s/pair):
  conda run -n py310_orca python align_images.py --pairs pairs.txt
  # pairs.txt lines: <target.fits> <reference.fits> <out_dir>
  # blank lines and #-comments ignored

Outputs:
  {out_dir}/{target_basename}_dewarp.fits    dewarped target
  {out_dir}/{target_basename}_flow.npy       flow offsets (H,W,2) array
"""
import argparse
import os
import sys
import time

# Persist JAX JIT compilation cache across processes  avoids ~43s
# recompilation overhead on every subprocess call.
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR",
                       os.path.expanduser("~/.cache/jax"))

import numpy as np


def align_one(target, reference, out_dir, preprocess_weight=1.5,
              alpha=1.3, gamma=150.0, scale_factor=0.7, qa=True,
              save_flow=False):
    """Align one target to a reference. Returns True on QA pass.
    Calling repeatedly within one process reuses the JIT-compiled
    kernels for each input shape."""
    from image_plane_correction.flow import calcflow

    os.makedirs(out_dir, exist_ok=True)
    print(f"  Target:    {os.path.basename(target)}")
    print(f"  Reference: {os.path.basename(reference)}")

    # Remove any existing output to prevent overwrite errors from calcflow
    # (calcflow's write=True doesn't pass overwrite=True to astropy)
    tgt_base = os.path.splitext(os.path.basename(target))[0]
    expected_output = os.path.join(out_dir, f"{tgt_base}_dewarp.fits")
    if os.path.exists(expected_output):
        os.remove(expected_output)
        print(f"  Removed existing: {os.path.basename(expected_output)}")

    result = calcflow(
        target,
        reference_sky_fn=reference,
        qa=qa,
        write=True,
        outroot=out_dir,
        preprocess_weight=preprocess_weight,
        alpha=alpha,
        gamma=gamma,
        scale_factor=scale_factor,
    )

    if qa and len(result) == 5:
        image, reference_sky, flow, dewarped, qa_passed = result
        print(f"  QA: {'PASSED' if qa_passed else 'FAILED'}")
    else:
        image, reference_sky, flow, dewarped = result[:4]
        qa_passed = True

    # Report flow statistics
    offsets = np.asarray(flow.offsets)
    magnitude = np.sqrt(offsets[:, :, 0]**2 + offsets[:, :, 1]**2)

    # Mask outside a central circle (horizon region has no signal)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    circle = (xx - cx)**2 + (yy - cy)**2 < (min(h, w) * 0.4)**2
    mag_valid = magnitude[circle]

    print(f"  Flow stats (inside FoV):")
    print(f"    Median shift: {np.median(mag_valid):.3f} pix")
    print(f"    MAD shift:    {np.median(np.abs(mag_valid - np.median(mag_valid))) * 1.4826:.3f} pix")
    print(f"    Max shift:    {np.max(mag_valid):.3f} pix")
    print(f"    Mean dx:      {np.mean(offsets[circle, 0]):.4f} pix")
    print(f"    Mean dy:      {np.mean(offsets[circle, 1]):.4f} pix")

    if save_flow:
        flow_path = os.path.join(out_dir, f"{tgt_base}_flow.npy")
        np.save(flow_path, offsets)
        print(f"  Flow saved: {flow_path}")

    dewarp_path = os.path.join(out_dir,
                               os.path.basename(target).replace(
                                   '.fits', '_dewarp.fits'))
    if os.path.exists(dewarp_path):
        print(f"  Dewarped: {dewarp_path}")
    else:
        print(f"  WARNING: Expected output not found at {dewarp_path}")
        alt = os.path.join(out_dir, f"{tgt_base}_dewarp.fits")
        if os.path.exists(alt):
            print(f"  Found at: {alt}")
    return qa_passed


def main():
    parser = argparse.ArgumentParser(
        description="Align target to reference via optical flow")
    parser.add_argument("--target", help="Target image FITS path")
    parser.add_argument("--reference", help="Reference image FITS path")
    parser.add_argument("--out_dir", help="Output directory")
    parser.add_argument("--pairs",
                        help="Batch file: '<target> <reference> <out_dir>' "
                             "per line; loops in-process so JIT compiles "
                             "once per image shape")
    parser.add_argument("--preprocess_weight", type=float, default=1.5,
                        help="Preprocessing power-law weight")
    parser.add_argument("--alpha", type=float, default=1.3,
                        help="Brox flow smoothness parameter")
    parser.add_argument("--gamma", type=float, default=150.0,
                        help="Brox flow gradient constancy weight")
    parser.add_argument("--scale_factor", type=float, default=0.7,
                        help="Brox flow pyramid scale factor")
    parser.add_argument("--no_qa", action='store_true',
                        help="Skip QA check")
    parser.add_argument("--save_flow", action='store_true',
                        help="Save flow field as .npy")
    args = parser.parse_args()

    if not args.pairs and not (args.target and args.reference
                               and args.out_dir):
        parser.error("either --pairs or all of "
                     "--target/--reference/--out_dir required")

    try:
        from image_plane_correction.flow import calcflow  # noqa: F401
    except ImportError as e:
        print(f"ERROR: Cannot import image_plane_correction: {e}")
        print("This script must run in the py310_orca environment.")
        sys.exit(1)

    kw = dict(preprocess_weight=args.preprocess_weight, alpha=args.alpha,
              gamma=args.gamma, scale_factor=args.scale_factor,
              qa=not args.no_qa, save_flow=args.save_flow)
    print(f"align_images.py")
    print(f"  Parameters: alpha={args.alpha}, gamma={args.gamma}, "
          f"scale_factor={args.scale_factor}, "
          f"weight={args.preprocess_weight}")

    if args.pairs:
        pairs = []
        with open(args.pairs) as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln or ln.startswith('#'):
                    continue
                parts = ln.split()
                if len(parts) != 3:
                    print(f"  SKIP malformed line: {ln}")
                    continue
                pairs.append(parts)
        print(f"  Batch: {len(pairs)} pairs")
        n_fail = 0
        for k, (tgt, ref, od) in enumerate(pairs):
            t0 = time.time()
            print(f"\n[{k + 1}/{len(pairs)}]")
            try:
                ok = align_one(tgt, ref, od, **kw)
            except Exception as e:
                print(f"  ERROR: {e}")
                ok = False
            n_fail += (not ok)
            print(f"  Time: {time.time() - t0:.1f}s")
        print(f"\n  STATUS: {'OK' if n_fail == 0 else f'{n_fail} FAILED'}")
        sys.exit(0 if n_fail == 0 else 2)
    else:
        t0 = time.time()
        ok = align_one(args.target, args.reference, args.out_dir, **kw)
        print(f"  Time: {time.time() - t0:.1f}s")
        print(f"  STATUS: {'OK' if ok else 'QA_FAILED'}")


if __name__ == "__main__":
    main()
