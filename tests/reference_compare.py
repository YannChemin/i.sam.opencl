#!/usr/bin/env python3
"""Compare i.sam.opencl debug dumps with the PyTorch segment_anything code.

Run i.sam.opencl with the environment variable I_SAM_OPENCL_DUMP set to a
directory; it then writes, for each crop, the 8-bit crop it segmented, the
normalized encoder input and the image embedding, plus the final mask list
(masks.csv). This script runs the reference implementation on the same
8-bit pixels and reports:

1. the maximum difference between the reference preprocessing
   (SamPredictor.set_image) and the dumped encoder input,
2. the relative error of the image embedding,
3. how the automatically generated masks match (box IoU pairing).

Requires torch and segment_anything (not needed by the module itself).

Usage:
    reference_compare.py DUMP_DIR CHECKPOINT [--points N] [--minsize N]
        [--iou 0.88] [--stability 0.95]
"""

import argparse
import glob
import os
import re

import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def load_crop(dump_dir):
    """Return the first crop's (rgb, input, embedding) arrays."""
    rgb_path = min(glob.glob(os.path.join(dump_dir, "crop000_rgb_*.u8")))
    h, w = map(int, re.search(r"_(\d+)x(\d+)\.u8$", rgb_path).groups())
    rgb = np.fromfile(rgb_path, dtype=np.uint8).reshape(h, w, 3)
    inp = np.fromfile(os.path.join(dump_dir, "crop000_input.f32"), np.float32)
    emb = np.fromfile(os.path.join(dump_dir, "crop000_embedding.f32"), np.float32)
    return rgb, inp.reshape(3, 1024, 1024), emb.reshape(64, 64, 256)


def box_iou(a, b):
    """IoU of inclusive xyxy boxes, as torchvision computes it."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dump_dir")
    parser.add_argument("checkpoint")
    parser.add_argument("--model", default="vit_h")
    parser.add_argument("--points", type=int, default=32)
    parser.add_argument("--minsize", type=int, default=100)
    parser.add_argument("--iou", type=float, default=0.88)
    parser.add_argument("--stability", type=float, default=0.95)
    args = parser.parse_args()

    rgb, inp, emb = load_crop(args.dump_dir)
    sam = sam_model_registry[args.model](checkpoint=args.checkpoint).eval()
    gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points,
        pred_iou_thresh=args.iou,
        stability_score_thresh=args.stability,
        min_mask_region_area=args.minsize,
    )

    with torch.no_grad():
        gen.predictor.set_image(rgb)
        ref_emb = gen.predictor.features[0].permute(1, 2, 0).numpy()
        resized = gen.predictor.transform.apply_image(rgb)
        ref_inp = sam.preprocess(
            torch.as_tensor(resized).permute(2, 0, 1)[None].float()
        )[0].numpy()

    d_inp = np.abs(ref_inp - inp).max()
    rel = np.linalg.norm(ref_emb - emb) / np.linalg.norm(ref_emb)
    cos = float((ref_emb * emb).sum() / (np.linalg.norm(ref_emb) * np.linalg.norm(emb)))
    print(f"encoder input: max abs difference {d_inp:.3g}")
    print(
        f"embedding: relative L2 error {rel:.3g}, cosine {cos:.6f}, "
        f"max abs difference {np.abs(ref_emb - emb).max():.3g}"
    )

    with torch.no_grad():
        ref = gen.generate(rgb)
    ours = np.loadtxt(
        os.path.join(args.dump_dir, "masks.csv"), delimiter=",", skiprows=1, ndmin=2
    )
    print(f"masks: reference {len(ref)}, i.sam.opencl {len(ours)}")

    ref_boxes = [
        (
            m["bbox"][0],
            m["bbox"][1],
            m["bbox"][0] + m["bbox"][2],
            m["bbox"][1] + m["bbox"][3],
        )
        for m in ref
    ]
    matched = 0
    ious = []
    for row in ours:
        best = max((box_iou(row[:4], rb) for rb in ref_boxes), default=0.0)
        ious.append(best)
        matched += best > 0.9
    if ious:
        print(
            f"masks with a reference box IoU > 0.9: {matched}/{len(ours)} "
            f"(median best IoU {np.median(ious):.3f})"
        )


if __name__ == "__main__":
    main()
