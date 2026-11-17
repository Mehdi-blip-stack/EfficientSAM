"""Step B2: run the frozen SAM teacher ONCE over the 800 training images and cache what SAMI needs.

For every training image and two views (original, horizontally mirrored):
  * the student input: the image squashed to 256 x 256 (uint8)
  * the SAMI target: SAM's image-encoder output (256 x 64 x 64 at 1024 px input) average-pooled 4 x 4
    to the student's 16 x 16 token grid (fp16)
The teacher sees the SAME squashed image as the student (resized straight to 1024 x 1024, no padding), so every
student token lines up with the 4 x 4 block of teacher features it has to reconstruct.

Paper vs ours (feasibility, report Section B8): teacher SAM ViT-H as in the paper; 800 COCO train images and
2 views instead of ImageNet-1K with augmentation; features pooled to the student grid.

Usage (Colab):  python -m repro.teacher_features --checkpoint .../sam_vit_h_4b8939.pth --out .../features
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .common import GRID, STUDENT_SIZE, TEACHER_SIZE, data_root, squash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="SAM checkpoint; 'none' = random weights (code test only)")
    ap.add_argument("--model_type", default="vit_h")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from segment_anything import sam_model_registry

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=None if args.checkpoint == "none" else args.checkpoint)
    teacher = sam.image_encoder.to(dev).eval()
    mean = torch.tensor([123.675, 116.28, 103.53], device=dev).view(1, 3, 1, 1)   # SAM's own normalisation
    std = torch.tensor([58.395, 57.12, 57.375], device=dev).view(1, 3, 1, 1)

    meta = json.load(open(data_root() / "annotations" / "subset_instances_train2017.json"))
    images = sorted(meta["images"], key=lambda r: r["id"])
    if args.limit:
        images = images[: args.limit]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    n = len(images)
    student_in = np.zeros((n, 2, 3, STUDENT_SIZE, STUDENT_SIZE), np.uint8)
    targets = np.zeros((n, 2, 256, GRID, GRID), np.float16)

    t0 = time.time()
    with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
        for i, info in enumerate(images):
            img = np.array(Image.open(data_root() / "images" / "train2017" / info["file_name"]).convert("RGB"))
            for v, view in enumerate([img, img[:, ::-1]]):                          # original, mirrored
                student_in[i, v] = squash(view, STUDENT_SIZE)[0].round().byte().numpy()
                x = squash(view, TEACHER_SIZE).to(dev)
                feat = teacher((x - mean) / std)                                       # 1 x 256 x 64 x 64
                targets[i, v] = F.adaptive_avg_pool2d(feat.float(), GRID)[0].cpu().numpy().astype(np.float16)
            if (i + 1) % 50 == 0 or i + 1 == n:
                print(f"  {i + 1}/{n} images, {time.time() - t0:.0f}s", flush=True)

    t = targets.astype(np.float32)
    stats = {"channel_mean": t.mean(axis=(0, 1, 3, 4)).tolist(), "channel_std": t.std(axis=(0, 1, 3, 4)).tolist(),
             "image_ids": [r["id"] for r in images], "teacher": args.model_type, "seconds": time.time() - t0}
    np.save(out / "student_inputs_256.npy", student_in)
    np.save(out / f"teacher_{args.model_type}_16x16.npy", targets)
    json.dump(stats, open(out / f"teacher_{args.model_type}_stats.json", "w"))
    print(f"saved {n} images x 2 views: inputs {student_in.nbytes / 1e6:.0f} MB, targets {targets.nbytes / 1e6:.0f} MB "
          f"-> {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
