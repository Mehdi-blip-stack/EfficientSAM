"""Evaluate test-set mIoU using deterministic interior-point or exact-box prompts."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as torch_functional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tinysam.config import NUMPY_TINY, TORCH_SMALL, TinySAMConfig
from tinysam.data import PetsDataset
from tinysam.metrics import mean_iou
from tinysam.numpy_impl.model import TinySAM as NumpyTinySAM
from tinysam.torch_impl.convert import load_numpy_weights, load_torch_weights
from tinysam.torch_impl.model import TinySAM as TorchTinySAM


def chamfer_distance(mask):
    """Two-pass 8-neighbor chamfer transform from background pixels."""
    distance = np.where(mask, 1e6, 0.0).astype(np.float64)
    diagonal = np.sqrt(2.0)
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if y:
                distance[y, x] = min(distance[y, x], distance[y - 1, x] + 1)
                if x:
                    distance[y, x] = min(distance[y, x], distance[y - 1, x - 1] + diagonal)
                if x + 1 < width:
                    distance[y, x] = min(distance[y, x], distance[y - 1, x + 1] + diagonal)
            if x:
                distance[y, x] = min(distance[y, x], distance[y, x - 1] + 1)
    for y in range(height - 1, -1, -1):
        for x in range(width - 1, -1, -1):
            if y + 1 < height:
                distance[y, x] = min(distance[y, x], distance[y + 1, x] + 1)
                if x:
                    distance[y, x] = min(distance[y, x], distance[y + 1, x - 1] + diagonal)
                if x + 1 < width:
                    distance[y, x] = min(distance[y, x], distance[y + 1, x + 1] + diagonal)
            if x + 1 < width:
                distance[y, x] = min(distance[y, x], distance[y, x + 1] + 1)
    return distance


def deterministic_prompt(mask, prompt):
    if prompt == "point":
        y, x = np.unravel_index(np.argmax(chamfer_distance(mask)), mask.shape)
        return np.asarray([[[x, y], [0, 0], [0, 0]]], dtype=np.float32), np.asarray([[1, -1, -1]])
    ys, xs = np.where(mask)
    points = np.zeros((1, 3, 2), dtype=np.float32)
    points[0, 0] = [xs.min(), ys.min()]
    points[0, 1] = [xs.max(), ys.max()]
    return points, np.asarray([[2, 3, -1]])


def resize_bilinear(image, height, width):
    """NCHW bilinear resize with half-pixel centers, for display/evaluation only."""
    b, c, old_h, old_w = image.shape
    y = np.clip((np.arange(height) + 0.5) * old_h / height - 0.5, 0, old_h - 1)
    x = np.clip((np.arange(width) + 0.5) * old_w / width - 0.5, 0, old_w - 1)
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    y1, x1 = np.minimum(y0 + 1, old_h - 1), np.minimum(x0 + 1, old_w - 1)
    wy, wx = y - y0, x - x0
    top = image[:, :, y0[:, None], x0[None, :]] * (1 - wx)[None, None, None, :] \
        + image[:, :, y0[:, None], x1[None, :]] * wx[None, None, None, :]
    bottom = image[:, :, y1[:, None], x0[None, :]] * (1 - wx)[None, None, None, :] \
        + image[:, :, y1[:, None], x1[None, :]] * wx[None, None, None, :]
    return top * (1 - wy)[None, None, :, None] + bottom * wy[None, None, :, None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("numpy", "torch"), required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--prompt", choices=("point", "box"), default="point")
    parser.add_argument("--root", default=str(ROOT / "data"))
    parser.add_argument("--size", type=int)
    parser.add_argument("--patch", type=int)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--tiny", action="store_true", help="Load a compact torch smoke-test checkpoint")
    args = parser.parse_args()
    size, patch = args.size or (64 if args.impl == "numpy" else 256), args.patch or (8 if args.impl == "numpy" else 16)
    config = (TinySAMConfig(img_size=size, patch=patch, enc_dim=32, enc_depth=1, enc_heads=4,
                            emb_dim=32, dec_depth=1, dec_heads=4) if args.tiny else
              NUMPY_TINY if (size, patch) == (64, 8) else TORCH_SMALL if (size, patch) == (256, 16)
              else TinySAMConfig(img_size=size, patch=patch, enc_dim=64, enc_depth=4, enc_heads=4,
                                 emb_dim=64, dec_depth=2, dec_heads=4))
    dataset = PetsDataset("test", size, args.root)
    if args.impl == "numpy":
        model = NumpyTinySAM(config)
        load_numpy_weights(model, args.weights)
    else:
        model = TorchTinySAM(config)
        load_torch_weights(model, args.weights)
        model.eval()
    predictions, targets = [], []
    count = min(len(dataset), args.max_images) if args.max_images else len(dataset)
    for index in range(count):
        image, target = dataset[index]
        points, labels = deterministic_prompt(target > 0.5, args.prompt)
        if args.impl == "numpy":
            logits, quality = model.forward(image[None], points, labels)
            best = int(np.argmax(quality[0]))
            upsampled = resize_bilinear(logits[:, best:best + 1], size, size)[0, 0]
        else:
            with torch.no_grad():
                logits, quality = model(torch.from_numpy(image[None]), torch.from_numpy(points),
                                        torch.from_numpy(labels))
                best = int(quality[0].argmax())
                upsampled = torch_functional.interpolate(
                    logits[:, best:best + 1], size=(size, size), mode="bilinear", align_corners=False
                )[0, 0].numpy()
        predictions.append(upsampled > 0)
        targets.append(target > 0.5)
    score = mean_iou(predictions, targets)
    result = {"impl": args.impl, "prompt": args.prompt, "images": count, "miou": score}
    print(json.dumps(result, indent=2))
    output = Path(args.output) if args.output else Path(args.weights).parent / f"evaluate_{args.impl}_{args.prompt}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
