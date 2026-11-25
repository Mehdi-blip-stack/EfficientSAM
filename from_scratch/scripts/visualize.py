"""Save a grid of images, prompts, ground truth, and predicted masks."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as torch_functional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tinysam.config import NUMPY_TINY, TORCH_SMALL, TinySAMConfig
from tinysam.data import IMAGENET_MEAN, IMAGENET_STD, PetsDataset
from tinysam.numpy_impl.model import TinySAM as NumpyTinySAM
from tinysam.torch_impl.convert import load_numpy_weights, load_torch_weights
from tinysam.torch_impl.model import TinySAM as TorchTinySAM
from evaluate import deterministic_prompt, resize_bilinear


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("numpy", "torch"), required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--root", default=str(ROOT / "data"))
    parser.add_argument("--output", default=str(ROOT / "runs" / "visualize.png"))
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    config = (TinySAMConfig(img_size=64, patch=8, enc_dim=32, enc_depth=1, enc_heads=4,
                            emb_dim=32, dec_depth=1, dec_heads=4) if args.tiny else
              NUMPY_TINY if args.impl == "numpy" else TORCH_SMALL)
    dataset = PetsDataset("test", config.img_size, args.root)
    if args.impl == "numpy":
        model = NumpyTinySAM(config)
        load_numpy_weights(model, args.weights)
    else:
        model = TorchTinySAM(config)
        load_torch_weights(model, args.weights)
        model.eval()
    count = min(args.count, len(dataset))
    figure, axes = plt.subplots(count, 3, figsize=(9, 3 * count), squeeze=False)
    for index in range(count):
        image, target = dataset[index]
        points, labels = deterministic_prompt(target > 0.5, "point")
        if args.impl == "numpy":
            logits, quality = model(image[None], points, labels)
            best = int(quality[0].argmax())
            low = logits[0, best]
            pred = resize_bilinear(low[None, None], config.img_size, config.img_size)[0, 0] > 0
        else:
            with torch.no_grad():
                logits, quality = model(torch.from_numpy(image[None]), torch.from_numpy(points),
                                        torch.from_numpy(labels))
                best = int(quality[0].argmax())
                pred = torch_functional.interpolate(logits[:, best:best + 1], size=(config.img_size, config.img_size),
                                                     mode="bilinear", align_corners=False)[0, 0].numpy() > 0
        rgb = np.clip((image * IMAGENET_STD + IMAGENET_MEAN).transpose(1, 2, 0), 0, 1)
        axes[index, 0].imshow(rgb)
        axes[index, 0].scatter(points[0, labels[0] == 1, 0], points[0, labels[0] == 1, 1], c="lime", s=12)
        axes[index, 1].imshow(target, cmap="gray")
        axes[index, 2].imshow(rgb)
        axes[index, 2].imshow(pred, alpha=0.45, cmap="spring")
        for axis, title in zip(axes[index], ("Image + prompt", "Ground truth", "Prediction")):
            axis.set_title(title)
            axis.axis("off")
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=120)
    print(output)


if __name__ == "__main__":
    main()
