"""Train the NumPy TinySAM implementation on Oxford-IIIT Pets."""

import argparse
import csv
import math
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tinysam.config import NUMPY_TINY, TinySAMConfig
from tinysam.data import PetsDataset, sample_prompt
from tinysam.metrics import mean_iou
from tinysam.numpy_impl.losses import tiny_sam_loss
from tinysam.numpy_impl.model import TinySAM
from tinysam.numpy_impl.optim import AdamW
from tinysam.torch_impl.convert import save_numpy_weights


def config_for_size(size, patch):
    return TinySAMConfig(img_size=size, patch=patch, enc_dim=64, enc_depth=4, enc_heads=4,
                         emb_dim=64, dec_depth=2, dec_heads=4)


def downsample_masks(masks, out_size):
    factor = masks.shape[-1] // out_size
    return (masks.reshape(len(masks), out_size, factor, out_size, factor)
            .mean(axis=(2, 4)) >= 0.5).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "data"))
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--patch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--name", default="numpy-tinysam")
    parser.add_argument("--out-dir", default=str(ROOT / "runs"))
    parser.add_argument("--val-subset", type=int, default=128)
    args = parser.parse_args()
    config = NUMPY_TINY if (args.size, args.patch) == (64, 8) else config_for_size(args.size, args.patch)
    rng = np.random.default_rng(args.seed)
    train = PetsDataset("train", args.size, args.root, augment=True, seed=args.seed)
    val = PetsDataset("val", args.size, args.root, seed=args.seed + 1)
    model = TinySAM(config, seed=args.seed)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    steps_epoch = args.steps_per_epoch or math.ceil(len(train) / args.batch_size)
    total_steps = args.max_steps or args.epochs * steps_epoch
    output = Path(args.out_dir) / args.name
    output.mkdir(parents=True, exist_ok=True)
    losses, validation, step = [], [], 0
    gradients = {name: _named_gradient(model, name) for name in model.parameters()}
    start = time.time()
    log_path = output / "loss.csv"
    with log_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step", "epoch", "loss", "lr"])
        for epoch in range(args.epochs):
            order = rng.permutation(len(train))
            epoch_loss = []
            for offset in range(0, min(len(order), steps_epoch * args.batch_size), args.batch_size):
                if step >= total_steps:
                    break
                indices = order[offset:offset + args.batch_size]
                if len(indices) < args.batch_size:
                    continue
                images, masks, points, labels = train.batch(indices, rng)
                mask_logits, iou_pred = model.forward(images, points, labels)
                target = downsample_masks(masks, mask_logits.shape[-1])
                loss, dmask, diou = tiny_sam_loss(mask_logits, iou_pred, target)
                model.zero_grad()
                model.backward(dmask, diou)
                warmup = min(1.0, (step + 1) / max(args.warmup_steps, 1))
                progress = step / max(total_steps, 1)
                rate = args.lr * warmup * 0.5 * (1 + math.cos(math.pi * progress))
                optimizer.step(gradients, rate)
                losses.append(loss)
                epoch_loss.append(loss)
                step += 1
                writer.writerow([step, epoch + 1, loss, rate])
                csv_file.flush()
                if step % 10 == 0:
                    print(f"step={step}/{total_steps} loss={loss:.5f} lr={rate:.3g}", flush=True)
            if not epoch_loss:
                break
            val_count = min(args.val_subset, len(val))
            val_indices = rng.choice(len(val), val_count, replace=False)
            predictions, targets = [], []
            for index in val_indices:
                image, mask = val[index]
                points, labels = sample_prompt(mask, rng)
                logits, quality = model.forward(image[None], points[None], labels[None])
                best = int(quality[0].argmax())
                pred = logits[0, best] > 0
                predictions.append(pred)
                targets.append(downsample_masks(mask[None], pred.shape[-1])[0] > 0)
            val_score = mean_iou(predictions, targets)
            validation.append(val_score)
            print(f"epoch={epoch + 1} train_loss={np.mean(epoch_loss):.5f} "
                  f"val_mIoU={val_score:.4f} elapsed={time.time()-start:.1f}s")
            save_numpy_weights(model, output / "weights.npz")
    save_numpy_weights(model, output / "weights.npz")
    with (output / "val_metrics.csv").open("w", newline="") as metric_file:
        writer = csv.writer(metric_file)
        writer.writerow(["epoch", "val_miou"])
        writer.writerows(enumerate(validation, 1))
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(losses) + 1), losses)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(output / "loss_curve.png", dpi=140)
    print(f"weights={output / 'weights.npz'}\nloss_curve={output / 'loss_curve.png'}")


def _named_gradient(model, name):
    # Each leaf parameter has a matching leaf gradient in the owning Module.
    parts = name.split(".")
    current = model
    for part in parts[:-1]:
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current.grads[parts[-1]]


if __name__ == "__main__":
    main()
