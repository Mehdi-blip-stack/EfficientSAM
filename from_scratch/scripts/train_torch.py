"""Train PyTorch TinySAM with the custom AdamW implementation."""

import argparse
import csv
import math
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tinysam.config import TORCH_SMALL, TinySAMConfig
from tinysam.data import PetsDataset, sample_prompt
from tinysam.torch_impl.losses import tiny_sam_loss
from tinysam.torch_impl.model import TinySAM
from tinysam.torch_impl.optim import AdamW
from tinysam.torch_impl.convert import save_torch_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "data"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--patch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init-encoder")
    parser.add_argument("--tiny", action="store_true", help="Use a smaller model for smoke tests")
    parser.add_argument("--name", default="torch-tinysam")
    parser.add_argument("--out-dir", default=str(ROOT / "runs"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-subset", type=int, default=64)
    args = parser.parse_args()
    config = (TinySAMConfig(img_size=args.size, patch=args.patch, enc_dim=32, enc_depth=1, enc_heads=4,
                            emb_dim=32, dec_depth=1, dec_heads=4)
              if args.tiny else TORCH_SMALL)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train = PetsDataset("train", config.img_size, args.root, augment=True, seed=args.seed)
    val = PetsDataset("val", config.img_size, args.root, seed=args.seed + 1)
    model = TinySAM(config, seed=args.seed).to(device)
    if args.init_encoder:
        state = torch.load(args.init_encoder, map_location=device, weights_only=True)
        model_state = model.state_dict()
        absent = [key for key in state if key not in model_state]
        result = model.load_state_dict(state, strict=False)
        if result.unexpected_keys or absent:
            raise RuntimeError(
                f"MAE encoder transfer mismatch: unexpected={result.unexpected_keys}, absent={absent}"
            )
        if not state or any(not key.startswith("image_encoder.") for key in state):
            raise RuntimeError("MAE checkpoint must contain image_encoder tensors only")
        print(f"loaded {len(state)} encoder tensors from {args.init_encoder}", flush=True)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    steps_epoch = args.steps_per_epoch or math.ceil(len(train) / args.batch_size)
    total_steps = args.max_steps or args.epochs * steps_epoch
    output = Path(args.out_dir) / args.name
    output.mkdir(parents=True, exist_ok=True)
    losses, validation = [], []
    step, start = 0, time.time()
    with (output / "loss.csv").open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step", "epoch", "loss", "lr"])
        for epoch in range(args.epochs):
            order = rng.permutation(len(train))
            epoch_loss = []
            for offset in range(0, min(len(order), steps_epoch * args.batch_size), args.batch_size):
                if step >= total_steps:
                    break
                indices = order[offset:offset + args.batch_size]
                if len(indices) != args.batch_size:
                    continue
                images, masks, points, labels = train.batch(indices, rng)
                image_tensor = torch.from_numpy(images).to(device)
                point_tensor = torch.from_numpy(points).to(device)
                label_tensor = torch.from_numpy(labels).to(device)
                logits, iou_pred = model(image_tensor, point_tensor, label_tensor)
                target = downsample_masks(masks, logits.shape[-1])
                target_tensor = torch.from_numpy(target).to(device)
                loss = tiny_sam_loss(logits, iou_pred, target_tensor)
                optimizer.zero_grad()
                loss.backward()
                warmup = min(1.0, (step + 1) / max(args.warmup_steps, 1))
                progress = step / max(total_steps, 1)
                rate = args.lr * warmup * 0.5 * (1 + math.cos(math.pi * progress))
                optimizer.step(rate)
                losses.append(float(loss.detach().cpu()))
                epoch_loss.append(losses[-1])
                step += 1
                writer.writerow([step, epoch + 1, losses[-1], rate])
                csv_file.flush()
                if step % 10 == 0:
                    print(f"step={step}/{total_steps} loss={losses[-1]:.5f} lr={rate:.3g}", flush=True)
            if not epoch_loss:
                break
            score = evaluate_subset(model, val, rng, device, args.val_subset)
            validation.append(score)
            print(f"epoch={epoch+1} train_loss={np.mean(epoch_loss):.5f} val_mIoU={score:.4f} "
                  f"elapsed={time.time()-start:.1f}s", flush=True)
            save_torch_weights(model, output / "weights.pt")
    save_torch_weights(model, output / "weights.pt")
    with (output / "val_metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "val_miou"])
        writer.writerows(enumerate(validation, 1))
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(losses) + 1), losses)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(output / "loss_curve.png", dpi=140)
    print(f"weights={output / 'weights.pt'}\nloss_curve={output / 'loss_curve.png'}")


def downsample_masks(masks, out_size):
    factor = masks.shape[-1] // out_size
    return (masks.reshape(len(masks), out_size, factor, out_size, factor)
            .mean(axis=(2, 4)) >= 0.5).astype(np.float32)


@torch.no_grad()
def evaluate_subset(model, dataset, rng, device, count):
    if not len(dataset):
        return 0.0
    indices = rng.choice(len(dataset), min(count, len(dataset)), replace=False)
    scores = []
    for index in indices:
        image, mask = dataset[index]
        points, labels = sample_prompt(mask, rng)
        logits, quality = model(torch.from_numpy(image[None]).to(device),
                                torch.from_numpy(points[None]).to(device),
                                torch.from_numpy(labels[None]).to(device))
        low = logits[0, int(quality[0].argmax())].cpu().numpy() > 0
        truth = downsample_masks(mask[None], low.shape[-1])[0] > 0
        scores.append(np.logical_and(low, truth).sum() / max(np.logical_or(low, truth).sum(), 1))
    return float(np.mean(scores))


if __name__ == "__main__":
    main()
