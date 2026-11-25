"""Steps B3/B4: pretrain the student encoder with SAMI (teacher features) or MAE (pixels). Written by our team.

Recipe (paper / MAE defaults, scaled to our data): AdamW, base lr 1.5e-4 x batch/256, weight decay 0.05,
betas (0.9, 0.95), linear warm-up then cosine decay, mixed precision. Saves a checkpoint to --out every epoch and
resumes automatically from it, so a Colab disconnect only loses the current epoch.

Usage:  python -m repro.pretrain --target sami --features .../features --out .../checkpoints/sami
        python -m repro.pretrain --target mae  --features .../features --out .../checkpoints/mae
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .common import build_student_sam, seed_everything
from .sami import SAMIPretrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["sami", "mae"], required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--teacher", default="vit_h")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=36219)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    seed_everything(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    feats = Path(args.features)
    inputs = torch.from_numpy(np.load(feats / "student_inputs_256.npy"))             # N,2,3,256,256 uint8
    targets = None
    if args.target == "sami":
        stats = json.load(open(feats / f"teacher_{args.teacher}_stats.json"))
        t = torch.from_numpy(np.load(feats / f"teacher_{args.teacher}_16x16.npy")).float()
        mu = torch.tensor(stats["channel_mean"]).view(1, 1, -1, 1, 1); sd = torch.tensor(stats["channel_std"]).view(1, 1, -1, 1, 1)
        targets = ((t - mu) / sd).half()                                              # standardised per channel
    if args.limit:
        inputs = inputs[: args.limit]; targets = targets[: args.limit] if targets is not None else None
    n = inputs.shape[0]
    inputs = inputs.to(dev); targets = targets.to(dev) if targets is not None else None    # small enough for GPU memory

    model = SAMIPretrainer(build_student_sam().image_encoder, target=args.target).to(dev)
    lr = 1.5e-4 * args.batch / 256
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    steps_per_epoch = max(1, math.ceil(2 * n / args.batch)); total = args.epochs * steps_per_epoch
    warm = args.warmup * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

    start, log = 0, []
    ck = out / "last.pt"
    if ck.exists():                                                                    # resume after a disconnect
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        scaler.load_state_dict(s["scaler"]); start, log = s["epoch"] + 1, s["log"]
        print(f"resumed from epoch {start}")

    for epoch in range(start, args.epochs):
        model.train(); t0 = time.time(); losses = []
        perm = torch.randperm(2 * n, device=dev)                                      # every (image, view) pair once
        for i in range(0, 2 * n, args.batch):
            idx = perm[i:i + args.batch]; img_i, view = idx // 2, idx % 2
            x = inputs[img_i, view]
            y = targets[img_i, view] if targets is not None else None
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(loss.item())
        log.append({"epoch": epoch, "loss": float(np.mean(losses)), "lr": sched.get_last_lr()[0], "sec": time.time() - t0})
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "log": log, "args": vars(args)}, ck)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"[{args.target}] epoch {epoch + 1}/{args.epochs} loss {log[-1]['loss']:.4f} ({log[-1]['sec']:.1f}s/epoch)", flush=True)

    torch.save({"encoder": model.encoder.state_dict(), "target": args.target, "log": log, "args": vars(args)},
               out / "encoder_final.pt")
    json.dump(log, open(out / "pretrain_log.json", "w"))
    print("saved", out / "encoder_final.pt")


if __name__ == "__main__":
    main()
