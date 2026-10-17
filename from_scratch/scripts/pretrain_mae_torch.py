"""Run masked-pixel autoencoder pretraining on Pets images."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tinysam.config import TORCH_SMALL, TinySAMConfig
from tinysam.data import PetsDataset
from tinysam.torch_impl.mae import MAEPretrainer, mae_encoder_state_dict
from tinysam.torch_impl.optim import AdamW


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "data"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--patch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=str(ROOT / "runs" / "mae_encoder.pt"))
    parser.add_argument("--tiny", action="store_true", help="Use a compact architecture for CPU smoke tests")
    args = parser.parse_args()
    if args.tiny:
        config = TinySAMConfig(img_size=args.size, patch=args.patch, enc_dim=32, enc_depth=1, enc_heads=4,
                               emb_dim=32, dec_depth=1, dec_heads=4)
    else:
        config = TORCH_SMALL
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    dataset = PetsDataset("train", config.img_size, args.root, augment=True)
    model = MAEPretrainer(config).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    rng = torch.Generator().manual_seed(17)
    start = time.time()
    for step in range(args.steps):
        indices = torch.randint(len(dataset), (args.batch_size,), generator=rng).numpy()
        images = torch.from_numpy(np.stack([dataset[int(i)][0] for i in indices])).to(device)
        optimizer.zero_grad()
        loss = model(images)
        loss.backward()
        optimizer.step()
        if (step + 1) % 10 == 0 or step == 0:
            print(f"step={step+1}/{args.steps} mae_loss={loss.item():.6f} elapsed={time.time()-start:.1f}s",
                  flush=True)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mae_encoder_state_dict(model), output)
    print(f"encoder_weights={output}")


if __name__ == "__main__":
    main()
