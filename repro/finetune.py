"""Step B5: fine-tune the student for promptable segmentation on our 800 COCO training images.

Model   : EfficientSAM-Ti architecture at 256 px (official encoder/prompt encoder/mask decoder, see common.py).
Init    : encoder from SAMI pretraining, from MAE pretraining, or random (--init); prompt encoder and mask decoder
          always start random, so the three runs differ ONLY in how the encoder was pretrained.
Targets : non-crowd COCO masks with >= 4 px (report Section 6.8), resized to 256 x 256 like the image.
Prompts : each training object gets either a box (jittered by up to 10% of its side, as in SAM) or one click
          sampled uniformly inside its mask; up to 16 objects per image.
Loss    : SAM recipe -- 20 x focal + Dice on the best of the 3 predicted masks, plus MSE between the predicted IoU
          and the true IoU of every mask. ADAPTED from our team's from_scratch/tinysam/torch_impl/losses.py
          (vectorised and extended with a padding mask).
Written by our team. Saves every epoch to --out and resumes automatically.

Usage:  python -m repro.finetune --init sami   --encoder .../checkpoints/sami/encoder_final.pt --out .../checkpoints/ft_sami
        python -m repro.finetune --init mae    --encoder .../checkpoints/mae/encoder_final.pt  --out .../checkpoints/ft_mae
        python -m repro.finetune --init random --out .../checkpoints/ft_random
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .common import MIN_MASK_PIXELS, STUDENT_SIZE, build_student_sam, data_root, decode, seed_everything

S = STUDENT_SIZE


# ------------------------------------------------------------------------------------------------ targets
def build_targets(features_dir, cache_path, limit=0):
    """Decode every usable training mask once and store it at 256 x 256 (bit-packed) next to its box."""
    from pycocotools.coco import COCO
    ids = json.load(open(Path(features_dir) / "teacher_vit_h_stats.json"))["image_ids"] if \
        (Path(features_dir) / "teacher_vit_h_stats.json").exists() else None
    coco = COCO(str(data_root() / "annotations" / "subset_instances_train2017.json"))
    img_ids = ids if ids is not None else sorted(coco.imgs)
    if limit:
        img_ids = img_ids[:limit]
    rows, packed, dropped = [], [], {"crowd": 0, "tiny": 0, "vanished_at_256": 0}
    for i, iid in enumerate(img_ids):
        info = coco.imgs[iid]; h, w = info["height"], info["width"]
        for ann in coco.imgToAnns.get(iid, []):
            if ann["iscrowd"]:
                dropped["crowd"] += 1; continue
            m = decode(ann, h, w)
            if m.sum() < MIN_MASK_PIXELS:
                dropped["tiny"] += 1; continue
            m256 = F.interpolate(torch.from_numpy(m)[None, None].float(), (S, S), mode="bilinear")[0, 0] > 0.5
            if m256.sum() == 0:                                  # thinner than one pixel after resizing
                dropped["vanished_at_256"] += 1; continue
            ys, xs = torch.where(m256)
            rows.append([i, xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1])
            packed.append(np.packbits(m256.numpy().reshape(-1)))
    np.savez_compressed(cache_path, rows=np.array(rows, np.int32), masks=np.stack(packed), image_ids=np.array(img_ids))
    print(f"targets: {len(rows)} objects in {len(img_ids)} images, dropped {dropped}")


# ------------------------------------------------------------------------------------------------ loss
def sam_loss(logits, iou_pred, target, valid, focal_w=20.0):
    """logits: B,Q,3,S,S  iou_pred: B,Q,3  target: B,Q,S,S (0/1)  valid: B,Q (False = padding query)."""
    t = target[:, :, None].expand_as(logits)
    p = torch.sigmoid(logits)
    pt = t * p + (1 - t) * (1 - p)
    alpha = t * 0.25 + (1 - t) * 0.75
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    focal = (alpha * (1 - pt) ** 2 * bce).mean((-2, -1))
    dice = 1 - (2 * (p * t).sum((-2, -1)) + 1) / (p.sum((-2, -1)) + t.sum((-2, -1)) + 1)
    per_mask = focal_w * focal + dice                                           # B,Q,3
    best = per_mask.argmin(-1, keepdim=True)
    mask_loss = per_mask.gather(-1, best)[..., 0]
    with torch.no_grad():
        b = logits > 0; tt = t > 0.5
        true_iou = (b & tt).sum((-2, -1)).float() / (b | tt).sum((-2, -1)).float().clamp(min=1)
    iou_loss = ((iou_pred.float() - true_iou) ** 2).mean(-1)
    v = valid.float()
    return ((mask_loss + iou_loss) * v).sum() / v.sum(), (mask_loss * v).sum() / v.sum()


# ------------------------------------------------------------------------------------------------ prompts
def make_prompts(boxes, masks, gen):
    """boxes: K,4 (x0,y0,x1,y1 at 256 px), masks: K,S,S bool -> points K,2,2 and labels K,2."""
    K = boxes.shape[0]
    pts = torch.full((K, 2, 2), -1.0); lbl = torch.full((K, 2), -1.0)
    use_box = torch.rand(K, generator=gen) < 0.5
    for k in range(K):
        if use_box[k]:
            x0, y0, x1, y1 = boxes[k].float()
            noise = (torch.randn(4, generator=gen) * 0.1 * torch.stack([x1 - x0, y1 - y0, x1 - x0, y1 - y0])).clamp(-5, 5)
            b = (torch.stack([x0, y0, x1, y1]) + noise).clamp(0, S)
            pts[k] = torch.stack([b[:2], b[2:]]); lbl[k] = torch.tensor([2.0, 3.0])
        else:
            ys, xs = torch.where(masks[k])
            j = torch.randint(len(ys), (1,), generator=gen).item()
            pts[k, 0] = torch.tensor([xs[j] + 0.5, ys[j] + 0.5]); lbl[k, 0] = 1.0
    return pts, lbl


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", choices=["sami", "mae", "random"], required=True)
    ap.add_argument("--encoder", default="")
    ap.add_argument("--features", required=True, help="folder with student_inputs_256.npy (step B2)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_objects", type=int, default=16)
    ap.add_argument("--lr_encoder", type=float, default=1e-4)
    ap.add_argument("--lr_decoder", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=36219)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--freeze_encoder", action="store_true",
                    help="keep the pretrained patch embedding + ViT blocks fixed; train only the neck, prompt encoder "
                         "and mask decoder. Tests the quality of the pretrained FEATURES themselves.")
    args = ap.parse_args()
    seed_everything(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    feats = Path(args.features)

    cache = feats / ("finetune_targets_256.npz" if not args.limit else f"finetune_targets_256_limit{args.limit}.npz")
    if not cache.exists():
        build_targets(feats, cache, args.limit)
    z = np.load(cache)
    rows = torch.from_numpy(z["rows"]); n_img = len(z["image_ids"])
    masks = torch.from_numpy(np.unpackbits(z["masks"], axis=1)[:, : S * S].reshape(-1, S, S).astype(bool))
    inputs = torch.from_numpy(np.load(feats / "student_inputs_256.npy")[:n_img])     # N,2,3,S,S uint8
    by_img = [torch.where(rows[:, 0] == i)[0] for i in range(n_img)]
    img_list = [i for i in range(n_img) if len(by_img[i])]

    model = build_student_sam().to(dev)
    if args.init != "random":
        sd = torch.load(args.encoder, map_location="cpu")
        assert sd["target"] == args.init, f"encoder checkpoint was pretrained with {sd['target']}, not {args.init}"
        print("encoder load:", model.image_encoder.load_state_dict(sd["encoder"]))
    tokens = None
    if args.freeze_encoder:
        # The neck was never trained during pretraining (SAMI/MAE use the ViT blocks only), so it stays trainable.
        # Everything before it is frozen, so its output is computed ONCE per (image, view) and cached.
        from efficient_sam.efficient_sam_encoder import get_abs_pos
        enc = model.image_encoder
        for n, p in enc.named_parameters():
            p.requires_grad = n.startswith("neck.")
        enc.eval()
        chunks = []
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            flat = inputs.reshape(-1, 3, S, S)
            for i in range(0, len(flat), 64):
                t = enc.patch_embed(model.preprocess(flat[i:i + 64].to(dev).float() / 255)).permute(0, 2, 3, 1)
                t = t + get_abs_pos(enc.pos_embed, enc.pretrain_use_cls_token, [t.shape[1], t.shape[2]])
                g = t.shape[1]; t = t.reshape(t.shape[0], g * g, t.shape[-1])
                for blk in enc.blocks:
                    t = blk(t)
                chunks.append(t.reshape(t.shape[0], g, g, -1).permute(0, 3, 1, 2).half())
        tokens = torch.cat(chunks).reshape(n_img, 2, -1, chunks[0].shape[-2], chunks[0].shape[-1])   # N,2,192,16,16
        print(f"frozen encoder: cached block outputs {tuple(tokens.shape)}")
    enc_params = [p for n, p in model.image_encoder.named_parameters() if p.requires_grad and not n.startswith("neck.")]
    other = [p for n, p in model.named_parameters() if p.requires_grad and (not n.startswith("image_encoder.") or n.startswith("image_encoder.neck."))]
    groups = [{"params": other, "lr": args.lr_decoder}] + ([{"params": enc_params, "lr": args.lr_encoder}] if enc_params else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    steps_per_epoch = math.ceil(len(img_list) / args.batch); total = args.epochs * steps_per_epoch
    warm = args.warmup * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    gen = torch.Generator().manual_seed(args.seed)

    start, log, ck = 0, [], out / "last.pt"
    if ck.exists():
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        scaler.load_state_dict(s["scaler"]); start, log = s["epoch"] + 1, s["log"]; gen.set_state(s["gen"])
        print(f"resumed from epoch {start}")

    for epoch in range(start, args.epochs):
        model.train(); t0 = time.time(); L, ML = [], []
        views = []
        order = torch.randperm(len(img_list), generator=gen)
        for st in range(0, len(img_list), args.batch):
            imgs, P, Lb, T, V = [], [], [], [], []
            batch = [img_list[j] for j in order[st:st + args.batch]]
            Q = min(args.max_objects, max(len(by_img[i]) for i in batch))
            for i in batch:
                objs = by_img[i]
                if len(objs) > Q:
                    objs = objs[torch.randperm(len(objs), generator=gen)[:Q]]
                m = masks[objs]; b = rows[objs, 1:].clone()
                view = int(torch.rand(1, generator=gen) < 0.5)                   # random mirror (view 1)
                if view:
                    m = m.flip(-1); b = torch.stack([S - b[:, 2], b[:, 1], S - b[:, 0], b[:, 3]], 1)
                p, l = make_prompts(b, m, gen)
                k = len(objs); pad = Q - k
                imgs.append(inputs[i, view]); views.append((i, view))
                P.append(torch.cat([p, torch.full((pad, 2, 2), -1.0)])); Lb.append(torch.cat([l, torch.full((pad, 2), -1.0)]))
                T.append(torch.cat([m.float(), torch.zeros(pad, S, S)])); V.append(torch.arange(Q) < k)
            x = None if tokens is not None else torch.stack(imgs).to(dev).float() / 255
            bv = views[-len(batch):]
            P, Lb, T, V = torch.stack(P).to(dev), torch.stack(Lb).to(dev), torch.stack(T).to(dev), torch.stack(V).to(dev)
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                if tokens is not None:
                    emb = model.image_encoder.neck(torch.stack([tokens[i, v] for i, v in bv]).float())
                else:
                    emb = model.get_image_embeddings(x)
                logits, iou_pred = model.predict_masks(emb, P, Lb, multimask_output=True, input_h=S, input_w=S,
                                                       output_h=S, output_w=S)
            loss, mloss = sam_loss(logits.float(), iou_pred, T, V)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            L.append(loss.item()); ML.append(mloss.item())
        log.append({"epoch": epoch, "loss": float(np.mean(L)), "mask_loss": float(np.mean(ML)), "sec": time.time() - t0})
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "log": log, "gen": gen.get_state(), "args": vars(args)}, ck)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"[ft-{args.init}{'-frozen' if args.freeze_encoder else ''}] epoch {epoch + 1}/{args.epochs} loss {log[-1]['loss']:.4f} "
                  f"(mask {log[-1]['mask_loss']:.4f}) {log[-1]['sec']:.1f}s/epoch", flush=True)

    torch.save({"model": model.state_dict(), "init": args.init, "log": log, "args": vars(args)}, out / "model_final.pt")
    json.dump(log, open(out / "finetune_log.json", "w"))
    print("saved", out / "model_final.pt")


if __name__ == "__main__":
    main()
