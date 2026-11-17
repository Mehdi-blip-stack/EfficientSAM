"""Evaluate ONE promptable-segmentation model on the frozen prompts (run once per model, in its own process
so that peak memory is measured per model).

Models (all from their official code and released checkpoints, no retraining):
    effsam_ti, effsam_s   EfficientSAM   (yformer/EfficientSAM, code in this repo)
    mobile_sam            MobileSAM      (ChaoningZhang/MobileSAM)
    sam_vit_b, sam_vit_h  SAM            (facebookresearch/segment-anything)

Protocol (identical for every model; report Sections 6.6-6.7):
  * prompts: ground-truth box, and one click at the deepest interior pixel (from eval_prompts.json);
  * each model uses its OWN official preprocessing (EfficientSAM: direct resize to 1024x1024;
    SAM/MobileSAM: longest side to 1024 + padding);
  * the model returns 3 candidate masks; we keep the one with the highest PREDICTED IoU
    (and also record the best-of-three "oracle" IoU);
  * masks are compared with ground truth at the ORIGINAL image resolution;
  * metrics per object: IoU, Dice, Boundary IoU.
Efficiency: parameter counts, encoder time per image, decoder time per prompt, peak process memory.

Usage:
    python -m repro.evaluate_model --model effsam_ti --weights weights/efficient_sam_vitt.pt
"""
import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from scipy import ndimage

try:
    import resource                      # Unix only (peak memory); absent on Windows
except ImportError:
    resource = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_eval_prompts import decode  # same mask decoding as the prompt file
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(os.environ.get("EFFSAM_REPO", Path(__file__).resolve().parents[1]))
WARMUP_IMAGES = 3          # first images are excluded from timing (lazy allocations, caches)
QUERY_CHUNK = 16           # prompts decoded per call, bounds memory on 8 GB machines


# ----------------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------------
def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def dice(a, b):
    t = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / t) if t else 1.0


def boundary_iou(pred, true, ratio=0.02):
    """IoU restricted to a band of width d (2% of the image diagonal) inside each contour (Cheng et al., 2021).
    Computed on a crop around both masks; the image border counts as outside the object."""
    d = max(1, int(round(ratio * np.hypot(*true.shape))))
    ys, xs = np.where(pred | true)
    if len(ys) == 0:
        return 1.0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    p = np.pad(pred[y0:y1, x0:x1], 1); t = np.pad(true[y0:y1, x0:x1], 1)
    band = lambda m: m & ~ndimage.binary_erosion(m, iterations=d, border_value=0)
    return iou(band(p), band(t))


# ----------------------------------------------------------------------------------------------------
# Model adapters: each exposes encode(image_uint8) and decode_prompts(points, labels) -> (masks, iou_pred)
#   points: K x 2 x 2 in ORIGINAL pixel coordinates, labels: K x 2 (1 click, 2/3 box corners, -1 padding)
#   masks : K x 3 x H x W boolean at the original size, iou_pred: K x 3
# ----------------------------------------------------------------------------------------------------
class EfficientSAMAdapter:
    def __init__(self, variant, weights, device):
        sys.path.insert(0, str(REPO))
        from efficient_sam.efficient_sam import build_efficient_sam
        dim, heads = {"effsam_ti": (192, 3), "effsam_s": (384, 6)}[variant]
        self.model = build_efficient_sam(encoder_patch_embed_dim=dim, encoder_num_heads=heads,
                                         checkpoint=str(weights)).eval().to(device)
        self.encoder, self.device = self.model.image_encoder, device

    def encode(self, img):
        self.hw = img.shape[:2]
        x = torch.from_numpy(img).permute(2, 0, 1).float().div(255)[None].to(self.device)
        # Same call as EfficientSam.preprocess (direct bilinear resize, no padding); the model normalises itself.
        x = F.interpolate(x, (1024, 1024), mode="bilinear")
        self.emb = self.model.get_image_embeddings(x)

    def decode_prompts(self, points, labels):
        H, W = self.hw
        pts = points.clone(); real = labels >= 0
        pts[..., 0] = torch.where(real, pts[..., 0] * 1024 / W, pts[..., 0])     # anisotropic rescale
        pts[..., 1] = torch.where(real, pts[..., 1] * 1024 / H, pts[..., 1])
        logits, iou_pred = self.model.predict_masks(self.emb, pts[None].to(self.device), labels[None].to(self.device),
                                                    multimask_output=True, input_h=1024, input_w=1024)
        masks = F.interpolate(logits[0], (H, W), mode="bicubic") >= 0
        return masks.cpu().numpy(), iou_pred[0].cpu().numpy()


class SAMAdapter:
    """SAM and MobileSAM share the SamPredictor interface (longest-side resize + pad, own normalisation)."""
    def __init__(self, variant, weights, device):
        if variant == "mobile_sam":
            from mobile_sam import SamPredictor, sam_model_registry
            sam = sam_model_registry["vit_t"](checkpoint=str(weights))
        else:
            from segment_anything import SamPredictor, sam_model_registry
            sam = sam_model_registry[variant.replace("sam_", "")](checkpoint=str(weights))
        self.model = sam.eval().to(device)
        self.encoder, self.device = sam.image_encoder, device
        self.pred = SamPredictor(self.model)

    def encode(self, img):
        self.hw = img.shape[:2]
        self.pred.set_image(img)                       # RGB uint8

    def decode_prompts(self, points, labels):
        tr = self.pred.transform
        if (labels[:, 0] == 2).all():                  # box prompt: corners -> K x 4
            boxes = tr.apply_boxes_torch(points.reshape(-1, 4).to(self.device), self.hw)
            masks, iou_pred, _ = self.pred.predict_torch(None, None, boxes=boxes, multimask_output=True)
        else:                                          # one foreground click -> K x 1 x 2
            pc = tr.apply_coords_torch(points[:, :1].to(self.device), self.hw)
            masks, iou_pred, _ = self.pred.predict_torch(pc, torch.ones(len(pc), 1, device=self.device),
                                                         multimask_output=True)
        return masks.cpu().numpy(), iou_pred.cpu().numpy()


class StudentAdapter(EfficientSAMAdapter):
    """OUR trained models: EfficientSAM-Ti architecture at 256 px (repro.finetune). Same official preprocessing
    (direct resize, model-internal normalisation), at the student's input size."""
    def __init__(self, variant, weights, device):
        from repro.common import build_student_sam, STUDENT_SIZE
        self.size = STUDENT_SIZE
        self.model = build_student_sam().eval().to(device)
        self.model.load_state_dict(torch.load(weights, map_location="cpu")["model"])
        self.encoder, self.device = self.model.image_encoder, device

    def encode(self, img):
        self.hw = img.shape[:2]
        x = torch.from_numpy(img).permute(2, 0, 1).float().div(255)[None].to(self.device)
        self.emb = self.model.get_image_embeddings(F.interpolate(x, (self.size, self.size), mode="bilinear"))

    def decode_prompts(self, points, labels):
        H, W = self.hw
        pts = points.clone(); real = labels >= 0
        pts[..., 0] = torch.where(real, pts[..., 0] * self.size / W, pts[..., 0])
        pts[..., 1] = torch.where(real, pts[..., 1] * self.size / H, pts[..., 1])
        logits, iou_pred = self.model.predict_masks(self.emb, pts[None].to(self.device), labels[None].to(self.device),
                                                    multimask_output=True, input_h=self.size, input_w=self.size)
        masks = F.interpolate(logits[0].float(), (H, W), mode="bicubic") >= 0
        return masks.cpu().numpy(), iou_pred[0].float().cpu().numpy()


def build(variant, weights, device):
    if variant.startswith("ours_"):
        return StudentAdapter(variant, weights, device)
    return (EfficientSAMAdapter if variant.startswith("effsam") else SAMAdapter)(variant, weights, device)


# ----------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="effsam_ti | effsam_s | mobile_sam | sam_vit_b | sam_vit_h | ours_<name> (any of our fine-tuned students)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (smoke test)")
    args = ap.parse_args()
    if not (args.model in ("effsam_ti", "effsam_s", "mobile_sam", "sam_vit_b", "sam_vit_h") or args.model.startswith("ours_")):
        ap.error(f"unknown model {args.model}")

    root = Path(os.environ.get("COCO_DATA_ROOT", r"C:\Users\Abrar\Desktop\abrar\iba\ML\project\COCO_dataset"))
    out = Path(os.environ.get("REPRO_OUT", "output/repro")); out.mkdir(parents=True, exist_ok=True)
    prompts = json.load(open(out / "eval_prompts.json"))["images"]
    if args.limit:
        prompts = prompts[: args.limit]
    coco = COCO(str(root / "annotations" / "subset_instances_val2017.json"))

    torch.manual_seed(0)
    model = build(args.model, args.weights, args.device)
    n_total = sum(p.numel() for p in model.model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    sync = (lambda: torch.cuda.synchronize()) if args.device == "cuda" else (lambda: None)

    rows, enc_t, dec_t = [], [], []
    t_start = time.time()
    with torch.no_grad():
        for n, im in enumerate(prompts):
            with Image.open(root / "images" / "val2017" / im["file_name"]) as f:
                img = np.array(f.convert("RGB"))                     # always 3-channel RGB
            sync(); t0 = time.perf_counter(); model.encode(img); sync()
            if n >= WARMUP_IMAGES:
                enc_t.append(time.perf_counter() - t0)
            objs = im["objects"]
            gts = [decode(coco.anns[o["ann_id"]], im["height"], im["width"]) for o in objs]
            res = {}
            for mode in ["box", "click"]:
                if mode == "box":
                    pts = torch.tensor([[o["box_xyxy"][:2], o["box_xyxy"][2:]] for o in objs], dtype=torch.float32)
                    lbl = torch.tensor([[2.0, 3.0]] * len(objs))
                else:
                    pts = torch.tensor([[o["click_xy"], [-1.0, -1.0]] for o in objs], dtype=torch.float32)
                    lbl = torch.tensor([[1.0, -1.0]] * len(objs))
                masks, ious = [], []
                for i in range(0, len(objs), QUERY_CHUNK):
                    sync(); t0 = time.perf_counter()
                    m, q = model.decode_prompts(pts[i:i + QUERY_CHUNK], lbl[i:i + QUERY_CHUNK]); sync()
                    if n >= WARMUP_IMAGES:
                        dec_t.append((time.perf_counter() - t0) / len(m))     # seconds per prompt
                    masks.append(m); ious.append(q)
                res[mode] = (np.concatenate(masks), np.concatenate(ious))
            for k, (o, gt) in enumerate(zip(objs, gts)):
                row = dict(model=args.model, image_id=im["image_id"], ann_id=o["ann_id"], category=o["category"],
                           supercategory=o["supercategory"], size=o["size"], area=o["area"])
                for mode, (masks, ious) in res.items():
                    cand = [iou(masks[k, j], gt) for j in range(masks.shape[1])]
                    best = int(np.argmax(ious[k]))                   # the model's own most confident mask
                    row[f"{mode}_iou"] = cand[best]
                    row[f"{mode}_dice"] = dice(masks[k, best], gt)
                    row[f"{mode}_biou"] = boundary_iou(masks[k, best], gt)
                    row[f"{mode}_oracle_iou"] = max(cand)            # best of three (upper bound on selection)
                rows.append(row)
            if (n + 1) % 25 == 0:
                print(f"  {args.model}: {n + 1}/{len(prompts)} images, {time.time() - t_start:.0f}s", flush=True)

    with open(out / f"per_instance_{args.model}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 if resource else float('nan')   # Linux: kB -> MB
    med = lambda v: 1000 * float(np.median(v)) if v else float('nan')
    p90 = lambda v: 1000 * float(np.percentile(v, 90)) if v else float('nan')
    eff = dict(model=args.model, device=args.device, cpu=platform.processor() or platform.machine(),
               threads=torch.get_num_threads(), torch=torch.__version__,
               params_total_M=n_total / 1e6, params_encoder_M=n_enc / 1e6,
               encoder_ms_median=med(enc_t), encoder_ms_p90=p90(enc_t),
               decoder_ms_per_prompt_median=med(dec_t),
               peak_memory_MB=peak, cuda_peak_MB=(torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else None),
               images=len(prompts), objects=len(rows), timed_images=len(enc_t))
    json.dump(eff, open(out / f"efficiency_{args.model}.json", "w"), indent=1)
    print(json.dumps(eff))
    print({k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0] if k.endswith(("iou", "dice"))})


if __name__ == "__main__":
    main()
