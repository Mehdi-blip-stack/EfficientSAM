"""Model loading and inference for the Streamlit app (written by our team).

Pipeline (assignment Section 8):  image -> preprocessing -> trained model -> prediction -> output
  * preprocessing : RGB uint8 image -> float in [0, 1] -> direct bilinear resize to the model's input size
                    (256 x 256 for our students, 1024 x 1024 for the released EfficientSAM), exactly as in our
                    evaluation (repro/evaluate_model.py). The models normalise internally.
  * prompt        : one click (label 1) or one box (corner labels 2 and 3), rescaled to the input size.
  * prediction    : 3 candidate masks + a predicted IoU for each; logits are resized back to the original image
                    size and thresholded at 0. The app shows the candidate with the highest predicted IoU.
The image encoder runs once per image; every new prompt only runs the (cheap) prompt encoder and mask decoder.
"""
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).absolute().parents[1]          # repository root: efficient_sam/, repro/, weights/
sys.path.insert(0, str(REPO))

OURS_DIR = Path(__file__).absolute().parent / "weights"
RELEASED = {"EfficientSAM-Ti (released)": ("efficient_sam_vitt.pt", 192, 3),
            "EfficientSAM-S (released)": ("efficient_sam_vits.pt", 384, 6)}


@dataclass
class Prediction:
    masks: np.ndarray          # 3 x H x W bool
    iou: np.ndarray            # 3 predicted IoUs
    best: int                  # index of the mask with the highest predicted IoU
    decoder_ms: float


def available_models():
    """Our fine-tuned checkpoints in app/weights/*.pt first (the 100,000-image model, our best, at the top),
    then the released EfficientSAM models."""
    ours = sorted(OURS_DIR.glob("*.pt"), key=lambda p: (0 if "100k" in p.stem else 1 if "e100" in p.stem else 2 if "5k" in p.stem else 3, p.stem))
    models = {f"Ours: {p.stem}": ("ours", p) for p in ours}
    for name, (fname, _, _) in RELEASED.items():
        w = REPO / "weights" / fname
        if w.exists() or (REPO / "weights" / (fname + ".zip")).exists():
            models[name] = ("released", w)
    return models


class Segmenter:
    def __init__(self, name, kind, weights, device="cpu"):
        self.name, self.kind, self.device = name, kind, device
        self.info = {}
        if kind == "ours":
            from repro.common import build_student_sam, STUDENT_SIZE
            ck = torch.load(weights, map_location="cpu", weights_only=False)
            self.model = build_student_sam()
            self.model.load_state_dict(ck["model"])
            self.size = STUDENT_SIZE
            args = ck.get("args", {})
            self.info = {"encoder start": ck.get("init", "?"),
                         "encoder": "frozen" if args.get("freeze_encoder") else "fine-tuned",
                         "epochs": args.get("epochs", "?"), "encoder LR": args.get("lr_encoder", "?")}
        else:
            from efficient_sam.efficient_sam import build_efficient_sam
            fname, dim, heads = RELEASED[name]
            if not weights.exists():                                   # EfficientSAM-S ships as a zip
                with zipfile.ZipFile(str(weights) + ".zip") as z:
                    z.extract(fname, weights.parent)
            self.model = build_efficient_sam(encoder_patch_embed_dim=dim, encoder_num_heads=heads,
                                             checkpoint=str(weights))
            self.size = 1024
        self.model.eval().to(device)
        self.params_m = sum(p.numel() for p in self.model.parameters()) / 1e6

    @torch.no_grad()
    def encode(self, img):
        """img: H x W x 3 uint8 RGB -> (image embedding, encoder time in ms)."""
        t0 = time.perf_counter()
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255)[None].to(self.device)
        x = F.interpolate(x, (self.size, self.size), mode="bilinear")
        emb = self.model.get_image_embeddings(x)
        return emb, (time.perf_counter() - t0) * 1000

    @torch.no_grad()
    def predict(self, emb, hw, point=None, box=None):
        """point: (x, y) or box: (x0, y0, x1, y1), in ORIGINAL image pixels."""
        H, W = hw
        if box is not None:
            pts, lbl = [[box[0], box[1]], [box[2], box[3]]], [2, 3]
        else:
            pts, lbl = [[point[0], point[1]], [-1, -1]], [1, -1]
        pts = torch.tensor(pts, dtype=torch.float32)
        lbl = torch.tensor(lbl, dtype=torch.float32)
        real = lbl >= 0
        pts[:, 0] = torch.where(real, pts[:, 0] * self.size / W, pts[:, 0])
        pts[:, 1] = torch.where(real, pts[:, 1] * self.size / H, pts[:, 1])
        t0 = time.perf_counter()
        logits, iou = self.model.predict_masks(emb, pts[None, None].to(self.device), lbl[None, None].to(self.device),
                                               multimask_output=True, input_h=self.size, input_w=self.size)
        masks = F.interpolate(logits[0].float(), (H, W), mode="bicubic")[0] >= 0      # 3 x H x W
        ms = (time.perf_counter() - t0) * 1000
        iou = iou[0, 0].float().cpu().numpy()
        return Prediction(masks.cpu().numpy(), iou, int(iou.argmax()), ms)
