"""Shared helpers for the reproduction pipeline (written by our team).

Origins of code used in this package (template Sections 6 and 11):
  * efficient_sam/  : OFFICIAL EfficientSAM code (yformer/EfficientSAM), reused unchanged
                      (ImageEncoderViT, PromptEncoder, MaskDecoder, TwoWayTransformer, EfficientSam).
  * build_student_sam() below : ADAPTED from the official build_efficient_sam(); the only change is
                      img_size 1024 -> 256 (our compute budget) and no released checkpoint is loaded.
  * everything else in repro/ : WRITTEN BY OUR TEAM unless a docstring says otherwise.
"""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch import nn

REPO = Path(os.environ.get("EFFSAM_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))

STUDENT_SIZE = 256        # student input resolution (paper: 224 for SAMI, 1024 for fine-tuning)
PATCH = 16                # -> 16 x 16 = 256 tokens
GRID = STUDENT_SIZE // PATCH
TEACHER_SIZE = 1024       # SAM's native input resolution
MIN_MASK_PIXELS = 4       # report Section 6.8, rule 4


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def data_root():
    return Path(os.environ.get("COCO_DATA_ROOT", r"C:\Users\Abrar\Desktop\abrar\iba\ML\project\COCO_dataset"))


def decode(ann, h, w):
    """COCO polygons/RLE -> H x W bool mask; polygons merged with a UNION (report Section 6.5)."""
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
    elif isinstance(seg["counts"], list):
        rle = mask_utils.frPyObjects(seg, h, w)
    else:
        rle = seg
    return mask_utils.decode(rle).astype(bool)


def build_student_sam(embed_dim=192, num_heads=3, img_size=STUDENT_SIZE):
    """EfficientSAM-Ti architecture at a smaller input size, randomly initialised.
    ADAPTED from efficient_sam.build_efficient_sam (official): same encoder depth (12), width (192, Ti),
    heads, neck, prompt encoder and mask decoder; only img_size differs."""
    from efficient_sam.efficient_sam import EfficientSam, PromptEncoder, MaskDecoder, TwoWayTransformer
    from efficient_sam.efficient_sam_encoder import ImageEncoderViT
    enc = ImageEncoderViT(img_size=img_size, patch_size=PATCH, in_chans=3, patch_embed_dim=embed_dim,
                          normalization_type="layer_norm", depth=12, num_heads=num_heads, mlp_ratio=4.0,
                          neck_dims=[256, 256], act_layer=nn.GELU)
    return EfficientSam(
        image_encoder=enc,
        prompt_encoder=PromptEncoder(embed_dim=256, image_embedding_size=(enc.image_embedding_size,) * 2,
                                     input_image_size=(img_size, img_size)),
        decoder_max_num_input_points=6,
        mask_decoder=MaskDecoder(
            transformer_dim=256,
            transformer=TwoWayTransformer(depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048,
                                          activation=nn.GELU, normalize_before_activation=False),
            num_multimask_outputs=3, activation=nn.GELU, normalization_type="layer_norm",
            normalize_before_activation=False, iou_head_depth=2, iou_head_hidden_dim=256,
            upscaling_layer_dims=[64, 32]),
        pixel_mean=[0.485, 0.456, 0.406], pixel_std=[0.229, 0.224, 0.225])


def squash(img_uint8_hwc, size):
    """Direct (anisotropic) resize to size x size, exactly like EfficientSam.preprocess (report Section 6.4)."""
    t = torch.from_numpy(np.ascontiguousarray(img_uint8_hwc)).permute(2, 0, 1).float()[None]
    return torch.nn.functional.interpolate(t, (size, size), mode="bilinear").clamp(0, 255)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1)
