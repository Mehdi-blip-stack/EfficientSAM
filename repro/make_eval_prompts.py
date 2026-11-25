"""Freeze the evaluation prompts ONCE so that every model is tested on exactly the same inputs.

For each eligible object of the 200 held-out images (non-crowd, decoded mask >= 4 px; report Section 6.8)
we store: the ground-truth box prompt, the single-click prompt (deepest interior pixel = maximum of the
distance transform), and the tags needed for stratified metrics (COCO size bucket from the RECOMPUTED
mask area, category, supercategory).

Usage:
    python -m repro.make_eval_prompts              # reads COCO_DATA_ROOT, writes $REPRO_OUT/eval_prompts.json
"""
import json
import os
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from scipy import ndimage

MIN_MASK_PIXELS = 4


def decode(ann, h, w):
    """COCO segmentation (polygons or RLE) -> H x W boolean mask; polygons are merged with a UNION."""
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
    elif isinstance(seg["counts"], list):
        rle = mask_utils.frPyObjects(seg, h, w)
    else:
        rle = seg
    return mask_utils.decode(rle).astype(bool)


def size_bucket(area):
    return "small" if area < 32 ** 2 else ("medium" if area < 96 ** 2 else "large")


def main():
    root = Path(os.environ.get("COCO_DATA_ROOT", r"C:\Users\Abrar\Desktop\abrar\iba\ML\project\COCO_dataset"))
    out = Path(os.environ.get("REPRO_OUT", "output/repro"))
    out.mkdir(parents=True, exist_ok=True)
    coco = COCO(str(root / "annotations" / "subset_instances_val2017.json"))
    cat = {k: (v["name"], v["supercategory"]) for k, v in coco.cats.items()}

    images, n_obj, dropped = [], 0, {"crowd": 0, "tiny": 0, "empty_images": 0}
    for img_id in sorted(coco.imgs):
        info = coco.imgs[img_id]
        h, w = info["height"], info["width"]
        objects = []
        for ann in coco.imgToAnns.get(img_id, []):
            if ann["iscrowd"]:
                dropped["crowd"] += 1
                continue
            m = decode(ann, h, w)
            area = int(m.sum())
            if area < MIN_MASK_PIXELS:
                dropped["tiny"] += 1
                continue
            x, y, bw, bh = ann["bbox"]
            # Deepest interior pixel (pad so objects touching the border still get distance 0 at the border).
            dist = ndimage.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]
            cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
            objects.append({
                "ann_id": ann["id"],
                "box_xyxy": [float(x), float(y), float(x + bw), float(y + bh)],
                "click_xy": [float(cx + 0.5), float(cy + 0.5)],
                "area": area,
                "size": size_bucket(area),
                "category": cat[ann["category_id"]][0],
                "supercategory": cat[ann["category_id"]][1],
            })
        if not objects:
            dropped["empty_images"] += 1
            continue
        n_obj += len(objects)
        images.append({"image_id": img_id, "file_name": info["file_name"], "height": h, "width": w, "objects": objects})

    json.dump({"split": "val2017 subset (seed 36220)", "images": images}, open(out / "eval_prompts.json", "w"))
    print(f"images={len(images)} objects={n_obj} dropped={dropped} -> {out / 'eval_prompts.json'}")


if __name__ == "__main__":
    main()
