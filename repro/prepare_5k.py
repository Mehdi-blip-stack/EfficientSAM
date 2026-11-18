"""Experiment E, step 1: build a larger fine-tuning set (default 5,000 COCO train2017 images). Written by our team.

The set contains our original 800 training images plus images drawn at random (fixed seed) from the rest of
COCO train2017, so it never overlaps our 200 held-out val2017 evaluation images. Only fine-tuning uses it:
the SAMI and MAE encoders stay the ones pretrained on the original 800 images.

Stages (each is skipped if its output already exists, so the script can be re-run after a disconnect):
  1. full COCO train2017 annotations (downloaded once, ~250 MB zip)
  2. subset annotation file with the selected images
  3. images: copied from our local 800-image copy when available, otherwise downloaded from images.cocodataset.org
  4. student inputs: every image + its mirror, squashed to 256 x 256 exactly as in teacher_features.py
  5. fine-tuning targets cache (masks at 256 x 256), as finetune.py would build it
Needs no GPU.

Usage:
  python -m repro.prepare_5k --base_data /content/data/COCO_dataset --work_data /content/data/COCO_5k \
                             --out_features .../effsam_work/features_5k
"""
import argparse
import json
import os
import random
import shutil
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .common import STUDENT_SIZE, squash

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def fetch(url, dest, tries=4):
    for k in range(tries):
        try:
            tmp = str(dest) + ".part"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, dest)
            return True
        except Exception as e:                                   # network hiccup: wait and retry
            if k == tries - 1:
                print("  failed:", url, e)
                return False
            time.sleep(2 * (k + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_data", required=True, help="our COCO subset (annotations/subset_instances_train2017.json, images/train2017)")
    ap.add_argument("--work_data", required=True, help="local folder for the large set (raw annotations, images)")
    ap.add_argument("--out_features", required=True, help="where student_inputs_256.npy and the targets cache go")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=36221)
    ap.add_argument("--full_ann", default="", help="instances_train2017.json if already available (skips the download)")
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()
    t0 = time.time()
    base, work, feats = Path(args.base_data), Path(args.work_data), Path(args.out_features)
    (work / "annotations").mkdir(parents=True, exist_ok=True); (work / "images" / "train2017").mkdir(parents=True, exist_ok=True)
    feats.mkdir(parents=True, exist_ok=True)

    # 1. full train2017 annotations
    full = Path(args.full_ann) if args.full_ann else work / "raw" / "annotations" / "instances_train2017.json"
    if not full.exists():
        (work / "raw").mkdir(exist_ok=True)
        z = work / "raw" / "annotations_trainval2017.zip"
        if not z.exists():
            print("downloading COCO annotations (~250 MB) ...", flush=True)
            assert fetch(ANN_URL, z), "could not download the COCO annotations"
        with zipfile.ZipFile(z) as f:
            f.extract("annotations/instances_train2017.json", work / "raw")
    sub_path = work / "annotations" / "subset_instances_train2017.json"

    # 2. subset: our 800 + random others
    if not sub_path.exists():
        print("loading full annotations ...", flush=True)
        D = json.load(open(full))
        ours = json.load(open(base / "annotations" / "subset_instances_train2017.json"))
        keep = {im["id"] for im in ours["images"]}
        others = sorted(im["id"] for im in D["images"] if im["id"] not in keep)
        random.Random(args.seed).shuffle(others)
        keep |= set(others[: max(0, args.n - len(keep))])
        S = {"info": D.get("info", {}), "licenses": D.get("licenses", []), "categories": D["categories"],
             "images": sorted([im for im in D["images"] if im["id"] in keep], key=lambda r: r["id"]),
             "annotations": [a for a in D["annotations"] if a["image_id"] in keep]}
        json.dump(S, open(sub_path, "w"))
        del D
    S = json.load(open(sub_path))
    images = S["images"]
    print(f"subset: {len(images)} images, {len(S['annotations'])} annotations", flush=True)
    shutil.copy(sub_path, feats / "subset_instances_train2017.json")             # keep a record next to the features

    # 3. images
    img_dir = work / "images" / "train2017"
    todo = []
    for im in images:
        dst = img_dir / im["file_name"]
        if dst.exists():
            continue
        local = base / "images" / "train2017" / im["file_name"]
        if local.exists():
            shutil.copy(local, dst)
        else:
            todo.append(im)
    if todo:
        print(f"downloading {len(todo)} images with {args.threads} threads ...", flush=True)
        with ThreadPoolExecutor(args.threads) as ex:
            ok = list(ex.map(lambda im: fetch(im["coco_url"], img_dir / im["file_name"]), todo))
        missing = sum(not o for o in ok)
        assert missing == 0, f"{missing} images could not be downloaded: run the cell again"
    print(f"images ready ({time.time() - t0:.0f}s)", flush=True)

    # 4. student inputs (original + mirror, 256 x 256), in image-id order like teacher_features.py
    out_in = feats / "student_inputs_256.npy"
    if not out_in.exists():
        n = len(images)
        arr = np.lib.format.open_memmap(str(out_in) + ".part.npy", mode="w+", dtype=np.uint8,
                                        shape=(n, 2, 3, STUDENT_SIZE, STUDENT_SIZE))

        def one(i):
            img = np.array(Image.open(img_dir / images[i]["file_name"]).convert("RGB"))
            for v, view in enumerate([img, img[:, ::-1]]):
                arr[i, v] = squash(view, STUDENT_SIZE)[0].round().byte().numpy()

        with ThreadPoolExecutor(8) as ex:
            for k, _ in enumerate(ex.map(one, range(n))):
                if (k + 1) % 500 == 0 or k + 1 == n:
                    print(f"  inputs {k + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
        arr.flush(); del arr
        os.replace(str(out_in) + ".part.npy", out_in)

    # 5. fine-tuning targets cache (same function finetune.py uses)
    cache = feats / "finetune_targets_256.npz"
    if not cache.exists():
        os.environ["COCO_DATA_ROOT"] = str(work)
        from .finetune import build_targets
        build_targets(feats, cache)
    z = np.load(cache)
    info = {"images": len(images), "objects": int(len(z["rows"])), "seconds": round(time.time() - t0)}
    json.dump(info, open(feats / "prepare_5k_info.json", "w"))
    print("done:", info)


if __name__ == "__main__":
    main()
