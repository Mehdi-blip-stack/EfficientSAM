"""Experiment G, step 1: build a 100,000-image fine-tuning set from COCO train2017. Written by our team.

The set is our 5,000-image set of Experiment E (which contains our original 800 training images) plus images
drawn at random (fixed seed) from the rest of COCO train2017, so 800 < 5,000 < 100,000 are nested and none of
them overlaps our 200 held-out val2017 evaluation images. Only fine-tuning uses it: the SAMI and MAE encoders
stay the ones pretrained on the original 800 images.

Why a separate script (prepare_5k.py builds the 5,000-image set): 100,000 images do not fit in Colab's memory
the way 5,000 do (all images + their mirrors would be ~40 GB, all masks unpacked ~48 GB). This script
  * streams every image from images.cocodataset.org straight into an on-disk array at 256 x 256 (no raw image is
    kept), WITHOUT the mirrored copy: finetune.py --large flips the image on the fly, which gives exactly the same
    pixels because the bilinear resize is symmetric;
  * stores the masks bit-packed in an uncompressed on-disk array, so finetune.py --large reads only the masks of
    the current batch.
Inputs, masks and boxes are made exactly as in prepare_5k.py / finetune.build_targets (same squash() resize,
same mask rules: non-crowd, >= 4 px, still visible at 256 px).

Outputs in --out (local disk; ~20 GB inputs + ~6 GB masks):
  image_ids.npy, student_inputs_256_single.npy (N,3,256,256 uint8), targets_rows.npy (K,5 int32: image index,
  x0, y0, x1, y1), targets_masks.npy (>=K, 8192 uint8, bit-packed 256x256), prepare_100k_info.json
Every stage is skipped/resumed if its output exists, so re-running after a disconnect continues.
Deterministic: the same seed gives the same images, masks and order on every machine (checked by a fingerprint
in prepare_100k_info.json, which finetune.py --large stores in its checkpoint).

Usage:
  python -m repro.prepare_100k --base_subset .../effsam_work/features_5k/subset_instances_train2017.json \
                               --out /content/features_100k
"""
import argparse
import hashlib
import io
import json
import os
import random
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .common import MIN_MASK_PIXELS, STUDENT_SIZE, decode, squash

S = STUDENT_SIZE
PACKED = S * S // 8
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def fetch_bytes(url, tries=6):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:                                   # network hiccup: wait and retry
            if k == tries - 1:
                print("  failed:", url, e, flush=True)
                return None
            time.sleep(2 * (k + 1))


# ------------------------------------------------------------------------------------------------ masks (worker)
G = {}                                                         # filled before the pool forks (copy-on-write)


def masks_for_chunk(lo_hi):
    torch.set_num_threads(1)
    lo, hi = lo_hi
    rows, packed = [], []
    for i in range(lo, hi):
        h, w, anns = G["imgs"][i]
        for ann in anns:
            m = decode(ann, h, w)
            if m.sum() < MIN_MASK_PIXELS:
                continue
            m256 = F.interpolate(torch.from_numpy(m)[None, None].float(), (S, S), mode="bilinear")[0, 0] > 0.5
            if m256.sum() == 0:                                  # thinner than one pixel after resizing
                continue
            ys, xs = torch.where(m256)
            rows.append([i, xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1])
            packed.append(np.packbits(m256.numpy().reshape(-1)))
    return (np.array(rows, np.int32).reshape(-1, 5),
            np.stack(packed) if packed else np.zeros((0, PACKED), np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_subset", required=True, help="subset_instances_train2017.json of the 5,000-image set")
    ap.add_argument("--out", required=True, help="local folder for the arrays (~26 GB)")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=36222)
    ap.add_argument("--threads", type=int, default=48, help="download threads")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2, help="processes for the masks")
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # 1. full train2017 annotations (~250 MB zip)
    full = out / "raw" / "annotations" / "instances_train2017.json"
    if not full.exists():
        (out / "raw").mkdir(exist_ok=True)
        z = out / "raw" / "annotations_trainval2017.zip"
        if not z.exists():
            print("downloading COCO annotations (~250 MB) ...", flush=True)
            data = fetch_bytes(ANN_URL); assert data, "could not download the COCO annotations"
            z.write_bytes(data); del data
        with zipfile.ZipFile(z) as f:
            f.extract("annotations/instances_train2017.json", out / "raw")
    print(f"loading annotations ... ({time.time() - t0:.0f}s)", flush=True)
    D = json.load(open(full))

    # 2. subset: the 5,000-image set + random others (sorted by image id, like the other feature sets)
    base = {im["id"] for im in json.load(open(args.base_subset))["images"]}
    others = sorted(im["id"] for im in D["images"] if im["id"] not in base)
    random.Random(args.seed).shuffle(others)
    keep = base | set(others[: max(0, args.n - len(base))])
    images = sorted([im for im in D["images"] if im["id"] in keep], key=lambda r: r["id"])
    ids = np.array([im["id"] for im in images], np.int64)
    if (out / "image_ids.npy").exists():
        assert np.array_equal(np.load(out / "image_ids.npy"), ids), "image list differs from the one on disk"
    np.save(out / "image_ids.npy", ids)
    n = len(images)
    print(f"subset: {n} images ({len(base)} from the 5,000-image set) ({time.time() - t0:.0f}s)", flush=True)

    # 3. student inputs: stream, squash to 256 x 256, write; done flags make it resumable
    inp = out / "student_inputs_256_single.npy"
    flags_p = out / "inputs_done.npy"
    if not (out / "inputs_complete").exists():
        arr = np.lib.format.open_memmap(str(inp), mode="r+" if inp.exists() else "w+", dtype=np.uint8, shape=(n, 3, S, S))
        flags = np.lib.format.open_memmap(str(flags_p), mode="r+" if flags_p.exists() else "w+", dtype=np.bool_, shape=(n,))
        todo = [i for i in range(n) if not flags[i]]
        print(f"images: {n - len(todo)} already done, downloading {len(todo)} with {args.threads} threads ...", flush=True)
        torch.set_num_threads(1)
        lock, count = threading.Lock(), [0]

        def one(i):
            data = fetch_bytes(images[i]["coco_url"])
            if data is None:
                return False
            img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
            arr[i] = squash(img, S)[0].round().byte().numpy()
            flags[i] = True
            with lock:
                count[0] += 1
                if count[0] % 5000 == 0:
                    arr.flush(); flags.flush()
                    print(f"  images {n - len(todo) + count[0]}/{n} ({time.time() - t0:.0f}s)", flush=True)
            return True

        for attempt in range(3):
            with ThreadPoolExecutor(args.threads) as ex:
                list(ex.map(one, todo))
            arr.flush(); flags.flush()
            todo = [i for i in range(n) if not flags[i]]
            if not todo:
                break
            print(f"  {len(todo)} images failed, retrying ...", flush=True)
        assert not todo, f"{len(todo)} images could not be downloaded: run the cell again (it continues)"
        del arr, flags
        (out / "inputs_complete").write_text("ok")
    print(f"images ready ({time.time() - t0:.0f}s)", flush=True)

    # 4. masks + boxes (same rules as finetune.build_targets), written in image order
    rows_p, masks_p = out / "targets_rows.npy", out / "targets_masks.npy"
    if not rows_p.exists():
        by_img = {}
        for a in D["annotations"]:
            if a["image_id"] in keep and not a["iscrowd"]:
                by_img.setdefault(a["image_id"], []).append({"segmentation": a["segmentation"]})
        G["imgs"] = [(im["height"], im["width"], by_img.get(im["id"], [])) for im in images]
        upper = sum(len(v) for v in by_img.values())
        del D, by_img
        print(f"masks: decoding up to {upper} objects with {args.workers} processes ...", flush=True)
        mm = np.lib.format.open_memmap(str(masks_p), mode="w+", dtype=np.uint8, shape=(upper, PACKED))
        chunks = [(lo, min(n, lo + 500)) for lo in range(0, n, 500)]
        rows, k = [], 0
        with Pool(args.workers) as pool:
            for c, (r, p) in enumerate(pool.imap(masks_for_chunk, chunks)):
                mm[k:k + len(p)] = p; k += len(p); rows.append(r)
                if (c + 1) % 20 == 0 or c + 1 == len(chunks):
                    print(f"  masks: images {chunks[c][1]}/{n}, objects {k} ({time.time() - t0:.0f}s)", flush=True)
        mm.flush(); del mm
        rows = np.concatenate(rows)
        np.save(str(rows_p) + ".part.npy", rows); os.replace(str(rows_p) + ".part.npy", rows_p)
    rows = np.load(rows_p)
    fp = hashlib.sha1(np.load(out / "image_ids.npy").tobytes() + rows.tobytes()).hexdigest()[:16]
    info = {"images": n, "objects": int(len(rows)), "images_with_objects": int(len(np.unique(rows[:, 0]))),
            "seed": args.seed, "fingerprint": fp, "seconds": round(time.time() - t0)}
    json.dump(info, open(out / "prepare_100k_info.json", "w"))
    print("done:", info, flush=True)


if __name__ == "__main__":
    main()
