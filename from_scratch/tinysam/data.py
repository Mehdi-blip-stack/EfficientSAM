"""Oxford-IIIT Pets loading, caching, and prompt sampling (NumPy/Pillow only)."""

from __future__ import annotations

import io
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


IMAGE_URL = "https://thor.robots.ox.ac.uk/pets/images.tar.gz"
ANNOTATION_URL = "https://thor.robots.ox.ac.uk/pets/annotations.tar.gz"
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def download_data(root: str | Path) -> None:
    """Download and extract the official images and annotation archives."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, url in (("images.tar.gz", IMAGE_URL), ("annotations.tar.gz", ANNOTATION_URL)):
        path = root / name
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        with tarfile.open(path, "r:gz") as archive:
            archive.extractall(root)


def _read_split(root: Path, split: str) -> list[tuple[str, str]]:
    annotations = root / "annotations"
    source = annotations / ("trainval.txt" if split in ("train", "val", "trainval") else "test.txt")
    rows = []
    for line in source.read_text().splitlines():
        fields = line.split()
        if fields:
            rows.append((fields[0], fields[1]))
    if split in ("train", "val"):
        # The official trainval list has no validation split; a stable 90/10
        # split is made by the dataset index to keep evaluation reproducible.
        rows = rows[::10] if split == "val" else [row for i, row in enumerate(rows) if i % 10]
    return rows


def build_cache(split: str, size: int, root: str | Path = "from_scratch/data") -> Path:
    """Resize a split and save compact uint8 images/masks to an NPZ cache."""
    root = Path(root)
    raw = root / "raw"
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"pets_{split}_{size}.npz"
    if output.exists():
        return output
    images, masks = [], []
    rows = _read_split(raw, split)
    for stem, _class_id in rows:
        image_path = raw / "images" / f"{stem}.jpg"
        trimap_path = raw / "annotations" / "trimaps" / f"{stem}.png"
        try:
            image = Image.open(image_path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            trimap = Image.open(trimap_path).convert("L").resize((size, size), Image.Resampling.NEAREST)
            mask = (np.asarray(trimap) != 2).astype(np.uint8)
            if not mask.any() or mask.all():
                continue
            images.append(np.asarray(image, dtype=np.uint8).transpose(2, 0, 1))
            masks.append(mask)
        except (OSError, ValueError):
            continue
    np.savez_compressed(output, images=np.asarray(images, dtype=np.uint8), masks=np.asarray(masks, dtype=np.uint8))
    return output


class PetsDataset:
    """Array-backed Pets split, normalized only when an item is requested."""

    def __init__(self, split: str, size: int, root: str | Path = "from_scratch/data",
                 augment: bool = False, seed: int = 0):
        self.path = build_cache(split, size, root)
        data = np.load(self.path)
        self.images = data["images"]
        self.masks = data["masks"]
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.size = size

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image = self.images[index].copy()
        mask = self.masks[index].copy()
        if self.augment and self.rng.random() < 0.5:
            image = image[:, :, ::-1].copy()
            mask = mask[:, ::-1].copy()
        image = image.astype(np.float32) / 255.0
        return (image - IMAGENET_MEAN) / IMAGENET_STD, mask.astype(np.float32)

    def batch(self, indices: np.ndarray, rng: Optional[np.random.Generator] = None):
        """Return normalized BCHW images, binary masks, and sampled prompts."""
        rng = rng or self.rng
        images, masks, coords, labels = [], [], [], []
        for index in indices:
            image, mask = self[index]
            point, label = sample_prompt(mask, rng)
            images.append(image)
            masks.append(mask)
            coords.append(point)
            labels.append(label)
        return (np.stack(images), np.stack(masks), np.stack(coords).astype(np.float32),
                np.stack(labels).astype(np.int64))


def sample_prompt(mask: np.ndarray, rng: Optional[np.random.Generator] = None,
                  max_points: int = 3, box_probability: float = 0.5):
    """Sample points or jittered box corners; coords are (x,y) pixel locations."""
    rng = rng or np.random.default_rng()
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any() or mask.all():
        raise ValueError("prompt sampling requires both foreground and background")
    h, w = mask.shape
    coords = np.zeros((max_points, 2), dtype=np.float32)
    labels = np.full(max_points, -1, dtype=np.int64)
    if rng.random() < box_probability:
        ys, xs = np.where(mask)
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        dx, dy = max(x1 - x0, 1) * 0.1, max(y1 - y0, 1) * 0.1
        coords[0] = [np.clip(x0 + rng.uniform(-dx, dx), 0, w - 1),
                     np.clip(y0 + rng.uniform(-dy, dy), 0, h - 1)]
        coords[1] = [np.clip(x1 + rng.uniform(-dx, dx), 0, w - 1),
                     np.clip(y1 + rng.uniform(-dy, dy), 0, h - 1)]
        labels[:2] = [2, 3]
    else:
        fg_y, fg_x = np.where(mask)
        bg_y, bg_x = np.where(~mask)
        first = int(rng.integers(len(fg_y)))
        coords[0] = [fg_x[first], fg_y[first]]
        labels[0] = 1
        point_count = int(rng.integers(1, max_points + 1))
        for i in range(1, point_count):
            choose_fg = rng.random() < 0.5
            yy, xx = (fg_y, fg_x) if choose_fg else (bg_y, bg_x)
            j = int(rng.integers(len(yy)))
            coords[i] = [xx[j], yy[j]]
            labels[i] = 1 if choose_fg else 0
    return coords, labels


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize an RGB uint8 CHW or HWC array with ImageNet statistics."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3 and image.shape[-1] == 3:
        image = image.transpose(2, 0, 1)
    return (image / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
