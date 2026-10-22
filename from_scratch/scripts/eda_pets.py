"""Produce a raw-data EDA for the Oxford-IIIT Pets segmentation dataset.

This intentionally reads the *original* RGB images, trimaps, and official
split lists.  It does not resize, normalize, cache, augment, or sample prompts:
those actions belong to the following preprocessing deliverable.

Run from ``from_scratch`` after downloading only the raw data::

    python scripts/eda_pets.py

The command writes figures, a JSON summary, and a Markdown section ready to
paste into the project report under ``runs/eda/oxford_iiit_pets``.
"""

from __future__ import annotations


import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Keep Matplotlib's cache inside the project. This avoids permission errors on
# locked-down Windows profiles and makes the script work without a GUI backend.
MPL_CONFIG_DIR = PROJECT_ROOT / "runs" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
import numpy as np
from PIL import Image, UnidentifiedImageError


VALID_TRIMAP_LABELS = {1, 2, 3}
SPECIES_NAMES = {1: "cat", 2: "dog"}


@dataclass(frozen=True)
class Sample:
    """One row from an official Oxford-IIIT Pets split file."""

    image_id: str
    source_split: str
    project_split: str
    class_id: int
    species_id: int
    breed_id: int
    breed: str


def breed_from_image_id(image_id: str) -> str:
    """Remove the final numeric image index from an official image identifier."""

    return image_id.rsplit("_", maxsplit=1)[0]


def read_manifest(path: Path, source_split: str) -> tuple[list[Sample], list[str]]:
    """Read an official annotation list and retain malformed rows for reporting."""

    samples: list[Sample] = []
    malformed_rows: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 4:
            malformed_rows.append(f"{path.name}:{line_number}: expected four columns")
            continue
        image_id, class_id, species_id, breed_id = fields
        try:
            samples.append(
                Sample(
                    image_id=image_id,
                    source_split=source_split,
                    project_split=source_split,
                    class_id=int(class_id),
                    species_id=int(species_id),
                    breed_id=int(breed_id),
                    breed=breed_from_image_id(image_id),
                )
            )
        except ValueError:
            malformed_rows.append(f"{path.name}:{line_number}: metadata fields must be integers")
    return samples, malformed_rows


def project_splits(trainval: list[Sample], test: list[Sample]) -> list[Sample]:
    """Apply the PR's deterministic 90/10 train/validation split to trainval."""

    assigned: list[Sample] = []
    for index, sample in enumerate(trainval):
        assigned.append(
            Sample(
                **{
                    **asdict(sample),
                    "project_split": "val" if index % 10 == 0 else "train",
                }
            )
        )
    assigned.extend(
        Sample(**{**asdict(sample), "project_split": "test"}) for sample in test
    )
    return assigned


def require_raw_dataset(raw_root: Path) -> None:
    """Fail early with an actionable message instead of silently analysing caches."""

    required = (
        raw_root / "images",
        raw_root / "annotations" / "trimaps",
        raw_root / "annotations" / "trainval.txt",
        raw_root / "annotations" / "test.txt",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        missing_text = "\n  - ".join(missing)
        raise FileNotFoundError(
            "The original Oxford-IIIT Pets files are required for raw-data EDA. "
            "Missing:\n  - "
            f"{missing_text}\n\n"
            "Download and extract the official image and annotation archives into "
            f"{raw_root} first. Do not point this script at a resized NPZ cache."
        )


def image_and_mask_paths(sample: Sample, raw_root: Path) -> tuple[Path, Path]:
    return (
        raw_root / "images" / f"{sample.image_id}.jpg",
        raw_root / "annotations" / "trimaps" / f"{sample.image_id}.png",
    )


def load_rgb_and_trimap(sample: Sample, raw_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one original pair without changing dimensions or labels."""

    image_path, trimap_path = image_and_mask_paths(sample, raw_root)
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    with Image.open(trimap_path) as trimap:
        trimap_array = np.asarray(trimap)
    return rgb, trimap_array


def inspect_sample(sample: Sample, raw_root: Path) -> dict:
    """Record data-quality facts for one expected image/trimap pair."""

    image_path, trimap_path = image_and_mask_paths(sample, raw_root)
    record = {
        **asdict(sample),
        "image_path": str(image_path),
        "trimap_path": str(trimap_path),
        "image_exists": image_path.exists(),
        "trimap_exists": trimap_path.exists(),
        "readable": False,
        "image_mode": None,
        "width": None,
        "height": None,
        "trimap_width": None,
        "trimap_height": None,
        "dimension_match": None,
        "unexpected_trimap_labels": [],
        "project_foreground_fraction": None,
        "strict_foreground_fraction": None,
        "unclassified_fraction": None,
        "empty_foreground": None,
        "full_foreground": None,
        "prompt_feasible": None,
        "analysis_eligible": False,
        "error": None,
    }
    if not record["image_exists"] or not record["trimap_exists"]:
        missing = []
        if not record["image_exists"]:
            missing.append("image")
        if not record["trimap_exists"]:
            missing.append("trimap")
        record["error"] = f"missing {' and '.join(missing)}"
        return record

    try:
        with Image.open(image_path) as image:
            image.load()
            image_size = image.size
            image_mode = image.mode
        with Image.open(trimap_path) as trimap:
            trimap.load()
            trimap_size = trimap.size
            trimap_array = np.asarray(trimap)
    except (OSError, ValueError, UnidentifiedImageError) as error:
        record["error"] = f"unreadable: {error}"
        return record

    record.update(
        {
            "readable": True,
            "image_mode": image_mode,
            "width": int(image_size[0]),
            "height": int(image_size[1]),
            "trimap_width": int(trimap_size[0]),
            "trimap_height": int(trimap_size[1]),
            "dimension_match": image_size == trimap_size,
        }
    )
    if trimap_array.ndim != 2:
        record["error"] = f"trimap has {trimap_array.ndim} dimensions instead of one label channel"
        return record

    labels, label_counts = np.unique(trimap_array, return_counts=True)
    record["unexpected_trimap_labels"] = [
        int(label) for label in labels if int(label) not in VALID_TRIMAP_LABELS
    ]
    if record["unexpected_trimap_labels"]:
        # Do not interpret an unknown source value as either foreground or
        # background. It remains visible in the integrity audit only.
        record["error"] = "unexpected trimap label values"
        return record

    project_binary_mask = trimap_array != 2
    strict_foreground_mask = trimap_array == 1
    pixels = int(project_binary_mask.size)
    foreground_pixels = int(project_binary_mask.sum())
    record.update(
        {
            # Label 3 is "not classified" in the supplied dataset README. The
            # current TinySAM project policy treats it as foreground, so retain
            # both this project-target coverage and the strict label-1 coverage.
            "project_foreground_fraction": foreground_pixels / pixels,
            "strict_foreground_fraction": int(strict_foreground_mask.sum()) / pixels,
            "unclassified_fraction": int((trimap_array == 3).sum()) / pixels,
            "empty_foreground": foreground_pixels == 0,
            "full_foreground": foreground_pixels == pixels,
            "prompt_feasible": 0 < foreground_pixels < pixels,
            "analysis_eligible": bool(record["dimension_match"]),
            "trimap_label_counts": {str(int(label)): int(count) for label, count in zip(labels, label_counts)},
        }
    )
    if not record["dimension_match"]:
        record["error"] = "image and trimap dimensions differ"
    return record


def count_by(records: Iterable[dict], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record[key]) for record in records).items()))


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    values = list(values)
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def integrity_summary(records: list[dict], malformed_rows: list[str], raw_lists: dict[str, list[Sample]]) -> dict:
    """Summarise missing files, metadata overlap, and raw-mask anomalies."""

    source_ids = {name: [sample.image_id for sample in samples] for name, samples in raw_lists.items()}
    trainval_ids = set(source_ids["trainval"])
    test_ids = set(source_ids["test"])
    duplicate_ids = {
        name: sorted(image_id for image_id, count in Counter(ids).items() if count > 1)
        for name, ids in source_ids.items()
    }
    unreadable = [record for record in records if not record["readable"]]
    loaded = [record for record in records if record["readable"]]
    unexpected_labels = [record for record in loaded if record["unexpected_trimap_labels"]]
    dimension_mismatches = [record for record in loaded if not record["dimension_match"]]
    analysable = [record for record in loaded if record["analysis_eligible"]]
    empty_masks = [record for record in analysable if record["empty_foreground"]]
    full_masks = [record for record in analysable if record["full_foreground"]]
    outlier_masks = [
        record
        for record in analysable
        if record["project_foreground_fraction"] < 0.01
        or record["project_foreground_fraction"] > 0.99
    ]
    return {
        "expected_pairs": len(records),
        "readable_pairs": len(loaded),
        "missing_images": sum(not record["image_exists"] for record in records),
        "missing_trimaps": sum(not record["trimap_exists"] for record in records),
        "unreadable_pairs": len(unreadable),
        "image_modes": count_by(loaded, "image_mode"),
        "non_rgb_images": [record["image_id"] for record in loaded if record["image_mode"] != "RGB"],
        "metadata_malformed_rows": malformed_rows,
        "duplicate_image_ids": duplicate_ids,
        "trainval_test_overlap": sorted(trainval_ids & test_ids),
        "dimension_mismatches": [record["image_id"] for record in dimension_mismatches],
        "unexpected_trimap_labels": {
            record["image_id"]: record["unexpected_trimap_labels"] for record in unexpected_labels
        },
        "analysis_excluded_pairs": [
            record["image_id"] for record in loaded if not record["analysis_eligible"]
        ],
        "empty_foreground_masks": [record["image_id"] for record in empty_masks],
        "full_foreground_masks": [record["image_id"] for record in full_masks],
        "foreground_coverage_outliers": [record["image_id"] for record in outlier_masks],
        "prompt_infeasible_masks": sum(not record["prompt_feasible"] for record in analysable),
    }


def plot_examples(records: list[dict], raw_root: Path, output: Path, seed: int, count: int) -> None:
    """Show original RGB, original trimap, and derived binary target together."""

    feasible = [record for record in records if record["analysis_eligible"] and record["prompt_feasible"]]
    if not feasible:
        return
    rng = random.Random(seed)
    selected = rng.sample(feasible, k=min(count, len(feasible)))
    figure, axes = plt.subplots(len(selected), 3, figsize=(11, 3.4 * len(selected)), squeeze=False)
    trimap_colours = {1: "#44aa99", 2: "#1f3b5c", 3: "#e6ab02"}
    legend = [
        Patch(color=trimap_colours[1], label="1: pet"),
        Patch(color=trimap_colours[2], label="2: background"),
        Patch(color=trimap_colours[3], label="3: not classified (included by project policy)"),
    ]
    for row, record in enumerate(selected):
        sample = Sample(**{key: record[key] for key in Sample.__dataclass_fields__})
        rgb, trimap = load_rgb_and_trimap(sample, raw_root)
        binary_target = trimap != 2
        colour_map = np.zeros((*trimap.shape, 3), dtype=np.float32)
        for label, colour in trimap_colours.items():
            colour_map[trimap == label] = to_rgb(colour)
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"Raw RGB: {record['image_id']}")
        axes[row, 1].imshow(colour_map)
        axes[row, 1].set_title("Original trimap")
        axes[row, 2].imshow(binary_target, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title(
            f"Project target (foreground={record['project_foreground_fraction']:.1%})"
        )
        for axis in axes[row]:
            axis.axis("off")
    figure.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("Raw Oxford-IIIT Pets examples and the project binary-target policy", y=0.995)
    figure.tight_layout(rect=(0, 0.05, 1, 0.98))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_breed_counts(records: list[dict], output: Path) -> None:
    """Show manifest-level training-breed counts, independent of file health."""

    train_records = [record for record in records if record["project_split"] == "train"]
    counts = Counter(record["breed"] for record in train_records)
    labels, values = zip(*sorted(counts.items(), key=lambda item: (item[1], item[0])))
    height = max(7, len(labels) * 0.25)
    figure, axis = plt.subplots(figsize=(9, height))
    axis.barh(labels, values, color="#4c78a8")
    axis.set_title("Training manifest entries per breed (official split)")
    axis.set_xlabel("Images")
    axis.set_ylabel("Breed")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_image_geometry(records: list[dict], output: Path) -> None:
    """Display raw width, height, and aspect-ratio distributions."""

    readable = [
        record
        for record in records
        if record["analysis_eligible"]
    ]
    widths = [record["width"] for record in readable]
    heights = [record["height"] for record in readable]
    ratios = [record["width"] / record["height"] for record in readable]
    figure, axes = plt.subplots(1, 3, figsize=(14, 3.7))
    for axis, values, title, xlabel in zip(
        axes,
        (widths, heights, ratios),
        ("Raw image widths", "Raw image heights", "Raw aspect ratios"),
        ("Pixels", "Pixels", "width / height"),
    ):
        axis.hist(values, bins=35, color="#72b7b2", edgecolor="white")
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Images")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Image geometry before resizing", y=1.02)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_mask_coverage(records: list[dict], output: Path) -> None:
    """Visualise project foreground and raw not-classified-label coverage."""

    readable = [
        record
        for record in records
        if record["analysis_eligible"]
    ]
    foreground = [100 * record["project_foreground_fraction"] for record in readable]
    unclassified = [100 * record["unclassified_fraction"] for record in readable]
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    for axis, values, title, colour in (
        (axes[0], foreground, "Foreground coverage", "#f58518"),
        (axes[1], unclassified, "Not-classified-label coverage", "#54a24b"),
    ):
        axis.hist(values, bins=35, color=colour, edgecolor="white")
        axis.set_title(title)
        axis.set_xlabel("Percent of image pixels")
        axis.set_ylabel("Images")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Project target balance and raw not-classified-label prevalence", y=1.02)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def report_table(rows: list[tuple[str, int]]) -> str:
    header = "| Split | Expected pairs |\n|---|---:|\n"
    return header + "".join(f"| {name} | {count:,} |\n" for name, count in rows)


def write_report(output: Path, summary: dict) -> None:
    """Create a concise Markdown EDA section after real statistics are available."""

    integrity = summary["integrity"]
    geometry = summary["image_geometry"]
    coverage = summary["project_foreground_coverage"]
    strict_coverage = summary["strict_foreground_coverage"]
    unclassified_coverage = summary["unclassified_coverage"]
    breed = summary["breed_distribution"]
    split_rows = [(name, count) for name, count in summary["project_split_counts"].items()]
    breeds = breed["counts"]
    species_breakdown = ", ".join(
        f"{name}: {count:,}" for name, count in sorted(summary["species_counts"].items())
    )
    imbalance = breed["max_count"] / breed["min_count"] if breed["min_count"] else None
    outlier_count = len(integrity["foreground_coverage_outliers"])
    report = f"""# Oxford-IIIT Pets: Exploratory Data Analysis

> Generated by `scripts/eda_pets.py` from the original Oxford-IIIT Pets files. This analysis precedes resizing, normalization, augmentation, caching, and prompt sampling.

## 1. Dataset overview and split integrity

TinySAM is evaluated as a promptable **binary foreground-segmentation** system on Oxford-IIIT Pets. The supplied annotation README defines label 1 as foreground, label 2 as background, and label 3 as not classified. The project policy converts the trimap conceptually to `trimap != 2`, thereby including the not-classified label in its foreground target. This report does not apply that conversion to a cache; it uses it only to describe the label distribution and make that policy explicit.

{report_table(split_rows)}

The official `trainval.txt` list is divided deterministically into the project train/validation split (every tenth entry becomes validation), while `test.txt` remains held out. The raw manifests contain **{integrity['expected_pairs']:,} samples**, **{len(breeds):,} breeds**, and two species ({species_breakdown}). In the training split, the smallest breed has {breed['min_count']:,} images and the largest has {breed['max_count']:,}; the max/min imbalance ratio is {imbalance:.2f}.

Integrity checks found {integrity['missing_images']:,} missing image files, {integrity['missing_trimaps']:,} missing trimaps, {integrity['unreadable_pairs']:,} unreadable pairs, {len(integrity['non_rgb_images']):,} non-RGB images, {len(integrity['dimension_mismatches']):,} image/mask dimension mismatches, and {len(integrity['unexpected_trimap_labels']):,} trimaps with labels outside {{1, 2, 3}}. The {len(integrity['analysis_excluded_pairs']):,} readable pairs with invalid labels or mismatched dimensions are retained in the audit but excluded from quantitative image/mask summaries. There are {len(integrity['trainval_test_overlap']):,} image IDs shared by the official train/validation and test lists.

![Training images per breed](breed_distribution.png)

## 2. Raw visual inspection and label balance

The examples below show the unmodified RGB image, the original three-value trimap, and the binary target used by the project. Because label 3 is not classified rather than a confirmed foreground boundary, including it in the project target is a documented modelling choice that must remain consistent in preprocessing and evaluation.

![Raw examples, trimaps, and derived targets](raw_examples.png)

Across valid, dimension-matched pairs, project-target foreground coverage has median {coverage['median']:.2%} (mean {coverage['mean']:.2%}); strict label-1 foreground coverage has median {strict_coverage['median']:.2%}, and the raw not-classified label has median coverage {unclassified_coverage['median']:.2%}. {outlier_count:,} masks have project-target foreground coverage below 1% or above 99%. Pixel accuracy alone would therefore be misleading when background dominates an image.

![Project foreground and not-classified-label coverage](mask_coverage.png)

## 3. Image geometry and data challenges

Raw image widths range from {geometry['width']['min']:.0f} to {geometry['width']['max']:.0f} pixels and heights from {geometry['height']['min']:.0f} to {geometry['height']['max']:.0f} pixels. Aspect ratios range from {geometry['aspect_ratio']['min']:.2f} to {geometry['aspect_ratio']['max']:.2f}. The varying source geometry motivates aspect-distorting square resizing as an explicit, documented preprocessing trade-off rather than an unnoticed assumption.

![Raw image geometry](image_geometry.png)

The data audit also identifies {len(integrity['empty_foreground_masks']):,} empty masks, {len(integrity['full_foreground_masks']):,} all-foreground masks, and {integrity['prompt_infeasible_masks']:,} masks that cannot support both positive and negative point prompts. These cases must be handled transparently in the later prompt-sampling pipeline; they are not removed in this EDA.

## 4. Evaluation metrics

The primary test metric will be **per-image foreground IoU**, averaged over the held-out test images (mIoU). It directly measures mask overlap and avoids the inflated impression that a background-heavy pixel-accuracy score can give. **Dice/F1** will be reported alongside IoU because it is especially informative when foreground objects are small. To make class imbalance visible, report macro breed/species summaries in addition to the overall mean. Finally, because TinySAM is promptable, report point-prompt mIoU and box-prompt mIoU separately, using a fixed random seed and the same held-out split.
"""
    output.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory containing original images/ and annotations/ folders.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "runs" / "eda" / "oxford_iiit_pets",
        help="Directory for figures, summary JSON, and Markdown report.",
    )
    parser.add_argument("--seed", type=int, default=17, help="Seed for reproducible visual examples.")
    parser.add_argument("--examples", type=int, default=4, help="Number of sample rows in the example figure.")
    args = parser.parse_args()

    raw_root = args.root.resolve()
    output = args.out.resolve()
    require_raw_dataset(raw_root)
    output.mkdir(parents=True, exist_ok=True)

    trainval, malformed_trainval = read_manifest(raw_root / "annotations" / "trainval.txt", "trainval")
    test, malformed_test = read_manifest(raw_root / "annotations" / "test.txt", "test")
    samples = project_splits(trainval, test)
    records = [inspect_sample(sample, raw_root) for sample in samples]
    readable = [record for record in records if record["readable"]]
    analysable = [record for record in readable if record["analysis_eligible"]]
    if not analysable:
        raise RuntimeError(
            "No image/trimap pairs could be analysed. Inspect the integrity findings "
            "and confirm that the original data was extracted correctly."
        )

    project_counts = count_by(records, "project_split")
    breed_counts = Counter(record["breed"] for record in records if record["project_split"] == "train")
    species_counts = count_by(records, "species_id")
    image_widths = [record["width"] for record in analysable]
    image_heights = [record["height"] for record in analysable]
    project_foreground = [record["project_foreground_fraction"] for record in analysable]
    strict_foreground = [record["strict_foreground_fraction"] for record in analysable]
    unclassified = [record["unclassified_fraction"] for record in analysable]
    summary = {
        "dataset": "Oxford-IIIT Pets",
        "analysis_scope": "original RGB images, trimaps, and split lists only",
        "project_split_counts": project_counts,
        "species_counts": {SPECIES_NAMES.get(int(key), f"unknown-{key}"): value for key, value in species_counts.items()},
        "breed_distribution": {
            "counts": dict(sorted(breed_counts.items())),
            "min_count": min(breed_counts.values(), default=0),
            "median_count": float(np.median(list(breed_counts.values()))) if breed_counts else None,
            "max_count": max(breed_counts.values(), default=0),
        },
        "image_geometry": {
            "width": numeric_summary(image_widths),
            "height": numeric_summary(image_heights),
            "aspect_ratio": numeric_summary(width / height for width, height in zip(image_widths, image_heights)),
        },
        "project_foreground_coverage": numeric_summary(project_foreground),
        "strict_foreground_coverage": numeric_summary(strict_foreground),
        "unclassified_coverage": numeric_summary(unclassified),
        "integrity": integrity_summary(
            records,
            malformed_trainval + malformed_test,
            {"trainval": trainval, "test": test},
        ),
    }

    plot_examples(records, raw_root, output / "raw_examples.png", args.seed, args.examples)
    plot_breed_counts(records, output / "breed_distribution.png")
    plot_image_geometry(records, output / "image_geometry.png")
    plot_mask_coverage(records, output / "mask_coverage.png")
    (output / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(output / "eda_report.md", summary)

    print(
        f"Read {summary['integrity']['readable_pairs']:,} image/trimap pairs; "
        f"used {len(analysable):,} valid, dimension-matched pairs for quantitative summaries."
    )
    print(f"Saved report and figures to: {output}")


if __name__ == "__main__":
    main()
