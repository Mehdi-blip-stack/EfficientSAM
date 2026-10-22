# EDA phase: Oxford-IIIT Pets

This is the first project deliverable. It deliberately examines the original
Oxford-IIIT Pets files before the later loading/preprocessing pipeline changes
their size, numerical range, or sample content.

## What it produces

`scripts/eda_pets.py` writes the following under `runs/eda/oxford_iiit_pets/`:

- `eda_report.md` — a data-driven 3–4 page report section ready to integrate
  into the final PDF;
- `eda_summary.json` — exact computed figures used in that report;
- `raw_examples.png` — RGB image, source trimap, and derived binary target;
- `breed_distribution.png`, `image_geometry.png`, and `mask_coverage.png`.

The analysis checks image/trimap pairing, corrupted or missing files, split
overlap and duplicate IDs, label values, image-mask dimension mismatches,
empty/full masks, prompt feasibility, breed imbalance, source image geometry,
and binary foreground coverage.

The dataset README defines trimap label 1 as foreground, 2 as background, and
3 as not classified. The current TinySAM project policy uses `trimap != 2`, so
the EDA reports both strict label-1 coverage and the project-target coverage
that includes label 3. This makes the modelling choice visible before the
later preprocessing work adopts it.

## Run it only after the raw data is present

Use an isolated Python environment with NumPy, Pillow, and Matplotlib. This
workspace uses the existing Python 3.14 interpreter for EDA only; the original
guide's Python 3.11 recommendation still applies when the later PyTorch model
work begins. Obtain the original Oxford-IIIT Pets archives and extract them so
this structure exists:

```text
data/raw/
  images/
  annotations/
    trimaps/
    trainval.txt
    test.txt
```

Then run:

```powershell
python scripts/eda_pets.py
```

The EDA command does **not** download data, make an NPZ cache, resize images,
normalize pixels, augment samples, or generate prompts. Those are intentionally
reserved for the next deliverable: the data loading and preprocessing notebook.

If the original image archive is still downloading, the optional
`scripts/run_eda_after_raw_download.ps1` helper can wait until every JPEG and
trimap named by the 7,349 manifest rows is present, then run this same EDA
command automatically.
