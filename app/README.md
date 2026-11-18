# EfficientSAM Streamlit demo

Interactive promptable segmentation with **our trained model** (EfficientSAM-Ti architecture, SAMI-pretrained and
fine-tuned by our team, see `repro/`), with the released EfficientSAM-Ti/S models for comparison.

```
Input image ─► preprocessing ─► trained model ─► 3 candidate masks + predicted IoU ─► mask overlay, timing, PNG download
              (RGB, resize to        (image encoder once per image;
               256 / 1024 px)         prompt encoder + mask decoder per prompt)
```

## 1. Install (once)
From the repository root (the folder that contains `app/`, `repro/`, `efficient_sam/` and `weights/`):

**Windows (PowerShell)**
```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r app\requirements-app.txt
```
**macOS / Linux**
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r app/requirements-app.txt
```
Dependencies: `streamlit`, `streamlit-image-coordinates`, `torch`, `numpy`, `pillow`, `pycocotools`
(exact versions in `app/requirements-app.txt`). A GPU is optional; the app uses one automatically if PyTorch
can see it.

## 2. Add the model weights
* **Our trained models:** download them from the GitHub releases and copy them into `app/weights/`:
  * **[v1.1-weights](https://github.com/Mehdi-blip-stack/EfficientSAM/releases/tag/v1.1-weights)**:
    `SAMI_finetuned_5k.pt`, fine-tuned on 5,000 images: **our best model**, listed first in the app
  * **[v1.0-weights](https://github.com/Mehdi-blip-stack/EfficientSAM/releases/tag/v1.0-weights)**:
    `SAMI_finetuned.pt` (800 images) and `SAMI_frozen_encoder.pt` (800 images, encoder frozen)

  Every `*.pt` file in `app/weights/` appears in the model list as "Ours: <file name>".
* **Released EfficientSAM:** already in the repository's `weights/` folder (`efficient_sam_vitt.pt`,
  `efficient_sam_vits.pt.zip`; the zip is extracted automatically the first time EfficientSAM-S is used).

## 3. Start the app
```powershell
.\.venv\Scripts\python.exe -m streamlit run app\app.py        # Windows
```
```bash
.venv/bin/python -m streamlit run app/app.py                   # macOS / Linux
```
The browser opens at http://localhost:8501.

## 4. How to use it
| | |
|---|---|
| **Input** | A JPEG or PNG image (upload, or pick one of 4 example images from our held-out COCO set). Greyscale and RGBA images are converted to RGB. |
| **Prompt** | **Click** on an object for a point prompt, or **click and drag** a rectangle for a box prompt. A new click or drag replaces the old prompt; *Clear prompt* removes it. |
| **Output** | The mask with the highest predicted IoU, overlaid on the image (blue; orange for the second model), plus: encoder time, decoder time, predicted IoU, mask area, and *Download mask (PNG)* (white = object, black = background, same size as the input) and *Download overlay (PNG)*. |
| **Options** (sidebar) | Choose the model; *Compare with a second model* runs two models on the same prompt side by side with a comparison table and the IoU between their masks; *Show all 3 candidate masks* shows every mask the model returned with its predicted IoU. |

The image encoder runs **once per image and model**; later prompts on the same image reuse the stored embedding
("cached" under the encoder time), so only the fast decoder runs.

## 5. Example inference
Example image "Cat, toilet and sink", one click on the toilet tank, our model vs the released EfficientSAM-Ti in
comparison mode, on our laptop's CPU (screenshots: report Figures 9 and 10):

| | Ours: SAMI_finetuned | EfficientSAM-Ti (released) |
|---|---|---|
| Parameters | 10.2 M | 10.2 M |
| Encoder (first prompt on the image) | 146 ms | 4,903 ms |
| Decoder (per prompt) | 48 ms | 139 ms |
| Predicted IoU of the shown mask | 0.54 | 0.86 |
| Mask area | 6.0% of the image | 4.7% of the image |
| Agreement between the two masks (IoU) | 0.67 | |

Both models find the clicked object. Ours is about 33× faster because it works at 256 px, but its mask is
coarser and has small stray regions.

Our trained models work at 256×256 px, so their encoder is much faster, but their masks are coarser. On our
1,620 held-out objects, mIoU (box / click) is 0.58 / 0.31 for `SAMI_finetuned_5k` and 0.50 / 0.27 for
`SAMI_finetuned`, against 0.75 / 0.50 for the released EfficientSAM-Ti (report, Sections 6 and 7.5).

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface |
| `segmenter.py` | Model loading, preprocessing, inference (no Streamlit code; reusable) |
| `examples/` | 4 COCO val2017 images with their licences (`CREDITS.md`) |
| `weights/` | Put our fine-tuned `*.pt` checkpoints here |
| `requirements-app.txt` | Dependencies |

All code in `app/` was written by our team. It uses the official `efficient_sam/` code and our `repro/common.py`.
