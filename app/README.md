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
* **Our trained model:** download our fine-tuned checkpoints (link: **TO DO**) and copy them into `app/weights/`,
  e.g. `app/weights/SAMI_finetuned.pt` and `app/weights/SAMI_frozen_encoder.pt`. Every `*.pt` file there appears
  in the model list as "Ours: <file name>".
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
Example image "Cat, toilet and sink", one click on the toilet tank, released EfficientSAM-Ti vs -S, on a
2-thread cloud CPU:

| | EfficientSAM-Ti (released) | EfficientSAM-S (released) |
|---|---|---|
| Encoder (first prompt on the image) | 4,893 ms | 10,121 ms |
| Decoder | 247 ms | 184 ms |
| Predicted IoU of the shown mask | 0.85 | 0.97 |
| Mask area | 4.8% of the image | 4.8% of the image |
| Agreement between the two masks (IoU) | 0.98 | |

**TO DO:** the same example with our trained model, and a screenshot.

Our trained model works at 256×256 px, so its encoder is much faster, but its masks are coarser: on our
1,620 held-out objects it reaches 0.50 box / 0.27 click mIoU, against 0.75 / 0.50 for the released
EfficientSAM-Ti (report, Section 6).

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface |
| `segmenter.py` | Model loading, preprocessing, inference (no Streamlit code; reusable) |
| `examples/` | 4 COCO val2017 images with their licences (`CREDITS.md`) |
| `weights/` | Put our fine-tuned `*.pt` checkpoints here |
| `requirements-app.txt` | Dependencies |

All code in `app/` was written by our team. It uses the official `efficient_sam/` code and our `repro/common.py`.
