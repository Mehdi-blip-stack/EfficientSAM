# `repro/`: EfficientSAM reproduction code

Our re-implementation of **SAMI pretraining** (Xiong et al., *EfficientSAM*, CVPR 2024), scaled down to our
COCO subset and one Colab T4 GPU, plus the evaluation code used for every model in our report.

* **Run it:** `notebooks/EfficientSAM_Reproduction_Colab.ipynb` (step by step, resumable, on a free T4 GPU).
* **Our results:** `output/repro_results/` (per-object scores, summary tables, figures, logs).

## Pipeline

| Step | Module | What it does |
|---|---|---|
| Prompts | `make_eval_prompts.py` | Freezes one box and one click (distance-transform maximum) per held-out object: 1,620 objects in 199 images |
| Teacher | `teacher_features.py` | Runs frozen SAM ViT-H on the 800 training images + mirrored copies at 1024 px; pools 64×64 features to 16×16 |
| Pretrain | `pretrain.py`, `sami.py` | ViT-Tiny student, 75% masking, cross-attention decoder, linear projection. `--target sami` (SAM features, MSE over all tokens) or `--target mae` (normalised pixels, masked tokens) |
| Fine-tune | `finetune.py` | EfficientSAM-Ti architecture at 256 px, box or click prompts, SAM loss. `--init sami/mae/random`, `--freeze_encoder`, `--epochs`, `--lr_encoder`, `--limit N` (first N images only), `--save_every` |
| 5,000-image set | `prepare_5k.py` | Experiment E: our 800 + 4,200 random COCO train2017 images (downloaded), inputs and targets prepared exactly as for the 800; needs no GPU |
| Evaluate | `evaluate_model.py` | One model per process: mIoU, Dice, Boundary IoU, best-of-3 oracle, encoder/decoder time, memory |
| Summarise | `summarize_results.py` | Tables by size, supercategory, category-macro; paired differences with 95% bootstrap CIs over images |

All scripts are run from the repository root as modules, e.g. `python -m repro.finetune --init sami ...`, and read
three environment variables: `COCO_DATA_ROOT` (our COCO subset), `REPRO_OUT` (results folder) and optionally
`EFFSAM_REPO` (repository root, for `efficient_sam/`).

## Which code is ours

| Component | Source | Status |
|---|---|---|
| `efficient_sam/` (encoder, prompt encoder, mask decoder) | Official EfficientSAM code | Reused unchanged |
| `segment_anything`, `mobile_sam` packages | Official SAM / MobileSAM code | Reused unchanged (teacher and baselines) |
| `common.py: build_student_sam()` | Official `build_efficient_sam()` | Adapted: input size 1024 → 256, no checkpoint |
| `finetune.py: sam_loss()` | Our earlier `from_scratch/tinysam` losses | Adapted: vectorised, padding mask |
| `teacher_features.py`, `sami.py`, `pretrain.py`, rest of `finetune.py`, `prepare_5k.py` | – | Written by our team |
| `make_eval_prompts.py`, `evaluate_model.py`, `summarize_results.py` | – | Written by our team |

## Main results (T4 GPU, 1,620 held-out objects, mIoU, box / click)

**Released models** (evaluated by us with identical prompts)

| Model | Box | Click |
|---|---|---|
| SAM ViT-H | 0.780 | 0.591 |
| EfficientSAM-S | 0.764 | 0.598 |
| SAM ViT-B | 0.764 | 0.546 |
| EfficientSAM-Ti | 0.751 | 0.501 |
| MobileSAM | 0.735 | 0.535 |

**Our students** (EfficientSAM-Ti architecture at 256 px; encoders pretrained by us on 800 images)

| Experiment | Setting | SAMI start | MAE start | Random start |
|---|---|---|---|---|
| Reproduction | 800 images, full fine-tuning | 0.497 / 0.268 | 0.510 / 0.269 | 0.513 / 0.275 |
| B: ablation | 800 images, encoder frozen | 0.487 / 0.257 | 0.482 / 0.226 | 0.504 / 0.231 |
| C: gentler fine-tuning | 800 images, 20 epochs | 0.463 / 0.247 | 0.476 / 0.235 | 0.476 / 0.226 |
| C: gentler fine-tuning | 800 images, encoder LR 1e-5 | 0.476 / 0.253 | 0.486 / 0.252 | 0.496 / 0.242 |
| D: fewer labels | 100 labelled images | 0.151 / 0.101 | 0.159 / 0.105 | 0.190 / 0.110 |
| D: fewer labels | 200 labelled images | 0.241 / 0.140 | 0.243 / 0.139 | 0.275 / 0.144 |
| E: more data | 5,000 images, 16 epochs | 0.577 / 0.313 | 0.571 / 0.308 | 0.568 / 0.297 |
| F: longer training | 5,000 images, 100 epochs | **0.611 / 0.341** | 0.612 / 0.348 | 0.613 / 0.346 |

Key findings:
* With the encoder **frozen**, SAMI beats MAE on single clicks by +0.031 [95% CI +0.019, +0.043]; full fine-tuning
  on 800 images erases this, and a 20-epoch fine-tuning keeps part of it (+0.012).
* **More data is the strongest lever:** 5,000 instead of 800 images raises the SAMI student by +0.080 box mIoU, and
  SAMI becomes the best start (SAMI > MAE > random, as in the paper).
* **Longer training (Experiment F):** 100 instead of 16 epochs on 5,000 images adds +0.033 to +0.045 box mIoU to
  every student, but then all three starts are tied (SAMI − MAE −0.001 [−0.005, +0.003]): SAMI's lead in
  Experiment E was a faster start, not a better end point.
* With only **100 or 200 labelled images**, pretraining does not help: the random start is best on boxes.

Full tables, confidence intervals and timings: `output/repro_results/gpu_t4/results_summary.md`.

## Trained weights
Checkpoints are not stored in git (≈40 MB each). Download them from the GitHub releases:
* **[v1.2-weights](https://github.com/Mehdi-blip-stack/EfficientSAM/releases/tag/v1.2-weights)**:
  `SAMI_finetuned_5k_e100.pt`, SAMI-pretrained encoder fine-tuned on 5,000 images for 100 epochs (`checkpoints/ft_sami_5k_e100`), **our best model**
* **[v1.1-weights](https://github.com/Mehdi-blip-stack/EfficientSAM/releases/tag/v1.1-weights)**:
  `SAMI_finetuned_5k.pt`, SAMI-pretrained encoder fine-tuned on 5,000 images for 16 epochs (`checkpoints/ft_sami_5k`)
* **[v1.0-weights](https://github.com/Mehdi-blip-stack/EfficientSAM/releases/tag/v1.0-weights)**:
  `SAMI_finetuned.pt`, fully fine-tuned on 800 images (`checkpoints/ft_sami`), and
  `SAMI_frozen_encoder.pt`, encoder kept frozen (`checkpoints/ft_sami_frozen`)

Load a fine-tuned student with:
```python
import torch
from repro.common import build_student_sam
model = build_student_sam()
model.load_state_dict(torch.load("model_final.pt", map_location="cpu")["model"])
model.eval()   # input: RGB image resized to 256x256, values in [0, 1]
```

## History
* First version: failed on the GPU under fp16 mixed precision (`scatter` dtype error in `sami.py`); fixed with
  `torch.promote_types` before any result was produced.
* `--freeze_encoder` was added for Experiment B; it does not change runs made without it.
* Experiment C: run names `*_short`, `*_lowlr` added to the evaluation and summary scripts.
* Experiment D: `--limit` documented and `--save_every` added to `finetune.py` (saving every epoch is the default,
  so earlier runs are unaffected); run names `*_n100`, `*_n200`.
* Experiment E: `prepare_5k.py` added; run names `*_5k`.
* Experiment F: run names `*_5k_e100`. Fixed resuming in `finetune.py`: the saved random-generator state is moved
  back to the CPU before it is restored (`gen.set_state(s["gen"].cpu())`); before, resuming on a GPU failed with
  "RNG state must be a torch.ByteTensor". It does not change results. `summarize_results.py` now skips models
  without an `efficiency_<model>.json` in the timing table and figure.

The training and evaluation logic is the same code that produced every result in `output/repro_results/`.
Earlier versions are not kept separately: git history records every change from here on.
