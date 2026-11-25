# EfficientSAM: Reproduction, Experiments and Deployment

Course project *Machine Learning Paper Reproduction & Deployment* (Fall 2026) on
**EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything** (Xiong et al., CVPR 2024).

**Team:** Abrar Jameel Khatri (36241), Muhammad Taqi Salehi (36235), Syed Muhammad Mehdi Naqvi (36219)

This repository is a fork of the official [yformer/EfficientSAM](https://github.com/yformer/EfficientSAM).
Our work is on the branch **`coco-eda-sharing`**. The original authors' README is kept unchanged at the end of this file.

## Links

| What | Where |
|---|---|
| Code (this branch) | https://github.com/Mehdi-blip-stack/EfficientSAM/tree/coco-eda-sharing |
| Trained models | [v1.3-weights](https://github.com/Mehdi-blip-stack/EfficientSAM/releases/tag/v1.3-weights) (`SAMI_finetuned_100k.pt`, our best model), [v1.2-weights](https://github.com/Mehdi-blip-stack/EfficientSAM/releases/tag/v1.2-weights) (`SAMI_finetuned_5k_e100.pt`), [v1.1-weights](https://github.com/Mehdi-blip-stack/EfficientSAM/releases/tag/v1.1-weights) (`SAMI_finetuned_5k.pt`), [v1.0-weights](https://github.com/Mehdi-blip-stack/EfficientSAM/releases/tag/v1.0-weights) (`SAMI_finetuned.pt`, `SAMI_frozen_encoder.pt`) |
| Results (tables, per-object scores, figures, logs) | [`output/repro_results/`](output/repro_results/) |
| Reproduction code and its documentation | [`repro/`](repro/), [`repro/README.md`](repro/README.md) |
| Streamlit application | [`app/`](app/), [`app/README.md`](app/README.md) |

## What is in this repository

| Folder / file | Contents | Origin |
|---|---|---|
| `efficient_sam/`, `weights/` | EfficientSAM model code and released checkpoints | Official code, reused unchanged |
| `repro/` | SAMI and MAE pretraining, fine-tuning, data preparation, evaluation, summary tables | Written by our team (two functions adapted, see below) |
| `notebooks/EfficientSAM_Reproduction_Colab.ipynb` | The whole pipeline on a free Colab T4 GPU, all experiments, resumable | Written by our team |
| `notebooks/EfficientSAM_EDA_and_Preprocessing.ipynb` | Exploratory data analysis of our COCO subset | Written by our team |
| `app/` | Streamlit application: our trained model and the released models, side by side | Written by our team |
| `output/` | EDA figures and reports, all reproduction and experiment results | Produced by our code |
| `from_scratch/` | Our earlier from-scratch TinySAM experiments (the SAM loss was adapted from here) | Written by our team |
| `requirements.txt` | All dependencies (`requirements-repro.txt` + `app/requirements-app.txt`); `requirements-eda.txt` for the EDA notebook | |

## 1. Setup

Python 3 (we used Python 3.14 locally and Colab’s default Python); a GPU is needed for training (we used free Colab T4 GPUs), not for the app.

```bash
git clone -b coco-eda-sharing https://github.com/Mehdi-blip-stack/EfficientSAM.git
cd EfficientSAM
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt          # Windows: .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Run the application (deployment)

1. Download `SAMI_finetuned_100k.pt` from the [v1.3-weights release](https://github.com/Mehdi-blip-stack/EfficientSAM/releases/tag/v1.3-weights) and put it in `app/weights/` (optionally also the v1.0–v1.2 models).
2. From the repository root:
   ```bash
   .venv/bin/python -m streamlit run app/app.py                 # Windows: .\.venv\Scripts\python.exe -m streamlit run app\app.py
   ```
3. The browser opens at http://localhost:8501. Upload an image (or pick an example), **click** on an object for a point prompt or **drag** a box. The app shows the predicted mask, timings and predicted IoU, and lets you download the mask as a PNG.

Pipeline: input image → preprocessing (RGB, resize to 256 px for our model) → our trained model → 3 candidate masks + predicted IoU → mask overlay and PNG output. Details: [`app/README.md`](app/README.md).

## 3. Reproduce the experiments

Open [`notebooks/EfficientSAM_Reproduction_Colab.ipynb`](notebooks/EfficientSAM_Reproduction_Colab.ipynb) in Colab (T4 GPU) and run the cells in order;
every step saves to Google Drive and resumes after a disconnect. The scripts can also be run directly from the repository root,
e.g. `python -m repro.finetune --init sami --encoder ... --features ... --out ...` (all options: [`repro/README.md`](repro/README.md)).

| Experiment | Question | Run names |
|---|---|---|
| Reproduction | Does SAMI pretraining beat MAE and random initialisation? (800 COCO images) | `ft_sami`, `ft_mae`, `ft_random` |
| A: computational analysis | Accuracy, size and speed of the released EfficientSAM, SAM and MobileSAM models (GPU and CPU) | `effsam_ti`, `effsam_s`, `mobile_sam`, `sam_vit_b`, `sam_vit_h` |
| B: ablation | Are SAMI features better when the encoder is frozen? | `ft_*_frozen` |
| C: gentler fine-tuning | Does shorter training or a lower encoder learning rate keep the SAMI advantage? | `ft_*_short`, `ft_*_lowlr` |
| D: fewer labels | Does SAMI help more with 100 / 200 labelled images? | `ft_*_n100`, `ft_*_n200` |
| E: more labels | Does SAMI's advantage appear with 5,000 images? | `ft_*_5k` |
| F: longer training | 5,000 images for 100 epochs: does SAMI stay ahead? | `ft_*_5k_e100` |
| G: more variety | 100,000 images for 5 epochs (same steps as F): does more variety help? Notebook: `notebooks/Experiment_G.ipynb` | `ft_*_100k` |

Evaluation: 1,620 held-out objects in 199 COCO val2017 images, identical frozen box and click prompts for every model,
mIoU / Dice / Boundary IoU with 95% bootstrap confidence intervals. Summary: [`output/repro_results/gpu_t4/results_summary.md`](output/repro_results/gpu_t4/results_summary.md).

## 4. Code and result attribution

* **(c) Reused unchanged:** `efficient_sam/` and the released EfficientSAM weights (official repository); the `segment_anything` and `mobile_sam` packages (used only as the SAMI teacher and as baselines).
* **(b) Adapted:** `repro/common.py: build_student_sam()` from the official `build_efficient_sam()` (input size 1024 → 256, no checkpoint loaded); `repro/finetune.py: sam_loss()` from our own earlier `from_scratch/tinysam` losses (vectorised, padding mask).
* **(a) Written by our team:** everything else in `repro/` (teacher features, SAMI/MAE pretraining, fine-tuning, data preparation, evaluation, statistics), the notebooks, and the Streamlit app in `app/`.
* **(d) Results:** numbers cited from the paper are the authors' published results and are always marked as such in our report. Every other number, table and figure (including the evaluation of the released checkpoints) was produced by our team with this code.

The official repository releases model code and weights but no training code; SAMI pretraining and fine-tuning were re-implemented by us from the paper.

---

# Original EfficientSAM README (yformer/EfficientSAM, unchanged)

## Update -- Efficient Track Anything

Our new release on [**Efficient Track Anything**](https://github.com/yformer/EfficientTAM).

* Efficient Track Anything code: https://github.com/yformer/EfficientTAM
* Efficient Track Anything project (with gradio demo): https://yformer.github.io/efficient-track-anything/
* Efficient Track Anything paper: https://arxiv.org/pdf/2411.18933

![Efficient Track Anything design](figs/examples/overview.png)

**Efficient Track Anything** is an efficient foundation model for promptable unified image and video segmentation.

[`🤗Efficient Track Anything for video segmentation`](https://5c9036562e75ee2d4d.gradio.live)

[`🤗Efficient Track Anything for image segment everything`](https://5239f8e221db7ee8a0.gradio.live)

[`🤗Efficient Track Anything checkpoints`](https://huggingface.co/yunyangx/efficient-track-anything/tree/main)

# EfficientSAM
EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything

## News
[Jan.12 2024] ONNX version of EfficientSAM including separate encoder and decoder is available on the [Hugging Face Space](https://huggingface.co/spaces/yunyangx/EfficientSAM/tree/main) (thanks to @wkentaro Kentaro Wada for implementing onnx export)

[Dec.31 2023] EfficientSAM is integrated into the annotation tool, [Labelme](https://github.com/labelmeai/labelme) (huge thanks to lableme team and @wkentaro Kentaro Wada)

[Dec.11 2023] The EfficientSAM model code with checkpoints is fully available in this repository. The [example](https://github.com/yformer/EfficientSAM/blob/main/EfficientSAM_example.py) script shows how to instantiate the model with checkpoint and query points on an image.

[Dec.10 2023] Grounded EfficientSAM demo is available on [Grounded-Efficient-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main/EfficientSAM) (huge thanks to IDEA-Research team and @rentainhe for supporting [grounded-efficient-sam demo](https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/EfficientSAM/grounded_efficient_sam.py) under [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)).

[Dec.6 2023] EfficientSAM demo is available on the [Hugging Face Space](https://huggingface.co/spaces/yunyangx/EfficientSAM) (huge thanks to all the HF team for their support).

[Dec.5 2023] We release the torchscript version of EfficientSAM and share a colab.

## Online Demo & Examples
Online demo and examples can be found in the [project page](https://yformer.github.io/efficient-sam/).

## EfficientSAM Instance Segmentation Examples
  |   |   |
:-------------------------:|:-------------------------:
Point-prompt | ![point-prompt](figs/examples/demo_point.png)
Box-prompt |  ![box-prompt](figs/examples/demo_box.png)
Segment everything |![segment everything](figs/examples/demo_everything.png)
Saliency | ![Saliency](figs/examples/demo_saliency.png)

## Model
EfficientSAM checkpoints are available under the weights folder of this github repository. Example instantiations and run of the models can be found in [EfficientSAM_example.py](https://github.com/yformer/EfficientSAM/blob/main/EfficientSAM_example.py).

| EfficientSAM-S | EfficientSAM-Ti |
|------------------------------|------------------------------|
| [Download](https://github.com/yformer/EfficientSAM/blob/main/weights/efficient_sam_vits.pt.zip) |[Download](https://github.com/yformer/EfficientSAM/blob/main/weights/efficient_sam_vitt.pt)|

You can directly use EfficientSAM with checkpoints,
```
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
efficientsam = build_efficient_sam_vitt()
```

## Jupyter Notebook Example
The notebook is shared [here](https://github.com/yformer/EfficientSAM/blob/main/notebooks)


## Acknowledgement

+ [SAM](https://github.com/facebookresearch/segment-anything)
+ [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
+ [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)
+ [U-2-Net](https://github.com/xuebinqin/U-2-Net)


If you're using EfficientSAM in your research or applications, please cite using this BibTeX:
```bibtex


@article{xiong2023efficientsam,
  title={EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything},
  author={Yunyang Xiong, Bala Varadarajan, Lemeng Wu, Xiaoyu Xiang, Fanyi Xiao, Chenchen Zhu, Xiaoliang Dai, Dilin Wang, Fei Sun, Forrest Iandola, Raghuraman Krishnamoorthi, Vikas Chandra},
  journal={arXiv:2312.00863},
  year={2023}
}
```
