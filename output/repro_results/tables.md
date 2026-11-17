## Table R1. Released models on our 200 held-out COCO images (1,620 objects)

| Model | Params (M) | Box mIoU | Click mIoU | Box Boundary IoU | Encoder ms/image, T4 | Encoder ms/image, CPU (2 threads) |
|---|---|---|---|---|---|---|
| MobileSAM | 10.1 | 0.735 [0.722, 0.749] | 0.535 [0.516, 0.555] | 0.657 | 61 | 1,232 |
| EfficientSAM-Ti | 10.2 | 0.751 [0.739, 0.763] | 0.501 [0.480, 0.520] | 0.671 | 110 | 3,763 |
| EfficientSAM-S | 26.4 | 0.764 [0.752, 0.777] | 0.598 [0.578, 0.617] | 0.687 | 276 | 8,413 |
| SAM ViT-B | 93.7 | 0.764 [0.753, 0.777] | 0.546 [0.525, 0.566] | 0.686 | 485 | – |
| SAM ViT-H | 641.1 | 0.780 [0.768, 0.791] | 0.591 [0.569, 0.614] | 0.708 | 2,181 | – |

T4: PyTorch fp32, batch 1, median over 196 images after 3 warm-up images. CPU: 2-thread cloud CPU; SAM ViT-B/H were not timed on CPU.

## Table R2. Our reproduction: student (EfficientSAM-Ti architecture, 256 px) by pretraining and fine-tuning regime

| Encoder start | Fine-tuning | Box mIoU | Click mIoU | Box Dice | Click Dice | Box Boundary IoU |
|---|---|---|---|---|---|---|
| SAMI | full | 0.497 [0.479, 0.516] | 0.268 [0.253, 0.284] | 0.636 | 0.382 | 0.366 |
| MAE | full | 0.510 [0.491, 0.528] | 0.269 [0.254, 0.285] | 0.649 | 0.384 | 0.375 |
| Random | full | 0.513 [0.496, 0.530] | 0.275 [0.259, 0.291] | 0.652 | 0.393 | 0.380 |
| SAMI | frozen encoder | 0.487 [0.469, 0.507] | 0.257 [0.242, 0.273] | 0.627 | 0.370 | 0.352 |
| MAE | frozen encoder | 0.482 [0.462, 0.502] | 0.226 [0.212, 0.241] | 0.621 | 0.334 | 0.340 |
| Random | frozen encoder | 0.504 [0.486, 0.520] | 0.231 [0.217, 0.244] | 0.645 | 0.343 | 0.361 |

## Table R3. Paired differences in mIoU (same 1,620 objects; 95% CI; positive = first better)

| Comparison | Box | Click |
|---|---|---|
| EfficientSAM-S − MobileSAM | +0.029 [+0.023, +0.034] | +0.062 [+0.048, +0.076] |
| EfficientSAM-Ti − MobileSAM | +0.016 [+0.010, +0.022] | -0.035 [-0.052, -0.020] |
| SAM ViT-H − EfficientSAM-S | +0.016 [+0.010, +0.022] | -0.006 [-0.024, +0.009] |
| Ours, full FT: SAMI − MAE | -0.013 [-0.020, -0.006] | -0.001 [-0.008, +0.007] |
| Ours, full FT: SAMI − random | -0.017 [-0.023, -0.010] | -0.007 [-0.016, +0.001] |
| Ours, frozen: SAMI − MAE | +0.005 [-0.002, +0.013] | +0.031 [+0.019, +0.043] |
| Ours, frozen: SAMI − random | -0.016 [-0.023, -0.009] | +0.026 [+0.011, +0.040] |
| Ours: SAMI frozen − SAMI full FT | -0.009 [-0.017, -0.002] | -0.011 [-0.021, -0.001] |

## Table R4. Comparison with the paper

| Model | Paper: COCO AP (detector boxes) [15] | Paper: % of SAM | Ours: box mIoU (GT boxes) | Ours: % of SAM ViT-H |
|---|---|---|---|---|
| SAM ViT-H | 46.5 | 100.0% | 0.780 | 100.0% |
| EfficientSAM-S | 44.4 | 95.5% | 0.764 | 98.0% |
| EfficientSAM-Ti | 42.3 | 91.0% | 0.751 | 96.3% |
| MobileSAM | 38.7 | 83.2% | 0.735 | 94.3% |

| Claim | Paper [15] | Ours |
|---|---|---|
| SAMI vs MAE pretraining, 1-click COCO mIoU (ViT-Ti) | 45.5 vs 43.3 (+2.2) | frozen: 0.257 vs 0.226 (+0.031); full FT: 0.268 vs 0.269 (-0.001) |
| EfficientSAM-S vs SAM: parameters | about 20x fewer | 24x fewer |
| EfficientSAM-S vs SAM: speed | about 20x (A100 throughput) | 7.9x (T4 encoder latency, batch 1) |

## Table R5. Changes made to fit our environment (template Section 6)

| Aspect | Paper [15] | Ours |
|---|---|---|
| SAMI teacher | SAM ViT-H encoder | SAM ViT-H encoder (same) |
| Pretraining data | ImageNet-1K, 1.28 M images | 800 COCO train images × 2 views (original, mirrored) |
| Pretraining resolution / epochs | 224 px, 400 epochs | 256 px, 400 epochs |
| SAMI target | teacher features at full grid | teacher features (64×64) average-pooled to the student's 16×16 grid |
| Student encoder | ViT-Tiny / ViT-Small | ViT-Tiny (EfficientSAM-Ti architecture, 10.2 M params) |
| Fine-tuning data | SA-1B (11 M images, 1.1 B masks) | 800 COCO train images, 5,827 masks |
| Fine-tuning resolution / epochs | 1024 px, 5 epochs | 256 px, 100 epochs |
| Hardware | 64 A100 GPUs | 1 Colab T4 GPU (free tier) |
| Evaluation | COCO AP with ViTDet boxes; 1-click mIoU | prompted mIoU, Dice, Boundary IoU with GT boxes and 1 click, 200 images |

## Table R6. Which code was reused, adapted or written by our team

| Component | Source | Status |
|---|---|---|
| efficient_sam/ (encoder, prompt encoder, mask decoder, EfficientSam) | Official EfficientSAM code | Reused unchanged |
| segment_anything, mobile_sam packages | Official SAM / MobileSAM code | Reused unchanged (baselines only) |
| repro/common.py: build_student_sam() | Official build_efficient_sam() | Adapted: img_size 1024 → 256, no checkpoint |
| repro/finetune.py: sam_loss() | Team's from_scratch/tinysam losses.py | Adapted: vectorised, padding mask |
| repro/teacher_features.py | – | Written by us: teacher pass, view alignment, pooling, caching |
| repro/sami.py | Paper Sec. 3.1, Eq. 1 | Written by us: masked encoder pass, cross-attention decoder, projection, SAMI and MAE losses |
| repro/pretrain.py | – | Written by us: training loop, schedule, mixed precision, resume |
| repro/finetune.py (rest) | – | Written by us: targets, prompt sampling, training loop, frozen-encoder mode |
| repro/make_eval_prompts.py, evaluate_model.py, summarize_results.py | – | Written by us: frozen prompts, model adapters, metrics, timing, bootstrap |
| EDA and preprocessing notebook | – | Written by us |