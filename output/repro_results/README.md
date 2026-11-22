# Reproduction and experiment results

Produced with `repro/` and `notebooks/EfficientSAM_Reproduction_Colab.ipynb`.
Evaluation set: 1,620 non-crowd objects in 199 held-out COCO val2017 images, identical frozen prompts for every model.

| Folder / file | Contents |
|---|---|
| `gpu_t4/` | All 29 models on Colab T4 GPUs: `per_instance_<model>.csv` (one row per object), `efficiency_<model>.json` (parameters, timing, memory; not measured for the `_5k_e100` runs, which have the same architecture), `eval_prompts.json` (the frozen prompts), `results_summary.md` and `results_tables.json` (all tables with 95% bootstrap CIs) |
| `cpu_2threads/` | EfficientSAM-Ti, EfficientSAM-S and MobileSAM on a 2-thread CPU (accuracy identical to the GPU run; CPU timings) |
| `figures/` | Report figures: R1–R5 (released models, reproduction, paired differences, training curves), `fig_r6_expC_*` (Experiment C), `fig_r7_labelled_data` (Experiments D and E), `fig_r8_expB_frozen` (Experiment B), `fig_r9_expD_few_labels` (Experiment D), `fig_r10_expF_longer_training` (Experiment F) |
| `figure_scripts/` | The scripts that draw those figures from `gpu_t4/` |
| `tables.md`, `make_report_assets.py` | Report tables R1–R6 and the script that builds them |
| `logs/` | `pretrain_loss.csv` (SAMI and MAE loss every 10 epochs), `finetune_full_log.txt`, `eval_sam_vit_h_log.txt` |

**Model names**
* Released checkpoints: `effsam_ti`, `effsam_s`, `mobile_sam`, `sam_vit_b`, `sam_vit_h`
* Our students, `ours_{sami,mae,random}<suffix>`, where the suffix is the setting:

| Suffix | Setting | Report section |
|---|---|---|
| (none) | 800 images, encoder fine-tuned, 100 epochs | 6.2 |
| `_frozen` | 800 images, encoder frozen | 7.2 (Experiment B) |
| `_short` | 800 images, 20 epochs | 7.3 (Experiment C) |
| `_lowlr` | 800 images, encoder learning rate 1e-5 | 7.3 (Experiment C) |
| `_n100`, `_n200` | only 100 / 200 labelled images, same number of steps | 7.4 (Experiment D) |
| `_5k` | 5,000 images, same number of steps (16 epochs) | 7.5 (Experiment E) |
| `_5k_e100` | 5,000 images, 100 epochs (6.25× more steps) | 7.6 (Experiment F) |
