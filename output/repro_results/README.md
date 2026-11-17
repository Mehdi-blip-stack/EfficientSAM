# Reproduction results

Produced with `repro/` and `notebooks/EfficientSAM_Reproduction_Colab.ipynb`.
Evaluation set: 1,620 non-crowd objects in 199 held-out COCO val2017 images, identical frozen prompts for every model.

| Folder / file | Contents |
|---|---|
| `gpu_t4/` | All 11 models on a Colab T4 GPU: `per_instance_<model>.csv` (one row per object and model), `efficiency_<model>.json` (parameters, timing, memory), `eval_prompts.json` (the frozen prompts), `results_summary.md` and `results_tables.json` (all tables with 95% bootstrap CIs) |
| `cpu_2threads/` | EfficientSAM-Ti, EfficientSAM-S and MobileSAM on a 2-thread CPU (accuracy identical to the GPU run; CPU timings) |
| `figures/` | Report figures R1–R5 |
| `tables.md` | Report tables R1–R6 |
| `make_report_assets.py` | Rebuilds `figures/` and `tables.md`: `python make_report_assets.py gpu_t4 cpu_2threads logs/finetune_full_log.txt out` |
| `logs/` | `pretrain_loss.csv` (SAMI and MAE loss every 10 epochs), `finetune_full_log.txt` (fine-tuning, encoder trained), `eval_sam_vit_h_log.txt` |

Model names: `effsam_ti`, `effsam_s`, `mobile_sam`, `sam_vit_b`, `sam_vit_h` (released checkpoints);
`ours_{sami,mae,random}` (our students, encoder fine-tuned); `ours_{sami,mae,random}_frozen` (encoder frozen).
