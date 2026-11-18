Put our fine-tuned checkpoints here (`*.pt`, the `model_final.pt` files written by `repro/finetune.py`).
Rename each one so the app shows a clear name, e.g.

    app/weights/SAMI_finetuned.pt          <- effsam_work/checkpoints/ft_sami/model_final.pt
    app/weights/SAMI_frozen_encoder.pt     <- effsam_work/checkpoints/ft_sami_frozen/model_final.pt

Every `*.pt` file in this folder appears in the app's model list as "Ours: <file name>".
The released EfficientSAM weights are read from the repository's `weights/` folder.
