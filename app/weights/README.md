Put our fine-tuned checkpoints here (`*.pt`). Download them from the GitHub releases:

* [v1.1-weights](https://github.com/Mehdi-blip-stack/EfficientSAM/releases/tag/v1.1-weights):
  `SAMI_finetuned_5k.pt`, fine-tuned on 5,000 COCO images (**our best model; the app selects it first**)
* [v1.0-weights](https://github.com/Mehdi-blip-stack/EfficientSAM/releases/tag/v1.0-weights):
  `SAMI_finetuned.pt` (800 images) and `SAMI_frozen_encoder.pt` (800 images, encoder frozen)

Every `*.pt` file in this folder appears in the app's model list as "Ours: <file name>".
The released EfficientSAM weights are read from the repository's `weights/` folder.
