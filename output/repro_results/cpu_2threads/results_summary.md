# Baseline reproduction results

1620 non-crowd objects in 199 held-out COCO val2017 images; identical frozen prompts for every model; 3 candidate masks, the one with the highest predicted IoU is scored; masks compared at original resolution. Brackets: 95% bootstrap CI over images (2000 resamples).

## A. Segmentation quality (object-level means)

| Model | Prompt | mIoU | Dice | Boundary IoU | Oracle mIoU (best of 3) |
|---|---|---|---|---|---|
| EfficientSAM-Ti | box | 0.751 [0.739, 0.763] | 0.844 [0.836, 0.853] | 0.671 [0.658, 0.684] | 0.788 [0.778, 0.798] |
| EfficientSAM-Ti | click | 0.501 [0.480, 0.520] | 0.614 [0.594, 0.633] | 0.440 [0.417, 0.463] | 0.681 [0.667, 0.696] |
| EfficientSAM-S | box | 0.764 [0.752, 0.777] | 0.853 [0.844, 0.863] | 0.687 [0.675, 0.699] | 0.796 [0.785, 0.807] |
| EfficientSAM-S | click | 0.598 [0.578, 0.617] | 0.699 [0.681, 0.718] | 0.528 [0.506, 0.548] | 0.710 [0.696, 0.724] |
| MobileSAM | box | 0.735 [0.722, 0.749] | 0.832 [0.821, 0.843] | 0.657 [0.644, 0.670] | 0.777 [0.766, 0.788] |
| MobileSAM | click | 0.535 [0.516, 0.555] | 0.643 [0.624, 0.663] | 0.465 [0.444, 0.485] | 0.687 [0.669, 0.703] |

## B. mIoU by COCO object size (recomputed mask area)

| Model | Prompt | small (n=660) | medium (n=600) | large (n=360) |
|---|---|---|---|---|
| EfficientSAM-Ti | box | 0.712 [0.693, 0.730] | 0.767 [0.751, 0.783] | 0.796 [0.776, 0.814] |
| EfficientSAM-Ti | click | 0.556 [0.526, 0.584] | 0.492 [0.463, 0.524] | 0.413 [0.375, 0.453] |
| EfficientSAM-S | box | 0.710 [0.691, 0.728] | 0.794 [0.781, 0.809] | 0.813 [0.794, 0.833] |
| EfficientSAM-S | click | 0.601 [0.572, 0.628] | 0.604 [0.572, 0.634] | 0.580 [0.543, 0.617] |
| MobileSAM | box | 0.677 [0.657, 0.697] | 0.762 [0.745, 0.780] | 0.798 [0.773, 0.821] |
| MobileSAM | click | 0.477 [0.440, 0.509] | 0.580 [0.555, 0.606] | 0.567 [0.529, 0.603] |

## C. Box-prompt mIoU by supercategory

| Supercategory (n) | EfficientSAM-Ti | EfficientSAM-S | MobileSAM |
|---|---|---|---|
| person (497) | 0.760 | 0.778 | 0.739 |
| furniture (177) | 0.656 | 0.680 | 0.649 |
| food (140) | 0.800 | 0.807 | 0.799 |
| vehicle (138) | 0.768 | 0.776 | 0.761 |
| animal (135) | 0.758 | 0.778 | 0.735 |
| kitchen (134) | 0.749 | 0.764 | 0.737 |
| sports (102) | 0.731 | 0.739 | 0.708 |
| indoor (94) | 0.771 | 0.746 | 0.726 |
| accessory (90) | 0.740 | 0.756 | 0.731 |
| outdoor (50) | 0.743 | 0.786 | 0.762 |
| electronic (49) | 0.790 | 0.764 | 0.754 |
| appliance (14) | 0.876 | 0.897 | 0.901 |

Same table for the single-click prompt:

| Supercategory (n) | EfficientSAM-Ti | EfficientSAM-S | MobileSAM |
|---|---|---|---|
| person (497) | 0.443 | 0.621 | 0.535 |
| furniture (177) | 0.420 | 0.484 | 0.441 |
| food (140) | 0.567 | 0.584 | 0.590 |
| vehicle (138) | 0.489 | 0.614 | 0.540 |
| animal (135) | 0.592 | 0.667 | 0.632 |
| kitchen (134) | 0.513 | 0.590 | 0.554 |
| sports (102) | 0.600 | 0.616 | 0.547 |
| indoor (94) | 0.554 | 0.606 | 0.508 |
| accessory (90) | 0.526 | 0.568 | 0.497 |
| outdoor (50) | 0.546 | 0.583 | 0.476 |
| electronic (49) | 0.521 | 0.582 | 0.509 |
| appliance (14) | 0.527 | 0.682 | 0.661 |

## D. Category-macro mIoU (mean over the 44 categories with at least 10 eval objects; every category counts equally)

| Model | box | click |
|---|---|---|
| EfficientSAM-Ti | 0.743 | 0.516 |
| EfficientSAM-S | 0.756 | 0.571 |
| MobileSAM | 0.731 | 0.521 |

Categories with fewer than 10 eval objects are excluded from the macro average; categories with none (fire hydrant, bear, snowboard, toaster) are unavailable.

## E. Paired differences in mIoU (same objects; positive = first model better)

| Comparison | box | click |
|---|---|---|
| EfficientSAM-Ti − MobileSAM | +0.016 [+0.010, +0.021] | -0.035 [-0.051, -0.019] |
| EfficientSAM-S − MobileSAM | +0.029 [+0.023, +0.034] | +0.062 [+0.049, +0.076] |
| EfficientSAM-S − EfficientSAM-Ti | +0.013 [+0.008, +0.018] | +0.097 [+0.083, +0.113] |

## F. Efficiency (CPU, 2 threads, PyTorch 2.14.1+cu130, batch size 1)

| Model | Params total (M) | Params encoder (M) | Encoder ms/image (median) | Decoder ms/prompt (median) | Peak process memory (MB) |
|---|---|---|---|---|---|
| EfficientSAM-Ti | 10.22 | 6.16 | 3763 | 97 | 1483 |
| EfficientSAM-S | 26.41 | 22.35 | 8413 | 108 | 1922 |
| MobileSAM | 10.13 | 6.07 | 1232 | 107 | 1501 |

Encoder time is one forward pass per image (first 3 images excluded as warm-up). Decoder time includes upsampling the masks to the original resolution. Peak memory is the whole Python process (model, image, masks).