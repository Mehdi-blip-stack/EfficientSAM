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
| SAM ViT-B | box | 0.764 [0.753, 0.777] | 0.853 [0.845, 0.862] | 0.686 [0.673, 0.699] | 0.800 [0.790, 0.810] |
| SAM ViT-B | click | 0.546 [0.525, 0.566] | 0.654 [0.634, 0.674] | 0.480 [0.457, 0.503] | 0.712 [0.698, 0.727] |
| SAM ViT-H | box | 0.780 [0.768, 0.791] | 0.864 [0.855, 0.873] | 0.708 [0.695, 0.720] | 0.809 [0.799, 0.819] |
| SAM ViT-H | click | 0.591 [0.569, 0.614] | 0.691 [0.669, 0.713] | 0.525 [0.501, 0.550] | 0.749 [0.734, 0.764] |
| Ours: SAMI init | box | 0.497 [0.479, 0.516] | 0.636 [0.618, 0.654] | 0.366 [0.353, 0.378] | 0.531 [0.514, 0.547] |
| Ours: SAMI init | click | 0.268 [0.253, 0.284] | 0.382 [0.364, 0.401] | 0.172 [0.161, 0.183] | 0.358 [0.343, 0.373] |
| Ours: MAE init | box | 0.510 [0.491, 0.528] | 0.649 [0.631, 0.667] | 0.375 [0.363, 0.387] | 0.540 [0.522, 0.556] |
| Ours: MAE init | click | 0.269 [0.254, 0.285] | 0.384 [0.366, 0.404] | 0.173 [0.161, 0.184] | 0.367 [0.352, 0.382] |
| Ours: random init | box | 0.513 [0.496, 0.530] | 0.652 [0.634, 0.671] | 0.380 [0.366, 0.393] | 0.551 [0.535, 0.567] |
| Ours: random init | click | 0.275 [0.259, 0.291] | 0.393 [0.371, 0.413] | 0.174 [0.160, 0.186] | 0.374 [0.359, 0.389] |
| Ours: SAMI frozen | box | 0.487 [0.469, 0.507] | 0.627 [0.606, 0.647] | 0.352 [0.339, 0.365] | 0.531 [0.515, 0.547] |
| Ours: SAMI frozen | click | 0.257 [0.242, 0.273] | 0.370 [0.351, 0.391] | 0.156 [0.145, 0.167] | 0.367 [0.350, 0.382] |
| Ours: MAE frozen | box | 0.482 [0.462, 0.502] | 0.621 [0.599, 0.642] | 0.340 [0.325, 0.355] | 0.512 [0.493, 0.530] |
| Ours: MAE frozen | click | 0.226 [0.212, 0.241] | 0.334 [0.316, 0.353] | 0.135 [0.126, 0.145] | 0.335 [0.320, 0.350] |
| Ours: random frozen | box | 0.504 [0.486, 0.520] | 0.645 [0.628, 0.662] | 0.361 [0.345, 0.374] | 0.531 [0.516, 0.547] |
| Ours: random frozen | click | 0.231 [0.217, 0.244] | 0.343 [0.327, 0.361] | 0.143 [0.133, 0.153] | 0.331 [0.318, 0.346] |

## B. mIoU by COCO object size (recomputed mask area)

| Model | Prompt | small (n=660) | medium (n=600) | large (n=360) |
|---|---|---|---|---|
| EfficientSAM-Ti | box | 0.712 [0.694, 0.730] | 0.767 [0.751, 0.782] | 0.796 [0.776, 0.815] |
| EfficientSAM-Ti | click | 0.556 [0.526, 0.584] | 0.492 [0.463, 0.522] | 0.413 [0.374, 0.450] |
| EfficientSAM-S | box | 0.710 [0.693, 0.728] | 0.794 [0.781, 0.808] | 0.813 [0.792, 0.833] |
| EfficientSAM-S | click | 0.601 [0.573, 0.628] | 0.604 [0.575, 0.636] | 0.580 [0.544, 0.618] |
| MobileSAM | box | 0.677 [0.658, 0.696] | 0.762 [0.745, 0.780] | 0.798 [0.774, 0.823] |
| MobileSAM | click | 0.477 [0.439, 0.509] | 0.580 [0.556, 0.605] | 0.567 [0.530, 0.604] |
| SAM ViT-B | box | 0.717 [0.698, 0.735] | 0.788 [0.774, 0.803] | 0.809 [0.788, 0.830] |
| SAM ViT-B | click | 0.560 [0.524, 0.595] | 0.563 [0.534, 0.593] | 0.490 [0.451, 0.528] |
| SAM ViT-H | box | 0.732 [0.712, 0.753] | 0.810 [0.798, 0.822] | 0.816 [0.792, 0.838] |
| SAM ViT-H | click | 0.583 [0.542, 0.622] | 0.616 [0.589, 0.644] | 0.565 [0.530, 0.605] |
| Ours: SAMI init | box | 0.383 [0.362, 0.402] | 0.547 [0.530, 0.564] | 0.621 [0.599, 0.640] |
| Ours: SAMI init | click | 0.163 [0.148, 0.178] | 0.330 [0.312, 0.349] | 0.358 [0.335, 0.384] |
| Ours: MAE init | box | 0.399 [0.377, 0.420] | 0.555 [0.538, 0.572] | 0.636 [0.617, 0.655] |
| Ours: MAE init | click | 0.167 [0.150, 0.185] | 0.328 [0.310, 0.346] | 0.358 [0.334, 0.381] |
| Ours: random init | box | 0.402 [0.380, 0.423] | 0.558 [0.540, 0.575] | 0.643 [0.625, 0.661] |
| Ours: random init | click | 0.174 [0.156, 0.193] | 0.337 [0.320, 0.355] | 0.358 [0.336, 0.381] |
| Ours: SAMI frozen | box | 0.366 [0.341, 0.389] | 0.539 [0.523, 0.554] | 0.623 [0.603, 0.643] |
| Ours: SAMI frozen | click | 0.153 [0.138, 0.169] | 0.319 [0.300, 0.339] | 0.345 [0.322, 0.367] |
| Ours: MAE frozen | box | 0.354 [0.327, 0.379] | 0.537 [0.520, 0.553] | 0.624 [0.604, 0.642] |
| Ours: MAE frozen | click | 0.142 [0.124, 0.161] | 0.280 [0.264, 0.298] | 0.291 [0.266, 0.315] |
| Ours: random frozen | box | 0.404 [0.382, 0.425] | 0.545 [0.527, 0.562] | 0.617 [0.598, 0.637] |
| Ours: random frozen | click | 0.160 [0.145, 0.177] | 0.285 [0.267, 0.302] | 0.272 [0.251, 0.294] |

## C. Box-prompt mIoU by supercategory

| Supercategory (n) | EfficientSAM-Ti | EfficientSAM-S | MobileSAM | SAM ViT-B | SAM ViT-H | Ours: SAMI init | Ours: MAE init | Ours: random init | Ours: SAMI frozen | Ours: MAE frozen | Ours: random frozen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| person (497) | 0.760 | 0.778 | 0.739 | 0.779 | 0.796 | 0.520 | 0.529 | 0.529 | 0.510 | 0.498 | 0.506 |
| furniture (177) | 0.656 | 0.680 | 0.649 | 0.668 | 0.696 | 0.470 | 0.485 | 0.498 | 0.468 | 0.473 | 0.494 |
| food (140) | 0.800 | 0.807 | 0.799 | 0.809 | 0.818 | 0.588 | 0.598 | 0.609 | 0.552 | 0.563 | 0.588 |
| vehicle (138) | 0.768 | 0.776 | 0.761 | 0.777 | 0.789 | 0.498 | 0.503 | 0.503 | 0.485 | 0.486 | 0.518 |
| animal (135) | 0.758 | 0.778 | 0.735 | 0.766 | 0.792 | 0.505 | 0.508 | 0.520 | 0.484 | 0.484 | 0.490 |
| kitchen (134) | 0.749 | 0.764 | 0.737 | 0.770 | 0.778 | 0.506 | 0.510 | 0.522 | 0.499 | 0.502 | 0.515 |
| sports (102) | 0.731 | 0.739 | 0.708 | 0.742 | 0.756 | 0.349 | 0.357 | 0.377 | 0.332 | 0.327 | 0.358 |
| indoor (94) | 0.771 | 0.746 | 0.726 | 0.756 | 0.770 | 0.441 | 0.475 | 0.466 | 0.458 | 0.423 | 0.486 |
| accessory (90) | 0.740 | 0.756 | 0.731 | 0.757 | 0.775 | 0.472 | 0.498 | 0.488 | 0.490 | 0.471 | 0.501 |
| outdoor (50) | 0.743 | 0.786 | 0.762 | 0.773 | 0.760 | 0.477 | 0.509 | 0.489 | 0.461 | 0.461 | 0.511 |
| electronic (49) | 0.790 | 0.764 | 0.754 | 0.789 | 0.806 | 0.500 | 0.530 | 0.532 | 0.478 | 0.478 | 0.518 |
| appliance (14) | 0.876 | 0.897 | 0.901 | 0.914 | 0.913 | 0.574 | 0.661 | 0.664 | 0.658 | 0.642 | 0.686 |

Same table for the single-click prompt:

| Supercategory (n) | EfficientSAM-Ti | EfficientSAM-S | MobileSAM | SAM ViT-B | SAM ViT-H | Ours: SAMI init | Ours: MAE init | Ours: random init | Ours: SAMI frozen | Ours: MAE frozen | Ours: random frozen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| person (497) | 0.443 | 0.621 | 0.535 | 0.503 | 0.555 | 0.306 | 0.297 | 0.320 | 0.289 | 0.252 | 0.263 |
| furniture (177) | 0.420 | 0.484 | 0.441 | 0.460 | 0.494 | 0.273 | 0.271 | 0.273 | 0.272 | 0.234 | 0.237 |
| food (140) | 0.567 | 0.584 | 0.590 | 0.590 | 0.624 | 0.259 | 0.260 | 0.245 | 0.231 | 0.195 | 0.218 |
| vehicle (138) | 0.489 | 0.614 | 0.540 | 0.576 | 0.643 | 0.243 | 0.259 | 0.253 | 0.219 | 0.223 | 0.203 |
| animal (135) | 0.592 | 0.667 | 0.632 | 0.658 | 0.734 | 0.302 | 0.293 | 0.291 | 0.301 | 0.256 | 0.272 |
| kitchen (134) | 0.513 | 0.590 | 0.554 | 0.564 | 0.614 | 0.271 | 0.293 | 0.273 | 0.276 | 0.246 | 0.247 |
| sports (102) | 0.600 | 0.616 | 0.547 | 0.624 | 0.626 | 0.174 | 0.165 | 0.181 | 0.160 | 0.143 | 0.170 |
| indoor (94) | 0.554 | 0.606 | 0.508 | 0.509 | 0.557 | 0.187 | 0.219 | 0.208 | 0.207 | 0.189 | 0.195 |
| accessory (90) | 0.526 | 0.568 | 0.497 | 0.573 | 0.599 | 0.253 | 0.242 | 0.247 | 0.227 | 0.180 | 0.173 |
| outdoor (50) | 0.546 | 0.583 | 0.476 | 0.530 | 0.547 | 0.264 | 0.276 | 0.318 | 0.245 | 0.224 | 0.215 |
| electronic (49) | 0.521 | 0.582 | 0.509 | 0.564 | 0.601 | 0.237 | 0.262 | 0.255 | 0.231 | 0.229 | 0.165 |
| appliance (14) | 0.527 | 0.682 | 0.661 | 0.642 | 0.744 | 0.292 | 0.266 | 0.322 | 0.328 | 0.250 | 0.227 |

## D. Category-macro mIoU (mean over the 44 categories with at least 10 eval objects; every category counts equally)

| Model | box | click |
|---|---|---|
| EfficientSAM-Ti | 0.743 | 0.516 |
| EfficientSAM-S | 0.756 | 0.571 |
| MobileSAM | 0.731 | 0.521 |
| SAM ViT-B | 0.755 | 0.552 |
| SAM ViT-H | 0.768 | 0.595 |
| Ours: SAMI init | 0.482 | 0.242 |
| Ours: MAE init | 0.496 | 0.247 |
| Ours: random init | 0.499 | 0.241 |
| Ours: SAMI frozen | 0.470 | 0.229 |
| Ours: MAE frozen | 0.471 | 0.205 |
| Ours: random frozen | 0.494 | 0.208 |

Categories with fewer than 10 eval objects are excluded from the macro average; categories with none (fire hydrant, bear, snowboard, toaster) are unavailable.

## E. Paired differences in mIoU (same objects; positive = first model better)

| Comparison | box | click |
|---|---|---|
| EfficientSAM-Ti − MobileSAM | +0.016 [+0.010, +0.022] | -0.035 [-0.052, -0.020] |
| EfficientSAM-S − MobileSAM | +0.029 [+0.023, +0.034] | +0.062 [+0.048, +0.076] |
| EfficientSAM-S − EfficientSAM-Ti | +0.013 [+0.008, +0.018] | +0.097 [+0.082, +0.111] |
| SAM ViT-B − EfficientSAM-S | +0.000 [-0.005, +0.004] | -0.052 [-0.065, -0.040] |
| SAM ViT-H − EfficientSAM-S | +0.016 [+0.010, +0.022] | -0.006 [-0.024, +0.009] |
| Ours: SAMI init − Ours: MAE init | -0.013 [-0.020, -0.006] | -0.001 [-0.008, +0.007] |
| Ours: SAMI init − Ours: random init | -0.017 [-0.023, -0.010] | -0.007 [-0.016, +0.001] |
| Ours: MAE init − Ours: random init | -0.004 [-0.011, +0.003] | -0.006 [-0.014, +0.002] |
| Ours: SAMI frozen − Ours: MAE frozen | +0.005 [-0.002, +0.013] | +0.031 [+0.019, +0.043] |
| Ours: SAMI frozen − Ours: random frozen | -0.016 [-0.023, -0.009] | +0.026 [+0.011, +0.040] |
| Ours: MAE frozen − Ours: random frozen | -0.022 [-0.029, -0.015] | -0.005 [-0.016, +0.007] |
| Ours: SAMI frozen − Ours: SAMI init | -0.009 [-0.017, -0.002] | -0.011 [-0.021, -0.001] |

## F. Efficiency (CUDA, 1 threads, PyTorch 2.14.1+cu130, batch size 1)

| Model | Params total (M) | Params encoder (M) | Encoder ms/image (median) | Decoder ms/prompt (median) | Peak process memory (MB) |
|---|---|---|---|---|---|
| EfficientSAM-Ti | 10.22 | 6.16 | 110 | 5 | 1563 |
| EfficientSAM-S | 26.41 | 22.35 | 276 | 5 | 1560 |
| MobileSAM | 10.13 | 6.07 | 61 | 5 | 1906 |
| SAM ViT-B | 93.74 | 89.67 | 485 | 5 | 1804 |
| SAM ViT-H | 641.09 | 637.03 | 2181 | 5 | 5562 |
| Ours: SAMI init | 10.22 | 6.16 | 13 | 3 | 1561 |
| Ours: MAE init | 10.22 | 6.16 | 13 | 3 | 1553 |
| Ours: random init | 10.22 | 6.16 | 13 | 3 | 1548 |
| Ours: SAMI frozen | 10.22 | 6.16 | 12 | 2 | 1551 |
| Ours: MAE frozen | 10.22 | 6.16 | 11 | 2 | 1551 |
| Ours: random frozen | 10.22 | 6.16 | 12 | 2 | 1550 |

Encoder time is one forward pass per image (first 3 images excluded as warm-up). Decoder time includes upsampling the masks to the original resolution. Peak memory is the whole Python process (model, image, masks).