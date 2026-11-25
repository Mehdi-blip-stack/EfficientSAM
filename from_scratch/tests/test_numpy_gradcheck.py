"""Finite-difference checks for NumPy layers, losses, and a complete TinySAM."""

import numpy as np

from tinysam.config import TinySAMConfig
from tinysam.numpy_impl.gradcheck import finite_difference, relative_error, sampled_indices
from tinysam.numpy_impl.layers import (
    GELU, LayerNorm, LayerNorm2d, Linear, MLP, MLP3, MultiHeadAttention, PatchEmbed,
    ReLU, UpsampleConvT2x, softmax, softmax_backward,
)
from tinysam.numpy_impl.losses import dice_loss, focal_loss, tiny_sam_loss
from tinysam.numpy_impl.model import TinySAM


RNG = np.random.default_rng(23)
DTYPE = np.float64


def assert_close(actual, expected, limit=1e-5):
    error = relative_error(actual, expected)
    assert np.max(error) < limit, f"max symmetric relative error={np.max(error):.3g}"


def assert_sampled_close(analytic, numeric, indices, limit=1e-5):
    selected = tuple(np.asarray(indices).T)
    assert_close(analytic[selected], numeric[selected], limit)


def owner_and_leaf(root, name):
    parts = name.split(".")
    owner = root
    for part in parts[:-1]:
        owner = owner[int(part)] if part.isdigit() else getattr(owner, part)
    return owner, parts[-1]


def check_module(module, x, seed=0, count=2):
    rng = np.random.default_rng(seed)
    x = x.astype(DTYPE)
    output = module.forward(x)
    dout = rng.normal(size=output.shape).astype(DTYPE)
    module.zero_grad()
    analytic_input = module.backward(dout)
    analytic_params = {name: owner_and_leaf(module, name)[0].grads[owner_and_leaf(module, name)[1]].copy()
                       for name, _ in module.named_parameters()}
    input_indices = sampled_indices(x.shape, count, seed)
    numeric_input = finite_difference(x, lambda: float(np.sum(module.forward(x) * dout)), indices=input_indices)
    assert_sampled_close(analytic_input, numeric_input, input_indices)
    for name, parameter in module.named_parameters():
        owner, leaf = owner_and_leaf(module, name)
        indices = sampled_indices(parameter.shape, count, seed + 1)
        numeric = finite_difference(parameter, lambda: float(np.sum(module.forward(x) * dout)), indices=indices)
        assert_sampled_close(analytic_params[name], numeric, indices)


def test_every_numpy_layer_input_and_parameter_gradients():
    check_module(Linear(3, 4, np.random.default_rng(1), DTYPE),
                 RNG.normal(size=(2, 3)), seed=1)
    check_module(LayerNorm(4, dtype=DTYPE), RNG.normal(size=(2, 3, 4)), seed=2)
    check_module(GELU(), RNG.normal(size=(3, 4)), seed=3)
    check_module(ReLU(), RNG.uniform(0.1, 1.2, size=(3, 4)), seed=4)
    check_module(MLP(4, 5, 3, np.random.default_rng(5), DTYPE), RNG.normal(size=(2, 4)), seed=5)
    check_module(MLP3(4, 5, 3, np.random.default_rng(51), DTYPE), RNG.normal(size=(2, 4)), seed=51)
    check_module(PatchEmbed(4, 2, 4, np.random.default_rng(6), DTYPE),
                 RNG.normal(size=(1, 3, 4, 4)), seed=6)
    check_module(UpsampleConvT2x(2, 2, np.random.default_rng(7), DTYPE),
                 RNG.normal(size=(1, 2, 2, 2)), seed=7)
    check_module(LayerNorm2d(3, dtype=DTYPE), RNG.normal(size=(1, 3, 2, 2)), seed=8)


def test_attention_and_softmax_gradients():
    attention = MultiHeadAttention(4, 2, np.random.default_rng(9), DTYPE)
    q = RNG.normal(size=(1, 2, 4)).astype(DTYPE)
    k = RNG.normal(size=(1, 3, 4)).astype(DTYPE)
    v = RNG.normal(size=(1, 3, 4)).astype(DTYPE)
    output = attention.forward(q, k, v)
    dout = RNG.normal(size=output.shape).astype(DTYPE)
    attention.zero_grad()
    dq, dk, dv = attention.backward(dout)
    for array, analytic in ((q, dq), (k, dk), (v, dv)):
        indices = sampled_indices(array.shape, 4, 10)
        numeric = finite_difference(array, lambda: float(np.sum(attention.forward(q, k, v) * dout)), indices=indices)
        assert_sampled_close(analytic, numeric, indices)
    for name, param in attention.named_parameters():
        owner, leaf = owner_and_leaf(attention, name)
        analytic = owner.grads[leaf].copy()
        indices = sampled_indices(param.shape, 2, 11)
        numeric = finite_difference(param, lambda: float(np.sum(attention.forward(q, k, v) * dout)), indices=indices)
        assert_sampled_close(analytic, numeric, indices)
    scores = RNG.normal(size=(2, 3)).astype(DTYPE)
    probabilities = softmax(scores)
    dp = RNG.normal(size=scores.shape).astype(DTYPE)
    numeric = finite_difference(scores, lambda: float(np.sum(softmax(scores) * dp)))
    assert_close(softmax_backward(probabilities, dp), numeric)


def test_losses_and_analytic_gradients():
    logits = RNG.normal(size=(3, 4)).astype(DTYPE)
    target = RNG.integers(0, 2, size=(3, 4)).astype(DTYPE)
    for loss_fn in (focal_loss, dice_loss):
        loss, analytic = loss_fn(logits, target)
        numeric = finite_difference(logits, lambda: loss_fn(logits, target)[0])
        assert_close(analytic, numeric)
    masks = RNG.normal(size=(2, 3, 4, 4)).astype(DTYPE)
    iou = RNG.normal(size=(2, 3)).astype(DTYPE)
    gt = RNG.integers(0, 2, size=(2, 4, 4)).astype(DTYPE)
    loss, dmask, diou = tiny_sam_loss(masks, iou, gt)
    mask_indices = sampled_indices(masks.shape, 12, 12)
    numeric_mask = finite_difference(masks, lambda: tiny_sam_loss(masks, iou, gt)[0],
                                     indices=mask_indices)
    numeric_iou = finite_difference(iou, lambda: tiny_sam_loss(masks, iou, gt)[0])
    assert_sampled_close(dmask, numeric_mask, mask_indices)
    assert_close(diou, numeric_iou)


def test_full_tinysam_total_loss_parameter_gradients():
    config = TinySAMConfig(img_size=16, patch=4, enc_dim=16, enc_depth=1, enc_heads=2,
                           emb_dim=16, dec_depth=1, dec_heads=2)
    model = TinySAM(config, seed=12, dtype=DTYPE)
    images = RNG.normal(size=(1, 3, 16, 16)).astype(DTYPE)
    points = np.asarray([[[5, 6], [0, 0], [0, 0]]], dtype=DTYPE)
    labels = np.asarray([[1, -1, -1]], dtype=np.int64)
    target = RNG.integers(0, 2, size=(1, 16, 16)).astype(DTYPE)
    gradcheck_scale = 1e-4

    def objective():
        masks, quality = model.forward(images, points, labels)
        target_low = (target.reshape(1, masks.shape[-1], 16 // masks.shape[-1],
                                     masks.shape[-1], 16 // masks.shape[-1]).mean(axis=(2, 4)) >= 0.5)
        return gradcheck_scale * tiny_sam_loss(masks, quality, target_low.astype(DTYPE))[0]

    masks, quality = model.forward(images, points, labels)
    out_size = masks.shape[-1]
    target_low = (target.reshape(1, out_size, 16 // out_size, out_size, 16 // out_size)
                  .mean(axis=(2, 4)) >= 0.5).astype(DTYPE)
    loss, dmask, diou = tiny_sam_loss(masks, quality, target_low)
    model.zero_grad()
    model.backward(gradcheck_scale * dmask, gradcheck_scale * diou)
    errors = []
    for index, (name, parameter) in enumerate(model.named_parameters()):
        owner, leaf = owner_and_leaf(model, name)
        analytic = owner.grads[leaf].copy()
        indices = sampled_indices(parameter.shape, 1, 100 + index)
        numeric = finite_difference(parameter, objective, indices=indices)
        selected = tuple(np.asarray(indices).T)
        errors.append((float(np.max(relative_error(analytic[selected], numeric[selected]))),
                       name, analytic[selected].copy(), numeric[selected].copy()))
    worst = max(errors, key=lambda item: item[0])
    assert worst[0] < 1e-5, f"full-model max symmetric relative error={worst[0]:.3g} for {worst[1]}: {worst[2]} vs {worst[3]}"
