"""Focal, soft Dice, and IoU-regression losses with analytic gradients."""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def focal_loss(logits, target, alpha=0.25, gamma=2.0):
    """Mean sigmoid focal loss and its derivative with respect to logits."""
    p = sigmoid(logits)
    target = np.asarray(target, dtype=logits.dtype)
    pt = target * p + (1 - target) * (1 - p)
    at = target * alpha + (1 - target) * (1 - alpha)
    bce = -(target * np.log(np.maximum(p, 1e-12)) + (1-target) * np.log(np.maximum(1-p, 1e-12)))
    loss = at * (1 - pt) ** gamma * bce
    dpt = (2 * target - 1) * p * (1-p)
    grad = at * ((1-pt) ** gamma * (p-target) - gamma * (1-pt) ** (gamma-1) * bce * dpt)
    return float(loss.mean()), grad / loss.size


def dice_loss(logits, target, eps=1e-6):
    """Soft Dice loss over the final two dimensions and its logit gradient."""
    p = sigmoid(logits)
    target = np.asarray(target, dtype=logits.dtype)
    axes = (-2, -1)
    intersection = np.sum(p * target, axis=axes, keepdims=True)
    p_sum = np.sum(p, axis=axes, keepdims=True)
    y_sum = np.sum(target, axis=axes, keepdims=True)
    numerator = 2 * intersection + eps
    denominator = p_sum + y_sum + eps
    loss = 1 - numerator / denominator
    dloss_dp = -(2 * target * denominator - numerator) / (denominator ** 2)
    grad = dloss_dp * p * (1-p)
    leading_items = max(1, int(np.prod(logits.shape[:-2])))
    return float(loss.mean()), grad / leading_items


def tiny_sam_loss(mask_logits, iou_pred, target, focal_weight=20.0):
    """Per-sample best-of-K mask loss plus IoU-head MSE; returns all gradients."""
    batch, masks, height, width = mask_logits.shape
    target = np.asarray(target, dtype=mask_logits.dtype)
    if target.shape != (batch, height, width):
        raise ValueError("target must be downsampled to the mask-logit resolution")
    all_losses = np.zeros((batch, masks), dtype=mask_logits.dtype)
    all_mask_grads = np.zeros_like(mask_logits)
    for b in range(batch):
        for k in range(masks):
            focal, gf = focal_loss(mask_logits[b, k], target[b])
            dice, gd = dice_loss(mask_logits[b, k], target[b])
            all_losses[b, k] = focal_weight * focal + dice
            all_mask_grads[b, k] = focal_weight * gf + gd
    winners = np.argmin(all_losses, axis=1)
    dmask = np.zeros_like(mask_logits)
    for b, k in enumerate(winners):
        dmask[b, k] = all_mask_grads[b, k] / batch
    binary = mask_logits > 0
    intersection = (binary & (target[:, None] > 0.5)).sum(axis=(-2, -1))
    union = (binary | (target[:, None] > 0.5)).sum(axis=(-2, -1))
    iou_target = intersection / np.maximum(union, 1)
    error = iou_pred - iou_target
    iou_loss = float(np.mean(error ** 2))
    diou = 2 * error / (batch * masks)
    return float(all_losses[np.arange(batch), winners].mean() + iou_loss), dmask, diou
