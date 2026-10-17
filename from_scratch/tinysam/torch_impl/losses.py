"""Focal/Dice mask losses and IoU-head regression for PyTorch autograd."""

import torch


def focal_loss(logits, target, alpha=0.25, gamma=2.0):
    """Mean sigmoid focal loss, composed from basic tensor operations."""
    probability = torch.sigmoid(logits)
    pt = target * probability + (1 - target) * (1 - probability)
    alpha_t = target * alpha + (1 - target) * (1 - alpha)
    bce = -(target * torch.log(probability.clamp_min(1e-12))
            + (1 - target) * torch.log((1 - probability).clamp_min(1e-12)))
    return (alpha_t * (1 - pt) ** gamma * bce).mean()


def dice_loss(logits, target, eps=1e-6):
    """Soft Dice loss independently for each leading index."""
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (1 - (2 * intersection + eps) / (denominator + eps)).mean()


def tiny_sam_loss(mask_logits, iou_pred, target, focal_weight=20.0):
    """Best-of-K mask loss per sample plus IoU regression MSE."""
    batch, masks, height, width = mask_logits.shape
    target = target[:, None].expand(batch, masks, height, width)
    per_mask = focal_weight * focal_loss_per_mask(mask_logits, target)
    per_mask = per_mask + dice_loss_per_mask(mask_logits, target)
    winners = per_mask.argmin(dim=1)
    mask_loss = per_mask.gather(1, winners[:, None]).mean()
    with torch.no_grad():
        binary = mask_logits > 0
        truth = target > 0.5
        intersection = (binary & truth).sum(dim=(-2, -1)).float()
        union = (binary | truth).sum(dim=(-2, -1)).float()
        iou_target = intersection / union.clamp_min(1)
    return mask_loss + ((iou_pred - iou_target) ** 2).mean()


def focal_loss_per_mask(logits, target, alpha=0.25, gamma=2.0):
    probability = torch.sigmoid(logits)
    pt = target * probability + (1 - target) * (1 - probability)
    alpha_t = target * alpha + (1 - target) * (1 - alpha)
    bce = -(target * torch.log(probability.clamp_min(1e-12))
            + (1 - target) * torch.log((1 - probability).clamp_min(1e-12)))
    return (alpha_t * (1 - pt) ** gamma * bce).mean(dim=(-2, -1))


def dice_loss_per_mask(logits, target, eps=1e-6):
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return 1 - (2 * intersection + eps) / (denominator + eps)
