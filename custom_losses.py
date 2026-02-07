#!/usr/bin/env python3
"""
Custom Loss Functions for YOLO Detection

Proper v8DetectionLoss subclasses that inject into the YOLO training pipeline.
Implements: Focal Loss (BCE replacement), DIoU Loss (BboxLoss replacement),
            Weighted Loss (per-class BCE weights).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist


# ---------------------------------------------------------------------------
# Focal Loss (replaces nn.BCEWithLogitsLoss inside v8DetectionLoss)
# ---------------------------------------------------------------------------

class FocalBCEWithLogitsLoss(nn.Module):
    """
    Drop-in replacement for nn.BCEWithLogitsLoss(reduction='none')
    that applies focal weighting: (1 - p_t)^gamma * BCE.

    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

        return focal_weight * bce


class FocalDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss with BCE replaced by Focal BCE."""

    def __init__(self, model, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__(model)
        self.bce = FocalBCEWithLogitsLoss(alpha=alpha, gamma=gamma)


# ---------------------------------------------------------------------------
# DIoU Loss (replaces BboxLoss inside v8DetectionLoss)
# ---------------------------------------------------------------------------

class DIoUBboxLoss(BboxLoss):
    """BboxLoss that uses DIoU instead of CIoU for bounding-box regression."""

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, DIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = (
                self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask])
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none")
                .mean(-1, keepdim=True)
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class DIoUDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss with BboxLoss replaced by DIoU variant."""

    def __init__(self, model):
        super().__init__(model)
        m = model.model[-1]
        self.bbox_loss = DIoUBboxLoss(m.reg_max).to(self.device)


# ---------------------------------------------------------------------------
# Weighted Loss (per-class weights applied to BCE)
# ---------------------------------------------------------------------------

class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Drop-in replacement for nn.BCEWithLogitsLoss(reduction='none')
    that applies per-class weights to the binary cross-entropy.
    """

    def __init__(self, class_weights: list[float] | None = None):
        super().__init__()
        if class_weights is None:
            class_weights = [1.0, 2.5]
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        # class_weights shape: (C,) → broadcast over (B, C) or (N, C)
        weights = self.class_weights.to(inputs.device)
        if inputs.dim() >= 2 and inputs.shape[-1] == weights.shape[0]:
            bce = bce * weights
        return bce


class WeightedDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss with BCE replaced by per-class weighted BCE."""

    def __init__(self, model, class_weights: list[float] | None = None):
        super().__init__(model)
        self.bce = WeightedBCEWithLogitsLoss(class_weights=class_weights)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_detection_loss(loss_method: str, model, loss_params: dict | None = None):
    """
    Create a v8DetectionLoss (or subclass) for the given method.

    Args:
        loss_method: 'default', 'focal', 'diou', 'weighted'
        model:       The YOLO model (needed by v8DetectionLoss.__init__)
        loss_params: Extra parameters forwarded to the loss constructor

    Returns:
        A v8DetectionLoss instance ready to assign to model.criterion
    """
    if loss_params is None:
        loss_params = {}

    if loss_method == "default":
        return v8DetectionLoss(model)

    elif loss_method == "focal":
        alpha = loss_params.get("alpha", 0.25)
        gamma = loss_params.get("gamma", 2.0)
        return FocalDetectionLoss(model, alpha=alpha, gamma=gamma)

    elif loss_method == "diou":
        return DIoUDetectionLoss(model)

    elif loss_method == "weighted":
        class_weights = loss_params.get("class_weights", [1.0, 2.0])
        return WeightedDetectionLoss(model, class_weights=class_weights)

    else:
        raise ValueError(f"Unknown loss method: {loss_method}")
