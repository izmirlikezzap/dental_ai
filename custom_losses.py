#!/usr/bin/env python3
"""
Custom Loss Functions for YOLO Detection

In-place component swaps on the model's existing criterion.
Works with any detection head (v8Detect, v10Detect, YOLO26, etc.)
by modifying .bce or .bbox_loss attributes rather than replacing
the entire criterion class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist


# ---------------------------------------------------------------------------
# Focal Loss (replaces .bce inside any DetectionLoss)
# ---------------------------------------------------------------------------

class FocalBCEWithLogitsLoss(nn.Module):
    """
    Drop-in replacement for nn.BCEWithLogitsLoss(reduction='none')
    that applies focal weighting: (1 - p_t)^gamma * BCE.
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


# ---------------------------------------------------------------------------
# DIoU Loss (replaces .bbox_loss inside any DetectionLoss)
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


# ---------------------------------------------------------------------------
# Weighted Loss (per-class weights applied to .bce)
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
        weights = self.class_weights.to(inputs.device)
        if inputs.dim() >= 2 and inputs.shape[-1] == weights.shape[0]:
            bce = bce * weights
        return bce


# ---------------------------------------------------------------------------
# In-place injection (modifies existing criterion, doesn't replace it)
# ---------------------------------------------------------------------------

def inject_custom_loss(criterion, loss_method: str, loss_params: dict | None = None):
    """
    Modify the model's existing criterion in-place by swapping components.

    This works with ANY detection loss class (v8DetectionLoss, v10DetectionLoss,
    YOLO26 loss, etc.) as long as it has .bce and/or .bbox_loss attributes.

    Args:
        criterion: The model's existing criterion (trainer.model.criterion)
        loss_method: 'default', 'focal', 'diou', 'weighted'
        loss_params: Extra parameters for the loss
    """
    if loss_params is None:
        loss_params = {}

    if loss_method == "default":
        print(f"[LOSS] Using model's native loss (no modification)")
        return

    elif loss_method == "focal":
        if hasattr(criterion, "bce"):
            alpha = loss_params.get("alpha", 0.25)
            gamma = loss_params.get("gamma", 2.0)
            criterion.bce = FocalBCEWithLogitsLoss(alpha=alpha, gamma=gamma)
            print(f"[LOSS] Swapped criterion.bce → FocalBCE (alpha={alpha}, gamma={gamma})")
        else:
            print(f"[LOSS WARNING] criterion has no 'bce' attribute, skipping focal injection")

    elif loss_method == "diou":
        if hasattr(criterion, "bbox_loss"):
            old_bbox = criterion.bbox_loss
            reg_max = old_bbox.dfl_loss.reg_max if old_bbox.dfl_loss else 16
            device = next(old_bbox.parameters()).device if list(old_bbox.parameters()) else "cpu"
            criterion.bbox_loss = DIoUBboxLoss(reg_max).to(device)
            print(f"[LOSS] Swapped criterion.bbox_loss → DIoUBboxLoss")
        else:
            print(f"[LOSS WARNING] criterion has no 'bbox_loss' attribute, skipping DIoU injection")

    elif loss_method == "weighted":
        if hasattr(criterion, "bce"):
            class_weights = loss_params.get("class_weights", [1.0, 2.5])
            criterion.bce = WeightedBCEWithLogitsLoss(class_weights=class_weights)
            print(f"[LOSS] Swapped criterion.bce → WeightedBCE (weights={class_weights})")
        else:
            print(f"[LOSS WARNING] criterion has no 'bce' attribute, skipping weighted injection")

    else:
        raise ValueError(f"Unknown loss method: {loss_method}")
