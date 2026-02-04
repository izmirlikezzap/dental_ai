#!/usr/bin/env python3
"""
Custom Loss Functions for YOLO Detection
Implements: Focal Loss, DIoU/CIoU Loss, Weighted Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.tal import bbox2dist
from typing import Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for classification
    Focuses training on hard examples

    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor in [0, 1] for class balance
            gamma: Focusing parameter >= 0 (gamma=0 is equivalent to CE)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (B, C) logits before sigmoid
            targets: Ground truth labels (B,) or (B, C) one-hot

        Returns:
            Focal loss value
        """
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(inputs)

        # Convert targets to one-hot if needed
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=inputs.size(-1)).float()

        # Calculate focal loss
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DIoULoss(nn.Module):
    """
    Distance-IoU Loss for bounding box regression
    Considers center distance in addition to IoU

    Reference: https://arxiv.org/abs/1911.08287
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_boxes: Predicted boxes (N, 4) in xyxy format
            target_boxes: Target boxes (N, 4) in xyxy format

        Returns:
            DIoU loss
        """
        # Calculate IoU
        iou = self._calculate_iou(pred_boxes, target_boxes)

        # Calculate center distance
        pred_center = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
        target_center = (target_boxes[:, :2] + target_boxes[:, 2:]) / 2
        center_distance = torch.sum((pred_center - target_center) ** 2, dim=1)

        # Calculate diagonal length of smallest enclosing box
        enclose_x1y1 = torch.min(pred_boxes[:, :2], target_boxes[:, :2])
        enclose_x2y2 = torch.max(pred_boxes[:, 2:], target_boxes[:, 2:])
        enclose_wh = enclose_x2y2 - enclose_x1y1
        diagonal_distance = torch.sum(enclose_wh ** 2, dim=1) + self.eps

        # DIoU = IoU - (center_distance^2 / diagonal_distance^2)
        diou = iou - (center_distance / diagonal_distance)

        # Loss is 1 - DIoU
        loss = 1 - diou

        return loss.mean()

    def _calculate_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between two sets of boxes"""
        # Intersection area
        inter_x1y1 = torch.max(boxes1[:, :2], boxes2[:, :2])
        inter_x2y2 = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        inter_wh = (inter_x2y2 - inter_x1y1).clamp(min=0)
        inter_area = inter_wh[:, 0] * inter_wh[:, 1]

        # Union area
        boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union_area = boxes1_area + boxes2_area - inter_area + self.eps

        # IoU
        iou = inter_area / union_area

        return iou


class CIoULoss(nn.Module):
    """
    Complete-IoU Loss for bounding box regression
    Considers IoU, center distance, and aspect ratio

    Reference: https://arxiv.org/abs/1911.08287
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_boxes: Predicted boxes (N, 4) in xyxy format
            target_boxes: Target boxes (N, 4) in xyxy format

        Returns:
            CIoU loss
        """
        # Calculate IoU
        iou = self._calculate_iou(pred_boxes, target_boxes)

        # Calculate center distance
        pred_center = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
        target_center = (target_boxes[:, :2] + target_boxes[:, 2:]) / 2
        center_distance = torch.sum((pred_center - target_center) ** 2, dim=1)

        # Calculate diagonal length of smallest enclosing box
        enclose_x1y1 = torch.min(pred_boxes[:, :2], target_boxes[:, :2])
        enclose_x2y2 = torch.max(pred_boxes[:, 2:], target_boxes[:, 2:])
        enclose_wh = enclose_x2y2 - enclose_x1y1
        diagonal_distance = torch.sum(enclose_wh ** 2, dim=1) + self.eps

        # Calculate aspect ratio consistency
        pred_wh = pred_boxes[:, 2:] - pred_boxes[:, :2]
        target_wh = target_boxes[:, 2:] - target_boxes[:, :2]

        v = (4 / (torch.pi ** 2)) * torch.pow(
            torch.atan(target_wh[:, 0] / (target_wh[:, 1] + self.eps)) -
            torch.atan(pred_wh[:, 0] / (pred_wh[:, 1] + self.eps)), 2
        )

        with torch.no_grad():
            alpha = v / (1 - iou + v + self.eps)

        # CIoU = IoU - (center_distance^2 / diagonal_distance^2) - alpha * v
        ciou = iou - (center_distance / diagonal_distance) - alpha * v

        # Loss is 1 - CIoU
        loss = 1 - ciou

        return loss.mean()

    def _calculate_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between two sets of boxes"""
        # Intersection area
        inter_x1y1 = torch.max(boxes1[:, :2], boxes2[:, :2])
        inter_x2y2 = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        inter_wh = (inter_x2y2 - inter_x1y1).clamp(min=0)
        inter_area = inter_wh[:, 0] * inter_wh[:, 1]

        # Union area
        boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union_area = boxes1_area + boxes2_area - inter_area + self.eps

        # IoU
        iou = inter_area / union_area

        return iou


class WeightedDetectionLoss(nn.Module):
    """
    Weighted loss for handling class imbalance in object detection
    Applies class-specific weights to classification loss
    """

    def __init__(self, class_weights: list = None, num_classes: int = 3):
        """
        Args:
            class_weights: List of weights for each class [healthy, meziodens, supernumere]
            num_classes: Number of classes
        """
        super().__init__()

        if class_weights is None:
            # Default weights: higher weight for rare classes
            class_weights = [1.0, 2.0, 2.0]

        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.num_classes = num_classes

    def forward(self, pred_cls: torch.Tensor, target_cls: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_cls: Predicted class scores (B, C)
            target_cls: Target class labels (B,)

        Returns:
            Weighted classification loss
        """
        # Move weights to same device as predictions
        weights = self.class_weights.to(pred_cls.device)

        # Calculate weighted cross entropy
        loss = F.cross_entropy(pred_cls, target_cls, weight=weights, reduction='mean')

        return loss


def get_loss_function(loss_method: str, loss_params: dict = None):
    """
    Factory function to get the appropriate loss function

    Args:
        loss_method: 'default', 'focal', 'diou', 'ciou', 'weighted'
        loss_params: Dictionary of loss-specific parameters

    Returns:
        Loss function object or None (for default YOLO loss)
    """
    if loss_params is None:
        loss_params = {}

    if loss_method == 'default':
        # Use YOLO's built-in loss (return None)
        return None

    elif loss_method == 'focal':
        alpha = loss_params.get('alpha', 0.25)
        gamma = loss_params.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)

    elif loss_method == 'diou':
        return DIoULoss()

    elif loss_method == 'ciou':
        return CIoULoss()

    elif loss_method == 'weighted':
        class_weights = loss_params.get('class_weights', [1.0, 2.0, 2.0])
        num_classes = loss_params.get('num_classes', 3)
        return WeightedDetectionLoss(class_weights=class_weights, num_classes=num_classes)

    else:
        raise ValueError(f"Unknown loss method: {loss_method}")


# Example usage and testing
if __name__ == "__main__":
    # Test Focal Loss
    print("Testing Focal Loss...")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    pred = torch.randn(10, 3)  # 10 samples, 3 classes
    target = torch.randint(0, 3, (10,))  # 10 labels
    loss = focal_loss(pred, target)
    print(f"Focal Loss: {loss.item():.4f}")

    # Test DIoU Loss
    print("\nTesting DIoU Loss...")
    diou_loss = DIoULoss()
    pred_boxes = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32)
    target_boxes = torch.tensor([[15, 15, 55, 55], [25, 25, 65, 65]], dtype=torch.float32)
    loss = diou_loss(pred_boxes, target_boxes)
    print(f"DIoU Loss: {loss.item():.4f}")

    # Test CIoU Loss
    print("\nTesting CIoU Loss...")
    ciou_loss = CIoULoss()
    loss = ciou_loss(pred_boxes, target_boxes)
    print(f"CIoU Loss: {loss.item():.4f}")

    # Test Weighted Loss
    print("\nTesting Weighted Loss...")
    weighted_loss = WeightedDetectionLoss(class_weights=[1.0, 2.0, 2.0])
    pred = torch.randn(10, 3)
    target = torch.randint(0, 3, (10,))
    loss = weighted_loss(pred, target)
    print(f"Weighted Loss: {loss.item():.4f}")

    print("\nAll loss functions tested successfully!")
