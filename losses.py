import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedSoftIoULoss(nn.Module):
    def __init__(self, obj_weight=4.0, eps=1e-6):
        """
        Weighted soft IoU loss for binary segmentation with class imbalance.

        Args:
            obj_weight (float): How much more important the object IoU is compared to background IoU.
            eps (float): Small constant to prevent division by zero.
        """
        super().__init__()
        self.obj_weight = obj_weight
        self.eps = eps

    def forward(self, preds, targets):
        """
        Args:
            preds (Tensor): Sigmoid probabilities of shape (N, 1, H, W)
            targets (Tensor): Binary ground truth of shape (N, 1, H, W)
        Returns:
            loss (Tensor): Scalar weighted IoU loss
        """
        # Flatten for per-image IoU computation
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # Binary masks
        obj_mask = targets == 1
        bkg_mask = targets == 0

        # Soft intersection and union
        inter_obj = (preds * targets * obj_mask).sum(dim=1)
        union_obj = ((preds + targets) * obj_mask).sum(dim=1) - inter_obj

        inter_bkg = ((1 - preds) * (1 - targets) * bkg_mask).sum(dim=1)
        union_bkg = (((1 - preds) + (1 - targets)) * bkg_mask).sum(dim=1) - inter_bkg

        iou_obj = (inter_obj + self.eps) / (union_obj + self.eps)
        iou_bkg = (inter_bkg + self.eps) / (union_bkg + self.eps)

        # Weighted inverse mean IoU
        weighted_iou = (self.obj_weight * iou_obj + iou_bkg) / (self.obj_weight + 1)
        loss = 1 - weighted_iou.mean()

        return loss

class WeightedBCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=4.0, smooth=1e-6):
        """
        pos_weight: float, multiplier for positive class (object) pixels in BCE.
                    For example, if object pixels are rare, use >1 to emphasize.
        smooth: small constant to avoid division by zero in Dice loss.
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: predicted raw outputs from model (no sigmoid applied), shape (N, 1, H, W)
        targets: ground truth masks, same shape, values 0 or 1
        """

        # Weighted BCE loss
        # pos_weight is a tensor with shape [1] or broadcastable to logits shape
        pos_weight_tensor = torch.tensor(self.pos_weight).to(logits.device)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight_tensor)

        # Dice loss
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice_score

        return bce_loss + dice_loss

class ProbWeightedBCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=4.0, smooth=1e-6):
        """
        BCE + Dice Loss for models that output probabilities (sigmoid already applied).
        Emulates pos_weight manually.
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.smooth = smooth

    def forward(self, probs, targets):
        """
        Args:
            probs: model output after sigmoid, shape (N, 1, H, W), values in [0, 1]
            targets: ground truth binary masks, same shape, values 0 or 1
        """
        # --- 1. Weighted BCE ---
        # Manual weighting per pixel
        weights = torch.where(targets == 1, self.pos_weight, 1.0).to(probs.device) # type: ignore
        bce_loss = F.binary_cross_entropy(probs, targets, weight=weights)

        # --- 2. Dice Loss ---
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()

        dice_score = (2 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1 - dice_score

        return bce_loss + dice_loss