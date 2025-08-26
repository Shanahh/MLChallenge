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
    def __init__(self, pos_weight=1.0, smooth=1e-6, dice_weight=1.0):
        """
        takes raw logits as input
        pos_weight: float, multiplier for positive class pixels in BCE.
        smooth: float, smoothing constant for Dice.
        dice_weight: float, weight factor for the Dice loss term (default 1.0).
        """
        super().__init__()
        self.smooth = smooth
        self.dice_weight = dice_weight
        # Store pos_weight as a buffer so it moves with .to(device)
        self.register_buffer("pos_weight_tensor", torch.tensor(pos_weight))

    def forward(self, logits, targets):
        """
        logits: raw outputs from the model, shape (N, 1, H, W)
        targets: ground truth masks, same shape, values {0,1}
        """
        # Binary cross-entropy with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight_tensor
        )

        # Dice loss (per image, averaged across batch)
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(1)
        dice_score = (2. * intersection + self.smooth) / (
            probs_flat.sum(1) + targets_flat.sum(1) + self.smooth
        )
        dice_loss = 1. - dice_score.mean()

        return bce_loss + self.dice_weight * dice_loss

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    gt_sorted: ground truth sorted by prediction errors
    """
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if gt_sorted.numel() > 1:  # if not a single pixel
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard

def lovasz_hinge_flat(logits, targets):
    """
    Binary Lovasz hinge loss
    logits: [P] variable, logits at each prediction (flattened)
    targets: [P] binary ground truth labels (flattened)
    """
    if len(targets) == 0:
        return logits.sum() * 0.
    signs = 2. * targets.float() - 1.
    errors = 1. - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    targets_sorted = targets[perm]
    grad = lovasz_grad(targets_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss

def flatten_binary_scores(scores, labels):
    """
    Flattens predictions and labels, removes ignore label (-1)
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    valid = labels != -1
    return scores[valid], labels[valid]

class BCELovaszLoss(nn.Module):
    def __init__(self, pos_weight=1.0, lovasz_weight=1.0):
        """
        BCE + Lovasz hinge loss
        pos_weight: positive pixel weighting for BCE
        lovasz_weight: weight factor for Lovasz term
        """
        super().__init__()
        self.register_buffer("pos_weight_tensor", torch.tensor(pos_weight))
        self.lovasz_weight = lovasz_weight

    def forward(self, logits, targets):
        # BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight_tensor
        )

        # Lovasz hinge (direct IoU optimization)
        logits_flat, targets_flat = flatten_binary_scores(logits.squeeze(1), targets.squeeze(1))
        lovasz_loss = lovasz_hinge_flat(logits_flat, targets_flat)

        return bce_loss + self.lovasz_weight * lovasz_loss

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


class WeightedBCEJaccardLoss(nn.Module):
    def __init__(self, pos_weight=1.0, smooth=1e-6, jaccard_weight=1.0):
        """
        pos_weight: emphasis on positive pixels in BCE
        smooth: for numerical stability in Jaccard
        jaccard_weight: balance between BCE and Jaccard
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.smooth = smooth
        self.jaccard_weight = jaccard_weight

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        # BCE
        pos_weight_tensor = torch.tensor(self.pos_weight).to(logits.device)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight_tensor
        )

        # Jaccard (IoU) Loss
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum() - intersection
        jaccard_score = (intersection + self.smooth) / (union + self.smooth)
        jaccard_loss = 1 - jaccard_score

        return bce_loss + self.jaccard_weight * jaccard_loss

def estimate_pos_loss_weight(data_loader):
    num_pos = 0
    num_neg = 0
    for _, masks in data_loader:
        num_pos = masks.float().sum().item()
        num_neg = (1 - masks).float().sum().item()

    pos_weight = num_neg / num_pos
    return pos_weight