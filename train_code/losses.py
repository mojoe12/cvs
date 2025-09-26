import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        """
        logits: Tensor of shape (B, num_labels)
        targets: Tensor of shape (B, num_labels), with 0s and 1s
        """
        # Flatten to (B*T, C)
        dim = logits.dim()
        if dim == 3:
            B, T = logits.size(0), logits.size(1)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1, targets.size(-1))

        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        pt = probs * targets + (1 - probs) * (1 - targets)  # pt = p_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.view(B, T, -1) if dim == 3 else focal_loss

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8):
        """
        gamma_pos: Focusing parameter for positive samples
        gamma_neg: Focusing parameter for negative samples
        clip: Optional clipping of negative prediction probabilities (to avoid overconfidence)
        eps: Numerical stability
        reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        """
        logits: raw predictions [batch_size, num_classes]
        targets: binary labels [batch_size, num_classes]
        """
        # Flatten to (B*T, C)
        dim = logits.dim()
        if dim == 3:
            B, T = logits.size(0), logits.size(1)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1, targets.size(-1))

        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1 - probs

        # Optional clip on negative probs to prevent over-confidence
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)

        # Compute log-likelihood
        log_pos = torch.log(probs_pos.clamp(min=self.eps))
        log_neg = torch.log(probs_neg.clamp(min=self.eps))

        # Asymmetric focusing
        loss_pos = targets * (1 - probs_pos) ** self.gamma_pos * log_pos
        loss_neg = (1 - targets) * probs_pos ** self.gamma_neg * log_neg

        loss = - (loss_pos + loss_neg)
        return loss.view(B, T, -1) if dim == 3 else loss
