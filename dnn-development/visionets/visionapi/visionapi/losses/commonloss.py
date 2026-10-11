import torch.nn as nn

from ._internalosses import AsymmetricLossOptimized, MCLoss


class CommonLoss(nn.Module):
    """Loss module that inherits from torch.nn.Module and is used as
    a wrapper for different losses (bce, asymetric, mcloss and distillation)"""

    def __init__(self, pw, distillation, hierarchy, loss):
        super(CommonLoss, self).__init__()
        self.pw = pw
        self.distillation = distillation
        self.hierarchy = hierarchy
        """init for CommonLoss

        Args:
        pw: Array of two positive weights to use on both loss components
        distillation: Wheter to apply distillation loss (only for deit-distilled) 
        hierarchy: Hierarchy matrix tensor to use in case of mcloss
        loss: Indicates the loss function to be applied (bce, asym, mc)
        """
        if loss == "bce":  # BCE loss
            self.classifier_lossfn = nn.BCEWithLogitsLoss(pos_weight=pw[0])
            self.distill_lossfn = nn.BCEWithLogitsLoss(pos_weight=pw[1])
        elif loss == "asym":  # Asymm loss
            self.classifier_lossfn = AsymmetricLossOptimized()
            self.distill_lossfn = AsymmetricLossOptimized()
        elif loss == "mc":  # MCLoss (only on classifier)
            self.classifier_lossfn = MCLoss(pos_weight=pw[0], hierarchy=hierarchy)
            self.distill_lossfn = nn.BCEWithLogitsLoss(
                pos_weight=pw[1]
            )  # Distillation learns from teacher with simple BCE
        else:
            raise ValueError(f"{loss} not supported as loss.")

    def forward(self, preds, precomputed, label):
        """Forward method for loss object

        Args:
        Preds: Output of the model
        Precomputed: Output of the teacher model on distillation loss
        label: Expected output of the model
        """
        if self.distillation == 0:
            loss = self.classifier_lossfn(preds, label)
        else:
            classifier_loss = self.classifier_lossfn(preds[0], label)
            distill_loss = self.distill_lossfn(preds[1], precomputed)
            loss = (classifier_loss + distill_loss) / 2
        return loss
