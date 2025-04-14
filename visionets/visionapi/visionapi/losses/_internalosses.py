import torch
import torch.nn as nn


class CustomWeightedBCELoss(torch.nn.Module):
    """BCELoss class on sigmoid activations with pos_weight
    Used when hierarchy+class_weights
    """

    def __init__(self, pos_weight):
        super(CustomWeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.fn = CustomWeightedBCELossFunct.apply

    def forward(self, outputs, label):
        return self.fn(outputs, label, self.pos_weight)


class CustomWeightedBCELossFunct(torch.autograd.Function):
    """BCELoss autograd function on sigmoid activations
    with pos_weight. Used when hierarchy+class_weights
    """

    @staticmethod
    def forward(ctx, input, target, pos_weight):
        ctx.save_for_backward(input, target, pos_weight)
        output = (target - 1) * torch.maximum(
            torch.log1p(-input), input.new_full((), -100)
        ) - target * pos_weight * torch.maximum(
            torch.log(input), input.new_full((), -100)
        )
        return torch.mean(output)

    @staticmethod
    def backward(ctx, grad_output):
        input, target, pos_weight = ctx.saved_tensors
        EPSILON = 1e-12
        result = (
            grad_output
            * ((1 - target) * input - target * pos_weight * (1 - input))
            / torch.clamp(input * (1 - input), min=EPSILON)
        )
        return result / input.numel(), None, None


class MCLoss(nn.Module):
    """MCloss class that applies hierarchy restrictions on training"""

    def __init__(self, pos_weight, hierarchy):
        super(MCLoss, self).__init__()
        self.lossfn = CustomWeightedBCELoss(pos_weight)
        self.hierarchy = hierarchy

    def get_constr_out(self, outputs):
        """Applies restrictions to the model output"""
        # Read label shape (number of labels)
        num_label = self.hierarchy.shape[1]
        # Convert to double and unsqueeze
        c_out = outputs.double()
        c_out = c_out.unsqueeze(1)
        # Add extra dimension to have a matrix
        c_out = c_out.expand(outputs.shape[0], num_label, num_label)
        # Add extra batch dimension to hierarchy
        R_batch = self.hierarchy.expand(outputs.shape[0], num_label, num_label)
        # Compute final out
        final_out, _ = torch.max(R_batch * c_out.double(), dim=2)
        return final_out

    def forward(self, outputs, label):
        """Applies hierarchy loss logic to enforce hierarchy relationships"""
        outputs = torch.sigmoid(outputs)

        # Compute constraint output
        constr_output = self.get_constr_out(outputs)
        # Compute MCLoss
        train_output = label * outputs.double()
        train_output = self.get_constr_out(train_output)
        train_output = (1 - label) * constr_output.double() + label * train_output
        loss = self.lossfn(train_output, label)

        return loss


class AsymmetricLossOptimized(nn.Module):
    """External asymetricloss implementation"""

    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=False,
    ):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = (
            self.asymmetric_w
        ) = self.loss = None

    def forward(self, x, y):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Binarize y
        bin_y = torch.round(y).int()

        self.targets = bin_y
        self.anti_targets = 1 - bin_y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(
                1 - self.xs_pos - self.xs_neg,
                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets,
            )
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()
