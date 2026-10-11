import torch
from torch import nn

from ._dlgetters import get_fullmodel


class ChestModel(nn.Module):
    """RELIANCE Model and methods definitions"""

    def __init__(
        self,
        pre="convfixed",
        base_model_name="densenet121",
        input_size=-1,
        layers=0,
        neurons=0,
        norm_type="none",
        drop_type="none",
        drop_rate=0.1,
        ptdrop_rate=0.0,
        activation="relu",
        num_label=32,
        hierarchy="none",
    ):
        """Initilization function for reliance model

        Args:
        pre: Preprocessing technique applied (convrand, convfixed, resize)
        base_model_name: Name of the pretrained model on timm library
        input_size: Desired input size of the model (-1 selects input size of timm model cfg)
        layers: Number of hidden classifier layers
        neurons: Number of neurons of the first hidden classifier layer
        norm_type: Normalization type in classifier (None for no normalization applied)
        drop_type: Dropout type in classifier (None for no dropout applied)
        drop_rate: Dropout rate in classifier
        ptdrop_rate: Patch_dropout rate applied on input (Available only on vit and deit)
        activation: Activation applied on hidden classifier layers (gelu, relu, prelu, selu)
        num_label: Number of labels of the final classifier layer
        hierarchy: Hierarchy matrix in case of mcloss ("none" for ignoring)
        """

        # Initialize
        super().__init__()
        # Assign hierarchy
        self.hierarchy = hierarchy
        self.num_label = num_label
        # Get model, preprocess andkwargs
        self.model, self.preprocess, self.kwargs = get_fullmodel(
            pre,
            base_model_name,
            input_size,
            layers,
            neurons,
            norm_type,
            drop_type,
            drop_rate,
            ptdrop_rate,
            activation,
            num_label,
            hierarchy,
        )

    def set_hierarchy_device(self, device):
        """Sets the device of the hierarchy matrix if exists"""
        if self.hierarchy != "none":
            self.hierarchy = self.hierarchy.to(device)

    def get_classifier_parameters(self):
        """Returns parameters of the classifier"""
        classifier = self.model.get_classifier()
        if isinstance(classifier, tuple):
            return list(classifier[0].parameters()) + list(classifier[1].parameters())
        else:
            return classifier.parameters()

    def apply_hierarchy(self, outputs):
        """Function that applies hierarchy restrictions to classifier output
        Used mainly in mcloss training, but can be used in normal training as
        postprocessing technique
        """
        # Convert to double and unsqueeze
        c_out = outputs.double()
        c_out = c_out.unsqueeze(1)
        # Add extra dimension to have a matrix
        c_out = c_out.expand(outputs.shape[0], self.num_label, self.num_label)
        # Add extra batch dimension to hierarchy
        R_batch = self.hierarchy.expand(
            outputs.shape[0], self.num_label, self.num_label
        )
        # Compute final out
        final_out, _ = torch.max(R_batch * c_out.double(), dim=2)
        return final_out

    def process_logits(self, logits):
        """Function that process logits using sigmoid, apply_hierarchy if
        needed and combines the output of double classifier in distillation
        """
        # Mean in case of tuple
        if isinstance(logits, tuple):
            logits = (logits[0] + logits[1]) / 2

        # Apply sigmoid
        preds = torch.sigmoid(logits)

        # Apply hierarchy restriction if present
        if self.hierarchy != "none":
            preds = self.apply_hierarchy(preds)

        return preds

    def forward(self, x):
        preprocessed = self.preprocess(x)
        logits = self.model(preprocessed)
        if self.training:
            return logits
        else:
            return self.process_logits(logits)


def load_model(model_path) -> ChestModel:
    """Function that loads model into cpu

    Args:
    model_path: Path where model is saved
    """
    device = torch.device("cpu")
    kwargs, state = torch.load(model_path, map_location=device)
    model = ChestModel(**kwargs)
    model.load_state_dict(state)
    model.set_hierarchy_device(device)
    return model


def freeze_extractor(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.module.get_classifier_parameters():
        param.requires_grad = True


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True
