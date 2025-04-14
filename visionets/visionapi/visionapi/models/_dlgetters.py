import torch
from torch import nn
from torch.nn import GELU, PReLU, ReLU, SELU
from torch.nn import Dropout, AlphaDropout
from torch.nn import BatchNorm1d, LayerNorm
from torch.nn.init import constant_, ones_, zeros_
from torch.nn import Conv2d, Linear, Sequential
from torchvision.transforms import Normalize, Resize
from timm import create_model


TIMM_IDS = ["densenet169", "deit_base_distilled_patch16_384", "regnety_160"]


class ChestModelPreprocess(nn.Module):
    """Class that implements the preprocessor for the ChestModel"""

    def __init__(self, input_size, mean, std):
        """
        Args:
        input_size: Final input size the model requires
        mean: Mean to normalize images
        std: Std to normalize images"""
        super().__init__()

        self.resize = Resize(size=(input_size, input_size))
        self.norm = Normalize(mean, std)

    def forward(self, input):

        out = self.resize(input)
        out = torch.cat([out, out, out], dim=1)
        out = self.norm(out)
        return out


def _rgetattr(obj, attr, default=None):
    """Recursive getter"""
    try:
        left, right = attr.split(".", 1)
    except:
        return getattr(obj, attr, default)
    return _rgetattr(getattr(obj, left), right, default)


def _rsetattr(obj, attr, val):
    """Recursive getattr"""
    try:
        left, right = attr.split(".", 1)
    except:
        return setattr(obj, attr, val)
    return _rsetattr(getattr(obj, left), right, val)


def get_activation(activation="relu"):
    """Function to select activation"""
    if activation == "gelu":
        return GELU
    elif activation == "prelu":
        return PReLU
    elif activation == "relu":
        return ReLU
    elif activation == "selu":
        return SELU
    else:
        return None


def get_dropout(drop_type="drop"):
    """Function to select dropout"""
    if drop_type == "drop":
        return Dropout
    elif drop_type == "alpha":
        return AlphaDropout
    else:
        return None


def get_norm(norm_type="bn"):
    """Function to select normalization"""
    if norm_type == "bn":
        return BatchNorm1d
    elif norm_type == "ln":
        return LayerNorm
    else:
        return None


def _get_model(base_model_name, input_size, ptdrop_rate):
    """Function that creates the base pretrained timm model

    Args:
    base_model_name: Name of the pretrained model on timm library
    input_size: Desired input size of the model (-1 selects input size of timm model cfg)
    ptdrop_rate: Patch_dropout rate applied on input (Available only on vit and deit)

    Returns:
    model: The pretrained timm model
    final_input_size: Final input size of model given input size parameter and timm model cfg
    clsidf: Idenfifier to access classifier
    last_size: Size of last tensor before classifier
    mean: Mean the model uses to normalize images, given by timm cfg
    std: Std the model uses to normalize images, given by timm cfg
    """
    # Create timm model
    if "deit" in base_model_name or "vit" in base_model_name:
        model = create_model(
            base_model_name,
            pretrained=True,
            num_classes=1000,
            patch_drop_rate=ptdrop_rate,
        )
    else:
        model = create_model(base_model_name, pretrained=True, num_classes=1000)

    # Access configuration
    cfg = model.default_cfg
    # Read cfg params
    mean = cfg["mean"]
    std = cfg["std"]
    clsidf = cfg["classifier"]

    # Read output shape
    if type(clsidf) is tuple:
        last_size = _rgetattr(model, clsidf[0]).in_features
    else:
        last_size = _rgetattr(model, clsidf).in_features

    # Select definitive input size
    final_input_size = cfg["input_size"][1]
    if input_size != -1:
        final_input_size = input_size
    return model, final_input_size, clsidf, last_size, mean, std


def _get_classifier(
    neurons, layers, last_size, norm, dropout, drop_rate, act, num_label
):
    """Function that creates the classifier used by the model

    Args:
    neurons: Number of neurons of the first hidden classifier layer
    layers: Number of hidden classifier layers
    last_size: Size of last tensor before classifier
    norm: Normalization torch module to use in classifier (None for no normalization applied)
    drop: Dropout torch module to use in classifier (None for no dropout applied)
    drop_rate: Dropout rate in classifier
    num_label: Number of labels of the final classifier layer

    Returns:
    torch.nn.Sequential model with all layers needed for classifier
    """

    # Empty array of layers
    modules = []
    # Set up initial vars
    current_size = neurons

    # Foreach layer --> Drop + Linear + Norm + Act
    for l in range(layers):
        modules.append(dropout(drop_rate))
        modules.append(Linear(in_features=last_size, out_features=current_size))
        if norm is not None:
            modules.append(norm(current_size))
        modules.append(act())

        last_size = current_size
        current_size = current_size // 2

    # Add final classification layer --> Drop + Linear
    modules.append(dropout(drop_rate))
    modules.append(Linear(in_features=last_size, out_features=num_label))

    # Return sequential module
    return Sequential(*modules)


def get_fullmodel(
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
    """Function that handles all the logic of timm model, preprocessing and
    classifier for better ChestModel interface

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

    Returns:
    model: Timm pretrained model + classifier
    preprocessing: Preprocessing needed before model forward
    kwargs: Dictionary of kwargs for easier model saving/loading
    """

    # Get modules selected by parameters
    act = get_activation(activation)
    dropout = get_dropout(drop_type)
    norm = get_norm(norm_type)

    # Build model
    model, input_size, clsidf, last_size, mean, std = _get_model(
        base_model_name, input_size, ptdrop_rate
    )
    kwargs = {
        "pre": pre,
        "base_model_name": base_model_name,
        "input_size": input_size,
        "layers": layers,
        "neurons": neurons,
        "norm_type": norm_type,
        "drop_type": drop_type,
        "drop_rate": drop_rate,
        "ptdrop_rate": ptdrop_rate,
        "activation": activation,
        "num_label": num_label,
        "hierarchy": hierarchy,
    }

    # Set up classifier
    if type(clsidf) is tuple:
        _rsetattr(
            model,
            clsidf[0],
            _get_classifier(
                neurons, layers, last_size, norm, dropout, drop_rate, act, num_label
            ),
        )
        _rsetattr(
            model,
            clsidf[1],
            _get_classifier(
                neurons, layers, last_size, norm, dropout, drop_rate, act, num_label
            ),
        )
    else:
        _rsetattr(
            model,
            clsidf,
            _get_classifier(
                neurons, layers, last_size, norm, dropout, drop_rate, act, num_label
            ),
        )

    # Set up preprocessing
    preprocess = ChestModelPreprocess(input_size=input_size, mean=mean, std=std)

    return model, preprocess, kwargs
