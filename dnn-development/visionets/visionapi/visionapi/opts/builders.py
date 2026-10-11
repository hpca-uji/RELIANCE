from torch.optim import Adam, AdamW, NAdam, RAdam, Adamax

from ._lrschedulers import MultiGPUReducelr


def _select_opt(opt_value):
    """Torch optimizer selector"""
    if opt_value == "adam":
        optclass = Adam
    elif opt_value == "adamw":
        optclass = AdamW
    elif opt_value == "nadam":
        optclass = NAdam
    elif opt_value == "radam":
        optclass = RAdam
    elif opt_value == "adamax":
        optclass = Adamax
    else:
        raise ValueError("{opt} is not valid as optimizer")

    return optclass


def setup_frozenopt(opt_value, pre, lr, model, device):
    """Sets up optimizer with frozen pretrained model

    Args:
    opt_value: Identifier of optimizer (adam, adamw, nadam, radam, adamax)
    pre: Preprocessing technique applied (convrand, convfixed, resize)
    lr: Learning rate of optimizer
    model: Neural model to be optimized
    device: Device where model runs

    Returns:
    opt: Torch optim object
    reduce_lr: Custom torch reduce lr for usage on multi-gpu
    """
    # Opt class selection
    optclass = _select_opt(opt_value)

    # Set up optimizer and reducelr (classifier parameters)
    if pre == "resize":  # Freeze parameters on preprocessing
        opt = optclass(
            lr=lr, params=[{"params": model.module.get_classifier_parameters()}]
        )
    else:
        opt = optclass(
            lr=lr,
            params=[
                {"params": model.module.get_classifier_parameters()},
                {"params": model.module.preprocess.parameters()},
            ],
        )
    reduce_lr = MultiGPUReducelr(
        optimizer=opt, patience=2, factor=0.1, mode="max", verbose_device=device
    )

    return opt, reduce_lr


def setup_opt(opt_value, pre, lr, model, device):
    """Sets up optimizer for full model training

    Args:
    opt_value: Identifier of optimizer (adam, adamw, nadam, radam, adamax)
    pre: Preprocessing technique applied (convrand, convfixed, resize)
    lr: Learning rate of optimizer
    model: Neural model to be optimized
    device: Device where model runs

    Returns:
    opt: Torch optim object
    reduce_lr: Custom torch reduce lr for usage on multi-gpu
    """
    # Opt class selection
    optclass = _select_opt(opt_value)

    # Set up optimizer and reducelr (classifier parameters)
    if pre == "resize":  # Freeze parameters on preprocessing
        opt = optclass(lr=lr, params=[{"params": model.module.model.parameters()}])
    else:
        opt = optclass(lr=lr, params=[{"params": model.module.parameters()}])
    reduce_lr = MultiGPUReducelr(
        optimizer=opt, patience=2, factor=0.1, mode="max", verbose_device=device
    )

    return opt, reduce_lr
