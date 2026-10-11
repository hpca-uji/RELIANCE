import os

from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import torch

from .datasets import DisillationChestDataset
from .datasets import InferenceChestDataset
from .datasets import ChestDataset
from .samplers import RASampler


def _load_label_path(label_path, idf):
    """Loads specific image paths and labels on a given idf"""

    paths = np.load(os.path.join(label_path, idf + "_paths.npy"), allow_pickle=True)
    with open(os.path.join(label_path, idf + "_label.pt"), "rb") as f:
        label = torch.load(f)
    return paths, label


def _load_pre_path(pre_path, idf):
    """Loads precompute predictios from teacher on a given path"""
    with open(os.path.join(pre_path, idf + "_pre.pt"), "rb") as f:
        pre = torch.load(f)
    return pre


def _setup_repeat_distributed_dataloader(
    dataset, num_repeats, batch_size, pin_memory, num_workers
):
    """Creates distributed sampler and dataloader with repeated augmentation

    Args:
    dataset: Dataset object to be transformed into dataloader
    batch_size: Dataloader single-gpu batch size

    Returns:
    data: Torch dataloader containing training samples
    sampler: Torch distributed sampler
    """

    sampler = RASampler(dataset, num_repeats=num_repeats)
    data = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    return data, sampler


def _setup_distributed_dataloader(dataset, batch_size, pin_memory, num_workers):
    """Creates distributed sampler and dataloader

    Args:
    dataset: Dataset object to be transformed into dataloader
    batch_size: Dataloader single-gpu batch size

    Returns:
    data: Torch dataloader containing training samples
    sampler: Torch distributed sampler
    """

    sampler = DistributedSampler(dataset)
    data = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    return data, sampler


def _setup_dataloader(dataset, batch_size, pin_memory, num_workers):
    """Creates simple dataloader

    Args:
    dataset: Dataset object to be transformed into dataloader
    batch_size: Dataloader single-gpu batch size

    Returns:
    data: Torch dataloader containing training samples
    """
    data = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    return data


def _compute_pw(pw_formula, device, label):
    """Function that compute positive weights from labels with a given formula

    Args:
    pw_formula: Which formula to use as pw (multilabel, multiclass)
    device: Which device to store pw into
    label: Labels on which pw are computed
    """

    if pw_formula == "multilabel":
        samples = label.shape[0]
        counts = label.sum(dim=0)
        pw = (samples - counts) / counts
    elif pw_formula == "multiclass":
        samples = label.shape[0]
        counts = label.sum(dim=0)
        classes = label.shape[1]
        pw = samples / (counts * classes)
    else:
        raise ValueError(f"{pw_formula} not valid as pos weight formula")
    return pw.to(device)


def _setup_pw(pw_apply, pw_formula, device, label):
    """Function that computes positive weights from labels with a given formula
    and sets it up in array for usage in dual classifier (classifier and distillator)
    Args:
    pw_apply: Where to apply pw (cls, distill, dual)
    pw_formula: Which formula to use as pw (multilabel, multiclass)
    device: Which device to store pw into
    label: Labels on which pw are computed
    """

    if pw_formula is None:
        pw = [None, None]
    else:
        pw = _compute_pw(pw_formula=pw_formula, device=device, label=label)
        if pw_apply == "cls":
            pw = [pw, None]
        elif pw_apply == "distill":
            pw = [None, pw]
        elif pw_apply == "dual":
            pw = [pw, pw]
    return pw


def setup_complex_dataloader(
    img_path,
    pre_path,
    label_path,
    semi_id,
    pw_apply,
    pw_formula,
    device,
    datransforms,
    num_repeats,
    batch_size,
    pin_memory,
    num_workers,
):
    """Sets up complex dataloader for different training methodologies, such as
    semi-supervised learning, distillation or simple training. Serves as a wrapper
    to call the proper Dataset class

    Args:
    img_path: Root path where all images are saved from several datasets
    pre_path: Root path where all outputs from teacher model are saved from several datasets
    label_path: Path to load label structure (paths + labels)
    semi_id: Name of semi-supervised dataset
    pw_apply: Where to apply pw (cls, distill, dual)
    pw_formula:  Which pw formula to compute (multilabel, multiclass)
    device: Which device to store pw into
    datransforms: Data augmentations transforms to be applied
    batch_size: Dataloader single-gpu batch size

    Returns:
    data: Torch dataloader containing training samples
    sampler: Torch distributed sampler
    pw: Python list containing pw calculations to provide to loss init
    """

    # Load paths and label files
    paths, label = _load_label_path(label_path=label_path, idf="tr")
    pw = _setup_pw(pw_apply=pw_apply, pw_formula=pw_formula, device=device, label=label)

    # Distillation active
    if pre_path is not None:
        # Load predictions from teacher on tr
        pre = _load_pre_path(pre_path=pre_path, idf="tr")
        if semi_id is not None:  # Mixed training
            # Load semisupervised dataset
            semi_pre = _load_pre_path(pre_path=pre_path, idf=semi_id)
            semi_label = torch.round(semi_pre)
            semi_paths = np.load(
                os.path.join(label_path, semi_id + "_paths.npy"), allow_pickle=True
            )
            # Connect both datasets
            paths = np.concatenate([paths, semi_paths])
            label = torch.cat([label, semi_label])
            pre = torch.cat([pre, semi_pre])
            # Build dataset
        # Build dataset
        dataset = DisillationChestDataset(
            img_path=img_path,
            label=label,
            pre=pre,
            paths=paths,
            datransforms=datransforms,
        )
    else:
        dataset = ChestDataset(
            img_path=img_path, label=label, paths=paths, datransforms=datransforms
        )

    # Prepare distributed dataloader (repeat sampler if indicated)
    if num_repeats > 1:
        data, sampler = _setup_repeat_distributed_dataloader(
            dataset=dataset,
            num_repeats=num_repeats,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
    else:
        data, sampler = _setup_distributed_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )

    return data, sampler, pw


def setup_simple_dataloader(
    img_path,
    label_path,
    idf_list,
    multigpu,
    datransforms,
    batch_size,
    pin_memory,
    num_workers,
):
    """Sets up a simple dataloader that yields img + label. Several idfs can be loaded at the same time

    Args:
    img_path: Root path where all images are saved from several datasets
    label_path: Path to load label structure (paths + labels)
    idfs_list: Which idfs to load (tr, vl, ts, at)
    multigpu: Whether multi-gpu is applied
    datransforms: Data augmentations transforms to be applied
    batch_size: Dataloader single-gpu batch size

    Returns:
    data: Torch dataloader containing training samples
    sampler: Torch distributed sampler if multigpu is active
    """

    # Loads paths and labels from all selected idfs and concatenate them
    paths_list = []
    label_list = []
    for idf in idf_list:
        paths, label = _load_label_path(label_path=label_path, idf=idf)
        paths_list.append(paths)
        label_list.append(label)
    paths = np.concatenate(paths_list)
    label = torch.cat(label_list)

    # Create dataloader and sampler
    dataset = ChestDataset(
        paths=paths, label=label, img_path=img_path, datransforms=datransforms
    )

    # Set up multi-gpu
    if multigpu == 0:
        data = _setup_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        return data, None
    else:
        data, sampler = _setup_distributed_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        return data, sampler


def setup_inference_dataloader(
    img_path, label_path, idf_list, batch_size, num_workers=10
):
    """Sets up a simple dataloader for inference (yields only images).
    Several idfs can be loaded at the same time

    Args:
    img_path: Root path where all images are saved from several datasets
    label_path: Path to load label structure (paths + labels)
    idf_list: Which idfs to load (tr, vl, ts)
    batch_size: Dataloader single-gpu batch size

    Returns:
    data: Torch dataloader containing inference samples
    """

    # Loads paths from all selected idfs and concatenate them
    paths_list = []
    for idf in idf_list:
        paths = np.load(os.path.join(label_path, idf + "_paths.npy"), allow_pickle=True)
        paths_list.append(paths)
    paths = np.concatenate(paths_list)

    # Create dataloader and sampler
    dataset = InferenceChestDataset(img_path=img_path, paths=paths)
    data = _setup_dataloader(
        dataset=dataset, batch_size=batch_size, pin_memory=False, num_workers=num_workers
    )
    return data


def setup_inference_chexpert_dataloader(img_path, paths, batch_size, num_workers=10):
    """Sets up an inference dataloader for inference (yields only images).
    Given root img path and individual path images in jpg or png format.

    Args:
    img_path: Path where chexpert is stored
    paths: Paths of individual images
    batch_size: Dataloader single-gpu batch size

    Returns:
    data: Torch dataloader containing inference samples
    """

    dataset = InferenceChestDataset(img_path=img_path, paths=paths)
    data = _setup_dataloader(
        dataset=dataset, batch_size=batch_size, pin_memory=False, num_workers=num_workers
    )
    return data


def setup_alpha_dataloader(
    img_path,
    ori_paths,
    ori_label,
    dst_paths,
    dst_label,
    datransforms,
    batch_size,
    prev_alpha,
    pin_memory,
    num_workers,
):
    """Sets up and resets dataloader with origin dataset and destination dataset
    in active learning scenario. Random samples are selected from origin dataset
    given prev_alpha parameter

    Args:
    img_path: Root path where all images are saved from several datasets
    ori_paths: Origin dataset specific paths
    ori_label: Origin dataset specific labels foreach path
    dst_paths: Destination dataset specific paths
    dst_label: Destination dataset specific labels foreach path
    datransforms: Data augmentations transforms to be applied
    batch_size: Dataloader single-gpu batch size
    prev_alpha: Random percent of samples reused from origin dataset

    Returns:
    data: Torch dataloader containing training samples
    sampler: Torch distributed sampler
    """

    # Get size of ori dataset and cut down dataset
    size = len(ori_paths)
    new_size = int(size * prev_alpha)
    # Generate list of indexes
    indexes = list(range(size))
    # Random choice with alpha
    selected_indexes = np.random.choice(indexes, size=new_size, replace=False)
    # Add new paths and new label
    paths = np.concatenate((dst_paths, ori_paths[selected_indexes]))
    label = torch.cat((dst_label, ori_label[selected_indexes]))

    # Build dataset and dataloader
    dataset = ChestDataset(
        img_path=img_path, label=label, paths=paths, datransforms=datransforms
    )
    data, sampler = _setup_distributed_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    return data, sampler
