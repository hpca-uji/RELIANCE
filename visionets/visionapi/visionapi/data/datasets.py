import os
import pickle

import torch
from torch.nn import Identity
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import Resize
from torchvision.io import read_image


class DisillationChestDataset(Dataset):
    """Dataset class that yields samples with labels and pseudolabels
    from teacher model. Distillation ONLY
    """

    def __init__(self, img_path, label, pre, paths, datransforms):
        self.img_path = img_path
        self.label = label
        self.pre = pre
        self.paths = paths
        self.datransforms = datransforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        # Load pickle
        with open(os.path.join(self.img_path, self.paths[idx]), "rb") as f:
            img = pickle.load(f)

        # Augment if transform is indicated
        if self.datransforms is not None:
            img = self.datransforms(img)

        # Transform to tensor
        img = to_tensor(img)

        return (img, self.pre[idx], self.label[idx])


class ChestDataset(Dataset):
    """Dataset class that yields samples with labels"""

    def __init__(self, img_path, label, paths, datransforms):
        self.label = label
        self.img_path = img_path
        self.paths = paths
        self.datransforms = datransforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        # Load image
        with open(os.path.join(self.img_path, self.paths[idx]), "rb") as f:
            img = pickle.load(f)

        # Augment image if indicated
        if self.datransforms is not None:
            img = self.datransforms(img)
        img = to_tensor(img)

        return (img, torch.Tensor(0), self.label[idx])


class InferenceChestDataset(Dataset):
    """Dataset class that yields samples for inference only"""

    def __init__(self, img_path, paths):
        self.img_path = img_path
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        # Load image
        with open(os.path.join(self.img_path, self.paths[idx]), "rb") as f:
            img = to_tensor(pickle.load(f))

        return img
