import json
import argparse
import os

import torch
import pydicom as pyd
import numpy as np
from torchvision.io import read_image

from visionapi.models import load_model


def get_images(args, device):
    """Function that reads images, labels and paths"""
    # Read vl paths
    vl_paths = np.load(os.path.join(args.label_path, "vl_paths.npy"), allow_pickle=True)
    with open(os.path.join(args.label_path, "vl_label.pt"), "rb") as f:
        vl_label = torch.load(f)

    # Read ts paths
    ts_paths = np.load(os.path.join(args.label_path, "ts_paths.npy"), allow_pickle=True)
    with open(os.path.join(args.label_path, "ts_label.pt"), "rb") as f:
        ts_label = torch.load(f)

    # Concatenate paths and labels
    paths = np.concatenate((vl_paths, ts_paths)).tolist()
    paths = [path.split("/")[1].replace(".pt", ".png") for path in paths]
    label = torch.cat((vl_label, ts_label))

    # Prepare list of images in vl or ts
    image_list = []
    label_list = []
    path_list = []

    for path in os.listdir(args.folder_path):

        # Check if path is part of validation or testing
        if path in paths:
            idx = paths.index(path)
            read_path = os.path.join(args.folder_path, path)
            image_list.append(read_image(read_path))
            label_list.append(label[idx])
            path_list.append(path)

    # Stack list into torch tensor
    inputs = torch.stack(image_list)

    return inputs.to(device), label_list, path_list


def load_args():
    """Parameter loading"""
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("label_path", type=str)
    parser.add_argument("labelnames_path", type=str)
    parser.add_argument("folder_path", type=str)
    args = parser.parse_args()

    return args


def setup_model(args, device):
    """Create and set up model"""
    model = load_model(args.model_path)
    model = model.to(device)
    model.set_hierarchy_device(device)
    model.eval()
    return model


def main():

    # Initialization
    args = load_args()
    device = torch.device("cpu")
    model = setup_model(args=args, device=device)
    inputs, label_list, path_list = get_images(args=args, device=device)

    # Read label names
    with open(args.labelnames_path) as f:
        labelnames = json.load(f)

    # Process the image
    with torch.no_grad():
        preds = model(inputs / 255.0)

    # Iterate over predictions
    for i, path in enumerate(path_list):
        print(path)
        for j, label in enumerate(labelnames):
            print(f"{label} : {round(preds[i,j].item(),2)} , {label_list[i][j].item()}")
        print()


if __name__ == "__main__":

    main()
