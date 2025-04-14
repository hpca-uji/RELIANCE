import argparse
import os

import torch
from torchvision.io import read_image
import pandas as pd

from visionapi.data import setup_paths_imgformat_dataloader
from visionapi.models import load_model


def load_args():
    """Parameter loading"""
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str)
    parser.add_argument("csv_path", type=str)
    args = parser.parse_args()

    return args


def process_chexpert_csv(csv_path):
    """Function that process chexpert csv to get an array of paths"""

    # Read csv
    dataf = pd.read_csv(csv_path)

    # Discard lateral
    dataf = dataf[dataf["Frontal/Lateral"] == "Frontal"]

    # Get paths
    paths = dataf["Path"].to_numpy()
    return paths


def readallimg(data):
    """Read all images"""
    for i, _ in enumerate(data):
        if i % 50 == 0:
            print(f"{i:04d} / {len(data)}", flush=True)


def main():

    args = load_args()
    paths = process_chexpert_csv(args.csv_path)

    data = setup_paths_imgformat_dataloader(
        img_path=args.img_path,
        paths=paths,
        size=None,
        return_images=False,
        batch_size=512,
    )
    readallimg(data)


if __name__ == "__main__":
    main()
