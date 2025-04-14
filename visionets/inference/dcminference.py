import argparse

import pydicom as pyd
import numpy as np
import torch
from torchvision.utils import save_image

from visionapi.models import load_model


def get_image(args, device):
    """Function that loads a dicom and returns torch image array"""

    # Read dicom and misc fields
    ds = pyd.dcmread(args.dicom_path)
    color_scheme = ds.get(0x20500020)
    bits_stored = ds.get(0x00280101)
    img = ds.pixel_array

    # Check for Nones
    if color_scheme is None:
        color_scheme = "IDENTITY"  # Assume Identity by default
    else:
        color_scheme = color_scheme.value
    if bits_stored is None:
        bits_stored = 16  # Assume 16 by default
    else:
        bits_stored = bits_stored.value

    # Convert to range 0-255 if needed
    if bits_stored != 8:
        img = img // (2 ** (bits_stored - 8))
        img = img.astype(np.uint8)
    img = torch.ByteTensor(img)

    # Inverse color if needed
    if color_scheme == "INVERSE":
        img = 255 - img

    # Expand batch dimension
    img = img.unsqueeze(0).unsqueeze(0)

    # Convert to [0-1] range and return
    img = img / 255.0

    return img


def load_args():
    """Parameter loading"""
    parser = argparse.ArgumentParser()
    parser.add_argument("dicom_path", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("--model_format", type=str, choices=["torch", "jit"])
    args = parser.parse_args()

    return args


def setup_model(args, device):
    """Create and set up model"""
    if args.model_format == "jit":
        model = torch.jit.load(args.model_path)
    else:
        model = load_model(model_path=args.model_path)
        model = model.to(device)
        model.set_hierarchy_device(device)
        model.eval()

    return model


def main():

    # Initialization
    args = load_args()
    device = torch.device("cpu")
    model = setup_model(args=args, device=device)
    img = get_image(args=args, device=device)
    print(img.shape)
    save_image(img[0], "processdicom.png")

    # Inference on image and print result
    with torch.no_grad():
        predict = model(img)
    print(predict[0])


if __name__ == "__main__":

    main()
