import os
import argparse

import pydicom as pyd
import numpy as np
import pandas as pd
import torch
import pickle
from torchvision.transforms.functional import resize, to_pil_image


def load_args():
    """Parameter loading"""
    parser = argparse.ArgumentParser(
        description="Create pickle raw files from png files"
    )
    parser.add_argument("csv_path", type=str, help="Path of the translated csv")
    parser.add_argument("img_path", type=str, help="Path where images are stored")
    parser.add_argument(
        "save_path", type=str, help="Path where the pkt files will be saved"
    )
    args = parser.parse_args()
    return args


def get_dicom_img(path):
    """Get image from dicom"""
    # Read the dicom
    ds = pyd.dcmread(path)

    # Access dicom fields
    color_scheme = ds.get(0x20500020)
    bits_stored = ds.get(0x00280101)
    img = ds.pixel_array

    # Check for Nones
    if color_scheme is None:
        color_scheme = "IDENTITY"
    else:
        color_scheme = color_scheme.value

    if bits_stored is None:
        bits_stored = 12
    else:
        bits_stored = bits_stored.value

    # Process data according to fields

    # Convert to range 0-255
    if bits_stored != 8:
        img = img // (2 ** (bits_stored - 8))
        img = img.astype(np.uint8)
    img = torch.ByteTensor(img)

    # Inverse color if needed
    if color_scheme == "INVERSE":
        img = 255 - img

    # Add channel dimension
    img = torch.unsqueeze(img, 0)

    # Reduce to 512*512
    img = to_pil_image(resize(img, [512, 512]))

    return img


def main():
    """This script takes the csv given by the preprocessing and creates a pkt file for each dicom inside de csv
    It also creates an npy file with all path files
    """

    # Load arguments
    args = load_args()

    # Read pandas dataframe
    dataf = pd.read_csv(args.csv_path)

    # Get paths where dicoms are stored originally
    dicom_ids = dataf["ImageID"].to_list()

    # Build paths where images are saved
    save_ids = [idf.split("/")[-1] for idf in dicom_ids]

    for dicom_id, save_id in zip(dicom_ids, save_ids):

        # Build path adding img_path folder and deleting last 3 letters of dicom_id (.pkt)
        read_path = os.path.join(args.img_path, dicom_id[:-4])
        img_save_path = os.path.join(args.save_path, save_id)

        # Get dicom img
        img = get_dicom_img(read_path)

        # Save data
        with open(img_save_path, "wb") as wfile:
            pickle.dump(img, wfile)

    # Save list of paths for use in model training
    father_folder = args.save_path.split("/")[-1]
    save_ids = [father_folder + "/" + save_id for save_id in save_ids]
    print(save_ids)
    np.save("at_paths.npy", np.array(save_ids))


if __name__ == "__main__":

    main()
