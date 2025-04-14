import os
import argparse

import pandas as pd
import pickle
from PIL import Image


def load_args():
    """Parameter loading"""
    parser = argparse.ArgumentParser(
        description="Create pickle files from png files loaded with PIL"
    )
    parser.add_argument("csv_path", type=str, help="Path of the dataset csv")
    parser.add_argument("img_path", type=str, help="Path where images are stored")
    parser.add_argument("save_path", type=str, help="Path where the npy will be saved")
    parser.add_argument(
        "--folder_column",
        type=str,
        help="Name of the column containing the id of the folder",
        default=None,
    )
    parser.add_argument(
        "--rm_path",
        "-remove_path",
        type=str,
        help="Optional path of filenames to be deleted",
        default=None,
    )
    args = parser.parse_args()
    return args


def create_img(img_read_path, img_save_path):
    """Create pillow-pickle image from png and save as pkt"""
    img = Image.open(img_read_path)
    with open(img_save_path, "wb") as wfile:
        pickle.dump(img, wfile)


def create_pkt_imgs(ids, dirs, img_path, save_path, df):
    """Creates pkt images from png images"""
    if dirs is None:
        for i in ids:
            img_read_path = os.path.join(img_path, i)
            i_save = i.split(".")[0]
            img_save_path = os.path.join(save_path, i_save + ".pkt")
            create_img(img_read_path, img_save_path)
    else:
        for i, d in zip(ids, dirs):
            img_read_path = os.path.join(img_path, str(d), i)
            i_save = i.split(".")[0]
            img_save_path = os.path.join(save_path, i_save + ".pkt")
            create_img(img_read_path, img_save_path)


if __name__ == "__main__":

    # Initialization
    args = load_args()
    # Create save_path folder if it does not exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Read csv
    df = pd.read_csv(args.csv_path)

    # Delete corrupted images
    if args.rm_path is not None:
        rm_file = open(args.rm_path, "r")
        for line in rm_file:
            df = df[df["ImageID"] != line.strip()]

    # Identifiers
    ids = df["ImageID"].to_numpy()
    if args.folder_column != None:
        dirs = df[args.folder_column].to_numpy()
    else:
        dirs = None

    # Create npy
    create_pkt_imgs(ids, dirs, args.img_path, args.save_path, df)
