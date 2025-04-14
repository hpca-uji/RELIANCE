import argparse

import torch
import pandas as pd

from visionapi.data import setup_inference_chexpert_dataloader
from visionapi.models import load_model


def load_args():
    """Parameter loading"""
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str)
    parser.add_argument("csv_path", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("save_path", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    return args


def setup_model(args, device):
    """Create and set up model"""
    model = load_model(model_path=args.model_path)
    model = model.to(device)
    model.set_hierarchy_device(device)
    model.eval()
    return model


def process_chexpert_csv(csv_path):
    """Function that process chexpert csv to get an array of paths"""

    # Read csv
    dataf = pd.read_csv(csv_path)

    # Discard lateral
    dataf = dataf[dataf["Frontal/Lateral"] == "Frontal"]

    # Subsitute jpg with pkt

    # Get paths
    paths = dataf["ImageID"].str.replace("jpg", "pkt").to_numpy()
    return paths


def inference_data(device, data, model):
    """Inference data on model"""

    preds_agg = []

    with torch.no_grad():  # No grad to avoid overflow
        for i, inputs in enumerate(data):

            # Load to GPU
            inputs = inputs.to(device)

            # Forward inputs and move to cpu
            preds = model(inputs).to("cpu").tolist()

            # Add to agg
            preds_agg.extend(preds)

            # Print progress
            if i % 100 == 0:
                print(f"{i:04d} / {len(data)}", flush=True)

    return preds_agg


def main():

    # Initialization
    args = load_args()
    device = torch.device("cpu")
    paths = process_chexpert_csv(args.csv_path)
    model = setup_model(args=args, device=device)
    print(model)

    # Load data
    data = setup_inference_chexpert_dataloader(
        img_path=args.img_path,
        paths=paths,
        batch_size=args.batch_size,
    )

    # Inference and save data
    preds = inference_data(device=device, data=data, model=model)
    torch.save(torch.FloatTensor(preds), args.save_path)

    print("Inference completed and saved")


if __name__ == "__main__":
    main()
