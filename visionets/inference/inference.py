import os
import argparse

import numpy as np
import torch

from visionapi.data import setup_inference_dataloader
from visionapi.models import load_model


def load_args():
    """Parameter loading"""
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str)
    parser.add_argument("label_path", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("save_path", type=str)
    parser.add_argument("idf_list", type=str, nargs="+")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    return args


def setup_model(args, device):
    """Create and set up model"""
    model = load_model(model_path=args.model_path)
    model = model.to(device)
    model.set_hierarchy_device(device)
    model.eval()
    return model


def inference_data(device, data, model):
    """Inference data on model"""

    preds_agg = []

    with torch.no_grad():  # No grad to avoid overflow
        for inputs in data:

            # Load to GPU
            inputs = inputs.to(device)

            # Forward inputs and move to cpu
            preds = model(inputs).to("cpu").tolist()

            # Add to agg
            preds_agg.extend(preds)

    return preds_agg


def main():

    # Initialization
    args = load_args()
    device = torch.device("cuda")
    model = setup_model(args=args, device=device)

    # Create predictions folder
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)

    # Inference for each requested idf
    for idf in args.idf_list:

        print(f"Starting inference on '{idf}' identifier", flush=True)

        # Load data
        data = setup_inference_dataloader(
            img_path=args.img_path,
            label_path=args.label_path,
            idf_list=[idf],
            batch_size=args.batch_size,
        )

        # Inference and save data
        preds = inference_data(device=device, data=data, model=model)
        torch.save(
            torch.FloatTensor(preds), os.path.join(args.save_path, idf + "_preds.pt")
        )

    print("Inference completed and saved")


if __name__ == "__main__":
    main()
