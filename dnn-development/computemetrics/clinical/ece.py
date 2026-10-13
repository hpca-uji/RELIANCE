import json
import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import torch


def load_args():

    """Parameter loading"""
    parser = argparse.ArgumentParser()
    parser.add_argument("internal_labels_path", type=str)
    parser.add_argument("internal_preds_path", type=str)
    parser.add_argument("hospital_path", type=str)
    parser.add_argument("labelnames_path")
    parser.add_argument("hospital_labelnames_path")
    parser.add_argument("--models_idfs", type=str, nargs="+")
    args = parser.parse_args()

    return args


def load_data(
    internal_labels_path: str,
    internal_preds_path: str,
    hospital_path: str, 
    labelnames_path: str,
    hospital_labelnames_path: str,
    models_idfs: List[str],
):
    
    # Load internal test labels
    internal_ts_labels = torch.load(os.path.join(internal_labels_path, "ts_label.pt"))

    # Load internal test preds
    internal_ts_preds = dict()
    for idf in models_idfs:
        load_path = os.path.join(internal_preds_path, idf, "inferencedata/", "ts_preds.pt")
        internal_ts_preds[(idf)] = torch.load(load_path)

    # Load labelnames 
    with open(labelnames_path, "r") as file:
        labelnames = json.load(file)

    with open(hospital_labelnames_path, "r") as file:
        hospital_labelnames = json.load(file)

    # Load hospital data
    hospital_labels_csv = pd.read_csv(os.path.join(hospital_path, "labels.csv"), sep="\t", index_col=["dicom_id"])
    # Keep labelnames only from labels
    hospital_labels_csv = hospital_labels_csv[labelnames]
    # Read dicom ids on labels to keep predictions of only those dicoms
    common_idx = hospital_labels_csv.index.tolist()

    # Load hospital predictions from different models
    hospital_preds = dict()
    for idf in models_idfs:

        # Read model preds
        model_preds_path = os.path.join(hospital_path, idf + ".csv")
        model_preds_csv = pd.read_csv(
            model_preds_path, sep="\t", index_col=["dicom_id"]
        )
        # Drop columns
        model_preds_csv = model_preds_csv.drop(
            columns=["Assumed Color Scheme", "Assumed Bits Stored"]
        )
        # Drop duplicated dicom_ids
        model_preds_csv = model_preds_csv[
            ~model_preds_csv.index.duplicated(keep="first")
        ]
        # Save into dict
        hospital_preds[idf] = model_preds_csv

        # Read list of idx
        model_preds_idx = model_preds_csv.index.tolist()
        # Compute smaller common idf
        common_idx = list(set(common_idx).intersection(set(model_preds_idx)))

    # As final step, iterate over preds to convert to numpy
    for idf, model_preds_csv in hospital_preds.items():
        model_preds = model_preds_csv.loc[common_idx].to_numpy()
        hospital_preds[idf] = torch.Tensor(model_preds)

    # Keep common idx on labels
    hospital_labels = torch.Tensor(hospital_labels_csv.loc[common_idx].to_numpy())


    return internal_ts_labels, internal_ts_preds, hospital_labels, hospital_preds, labelnames, hospital_labelnames


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        upper = p < edges[i + 1] if i < bins - 1 else p <= 1
        m = (p >= edges[i]) & upper
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def main():
    
    # Load args
    args = load_args()

    # Load data
    internal_ts_labels, internal_ts_preds, hospital_labels, hospital_preds, labelnames, hospital_labelnames = load_data(internal_labels_path=args.internal_labels_path, 
                                                                                                                        internal_preds_path=args.internal_preds_path,
                                                                                                                        hospital_path=args.hospital_path, 
                                                                                                                        models_idfs=args.models_idfs, 
                                                                                                                        labelnames_path=args.labelnames_path,
                                                                                                                        hospital_labelnames_path=args.hospital_labelnames_path)
    
    # Calibration internal test vs hospital
    for model_idf in internal_ts_preds.keys():
        
        # Print
        print(f"\n{model_idf}")
        print(f"{'label':30s} {'prev_int':>8s} {'meanp_int':>9s} {'ECE_int':>7s} {'prev_hos':>8s} {'meanp_hos':>9s} {'ECE_hos':>7s}")
        
        # Create empty lists
        ece_internal, ece_hospital = [], []

        # Iterate over labelnames
        for c in hospital_labelnames:

            # Get index
            j = labelnames.index(c)

            # Get predictions from model
            preds_internal = internal_ts_preds[model_idf][:, j].numpy()
            preds_hospital = hospital_preds[model_idf][:, j].numpy()

            # Get labels
            y_internal = internal_ts_labels[:, j].numpy()
            y_hospital = hospital_labels[:, j].numpy()

            # Compute and add to list
            ece_internal.append(ece(y_internal, preds_internal))
            ece_hospital.append(ece(y_hospital, preds_hospital))
            
            # Print 
            print(f"{c:30s} {y_internal.mean():8.3f} {preds_internal.mean():9.3f} {ece_internal[-1]:7.3f} {y_hospital.mean():8.3f} {preds_hospital.mean():9.3f} {ece_hospital[-1]:7.3f}")
        
        # Print macro
        print(f"{'Macro ECE':30s} {'':8s} {'':9s} {np.mean(ece_internal):7.3f} {'':8s} {'':9s} {np.mean(ece_hospital):7.3f}")


if __name__ == "__main__":
    main()