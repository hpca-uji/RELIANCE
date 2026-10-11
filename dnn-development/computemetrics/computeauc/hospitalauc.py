import argparse
from typing import List
import os
import json
from statistics import mean

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from aucops import delong_roc_variance, compute_auc_ci


def load_args():
    """Parameter loading"""
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_csv_path", type=str)
    parser.add_argument("labelnames_path", type=str)
    parser.add_argument("preds_path", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--models_idfs", type=str, nargs="+", required=True)
    parser.add_argument("--count_limit", type=int, default=10)
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--selected_labelnames_path",type=str,default=None)
    args = parser.parse_args()

    return args


def load_data(
    labels_csv_path: str,
    labelnames_path: str,
    preds_path: str,
    models_idfs: List[str],
    selected_labelnames_path: str = None,
):

    # Load labels csv
    labels_csv = pd.read_csv(labels_csv_path, sep="\t", index_col=["dicom_id"])

    # Load labelnames
    with open(labelnames_path) as labelfile:
        labelnames = json.load(labelfile)
    
    selected_labelnames = None
    if selected_labelnames_path is not None:
        with open(selected_labelnames_path) as file:
            selected_labelnames = json.load(file)
    if selected_labelnames is not None:
        label_indices = [
            labelnames.index(name)
            for name in selected_labelnames
        ]
        labelnames = selected_labelnames
    else:
        label_indices = list(range(len(labelnames)))

    # Keep labelnames only from labels
    labels_csv = labels_csv[labelnames]
    # Read dicom ids on labels to keep predictions of only those dicoms
    common_idx = labels_csv.index.tolist()

    # Load predictions from different models
    preds = dict()
    for idf in models_idfs:

        # Read model preds
        model_preds_path = os.path.join(preds_path, idf + ".csv")
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
        preds[idf] = model_preds_csv

        # Read list of idx
        model_preds_idx = model_preds_csv.index.tolist()
        # Compute smaller common idf
        common_idx = list(set(common_idx).intersection(set(model_preds_idx)))

    # As final step, iterate over preds to convert to numpy
    for idf, model_preds_csv in preds.items():
        model_preds = model_preds_csv.loc[common_idx].to_numpy()
        model_preds = model_preds[:,label_indices]
        preds[idf] = model_preds

    # Keep common idx on labels
    labels = labels_csv.loc[common_idx].to_numpy()

    return labels, preds, labelnames


def hospital_mean_auc(labels: np.array, preds: np.array, count_limit: int):

    # Create list of auc
    aucs = []
    n_labels = labels.shape[1]

    # Compute foreach label individually
    for i in range(n_labels):

        y_true = labels[:, i]
        y_pred = preds[:, i]

        # Compute number of positives
        pos = np.sum(y_true)
        if pos <= count_limit:
            continue

        aucs.append(roc_auc_score(y_true, y_pred))

    return mean(aucs)


def hospital_auc_calculator(
    labels: np.array,
    preds: dict,
    labelnames: list,
    count_limit: int,
    bootstraps: int,
    alpha: float,
):

    # Create empty dict
    aucs = dict()

    # Iterate over preds
    for model, model_preds in preds.items():

        # Print computing model
        print(f"Computing: {model}", flush=True)

        # Compute AUC foreach label
        for i, label_name in enumerate(labelnames):

            y_true = labels[:, i]
            y_pred = model_preds[:, i]

            # Compute number of positives
            pos = int(np.sum(y_true))
            if pos <= count_limit:
                continue

            # Compute delong
            auc, auc_cov = delong_roc_variance(y_true, y_pred)

            # Compute auc ci
            auc_ci = compute_auc_ci(auc=auc, auc_cov=auc_cov, alpha=alpha)

            # Introduce data in dict
            aucs[(model, label_name)] = [
                pos,
                float(auc),
                float(auc_ci[0]),
                float(auc_ci[1]),
            ]

        # Compute boostrapped mean auc
        n_samples = labels.shape[0]
        mean_aucs = []
        for _ in range(bootstraps):

            # Random selection with replacement
            indices = np.random.choice(
                n_samples,
                size=n_samples,
                replace=True,
            )

            # Computation
            mean_aucs.append(
                hospital_mean_auc(labels[indices], model_preds[indices], count_limit)
            )

        # Percentile computation
        mean_auc = np.mean(mean_aucs)
        lower = (1 - alpha) / 2 * 100
        upper = 100 - lower
        lower_auc = np.percentile(mean_aucs, lower)
        upper_auc = np.percentile(mean_aucs, upper)

        aucs[(model, "Avg")] = [mean_auc, lower_auc, upper_auc]

    return aucs


def parse_dict(dict):
    dict_string = {str(key): value for key, value in dict.items()}
    return dict_string


def main():

    # Load args
    args = load_args()

    # Load data
    labels, preds, labelnames = load_data(
        labels_csv_path=args.labels_csv_path,
        labelnames_path=args.labelnames_path,
        preds_path=args.preds_path,
        models_idfs=args.models_idfs,
        selected_labelnames_path=args.selected_labelnames_path
    )

    # Print shapes
    print(f"Labels shape: {labels.shape}")
    for idx, model_preds in preds.items():
        print(f"{idx} preds shape: {model_preds.shape}")

    # Compute auc
    aucs = hospital_auc_calculator(
        labels=labels,
        preds=preds,
        labelnames=labelnames,
        count_limit=args.count_limit,
        bootstraps=args.bootstraps,
        alpha=args.alpha,
    )

    # Convert keys to string
    aucs = parse_dict(aucs)

    # Save aucs
    with open(args.output_file, "w") as file:
        json.dump(aucs, file)


if __name__ == "__main__":
    main()
