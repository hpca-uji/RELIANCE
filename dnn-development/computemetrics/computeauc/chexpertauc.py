import argparse
from typing import List
import os
import json
from statistics import mean

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from aucops import delong_roc_variance, compute_auc_ci


def load_args():
    """Parameter loading"""
    parser = argparse.ArgumentParser()
    parser.add_argument("chexpert_train_csv_path", type=str)
    parser.add_argument("labelnames_path", type=str)
    parser.add_argument("preds_path", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--models_idfs", type=str, nargs="+", required=True)
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.95)
    args = parser.parse_args()

    return args


def convert_labels(labels: np.array):
    conversion_dict = {0: [31], 1: [8, 9, 10], 2: [8, 9, 10], 3: [1, 2], 4: [1, 2],
                       5: [1], 6: [1, 2, 5], 7: [1, 2, 4], 8: [1, 2, 6], 9: [14, 15, 17],
                       10: [14, 15, 16], 11: [14], 12: [14, 15], 13: [22]}
    converted_labels = np.full((len(labels), 32), -1)
    for position, targets in conversion_dict.items():
        col = labels[:, position].astype(float)
        pos = col == 1
        neg = (col == 0) | ((position == 0) & np.isnan(col))
        for t in targets:
            cur = converted_labels[:, t]
            cur[neg & (cur == -1)] = 0
            cur[pos] = 1
            converted_labels[:, t] = cur
    return converted_labels


def process_labels(labels: pd.DataFrame):

    # Hard coded label names
    chexpertnames = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]

    labels = labels[labels["Frontal/Lateral"] == "Frontal"]
    labels = labels[labels["AP/PA"].isin(["AP","PA"])]
    labels = labels[labels["Age"]>=18]
    labels = labels[chexpertnames].to_numpy()
    labels = convert_labels(labels)

    return labels


def load_data(
    chexpert_train_csv_path: str,
    labelnames_path: str,
    preds_path: str,
    models_idfs: List[str],
):

    # Load labels csv
    labels = pd.read_csv(chexpert_train_csv_path)
    # Process labels
    labels = process_labels(labels)

    # Load labelnames
    with open(labelnames_path) as labelfile:
        labelnames = json.load(labelfile)

    # Load predictions from different models
    preds = dict()
    for idf in models_idfs:
        model_preds_path = os.path.join(
            preds_path, idf, "inferencedata/", "chexpert_preds.pt"
        )
        preds[idf] = torch.load(model_preds_path, weights_only=False).numpy()

    return labels, preds, labelnames


def chexpert_mean_auc(labels: np.array, preds: np.array):

    # Create list of auc
    aucs = []
    n_labels = labels.shape[1]

    # Compute foreach label individually
    for i in range(n_labels):

        mask = labels[:, i] != -1
        y_true = labels[mask, i]
        y_pred = preds[mask, i]

        # Compute number of positives and negatives
        pos = np.sum(y_true == 1)
        neg = np.sum(y_true == 0)

        if len(y_true) == 0 or pos == 0 or neg == 0:
            continue

        aucs.append(roc_auc_score(y_true, y_pred))

    return mean(aucs)


def chexpert_auc_calculator(
    labels: np.array, preds: dict, labelnames: list, bootstraps: int, alpha: float
):

    # Create empty dict
    aucs = dict()

    # Iterate over preds
    for model, model_preds in preds.items():

        # Print computing model
        print(f"Computing: {model}", flush=True)

        # Compute AUC foreach label
        for i, label_name in enumerate(labelnames):

            # Discard -1 from computation
            mask = labels[:, i] != -1
            y_true = labels[mask, i]
            y_pred = model_preds[mask, i]

            # Compute number of positives and negatives
            pos = int(np.sum(y_true == 1))
            neg = int(np.sum(y_true == 0))

            if len(y_true) == 0 or pos == 0 or neg == 0:
                continue

            # Compute delong
            auc, auc_cov = delong_roc_variance(y_true, y_pred)

            # Compute auc ci
            auc_ci = compute_auc_ci(auc=auc, auc_cov=auc_cov, alpha=alpha)

            # Introduce data in dict
            aucs[(model, label_name)] = [
                pos,
                neg,
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
            mean_aucs.append(chexpert_mean_auc(labels[indices], model_preds[indices]))

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
        chexpert_train_csv_path=args.chexpert_train_csv_path,
        labelnames_path=args.labelnames_path,
        preds_path=args.preds_path,
        models_idfs=args.models_idfs,
    )
    print(labels.shape)

    # Compute auc
    aucs = chexpert_auc_calculator(
        labels=labels,
        preds=preds,
        labelnames=labelnames,
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
