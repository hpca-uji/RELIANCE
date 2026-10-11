import argparse
import json
import os
from typing import List

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from aucops import delong_roc_variance, compute_auc_ci


def load_args():
    """Parameter loading"""
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_path", type=str)
    parser.add_argument("labelnames_path")
    parser.add_argument("preds_path", type=str)
    parser.add_argument("output_file")
    parser.add_argument("--models_idfs", type=str, nargs="+")
    parser.add_argument("--partitions", type=str, nargs="+")
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--selected_labelnames_path", type=str, default=None)
    args = parser.parse_args()

    return args


def load_data(
    labels_path: str,
    labelnames_path: str,
    preds_path: str,
    models_idfs: List[str],
    partitions: List[str],
    selected_labelnames_path: str = None,
):

    # Load labels
    tr_label = torch.load(os.path.join(labels_path, "tr_label.pt"))
    vl_label = torch.load(os.path.join(labels_path, "vl_label.pt"))
    ts_label = torch.load(os.path.join(labels_path, "ts_label.pt"))
    labels = {"tr": tr_label, "vl": vl_label, "ts": ts_label}

    # Load preds
    preds = dict()
    for idf in models_idfs:
        inference_path = os.path.join(preds_path, idf, "inferencedata/")
        for part in partitions:
            load_path = os.path.join(inference_path, part + "_preds.pt")
            preds[(idf, part)] = torch.load(load_path)

    # Load labelnames
    with open(labelnames_path, "r") as file:
        labelnames = json.load(file)

    # Load selected labelnames (optional)
    if selected_labelnames_path is not None:

        with open(selected_labelnames_path, "r") as file:
            selected_labelnames = json.load(file)

        label_indices = [labelnames.index(name) for name in selected_labelnames]

        labelnames = selected_labelnames

        for part in labels:
            labels[part] = labels[part][:, label_indices]

        for key in preds:
            preds[key] = preds[key][:, label_indices]

    return labels, preds, labelnames


def parse_dict(dict):
    dict_string = {str(key): value for key, value in dict.items()}
    return dict_string


def internal_auc_calculator(
    labels: dict,
    preds: dict,
    labelnames: list,
    bootstraps: int = 1000,
    alpha: float = 0.95,
):

    # Create empty dictionary of aucs
    aucs = dict()

    # Iterate over keys from predictions
    for (model, part), model_part_preds in preds.items():

        # Print computing pair model, partition
        print(f"Computing: {model, part}", flush=True)

        # Load labels this partition
        part_label = labels[part]

        # Iterate over labels
        for i, label_name in enumerate(labelnames):

            # Read columns into contiguous format
            local_label = part_label[:, i].numpy()
            local_preds = model_part_preds[:, i].numpy()

            auc, auc_cov = delong_roc_variance(local_label, local_preds)

            # Compute auc ci
            auc_ci = compute_auc_ci(auc=auc, auc_cov=auc_cov, alpha=alpha)

            # Introduce data in dict
            aucs[(model, part, label_name)] = [
                float(auc),
                float(auc_ci[0]),
                float(auc_ci[1]),
            ]

        # Compute boostrapped mean auc
        n_samples = part_label.shape[0]
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
                roc_auc_score(part_label[indices], model_part_preds[indices])
            )

        # Percentile computation
        mean_auc = np.mean(mean_aucs)
        lower = (1 - alpha) / 2 * 100
        upper = 100 - lower
        lower_auc = np.percentile(mean_aucs, lower)
        upper_auc = np.percentile(mean_aucs, upper)

        aucs[(model, "Avg")] = [mean_auc, lower_auc, upper_auc]

    return aucs


def main():

    # Load args
    args = load_args()

    # Load data
    labels, preds, labelnames = load_data(
        labels_path=args.labels_path,
        labelnames_path=args.labelnames_path,
        preds_path=args.preds_path,
        models_idfs=args.models_idfs,
        partitions=args.partitions,
        selected_labelnames_path=args.selected_labelnames_path
    )

    # Compute data
    aucs = internal_auc_calculator(
        labels=labels,
        preds=preds,
        labelnames=labelnames,
        bootstraps=args.bootstraps,
        alpha=args.alpha,
    )

    # Convert keys to string
    aucs = parse_dict(aucs)

    # Save data
    with open(args.output_file, "w") as file:
        json.dump(aucs, file)


if __name__ == "__main__":
    main()
