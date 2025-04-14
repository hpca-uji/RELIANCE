import argparse

from statistics import mean, stdev
import os
import json

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from torchmetrics import AUROC, FBetaScore, ROC, StatScores


def load_args():
    """Parameter loading"""
    parser = argparse.ArgumentParser()
    parser.add_argument("train_csv_path", type=str)
    parser.add_argument("preds_path", type=str)
    parser.add_argument("labelnames_path", type=str)
    args = parser.parse_args()

    return args


def load_data(args):
    """Data loading"""
    labels = pd.read_csv(args.train_csv_path)
    preds = torch.load(args.preds_path, weights_only=True).numpy()
    with open(
        "/home/fsoler/projects/reliance/dbprocess/padchest/dictprocess/urgencylabel/labelnames.json"
    ) as labelfile:
        labelnames = json.load(labelfile)
    return labels, preds, labelnames


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
    labels = labels[chexpertnames].to_numpy()
    labels = convert_labels(labels)

    return labels


def convert_labels(labels: np.array):

    # Hard coded conversion dict between chexpert labels and 32labels
    conversion_dict = {
        0: [31],
        1: [8, 9, 10],
        2: [8, 9, 10],
        3: [1, 2],
        4: [1, 2],
        5: [1],
        6: [1, 2, 5],
        7: [1, 2, 4],
        8: [1, 2, 6],
        9: [14, 15, 17],
        10: [14, 15, 16],
        11: [14],
        12: [14, 15],
        13: [22],
    }

    # Convert labels on iteration
    converted_labels = np.full((len(labels), 32), -1)
    for i in range(labels.shape[0]):
        samplelabel = labels[i]
        for position, value in enumerate(samplelabel):
            if value == 1:
                converted_labels[i, conversion_dict[position]] = 1
            elif value == 0:
                converted_labels[i, conversion_dict[position]] = 0
            elif position == 0:  # No finding label put 0 on default
                converted_labels[i, conversion_dict[position]] = 0

    return converted_labels


def compute_auc(labels: np.array, preds: np.array, labelnames: list[str]):
    """Function that computes auc per label"""
    scores = []
    validnames = []
    for i, label_name in enumerate(labelnames):
        neg = (labels[:, i] == 0).sum(axis=0)
        pos = (labels[:, i] == 1).sum(axis=0)
        # If no positive or negative samples continue to next iteration
        if neg == 0 or pos == 0:
            continue
        # Update valid_names
        validnames.append(label_name)
        # Get indices where positive or negative
        valid_idx = np.where(labels[:, i] != -1)
        # Compute auc on valid index
        score = roc_auc_score(labels[valid_idx, i][0], preds[valid_idx, i][0])
        scores.append(score)
    scores.append(mean(scores))
    validnames.append("Mean")
    return scores, validnames


def print_auc(scores: list[float], validnames: list[str]):
    """Function that prints auc in pretty format"""
    for score, validname in zip(scores, validnames):
        print(f"{validname:16}: {score}")


def main():

    # Load args
    args = load_args()

    # Load data
    labels, preds, labelnames = load_data(args)

    # Process labels
    labels = process_labels(labels)

    # Compute auc
    scores, validnames = compute_auc(labels, preds, labelnames)

    # Print result
    print_auc(scores, validnames)


if __name__ == "__main__":
    main()
