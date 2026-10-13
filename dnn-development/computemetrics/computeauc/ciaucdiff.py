"""Additional paired bootstrap analyses reported in the paper.

Run from dnn-development/:

    python3 extra_analyses.py [--chexpert_csv /path/to/chexpert/labels.csv]

Outputs:
  * Paired bootstrap (1000 resamples of hospital images) of the difference in
    mean AUC over the 14 shared labels: DenseNet-169 - SwinTV2, ConvNeXt - DenseNet-169.
  * With --chexpert_csv: bootstrap CIs (1000 resamples of CheXpert images) of the
    mean AUC over the 14 shared labels for DenseNet-169 and SwinTV2, and their
    paired difference, using the corrected label conversion of chexpertauc.py.
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


def load_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("chexpert_train_csv_path", type=str)
    parser.add_argument("labelnames_path", type=str)
    parser.add_argument("common_labelnames_path", type=str)
    parser.add_argument("hospital_path", type=str)
    parser.add_argument("preds_path", type=str)
    args = parser.parse_args()

    return args


def load_hospital(names32, hospital_path):
    labels = pd.read_csv(f"{hospital_path}/labels.csv", sep="\t", index_col="dicom_id")
    preds = {}
    for m in MODELS:
        p = pd.read_csv(f"{hospital_path}/{m}.csv", sep="\t", index_col="dicom_id")
        p = p.drop(columns=["Assumed Color Scheme", "Assumed Bits Stored"])
        p = p[~p.index.duplicated(keep="first")]
        p.columns = names32
        preds[m] = p
    idx = sorted(set(labels.index).intersection(*[set(p.index) for p in preds.values()]))
    return labels.loc[idx], {m: p.loc[idx] for m, p in preds.items()}


CHEXPERT_NAMES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
CHEXPERT_CONVERSION = {
    0: [31], 1: [8, 9, 10], 2: [8, 9, 10], 3: [1, 2], 4: [1, 2], 5: [1], 6: [1, 2, 5],
    7: [1, 2, 4], 8: [1, 2, 6], 9: [14, 15, 17], 10: [14, 15, 16], 11: [14], 12: [14, 15], 13: [22],
}
MODELS = ["teacherdn169", "swintv2", "deit", "convnext"]


def chexpert_labels(csv_path):
    """Same filtering as chexpertauc.py; a category is positive if any mapped finding is positive."""
    df = pd.read_csv(csv_path)
    df = df[df["Frontal/Lateral"] == "Frontal"]
    df = df[df["AP/PA"].isin(["AP", "PA"])]
    df = df[df["Age"] >= 18]
    raw = df[CHEXPERT_NAMES].to_numpy(dtype=float)
    out = np.full((len(raw), 32), -1)
    for position, targets in CHEXPERT_CONVERSION.items():
        col = raw[:, position]
        pos = col == 1
        neg = (col == 0) | ((position == 0) & np.isnan(col))
        for t in targets:
            cur = out[:, t]
            cur[neg & (cur == -1)] = 0
            cur[pos] = 1
            out[:, t] = cur
    return out


def chexpert_bootstrap(chexpert_train_csv_path, names32, common, preds_path):
    labels = chexpert_labels(chexpert_train_csv_path)
    idx = [names32.index(c) for c in common]
    preds = {m: torch.load(f"{preds_path}{m}/inferencedata/chexpert_preds.pt", weights_only=True).numpy()
             for m in ("teacherdn169", "swintv2")}
    rng = np.random.default_rng(0)
    n = len(labels)
    boot = {m: [] for m in preds}
    for _ in range(1000):
        r = rng.integers(0, n, n)
        lab = labels[r]
        for m, p in preds.items():
            pr = p[r]
            vals = []
            for j in idx:
                mask = lab[:, j] != -1
                vals.append(roc_auc_score(lab[mask, j], pr[mask, j]))
            boot[m].append(np.mean(vals))
    for m, v in boot.items():
        print(f"CheXpert mean AUC 95% CI {m}: [{np.percentile(v, 2.5):.4f}, {np.percentile(v, 97.5):.4f}]")
    d = np.array(boot["teacherdn169"]) - np.array(boot["swintv2"])
    print(f"CheXpert paired bootstrap teacherdn169 - swintv2: {d.mean():+.4f} "
          f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")


def main():

    # Load args
    args = load_args()

    # Load labelnames
    names32 = json.load(open(args.labelnames_path))
    common = json.load(open(args.common_labelnames_path))

    # Load hospital data
    y_h, p_h = load_hospital(names32, args.hospital_path)
    print(f"Hospital images: {len(y_h)}")

    # Paired bootstrap over hospital images
    rng = np.random.default_rng(0)
    n = len(y_h)
    yv = {c: y_h[c].to_numpy() for c in common}
    pv = {m: {c: p_h[m][c].to_numpy() for c in common} for m in MODELS}
    diffs = {("teacherdn169", "swintv2"): [], ("convnext", "teacherdn169"): []}
    for _ in range(1000):
        r = rng.integers(0, n, n)
        mean_auc = {m: np.mean([roc_auc_score(yv[c][r], pv[m][c][r]) for c in common]) for m in MODELS}
        for a, b in diffs:
            diffs[(a, b)].append(mean_auc[a] - mean_auc[b])
    for (a, b), d in diffs.items():
        d = np.array(d)
        print(f"Paired bootstrap {a} - {b}: {d.mean():+.4f} "
              f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")

    chexpert_bootstrap(args.chexpert_train_csv_path, names32, common, args.preds_path)


if __name__ == "__main__":
    main()