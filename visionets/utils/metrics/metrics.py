from torchmetrics.functional import auroc
import torch
import argparse
import os
import json
import numpy as np
from statistics import mean


def load_args():
    """Parameter loading"""
    parser = argparse.ArgumentParser()
    parser.add_argument("label_path", type=str)
    parser.add_argument("list_path", type=str)
    parser.add_argument("preds_path", type=str)

    args = parser.parse_args()

    return args

def load_data(args):
    """Load validation and test labels, predictions and label name list"""

    # Load labeld and predictions
    with open(os.path.join(args.label_path, "vl_label.pt"), "rb") as f:
        vl_label = torch.load(f)
    with open(os.path.join(args.label_path, "ts_label.pt"), "rb") as f:
        ts_label = torch.load(f)
    with open(os.path.join(args.preds_path, "vl_preds.pt"), "rb") as f:
        vl_preds = torch.load(f)
    with open(os.path.join(args.preds_path, "ts_preds.pt"), "rb") as f:
        ts_preds = torch.load(f)
    with open(os.path.join(args.label_path, "vl_properties.npy"), "rb") as f:
        vl_prop = np.load(f, allow_pickle=True)
    with open(os.path.join(args.label_path, "ts_properties.npy"), "rb") as f:
        ts_prop = np.load(f, allow_pickle=True)

    with open(args.list_path) as f:
        disease_list = json.load(f)

    return np.array(vl_label), np.array(vl_preds), np.array(ts_label), np.array(ts_preds), vl_prop, ts_prop, disease_list

def false_positives_negatives(label, preds, th=0.5):
    """Compute fnr and fpr"""
    total_positives = 0
    false_positives = 0
    total_negatives = 0
    false_negatives = 0
    for l, p in zip(label, preds):
        if l == 1:
            total_positives+=1
            if p < th:
                false_negatives+=1
        else:
            total_negatives+=1
            if p > th:
                false_positives+=1
    return false_negatives/total_positives*100, false_positives/total_negatives*100

def compute_stats(header, disease_list, label, preds, prop, th):
    """Compute auc, fnr and fpr over every label"""
    print()
    print(header)
    scores = []
    fnrs = []
    fprs = []
    # Compute auc per label
    for i in range(label.shape[1]):
        score = auroc(torch.Tensor(preds[:,i]), torch.Tensor(label[:,i]).int(), task="binary").item()
        fnr, fpr = false_positives_negatives(label[:,i], preds[:,i], th)
        scores.append(score)
        fnrs.append(fnr)
        fprs.append(fpr)
        print(f"{disease_list[i]}: {score}, {fnr}, {fpr}")
    print(f"mean: {mean(scores)}, {mean(fnrs)}, {mean(fprs)}")

    # Compute auc per sex and projection
    f_preds = []
    f_label = []
    m_preds = []
    m_label = []
    pa_preds = []
    pa_label = []
    ap_preds = []
    ap_label = []
    aph_preds = []
    aph_label = []

    for lbl, pred, pr in zip(label, preds, prop):
        if pr[1] == "F":
            f_preds.append(pred)
            f_label.append(lbl)
        elif pr[1] == "M":
            m_preds.append(pred)
            m_label.append(lbl)
        if pr[2] == "PA":
            pa_preds.append(pred)
            pa_label.append(lbl)
        elif pr[2] == "AP":
            ap_preds.append(pred)
            ap_label.append(lbl)
        elif pr[2] == "AP_horizontal":
            aph_preds.append(pred)
            aph_label.append(lbl)

    f_score = auroc(torch.Tensor(np.array(f_preds)), torch.Tensor(np.array(f_label)).int(), task="multilabel", num_labels=32).item()
    m_score = auroc(torch.Tensor(np.array(m_preds)), torch.Tensor(np.array(m_label)).int(), task="multilabel", num_labels=32).item()
    pa_score = auroc(torch.Tensor(np.array(pa_preds)), torch.Tensor(np.array(pa_label)).int(), task="multilabel", num_labels=32).item()
    ap_score = auroc(torch.Tensor(np.array(ap_preds)), torch.Tensor(np.array(ap_label)).int(), task="multilabel", num_labels=32).item()
    aph_score = auroc(torch.Tensor(np.array(aph_preds)), torch.Tensor(np.array(aph_label)).int(), task="multilabel", num_labels=32).item()

    print(f"F: {f_score}")
    print(f"M: {m_score}")
    print(f"PA: {pa_score}")
    print(f"AP: {ap_score}")
    print(f"APH: {aph_score}")




def main():

    args = load_args()
    vl_label, vl_preds, ts_label, ts_preds, vl_prop, ts_prop, disease_list = load_data(args)
    compute_stats(header="vl", disease_list=disease_list, label=vl_label, preds=vl_preds, prop=vl_prop, th=0.2)
    compute_stats(header="ts", disease_list=disease_list, label=ts_label, preds=ts_preds, prop=ts_prop, th=0.2)


if __name__ == "__main__":
    main()