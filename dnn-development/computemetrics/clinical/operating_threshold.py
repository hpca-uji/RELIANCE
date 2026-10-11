import argparse
import json
import os
from typing import List
from statistics import mean

import torch
from torchmetrics import FBetaScore, StatScores
import pandas as pd


def load_args():

    """Parameter loading"""
    parser = argparse.ArgumentParser()
    parser.add_argument("internal_labels_path", type=str)
    parser.add_argument("internal_preds_path", type=str)
    parser.add_argument("hospital_path", type=str)
    parser.add_argument("labelnames_path")
    parser.add_argument("--models_idfs", type=str, nargs="+")
    parser.add_argument("--beta", type=float, default=2)
    args = parser.parse_args()

    return args


def load_data(
    internal_labels_path: str,
    internal_preds_path: str,
    hospital_path: str, 
    labelnames_path: str,
    models_idfs: List[str],
):
    
    # Load internal validation labels
    internal_vl_labels = torch.load(os.path.join(internal_labels_path, "vl_label.pt"))

    # Load internal validation preds
    internal_vl_preds = dict()
    for idf in models_idfs:
        load_path = os.path.join(internal_preds_path, idf, "inferencedata/", "vl_preds.pt")
        internal_vl_preds[(idf)] = torch.load(load_path)

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

    # Load HUGCS data
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


    return internal_vl_labels, internal_vl_preds, internal_ts_labels, internal_ts_preds, hospital_labels, hospital_preds, labelnames


def compute_best_multith(labels, preds, beta):

    # Create array to store all values
    selected_th = dict()

    # Compute number of labels
    nlabels = labels.shape[1]

    # Foreach model compute best th
    for model_idf, model_preds in preds.items():
    
        # Create list of equidistand thresholds
        th_list = [i/200 for i in range(200)]

        label_th = []

        # Foreach label compute best th
        for i in range(nlabels):

            # Save best fb score and threshold for it
            best_fb = 0
            best_th = 0

            # Iterate over thlist
            for th in th_list:

                # Create metric
                fbeta = FBetaScore(task="binary", beta=beta, threshold=th, average="macro", num_labels=1)

                # Compute metric and append
                score_fb = fbeta(model_preds[:,i], labels[:,i]).item()

                # Update best
                if score_fb > best_fb:
                    best_th = th
                    best_fb = score_fb
            # Add to list
            label_th.append(best_th)
        
        # Add to dict
        selected_th[model_idf] = label_th
    
    return selected_th


# Compute stats for each label
def compute_stats_summary(labelnames, labels, preds, id, thlist):

    sensitivies = []
    specificitys = []
    fnrs = []
    fprs = []
    space = " "

    print(f"{id:<36} th {space*3} sens {space*3} spec {space*3} fnr {space*3} fpr")

    for i, th in enumerate(thlist):

        counts = labels[:,i].sum()
        if counts < 10:
            continue

        # Define metric
        m_stats = StatScores(task="binary", threshold=th)

        # Compute metric
        stats = m_stats(preds[:,i], labels[:,i])

        # Read disease
        label = labelnames[i]

        # Read stats and convert to item
        tp = stats[0].item()
        fp = stats[1].item()
        tn = stats[2].item()
        fn = stats[3].item()
        sup = stats[4].item()

        # Compute stats
        sensitivity = tp/sup
        sensitivies.append(sensitivity)

        specificity = tn/(tn+fp)
        specificitys.append(specificity)

        fnr = fn/sup
        fnrs.append(fnr)

        fpr = fp/(tn+fp)
        fprs.append(fpr)

        # Print
        print(f"{label:<35}{th:.3f}\t{sensitivity:.3f}\t{specificity:.3f}\t{fnr:.3f}\t{fpr:.3f}")

    mean_sensitivity = mean(sensitivies)

    mean_specificity = mean(specificitys)

    mean_fnr = mean(fnrs)

    mean_fpr = mean(fprs)

    # Print
    print(f"Media {space*29}{mean_sensitivity:.3f}\t{mean_specificity:.3f}\t{mean_fnr:.3f}\t{mean_fpr:.3f}")
    print()


def main():

    # Load args
    args = load_args()

    # Load data
    internal_vl_labels, internal_vl_preds, internal_ts_labels, internal_ts_preds, hospital_labels, hospital_preds, labelnames = load_data(internal_labels_path=args.internal_labels_path, 
                                                                                          internal_preds_path=args.internal_preds_path,
                                                                                          hospital_path=args.hospital_path, 
                                                                                          models_idfs=args.models_idfs, 
                                                                                          labelnames_path=args.labelnames_path)

    # Compute best threshold foreach model
    selected_th = compute_best_multith(labels=internal_vl_labels, preds=internal_vl_preds, beta=args.beta)

    # Compute metrics over Hospital
    print("Hospital metrics")
    for model_idf, model_preds in hospital_preds.items():
        compute_stats_summary(labelnames=labelnames, labels=hospital_labels, preds=model_preds, id=model_idf, thlist=selected_th[model_idf])

    print("Test metrics")
    # Compute metrics over test data using validation thresholds
    for model_idf, model_preds in internal_ts_preds.items():
        compute_stats_summary(labelnames=labelnames, labels=internal_ts_labels, preds=model_preds, id=model_idf, thlist=selected_th[model_idf])
    



if __name__ == "__main__":
    main()