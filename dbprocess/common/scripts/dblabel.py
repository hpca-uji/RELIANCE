# Imports
import pandas as pd
import numpy as np
import os
import argparse
import json


def load_args():

    # Read parameters
    parser = argparse.ArgumentParser(description="Creates a labeled dataset")
    parser.add_argument("csv_path", type=str, help="Path of the processed csv")
    parser.add_argument(
        "assigns_path", type=str, help="Path where disease dictionary is stored"
    )
    parser.add_argument(
        "labelnames_path", type=str, help="Path where label list is stored"
    )
    parser.add_argument(
        "save_path", type=str, help="Path where the labeled csv will be saved"
    )
    parser.add_argument(
        "--normalcode",
        type=str,
        help="Code used by dataset to indicate healthy patients",
    )
    args = parser.parse_args()

    return args


def load_data(args):

    # Read csv
    df = pd.read_csv(args.csv_path)

    # Read assigns
    with open(args.assigns_path) as f:
        assigns = json.load(f)

    # Read labelnames
    with open(args.labelnames_path) as f:
        labelnames = json.load(f)

    return df, assigns, labelnames


def create_label_matrix(df, assigns_positions, labelnames, normalcode):

    # Create empty set of diseases not present in the dictionary
    discarded = set()
    discarded_cls_rows = 0
    discarded_contradiction_rows = 0
    discarded_unchangedsolo_rows = 0
    label_matrix = np.zeros(shape=[len(df), len(labelnames)], dtype=np.int8)

    # Iterate over the df
    for i, row in df.iterrows():

        # Special states control
        varNormal = False
        varDisease = False
        varDetected = False
        varUnchanged = False
        varExclude = False

        # Read diseases from row
        diseases = row["Labels"].split("|")
        for disease in diseases:

            if disease == normalcode:
                varNormal = True
            elif disease == "exclude":
                varExclude = True
            elif disease == "unchanged":
                varUnchanged = True
            elif disease in assigns_positions:
                varDetected = True
                curr_assigns = assigns_positions[disease]
                if (
                    set([i for i in range(22)]) & curr_assigns
                ):  # There are elements to assign outside of normal compatibles
                    varDisease = True
                label_matrix[i, list(curr_assigns)] = 1
            else:
                discarded.add(disease)

        # Check special vars
        if varUnchanged:
            label_matrix[i, -4] = 1
        if varExclude:
            label_matrix[i, -3] = 1
        if varNormal:
            label_matrix[i, -5] = 1
        if not varDisease:
            label_matrix[i, -5] = 1

        # Check special var cases
        if (
            varUnchanged and not varDetected and not varNormal
        ):  # Unchanged but diagnostic is not present: discard for cnn
            label_matrix[i, -1] = 1
            discarded_unchangedsolo_rows += 1
        elif (
            varNormal and varDisease
        ):  # Both normal and disease are present (contradiction -> discard)
            label_matrix[i, -2] = 1
            discarded_contradiction_rows += 1
        elif (
            not varExclude and not varDetected and not varNormal
        ):  # No detection according to our classification
            label_matrix[i, -2] = 1
            discarded_cls_rows += 1

    # Print stats
    print(f"{discarded_contradiction_rows} rows were excluded for contradictions")
    print(
        f"{discarded_unchangedsolo_rows} rows were excluded since only unchanged was present"
    )
    print(
        f"{discarded_cls_rows} rows were excluded since did not contain any disease in the dictionary. The next diseases were discarded:"
    )
    print(discarded)

    return label_matrix


def main():

    # Load args
    args = load_args()

    # Load data
    df, assigns, labelnames = load_data(args)

    # Add Unchanged, Exclude, Discard and DiscardCNN
    labelnames.extend(["Unchanged", "Exclude", "Discard", "DiscardCNN"])

    # Create dictionary of label - positions
    label_pos = {labelnames[i]: i for i in range(len(labelnames))}
    # Create dictionary of disease - positions
    assigns_pos = {
        disease: {label_pos[assign] for assign in disease_assigns}
        for disease, disease_assigns in assigns.items()
    }
    print(assigns_pos)

    # Create label matrix
    label_matrix = create_label_matrix(df, assigns_pos, labelnames, args.normalcode)

    # Create pandas with the labels
    label_df = pd.DataFrame(data=label_matrix, columns=labelnames)

    # Concat dataframes
    df = pd.concat([df, label_df], axis=1)

    # Save to csv
    df.to_csv(args.save_path, index=False)


if __name__ == "__main__":

    main()
