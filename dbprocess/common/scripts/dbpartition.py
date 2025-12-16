# Imports
import pandas as pd
import numpy as np
import os
import torch
import argparse
import json
from sklearn.model_selection import GroupShuffleSplit


def load_args():
    """Parameter loading"""
    parser = argparse.ArgumentParser(description="Create labeled database")
    parser.add_argument(
        "save_path", type=str, help="Path where the label data will be stored"
    )
    parser.add_argument(
        "labelnames_path", type=str, help="Path where the label list is stored"
    )
    parser.add_argument(
        "--path1", type=str, help="Path of a labeled dataset csv", default=None
    )
    parser.add_argument(
        "--path2", type=str, help="Path of another labeled dataset csv", default=None
    )
    parser.add_argument(
        "--discard", type=int, help="Discard under certain years old", default=1
    )
    parser.add_argument(
        "--rm_path",
        "-remove_path",
        type=str,
        help="Optional path of filenames from padchest to be deleted",
        default=None,
    )
    parser.add_argument(
        "--previous_labeling",
        type=str,
        help="Optional path of a previous labeling to copy the partitions",
        default=None,
    )
    parser.add_argument("--seed", type=int, help="Seed used in splitters", default=42)

    args = parser.parse_args()

    # Check that at least one csv is provided
    if args.path1 is None and args.path2 is None:
        parser.error("At least one csv must be provided as input")

    return args


def check_save_folder(save_path):
    """Create save_path folder if it does not exist"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        print("Label folder already exists. Use another name or delete the folder")
        exit(-1)


def compute_triple_proportion_value(y_tr, y_ts, y_vl):
    """Computes proportion values and returns the difference with the proportion on tr, vl, ts"""
    # Compute proportions of each label
    proportions_tr = y_tr.sum(axis=0) / len(y_tr)
    proportions_ts = y_ts.sum(axis=0) / len(y_ts)
    proportions_vl = y_vl.sum(axis=0) / len(y_vl)

    # Proportions should be similar among the three groups
    diff1 = abs(proportions_tr - proportions_ts).sum()
    diff2 = abs(proportions_tr - proportions_vl).sum()
    total_diff = diff1 + diff2

    return total_diff

def compute_double_proportion_value(y_tr, y_vl):
    """Computes proportion values and returns the difference with the proportion on tr, vl"""
    # Compute proportions of each label
    proportions_tr = y_tr.sum(axis=0) / len(y_tr)
    proportions_vl = y_vl.sum(axis=0) / len(y_vl)

    # Proportions should be similar among the three groups
    diff = abs(proportions_tr - proportions_vl).sum()

    return diff


def get_xyproperties(df, numeric_columns):
    """Gets image, labels and properties of dicom"""
    X = df["ImageID"].to_numpy(dtype=object)
    y = df[numeric_columns].to_numpy(dtype=np.float32)
    properties = df[["PatientAge", "PatientSex_DICOM", "Projection"]].to_numpy(
        dtype=object
    )
    return [X, y, properties]


def create_label_partition(best, save_path, partition):
    """Creates and saves partition data"""
    # Save paths
    np.save(os.path.join(save_path, partition + "_paths.npy"), best[0])

    # Save label
    label = torch.ByteTensor(best[1])
    torch.save(label, os.path.join(save_path, partition + "_label.pt"))

    # Save properties
    np.save(os.path.join(save_path, partition + "_properties.npy"), best[2])


def print_distribution_stats(df, header):
    """Prints distribution stats of the dataframe under a given header"""
    # If empty do not print anything
    if len(df) == 0:
        return

    # Print header
    print(f"{header}: {len(df)} counts")

    # Compute numeric columns
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    # Remove not needed columns
    numeric_columns.remove("PatientAge")
    if "ImageDir" in numeric_columns:
        numeric_columns.remove("ImageDir")
    elif "Follow-up #" in numeric_columns:
        numeric_columns.remove("Follow-up #")
        numeric_columns.remove("OriginalImage[Width")
        numeric_columns.remove("Height]")
        numeric_columns.remove("OriginalImagePixelSpacing[x")
        numeric_columns.remove("y]")

    # Compute and print distribution
    distribution = df[numeric_columns].sum(numeric_only=True)
    print(distribution)
    print()


def print_stats(df, header):
    """Prints all stats of dataframe under the given header"""
    # Print header
    print()
    print(f"{header}: {len(df)} counts")
    print()

    # Print unknown counts
    UNK_GENDER = len(df[df["PatientSex_DICOM"] == "UNK"])
    UNK_AGE = len(df[df["PatientAge"] == 200])
    UNDER_AGE = len(df[df["PatientAge"] < 16])
    print(f"UNK_GENDER: {UNK_GENDER}")
    print(f"UNK_AGE: {UNK_AGE}")
    print(f"UNDER_AGE: {UNDER_AGE}")

    # Distribution stats
    print_distribution_stats(df, "total")

    # Compute AP, AP, F, M, chestx, padchest number of studies
    print_distribution_stats(df[df["Projection"] == "PA"], "PA")
    print_distribution_stats(df[df["Projection"] == "AP"], "AP")
    print_distribution_stats(df[df["Projection"] == "AP_horizontal"], "AP_horizontal")
    print_distribution_stats(df[df["PatientSex_DICOM"] == "F"], "F")
    print_distribution_stats(df[df["PatientSex_DICOM"] == "M"], "M")

    # Compute number of studies of each dataset
    if header not in ("padchest", "chestx"):
        padchest_counts = len(df[df["ImageID"].str.contains("padchest")])
        chestx_counts = len(df[df["ImageID"].str.contains("chestx")])
        print(f"padchest counts: {padchest_counts}")
        print(f"chestx counts: {chestx_counts}")


def process_df(df, discard, header):
    """Common function of dataframe processing and cleaning"""
    # Fill nan values in gender and age
    df["PatientSex_DICOM"] = df["PatientSex_DICOM"].fillna("UNK")
    df["PatientAge"] = df["PatientAge"].fillna(200).astype(int)

    # Discard data from df with DiscardCNN = 1, Discard = 1 or Exclude
    df = df[df["DiscardCNN"] == 0]
    df = df[df["Discard"] == 0]
    df = df[df["Exclude"] == 0]

    # Reset index positions
    df = df.reset_index(drop=True)

    # Change png to pt
    df["ImageID"] = df["ImageID"].str.replace(".png", ".pkt", regex=False)

    # Add header identifier to PatientID and ImageID data
    df["PatientID"] = header + df["PatientID"].astype(str)
    df["ImageID"] = header + df["ImageID"].astype(str)

    # Drop Discard and DiscardCNN
    df = df.drop(labels=["Discard", "DiscardCNN", "Exclude", "Unchanged"], axis=1)

    # Discard underage if indicated
    if discard == 1:
        df = df[df["PatientAge"] >= 16].reset_index(drop=True)

    # Print stats
    print_stats(df, header)

    return df


def process_padchest(df, discard, rm_path):
    """Function used to process padchest only"""
    # Keep PA, AP, AP_horizontal projections
    df = df[df["Projection"].isin(["PA", "AP", "AP_horizontal"])]

    # Delete corrupted images
    if rm_path is not None:
        rm_file = open(rm_path, "r")
        for line in rm_file:
            df = df[df["ImageID"] != line.strip()]

    # Common processing
    df = process_df(df, discard, "padchestPKT/")

    return df


def process_chestx(df, discard):
    """Function to process chestx only"""
    df = process_df(df, discard, "chestxPKT/")

    return df


def main():

    # Initialization
    args = load_args()
    check_save_folder(args.save_path)

    # Process padchest if exists
    if args.path1 is not None:
        # Read and process padchest
        padchest_df = pd.read_csv(args.path1)
        padchest_df = process_padchest(padchest_df, args.discard, args.rm_path)

    # Process chestx if exists
    if args.path2 is not None:
        # Read and process padchest
        chestx_df = pd.read_csv(args.path2)
        chestx_df = process_chestx(chestx_df, args.discard)

    # Combine dfs
    if args.path2 is not None and args.path1 is not None:
        combined_df = pd.concat(objs=[padchest_df, chestx_df], axis=0)
        combined_df = combined_df.dropna(axis=1)
        combined_df = combined_df.reset_index(drop=True)
        # Print combined_df stats
        print_stats(df=combined_df, header="combined")
    elif args.path2 is not None:
        combined_df = chestx_df.copy(deep=True)
    else:
        combined_df = padchest_df.copy(deep=True)

    # Read label
    with open(args.labelnames_path) as f:
        labelnames = json.load(f)

    # Create new partition
    if args.previous_labeling is None:

        # Split into paths, labels, properties
        savedata = get_xyproperties(combined_df, labelnames)
        X = savedata[0]
        y = savedata[1]
        properties = savedata[2]
        groups = combined_df["PatientID"].to_numpy(dtype=object)

        # Split into tr, vl and ts
        spliter1 = GroupShuffleSplit(
            n_splits=100, train_size=0.7, random_state=args.seed
        )
        spliter2 = GroupShuffleSplit(
            n_splits=100, train_size=2 / 3, random_state=args.seed
        )

        # Placeholder vars
        best_tr = None
        best_ts = None
        best_vl = None
        best_proportion = float("infinity")

        # Try several splits
        for tr_idx, rest_idx in spliter1.split(X, y, groups):

            # Compute rest data
            X_rest = X[rest_idx]
            y_rest = y[rest_idx]
            groups_rest = groups[rest_idx]

            for ts_idx, vl_idx in spliter2.split(X_rest, y_rest, groups_rest):

                # Compute y for the three splits
                y_tr = y[tr_idx]
                y_ts = y_rest[ts_idx]
                y_vl = y_rest[vl_idx]

                # Check proportion value and minimize
                value = compute_triple_proportion_value(y_tr, y_ts, y_vl)
                if value < best_proportion:
                    best_tr = [X[tr_idx], y[tr_idx], properties[tr_idx]]
                    best_ts = [X_rest[ts_idx], y_rest[ts_idx], properties[ts_idx]]
                    best_vl = [X_rest[vl_idx], y_rest[vl_idx], properties[vl_idx]]
                    best_proportion = value
    elif os.path.isdir(args.previous_labeling):

        # We read a folder with all paths for each partition
        # Read paths
        tr_paths = np.load(
            os.path.join(args.previous_labeling, "tr_paths.npy"), allow_pickle=True
        )
        vl_paths = np.load(
            os.path.join(args.previous_labeling, "vl_paths.npy"), allow_pickle=True
        )
        ts_paths = np.load(
            os.path.join(args.previous_labeling, "ts_paths.npy"), allow_pickle=True
        )
        # Get slices
        tr_df = combined_df[combined_df["ImageID"].isin(tr_paths)]
        vl_df = combined_df[combined_df["ImageID"].isin(vl_paths)]
        ts_df = combined_df[combined_df["ImageID"].isin(ts_paths)]
        # Get savedata
        best_tr = get_xyproperties(tr_df, labelnames)
        best_vl = get_xyproperties(vl_df, labelnames)
        best_ts = get_xyproperties(ts_df, labelnames)
    else:

        # We read a file with ts paths by line
        # Read file by line to get each ts path
        ts_paths_file = open(args.previous_labeling, mode="r")
        ts_paths = []
        for line in ts_paths_file:
            ts_paths.append("chestxPKT/" + line.strip().replace(".png",".pkt")) # File with ts paths is only chestxray14 processing case
        # Convert to numpy
        ts_paths = np.array(ts_paths)

        # Split combined df
        trvl_df = combined_df[~combined_df["ImageID"].isin(ts_paths)]
        ts_df = combined_df[combined_df["ImageID"].isin(ts_paths)]

        # Final preparation for ts data
        best_ts = get_xyproperties(ts_df, labelnames)
    
        # Split tr/vl into paths, labels, properties
        savedata = get_xyproperties(trvl_df, labelnames)
        X = savedata[0]
        y = savedata[1]
        properties = savedata[2]
        groups = trvl_df["PatientID"].to_numpy(dtype=object)

        # Create splitter
        spliter = GroupShuffleSplit(
            n_splits=1000, train_size=0.9, random_state=args.seed
        )

        # Placeholder vars
        best_tr = None
        best_vl = None
        best_proportion = float("infinity")

        # Try several splits
        for tr_idx, vl_idx in spliter.split(X, y, groups):
            
            # Split y
            y_tr = y[tr_idx]
            y_vl = y[vl_idx]

            # Check proportion value and minimize
            value = compute_double_proportion_value(y_tr, y_vl)
            if value < best_proportion:
                best_tr = [X[tr_idx], y[tr_idx], properties[tr_idx]]
                best_vl = [X[vl_idx], y[vl_idx], properties[vl_idx]]
                best_proportion = value        
        
    # Show partition_stats
    print_stats(combined_df[combined_df["ImageID"].isin(best_tr[0])], "tr")
    print_stats(combined_df[combined_df["ImageID"].isin(best_ts[0])], "ts")
    print_stats(combined_df[combined_df["ImageID"].isin(best_vl[0])], "vl")

    # Save the partitions
    create_label_partition(best_tr, args.save_path, "tr")
    create_label_partition(best_ts, args.save_path, "ts")
    create_label_partition(best_vl, args.save_path, "vl")


if __name__ == "__main__":

    main()
