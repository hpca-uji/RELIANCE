import argparse
import json

from utils.loaders import load_labels, load_preds
from utils.aucops import auc_calculator


def load_args():
    """Parameter loading"""
    parser = argparse.ArgumentParser()
    parser.add_argument("label_path", type=str)
    parser.add_argument("models_dir", type=str)
    parser.add_argument("labelnames_path")
    parser.add_argument("output_file")
    parser.add_argument("--models_idfs", type=str, nargs="+")
    parser.add_argument("--partitions", type=str, nargs="+")
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.95)
    args = parser.parse_args()

    return args


def parse_dict(dict):
    dict_string = {str(key): value for key, value in dict.items()}
    return dict_string


def main():

    # Load args
    args = load_args()

    # Load labelnames
    with open(args.labelnames_path, "r") as file:
        labelnames = json.load(file)

    # Load data
    label = load_labels(label_path=args.label_path)
    preds = load_preds(models_dir=args.models_dir, models_idfs=args.models_idfs, partitions=args.partitions)

    # Compute data
    aucs = auc_calculator(
        label=label, preds=preds, labelnames=labelnames, n_bootstraps=args.bootstraps, alpha=args.alpha
    )

    # Convert keys to string
    aucs = parse_dict(aucs)

    # Save data
    with open(args.output_file, "w") as file:
        json.dump(aucs, file)


if __name__ == "__main__":
    main()
