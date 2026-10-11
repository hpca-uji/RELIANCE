import json
import sys
import pandas as pd
import argparse
import re


# Function that process each report individually to improve quality
def process_report(report):

    # Check if report is nan
    if isinstance(report, float):
        return report

    # Check if report is empty
    if report == "":
        return report

    # Delete space at beggining and end of sentence
    report = report.strip()

    # Delete multiple consecutive spaces into one space
    report = re.sub(" +", " ", report)

    # Delete multiple consecituve dot and spaces
    report = re.sub("(\. )+", ". ", report)

    # Check if report is empty after previous processing
    if report == "":
        return report

    # If starting with a dot delete first 3 chars
    if report[0] == ".":
        report = report[2:]

    return report


if __name__ == "__main__":

    # Read parameters
    parser = argparse.ArgumentParser(description="Create labeled database")
    parser.add_argument("csv_path", type=str, help="Path of the original padchest csv")
    parser.add_argument(
        "report_path", type=str, help="Path of the original reports csv"
    )
    parser.add_argument(
        "save_path", type=str, help="Path where the processed csv will be stored"
    )
    args = parser.parse_args()

    # Read csv
    data = pd.read_csv(args.csv_path)
    report = pd.read_csv(args.report_path)

    # Discard nans on report dataframe
    report = report.dropna(subset=["Report_full"]).reset_index(drop=True)

    # Drop nan in labels and print stats
    original_length = len(data)
    data = data.dropna(subset=["Labels"]).reset_index(drop=True)
    data = data[data["Labels"] != "['']"].reset_index(
        drop=True
    )  # Some labels are empty
    new_length = len(data)
    print(
        f"{new_length} rows remain from de original dataset of {original_length} initial rows by deleting {original_length-new_length} rows that did not contain any labeling information"
    )

    # Compute Age column
    data["StudyDate_DICOM"] = data["StudyDate_DICOM"].apply(
        str
    )  # Convert to string to extract year
    data["PatientAge"] = (
        data["StudyDate_DICOM"].str[:4].apply(float) - data["PatientBirth"]
    )

    # Create new full report column
    data = data.merge(report, how="left", on="ReportID")

    # Keep these columns
    data = data[
        [
            "Report_full",
            "Report",
            "Labels",
            "MethodLabel",
            "ImageID",
            "ImageDir",
            "PatientID",
            "PatientAge",
            "PatientSex_DICOM",
            "Projection",
        ]
    ]

    # Process Report full and Report columns
    data["Report_full"] = data["Report_full"].apply(process_report)
    data["Report"] = data["Report"].apply(process_report)

    # Process labeling column
    label_column = data["Labels"].tolist()
    processed_label_column = []

    for label in label_column:

        # Eliminar espacios al principio de las etiquetas
        new_label = label.replace("' ", "'")
        # Eliminar etiquetas vacías , '', - ['', ] - , '']
        new_label = new_label.replace("'', ", "")
        # Eliminar etiquetas vacías al final
        new_label = new_label.replace(", ''", "")
        # Eliminar comillas simples
        new_label = new_label.replace("'", "")
        # Sustituir , por |
        new_label = new_label.replace(", ", "|")
        # Eliminar corchetes
        new_label = new_label.replace("[", "").replace("]", "")
        # Añadir al array
        processed_label_column.append(new_label)

    data["Labels"] = processed_label_column
    data.to_csv(args.save_path, index=False)
