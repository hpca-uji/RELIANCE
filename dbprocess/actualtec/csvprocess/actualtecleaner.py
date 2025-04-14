import os
import re
import argparse

import pydicom as pyd
import pandas as pd


def load_args():
    """Parameter loading"""
    # Create parser object
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("csv_path", type=str)

    # Parse parameters
    args = parser.parse_args()

    return args


def main():

    # Load args
    args = load_args()

    # Create empty lists to create dataset
    report_list = []
    findings_list = []
    diagnostic_list = []
    projection_list = []
    imageid_list = []

    # List all folders inside the dataset folder given as parameters
    content = os.listdir(args.dataset_path)
    # Remove folder14
    content.remove("anonc14")

    # Age counters
    no_age_counter = 0
    under_age_counter = 0
    total_counter = 0

    # List of words that make the report invalid
    error_array = [
        "error envío estudio",
        "https://",
        "addendum",
        "pie",
        "mano",
        "muñeca",
        "rodilla",
        "brazo",
        "fémur",
        "femur",
        "mnp",
        "facial",
        "parrilla",
        "costal",
        "cervical",
        "cráneo",
        "craneo",
        "cráneo",
        "pierna",
        "hpn",
        "col.",
        "spn",
        "hombro",
        "humero",
        "abdomen",
        "oseo",
        "cavum",
        "calle",
        "maxilar",
        "columna",
        "nasales",
    ]

    # Ways that findings are specified in the dataset
    findings_pattern = [
        "hallazgos",
        "comentario",
        "encontrando",
        "==",
        "descripción",
        "descripcion",
        "se observa lo siguiente",
        "rx torax frente",
        "rx tórax frente",
        "rx torax",
        "radiografía de tórax frente",
        "radiografia de torax frente",
        "rx digital de torax (f)",
        "rx digital de torax (f y p)",
        "rx digital de torax (f-p)",
        "rx digital torax (f-p)",
        "rx digital torax (fp)",
        "rx digital de torax",
        "rx de tóraxdatos clínicos:",
        "rx de tórax datos clínicos:",
        "radiografía de torax f",
        "radiografía de tórax oblicuada portátil.",
        "radiografía de tórax portátil",
        "rayos x portátil en proyección ",
        "radiografía de tórax",
        "rayos x de tórax",
        "rx. tórax",
        "rx tórax",
        "informe:en el estudio realizado se observa:calidad del estudio: estudio radiológico de buena calidad con correcta exposición y sin artefactos detectables.",
        "informe:en el estudio realizado se observa:",
        "estudio de prueba",
        "fecha",
        "torax:",
    ]

    # Ways that diagnostic is specified
    diags_pattern = [
        "impresión diagnóstica",
        "impresión diagóstica",
        "impresión dx:",
        "impresión",
        "impresion",
        "impresion diagnostica",
        "i.d.",
        "diagnostico",
        "diagnostica",
        "conclusiones",
        "conclusión",
        "conclusion",
        "conlusión",
        "opinion",
        "opinión",
        "médico radiólogo",
        "conclucion",
        "conclución" "conclusiones",
    ]

    # Characters and patterns to remove from the initial report
    init_delpatt_array = [
        "[",
        "]",
        "correlacion clinica, de ser necesario estudios complementarios",
    ]

    # Characters and patterns to remove from the final report
    delpatt_array = ["-", ":", "atentamente." "atentamente"]

    # Iterate over anonc folders
    for anonc in content:

        print(anonc, flush=True)

        # Foreach study in the folder
        studies = os.listdir(os.path.join(args.dataset_path, anonc))
        for study in studies:

            # Reset control vars
            imageID = None
            findings = None
            diagnostic = None
            view_val = None
            error = False
            containsfp = False

            # Read all files of the folder study
            study_files = os.listdir(os.path.join(args.dataset_path, anonc, study))
            for study_f in study_files:

                # If the file is a dicom
                if study_f[0] == "C" or study_f[0] == "D":

                    # Read the dicom
                    ds = pyd.dcmread(
                        fp=os.path.join(args.dataset_path, anonc, study, study_f),
                        specific_tags=[
                            "0x00101010",
                            "0x00180015",
                            "0x00185101",
                            "0x20500020",
                            "0x00280101",
                        ],
                    )

                    # Access dicom fields
                    age = ds.get(0x00101010)
                    localization = ds.get(0x00180015)
                    view = ds.get(0x00185101)

                    # Check if dicom is PA projection
                    if (
                        localization is not None
                        and localization.value
                        in ("CHEST", "TORAX", "THORAX", "PECHO", "CHEST PA LAT")
                        and view is not None
                        and view.value
                        in (
                            "PA",
                            "PA LANDSCAPE",
                            "PA PORTRAIT",
                            "AP",
                            "AP LANDSCAPE",
                            "AP PORTRAIT",
                        )
                    ):
                        imageID = os.path.join(anonc, study, study_f + ".pkt")
                        view_val = view.value

                elif study_f[0] == "1":

                    # Read file in lower case
                    filepath = os.path.join(args.dataset_path, anonc, study, study_f)
                    f = open(filepath, "r")
                    report = f.read().lower()

                    # Delete undesired patterns
                    for pattern in init_delpatt_array:
                        report = report.replace(pattern, "")
                    report = (
                        report.replace("\n", " ")
                        .replace("\r", "")
                        .replace("\u2028", " ")
                    )

                    # Replace consecutive = by only ==
                    report = re.sub(r"(=)\1+", r"\1\1", report)

                    # If the report contains any string in the error array discard
                    for e in error_array:
                        if e in report:
                            error = True
                            break
                    if error:
                        break

                    # If report is short discard
                    if len(report) < 20:
                        break

                    # Detect findings section
                    for fp in findings_pattern:
                        # If finding pattern is contained within the report
                        if fp in report:

                            # Mark finding
                            containsfp = True
                            # Split by the pattern
                            split = report.split(fp)[1]
                            # If longer than 10 chars assign to findings
                            if len(split) >= 6:
                                findings = split
                            break

                    if findings is None:
                        if containsfp:  # Contained fp but text was empty --> discard
                            break
                        else:  # Did not contain an fp so everything is the finding
                            findings = report

                    # Detect diagnostic section within the findings
                    for dp in diags_pattern:
                        # If diagnostic pattern is contained within the findings
                        if dp in findings:
                            # Split by the pattern
                            splits = findings.split(dp)
                            # If longer than 6 chars assign to diagnostic
                            if len(splits[1]) >= 6:
                                diagnostic = splits[1]
                            findings = splits[0]
                            break

                    # If diagnostic is None asign findings
                    if diagnostic is None:
                        diagnostic = findings

                    # Delete undesired patterns
                    if diagnostic is not None:
                        for pattern in delpatt_array:
                            diagnostic = diagnostic.replace(pattern, "")
                    for pattern in delpatt_array:
                        findings = findings.replace(pattern, "")

                    # Delete Dr, Dra if exists
                    if diagnostic is not None:
                        if "dr" in diagnostic:
                            diagnostic = diagnostic.split("dr")[0]
                        elif "dra" in diagnostic:
                            diagnostic = diagnostic.split("dra")[0]
                    if "dr" in findings:
                        findings = findings.split("dr")[0]
                    elif "dra" in findings:
                        findings = findings.split("dra")[0]

            if imageID is not None and findings is not None:

                # Check dicom age
                if age is None or len(age.value) <= 1:
                    no_age_counter += 1
                    continue
                elif int(age.value[:-1]) < 16:
                    under_age_counter += 1
                    continue
                total_counter += 1

                # Add to lists
                report_list.append(report)
                findings_list.append(findings)
                diagnostic_list.append(diagnostic)
                projection_list.append(view_val)
                imageid_list.append(imageID)

    # Create dataframe and save
    df = pd.DataFrame(
        list(
            zip(
                report_list,
                findings_list,
                diagnostic_list,
                projection_list,
                imageid_list,
            )
        ),
        columns=["Report", "Findings", "Diagnostic", "Projection", "ImageID"],
    )
    df.to_csv(args.csv_path)

    # Print stats
    print(f"No age: {no_age_counter}")
    print(f"Under age: {under_age_counter}")
    print(f"Total count: {total_counter}")


if __name__ == "__main__":
    main()
