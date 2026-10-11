# <REDACTED_PROJECT_NAME> DNN model development

# Requirements

This repository is based on Python 3.11.10 and PyTorch 2.4. The full environment used in the project can be installed through the following steps:

1. `pip install -r requirements.txt`
2. `cd visionets/visionapi`
3. `pip install -e .`

The last step will install (in editable mode) the `visionapi` library specifically developed for this project. This library is responsible for model creation, data loading, optimizer setup, and includes several implemented loss functions. Other thins to note:

- Example commands and configurations can be found in `scripts.sh`.
- Source data from datasets such as CheXpert, PadChest, ChestX-ray14 are not included in this repository and should be download from official sites.
- Hospital labels performed by radiologist are stored in `dbprocess/hospital`, together with multiple models predictions.
- Commands output are stored in progress.txt files which are included in `.gitignore` together with files generated from SLURM system.

# Project Structure

The repository is organized into three main components:

- **`computemetrics`**: Scripts to compute AUC and confidence intervals (CI).
- **`dbprocess`**: Scripts for processing datasets, including the private Actualtec dataset, ChestX-ray14, and PadChest. This module also includes utilities to generate training, validation, and test splits.
- **`modelpreds`**: Predictions for CheXpert and development dataset with all partitions (tr, vl, ts) in .pt format and Hospital cohort data in .csv format. teacherdn169 additionaly contains `precomputed` model outputs for distilled training
- **`visionets`**: Scripts for model training, inference, and export. These scripts rely on a custom-developed package, `visionapi`, and also support hyperparameter search over a predefined space.

## `computemetrics`

This folder contains the scripts used to evaluate model performance for the RELIANCE project. The utilities focus on area under the ROC curve (AUC), confidence intervals (CI), and clinically-oriented operating thresholds for multi-label chest X-ray classification. 

- `computeauc/`: Computes AUC and confidence intervals for internal validation, Hospital, and CheXpert evaluations.
    - `internalauc.py`: evaluates model AUC on internal train/validation/test partitions.
    - `hospitalauc.py`: evaluates AUC on the Hospital dataset.
    - `chexpertauc.py`: evaluates AUC on the CheXpert external benchmark.
    - `aucops.py`: contains the DeLong variance and confidence interval helper functions.
    - `results`: contains AUC results stored into JSON files for different datasets, models and labels.
        - `internal.json`: Performance over different splits of the development dataset PadChest + CX14 over the whole taxonomy of labels.
        - `interal_hospital.json`: Performance over different splits of the development dataset PadChest + CX14 over the restricted space of available labels at the Hospital.
        - `internal_common.json`: Performance over different splits of the development dataset PadChest + CX14 over the restricted space of available labels at the common intersection of CheXpert, development dataset and Hospital.
        - `hospital.json`: Performance over hospital cohort data over all available labels of the hospital cohort.
        - `hospital_common.json`: Performance over hospital cohort data over the restricted space of available labels at the common intersection of CheXpert, development dataset and Hospital.
        - `chexpert.json` and `cx14.json`: Self explanatory.

- `clinical/`: Estimates the operating threshold that maximizes multi-label F-beta performance and reports sensitivity/specificity metrics over test partition of the development dataset and hospital cohort data, with thresholds adjusted using validation split of the development dataset.
    - `operating_performance.txt`: Clinical operating point results

The code in this folder is provided for quality assurance, to document and verify the correctness of the reported AUC values and clinical operating points. Metrics reported in the paper can be reproduced using this scripts together with files from `modelpreds` and `dbprocess`. 

## `dbprocess`

One of the most important directories is `common/trainlabel`, which contains multiple processed versions of dataset labels which are directly consumed by the training scripts:

- **`chestx14label`**: Training, validation, and test splits for the ChestX-ray14 dataset using the original 14-label schema.
- **`chestxlabel`**: Training, validation, and test splits for the ChestX-ray14 dataset using our 32-category pathology system.
- **`padchestlabel`**: Training, validation, and test splits for the PadChest dataset using our 32-category pathology system.
- **`urgencylabel`**: Training, validation, and test splits for the combined PadChest + ChestX-ray14 dataset using our 32-category pathology system.

Each of these dataset labels folders contains a `stats.txt` file that describes label prevalence and demographics of the different partitions, which are partially published with the paper.

Other important files:

- To preprocess images into the required format, run the `createimg.py` script located in `common/scripts`.
- `hospital/labels.csv` contains labeling performed by radiologist over the hospital cohort data.

## `visionets`

This module contains scripts related to model training, evaluation, and experimentation:

- **`train.py`**: Main script for fine-tuning models with configurable hyperparameters in a distributed training setup.
- **`activelearn.py`**: Script for studying model adaptation to new samples from a different domain. This script is not used in the BIOMAGING 2026 paper.
- **`search.py`**: Hyperparameter search orchestration script. It invokes `train.py` using `Popen`.
- **`inference`**: Scripts for running inference on single DICOM images, directories of images, and dataset partitions generated by `dbprocess`.
- **`utils`**: Utility scripts for model export and metric computation.