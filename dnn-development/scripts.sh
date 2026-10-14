#!/bin/bash

# Activate virtual environment
ENV_PATH=/path/to/env
source $ENV_PATH/bin/activate

# General paths (may point to files in the repository or files to be added)
IMG_DIR=/path/to/processed/images
CHEXPERT_TRAIN_CSV_PATH=/path/to/chexpert/labels.csv
PRECOMPUTED_DIR=modelpreds/teacherdn169/precomputed
MODELS_DIR=modelpreds/
LABEL_DIR=dbprocess/common/trainlabel
LABELNAMES_PATH=dbprocess/padchest/dictprocess/urgencylabel/labelnames.json
HOSPITAL_LABELNAMES_PATH=computemetrics/computeauc/hospital_labelnames.json
COMMON_LABELNAMES_PATH=computemetrics/computeauc/common_labelnames.json
HOSPITAL_PATH=dbprocess/hospital

# Global parameters
# GPUs for training scripts
NGPUS=8

## TRAINING SCRIPTS ##

# Distilled train over PadChest + ChestX-ray14 on the 32 category pathology system                                            
torchrun --nproc_per_node=$NGPUS visionets/training/train.py $IMG_DIR \
                                              $LABEL_DIR/urgencylabel/ \
                                              --pre_path $PRECOMPUTED_DIR \
                                              --pw_apply cls \
                                              --pw_formula multilabel \
                                              --loss bce \
                                              --epochs 15 \
                                              --epochs_freeze 1 \
                                              --base_model_name deit_base_distilled_patch16_384 \
                                              --input_size 384 \
                                              --randaug 3 0 0.5 \
                                              --num_repeats 1 \
                                              --batch_size 8 \
                                              --pin_memory \
                                              -j 10 \
                                              &> progress_train_distilled.txt                                                 
                                            

# Distilled train over PadChest + ChestX-ray14 on the 32 category pathology system + Semi-supervised Train over independent source (to be disclosed upon acceptance)
torchrun --nproc_per_node=$NGPUS visionets/training/train.py $IMG_DIR \
                                              $LABEL_DIR/urgencylabel/ \
                                              --save_path deit.pt \
                                              --pre_path $PRECOMPUTED_DIR \
                                              --semi_id at \
                                              --pw_apply dual \
                                              --pw_formula multilabel \
                                              --loss bce \
                                              --epochs 15 \
                                              --epochs_freeze 1 \
                                              --base_model_name deit_base_distilled_patch16_384 \
                                              --randaug 2 6 0.5 \
                                              --num_repeats 3 \
                                              --batch_size 8 \
                                              --pin_memory \
                                              -j 10 \
                                              &> progress_train_distilled_semi.txt                                             

# Traditional train
torchrun --nproc_per_node=$NGPUS visionets/training/train.py $IMG_DIR \
                                              $LABEL_DIR/urgencylabel/ \
                                              --pw_apply cls \
                                              --pw_formula multilabel \
                                              --loss bce \
                                              --epochs_freeze 1 \
                                              --epochs 1 \
                                              --base_model_name densenet169 \
                                              --batch_size 8 \
                                              --num_repeats 1 \
                                              --randaug 5 3 0.5 \
                                              --input_size 512 \
                                              --pin_memory \
                                              -j 10 \
                                              &> progress_train.txt

## INFERENCE ##

# Inference
python3 visionets/inference/inference.py $IMG_DIR \
                               $LABEL_DIR/urgencylabel/ \
                               model.pt \
                               inferencedata \
                               tr vl ts \
                               &> progress_inference.txt
# Inference dicom
python3 visionets/inference/dcminference.py dicom model.pt --model_format jit &> progress_dicom_inference.txt

# Computation of prediction over teacherdn169
python3 visionets/inference/chexpert/chexpertinference.py $IMG_DIR/chexpertPKT \
                                                CHEXPERT_TRAIN_CSV_PATH \
                                                $MODELS_DIR/teacherdn169/dn169.pt \
                                                $MODELS_DIR/teacherdn169/inferencedata/chexpert_preds.pt &> progress.txt

## EXPORTERS ##

# Export new model
python3 visionets/utils/exporters/export_new_model.py modeljit.pt jit --subformat trace --opset 11 &> progress_export_new.txt

# Export saved model
python3 visionets/utils/exporters/export_saved_model.py model.pt jitmodel.jit jit --subformat trace &> progress_export_saved.txt

## Utils/metrics ##

# Metrics
python3 visionets/utils/metrics/metrics.py $LABEL_DIR/urgencylabel/ $LABELNAMES_PATH inferencedata &> progress_metrics.txt

## COMPUTEMETRICS ##

# Clinical operating points
python computemetrics/clinical/operating_threshold.py $LABEL_DIR \
    $MODELS_DIR \
    $HOSPITAL_PATH \
    $LABELNAMES_PATH \
    --models_idfs convnext deit swintv2 teacherdn169 \
    --beta 2 &> progress.txt

# ECE computation
python computemetrics/clinical/ece.py $LABEL_DIR \
    $MODELS_DIR \
    $HOSPITAL_PATH \
    $LABELNAMES_PATH \
    $HOSPITAL_LABELNAMES_PATH \
    --models_idfs convnext deit swintv2 teacherdn169 &> progress.txt

# Internal AUC computation over restricted Hospital labelnames subset
python computemetrics/computeauc/internalauc.py $LABEL_DIR \
    $LABELNAMES_PATH \
    $MODELS_DIR \
    internalauc_hospital.json \
    --models_idfs convnext deit swintv2 teacherdn169 \
    --partitions ts \
    --bootstraps 1000 --alpha 0.95 \
    --selected_labelnames_path hospital_labelnames.json &> progress.txt

# Hospital AUC computation over restricted common labelnames subset
python computemetrics/computeauc/hospitalauc.py $HOSPITAL_PATH/labels.csv \
    $LABELNAMES_PATH \
    $HOSPITAL_PATH \
    hospital_common.json \
    --models_idfs convnext deit swintv2 teacherdn169 \
    --count_limit 10 --bootstraps 1000 --alpha 0.95 \
    --selected_labelnames_path common_labelnames.json &> progress.txt

# Chexpert AUC computation
python computemetrics/computeauc/chexpertauc.py $CHEXPERT_TRAIN_CSV_PATH \
    $LABELNAMES_PATH \
    $MODELS_DIR \
    chexpertauc.json \
    --models_idfs convnext deit swintv2 teacherdn169 \
    --bootstraps 1000 --alpha 0.95 &> progress.txt

# CI AUC Diff
python computemetrics/computeauc/ciaucdiff.py $CHEXPERT_TRAIN_CSV_PATH \
    $LABELNAMES_PATH \
    $COMMON_LABELNAMES_PATH \
    $HOSPITAL_PATH \
    $MODELS_DIR &> progress.txt