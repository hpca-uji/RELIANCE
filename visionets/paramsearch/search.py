import argparse
import subprocess
import os

import joblib
import optuna


def get_model_fullname(base_model, base_extension, base_res):

    if base_model == "beit":
        # Build extension string
        if base_extension == 0:
            base_extension_str = "base_patch16"
        else:
            base_extension_str = "large_patch16"
        # Build resolution int
        if base_res == 0:
            if base_extension == 0:
                base_res_int = 224
            else:
                base_res_int = 384
        else:
            if base_extension == 0:
                base_res_int = 384
            else:
                base_res_int = 512
        fullname = f"{base_model}_{base_extension_str}_{base_res_int}"
    elif base_model == "vit":
        # Build extension string
        if base_extension == 0:
            base_extension_str = "base_patch16"
        else:
            base_extension_str = "large_patch16"
        # Build resolution int
        base_res_int = 384
        fullname = f"{base_model}_{base_extension_str}_{base_res_int}"
    elif base_model == "deit":
        # Build extension string
        if base_extension == 0:
            base_extension_str = "base_patch16"
        else:
            base_extension_str = "base_distilled_patch16"
        # Build resolution int
        if base_res == 0:
            base_res_int = 224
        else:
            base_res_int = 384
        fullname = f"{base_model}_{base_extension_str}_{base_res_int}"
    elif base_model == "swinv2":
        # Build extension string
        if base_extension == 0 and base_res == 0:
            fullname = "swinv2_base_window12to16_192to256"
            base_res_int = 256
        elif base_extension == 0 and base_res == 1:
            fullname = "swinv2_base_window12to24_192to384"
            base_res_int = 384
        elif base_extension == 1 and base_res == 0:
            fullname = "swinv2_large_window12to16_192to256"
            base_res_int = 256
        elif base_extension == 1 and base_res == 1:
            fullname = "swinv2_large_window12to24_192to384"
            base_res_int = 384
    elif base_model == "convnext":
        # Build extension string
        if base_extension == 0:
            base_extension_str = "large"
        else:
            base_extension_str = "xlarge"
        # Build resolution int
        if base_res == 0:
            base_res_int = 224
        else:
            base_res_int = 384
        fullname = f"{base_model}_{base_extension_str}"
    elif base_model == "densenet":
        # Build extension string
        if base_extension == 0:
            base_extension_str = "169"
        else:
            base_extension_str = "161"
        # Build resolution int
        if base_res == 0:
            base_res_int = 256
        else:
            base_res_int = 512
        fullname = f"{base_model}{base_extension_str}"
    return fullname, base_res_int


# Function that creates and return the best validacion auc
def objective(trial):

    # Training parameters
    batch_size = trial.suggest_int("batch_size", 8, 12, 2)
    lr = trial.suggest_categorical(
        "lr",
        [
            10 ** (-6),
            5 * (10 ** (-6)),
            10 ** (-5),
            5 * (10 ** (-5)),
            10 ** (-4),
            5 * (10 ** (-4)),
        ],
    )
    opt = trial.suggest_categorical(
        "opt", ["adam", "adamw", "nadam", "radam", "adamax"]
    )

    # Data augmentation parameters
    aug_magnitude = trial.suggest_int("aug_magnitude", 1, 10, 1)
    aug_layers = trial.suggest_int("aug_layers", 0, 6, 1)
    aug_mstd = trial.suggest_categorical("aug_mstd", ["inf", "0.5", "1", "1.5", "2"])
    num_repeats = trial.suggest_int("num_repeats", 1, 4, 1)

    # Model parameters
    pre = "resize"
    base_model = trial.suggest_categorical(
        "base_model", ["beit", "vit", "deit", "swinv2", "convnext", "densenet"]
    )
    base_extension = trial.suggest_int("base_extension", 0, 1, 1)
    base_res = trial.suggest_int("base_res", 0, 1, 1)

    # Build full model name
    base_model_name, base_model_res = get_model_fullname(
        base_model, base_extension, base_res
    )
    # Classifier parameters
    layers = trial.suggest_int("layers", 0, 1, 1)
    neurons = trial.suggest_int("neurons", 128, 512, 128)
    norm_type = trial.suggest_categorical("norm_type", ["bn", "ln", "none"])
    drop_type = trial.suggest_categorical("drop_type", ["drop", "alpha"])
    drop_rate = trial.suggest_discrete_uniform("drop_rate", 0.0, 0.5, 0.1)
    activation = "relu"  # Irrelevant since there are not hidden layers

    for t in trial.study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        if t.params == trial.params:
            raise optuna.exceptions.TrialPruned("Duplicate parameter set")

    # Create subprocess
    if "distilled" in base_model_name:
        p = subprocess.Popen(
            f"torchrun --nproc_per_node=8 ../training/train.py \
                             /scratch1/fsoler/visiondata/img \
                             /scratch1/fsoler/visiondata/label/urgencylabel/ \
                             --pre_path ../training/trainedmodels/teacherdn169/precomputed \
                             --semi_id at \
                             --hierarchy_path /scratch1/fsoler/visiondata/hierarchy.pt \
                             --metric_save_path progressmetric.txt \
                             --pw_apply dual \
                             --pw_formula multilabel \
                             --loss bce \
                             --batch_size {batch_size} \
                             --epochs 1 \
                             --epochs_freeze 1 \
                             --lr {lr} \
                             --opt {opt} \
                             --randaug {aug_magnitude} {aug_layers} {aug_mstd} \
                             --num_repeats {num_repeats} \
                             --ptdrop_rate 0 \
                             --pre {pre} \
                             --base_model_name {base_model_name} \
                             --input_size {base_model_res} \
                             --layers {layers} \
                             --neurons {neurons} \
                             --norm_type {norm_type} \
                             --drop_type {drop_type} \
                             --drop_rate {drop_rate} \
                             --activation {activation} \
                             --num_label 32 &>> progress.txt",
            shell=True,
        )
    else:
        p = subprocess.Popen(
            f"torchrun --nproc_per_node=8 ../training/train.py \
                             /scratch1/fsoler/visiondata/img \
                             /scratch1/fsoler/visiondata/label/urgencylabel/ \
                             --hierarchy_path /scratch1/fsoler/visiondata/hierarchy.pt \
                             --metric_save_path progressmetric.txt \
                             --pw_apply dual \
                             --pw_formula multilabel \
                             --loss bce \
                             --batch_size {batch_size} \
                             --epochs 1 \
                             --epochs_freeze 1 \
                             --lr {lr} \
                             --opt {opt} \
                             --randaug {aug_magnitude} {aug_layers} {aug_mstd} \
                             --num_repeats {num_repeats} \
                             --ptdrop_rate 0 \
                             --pre {pre} \
                             --base_model_name {base_model_name} \
                             --input_size {base_model_res} \
                             --layers {layers} \
                             --neurons {neurons} \
                             --norm_type {norm_type} \
                             --drop_type {drop_type} \
                             --drop_rate {drop_rate} \
                             --activation {activation} \
                             --num_label 32 &>> progress.txt",
            shell=True,
        )
    # Communicate and wait for result
    (output, err) = p.communicate()
    p_status = p.wait()

    # Read results
    eval_file = open("progressmetric.txt", "r")
    auc = float(eval_file.readline())
    eval_file.close()
    os.remove("progressmetric.txt")

    # Return results
    return auc


if __name__ == "__main__":

    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("trials_path", type=str)
    args = parser.parse_args()

    # Check if trials objects exists and create if it doesnt
    if os.path.exists(args.trials_path):
        study = joblib.load(args.trials_path)
    else:
        study = optuna.create_study(
            direction="maximize",
            study_name="Optuna-RELIANCE",
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=50),
        )

    # Call optimize function
    study.optimize(objective, n_trials=10)

    # Save study object
    joblib.dump(study, args.trials_path)

    # Print study object
    print(study.trials)
