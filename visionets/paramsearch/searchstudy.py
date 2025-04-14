import optuna
import argparse
import joblib
import datetime


def clean_print_trial(trial):

    params = list(trial.params.values())
    for p in params:
        print(p, end=":")
    if trial.values is not None:
        print(trial.values[0], end=":")
    else:
        print("Repeated param", end=":")
    print((trial.datetime_complete - trial.datetime_start).total_seconds())


if __name__ == "__main__":

    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("trials_path", type=str)
    args = parser.parse_args()

    # Read study
    study = joblib.load(args.trials_path)

    # Show all trials ordered
    print()
    print("**************** Ordered Trials ****************")
    print()
    ordered_trials = sorted(
        study.trials,
        key=lambda trial: (
            -trial.values[0] if trial.values is not None else float("inf")
        ),
    )
    for trial in ordered_trials:
        clean_print_trial(trial)
