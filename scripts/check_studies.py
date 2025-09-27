# Standard library imports
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import yaml
import optuna
import numpy as np

def main():
    # (0) Read in configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    OPTUNA_STORAGE = config['optuna_storage']

    study_names = optuna.study.get_all_study_names(storage=OPTUNA_STORAGE)

    print(f"Existing studies at '{OPTUNA_STORAGE}':")
    for name in study_names:
        study = optuna.load_study(study_name=name, storage=OPTUNA_STORAGE)

        n_trials = len(study.trials)
        best_value = study.best_trial.value if study.best_trial is not None else "No completed trials"

        print(f" * {name}")
        print(f" * * Number of trials: {n_trials}")
        print(f" * * Best trial value: {best_value:.3f}")
        
def main():
    # (0) Read in configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    OPTUNA_STORAGE = config['optuna_storage']

    study_names = optuna.study.get_all_study_names(storage=OPTUNA_STORAGE)

    print(f"Existing studies at '{OPTUNA_STORAGE}':")
    for name in study_names:
        study = optuna.load_study(study_name=name, storage=OPTUNA_STORAGE)

        n_trials = len(study.trials)
        if study.best_trial is not None:
            best = study.best_trial
            best_num = best.number
            best_value = best.value
            metrics = best.user_attrs.get("metrics", {})
        else:
            best_num = "N/A"
            best_value = "No completed trials"
            metrics = {}

        print(f"\n #### {name} ####")
        print(f" * Number of trials: {n_trials}")
        print(f" * Best trial number: {best_num}")
        print(f" * Best trial value: {best_value:.3f}")

        if metrics:
            print(" * Metrics:")
            for k, v in metrics.items():
                print(f" --- {k}: {np.mean(v):.3f}")
        else:
            print(" * Metrics: (none)")
    
if __name__ == "__main__":
    main()