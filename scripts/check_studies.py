# Standard library imports
import sys
import os
import argparse

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import yaml
import optuna
import numpy as np

# Internal imports 
from src.utils import season_int_to_str

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

        # Header
        print(" * Metrics:")

        # Find the max number of values across all metrics
        max_len = max(len(v) for v in metrics.values())

        # Build header
        header = ["Split"] + [f"{i+1}" for i in range(max_len)] + ["Mean"]
        print(" | ".join(h.ljust(10) for h in header))
        print("-" * (13 * len(header)))

        # Rows
        for k, v in metrics.items():
            row = [k.ljust(10)]
            if k == "val_season":
                vals = [season_int_to_str(szn) for szn in v]
                mean_str = "-"
            else:
                vals = [f"{num:.3f}" for num in v]
                mean_str = f"{np.mean(v):.3f}"
            
            # Pad with blanks so all rows have same length
            vals += [""] * (max_len - len(vals))
            row.extend(val.ljust(10) for val in vals)
            row.append(mean_str.ljust(10))
            
            print(" | ".join(row))

    
if __name__ == "__main__":
    main()