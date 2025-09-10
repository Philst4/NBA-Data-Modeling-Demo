import sys
import os
import argparse
import importlib.util
import sys
from pathlib import Path
from time import time

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import optuna

# Internal imports
from src.utils import (
    get_prev_0_modeling_data
)

# Required variables that must exist in the config
REQUIRED_CONFIG_VARS = [
    "model_class",
    "hyperparam_space",
    "objective_fn",
    "val_seasons",
    "study_name",
]

def load_config(config_path):
    """Load a Python config file as a module."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found.")

    spec = importlib.util.spec_from_file_location("config", str(config_path))
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)

    # Validate required variables
    missing_vars = [var for var in REQUIRED_CONFIG_VARS if not hasattr(config, var)]
    if missing_vars:
        raise ValueError(f"Config file is missing required variables: {missing_vars}")

    return config

def acc(y, y_preds):
    return ((y > 0) == (y_preds > 0)).astype(int).mean()

def backtest(
    model_class, 
    hyperparams, 
    modeling_data, 
    features, 
    target, 
    val_seasons,
    n_train_seasons,
    objective_fn,
    verbose=True
):
    """
    Accumulates objective scores of models trained/validated on specified train-val splits.
    """
    scores = []
    
    # For verbose
    maes = []
    rmses = []
    accs = []
    times = []
    
    for i, val_season in enumerate(val_seasons):
        
        # Print statement
        print(f"\n * Tuning on split {i+1} (val_season={val_season}, n_train_seasons={n_train_seasons} -> training from {val_season - n_train_seasons}+)...")
        
        # Start tracking time
        iter_start_time = time()
        
        # Get train-val split
        train_condn1 = modeling_data['SEASON_ID'] - 20_000 < val_season
        train_condn2 = val_season - (modeling_data['SEASON_ID'] - 20_000) <= n_train_seasons
        training_data = modeling_data.loc[train_condn1 & train_condn2, :]
        
        val_condn = modeling_data['SEASON_ID'] - 20_000 == val_season
        val_data = modeling_data.loc[val_condn, :]
        
        assert len(training_data) > 0, f"No training data when val_season={val_season}"
        assert len(val_data) > 0, f"No validation data when val_season={val_season}"
        
        # Get features/target
        X_tr = training_data[features]
        y_tr = training_data[target]
        X_val = val_data[features]
        y_val = val_data[target]

        # Initialize a new model instance
        model = model_class(**hyperparams)
        
        # Fit the model on the training data
        model.fit(X_tr, y_tr)
        
        # Evaluate the model on the validation data
        y_val_preds = model.predict(X_val)
        
        # Calculate score from objective function
        score = objective_fn(y_val_preds, y_val)
        scores.append(score)
        
        iter_end_time = time()
        
        maes.append(mean_absolute_error(y_val, y_val_preds))
        rmses.append(root_mean_squared_error(y_val, y_val_preds))
        accs.append(acc(y_val, y_val_preds))
        times.append(iter_end_time - iter_start_time)
        
        if verbose:
            # Print metrics
            print(f" -> MAE: {maes[-1]:.3f}")
            print(f" -> RMSE: {rmses[-1]:.3f}")
            print(f" -> Accuracy: {accs[-1]:.3f}")
            print(f" -> Time to fit: {times[-1]:3f} seconds")
            
    
    if verbose:
        # Print overall metrics
        print(f"\n * Average Overall Metrics:")
        print(f" -> MAE: {np.mean(maes):.3f}")
        print(f" -> RMSE: {np.mean(rmses):.3f}")
        print(f" -> Accuracy: {np.mean(accs):.3f}")
        print(f" -> Time to fit: {np.mean(times):3f} seconds")
         
    return scores

def make_objective(
    model_class, 
    hyperparam_space, 
    modeling_data, 
    features, 
    target,
    val_seasons,
    n_train_seasons_suggestion,
    objective_fn
):
    """
    Makes a backtesting optuna objective function for some configuration of the following:
        * model_class
        * hyperparam_space
        * Features
        * Target
        * Val seasons
        * Objective function (E.g. RMSE)
    """
    
    def objective(trial):
        # Dynamically sample hyperparameters
        hyperparams = {}
        for name, specs in hyperparam_space.items():
            if not isinstance(specs, tuple):
                # For non-tunable
                hyperparams[name] = specs
            else:
                # For tunable
                ptype = specs[0]
                if ptype == "float":
                    hyperparams[name] = trial.suggest_float(name, **specs[1])
                elif ptype == "int":
                    hyperparams[name] = trial.suggest_int(name, **specs[1])
                elif ptype == "categorical":
                    hyperparams[name] = trial.suggest_categorical(name, **specs[1]) 
                else:
                    raise ValueError(f"Unknown param type: {ptype}")
        
        # Handle n_train_seasons
        n_train_seasons = trial.suggest_int("n_train_seasons", **n_train_seasons_suggestion[1])
        
        # Backtest the model
        scores = backtest(
            model_class, 
            hyperparams, 
            modeling_data, 
            features, 
            target, 
            val_seasons,
            n_train_seasons,
            objective_fn
        )
        return np.mean(scores)
    
    return objective
    
def main():
    """
    Makes an objective function using the specified configuration, 
    and tunes according to that objective function.
    """
    
    # From base config TODO
    optuna_storage = "sqlite:///optuna_studies.db"
    
    # Create argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeling_config", type=str, default="modeling_configs/lasso.py")
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--n_jobs", type=int, default=1, help="Used to parallelize building models ATM.")
        
    # Extract args (including modeling config)
    args = parser.parse_args()
    modeling_config = args.modeling_config
    n_trials = args.n_trials
    n_jobs = args.n_jobs
        
    # Read modeling config
    modeling_config = load_config(args.modeling_config)
    
    # Get modeling data
    modeling_data, features, target = get_prev_0_modeling_data()
    
    # Make objective
    objective = make_objective(
        modeling_config.model_class,
        modeling_config.hyperparam_space,
        modeling_data,
        features,
        target,
        modeling_config.val_seasons, 
        modeling_config.n_train_seasons_suggestion,
        modeling_config.objective_fn
    )
    
    # Make study
    study = optuna.create_study(
        study_name=modeling_config.study_name,
        direction="minimize",
        storage=optuna_storage,
        load_if_exists=True
    )
    
    # Tune!
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    
    print(f"\nBest Trial:\n")
    print(study.best_trial)

if __name__ == "__main__":
    main()