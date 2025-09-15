import sys
import os
import argparse
from time import time

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, 
    root_mean_squared_error,
    r2_score,
    roc_auc_score,
)
import optuna
import torch
import torch.nn as nn

# Internal imports
from src.utils import (
    load_modeling_config,
    get_prev_0_modeling_data
)
from train_model import (
    train_sklearn,
    train_torch,
    predict_torch
)

def sample_hyperparam(trial, name, specs):
    """
    pass
    """
    if not isinstance(specs, tuple):
        # For non-tunable
        return specs
    else:
        # For tunable
        ptype = specs[0]
        if ptype == "float":
            return trial.suggest_float(name, **specs[1])
        elif ptype == "int":
            return trial.suggest_int(name, **specs[1])
        elif ptype == "categorical":
            return trial.suggest_categorical(name, **specs[1]) 
        else:
            raise ValueError(f"Unknown param type: {ptype}")

def sample_hyperparams(trial, hyperparam_space):
    """
    Dynamically samples hyperparameters optuna-style, given sampling space.
    """
    hyperparams = {}
    for name, specs in hyperparam_space.items():
        hyperparams[name] = sample_hyperparam(trial, name, specs)
    return hyperparams

def acc(y, y_preds):
    return ((y > 0) == (y_preds > 0)).astype(int).mean()

def backtest(
    model_class, 
    model_hyperparams, 
    modeling_data, 
    features, 
    target, 
    val_seasons,
    n_train_seasons,
    objective_fn,
    batch_size=None, # Start of torch-specific args
    optimizer_class=None,
    optimizer_hyperparams=None,
    n_epochs=None,
    verbose=True
):
    """
    Accumulates metrics of models trained/validated on specified train-val splits.
    """
    scores = []
    
    # Dictionary to store metrics
    metrics = {
        "mae" : [],
        "rmse" : [],
        "r2" : [],
        "acc" : [],
        "roc_auc" : [],
        "time" : [],
        "score" : [], # What given objective function calculates
    }
    
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
        
        if not issubclass(model_class, nn.Module):
            # For sklearn-style model setups         
            # Get model instance fit on the training data
            model = train_sklearn(
                model_class,
                model_hyperparams,
                training_data,
                features,
                target
            )
            
            # Predict for validation set
            X_val = val_data[features]
            y_val = val_data[target]
            y_val_preds = model.predict(X_val)
            
            # Evaluate
            score = objective_fn(y_val_preds, y_val)
            
        else:
            # For torch-style model setups
            # Get model instance fit on the training data
            
            model = train_torch(
                model_class,
                model_hyperparams,
                training_data,
                features,
                target,
                batch_size,
                optimizer_class,
                optimizer_hyperparams,
                objective_fn,
                n_epochs
            )
            
            # Predict for validation set
            # TODO: make a 'predict_torch' function.
            y_val_preds = predict_torch(
                model, 
                val_data, 
                features, 
                batch_size
            )
            y_val = torch.tensor(val_data[[target]].values)
            score = objective_fn(y_val_preds, y_val).item()
            
            # Convert to numpy for next part
            y_val_preds = y_val_preds.numpy()
            y_val = y_val.numpy()
            
        
        # Evaluate the model on the validation data
        metrics['score'].append(score)
        
        iter_end_time = time()
    
        metrics['mae'].append(mean_absolute_error(y_val, y_val_preds))
        metrics['rmse'].append(root_mean_squared_error(y_val, y_val_preds))
        metrics['r2'].append(r2_score(y_val, y_val_preds))
        metrics['acc'].append(acc(y_val, y_val_preds))
        metrics['roc_auc'].append(
            roc_auc_score(
                (y_val > 0).astype(int), 
                y_val_preds
            )
        )
        metrics['time'].append(iter_end_time - iter_start_time)
        
        if verbose:
            # Print metrics
            print(f" -> MAE: {metrics['mae'][-1]:.3f}")
            print(f" -> RMSE: {metrics['rmse'][-1]:.3f}")
            print(f" -> R^2 Score: {metrics['r2'][-1]:.3f}")
            print(f" -> Accuracy: {metrics['acc'][-1]:.3f}")
            print(f" -> ROC AUC Score: {metrics['roc_auc'][-1]:.3f}")
            print(f" -> Time to fit: {metrics['time'][-1]:3f} seconds")
            
    
    if verbose:
        # Print overall metrics
        print(f"\n * Average Overall Metrics:")
        print(f" -> MAE: {np.mean(metrics['mae']):.3f}")
        print(f" -> RMSE: {np.mean(metrics['rmse']):.3f}")
        print(f" -> R^2 Score: {np.mean(metrics['r2']):.3f}")
        print(f" -> Accuracy: {np.mean(metrics['acc']):.3f}")
        print(f" -> ROC AUC Score: {np.mean(metrics['roc_auc']):.3f}")
        print(f" -> Time to fit: {np.mean(metrics['time']):3f} seconds")
         
    return metrics

def make_objective(
    model_class, 
    model_hyperparam_space, 
    modeling_data, 
    features, 
    target,
    val_seasons,
    n_train_seasons_space,
    objective_fn,
    batch_size=None,
    optimizer_class=None,
    optimizer_hyperparam_space=None,
    n_epochs=None,
    verbose=True
    
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
        # Sample model hyperparameters
        model_hyperparams = sample_hyperparams(trial, model_hyperparam_space)
        
        # Sample n_train_seasons
        n_train_seasons = sample_hyperparam(trial, "n_train_seasons", n_train_seasons_space)
        
        # For torch setup only
        if issubclass(model_class, nn.Module):
            # Sample optimizer hyperparameters
            optimizer_hyperparams = sample_hyperparams(trial, optimizer_hyperparam_space)
            
        else:
            optimizer_hyperparams = None
            
        # Backtest the model
        metrics = backtest(
            model_class, 
            model_hyperparams, 
            modeling_data, 
            features, 
            target, 
            val_seasons,
            n_train_seasons,
            objective_fn,
            batch_size,
            optimizer_class,
            optimizer_hyperparams,
            n_epochs,
            verbose=True
        )
        
        # Log metrics as user attributes
        trial.set_user_attr("metrics", metrics)
        
        # Return objective function's score
        return np.mean(metrics['score'])
    
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
    modeling_config = load_modeling_config(args.modeling_config)
    
    # Get modeling data
    modeling_data, features, target = get_prev_0_modeling_data()
    
    # Make objective
    objective = make_objective(
        modeling_config.model_class,
        modeling_config.model_hyperparam_space,
        modeling_data,
        features,
        target,
        modeling_config.val_seasons, 
        modeling_config.n_train_seasons_space,
        modeling_config.objective_fn,
        modeling_config.batch_size,
        modeling_config.optimizer_class,
        modeling_config.optimizer_hyperparam_space,
        modeling_config.n_epochs
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