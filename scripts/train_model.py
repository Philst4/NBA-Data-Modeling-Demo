import sys
import os
import argparse
from joblib import dump

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import yaml
import pandas as pd
import numpy as np
import optuna
import torch
import torch.nn as nn

# Internal imports
from src.model.config_mgmt import (
    load_modeling_config,
)

from src.data.io import (
    get_modeling_data
)

from src.model.training import (
    train_sklearn,
    train_torch,
)       

def main(args):
    # Read configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    OPTUNA_STORAGE = config["optuna_storage"]
    MODEL_STORAGE = config['model_storage']
    
    # Read in modeling config
    modeling_config = load_modeling_config(args.modeling_config)
    
    # Load in study
    study = optuna.load_study(
        storage=OPTUNA_STORAGE,
        study_name=modeling_config.study_name
    )
    
    # Extract model class from modeling config
    model_class = modeling_config.model_class
    
    # Extract best hyperparams from study
    print(f"Best hyperparams of '{model_class}' from '{modeling_config.study_name}':")
    print(study.best_trial)
    all_hyperparams = study.best_trial.params
    
    # Extract best model hyperperams from study
    model_hyperparams = {
        key: all_hyperparams[key] if key in all_hyperparams else val
        for key, val in modeling_config.model_hyperparam_space.items()
    }
    
    # Load in data, get training data
    modeling_data, features, target = get_modeling_data()
    last_train_season = args.last_train_season
    n_train_seasons = all_hyperparams['n_train_seasons']
    train_condn1 = modeling_data['SEASON_ID'] - 20_000 <= last_train_season
    train_condn2 = last_train_season - (modeling_data['SEASON_ID'] - 20_000) <= n_train_seasons
    training_data = modeling_data.loc[train_condn1 & train_condn2, :]
    
    # Train the model
    print(f"\nTraining '{model_class}' using best hyperparams from '{modeling_config.study_name}'...")
    if not issubclass(model_class, nn.Module):
        # For sklearn-style modeling setups
        model = train_sklearn(
            modeling_config.model_class, 
            model_hyperparams,
            training_data,
            features,
            target
        )
        
    else:
        # For torch-style modeling setups
        # Extract optimizer hyperparams from study
        optimizer_hyperparams = {
            key : all_hyperparams[key] if key in all_hyperparams else val 
            for key, val in modeling_config.optimizer_hyperparam_space.items()
        }
        
        # Train model
        model = train_torch(
            modeling_config.model_class,
            model_hyperparams,
            training_data,
            features,
            target,
            modeling_config.batch_size,
            modeling_config.optimizer_class,
            optimizer_hyperparams,
            modeling_config.objective_fn,
            modeling_config.n_epochs
        )
    
    # Ensure storage path exists
    os.makedirs(MODEL_STORAGE, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(MODEL_STORAGE, modeling_config.model_filename)
    print(f"Saving model to '{model_path}'")
    if not isinstance(model, nn.Module):
        dump(model, model_path)
    else:
        torch.save(model, model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeling_config", type=str, default="modeling_configs/lasso.py", help="Modeling config")
    parser.add_argument("--last_train_season", type=int, default=2024, help="Season to train the model up to (inclusive).")
    args = parser.parse_args()
    main(args)