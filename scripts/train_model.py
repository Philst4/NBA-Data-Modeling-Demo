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
from src.utils import set_seed

from src.model.config_mgmt import (
    load_modeling_config,
)

from src.data.io import (
    get_modeling_data
)

from src.model.initialization import (
    extract_best_model_hyperparams_from_study
)

from src.model.training import (
    train_sklearn,
    train_torch,
)       

def main(args):
    # Read configuration
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    CLEAN_DATA_DIR = config['clean_data_dir']
    DB_NAME = config['db_name']
    DB_PATH = os.path.join(CLEAN_DATA_DIR, DB_NAME)
    MODELING_CONFIG_DIR = config['modeling_config_dir']
    OPTUNA_STORAGE = config["optuna_storage"]
    MODEL_STORAGE = config['model_storage']
    
    # Read modeling config
    modeling_config = load_modeling_config(
        os.path.join(
            MODELING_CONFIG_DIR,
            args.modeling_config
        )
    )
    
    # Set seed
    set_seed()
    
    # Load in study
    study_name = f"{modeling_config.model_name}_using_{config['config_name']}"
    study = optuna.load_study(
        storage=OPTUNA_STORAGE,
        study_name=study_name
    ) 
    
    # Extract model class from modeling config
    model_class = modeling_config.model_class
    
    # Extract best hyperparams from study
    print(f"Best hyperparams of '{model_class}' from study '{study_name}':")
    print(study.best_trial)
    all_hyperparams = study.best_trial.params
    
    # Extract best model hyperperams from study
    model_hyperparams = extract_best_model_hyperparams_from_study(study, modeling_config)
    
    # Load in data, get training data
    modeling_data, features, target = get_modeling_data(
        DB_PATH,
        config=config
    )
    last_train_season = args.last_train_season
    n_train_seasons = all_hyperparams['n_train_seasons']
    train_condn1 = modeling_data['SEASON_ID'] - 20_000 <= last_train_season
    train_condn2 = last_train_season - (modeling_data['SEASON_ID'] - 20_000) <= n_train_seasons
    training_data = modeling_data.loc[train_condn1 & train_condn2, :]
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train the model
    print(f"\nTraining '{model_class}' using best hyperparams from '{study_name}'...")
    if not issubclass(model_class, nn.Module):
        # For sklearn-style modeling setups
        model = train_sklearn(
            modeling_config.model_class, 
            model_hyperparams,
            training_data,
            features,
            target,
            device
        )
        
    else:
        # For torch-style modeling setups
        # Extract optimizer hyperparams from study
        optimizer_hyperparams = {
            key : all_hyperparams[key] if key in all_hyperparams else val 
            for key, val in modeling_config.optimizer_hyperparam_space.items()
        }
        
        # Other hparams for torch
        n_epochs = all_hyperparams['n_epochs']
        
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
            n_epochs,
            device
        )
    
    # Ensure storage path exists
    os.makedirs(MODEL_STORAGE, exist_ok=True)
    
    # Save the model
    if not args.model_filename:
        model_path = os.path.join(MODEL_STORAGE, f"{study_name}.{modeling_config.model_extension}")
    else:
        model_path = os.path.join(MODEL_STORAGE, args.model_filename)
    print(f"Saving model to '{model_path}'")
    if not isinstance(model, nn.Module):
        dump(model, model_path)
    else:
        torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=os.environ.get("CONFIG_PATH", "configs/config.yaml"), help="Config path")
    parser.add_argument("--modeling_config", type=str, default="lasso.py", help="Modeling config")
    parser.add_argument("--last_train_season", type=int, default=2024, help="Season to train the model up to (inclusive).")
    parser.add_argument("--model_filename", type=str, default="", help="What to name model file. Overrides name specs in modeling_config if provided. Include extension")
    
    args = parser.parse_args()
    main(args)