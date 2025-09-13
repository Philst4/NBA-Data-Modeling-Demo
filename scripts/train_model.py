import sys
import os
import argparse
from joblib import dump

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import pandas as pd
import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Internal imports
from src.utils import (
    load_modeling_config,
    get_prev_0_modeling_data
)
from src.gameset import Gameset


def train_sklearn(
    model_class,
    model_hyperparams,
    training_data,
    features, 
    target
):
     
    assert len(training_data) > 0, f"No training data."
        
    # Get features/target
    X_tr = training_data[features]
    y_tr = training_data[target]
    
    # Instantiate model
    model = model_class(**model_hyperparams)
        
    # Fit the model on the training data
    model.fit(X_tr, y_tr)
    
    # Return model
    return model

def train_torch(
    model_class,
    model_hyperparams,
    training_data,
    features, 
    target,
    batch_size, # For Initializing DataLoader 
    optimizer_class, 
    optimizer_hyperparams,
    objective_fn, # For calculating loss/stepping w/ optimizer
    n_epochs, # How many passes over entire dataset to give model!
    
):  
    # Initialize DataLoader
    trainset = Gameset(training_data, features, [target])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = model_class(
        input_dim=trainset.get_input_dim(),
        output_dim=trainset.get_output_dim(),      
        **model_hyperparams
    )
    
    # Initialize optimizer
    optimizer = optimizer_class(params=model.parameters(), **optimizer_hyperparams)

    # Iteratively train model!
    for epoch in range(n_epochs):
        
        # Iterate over batch
        for batch_idx, (X, y) in enumerate(tqdm(trainloader, desc=f" -- Epoch {epoch + 1}/{n_epochs} -- ")):
            # Move tensors to proper device
            pass
        
            # Forward pass
            y_preds = model(X)
            loss = objective_fn(y_preds, y) # AKA loss_fn
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Print progress?
            pass

    # Return trained model instance
    return model

def predict_torch(
    model,
    data,
    features,
    batch_size,
    w_unique_ids=False,
):
    
    # Initialize DataLoader
    gameset = Gameset(data, features, [])
    gameloader = DataLoader(gameset, batch_size=batch_size, shuffle=False)
    
    # Initialize list to hold all predictions
    all_y_preds = []
    
    with torch.no_grad():
        # Iterate over batch
        for batch_idx, (X, _) in enumerate(gameloader):
            
            # Move tensors to proper device
            pass
        
            # Forward pass
            y_preds = model(X)
            
            # Add to list of results
            all_y_preds.append(y_preds)
    
    # Convert to numpy
    all_y_preds = torch.cat(all_y_preds, dim=0)
    
    # Return
    return all_y_preds
        

def main(args):
    # From base config TODO
    optuna_storage = "sqlite:///optuna_studies.db"
    
    # Read in modeling config
    modeling_config = load_modeling_config(args.modeling_config)
    
    # Load in study
    study = optuna.load_study(
        storage=optuna_storage,
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
    modeling_data, features, target = get_prev_0_modeling_data()
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
    
    # Save the model
    print(f"Saving model to '{modeling_config.model_filename}'")
    if not isinstance(model, nn.Module):
        dump(model, modeling_config.model_filename)
    else:
        torch.save(model, modeling_config.model_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeling_config", type=str, default="modeling_configs/lasso.py", help="Modeling config")
    parser.add_argument("--last_train_season", type=int, default=2024, help="Season to train the model up to (inclusive).")
    args = parser.parse_args()
    main(args)