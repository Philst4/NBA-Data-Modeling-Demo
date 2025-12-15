from time import time
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, 
    root_mean_squared_error,
    roc_auc_score,
)
import torch
import torch.nn as nn

# Internal imports
from src.model.training import (
    train_sklearn,
    train_torch
)

from src.model.predicting import (
    predict_sklearn,
    predict_torch
)

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
    device=None,
    verbose=True
):
    """
    Accumulates metrics of models trained/validated on specified train-val splits.
    """

    # Dictionary to store metrics
    metrics = {
        "val_season" : [],
        "mae" : [],
        "rmse" : [],
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
                target,
                device
            )
            
            # Predict for validation set
            y_val = val_data[target]
            y_val_preds = predict_sklearn(
                model, 
                val_data, 
                features
            )
            
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
                n_epochs,
                device
            )
            
            # Predict for validation set
            y_val_preds = predict_torch(
                model, 
                val_data, 
                features, 
                batch_size,
                device
            )
            y_val = torch.tensor(val_data[[target]].values).to(device)
            score = objective_fn(y_val_preds, y_val).item()
            
            # Convert to numpy for next part
            y_val_preds = y_val_preds.detach().cpu().numpy()
            y_val = y_val.detach().cpu().numpy()
            
        
        # Evaluate the model on the validation data
        metrics['score'].append(score)
        
        iter_end_time = time()

        metrics['val_season'].append(val_season)
        metrics['mae'].append(mean_absolute_error(y_val, y_val_preds))
        metrics['rmse'].append(root_mean_squared_error(y_val, y_val_preds))
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
            print(f" -> Val Season: {metrics['val_season'][-1]}")
            print(f" -> MAE: {metrics['mae'][-1]:.3f}")
            print(f" -> RMSE: {metrics['rmse'][-1]:.3f}")
            print(f" -> Accuracy: {metrics['acc'][-1]:.3f}")
            print(f" -> ROC AUC Score: {metrics['roc_auc'][-1]:.3f}")
            print(f" -> Time to fit: {metrics['time'][-1]:3f} seconds")
            
    
    if verbose:
        # Print overall metrics
        print(f"\n * Average Overall Metrics:")
        print(f" -> MAE: {np.mean(metrics['mae']):.3f}")
        print(f" -> RMSE: {np.mean(metrics['rmse']):.3f}")
        print(f" -> Accuracy: {np.mean(metrics['acc']):.3f}")
        print(f" -> ROC AUC Score: {np.mean(metrics['roc_auc']):.3f}")
        print(f" -> Time to fit: {np.mean(metrics['time']):3f} seconds")
         
    return metrics