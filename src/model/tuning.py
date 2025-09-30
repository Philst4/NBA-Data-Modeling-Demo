#### FOR TUNING ####
import optuna
import numpy as np
import torch
import torch.nn as nn

# Internal imports
from src.utils import set_seed
from src.model.evaluating import backtest

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


def make_backtest_objective(
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
    n_epochs_space=None,
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
        # Set seed
        set_seed()
        
        # Sample model hyperparameters
        model_hyperparams = sample_hyperparams(trial, model_hyperparam_space)
        
        # Sample n_train_seasons
        n_train_seasons = sample_hyperparam(trial, "n_train_seasons", n_train_seasons_space)
        
        # For torch setup only
        if issubclass(model_class, nn.Module):
            
            # Sample optimizer hyperparameters
            optimizer_hyperparams = sample_hyperparams(trial, optimizer_hyperparam_space)
            n_epochs = sample_hyperparam(trial, "n_epochs", n_epochs_space)
            
        else:
            optimizer_hyperparams = None
            n_epochs = None
            
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