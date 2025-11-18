import os
from pathlib import Path
from src.model.config_mgmt import load_modeling_config
import re
from src.model.initialization import extract_best_model_hyperparams_from_study
import joblib
import torch
import torch.nn as nn


def load_models(
    modeling_config_dir, 
    optuna_storage, # For extracting hparams
    model_storage, # For extracting saved model file
    input_dim, # For properly initializing models
    config,
    modeling_config_filter="",
):
    # Collect modeling configs
    modeling_config_paths = []
    if modeling_config_filter != "":
        modeling_config_paths.append(os.path.join(modeling_config_dir, modeling_config_filter))
    else:
        for path in Path(modeling_config_dir).iterdir():
            if path.is_file() and not path.name.startswith('.') and not path.name == '__init__.py':
                modeling_config_paths.append(path)
    
    # Sort so that baseline files come last
    pattern = re.compile(r'baseline_\d+\.py')
    modeling_config_paths.sort(key=lambda p: (1 if pattern.match(Path(p).name) else 0, p.name))
    
    # Now get models
    models = {} # model_name : model_class
    
    for modeling_config_path in modeling_config_paths:
        modeling_config = load_modeling_config(str(modeling_config_path))
        
        model_name = f"{modeling_config.model_name}_using_{config['config_name']}"
        model_extension = modeling_config.model_extension
        model_path = os.path.join(model_storage, f"{model_name}.{model_extension}")

        if not Path(model_path).is_file():
            continue
        
        model_class = modeling_config.model_class
        
        if issubclass(model_class, nn.Module):
            model_hyperparams = extract_best_model_hyperparams_from_study(
                modeling_config, 
                optuna_storage
            )
            model = model_class(
                input_dim=input_dim,
                output_dim=1,
                **model_hyperparams
            )
            model.load_state_dict(torch.load(model_path))
            model.eval()
        else:
            model = joblib.load(model_path)
        
        models[model_name] = model
    
    return models