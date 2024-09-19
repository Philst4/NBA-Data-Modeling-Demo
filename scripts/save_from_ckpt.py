import sys
import os
import importlib

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import yaml
import torch

# Local imports
from src.training import (
    LightningModel
)

if __name__ == "__main__":    
    # Read in configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    MODEL_SRC_DIR = config['model_src_dir']
    sys.path.append(MODEL_SRC_DIR)
    MODEL_SAVE_DIR = config['model_save_dir']
    setup = config['setups']['A']

    # Necessary
    model_config = setup['model_config']
    model_name = model_config['name']
    model_file = importlib.import_module(model_name)
    hyperparams = model_config['hyperparams']
    
    # Additional setup
    model_ckpt_name = model_name + '.ckpt'
    model_ckpt_path = os.path.join(MODEL_SAVE_DIR, model_ckpt_name)
    model_pt_name = model_name + '.pt'
    model_pt_path = os.path.join(MODEL_SAVE_DIR, model_pt_name)
    
    # Recover checkpoint
    # NOTE: state of trained model saved in checkpoint, automatically loaded into 'model'
    print("* Loading in model architecture...")
    model = model_file.Model(**hyperparams)
    print("* Loading in checkpoint...")
    lightning_model = LightningModel.load_from_checkpoint(
        model_ckpt_path,
        model=model
    )
    
    # Extract trained model
    print("* Scripting model saved from checkpoint...")
    model_to_script = lightning_model.model
    scripted_model = torch.jit.script(model)
    scripted_model.save(model_pt_path)
    print(f"* Model saved to '{model_pt_path}'")
    
    