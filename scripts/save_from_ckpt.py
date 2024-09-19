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
from utils.training import (
    LightningModel
)

if __name__ == "__main__":    
    # Read in configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    ARCHITECTURE_DIR = config['architectures_dir']
    sys.path.append(ARCHITECTURE_DIR)
    setup = config['setups']['A']

    # Necessary
    model_config = setup['model_config']
    architecture = importlib.import_module(model_config['architecture'])
    
    # Logic
    lightning_model = LightningModel.load_from_checkpoint("models/model.ckpt")
    model = lightning_model.model
    scripted_model = torch.jit.script(model)
    scripted_model.save("models/scripted_model.pt")
    
    