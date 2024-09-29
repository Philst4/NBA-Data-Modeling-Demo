import sys
import os
import argparse

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import yaml
import pandas as pd
import matplotlib.pyplot as plt

# Internal imports
from src.utils import (
    get_training_logs
)


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="Name of trained model architecture")
    parser.add_argument("--version", required=True, type=int, help="Name of version")
    args = parser.parse_args()
    model_name = args.model
    version = args.version
    
    # To extract from yaml config (to revisit)
    logs_dir = "logs"
    
    # Read in training logs, display
    training_logs = get_training_logs(logs_dir, model_name, version)
    print(f"\n Training logs for '{model_name}', 'version_{version}':")
    print(f"{training_logs}")
    
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    training_logs['train_loss'].plot(ax=axes[0, 0], title='train_loss', color='y')
    training_logs['val_loss'].plot(ax=axes[0,1], title='test_loss', color='g')
    training_logs['train_acc'].plot(ax=axes[1, 0], title='train_acc', color='b')
    training_logs['val_acc'].plot(ax=axes[1, 1], title='test_acc', color='m')
    
    # Add layout adjustments
    plt.tight_layout()

    # Show plot
    plt.show()
    """
    