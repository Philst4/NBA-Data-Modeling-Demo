# Standard library imports
import sys
import os
import argparse

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import yaml
import optuna

def main(args):
    # (0) Read in configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    OPTUNA_STORAGE = config['optuna_storage']
    
    # Load in study
    study = optuna.load_study(
        study_name=args.study_name, 
        storage=OPTUNA_STORAGE
    )
    
    # Delete the specified study
    optuna.delete_study(
        study_name=args.study_name, 
        storage=OPTUNA_STORAGE,
    )
    print(f"Study '{args.study_name}' was successfully deleted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('study_name', type=str, help="Name of table to remove from clean database.")
    args = parser.parse_args()
    main(args)