# Standard library imports
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import yaml
import optuna

def main():
    # (0) Read in configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    OPTUNA_STORAGE = config['optuna_storage']
    
    study_names = optuna.study.get_all_study_names(
        storage=OPTUNA_STORAGE
    )

    print(f"Existing studies at '{OPTUNA_STORAGE}: \n{study_names}")
    
if __name__ == "__main__":
    main()