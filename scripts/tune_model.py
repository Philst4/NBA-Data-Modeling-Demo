import sys
import os
import argparse

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import yaml
import optuna
import torch

# Internal imports
from src.model.config_mgmt import (
    load_modeling_config,
)

from src.data.io import (
    get_modeling_data
)

from src.model.tuning import make_backtest_objective
    
def main(args):
    """
    Makes an objective function using the specified configuration, 
    and tunes according to that objective function.
    """
    
    # Read configuration
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    CLEAN_DIR = config['clean_data_dir']    
    DB_NAME = config['db_name']
    DB_PATH = os.path.join(CLEAN_DIR, DB_NAME)
    MODELING_CONFIG_DIR = config['modeling_config_dir']
    OPTUNA_STORAGE = config["optuna_storage"]
        
    # Extract args (including modeling config)
    modeling_config = args.modeling_config
    n_trials = args.n_trials
    n_jobs = args.n_jobs
        
    # Read modeling config
    modeling_config = load_modeling_config(
        os.path.join(
            MODELING_CONFIG_DIR,
            args.modeling_config
        )
    )
    
    # Get modeling data
    modeling_data, features, target = get_modeling_data(
        db_path=DB_PATH,
        config=config
    )
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Make objective
    objective = make_backtest_objective(
        modeling_config.model_class,
        modeling_config.model_hyperparam_space,
        modeling_data,
        features,
        target,
        modeling_config.val_seasons, 
        modeling_config.n_train_seasons_space,
        modeling_config.objective_fn,
        modeling_config.batch_size,
        modeling_config.optimizer_class,
        modeling_config.optimizer_hyperparam_space,
        modeling_config.n_epochs_space,
        device=device
    )
    
    # Make study (with seed)
    sampler = optuna.samplers.TPESampler(seed=42)
    study_name = f"{modeling_config.model_name}_using_{config['config_name']}"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=OPTUNA_STORAGE,
        load_if_exists=True,
        sampler=sampler
    )
    
    # Set config as attribute for the study
    study.set_user_attr("config", config)
    
    # Tune!
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    
    print(f"\nBest Trial:\n")
    print(study.best_trial)

if __name__ == "__main__":
    # Create argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/config.yaml", help="Config path")
    parser.add_argument("--modeling_config", type=str, default="lasso.py")
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--n_jobs", type=int, default=1, help="Used to parallelize building models ATM.")
    
    args = parser.parse_args()
    main(args)