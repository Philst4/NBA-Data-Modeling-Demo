import sys
import os
import argparse

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import yaml
import optuna

# Internal imports
from src.model.config_mgmt import (
    load_modeling_config,
)

from src.data.io import (
    get_modeling_data
)

from src.model.tuning import make_backtest_objective
    
def main():
    """
    Makes an objective function using the specified configuration, 
    and tunes according to that objective function.
    """
    
    # Read configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    optuna_storage = config["optuna_storage"]
    
    # Create argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeling_config", type=str, default="modeling_configs/lasso.py")
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--n_jobs", type=int, default=1, help="Used to parallelize building models ATM.")
        
    # Extract args (including modeling config)
    args = parser.parse_args()
    modeling_config = args.modeling_config
    n_trials = args.n_trials
    n_jobs = args.n_jobs
        
    # Read modeling config
    modeling_config = load_modeling_config(args.modeling_config)
    
    # Get modeling data
    modeling_data, features, target = get_modeling_data()
    
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
        modeling_config.n_epochs
    )
    
    # Make study
    study = optuna.create_study(
        study_name=modeling_config.study_name,
        direction="minimize",
        storage=optuna_storage,
        load_if_exists=True
    )
    
    # Tune!
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    
    print(f"\nBest Trial:\n")
    print(study.best_trial)

if __name__ == "__main__":
    main()