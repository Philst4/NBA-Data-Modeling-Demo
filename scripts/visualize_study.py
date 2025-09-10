# STL imports
import os
import sys
import argparse

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import optuna
import optuna.visualization as vis

def visualize_study(study):

    # Visualize the study
    vis.plot_optimization_history(study).show()
    vis.plot_parallel_coordinate(study).show()
    vis.plot_param_importances(study).show()
    vis.plot_slice(study).show()

def main(args):
    # What we need
    optuna_storage_path = "sqlite:///optuna_studies.db"
    
    # Load study into memory
    study = optuna.load_study(
        study_name=args.study_name, 
        storage=optuna_storage_path
    )
    
    visualize_study(study)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", type=str, default="lasso_study", help="Name of study to visualize")
    args = parser.parse_args()
    main(args)
