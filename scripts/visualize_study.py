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
import yaml
import plotly.tools as tls

# Internal imports
from src.utils import get_metric_quantiles_fig

def visualize_study(study, study_name=""):

    # Visualize the study using optuna.vis
    try:
        vis.plot_optimization_history(study).update_layout(
            title=f"'{study_name}' Optimization History Plot",
        ).show()
        vis.plot_parallel_coordinate(study).update_layout(
            title=f"'{study_name}' Parallel Coordinate Plot",
        ).show()
        vis.plot_param_importances(study).update_layout(
            title=f"'{study_name}' Param Importances Plot",
        ).show()
        vis.plot_slice(study).update_layout(
            title=f"'{study_name}' Slice Plot",
        ).show()
    except:
        print("Problem with optuna.visualizations, potentially b/c they require > 1 trial.")
    
    # Use plot_metric_quantiles as well
    metric_quantiles_fig = get_metric_quantiles_fig(study, study_name=study_name)
    
    tls.mpl_to_plotly(metric_quantiles_fig).update_layout(
        title=f"'{study_name}' Quantile Bands Across Metrics",
        margin=dict(t=100)
    ).show()
    
def main(args):
    # Read configuration
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    OPTUNA_STORAGE = config["optuna_storage"]
    
    # Load study into memory
    study = optuna.load_study(
        study_name=args.study_name, 
        storage=OPTUNA_STORAGE
    )
    
    visualize_study(study, args.study_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=os.environ.get("CONFIG_PATH", "configs/config.yaml"), help="Config path")
    parser.add_argument("--study_name", type=str, default="lasso_study", help="Name of study to visualize")
    args = parser.parse_args()
    main(args)
