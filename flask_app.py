import os
import argparse
import re
import yaml
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import date, datetime
from pathlib import Path
from flask import Flask, render_template, request, session

# Internal imports
from src.utils import set_seed
from src.data.io import get_modeling_data, query_db
from src.model.predicting import predict_sklearn, predict_torch
from src.model.config_mgmt import load_modeling_config
from src.model.initialization import extract_best_model_hyperparams_from_study


app = Flask(__name__)
app.secret_key = os.urandom(24)  # needed to use session

# Load config
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
CLEAN_DATA_DIR = config['clean_data_dir']
DB_NAME = config['db_name']
DB_PATH = os.path.join(CLEAN_DATA_DIR, DB_NAME)
MODELING_CONFIG_DIR = config['modeling_config_dir']
MODEL_STORAGE = config['model_storage']
OPTUNA_STORAGE = config['optuna_storage']

# Define min-max date
MIN_DATE = date(1983, 1, 1)
MAX_DATE = date.today()
DEFAULT_DATE = date(2025, 4, 13) # Don't change.

# Sample data to get model input dim
_, features, _, _ = get_modeling_data(DB_PATH, DEFAULT_DATE)
INPUT_DIM = len(features)

# Cache to avoid recomputations
# Key is '(date, modeling_config)' tuple ATM
CACHE = {}

def make_predictions_df(modeling_data, preds, targets):
    predictions_df = pd.DataFrame({
        "Away Team": modeling_data['TEAM_ABBREVIATION_ag'].values,
        "Home Team": modeling_data['TEAM_ABBREVIATION_for'].values,
        "Predicted Home Team Plus-Minus": preds,
        "True Home Team Plus-Minus": targets
    })

    predictions_df['Plus Minus Off By'] = (
        predictions_df['Predicted Home Team Plus-Minus'] - predictions_df['True Home Team Plus-Minus']
    )

    predictions_df['Predicted Winner'] = predictions_df.apply(
        lambda row: row['Home Team'] if row['Predicted Home Team Plus-Minus'] > 0 else row['Away Team'],
        axis=1
    )

    predictions_df['True Winner'] = predictions_df.apply(
        lambda row: row['Home Team'] if row['True Home Team Plus-Minus'] > 0 else row['Away Team'],
        axis=1
    )

    predictions_df['Predicted Correct Winner?'] = (
        predictions_df['Predicted Winner'] == predictions_df['True Winner']
    )

    predictions_df = predictions_df[
        [
            "Away Team",
            "Home Team",
            "Predicted Home Team Plus-Minus",
            "Predicted Winner",
            "True Home Team Plus-Minus",
            "True Winner",
            "Plus Minus Off By",
            "Predicted Correct Winner?"
        ]
    ]

    return predictions_df.reset_index(drop=True)


def load_models(
    modeling_config_dir, 
    optuna_storage, # For extracting hparams
    model_storage, # For extracting saved model file
    input_dim, # For properly initializing models
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
        model_path = os.path.join(model_storage, modeling_config.model_filename)

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
        
        models[modeling_config.model_name] = model
    
    return models
    
# Load in models (constant!)
MODELS = load_models(
    MODELING_CONFIG_DIR,
    OPTUNA_STORAGE,
    MODEL_STORAGE,
    INPUT_DIM,
)

def run_prediction_pipeline(
    models,
    selected_date, 
    modeling_config_filter=""
):

    # Set seed
    set_seed()

    # Load in modeling data
    modeling_data, features, target = get_modeling_data(DB_PATH, selected_date)
    modeling_data = modeling_data.sort_values(by=['UNIQUE_ID'])
    modeling_data = modeling_data[modeling_data['MATCHUP'].str.contains('vs.')]

    if modeling_data.shape[0] != 0:
        means_stds = query_db(DB_PATH, f"""
            SELECT SEASON_ID, PLUS_MINUS_for_mean, PLUS_MINUS_for_std
            FROM {config['main_table_name']}_means_stds
        """)
        
        season_condn = means_stds['SEASON_ID'] == list(modeling_data['SEASON_ID'])[0]
        season_mean = means_stds.loc[season_condn, "PLUS_MINUS_for_mean"].values
        season_std = means_stds.loc[season_condn, "PLUS_MINUS_for_std"].values
    
    pred_dfs = []
    for model_name, model in models.items():
        if modeling_data.shape[0] != 0:
            if isinstance(model, nn.Module):
                preds = predict_torch(model, modeling_data, features).squeeze().numpy()
            else:
                preds = predict_sklearn(model, modeling_data, features)

            preds = season_std * preds + season_mean
            targets = season_std * modeling_data[target] + season_mean

            preds = np.where(preds >= 0, np.ceil(preds), np.floor(preds)).astype(int)
            targets = targets.astype(int)
        else:
            preds, targets = np.array([]), np.array([])

        predictions_df = make_predictions_df(modeling_data, preds, targets)
        mae = abs(predictions_df['Plus Minus Off By']).mean()
        accuracy = (
            100 * predictions_df['Predicted Correct Winner?'].sum() / len(predictions_df)
            if len(predictions_df) else 0
        )

        pred_dfs.append({
            "model_name": model_name,
            "predictions": predictions_df,
            "mae": mae,
            "accuracy": accuracy
        })

    # Also load the true game results
    unique_ids = list(modeling_data['UNIQUE_ID'])
    points = query_db(DB_PATH, f"SELECT UNIQUE_ID, PTS_for, PTS_ag FROM game_data WHERE UNIQUE_ID in {tuple(unique_ids)}")
    modeling_data = pd.merge(modeling_data, points, on='UNIQUE_ID').sort_values(by=['UNIQUE_ID'])

    results_df = pd.DataFrame({
        "Away Team": modeling_data['TEAM_ABBREVIATION_ag'].values,
        "Home Team": modeling_data['TEAM_ABBREVIATION_for'].values,
        "Away Team Points": modeling_data['PTS_ag'].values,
        "Home Team Points": modeling_data['PTS_for'].values,
        "Home Team Plus-Minus": targets,
    }).reset_index(drop=True)

    return pred_dfs, results_df

@app.route("/", methods=["GET", "POST"])
def index():
    selected_date = DEFAULT_DATE
    pred_dfs, results_df = None, None

    if request.method == "POST":
        selected_date_str = request.form.get("game_date")
        selected_date = datetime.strptime(selected_date_str, "%Y-%m-%d").date()

    # Create cache key (unique per date + config)
    cache_key = selected_date

    # ✅ Use cache if available
    if cache_key in CACHE:
        pred_dfs, results_df = CACHE[cache_key]
        print(f"Using cached results for {selected_date}")
    else:
        print(f"Computing new results for {selected_date}")
        pred_dfs, results_df = run_prediction_pipeline(
            MODELS,
            selected_date,
        )
        CACHE[selected_date] = (pred_dfs, results_df)

    for df in pred_dfs or []:
        df["predictions_html"] = df["predictions"].to_html(
            index=False, classes="table table-sm table-striped align-middle text-center"
        )

    results_df_html = None
    if results_df is not None:
        results_df_html = results_df.to_html(
            index=False, classes="table table-sm table-bordered align-middle text-center"
        )


    return render_template(
        "index.html",
        min_date=MIN_DATE,
        max_date=MAX_DATE,
        pred_dfs=pred_dfs,
        results_df_html=results_df_html,
        selected_date=selected_date
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/config.yaml", help="Config path")
    app.run(debug=True)
