import sys
import os
import argparse
from pathlib import Path

# External imports
import yaml
import joblib
import streamlit as st
import pandas as pd
from datetime import date
import numpy as np
import torch
import torch.nn as nn
import re


# Internal imports
from src.utils import set_seed

from src.data.io import (
    get_modeling_data,
    query_db
)

from src.model.predicting import (
    predict_sklearn,
    predict_torch
)

from src.model.config_mgmt import (
    load_modeling_config
)

from src.model.initialization import (
    extract_best_model_hyperparams_from_study
)


def make_predictions_df(modeling_data, preds, targets):
    predictions_df = pd.DataFrame({
        "Away Team": modeling_data['TEAM_ABBREVIATION_ag'].values,
        "Home Team": modeling_data['TEAM_ABBREVIATION_for'].values,
        "Predicted Home Team Plus-Minus": preds,
        "True Home Team Plus-Minus": targets
    })

    # Absolute error
    predictions_df['Plus Minus Off By'] = (
        predictions_df['Predicted Home Team Plus-Minus'] - predictions_df['True Home Team Plus-Minus']
    )

    # Predicted winner
    predictions_df['Predicted Winner'] = predictions_df.apply(
        lambda row: row['Home Team'] if row['Predicted Home Team Plus-Minus'] > 0 else row['Away Team'],
        axis=1
    )

    # True winner
    predictions_df['True Winner'] = predictions_df.apply(
        lambda row: row['Home Team'] if row['True Home Team Plus-Minus'] > 0 else row['Away Team'],
        axis=1
    )

    # Correct prediction?
    predictions_df['Predicted Correct Winner?'] = predictions_df['Predicted Winner'] == predictions_df['True Winner']

    # Reorder columns
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


def main(args):
    
    # Read from config file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    CLEAN_DATA_DIR = config['clean_data_dir']
    DB_NAME = config['db_name']
    DB_PATH = os.path.join(CLEAN_DATA_DIR, DB_NAME)
    MODELING_CONFIG_DIR = config['modeling_config_dir']
    MODEL_STORAGE = config['model_storage']
    OPTUNA_STORAGE = config['optuna_storage']
    
    # Set seed
    set_seed()
    
    # Initial app setup (give date to select)
    st.title("NBA Game Outcome Predictor")
    
    min_date = date(1983, 1, 1)        # Jan 1, 1983
    max_date = date.today()  
    
    selected_date = st.date_input(
        "Select game date", 
        value="2025-04-13", # Last regular season NBA day
        min_value=min_date,
        max_value=max_date
    )
    
    # Load in modeling data using date
    modeling_data, features, target = get_modeling_data(DB_PATH, selected_date)
    modeling_data = modeling_data.sort_values(by=['UNIQUE_ID'])

    # Only keep one side of matchup
    modeling_data = modeling_data[modeling_data['MATCHUP'].str.contains('vs.')]
    
    # Loop over loading in modeling_configs, making predictions, displaying
    modeling_config_paths = []
    if args.modeling_config != "":
        modeling_config_paths.append(os.path.join(MODELING_CONFIG_DIR, args.modeling_config))
    else:
        for modeling_config_path in Path(MODELING_CONFIG_DIR).iterdir():
            if modeling_config_path.is_file() and not modeling_config_path.name.startswith('.') and not modeling_config_path.name == '__init__.py':
                modeling_config_paths.append(modeling_config_path)
    
    # Sort so that baseline_*.py files come LAST
    pattern = re.compile(r'baseline_\d+\.py')

    modeling_config_paths.sort(
        key=lambda p: (1 if pattern.match(Path(p).name) else 0, p.name)
    )
    
    for modeling_config_path in modeling_config_paths:
    
        # Load in modeling_config
        modeling_config = load_modeling_config(str(modeling_config_path))
        model_path = os.path.join(MODEL_STORAGE, modeling_config.model_filename)
        if not Path(model_path).is_file(): # or model_path.endswith("lgbm_model.joblib"):
            # Move onto next model path; this one doesn't exist
            continue
        
        model_class = modeling_config.model_class
        
        if issubclass(model_class, nn.Module):  
            # Load in study to get best hyperparams
            
            # Extract best hyperparams from study
            model_hyperparams = extract_best_model_hyperparams_from_study(
                modeling_config, 
                OPTUNA_STORAGE
            )
            
            # Initialize model with best hyperparams
            model = modeling_config.model_class(
                input_dim=len(features),
                output_dim=1,
                **model_hyperparams
            )
            
            # Load in weights
            model.load_state_dict(torch.load(model_path))
            model.eval()
        else:
            model = joblib.load(model_path)

        # Have the model make predictions
        if modeling_data.shape[0] != 0:
            if isinstance(model, nn.Module):
                preds = predict_torch(
                    model, 
                    modeling_data,
                    features,
                ).squeeze().numpy()
            else:
                # Revisit. Write a predict_sklearn
                preds = predict_sklearn(
                    model, 
                    modeling_data,
                    features
                )
            
            # Denormalize the predictions and targets
            means_stds = query_db(DB_PATH, f"SELECT SEASON_ID, PLUS_MINUS_for_mean, PLUS_MINUS_for_std FROM {config['main_table_name']}_means_stds")
            season_condn = means_stds['SEASON_ID'] == list(modeling_data['SEASON_ID'])[0]
            season_mean = means_stds.loc[season_condn, "PLUS_MINUS_for_mean"].values
            season_std = means_stds.loc[season_condn, "PLUS_MINUS_for_std"].values
                
            preds = season_std * preds + season_mean
            targets = season_std * modeling_data[target] + season_mean
            
            # Round preds
            preds = np.where(
                preds >= 0,
                np.ceil(preds),
                np.floor(preds)
            ).astype(int)
            targets = targets.astype(int)
                
        else:
            preds = np.array([])
            targets = np.array([])
    
    
        predictions_df = make_predictions_df(modeling_data, preds, targets)
        
        st.write(f"**---- '{modeling_config.model_name}' Predictions for NBA games: ----**")
        st.dataframe(predictions_df)
        st.write(f"Plus-Minus MAE: {abs(predictions_df['Plus Minus Off By']).mean():.3f}")
        n_correct = predictions_df['Predicted Correct Winner?'].sum()
        n_predicted = len(predictions_df)
        st.write(f"Accuracy: {n_correct}/{n_predicted} = {100 * n_correct / n_predicted :.1f}%")
    
    # Now display actual results
    
    # Load in points from game data as well (to display later, not for predictions)
    unique_ids = list(modeling_data['UNIQUE_ID'])
    points = query_db(DB_PATH, f"SELECT UNIQUE_ID, PTS_for, PTS_ag from game_data WHERE UNIQUE_ID in {tuple(unique_ids)}")
    modeling_data = pd.merge(modeling_data, points, on='UNIQUE_ID').sort_values(by=['UNIQUE_ID'])
    
    results_df = pd.DataFrame(
        {
            "Away Team" : modeling_data['TEAM_ABBREVIATION_ag'].values,
            "Home Team" : modeling_data['TEAM_ABBREVIATION_for'].values,
            "Away Team Points" : modeling_data['PTS_ag'].values,
            "Home Team Points" : modeling_data['PTS_for'].values,
            "Home Team Plus-Minus" : targets,
        }
    ).reset_index(drop=True)
    
    st.write(f"**---- Results for NBA games: ----**")
    st.dataframe(results_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeling_config", type=str, default="", help="Modeling config")
    args = parser.parse_args()
    main(args)
    
    

