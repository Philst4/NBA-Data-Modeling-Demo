# Internal imports
import os
from src.utils import set_seed
from src.data.io import get_modeling_data, query_db
from src.model.predicting import predict_sklearn, predict_torch
import pandas as pd
import numpy as np
from torch import nn
from src.model.config_mgmt import load_modeling_config
from src.model.initialization import extract_best_model_hyperparams_from_study

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

def run_prediction_pipeline(
    models,
    selected_date, 
    config,
    modeling_config_filter=""
):
    
    CLEAN_DIR = config['clean_data_dir']
    DB_NAME = config['db_name']
    DB_PATH = os.path.join(CLEAN_DIR, DB_NAME)
    
    # Set seed
    set_seed()

    # Load in modeling data
    modeling_data, features, target = get_modeling_data(DB_PATH, config, selected_date)
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
        
        targets = season_std * modeling_data[target] + season_mean
        targets = targets.astype(int)
    else:
        targets = np.array([])
    
    
    # Get preds
    pred_dfs = []
    for model_name, model in models.items():
        if modeling_data.shape[0] != 0:
            if isinstance(model, nn.Module):
                preds = predict_torch(model, modeling_data, features).squeeze().numpy()
            else:
                preds = predict_sklearn(model, modeling_data, features)

            preds = season_std * preds + season_mean
            preds = np.where(preds >= 0, np.ceil(preds), np.floor(preds)).astype(int)
        else:
            preds = np.array([])

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