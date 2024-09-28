import sys
import os
import argparse
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import yaml
import torch

#### LOCAL IMPORTS
# Local imports
from src.dataloading import (
    SeasonSequenceDataset,
    collate_fn
)

import random

#### MAIN PROGRAM

if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", required=True, type=str, help="Home team (3 letter abbreviation)")
    parser.add_argument("--away", required=True, type=str, help="Away team (3 letter abbreviation)")
    args = parser.parse_args()
    home_team_abbr = args.home
    away_team_abbr = args.away
    
    
    # Read in configuration
    with open('configs/old_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    CLEAN_DIR = config['clean_dir']
    DB_NAME = config['db_name']
    DB_PATH = os.path.join(CLEAN_DIR, DB_NAME)

    # ---- BEGIN SETUP ---- 
    setup = config['setups']['A']
    
    # (A) SET UP DATALOADING
    dataloading_config = setup['dataloading_config']
    ssd_blueprint = dataloading_config['ssd_blueprint']
    
    # Initialize ssd
    ssd  = SeasonSequenceDataset(
        db_path=DB_PATH,
        blueprint=ssd_blueprint
    )
    
    # (B) LOAD IN SCRIPTED MODEL FOR INFERENCE
    # Load in model, set it for inference
    model = torch.jit.load('models/base_model.pt')
    model.eval()
    
    date = None    
    if not date:
        date = str(datetime.now().date())
        date = '2024-04-16'
        
    # (1) Load in data up to that date
    partial_season_data, season = ssd.get_partial_season_data(date)
    # Pass through collator
    partial_season_data = collate_fn([partial_season_data])
        
    # (2) Find ID's of specified teams
    home_team_id = ssd._query_team_id(season, home_team_abbr)[0][0]
    away_team_id = ssd._query_team_id(season, away_team_abbr)[0][0]
        
    # (3) Get OHE's for matchup
    kq_new = torch.stack(
        [ssd.ohe(home_team_id), ssd.ohe(away_team_id)], dim=0
    )
        
    # (4) Add new vals to partial season data
    model_input = {}
    for key, val in partial_season_data.items():
        if isinstance(val, torch.Tensor):
            if key == "kq":
                new_val = kq_new
            elif key == "padding_masks":
                new_val = torch.ones((val.shape[2:]))
            else:
                new_val = torch.zeros((val.shape[2:]))
            # Unsqueeze along batch and time dimension
            new_val = new_val.unsqueeze(dim=0).unsqueeze(dim=1)
            # Add to original along time dimension
            model_input[key] = torch.concat((val, new_val), dim=1)
        else:
            # Model wants Dict[str, torch.Tensor]; won't 
            # add values of other type to model_input
            pass
        
    # (5) Pass sequence through model, only keep last prediction
    with torch.no_grad():
        model_output = (model(model_input))
        raw_pred = model_output[:, -1]
            
    # (6) Clean up prediction
    std = 15.88
    
    pred = int(torch.round(raw_pred * std).item())
    if pred == 0:
        pred = int(torch.sign(raw_pred).item())
    
    # (7) Print result
    output_str = f" * '{away_team_abbr}' @ '{home_team_abbr}' at end of 2023-24 season: "
    if pred > 0:
        output_str += f"{home_team_abbr} wins by {pred}"
    else:
        output_str += f"{away_team_abbr} wins by {-pred}"
    print(output_str)
        