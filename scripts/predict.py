import sys
import os
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
    collate_fn,
    DataLoader # Base PyTorch DataLoader
)

from src.training import (
    LightningModel
)

import random

#### MAIN PROGRAM

if __name__ == "__main__":
    # Read in configuration
    with open('config.yaml', 'r') as file:
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
    model = torch.jit.load('models/scripted_model.pt')
    model.eval()
    
    print("-- STARTING MAIN -- ")
    while True:
        home_team_abbr = input("Enter Home Team: ")
        away_team_abbr = input("Enter Away Team: ")
        date = input("Enter Date: ")
        
        if not date:
            date = str(datetime.now().date())
        
        # (1) Load in data up to that date
        partial_season_data, season = ssd.get_partial_season_data(date)
        
        # (2) Find ID's of specified teams
        if False:
            home_team_id = ssd._query_team_id(season, home_team_abbr)[0][0]
            away_team_id = ssd._query_team_id(season, away_team_abbr)[0][0]
            print(home_team_id)
            print(away_team_id)
        
        # (3) Get OHE's; need to rewrite ohe
        print(ssd.ohe([0]))
        print(type(ssd.ohe([0])))
        input()
        kq_new = torch.stack([ssd.ohe(0), ssd.ohe(0)], dim=0)
        print(f"kq_new : {kq_new.shape}")
        input()
        
        for key, val in partial_season_data.items():
            if isinstance(val, torch.Tensor):
                print(f"OLD : {key}, {val.shape}")
                if key == "kq":
                    new_val = kq_new
                elif key == "padding_masks":
                    new_val = torch.ones((val.shape[1:]))
                else:
                    new_val = torch.zeros((val.shape[1:]))
                new_val.unsqueeze(dim=0)
                partial_season_data[key] = torch.concat((val, new_val), dim=0)
                print(f"NEW : {key}, {val.shape}")
            else:
                pass
        
        # Pass through collator
        partial_season_data = collate_fn([partial_season_data])
        
        
        if False:
            # TODO: FIX UNDEFINED BEHAVIOR
            if not home_team_id:
                print(' * Home team not found')
                home_team_abbr = 'RNG_HOME'
                home_team_id = random.randint(0, 30)
            if not away_team_id:
                print(' * Away team not found')
                away_team_abbr = 'RNG_AWAY'
                away_team_id = random.randint(0, 30)
        
        # (3) Append matchup in question so model makes prediction
        # NOTE: Only need to append matchup correctly; since
        #   statistical data won't be used for the prediction

        kq_new = ssd.ohe([])
        
        for key, val in partial_season_data.items():
            if isinstance(val, torch.Tensor):
                print(f"{key} : {val.shape}")
            else:
                print(f"{key} : {len(val)} sequence")
        
        exit()
        # (4) Pass sequence through model, only keep last prediction
        sequence = collate_fn([sequence])
        kq, v, _, _ = sequence
        with torch.no_grad():
            v, means, stds = normalize(v)
            model_output = denormalize(model(kq, v), means, stds)
            raw_pred = model_output[:, -1]
            
        
        # (5) Clean up prediction
        pred = int(torch.round(raw_pred).item())
        if pred == 0:
            pred = int(torch.sign(raw_pred).item())
        
        # (6) Print result
        output_str = f"'{away_team_abbr}' @ '{home_team_abbr}' on '{date} => "
        if pred > 0:
            output_str += f"{home_team_abbr} wins by {pred}"
        else:
            output_str += f"{away_team_abbr} wins by {-pred}"
        print(output_str)
        