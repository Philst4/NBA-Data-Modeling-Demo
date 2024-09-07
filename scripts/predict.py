print(" * Handling imports ...")

#### STD IMPORTS
import sys
import os
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

#### EXTERNAL IMPORTS
import torch


#### LOCAL IMPORTS
from config import (
    CLEAN_DATA_DIR,
    MODEL_DIR
)
from utils.data_loading import (
    SeasonSequenceDataset
)

from utils.modeling import (
    normalize,
    denormalize
)
from architectures.model_0 import (
    collate_fn
)

import random

#### MAIN PROGRAM

if __name__ == "__main__":
    print(" * Setting configurations ... ")
    db_path = '/'.join((CLEAN_DATA_DIR, 'my_database.db'))
    table_name = 'my_table'
    kq_cols = ['NEW_TEAM_ID_for', 'NEW_TEAM_ID_ag']
    v_cols = ['PLUS_MINUS']
    t_cols = ['PLUS_MINUS']
    data_cols = kq_cols + v_cols + t_cols

    # Instantiate SSD for accessing data (use a lot of default values)
    print(" * Instantiating database access ...")
    ssd = SeasonSequenceDataset(
        db_path,
        table_name,
        data_cols=data_cols
    )
    
    # Load in model, set it for inference
    print(" * Loading in model ... ")
    model_path = '/'.join((MODEL_DIR, 'model_0', 'version_0', 'scripted_model.pt'))
    model = torch.jit.load(model_path)
    model.eval()
    
    print("-- STARTING MAIN -- ")
    while True:
        home_team_abbr = input("Enter Home Team: ")
        away_team_abbr = input("Enter Away Team: ")
        date = input("Enter Date: ")
        
        if not date:
            date = str(datetime.now().date())
        
        # (1) Load in data up to that date
        sequence, season = ssd.get_partial_season_sequence(date, include_meta=False)
        
        # (2) Find ID's of specified teams
        home_team_id = ssd.get_team_id(home_team_abbr, season)
        away_team_id = ssd.get_team_id(away_team_abbr, season)
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
        if not len(sequence) > 0:
            print('Please choose a valid date')
            break
        matchup_to_predict = [0] * len(sequence[-1])
        matchup_to_predict[0] = home_team_id
        matchup_to_predict[1] = away_team_id
        sequence.append(tuple(matchup_to_predict))
        
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
        