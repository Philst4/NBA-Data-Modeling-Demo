# Standard library imports
import sys
import os
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import pandas as pd

# Local imports
from config import (
    RAW_DATA_DIR,
    CLEAN_DATA_DIR,
)
from utils.data_cleaning import (
    read_raw_from_csv,
    drop_cols,
    convert_types, 
    add_cols,
    get_summary_stats,
    mirror,
    deal_w_NaNs,
    make_team_id_df,
    save_as_db
)

###### Data-specific functionality
def make_wl_map():
    return lambda x : {'L' : 0, 'W' : 1}[x]    

# Used for changing the 'TEAM_ID' to be more reasonable
def make_id_map(games : pd.DataFrame, old_id_col : str):
    unique_ids = sorted(list(games[old_id_col].unique()))
    return lambda x : dict(zip(unique_ids, range(len(unique_ids))))[x]

def make_is_home_map():
    return lambda x : int('vs.' in x)

def fill_plus_minus(games : pd.DataFrame) -> None:
    print(f" * Filling 'PLUS_MINUS' ...")
    games.loc[:, 'PLUS_MINUS'] = games.loc[:, 'PTS_for'].astype(int) - games.loc[:, 'PTS_ag'].astype(int)
    return


### MAIN SCRIPT ###

if __name__ == "__main__":
    print(" --- RUNNING DATA CLEANING SCRIPT --- ")
    
    # (0) Read in games
    games = read_raw_from_csv(RAW_DATA_DIR, 'raw.csv')
    
    # NOTE: Each function modifies a reference to 'games'
    # (1) Drop unnecessary columns
    cols_to_drop = ['FG_PCT', 'FT_PCT', 'FG3_PCT']
    drop_cols(games, cols_to_drop)
    
    # (2) Convert the data types appropriately 
    # TODO revisit
    convert_types(games)
    
    # (3) Engineer new features, add as columns
    cols_to_add = ['TARGET', 'NEW_TEAM_ID', 'IS_HOME']
    dependencies = [['WL'], ['TEAM_ID'], ['MATCHUP']]
    maps = [make_wl_map(), make_id_map(games, 'TEAM_ID'), make_is_home_map()]
    add_cols(games, cols_to_add, dependencies, maps)
    
    # (3.5) 
    team_id_df = make_team_id_df(games)
    
    # (4) 'Mirror' data to contain opposing stats
    leave_out_cols_mirroring = [
        'SEASON_ID',
        'GAME_DATE', 
        'GAME_ID', 
        'MATCHUP', 
        'PLUS_MINUS', 
        'WL', 
        'TARGET'
    ]
    mirror(games, leave_out_cols_mirroring)
    
    # (5) Data-specific cleaning processes
    fill_plus_minus(games)
    
    # (6) Calculate, save means + stds for each stat
    leave_out_cols_summary = [
        'SEASON_ID',
        'TEAM_ID',
        'TEAM_ABBREVIATION',
        'TEAM_NAME',
        'GAME_ID',
        'GAME_DATE',
        'MATCHUP',
        'WL',
        'TARGET',
        'NEW_TEAM_ID'
    ]
    summary_stats = get_summary_stats (games, leave_out_cols_summary)
    
    # (7) Deal with remaining NaN's
    deal_w_NaNs(games)
    
    # (8) Finally, save cleaned data + other relevant
    save_as_db(
        games, 
        CLEAN_DATA_DIR, 
        'my_database.db',
        'my_table'
    )
    save_as_db(
        summary_stats,
        CLEAN_DATA_DIR,
        'my_database.db',
        'summary_stats',
        index=True
    )
    save_as_db(
        team_id_df,
        CLEAN_DATA_DIR,
        'my_database.db',
        'team_metadata'
    )
    print('Success!')