# Standard library imports
import sys
import os
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import yaml
import pandas as pd
import sqlite3

# Local imports
from utils.etl import (
    read_from_csv, 
    make_id_map,
    add_cols,
    get_summary_stats,
    drop_cols,
    mirror,
    deal_w_NaNs,
    save_to_db,
)

# Used to fill 'PLUS_MINUS' N/A's
def fill_plus_minus(games : pd.DataFrame) -> None:
    print(f" * Filling 'PLUS_MINUS' ...")
    games.loc[:, 'PLUS_MINUS_for'] = games.loc[:, 'PTS_for'].astype(int) - games.loc[:, 'PTS_ag'].astype(int)
    games.loc[:, 'PLUS_MINUS_ag'] = -games.loc[:, 'PLUS_MINUS_for']
    return


if __name__ == "__main__":
    print("--- RUNNING DATA ETL SCRIPT ---")
    
    # (0) Read in configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    RAW_DIR = config['raw_dir']
    RAW_FILE_NAME = config['raw_file_name']
    RAW_FILE_PATH = os.path.join(RAW_DIR, RAW_FILE_NAME)
    CLEAN_DIR = config['clean_dir']
    DB_NAME = config['db_name']
    
    data_config = config['data']
    # Loop will start here
    curr_table = data_config[0]
    MAIN_TABLE_NAME = curr_table['table_name']
    SUMMARY_TABLE_NAME = MAIN_TABLE_NAME + '_summary'
    cols_to_drop = curr_table['cols_to_drop']
    
    # Make sure we don't lose info needed for the program to run correctly
    NEEDED_FOR_SUMMARY = ('SEASON_ID')
    NEEDED_FOR_MIRRORING = ('UNIQUE_ID', 'GAME_ID', 'IS_HOME')
    COLS_TO_DROP_A = [col for col in cols_to_drop if col not in NEEDED_FOR_SUMMARY and col not in NEEDED_FOR_MIRRORING]
    COLS_TO_DROP_B = [col for col in cols_to_drop if col not in COLS_TO_DROP_A and col not in NEEDED_FOR_MIRRORING]
    COLS_TO_DROP_C = [col for col in cols_to_drop if col not in COLS_TO_DROP_A and col not in COLS_TO_DROP_B]
    
    # (1) Read in raw data, drop unneeded
    games = read_from_csv(RAW_FILE_PATH)
    drop_cols(games, COLS_TO_DROP_A)
    
    # (2) Convert types (revisit)
    pass

    # (3) Add new features, drop unneeded after
    # For team_metadata, we want 'NEW_TEAM_ID'
    cols_to_add_t = ['NEW_TEAM_ID']
    dependencies_t = [['TEAM_ID']]
    maps_t = [make_id_map(games, 'TEAM_ID')]
    add_cols(
        games, 
        cols_to_add_t, 
        dependencies_t, 
        maps_t
    )
    drop_cols(games, ['TEAM_ID'])

    # (4) Generate table of summarizing stats, drop unneeded
    summary_stats = get_summary_stats(
        games, 
        leave_out_cols=['UNIQUE_ID', 'GAME_ID', 'NEW_TEAM_ID']
    )
    
    drop_cols(games, COLS_TO_DROP_B)

    # (5) 'Mirror' data to contain opposing stats, drop unneeded
    mirror(
        games, 
        cols_not_to_mirror=['UNIQUE_ID', 'GAME_ID']
    )
    drop_cols(games, COLS_TO_DROP_C)
    
    # (6) Data-specific cleaning processes
    fill_plus_minus(games)
    
    # (7) Deal with remaining NaN's
    deal_w_NaNs(games)
    
    # Convert types...?
    games['NEW_TEAM_ID_ag'] = games['NEW_TEAM_ID_ag'].astype(int)
    
    # (8) Save cleaned data
    save_to_db(
        games, 
        CLEAN_DIR,
        DB_NAME,
        MAIN_TABLE_NAME
    )
    
    save_to_db(
        summary_stats, 
        CLEAN_DIR,
        DB_NAME,
        SUMMARY_TABLE_NAME,
        index=True
    )