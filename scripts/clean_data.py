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
from src.data.io import (
    read_from_csv,
    save_to_db
)

from src.data.cleaning import (
    get_summary_stats,
    drop_cols,
    fill_plus_minus,
    mirror,
    deal_w_NaNs,
)

if __name__ == "__main__":
    print("--- RUNNING DATA ETL SCRIPT ---")
    
    # (0) Read in configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    RAW_DIR = config['raw_data_dir']
    RAW_FILE_NAME = config['raw_filename']
    RAW_FILE_PATH = os.path.join(RAW_DIR, RAW_FILE_NAME)
    CLEAN_DIR = config['clean_data_dir']
    DB_NAME = config['db_name']
    
    # Loop will start here
    MAIN_TABLE_NAME = config['main_table_name']
    SUMMARY_TABLE_NAME = MAIN_TABLE_NAME + '_summary'
    cols_to_drop = [
        'SEASON_ID',
        'GAME_ID',
        'TEAM_ABBREVIATION',
        'TEAM_NAME',
        'GAME_DATE',
        'MATCHUP',
        'WL',
        'FT_PCT', 
        'FG_PCT', 
        'FG3_PCT',
        ]
    
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
    drop_cols(games, ['TEAM_ID'])
    
    drop_cols(games, COLS_TO_DROP_B)

    # (4) 'Mirror' data to contain opposing stats, drop unneeded
    games = mirror(
        games, 
        cols_not_to_mirror=['UNIQUE_ID', 'GAME_ID']
    )
    drop_cols(games, COLS_TO_DROP_C)
    
    # (5) Data-specific cleaning processes
    fill_plus_minus(games)
    
    # (6) Deal with remaining NaN's
    deal_w_NaNs(games)
    
    # (7) Save cleaned data
    save_to_db(
        games, 
        CLEAN_DIR,
        DB_NAME,
        MAIN_TABLE_NAME
    )
    