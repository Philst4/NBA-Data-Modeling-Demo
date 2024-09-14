# Standard library imports
import sys
import os
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import yaml
import pandas as pd

# Local imports
from utils.etl import (
    read_from_csv,
    save_to_db
)

if __name__ == "__main__":
    print("--- RUNNING METADATA ETL SCRIPT --- ")
    
    # (0) Read in configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    RAW_DIR = config['raw_dir']
    RAW_FILE_NAME = config['raw_file_name']
    RAW_FILE_PATH = os.path.join(RAW_DIR, RAW_FILE_NAME)
    CLEAN_DIR = config['clean_dir']
    DB_NAME = config['db_name']
    GAME_METADATA_NAME = config['game_metadata_table_name']
    GAME_META_COLS = config['game_meta_cols']
    TEAM_META_COLS = config['team_meta_cols']
    TEAM_METADATA_NAME = config['team_metadata_table_name']
    
    # (1) Read in raw data
    games = read_from_csv(RAW_FILE_PATH)
    
    # (2) Keep what is needed
    # Keep relevant columns
    game_metadata = games.loc[:, GAME_META_COLS]
    team_metadata = games.loc[:, TEAM_META_COLS]
    
    # Remove repeats from team_metadata
    team_metadata = team_metadata.drop_duplicates(subset=['SEASON_ID', 'TEAM_ID'])
    team_metadata = team_metadata.reset_index(drop=True)
    
    # (3) Set types (N/A)
    # Not needed
    pass
    
    # (4) Save (TODO update save_to_db for primary keys)
    # Save game_metadata
    save_to_db(
        game_metadata,
        CLEAN_DIR,
        DB_NAME,
        GAME_METADATA_NAME
    )
    
    # Save team_metadata
    save_to_db(
        team_metadata,
        CLEAN_DIR,
        DB_NAME,
        TEAM_METADATA_NAME
    )
