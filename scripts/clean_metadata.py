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
from src.cleaning import (
    make_id_map,
    add_cols,
    mirror,
    drop_cols,
)

from src.utils import (
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
    
    metadata_config = config['metadata']
    GAME_METADATA_NAME = metadata_config['games']['table_name']
    GAME_META_COLS = metadata_config['games']['columns']
    TEAM_METADATA_NAME = metadata_config['teams']['table_name']
    TEAM_META_COLS = metadata_config['teams']['columns']
    
    # (1) Read in raw data
    games = read_from_csv(RAW_FILE_PATH)
    
    # (2) Keep what is needed
    # Keep relevant columns
    game_metadata = games.loc[:, GAME_META_COLS]
    team_metadata = games.loc[:, TEAM_META_COLS]
    
    # Remove repeats from team_metadata
    team_metadata = team_metadata.drop_duplicates(subset=['SEASON_ID', 'TEAM_ID'])
    team_metadata = team_metadata.reset_index(drop=True)
    
    # Add new team id to game metadata, team metadata
    cols_to_add = ['NEW_TEAM_ID']
    dependencies = [['TEAM_ID']]
    maps = [make_id_map(games, 'TEAM_ID')]
    add_cols(
        team_metadata, 
        cols_to_add, 
        dependencies, 
        maps
    )
    add_cols(
        game_metadata,
        cols_to_add, 
        dependencies,
        maps
    )
    
    # (3) Set types (N/A)
    # Not needed
    pass
    mirror(
        game_metadata, 
        cols_to_mirror=['NEW_TEAM_ID']
    ) 
    drop_cols(game_metadata, ['TEAM_ID'])
    
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
