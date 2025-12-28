# Standard library imports
import sys
import os
import argparse

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import yaml
import pandas as pd
import sqlite3

# Local imports
from src.data.io import (
    read_from_parquet,
    save_as_parquet,
)

from src.data.cleaning import (
    add_cols,
    make_id_map,
    drop_cols,
    fill_plus_minus,
    mirror,
    deal_w_NaNs,
)

from src.data.processing import (
    get_normalized_by_season
)

def clean_metadata(config):
    print("--- RUNNING METADATA CLEANING SCRIPT --- ")
    
    # (0) Read in configuration
    RAW_DIR = config['raw_data_dir']
    RAW_FILENAME = config['raw_filename']
    RAW_FILE_PATH = os.path.join(RAW_DIR, RAW_FILENAME)
    if not os.path.exists(os.path.join(RAW_DIR, RAW_FILENAME)):
        print(f"Raw file '{RAW_FILE_PATH}' from not found, make sure '{config}' specifies a valid 'raw_data_dir' and 'raw_filename'")
        return
        
    CLEAN_DIR = config['clean_data_dir']
    DB_NAME = config['db_name']
    
    metadata_config = config['metadata']
    GAME_METADATA_NAME = metadata_config['games']['table_name']
    GAME_META_COLS = metadata_config['games']['columns']
    TEAM_METADATA_NAME = metadata_config['teams']['table_name']
    TEAM_META_COLS = metadata_config['teams']['columns']
    
    # (1) Read in raw data
    games = read_from_parquet(RAW_FILE_PATH)
    
    # (1.5) Turn 'SEASON_ID' to int
    games['SEASON_ID'] = games['SEASON_ID'].astype(int)
    
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
    game_metadata = mirror(
        game_metadata, 
        cols_to_mirror=['NEW_TEAM_ID', 'TEAM_ABBREVIATION']
    ) 
    drop_cols(game_metadata, ['TEAM_ID'])
    
    # (4) Save game_metadata, team_metadata as parquet
    save_as_parquet(
        game_metadata,
        CLEAN_DIR,
        GAME_METADATA_NAME + '.parquet'
    )
    
    save_as_parquet(
        team_metadata,
        CLEAN_DIR,
        TEAM_METADATA_NAME + '.parquet'
    )


def clean_data(config):
    print("--- RUNNING DATA CLEANING SCRIPT ---")
    
    # (0) Read in configuration
    RAW_DIR = config['raw_data_dir']
    RAW_FILE_NAME = config['raw_filename']
    RAW_FILE_PATH = os.path.join(RAW_DIR, RAW_FILE_NAME)
    CLEAN_DIR = config['clean_data_dir']
    
    # Loop will start here
    MAIN_TABLE_NAME = config['main_table_name']
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
    games = read_from_parquet(RAW_FILE_PATH)
    games['GAME_WON'] = games['WL'].map({'W' : 1, 'L' : 0})
    
    drop_cols(games, COLS_TO_DROP_A)
    
    # (2) Get means, stds and save
    means_stds_by_season = (
        games
        .sort_values(by='SEASON_ID')
        .drop(columns=[
            'UNIQUE_ID', 
            'GAME_ID', 
            'TEAM_ID', 
            'IS_HOME', 
            'GAME_WON'
        ])
        .groupby('SEASON_ID')
        .agg(['mean', 'std'])
        .reset_index()
    )
    
    # Flatten MultiIndex columns
    means_stds_by_season.columns = [
        f"{col}_{stat}" if stat else col
        for col, stat in means_stds_by_season.columns
    ]
    
    means_stds_by_season['SEASON_ID'] = means_stds_by_season['SEASON_ID'].astype(int)

    save_as_parquet(
        means_stds_by_season,
        CLEAN_DIR,
        "team_stats_by_game_means_stds_by_season.parquet",
        w_reduced_precision=True
    )

    # (3) Add new features, drop unneeded after
    drop_cols(games, ['TEAM_ID'])
    
    drop_cols(games, COLS_TO_DROP_B)

    # (4) 'Mirror' data to contain opposing stats, drop unneeded
    games = mirror(
        games, 
        cols_not_to_mirror=['UNIQUE_ID', 'GAME_ID']
    )
    drop_cols(games, COLS_TO_DROP_C)
    games = games.drop(columns=games.filter(like='IS_HOME'))
    
    # (5) Data-specific cleaning processes
    fill_plus_minus(games)
    
    # (6) Deal with remaining NaN's
    deal_w_NaNs(games)
    
    # (7) Save cleaned data
    save_as_parquet(
        games.sort_values(by='UNIQUE_ID'),
        CLEAN_DIR,
        MAIN_TABLE_NAME + '.parquet'
    )
    

def main(args):
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    clean_metadata(config)
    clean_data(config)

if __name__ == "__main__":
    # (-1) Deal with arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=os.environ.get("CONFIG_PATH", "configs/config.yaml"), help="Config path")
    args = parser.parse_args()
    
    main(args)
    