import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import argparse
import yaml
import pandas as pd

# Local imports
from src.data.io import (
    save_as_parquet,
    read_from_parquet
)

from src.data.ingesting import (
    ingest_from_leaguegamefinder
)

def main(args):
    
    print("--- RUNNING DATA INGESTION SCRIPT ---")
    
    # (0) Read configuration
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    if not config['ingest_data']:
        print(f"'{args.config_path}' specifies to not ingest data")
        return
    
    RAW_DATA_DIR = config['raw_data_dir']
    RAW_FILE_NAME = config['raw_filename']
    RAW_FILE_PATH = os.path.join(RAW_DATA_DIR, RAW_FILE_NAME)
    
    # Loop starts here
    ingestion_fn = ingest_from_leaguegamefinder # ingestion function
    
    # (1) Check if raw exists, and
    # (2) Read in relevant data from NBA API
    if os.path.exists(RAW_FILE_PATH):
        existing_games = read_from_parquet(RAW_FILE_PATH)
        last_read = existing_games['GAME_DATE'].max()
        print(f"Existing reserve at '{RAW_FILE_PATH}' exists; last game in reserve on '{last_read}'")
        print(f" Will read in all seasons with potential new games")
        new_games = ingestion_fn(start_date=last_read)
    else:
        existing_games = None
        print(f"The file {RAW_FILE_PATH} does not exist")
        new_games = ingestion_fn(start_season=1983)
    
    # (3) Check how many games were read in
    if new_games is None:
        print(' * No games ingested')
        exit()
    len_new = len(new_games)
    print(f" * {len_new} total games ingested")

    # (4) Add 'IS_HOME'
    new_games['IS_HOME'] = new_games.apply(lambda row : int('vs.' in row['MATCHUP']), axis=1)
    
    # (5) Add 'UNIQUE_ID', composite key of 'GAME_ID' and 'IS_HOME'
    new_games['UNIQUE_ID'] = 10 * new_games['GAME_ID'].astype(int) + new_games['IS_HOME']
    
    # (6) Handle integrating into existing reserve
    if existing_games is not None:
        len_existing = len(existing_games)
        print(f" * {len_existing} games found in existing reserve")
        games = pd.concat((existing_games, new_games), ignore_index=True)    
        games = games.drop_duplicates(subset=['UNIQUE_ID'])
        len_total = len(games)
        n_duplicates = (len_existing + len_new) - len_total
        print(f" * {n_duplicates} of ingested games found in existing reserve, added {len_total - len_existing} new games")
    else:
        games = new_games.drop_duplicates(subset=['UNIQUE_ID'])
        len_total = len(games)
    
    # (7) Save file
    save_as_parquet(
        df=games, 
        write_dir=RAW_DATA_DIR,
        parquet_name=RAW_FILE_NAME,
    )
    print(f"{len_total} games in the reserve at '{RAW_FILE_PATH}'")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=os.environ.get("CONFIG_PATH", "configs/config.yaml"), help="Config path")
    args = parser.parse_args()
    main(args)