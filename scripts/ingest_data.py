import sys
import os
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import yaml
import pandas as pd

# Local imports
from src.ingesting import (
    ingestion_fns,
    save_as_csv
)

def read_last_run(file_path):
    if os.path.exists(file_path):
        # Read the file's contents
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Get the last line
        if lines:
            last_line = lines[-1].strip()
            # Extract the date part from the line
            if last_line.startswith("Last run: "):
                last_run_date = last_line[len("Last run: "):]
                return last_run_date
    return None

def update_last_run(file_path):
    # Get the current date in YYYY-MM-DD format
    current_date = datetime.now().date().isoformat()

    # Check if the file exists
    if os.path.exists(file_path):
        # Read the file's contents
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Update the last run line
        # Assuming the line you want to update is the last line
        if lines:
            lines[-1] = f"Last run: {current_date}\n"
        else:
            lines.append(f"Last run: {current_date}\n")
        
        # Write the contents back to the file
        with open(file_path, 'w') as file:
            file.writelines(lines)
    else:
        # File does not exist, create it and write the initial line
        with open(file_path, 'w') as file:
            file.write(f"Last run: {current_date}\n")

if __name__ == "__main__":
    
    print("--- RUNNING DATA INGESTION SCRIPT ---")
    
    last_read = datetime.today().strftime('%Y-%m-%d')
    
    # (0) Read configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    RAW_DIR = config['raw_dir']
    RAW_FILE_NAME = config['raw_file_name']
    RAW_FILE_PATH = os.path.join(RAW_DIR, RAW_FILE_NAME)
    
    ingest_config = config['ingest_data']
    # Loop starts here
    curr_endpoint = ingest_config['endpoints'][0]
    ingestion_fn = ingestion_fns[curr_endpoint] # ingestion function
    
    # (1) Check if raw exists, and
    # (2) Read in relevant data from NBA API
    if os.path.exists(RAW_FILE_PATH):
        print(f"'{RAW_FILE_PATH}' exists; last read on '{last_read}'")
        print(f" Will read in all potential new games")
        new_games = ingestion_fn(start_date=last_read)
    else:
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
    new_games['UNIQUE_ID'] = new_games.apply(lambda row: str(row['GAME_ID']) + '_' + str(row['IS_HOME']), axis=1)
    
    # (6) Handle integrating into existing reserve
    if os.path.exists(RAW_FILE_PATH):
        existing_games = pd.read_csv(RAW_FILE_PATH)
        len_existing = len(existing_games)
        print(f" * {len_existing} games found in existing reserve")
        games = pd.concat((existing_games, new_games), ignore_index=True)    
        games = games.drop_duplicates(subset=['UNIQUE_ID'])
        len_total = len(games)
        n_duplicates = (len_existing + len_new) - len_total
        print(f" * {n_duplicates} duplicates found, added {len_total - len_existing} new games to existing reserve")
    else:
        games = new_games
        len_total = len(games)
    
    # (7) Save file
    save_as_csv(
        games=games, 
        write_dir=RAW_DIR,
        csv_name=RAW_FILE_NAME,
    )
    print(f"{len_total} games in the reserve at '{RAW_FILE_PATH}'")