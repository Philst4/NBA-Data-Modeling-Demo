import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)


# Local imports
from config import RAW_DATA_DIR
from utils.data_ingestion import (
    ingest_from_nba_api,
    write_raw_to_csv
)


if __name__ == "__main__":
    
    # (1) Read data from NBA API
    games = ingest_from_nba_api(first_season=1983)
    
    # (2) Save data
    if games is None:
        print(' * No games ingested')
    else:
        write_raw_to_csv(
            games=games, 
            write_dir=RAW_DATA_DIR, 
            write_name='raw.csv'
        )