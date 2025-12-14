import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import yaml
import argparse

# Internal imports
from src.data.io import (
    query_db
)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=os.environ.get("CONFIG_PATH", "configs/config.yaml"), help="Config path")
    args = parser.parse_args()
    
    
    # Command line arguments (revisit)
    season_id = "22023"

    # Extract from config (revisit)
    # Read configuration
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    CLEAN_DIR = config['clean_data_dir']
    DB_NAME = config['db_name']
    DB_PATH = os.path.join(CLEAN_DIR, DB_NAME)
    
    # Query
    query = f"""SELECT TEAM_NAME, TEAM_ABBREVIATION 
    FROM team_metadata 
    WHERE SEASON_ID = {season_id}
    """
    
    team_abbreviations = query_db(
        db_path=DB_PATH,
        query=query
    )
    
    print(team_abbreviations.sort_values(by=['TEAM_ABBREVIATION'], ignore_index=True))