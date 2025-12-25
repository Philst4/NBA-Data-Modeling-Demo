import sys
import os
import argparse

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import yaml
import pandas as pd

# Internal imports
from src.data.io import (
    query_db,
    save_to_db
)
from src.data.processing import (
    get_temporal_spatial_features,
    get_rolling_avgs,
    add_rolling_avg_diffs,
    get_rolling_stats,
    add_rolling_diffs
)

def main(args):
    # Read configuration
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    CLEAN_DIR = config['clean_data_dir']    
    DB_NAME = config['db_name']
    DB_PATH = os.path.join(CLEAN_DIR, DB_NAME)
    METADATA_TABLE_NAME = config['metadata']['games']['table_name']
    MAIN_TABLE_NAME = config['main_table_name']

    game_metadata = query_db(DB_PATH, f"SELECT * from {METADATA_TABLE_NAME}")
    
    if not args.normalized:
        game_data = query_db(DB_PATH, f"SELECT * from {MAIN_TABLE_NAME}")
    else:
        game_data = query_db(DB_PATH, f"SELECT * from {MAIN_TABLE_NAME}_norm")
    
    
    # Get temporal-spatial features, save
    temporal_spatial = get_temporal_spatial_features(
        game_metadata
    )
    save_to_db(
        temporal_spatial.sort_values(by=['UNIQUE_ID']), 
        CLEAN_DIR,
        DB_NAME,
        "temporal_spatial"
    )
    
    # Get rolling temporal-spatial features
    temporal_spatial_rolling = get_rolling_avgs(
        temporal_spatial,
        game_metadata,
        windows=[args.window]
    )
    save_to_db(
        temporal_spatial_rolling.sort_values(by=['UNIQUE_ID']),
        CLEAN_DIR,
        DB_NAME,
        f"temporal_spatial_prev_{args.window}"
    )
    
    """
    # Get rolling averages
    game_data_rolling_avgs = get_rolling_avgs(
        game_data=game_data,
        game_metadata=game_metadata,
        windows=[args.window]
    )
    
    # Add rolling average differences
    game_data_rolling_avgs = add_rolling_avg_diffs(
        game_data_rolling_avgs
    )
    """
    
    game_data_rolling_stats = get_rolling_stats(
        game_data,
        game_metadata,
        windows=[args.window]
    )
    
    game_data_rolling_stats = add_rolling_diffs(
        game_data_rolling_stats
    )
    
    # Save rolling to database
    suffix = "_"
    if args.normalized:
        suffix += "norm_"
    suffix += "prev_" + str(args.window)
    
    save_to_db(
        game_data_rolling_stats.sort_values(by=['UNIQUE_ID']), 
        CLEAN_DIR,
        DB_NAME,
        MAIN_TABLE_NAME + suffix
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=os.environ.get("CONFIG_PATH", "configs/config.yaml"), help="Config path")
    parser.add_argument('--window', type=int, default=0, help="Window to make rolling averages over")
    parser.add_argument(
        "--normalized",
        action="store_true",
        default=True,
        help="Use normalized data"
    )
    parser.add_argument(
        "--not-normalized",
        dest="normalized",
        action="store_false",
        help="Disable normalization"
    )

    args = parser.parse_args()
    main(args)
    