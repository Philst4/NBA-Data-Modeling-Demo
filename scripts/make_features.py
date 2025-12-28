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
    save_to_db,
    read_from_parquet,
    save_as_parquet,
)
from src.data.processing import (
    get_temporal_spatial_features,
    get_rolling_avgs,
    get_rolling_stats,
    add_rolling_diffs
)

def main(args):
    # Read configuration
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    CLEAN_DIR = config['clean_data_dir']    
    METADATA_TABLE_NAME = config['metadata']['games']['table_name']
    MAIN_TABLE_NAME = config['main_table_name']

    game_metadata = read_from_parquet(
        os.path.join(CLEAN_DIR, METADATA_TABLE_NAME + '.parquet')
    ) 
    game_data = read_from_parquet(
        os.path.join(CLEAN_DIR, MAIN_TABLE_NAME + '.parquet')
    )
    
    # New table names
    ts_features_table_name = f"features_ts.parquet"
    
    # Get temporal-spatial features, save
    temporal_spatial = get_temporal_spatial_features(
        game_metadata,
        scale_0_1=args.ts_normalized
    )
    
    save_as_parquet(
        temporal_spatial.sort_values(by=['UNIQUE_ID']),
        CLEAN_DIR,
        ts_features_table_name,
        w_reduced_precision=True
    )
    
    for window in args.windows:
        rolling_features_table_name = f"features_prev_{window}.parquet"
        
        # Get base rolling stats
        game_data_rolling_stats = get_rolling_stats(
            game_data,
            game_metadata,
            windows=[window]
        )
        
        # Add diff rolling stats
        game_data_rolling_stats = add_rolling_diffs(
            game_data_rolling_stats
        )
        
        # Add temporal spatial rolling
        temporal_spatial_rolling = get_rolling_avgs(
            temporal_spatial,
            game_metadata,
            windows=[window]
        )
        
        game_data_rolling_stats = pd.merge(
            game_data_rolling_stats,
            temporal_spatial_rolling,
            on='UNIQUE_ID',
            how='left'
        )
        
        # Save rolling to database
        save_as_parquet(
            game_data_rolling_stats.sort_values(by=['UNIQUE_ID']), 
            CLEAN_DIR,
            rolling_features_table_name,
            w_reduced_precision=True
        )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=os.environ.get("CONFIG_PATH", "configs/config.yaml"), help="Config path")
    parser.add_argument('--windows', type=int, nargs='+', default=[5, 20, 0], help="Window to make rolling averages over")
    parser.add_argument(
        "--ts_normalized",
        action="store_true",
        default=True,
        help="Use normalized data"
    )
    parser.add_argument(
        "--not-ts_normalized",
        dest="ts_normalized",
        action="store_false",
        help="Disable normalization"
    )

    args = parser.parse_args()
    main(args)
    