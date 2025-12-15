import sys
import os

import pandas as pd
import sqlite3


def save_as_csv(
    df : pd.DataFrame, 
    write_dir : str,
    csv_name : str
) -> None:
    if not os.path.exists(write_dir):
        os.makedirs(write_dir, exist_ok=True)
    write_path = os.path.join(write_dir, csv_name)
    df.to_csv(write_path, index=False)
    print(f" * Data written to: {write_path}")
    return

def read_from_csv(read_path : str) -> pd.DataFrame:
    print(f" * Reading in '{read_path}' as pd.DataFrame ...")
    assert(os.path.exists(read_path)), f"{read_path} not found"
    return pd.read_csv(read_path)

def save_as_parquet(
    df : pd.DataFrame,
    write_dir : str,
    parquet_name : str
) -> None:
    if not os.path.exists(write_dir):
        os.makedirs(write_dir, exist_ok=True)
    write_path = os.path.join(write_dir, parquet_name)
    df.to_parquet(write_path, index=False, engine="pyarrow")
    print(f" * Data written to: {write_path}")
    
def read_from_parquet(read_path : str) -> pd.DataFrame:
    print(f" * Reading in '{read_path}' as pd.DataFrame ...")
    assert(os.path.exists(read_path)), f"{read_path} not found"
    return pd.read_parquet(read_path, engine="pyarrow")

def save_to_db(
    df : pd.DataFrame, 
    write_dir : str, 
    db_name : str, 
    table_name : str,
    index=False
) -> None:
    if not os.path.exists(write_dir):
        os.makedirs(write_dir, exist_ok=True)
    
    write_path = os.path.join(write_dir, db_name)
    conn = sqlite3.connect(write_path)
    df.to_sql(table_name, conn, if_exists='replace', index=index)
    conn.close()    
    print(f" * Table saved to '{write_path}' as '{table_name}' (run 'python scripts/check_db.py' for more info)")
    
def query_db(db_path : str, query : str) -> pd.DataFrame:
    assert(os.path.exists(db_path)), "Database not found at path"
    conn = sqlite3.connect(db_path)
    dataframe = pd.read_sql_query(query, conn)
    conn.close()
    return dataframe

def get_modeling_data(
    db_path,
    config,
    date=None,
    window=0,
    w_target_means_stds=True
    ):
    """
    Returns the modeling data, feature names, target name
    """
    
    GAME_METADATA_TABLE_NAME = config['metadata']['games']['table_name']
    GAME_DATA_TABLE_NAME =f"{config['main_table_name']}"
    GAME_DATA_NORM_TABLE_NAME = GAME_DATA_TABLE_NAME + '_norm'
    GAME_DATA_NORM_PREV_TABLE_NAME = f"{config['main_table_name']}_norm_prev_{window}"
    
    if not date:
        game_metadata = query_db(db_path, f"SELECT * FROM {GAME_METADATA_TABLE_NAME}")
        game_data_norm = query_db(db_path, f"SELECT * FROM {GAME_DATA_NORM_TABLE_NAME}")
        game_data_norm_prev = query_db(db_path, f"SELECT * FROM {GAME_DATA_NORM_PREV_TABLE_NAME}")
    else:
        
        # Find the game_metadata from the specified date
        game_metadata = query_db(db_path, f"SELECT * FROM {GAME_METADATA_TABLE_NAME} WHERE GAME_DATE = '{date}'")
        
        # Extract the proper UNIQUE_ID's from the date, use to query for relevant game_data + game_data_prev
        unique_ids = list(game_metadata['UNIQUE_ID'])
        
        # Query for data that has those UNIQUE_ID's
        game_data_norm = query_db(db_path, f"SELECT * FROM {GAME_DATA_NORM_TABLE_NAME} WHERE UNIQUE_ID in {tuple(unique_ids)}")
        game_data_norm_prev = query_db(db_path, f"SELECT * FROM {GAME_DATA_NORM_PREV_TABLE_NAME} WHERE UNIQUE_ID in {tuple(unique_ids)}")
    
    # Merge to get modeling data
    modeling_data = pd.merge(game_metadata, game_data_norm_prev, on='UNIQUE_ID')
    modeling_data = pd.merge(modeling_data, game_data_norm[['UNIQUE_ID', 'IS_HOME_for', 'IS_HOME_ag', 'PLUS_MINUS_for']], on='UNIQUE_ID')
    
    # Get features + target
    rolling_avg_feature_names = [col for col in list(game_data_norm_prev.columns) if col != 'UNIQUE_ID']
    features = ['IS_HOME_for', 'IS_HOME_ag'] + rolling_avg_feature_names
    target = 'PLUS_MINUS_for'
    
    target_means_stds = None
    
    if w_target_means_stds:
        means_stds = query_db(db_path, f"SELECT * FROM {GAME_DATA_TABLE_NAME}_means_stds")
        seasons = modeling_data['SEASON_ID'].unique()
        target_means_stds = means_stds.loc[
            means_stds['SEASON_ID'].isin(seasons), 
            ['SEASON_ID', target + '_mean', target + '_std']
        ]
    
    return modeling_data, features, target, target_means_stds