import sys
import os

import pandas as pd
import sqlite3

# Internal imports
from src.data.processing import (
    get_normalized_by_season,
)


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
    date=None,
    window=0
    ):
    """
    Returns the modeling data, feature names, target name
    """
    
    if not date:
        game_metadata = query_db(db_path, "SELECT * from game_metadata")
        game_data = query_db(db_path, f"SELECT * from game_data")
        game_data_prev = query_db(db_path, f"SELECT * from game_data_norm_prev_{window}")
    else:
        # Find the season of specified date
        game_metadata = query_db(db_path, f"SELECT * from game_metadata WHERE GAME_DATE is {date}")
        game_data = query_db(db_path, f"SELECT * from game_data WHERE game_date is {date}")
        game_data_prev = query_db(db_path, f"SELECT * from game_data_norm_prev_{window}")
    
    # Merge to get modeling data
    modeling_data = pd.merge(game_metadata, game_data_prev, on='UNIQUE_ID')
    modeling_data = pd.merge(modeling_data, game_data[['UNIQUE_ID', 'IS_HOME_for', 'IS_HOME_ag']], on='UNIQUE_ID')
    
    # Add the normalized target to modeling data
    normalized_target = get_normalized_by_season(
        game_data[['UNIQUE_ID', 'PLUS_MINUS_for']],
        game_metadata
    )
    modeling_data = pd.merge(modeling_data, normalized_target, on='UNIQUE_ID')
    
    # Get features + target
    rolling_avg_feature_names = [col for col in list(game_data_prev.columns) if col != 'UNIQUE_ID']
    features = ['IS_HOME_for', 'IS_HOME_ag'] + rolling_avg_feature_names
    target = 'PLUS_MINUS_for'
    
    return modeling_data, features, target