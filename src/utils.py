import sys
import os

# External imports
import warnings
import pandas as pd
import sqlite3

#### STRING PROCESSING UTILITIES ####
def season_int_to_str(season : int) -> str:
    """Converts reference of season start to reference of entire season.

    The resulting string can be used to specify queries for the NBA API.
    Example use: season_to_str(2023) gives '2023-24'

    Args: 
        season: an int meant to reference the start of an NBA season 
    
    Returns:
        A string referencing the entire season, meant to be used to 
        query from the NBA API. Example use: season_to_str(2023) gives '2023-24'
    """    
    return f"{season}-{str(season + 1)[-2:]}"

def season_str_to_int(season_str : str) -> int:
    return int(season_str[:4])

#### I/O UTILITIES ####
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