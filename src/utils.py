import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import yaml
import pandas as pd
import sqlite3

def season_to_str(season : int) -> str:
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

def save_as_csv(
    games : pd.DataFrame, 
    write_dir : str,
    csv_name : str
) -> None:
    if not os.path.exists(write_dir):
        os.makedirs(write_dir, exist_ok=True)
    write_path = os.path.join(write_dir, csv_name)
    games.to_csv(write_path, index=False)
    print(f" * Data written to: {write_path}")
    return


def read_from_csv(read_path : str) -> pd.DataFrame:
    print(" * Reading in data from csv ...")
    assert(os.path.exists(read_path)), f"{read_path} not found"
    return pd.read_csv(read_path)


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


def summarize(games : pd.DataFrame, check_game_counts : bool=True) -> None:
    """Summarizes basic stats of given pd.DataFrame.

    Prints the columns in games, the shape of games, and how many NaN's are in each column.
    If there are no NaN's in a given column, omits it from print.

    Args:
        games : pd.DataFrame
    Returns:
        None
    """

    # Prints datatypes
    print(f" --- DATATYPES --- ")
    print(f"{games.dtypes}\n")

    # Print number of rows and columns
    n_rows, n_cols = games.shape
    print(f" --- SHAPE --- ")
    print(f"{n_rows} rows, {n_cols} columns \n")

    # Prints number of NaN's by column
    nan_counts = games.isna().sum()
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
    print(f" --- NaN's PER COLUMN --- ")
    if len(nan_counts) == 0:
        print(f"No NaN's found")
    else:
        print(f"{nan_counts[nan_counts > 0].sort_values(ascending=False)}")

    # Print rows     
    rows_per_game = games['GAME_ID'].value_counts().unique()
    teams_per_game = games.groupby('GAME_ID')['UNIQUE_ID'].nunique().unique()
    print(f"\nUNIQUE ROW COUNTS PER GAME ID: {rows_per_game}")
    print(f"UNIQUE ID COUNTS PER GAME ID: {teams_per_game}")


def check_db(db_path : str) -> None:
    assert(os.path.exists(db_path)), f"Database not found at '{db_path}'"
    
    # Connect to your SQLite database
    conn = sqlite3.connect(db_path)

    # Create a cursor object to execute SQL queries
    cursor = conn.cursor()

    # Query to get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

    # Fetch and print table names
    tables = cursor.fetchall()

    for table_name in tables:
        print(f"\nTABLE: {table_name[0]}")
        
        cursor.execute(f"SELECT COUNT(*) FROM {table_name[0]}")
        row_count = cursor.fetchone()[0]
        
        # Query to get column names and types
        cursor.execute(f"PRAGMA table_info({table_name[0]});")
        columns = cursor.fetchall()

        print(f" * {row_count} rows, {len(columns)} columns")
        
        # Display columns information
        for col in columns:
            print(f" - - {col[1]} ({col[2]})")
    
    # Print number of tables 
    tables_str = ', '.join([name for (name,) in tables])
    print(f"\n{len(tables)} tables in database:", tables_str)
    
    # Get size of database
    cursor.execute("PRAGMA page_count;")
    page_count = cursor.fetchone()[0]

    cursor.execute("PRAGMA page_size;")
    page_size = cursor.fetchone()[0]

    db_size = page_count * page_size  # in bytes
    print(f"Size of database: {db_size} bytes => ~{db_size/1e9:.3f} GB")


def query_db(db_path : str, query : str) -> pd.DataFrame:
    assert(os.path.exists(db_path)), "Database not found at path"
    conn = sqlite3.connect(db_path)
    dataframe = pd.read_sql_query(query, conn)
    conn.close()
    return dataframe


def normalize(games : pd.DataFrame, summary_stats : pd.DataFrame) -> pd.DataFrame:
    return
    normalized_games = games.copy()
    normalized_games[numerical_cols] = normalized_games[numerical_cols].astype(float)
    for col in numerical_cols:
        mean_col, std_col = summary_cols_map[col]
        for season in list(summary_stats['SEASON_ID']):
            season_values = cleaned_games.loc[cleaned_games['SEASON_ID'] == season, col]
            season_mean = summary_stats.loc[summary_stats['SEASON_ID'] == season, mean_col].values
            season_std = summary_stats.loc[summary_stats['SEASON_ID'] == season, std_col].values
            normalized_season_values = (season_values - season_mean) / season_std
            normalized_games.loc[normalized_games['SEASON_ID'] == season, col] = normalized_season_values

    normalized_games = normalized_games.dropna()


def get_rolling(games : pd.DataFrame, window_size : int=0) -> None:
    return
    # Step 1: Sort by 'SEASON_ID', 'TEAM_ID_for', and 'GAME_DATE', groupby 'SEASON_ID', 'NEW_TEAM_ID_for'
    games = games.sort_values(by=['SEASON_ID', 'NEW_TEAM_ID_for', 'GAME_DATE'])
    team_groups = games.groupby(['SEASON_ID', 'NEW_TEAM_ID_for'])
    
    # Step 2: Define numerical columns for which we want the rolling average
    # First define cols not to roll (all metadata + some other)
    cols_not_to_roll = list(game_metadata.columns)
    cols_not_to_roll += [
        'UNIQUE_ID',
        'NEW_TEAM_ID_for', 'NEW_TEAM_ID_ag'
    ]
    
