import os
import sqlite3
import pandas as pd

#### COMMON INFO GATHERING/CHECKS ####

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


def check_df(games : pd.DataFrame, check_game_counts : bool=True) -> None:
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
    if 'GAME_ID' in games.columns:
        rows_per_game = games['GAME_ID'].value_counts().unique()
        teams_per_game = games.groupby('GAME_ID')['UNIQUE_ID'].nunique().unique()
        print(f"\nUNIQUE ROW COUNTS PER GAME ID: {rows_per_game}")
        print(f"UNIQUE ID COUNTS PER GAME ID: {teams_per_game}")
