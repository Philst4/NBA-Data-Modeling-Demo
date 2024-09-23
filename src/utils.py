import os

import pandas as pd
import sqlite3

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


def query_db(db_path : str, query : str) -> pd.DataFrame:
    assert(os.path.exists(db_path)), "Database not found at path"
    conn = sqlite3.connect(db_path)
    dataframe = pd.read_sql_query(query, conn)
    conn.close()
    return dataframe

game_metadata = query_db("SELECT * FROM game_metadata")
game_data = query_db("SELECT * FROM game_data")


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
    
