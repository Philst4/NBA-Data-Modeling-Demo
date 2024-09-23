import pandas as pd

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


def get_rolling(games : pd.DataFrame, window_size : int=0) -> None:
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
    
