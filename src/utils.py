import pandas as pd

def summarize(games : pd.DataFrame) -> None:
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
