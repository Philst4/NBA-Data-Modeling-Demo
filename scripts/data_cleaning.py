import pandas as pd
import numpy as np
import sqlite3


def read_from_csv(read_path='../data/raw/raw.csv') -> pd.DataFrame:
    print(" * Reading in data from csv...")
    return pd.read_csv(read_path)


def drop_pcts(games : pd.DataFrame) -> None:
    print(" * Dropping 'PCT' columns, ...")
    pct_cols = ['FT_PCT', 'FG3_PCT']
    games.drop(pct_cols, axis=1, inplace=True)
    return

def mirror(games : pd.DataFrame, leave_out_cols : list[str]=['MATCHUP', 'GAME_ID']) -> None:
    print(" * Mirroring data...")
    
    # Add necessary leave_out_cols for function to operate
    if 'MATCHUP' not in leave_out_cols:
        leave_out_cols.append('MATCHUP')
    elif 'GAME_ID' not in leave_out_cols:
        leave_out_cols.append('GAME_ID')
    
    # Identify columns to be mirrored; initialize new column names
    cols_to_mirror = [col for col in list(games.columns) if col not in leave_out_cols]
    col_for_mapping = {col: col + '_for' for col in cols_to_mirror}

    # Rename columns existing columns to cols_for
    games.rename(columns=col_for_mapping, inplace=True)

    # Initialize new columns for opposing stats
    cols_ag = [col + '_ag' for col in cols_to_mirror]
    games[cols_ag] = None

    # Fill in new columns with data from opposing team's game instance
    # Identify matching game instances using home-away conditions
    cols_for = list(col_for_mapping.values())

    away_condn = games['MATCHUP'].apply(lambda x : '@' in x)
    home_condn = games['MATCHUP'].apply(lambda x : 'vs.' in x)
    
    # Sort, then match
    games.sort_values(by='GAME_ID', inplace=True)
    games.loc[away_condn, cols_ag] = games.loc[home_condn, cols_for].values
    games.loc[home_condn, cols_ag] = games.loc[away_condn, cols_for].values
    return


def fill_plus_minus(games : pd.DataFrame) -> None:
    print(f" * Filling 'PLUS_MINUS'...")
    games.loc[:, 'PLUS_MINUS'] = games.loc[:, 'PTS_for'].astype(int) - games.loc[:, 'PTS_ag'].astype(int)
    return
    
    
def impute_NaNs(games : pd.DataFrame) -> None:
    num_NaNs = games.isna().sum(axis=0).sum()
    num_NaN_rows = games.isna().any(axis=1).sum()
    print(f" * {num_NaNs} remaining NaN's in {num_NaN_rows} rows; removing NaN's...")
    
    games.dropna(inplace=True)
    num_NaNs = games.isna().sum(axis=0).sum()
    num_NaN_rows = games.isna().any(axis=1).sum()
    print(f" * {num_NaNs} remaining NaN's in {num_NaN_rows} rows left!")
    return
    

def save_as_db(games : pd.DataFrame, write_path='../data/cleaned/my_database.db') -> None:
    print(" * Saving cleaned data as database... ")
    conn = sqlite3.connect(write_path)
    games.to_sql('my_table', conn, if_exists='replace', index=False)
    conn.close()    
    print(" * Data saved")
    return 


if __name__ == "__main__":
    print(" --- RUNNING DATA CLEANING SCRIPT --- ")
    
    games = read_from_csv()
    drop_pcts(games)
    
    leave_out_cols = ['SEASON_ID', 'GAME_DATE', 'GAME_ID', 'MATCHUP', 'PLUS_MINUS']
    mirror(games, leave_out_cols)
    
    fill_plus_minus(games)
    impute_NaNs(games)
    save_as_db(games)
    print('Success')
    