import os

import pandas as pd
import numpy as np
import sqlite3


def read_from_csv(read_path : str) -> pd.DataFrame:
    print(" * Reading in data from csv ...")
    assert(os.path.exists(read_path)), f"{read_path} not found"
    
    return pd.read_csv(read_path)


def drop_cols(games : pd.DataFrame, cols_to_drop, verbose=False) -> None:
    if verbose:
        print(f" * Dropping columns {cols_to_drop} ...")
    games.drop(cols_to_drop, axis=1, inplace=True)

def convert_types(games : pd.DataFrame) -> None:
    # NOTE: 
    #  * Handle 'O' to in next stage; write->read converts back to 'O'
    #  * Handle datetime elsewhere; write->read converts back to 'O'
    print(" * Converting types of remaining columns ... ")
    
    # Convert 'object' types to strings
    obj_cols = games.select_dtypes(include='O').columns
    games[obj_cols] = games[obj_cols].astype('string')
    
    # Convert 'number' types to float
    num_cols = games.select_dtypes(include='number').columns
    games[num_cols] = games[num_cols].astype('float64')

# Used for changing the 'TEAM_ID' to be more reasonable
def make_id_map(games : pd.DataFrame, old_id_col : str):
    unique_ids = sorted(list(games[old_id_col].unique()))
    return lambda x : dict(zip(unique_ids, range(len(unique_ids))))[x]


def add_cols(games : pd.DataFrame, cols_to_add : list[str], dependencies : list[list[str]], maps) -> None:
    for i in range(0, len(cols_to_add)):
        new_col = cols_to_add[i]
        cols_depending_on = dependencies[i]
        map = maps[i]
        games[new_col] = games[cols_depending_on].map(map)


def mirror(
    games : pd.DataFrame, 
    cols_to_mirror=None,
    cols_not_to_mirror=None,
) -> None:
    print(" * Mirroring data ...")
    
    error_str1 = "Neither cols_to_mirror or cols_not_to_mirror provided; please provide one (empty list accepted)"
    assert(not (cols_to_mirror is None and cols_not_to_mirror is None)), error_str1
    error_str2 = "Only provide one of cols_to_mirror and cols_not_to_mirror"
    assert(not (cols_to_mirror is not None and cols_not_to_mirror is not None)), error_str2


    
    # First, save away_condn, home_condn
    games.sort_values(by='GAME_ID', inplace=True)
    away_condn = games['IS_HOME'] == 0
    home_condn = games['IS_HOME'] == 1
    
    # Identify columns to be mirrored; initialize new column names
    if cols_not_to_mirror is not None:
        cols_to_mirror = [col for col in list(games.columns) if col not in cols_not_to_mirror]
    col_for_mapping = {col: col + '_for' for col in cols_to_mirror}

    # Rename columns existing columns to cols_for
    games.rename(columns=col_for_mapping, inplace=True)

    # Initialize new columns for opposing stats
    cols_ag = [col + '_ag' for col in cols_to_mirror]
    games[cols_ag] = None

    # Fill in new columns with data from opposing team's game instance
    # Identify matching game instances using home-away conditions
    cols_for = list(col_for_mapping.values())
    games.loc[away_condn, cols_ag] = games.loc[home_condn, cols_for].values
    games.loc[home_condn, cols_ag] = games.loc[away_condn, cols_for].values


def get_summary_stats(games : pd.DataFrame, leave_out_cols, debug=False) -> None:
    print(" * Making summary table with season means, stds of each stat ... ")
    
    # If stat has '_for' and '_ag', only using '_for' for calculations
    cols_to_summarize = [col for col in list(games.columns) if col not in leave_out_cols]
    if 'SEASON_ID' not in cols_to_summarize:
        cols_to_summarize.append('SEASON_ID')
    if debug:
        print(cols_to_summarize)
        input()
    
    # Get summary stats
    summary_stats = games[cols_to_summarize].groupby('SEASON_ID').agg(['mean', 'std'])
    summary_stats.columns = ['_'.join(col) for col in summary_stats.columns]
    # Remove '_for' from relevant column names
    summary_stats.columns = [col.replace('_for', '') for col in summary_stats.columns]
    if debug:
        print(summary_stats.columns)
        input()
    return summary_stats


def deal_w_NaNs(games : pd.DataFrame) -> None:
    num_NaNs = games.isna().sum(axis=0).sum()
    num_NaN_rows = games.isna().any(axis=1).sum()
    print(f" * {num_NaNs} remaining NaN's in {num_NaN_rows} rows; removing NaN's...")
    
    games.dropna(inplace=True)
    num_NaNs = games.isna().sum(axis=0).sum()
    num_NaN_rows = games.isna().any(axis=1).sum()
    print(f" * {num_NaNs} remaining NaN's in {num_NaN_rows} rows left!")
  
    
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
