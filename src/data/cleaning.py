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

# Used to fill 'PLUS_MINUS' N/A's
def fill_plus_minus(games : pd.DataFrame) -> None:
    print(f" * Filling 'PLUS_MINUS' ...")
    games.loc[:, 'PLUS_MINUS_for'] = games.loc[:, 'PTS_for'].astype(int) - games.loc[:, 'PTS_ag'].astype(int)
    games.loc[:, 'PLUS_MINUS_ag'] = -games.loc[:, 'PLUS_MINUS_for']
    return

def mirror(
    games : pd.DataFrame, 
    cols_to_mirror=None,
    cols_not_to_mirror=None,
) -> pd.DataFrame:
    """
    Adds opposing stats for game instances to each game instance.
    """
    print(" * Mirroring data ...")
    
    error_str1 = "Neither cols_to_mirror or cols_not_to_mirror provided; please provide one (empty list accepted)"
    assert(not (cols_to_mirror is None and cols_not_to_mirror is None)), error_str1
    error_str2 = "Only provide one of cols_to_mirror and cols_not_to_mirror"
    assert(not (cols_to_mirror is not None and cols_not_to_mirror is not None)), error_str2
    
    # Identify columns to be mirrored; initialize new column names
    if cols_not_to_mirror is not None:
        cols_to_mirror = [col for col in list(games.columns) if col not in cols_not_to_mirror]
    else:
        cols_not_to_mirror = [col for col in list(games.columns) if col not in cols_to_mirror]
    col_for_mapping = {col : col for col in cols_not_to_mirror} | {col : col + '_for' for col in cols_to_mirror}
    col_ag_mapping = {col : col + '_ag' for col in cols_to_mirror}

    # Sort by 'UNIQUE_ID'
    games.sort_values(by='UNIQUE_ID', inplace=True)

    # Flip 'UNIQUE_ID' on opposing games to join properly
    # If last digit is 0 make it 1, elif last digit is 1 make it 0
    games_ag = games[['UNIQUE_ID'] + cols_to_mirror].rename(columns=col_ag_mapping, inplace=False)
    f = lambda u_id : u_id + 1 if u_id % 10 == 0 else u_id - 1
    games_ag['UNIQUE_ID'] = games_ag.loc[:, 'UNIQUE_ID'].apply(f)
    games.rename(columns=col_for_mapping, inplace=True)
    return pd.merge(games, games_ag, on='UNIQUE_ID')

def get_summary_stats(
    games : pd.DataFrame, 
    leave_out_cols, 
    debug=False
) -> None:
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