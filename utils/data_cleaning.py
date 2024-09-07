import pandas as pd
import numpy as np
import sqlite3


def read_raw_from_csv(read_dir : str, read_name : str) -> pd.DataFrame:
    print(" * Reading in data from csv ...")
    read_path = '/'.join((read_dir, read_name))
    return pd.read_csv(read_path)

def drop_cols(games : pd.DataFrame, cols_to_drop) -> None:
    print(f" * Dropping columns {cols_to_drop} ...")
    games.drop(cols_to_drop, axis=1, inplace=True)
    return

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
    return

def add_cols(games : pd.DataFrame, cols_to_add : list[str], dependencies : list[list[str]], maps) -> None:
    for i in range(0, len(cols_to_add)):
        new_col = cols_to_add[i]
        cols_depending_on = dependencies[i]
        map = maps[i]
        games[new_col] = games[cols_depending_on].map(map)
    return

def make_team_id_df(
    games : pd.DataFrame, 
    season_id_col='SEASON_ID',
    team_id_col='NEW_TEAM_ID', 
    abbr_col='TEAM_ABBREVIATION',
    debug=False
):
    print(" * Making table containing team metadata ...")
    
    team_id_df = games.groupby([season_id_col, team_id_col])[abbr_col].unique().reset_index()
    team_id_df = team_id_df.explode(abbr_col).reset_index(drop=True)
    
    if debug:
        print(games.columns)
        input()
        print(team_id_df.columns)
        input()
        print(team_id_df.head())
        input()
    return team_id_df

def mirror(games : pd.DataFrame, leave_out_cols : list[str]=['MATCHUP', 'GAME_ID']) -> None:
    # NOTE requires that 'games' has a 'MATCHUP' and 'GAME_ID' column
    print(" * Mirroring data ...")
    
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

def get_summary_stats(games : pd.DataFrame, leave_out_cols, debug=False) -> None:
    print(" * Making table containing season means, stds by stat ... ")
    
    # If stat has '_for' and '_ag', only using '_for' for calculations
    cols_to_summarize = [col for col in list(games.columns) if '_ag' not in col]
    cols_to_summarize = [col for col in cols_to_summarize if not any(col2 in col for col2 in leave_out_cols)]
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
    return
    
def save_as_db(
    df : pd.DataFrame, 
    write_dir : str, 
    db_name : str, 
    table_name : str,
    index=False
) -> None:
    write_path = '/'.join((write_dir, db_name))
    conn = sqlite3.connect(write_path)
    df.to_sql(table_name, conn, if_exists='replace', index=index)
    conn.close()    
    print(f" * Data saved to '{write_path}' as '{table_name}'")
    return 
