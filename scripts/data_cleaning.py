import pandas as pd
import numpy as np
import sqlite3


def read_from_csv(read_path='../data/raw/raw.csv') -> pd.DataFrame:
    print(" * Reading in data from csv...")
    return pd.read_csv(read_path)


def drop_pcts(games : pd.DataFrame) -> None:
    print(" * Dropping 'PCT' columns, ...")
    pct_cols = ['FG_PCT', 'FT_PCT', 'FG3_PCT']
    games.drop(pct_cols, axis=1, inplace=True)
    return


def convert_types(games : pd.DataFrame) -> None:
    # NOTE: 
    #  * Handle 'O' to in next stage; write->read converts back to 'O'
    #  * Handle datetime elsewhere; write->read converts back to 'O'
    print(" * Converting types of existing data... ")
    
    # Convert 'object' types to strings
    obj_cols = games.select_dtypes(include='O').columns
    games[obj_cols] = games[obj_cols].astype('string')
    
    # Convert 'number' types to float
    num_cols = games.select_dtypes(include='number').columns
    games[num_cols] = games[num_cols].astype('float64')
    return


def make_wl_map():
    return lambda x : {'L' : 0, 'W' : 1}[x]    

# Used for changing the 'TEAM_ID' to be more reasonable
def make_id_map(games : pd.DataFrame, old_id_col : str):
    unique_ids = list(games[old_id_col].unique())
    return lambda x : dict(zip(unique_ids, range(len(unique_ids))))[x]

def make_is_home_map():
    return lambda x : int('vs.' in x)

def add_cols(games : pd.DataFrame, new_cols : list[str], dependencies : list[list[str]], maps) -> None:
    for i in range(0, len(new_cols)):
        new_col = new_cols[i]
        cols_depending_on = dependencies[i]
        map = maps[i]
        games[new_col] = games[cols_depending_on].map(map)
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
    
    
def deal_w_NaNs(games : pd.DataFrame) -> None:
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
    
    # Read in games
    games = read_from_csv()
    
    # Initial cleaning
    drop_pcts(games)
    convert_types(games)
    
    # Define new columns to add
    new_cols = ['TARGET', 'NEW_TEAM_ID', 'IS_HOME']
    dependencies = [['WL'], ['TEAM_ID'], ['MATCHUP']]
    maps = [make_wl_map(), make_id_map(games, 'TEAM_ID'), make_is_home_map()]
    add_cols(games, new_cols, dependencies, maps)
    
    # Mirror opposing stats
    leave_out_cols = ['SEASON_ID', 'GAME_DATE', 'GAME_ID', 'MATCHUP', 'PLUS_MINUS', 'WL', 'TARGET']
    mirror(games, leave_out_cols)
    
    # Deal with NaN's
    fill_plus_minus(games)
    deal_w_NaNs(games)
    
    # Save the cleaned data
    save_as_db(games)
    print('Success')