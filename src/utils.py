import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import yaml
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

#### STRING PROCESSING UTILITIES ####

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

def get_summary_from_main(main_col):
    stripped_col = main_col.replace('_for', '')
    stripped_col = stripped_col.replace('_ag', '')
    mean_col = stripped_col + '_mean'
    std_col = stripped_col + '_std'
    return mean_col, std_col

def get_main_to_summary_map(main_cols):
    map = {}
    for col in main_cols:
        mean_col, std_col = get_summary_from_main(col)
        map[col] = {
            'mean' : mean_col,
            'std' : std_col
        }
    return map

def rename_for_rolled_opp(col):
    if '_ag' in col:
        return col.replace('_ag', '_for_opp')
    elif '_for' in col:
        return col.replace('_for', '_ag_opp')
    return col + '_opp'


#### I/O UTILITIES ####

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
    

def query_db(db_path : str, query : str) -> pd.DataFrame:
    assert(os.path.exists(db_path)), "Database not found at path"
    conn = sqlite3.connect(db_path)
    dataframe = pd.read_sql_query(query, conn)
    conn.close()
    return dataframe

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


#### COMMON DF PROCESSING ####
def get_normalized_by_season(
        game_data : pd.DataFrame,
        game_metadata : pd.DataFrame
    ) -> pd.DataFrame:
    
    game_data = game_data.copy()
    game_metadata = game_metadata[['UNIQUE_ID', 'SEASON_ID']].copy()

    game_data = pd.merge(game_metadata, game_data, on='UNIQUE_ID')

    # List of columns to normalize (excluding identifiers)
    cols_to_normalize = [col for col in game_data.columns if col not in ('UNIQUE_ID', 'SEASON_ID')]

# Normalize each column by SEASON_ID
    game_data[cols_to_normalize] = game_data.groupby('SEASON_ID')[cols_to_normalize].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    return game_data.drop('SEASON_ID', axis=1)  # Return normalized data without the SEASON_ID column

def get_rolling_avgs(
        game_data : pd.DataFrame,
        game_metadata : pd.DataFrame,
        game_data_cols=None,
        window : int=1,
        set_na_to_0 : bool=True
    ) -> pd.DataFrame:
    
    if game_data_cols is not None:
        game_data = game_data[game_data_cols].copy()
    else:
        game_data = game_data.copy()
        
    metadata_cols = [
        'UNIQUE_ID', 'SEASON_ID', 
        'NEW_TEAM_ID_for', 'NEW_TEAM_ID_ag',
        'GAME_DATE'
    ]
    game_metadata = game_metadata[metadata_cols].copy()
    
    if window <= 0:
        window = 100
    
    # Get window str
    if window > 82:
        window_str = "0"
    else:
        window_str = str(window)

    # Define cols to roll (all of game_data except 'UNIQUE_ID')
    cols_to_roll = [col for col in list(game_data.columns) if col != 'UNIQUE_ID']

    # Merge metadata with game_data
    main_data = pd.merge(game_metadata, game_data, on='UNIQUE_ID')
    
    # Define rolling function
    roll = lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    
    #### GET ROLLING AVERAGES FOR '_for' TEAM ####
    # Sort and groupby
    main_data = main_data.sort_values(by=['SEASON_ID', 'NEW_TEAM_ID_for', 'GAME_DATE'])
    team_groups = main_data.groupby(['SEASON_ID', 'NEW_TEAM_ID_for'])
    
    # Apply roling and calculate mean, exclude current row
    rolling_avgs = team_groups[cols_to_roll].apply(roll)
    rolling_avgs = rolling_avgs.set_index(main_data.index).add_suffix('_prev_' + window_str)

    # Add 'UNIQUE_ID' (already sorted to match rolling_avgs)
    rolling_avgs['UNIQUE_ID'] = main_data['UNIQUE_ID']

    #### GET ROLLING AVERAGES FOR '_ag' TEAM ####
    # Sort and groupby
    main_data = main_data.sort_values(by=['SEASON_ID', 'NEW_TEAM_ID_ag', 'GAME_DATE'])
    team_groups = main_data.groupby(['SEASON_ID', 'NEW_TEAM_ID_ag'])
    
    # Apply roling and calculate mean, exclude current row
    rolling_avgs_opp = team_groups[cols_to_roll].apply(roll)
    rolling_avgs_opp = rolling_avgs_opp.set_index(main_data.index).add_suffix('_prev_' + window_str)

    # Name opposing properly
    rolling_avgs_opp = rolling_avgs_opp.rename(columns=rename_for_rolled_opp)

    # Add 'UNIQUE_ID' (already sorted to match rolling_avgs_opp)
    rolling_avgs_opp['UNIQUE_ID'] = main_data['UNIQUE_ID']
    
    # Merge rolling_avgs and rolling_avgs_opp
    all_rolling_avgs = pd.merge(rolling_avgs, rolling_avgs_opp, on='UNIQUE_ID')
    
    # First games of season for each team will have 'NA' rolling averages
    # Set NA's to 0 if specified
    if set_na_to_0:
        # Note: sets ALL NA's to 0, so will mask uncleaned data
        all_rolling_avgs[all_rolling_avgs.isna()] = 0
    return all_rolling_avgs


#### COMMON PLOT/VISUAL GENERATING ####

def generate_kde_plots(df, columns, suffix='_for'):
    """
    Generates KDE plots for the specified columns from the dataframe.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    columns (list of str): List of column names to generate plots for.
    home_column (str): The column name indicating home/away games.
    suffix (str): Suffix for column names indicating 'for' values (e.g., 'PTS_for').
    nrows (int): Number of rows for the subplot grid.
    ncols (int): Number of columns for the subplot grid.

    Returns:
    None
    """
    
    # Set up the grid of KDE plots
    fig, axes = plt.subplots(nrows=len(columns), ncols=2, figsize=(12, 4 * len(columns)))
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy iteration

    # Create a KDE plot for each column
    for i, column in enumerate(columns):
        # Define the axes for the home/away and combined plots
        ax1, ax2 = axes[2 * i], axes[2 * i + 1]

        # Get home and away values
        home_values = df.loc[df['UNIQUE_ID'].str.contains('_1', na=False), column + suffix]
        away_values = df.loc[df['UNIQUE_ID'].str.contains('_0', na=False), column + suffix]

        # Plot the KDE for home/away
        sns.kdeplot(home_values, ax=ax1, fill=True, color='blue', alpha=0.1, label='home')
        sns.kdeplot(away_values, ax=ax1, fill=True, color='red', alpha=0.1, label='away')
        
        ax1.legend()
        ax1.set_title(f'KDE Plot of {column} (home, away)')
        ax1.set_xlabel(f'{column}')
        ax1.set_ylabel('Density')
        ax1.set_ylim(0, 0.08)
        ax1.grid()

        # Plot the KDE for combined data
        values = df.loc[:, column + suffix]
        sns.kdeplot(values, ax=ax2, fill=True, color='orange', alpha=0.2, label='home and away')
        
        ax2.legend()
        ax2.set_title(f'KDE Plot of {column} (all)')
        ax2.set_xlabel(f'{column}')
        ax2.set_ylabel('Density')
        ax2.set_ylim(0, 0.08)
        ax2.grid()

    # Adjust layout
    plt.tight_layout()
    plt.show()


def generate_corr_matrices(games : pd.DataFrame):
    cols = [col for col in list(games.columns) if col != 'UNIQUE_ID']
    for_cols = [col for col in cols if '_for' in col]
    ag_cols = [col for col in cols if '_ag' in col]
    correlation_matrix = games[cols].corr()
    
    # Create a figure with 2 rows and 1 column of subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))  # Adjust height with figsize

    # First heatmap: correlation_matrix.loc[for_cols, for_cols]
    sns.heatmap(
        correlation_matrix.loc[for_cols, for_cols],
        annot=True,
        cmap='coolwarm',
        center=0,  # Center the colormap at zero
        vmin=-0.5,   # Set the minimum correlation value
        vmax=0.5,    # Set the maximum correlation value
        fmt=".1f",
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8},
        annot_kws={"size": 8},
        ax=axes[0]  # Plot on the first axis
    )
    axes[0].set_title("Heatmap of stats for with other stats for")

    # Second heatmap: correlation_matrix.loc[for_cols, ag_cols]
    sns.heatmap(
        correlation_matrix.loc[for_cols, ag_cols],
        annot=True,
        cmap='coolwarm',
        center=0,  # Center the colormap at zero
        vmin=-0.5,   # Set the minimum correlation value
        vmax=0.5,    # Set the maximum correlation value
        fmt=".1f",
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8},
        annot_kws={"size": 8},  # Adjust the size of annotations
        ax=axes[1]  # Plot on the second axis
    )
    axes[1].set_title("Heatmap of stats for with stats ag")

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()


def generate_corr_vs_window():
    pass