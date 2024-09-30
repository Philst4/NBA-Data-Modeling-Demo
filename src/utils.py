import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import yaml
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader

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
        windows : list=[1],
        need_opp : bool=True,
        set_na_to_0 : bool=True
    ) -> pd.DataFrame:
    
    if game_data_cols is not None:
        game_data = game_data[['UNIQUE_ID'] + game_data_cols].copy()
    else:
        game_data = game_data.copy()
        
    metadata_cols = [
        'UNIQUE_ID', 'SEASON_ID', 
        'NEW_TEAM_ID_for', 'NEW_TEAM_ID_ag',
        'GAME_DATE'
    ]
    game_metadata = game_metadata[metadata_cols].copy()

    # Define cols to roll (all of game_data except 'UNIQUE_ID')
    cols_to_roll = [col for col in list(game_data.columns) if col != 'UNIQUE_ID']

    # Merge metadata with game_data
    game_data = pd.merge(game_metadata, game_data, on='UNIQUE_ID')

    all_rolling_avgs = []

    for window in windows:
        if window <= 0:
            window = 100

        # Get window str
        if window > 82:
            window_str = "0"
        else:
            window_str = str(window)

        # Define rolling function
        roll = lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()

        #### GET ROLLING AVERAGES FOR '_for' TEAM ####
        # Sort and groupby
        game_data = game_data.sort_values(by=['SEASON_ID', 'NEW_TEAM_ID_for', 'GAME_DATE'])
        team_groups = game_data.groupby(['SEASON_ID', 'NEW_TEAM_ID_for'])

        # Apply rolling and calculate mean, exclude current row
        rolling_avgs = team_groups[cols_to_roll].apply(roll)
        rolling_avgs = rolling_avgs.set_index(game_data.index).add_suffix('_prev_' + window_str)

        # Add 'UNIQUE_ID' (already sorted to match rolling_avgs)
        rolling_avgs['UNIQUE_ID'] = game_data['UNIQUE_ID']

        #### GET ROLLING AVERAGES FOR '_ag' TEAM ####
        if need_opp:
            # Sort and groupby
            game_data = game_data.sort_values(by=['SEASON_ID', 'NEW_TEAM_ID_ag', 'GAME_DATE'])
            team_groups = game_data.groupby(['SEASON_ID', 'NEW_TEAM_ID_ag'])

            # Apply rolling and calculate mean, exclude current row
            rolling_avgs_opp = team_groups[cols_to_roll].apply(roll)
            rolling_avgs_opp = rolling_avgs_opp.set_index(game_data.index).add_suffix('_prev_' + window_str)

            # Name opposing properly
            rolling_avgs_opp = rolling_avgs_opp.rename(columns=rename_for_rolled_opp)

            # Add 'UNIQUE_ID' (already sorted to match rolling_avgs_opp)
            rolling_avgs_opp['UNIQUE_ID'] = game_data['UNIQUE_ID']

            # Merge rolling_avgs and rolling_avgs_opp
            rolling_avg_combined = pd.merge(rolling_avgs, rolling_avgs_opp, on='UNIQUE_ID')
        else:
            rolling_avg_combined = rolling_avgs

        all_rolling_avgs.append(rolling_avg_combined)

    # Concatenate all rolling averages for different windows
    final_rolling_avgs = pd.concat(all_rolling_avgs, axis=1)

    # Drop duplicate 'UNIQUE_ID' columns created in merging
    final_rolling_avgs = final_rolling_avgs.loc[:, ~final_rolling_avgs.columns.duplicated()]

    if set_na_to_0:
        # Note: sets ALL NA's to 0, so will mask uncleaned data
        final_rolling_avgs[final_rolling_avgs.isna()] = 0

    return final_rolling_avgs


#### COMMON PLOT/VISUAL GENERATING ####

def get_summary_stats(values):
    """
    Helper function to calculate statistics for a given set of values.

    Parameters:
    values (pd.Series or np.array): The data to calculate statistics for.

    Returns:
    dict: A dictionary containing the mean, median, standard deviation, min, and max of the values.
    """
    return {
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values)
    }

def generate_kde_plots(df, columns, suffix='_for', include_stats=False):
    """
    Generates KDE plots for the specified columns from the dataframe, 
    with optional statistics (mean, median, std, min, max) for each plot.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    columns (list of str): List of column names to generate plots for.
    suffix (str): Suffix for column names indicating 'for' values (e.g., 'PTS_for').
    include_stats (bool): If True, calculate and display stats on the plots.

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

        # If include_stats is True, calculate and display stats on the plots
        if include_stats:
            home_stats = get_summary_stats(home_values)
            away_stats = get_summary_stats(away_values)
            combined_stats = get_summary_stats(values)

            # Annotate stats for home and away on the first plot
            home_text = '\n'.join([f'home {key}: {val:.2f}' for key, val in home_stats.items()])
            away_text = '\n'.join([f'away {key}: {val:.2f}' for key, val in away_stats.items()])
            ax1.text(0.975, 0.8, home_text, transform=ax1.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right')
            ax1.text(0.025, 0.8, away_text, transform=ax1.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left')

            # Annotate stats for combined data on the second plot
            combined_text = '\n'.join([f'all {key}: {val:.2f}' for key, val in combined_stats.items()])
            ax2.text(0.975, 0.8, combined_text, transform=ax2.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def generate_hist_plot(
        game_data : pd.DataFrame, 
        stat : str, 
        suffix : str='_for', 
        bin_width : int=1,
        color='orange',
        include_stats=False
    ):
    # Create bins based on the bin width
    col = stat + suffix
    
    values = game_data[col]

    bins = range(int(values.min()), int(values.max()) + bin_width, bin_width)

    plt.figure(figsize=(12, 6))
    sns.histplot(values, bins=bins, kde=False, color=color, edgecolor='black', alpha=0.3)
    plt.xlabel(f"{stat}")
    plt.ylabel('Frequency')
    plt.title(f"Histogram of {stat}")
    plt.grid(axis='y')

    if include_stats:
        stats = get_summary_stats(values)

        # Annotate stats for home and away on the first plot
        text = '\n'.join([f'{key}: {val:.2f}' for key, val in stats.items()])
        plt.text(0.95, 0.95, text, fontsize=10, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right')
            

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


# Function to calculate correlation across window sizes for multiple stats
def calculate_corrs_for_windows(game_data, game_metadata, stats_to_analyze, max_exp=6):
    results = {}
    window_sizes = [2 ** i for i in range(max_exp + 1)] + [82]
    
    # Get rolling averages for all stats and all window sizes
    rolling_avgs = get_rolling_avgs(game_data, game_metadata, game_data_cols=stats_to_analyze, windows=window_sizes, need_opp=False)

    for stat in stats_to_analyze:
        correlations = []
        for window in window_sizes:
            # Use the rolling averages corresponding to the current window size
            rolling_stat_col = f"{stat}_prev_{window}"
            
            # Merge the rolling averages with the original game data
            temp = pd.merge(game_data, rolling_avgs[['UNIQUE_ID', rolling_stat_col]], on='UNIQUE_ID')
            
            # Calculate correlation between the rolling average and PLUS_MINUS_for
            correlation = temp[rolling_stat_col].corr(temp['PLUS_MINUS_for'])
            correlations.append(correlation)
        
        results[f'{stat}_prev_k'] = correlations
    
    return window_sizes, results

# Plot the results for multiple stats
def plot_corrs_for_windows(window_sizes, results):
    plt.figure(figsize=(10, 6))
    
    for stat, correlations in results.items():
        plt.plot(window_sizes, correlations, marker='o', label=f'{stat}')
    
    plt.xscale('log', base=2)
    plt.xlabel('Window Size k')
    plt.ylabel('Correlation with PLUS_MINUS_for')
    plt.title('Correlation of various stats with PLUS_MINUS_for over different window sizes')
    plt.grid(True)
    plt.legend(title="Stats")
    plt.show()
    
    
def visualize_regression_performance(targets : np.ndarray, preds : np.ndarray):

    mae_per_instance = (targets - preds)
    mse_per_instance = mae_per_instance ** 2

    # Plot histogram of MSE
    # Create a 3x2 grid of plots
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    fig.suptitle('Visualization of Regression Performance', fontsize=16)

    # Next
    axs[0,0].hist(preds, bins=40, color='yellow', edgecolor='black', alpha=0.7)
    axs[0,0].set_title('Histogram of preds')
    axs[0,0].set_xlabel('preds')
    axs[0,0].set_ylabel('Frequency')
    axs[0,0].grid(axis='y', alpha=0.75)


    axs[0,1].hist(targets, bins=40, color='lime', edgecolor='black', alpha=0.7)
    axs[0,1].set_title('Histogram of targets')
    axs[0,1].set_xlabel('targets')
    axs[0,1].set_ylabel('Frequency')
    axs[0,1].grid(axis='y', alpha=0.75)


    axs[1,0].hist(mae_per_instance, bins=40, color='skyblue', edgecolor='black', alpha=0.7)
    axs[1,0].set_title('Histogram of MAE per Instance')
    axs[1,0].set_xlabel('MAE')
    axs[1,0].set_ylabel('Frequency')
    axs[1,0].grid(axis='y', alpha=0.75)

    # Next
    axs[1,1].hist(mse_per_instance, bins=40, color='orange', edgecolor='black', alpha=0.7)
    axs[1,1].set_title('Histogram of MSE per Instance')
    axs[1,1].set_xlabel('MSE')
    axs[1,1].set_ylabel('Frequency')
    axs[1,1].grid(axis='y', alpha=0.75)

    correct = np.sign(preds) == np.sign(targets)

    axs[2,0].scatter(targets[correct], preds[correct], sizes=[1], color='skyblue', label='Correct predictions')
    axs[2,0].scatter(targets[~correct], preds[~correct], sizes=[1], color='orange', label='Incorrect predictions')
    axs[2,0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], color='red', linestyle='--', label='y = y_pred')  # Reference line
    axs[2,0].set_title('Targets vs Predicted Values')
    axs[2,0].set_xlabel('Targets')
    axs[2,0].set_ylabel('Preds')
    axs[2,0].axhline(0, color='black', linewidth=1, linestyle='-')
    axs[2,0].axvline(0, color='black', linewidth=1, linestyle='-')
    axs[2,0].legend()
    axs[2,0].grid()
    axs[2,0].axis('equal')  # Equal scaling for better visualization

    axs[2,1].scatter(targets[correct], mse_per_instance[correct], color='skyblue', sizes=[1])
    axs[2,1].scatter(targets[~correct], mse_per_instance[~correct], color='orange', sizes=[1])
    axs[2,1].set_title('Targets vs MSE')
    axs[2,1].set_xlabel('Targets')
    axs[2,1].set_ylabel('Mean Squared Error (MSE)')
    axs[2,1].axvline(0, color='black', linewidth=1, linestyle='-')
    axs[2,1].axhline(0, color='black', linewidth=1, linestyle='-')
    axs[2,1].grid()

    plt.show()

def plot_training_metrics(metrics : tuple[list[float]]):
    mses_tr, mses_val, accs_tr, accs_val = metrics
    
    best_mse_tr_idx = np.argmin(mses_tr)
    best_mse_val_idx = np.argmin(mses_val)
    best_acc_tr_idx = np.argmax(accs_tr)
    best_acc_val_idx = np.argmax(accs_val)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    fig.suptitle('TRAINING METRICS', fontsize=16)

    y_axis_mse = np.arange(.8, 1.3, .05)
    y_axis_acc = np.arange(.5, .7, .01)

    axes[0, 0].plot(mses_tr, color='y')
    axes[0, 0].set_title('train_loss')
    axes[0, 0].set_title('MSE TR vs EPOCH')
    axes[0, 0].set_xlabel('epoch')
    axes[0, 0].set_ylabel('mse_tr')
    axes[0, 0].set_yticks(y_axis_mse)
    axes[0, 0].grid(True)
    axes[0, 0].plot(best_mse_val_idx, mses_tr[best_mse_val_idx], 'rx', markersize=8,
                    label=f"Best val epoch : mse_tr = {mses_tr[best_mse_val_idx]:.3f}")
    axes[0, 0].axhline(mses_tr[best_mse_val_idx], color='r', linestyle='--')
    axes[0, 0].axvline(best_mse_val_idx, color='r', linestyle='--')
    axes[0, 0].plot(best_mse_tr_idx, mses_tr[best_mse_tr_idx], color='r', marker='*',
                    markersize=8, label=f"Best mse_tr : {mses_tr[best_mse_tr_idx]:.3f}")
    axes[0, 0].legend()
    

    axes[0, 1].plot(mses_val, color='g')
    axes[0, 1].set_title('val_loss')
    axes[0, 1].set_title('MSE VAL vs EPOCH')
    axes[0, 1].set_xlabel('epoch')
    axes[0, 1].set_ylabel('mse_val')
    axes[0, 1].set_yticks(y_axis_mse)
    axes[0, 1].grid(True)
    axes[0, 1].plot(best_mse_val_idx, mses_val[best_mse_val_idx], 'rx', markersize=8, 
                    label=f"Best val epoch : mse_val = {mses_val[best_mse_val_idx]:.3f}")
    axes[0, 1].axhline(mses_val[best_mse_val_idx], color='r', linestyle='--')
    axes[0, 1].axvline(best_mse_val_idx, color='r', linestyle='--')
    axes[0, 1].plot(best_mse_val_idx, mses_val[best_mse_val_idx], color='r', marker='*',
                    markersize=8, label=f"Best mse_val : {mses_val[best_mse_val_idx]:.3f}")
    axes[0, 1].legend()

    axes[1, 0].plot(accs_tr, color='b')
    axes[1, 0].set_title('train_acc')
    axes[1, 0].set_title('ACC TR vs EPOCH')
    axes[1, 0].set_xlabel('epoch')
    axes[1, 0].set_ylabel('acc_tr')
    axes[1, 0].set_yticks(y_axis_acc)
    axes[1, 0].grid(True)
    axes[1, 0].plot(best_mse_val_idx, accs_tr[best_mse_val_idx], 'rx', markersize=8,
                    label=f"Best val epoch : acc_tr = {accs_tr[best_mse_val_idx]:.3f}")
    axes[1, 0].axhline(accs_tr[best_mse_val_idx], color='r', linestyle='--')
    axes[1, 0].axvline(best_mse_val_idx, color='r', linestyle='--')
    axes[1, 0].plot(best_acc_tr_idx, accs_tr[best_acc_tr_idx], color='r', marker='*',
                    markersize=8, label=f"Best acc_tr : {accs_tr[best_acc_tr_idx]:.3f}")
    axes[1, 0].legend()

    axes[1, 1].plot(accs_val, color='m')
    axes[1, 1].set_title('val_loss')
    axes[1, 1].set_title('ACC VAL vs EPOCH')
    axes[1, 1].set_xlabel('epoch')
    axes[1, 1].set_ylabel('acc_val')
    axes[1, 1].set_yticks(y_axis_acc)
    axes[1, 1].grid(True)
    axes[1, 1].plot(best_mse_val_idx, accs_val[best_mse_val_idx], 'rx', markersize=8, 
                    label=f"Best val epoch : acc_val = {accs_val[best_mse_val_idx]:.3f}")
    axes[1, 1].axhline(accs_val[best_mse_val_idx], color='r', linestyle='--')
    axes[1, 1].axvline(best_mse_val_idx, color='r', linestyle='--')
    axes[1, 1].plot(best_acc_val_idx, accs_val[best_acc_val_idx], color='r', marker='*',
                    markersize=8, label=f"Best acc_val : {accs_val[best_acc_val_idx]:.3f}")
    axes[1, 1].legend()

    # Add layout adjustments
    plt.tight_layout()

    # Show plot
    plt.show()



def plot_heat_map(model : nn.Module, dataloader : DataLoader, n_games : int=51, vmax : int=0.25):
    device = next(model.parameters()).device
    
    data_iter = iter(dataloader)
    
    # Get attention maps
    with torch.no_grad():
        kq, v, targets, padding_masks = next(data_iter)
        kq = kq.to(device)
        v = v.to(device)
        targets = targets.to(device)
        padding_masks = padding_masks.to(device)
        _, attn_maps = model(kq, v)

    attn_maps = attn_maps[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    fig.suptitle("Attention Maps of Model Heads")

    sns.heatmap(attn_maps[0, :n_games, :n_games].cpu(), ax=axes[0, 0], cmap='plasma', cbar=True, vmin=0, vmax=vmax)
    axes[0, 0].set_title("Attention Head 1")
    axes[0, 0].set_xlabel("Game attended to")
    axes[0, 0].set_ylabel("Game attending")
    sns.heatmap(attn_maps[1, :n_games, :n_games].cpu(), ax=axes[0, 1], cmap='plasma', cbar=True, vmin=0, vmax=vmax)
    axes[0, 1].set_title("Attention Head 2")
    axes[0, 1].set_xlabel("Game attended to")
    axes[0, 1].set_ylabel("Game attending")
    sns.heatmap(attn_maps[2, :n_games, :n_games].cpu(), ax=axes[1, 0], cmap='plasma', cbar=True, vmin=0, vmax=vmax)
    axes[1, 0].set_title("Attention Head 3")
    axes[1, 0].set_xlabel("Game attended to")
    axes[1, 0].set_ylabel("Game attending")
    sns.heatmap(attn_maps[3, :n_games, :n_games].cpu(), ax=axes[1, 1], cmap='plasma', cbar=True, vmin=0, vmax=vmax)
    axes[1, 1].set_title("Attention Head 4")
    axes[1, 1].set_xlabel("Game attended to")
    axes[1, 1].set_ylabel("Game attending")
        
    # Add layout adjustments
    plt.tight_layout()

    # Show plot
    plt.show()

    
#### CHECKING LOGS

def get_training_logs(logs_dir, model_name, version):
    version_name = "version_" + str(version)
    
    logs_path = os.path.join(logs_dir, model_name, version_name, "metrics.csv")
    logs = pd.read_csv(logs_path)
    # Cleaning up dataframe
    # Fill NaN values in train columns with corresponding validation rows and vice versa
    logs = logs.groupby('epoch', as_index=False).apply(lambda x: x.ffill().bfill())

    # Drop duplicate rows (they might exist after filling)
    logs = logs.drop_duplicates(subset='epoch')

    # Drop 'step' column
    logs.drop(['step'], axis=1, inplace=True)

    # Reset the index after cleaning
    logs = logs.reset_index(drop=True)
    
    cols = ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc']
    logs = logs[cols]
    return logs

def save_from_ckpt():
    pass


#### USING FOR PREDICTION ####
def predict():
    pass
