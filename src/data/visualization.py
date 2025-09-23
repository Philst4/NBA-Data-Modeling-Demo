import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

# Internal import
from src.data.processing import (
    get_rolling_avgs
)

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