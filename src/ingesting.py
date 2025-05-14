import sys
import os
import warnings
import time

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
from nba_api.stats.endpoints import (
    leaguegamefinder,
    shotchartdetail,
)
from nba_api.stats.static import (
    teams
)
import pandas as pd
from typing import Optional

# Internal imports
from src.utils import (
    season_to_str
)

#### INGESTION FUNCTION VARIANTS

def ingest_from_leaguegamefinder(
    start_season: int = None,
    start_date: str = None,
    sleep_time : float = 1.0
) -> Optional[pd.DataFrame]:
    """Reads in data via NBA API from season first specified to latest or from start date.

    Args:
        start_season: An integer referencing the start of the first season to read in.
        start_date: A string in 'yyyy-mm-dd' format specifying the date from which to
            start reading games.

    Returns:
        A pd.DataFrame containing all specified data read in, following the structure
        of 'LeagueGameFinder' endpoint. If no data is found, returns None.
    """
    
    # Ensure only one of start_season or start_date is provided
    assert not (start_season and start_date), "Provide either 'start_season' or 'start_date', not both."

    # Suppress the FutureWarning to avoid seeing it for now
    warnings.simplefilter(action='ignore', category=FutureWarning)

    current_season = pd.Timestamp.now().year

    # If start_date is provided
    if start_date:
        # NOTE: granuilarity for working reads is season;
        # start_season set to earliest potential season 
        # Specified by start_date
        start_season = int(start_date[:4]) - 1

    # Deal with start_season
    if start_season is None or start_season < 1983:
        print(" * First season with good available data is '1983-84'")
        start_season = 1983  # Start season with reliable data
    
    elif start_season > current_season:
        print(f" * Season {season_to_str(start_season)} hasn't happened yet; starting with {season_to_str(current_season)}")
        start_season = current_season

    # Get season strings
    seasons = list(map(season_to_str, range(start_season, current_season + 1)))
    print(f" * Reading in regular season games from {seasons[0]} to {seasons[-1]}")

    # Read in data season by season
    season_dfs = []
    for season in seasons:
        print(" * Reading in games from ", season, "...")
        game_finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable='Regular Season',
            league_id_nullable='00'  # Specifies NBA league
        )
        # Sleep after each request
        time.sleep(sleep_time)

        # Handle received data
        season_df = game_finder.get_data_frames()[0]
        season_df_len = len(season_df)
        season_df = season_df.dropna(subset=['WL'])  # Drop games still in progress
        n_dropped = season_df_len - len(season_df)
        string = f" - - {season_df_len} games ingested"
        if n_dropped > 0:
            string += f" (dropped {n_dropped} games in progress)"
        print(string)    
        if len(season_df) > 0:
            season_dfs.append(season_df)

    if len(season_dfs) == 0:
        print(" * No games were found, please choose an earlier starting year.")
        return None
    else:
        new_games = pd.concat(season_dfs)
        return new_games

# Dictionary of ingestion functions
ingestion_fns = {
    'leaguegamefinder' : ingest_from_leaguegamefinder
}


    