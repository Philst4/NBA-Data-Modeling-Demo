from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
from typing import Optional

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



def ingest_from_nba_api(first_season : int=None) -> Optional[pd.DataFrame]:
    """Reads in data via NBA API from season first specified to latest

    This function reads in data from the NBA API's 'LeagueGameFinder' endpoint.
    It does so by making requests for each season since season must be specified 
    for access). Received data is then stored in its pd.DataFrame format. Once all
    requests are made the pd.DataFrame's are merged into a single pd.DataFrame, 
    which is returned to the caller. If no data is found, returns None. This is case
    if specified start season is set to current year, but no season has started yet.


    Args:
        first_season: An integer referencing the start of the first season to
            read in. Anecdotally, 1983 seems to be the first season with good
            available data via 'LeagueGameFinder' endpoint

    Returns:
        A pd.DataFrame containing all specified data read in, following the
        structure of 'LeagueGameFinder' endpoint (visit NBA_API docs for more info)

    NOTE: requests are handled and built on via the NBA API. Any unexpected errors
    may be a result of this dependence. For more info visit the following link:
    https://github.com/swar/nba_api
    
    """

    current_season = pd.Timestamp.now().year # Case of having not started handled later

    # Sets first season within reasonable range
    if first_season is None or first_season < 1983:
        print(" * First season with good available data is '1983-84'")
        first_season = 1983 # anecdotally first season with good data available
    
    elif first_season > current_season:
        print(f" * Season {season_to_str(first_season)} hasn't happened yet; starting with {season_to_str(current_season)}")
        first_season = current_season
    
    
    # Get season strings
    seasons = list(map(season_to_str, range(first_season, current_season+1)))
    print(f" * Reading in regular season games from {seasons[0]} to {seasons[-1]}")

    # Read in all data from first season specified to current season
    # Last date specifying season that hasn't 
    season_dfs = []
    for season in seasons:
        print(" * Reading in games from ", season, "...")
        game_finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable='Regular Season',
            league_id_nullable='00' # Specifies that the league is NBA
        )

        season_df = game_finder.get_data_frames()[0]
        # Drop data read in where N/A's in WL column; means game is still in progress
        season_df = season_df.dropna(subset=['WL'])
        
        # Adds dataframe to list if not empty
        if len(season_df) > 0:
            season_dfs.append(season_df)
    
    if len(season_dfs) == 0:
        print(" * No games were found, Please choose an earlier starting year ")
        new_games = None
    else:
        new_games = pd.concat(season_dfs) # gives dataframe
    return new_games


def write_raw_to_csv(games : pd.DataFrame, write_dir : str, write_name : str='raw.csv') -> None:
    write_path = '/'.join((write_dir, write_name))
    games.to_csv(write_path, index=False)
    print(" * Data written to: ", write_path)
    return


    