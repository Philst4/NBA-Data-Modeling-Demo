from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd


def season_to_str(season : int) -> str:
    """Converts reference of season start to reference of entire season.

    The resulting string can be used to specify queries for the NBA API.
    Example use: season_to_str(2023) gives '2023-24'

    Args: 
        season: an int meant to reference the start of an NBA season 
    
    Returns:
        A string referencing the entire season, meant to be used to 
        query from the NBA API. Example use: season_to_str(2023) gives '2023-24'
    
    Raises:
        TypeError: Input season is not an int
    
    """    
    if not isinstance(season, int):
        raise TypeError("Season start must be of type int ")
    
    return str(season) + '-' + str(season + 1)[-2:]


def read(first_season : int=None) -> pd.DataFrame:
    """Reads in data via NBA API from season first specified to latest

    This function reads in data from the NBA API's 'LeagueGameFinder' endpoint,
    season by season (because season must be specified for access). Data for each
    season is then stored in its pd.DataFrame format, and then eventually merged
    into a single pd.DataFrame, which is returned to the caller.


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


    if first_season is None:
        first_season = 1983 # anecdotally first season with good data available
    
    print(f" * Reading in regular season games from {season_to_str(first_season)} to current season")

    # Read in all data from first season specified to current season
    seasons = range(first_season, pd.Timestamp.now().year)
    season_dfs = []
    for season in seasons:
        season_str = season_to_str(season)
        print(" * Reading in games from ", season_str, "...")
        game_finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season_str,
            season_type_nullable='Regular Season',
            league_id_nullable='00' # Specifies that the league is NBA
        )

        season_df = game_finder.get_data_frames()[0]
        # Drop data read in where N/A's in WL column; means game is still in progress
        season_df = season_df.dropna(subset=['WL'])
        season_dfs.append(season_df)
    new_games = pd.concat(season_dfs) # gives dataframe
    return new_games


def write(games : pd.DataFrame, write_path : str='../data/raw/raw.csv') -> None:
    games.to_csv(write_path, index=False)
    print(" * Data written to: ", write_path)
    return

if __name__ == "__main__":
    games = read(first_season=1983)
    write(games)
    