# COMMON DF PROCESSING #
import pandas as pd

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

# Helper
def rename_for_rolled_opp(col):
    if '_ag' in col:
        return col.replace('_ag', '_for_opp')
    elif '_for' in col:
        return col.replace('_for', '_ag_opp')
    return col + '_opp'

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