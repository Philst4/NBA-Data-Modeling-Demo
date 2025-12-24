# COMMON DF PROCESSING #
import pandas as pd

def get_normalized_by_season(
    game_data: pd.DataFrame,
    game_metadata: pd.DataFrame,
    use_prev_season_to_scale: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:

    game_data = game_data.copy()
    meta = game_metadata[['UNIQUE_ID', 'SEASON_ID']].copy()
    game_data = game_data.merge(meta, on='UNIQUE_ID')

    cols_to_normalize = [
        c for c in game_data.columns
        if c not in ('UNIQUE_ID', 'SEASON_ID')
    ]

    # --- raw per-season stats ---
    season_stats = (
        game_data
        .groupby('SEASON_ID')[cols_to_normalize]
        .agg(['mean', 'std'])
    )

    # --- stats used for scaling ---
    if use_prev_season_to_scale:
        scaling_stats = season_stats.shift(1)

        # first season falls back to its own stats
        first_season = season_stats.index[0]
        scaling_stats.loc[first_season] = season_stats.loc[first_season]
    else:
        scaling_stats = season_stats

    means = scaling_stats.xs('mean', level=1, axis=1)
    stds  = scaling_stats.xs('std', level=1, axis=1)

    # --- normalize ---
    def scale_season(df):
        season = df.name
        return (df - means.loc[season]) / stds.loc[season]

    game_data[cols_to_normalize] = (
        game_data
        .groupby('SEASON_ID', group_keys=False)[cols_to_normalize]
        .apply(scale_season)
    )

    game_data_normalized = game_data.drop(columns='SEASON_ID')

    # --- format raw stats for return ---
    scaling_stats.columns = [
        f"{col}_{stat}" for col, stat in scaling_stats.columns
    ]
    game_data_means_stds = scaling_stats.reset_index()

    return game_data_normalized, game_data_means_stds

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

def add_rolling_avg_diffs(df):
    """
    'STAT_diff_for_prev_0' is 'STAT_for_prev_0' - 'STAT_ag_opp_prev_0'
    'STAT_diff_ag_prev_0' is 'STAT_ag_prev_0' - 'STAT_ag_opp_prev_0'
    """
    
    for_cols = df.filter(regex=r"_for_prev_")
    ag_opp_cols = df.filter(regex=r"_ag_opp_prev_")
    diff_for = for_cols.values - ag_opp_cols.values
    diff_for_cols = [
        c.replace("_for_prev_", "_diff_for_prev_")
        for c in for_cols.columns
    ]
    df[diff_for_cols] = diff_for

    ag_cols = df.filter(regex=r"_ag_prev_")
    for_opp_cols = df.filter(regex=r"_for_opp_prev_")

    diff_ag = ag_cols.values - for_opp_cols.values
    diff_ag_cols = [
        c.replace("_ag_prev_", "_diff_ag_prev_")
        for c in ag_cols.columns
    ]
    df[diff_ag_cols] = diff_ag

    return df

import pandas as pd
import numpy as np

def get_temporal_spatial_features(game_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Creates basic temporal and spatial features from game metadata.
    
    Returns per-row features:
        - B2B_for, B2B_ag
        - 3of4_for, 3of4_ag
        - travelled_for, travelled_ag
        - IS_HOME_for, IS_HOME_ag
        - UNIQUE_ID
    """

    df = game_metadata.copy()

    # Ensure datetime
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Sort for temporal ops
    df = df.sort_values(["NEW_TEAM_ID_for", "GAME_DATE"])

    # --- TEMPORAL FEATURES (FOR TEAM) ---------------------------------------

    # Days since last game for the for-team
    df["days_since_last_for"] = (
        df.groupby("NEW_TEAM_ID_for")["GAME_DATE"]
          .diff()
          .dt.days
    )

    df["B2B_for"] = (df["days_since_last_for"] == 1).astype(int)

    # Count games in previous 3 days (inclusive window size = 4 days)
    def games_last_3_days(dates):
        return [
            (dates >= d - pd.Timedelta(days=3)).sum() - 1
            for d in dates
        ]

    df["games_last_3_days_for"] = (
        df.groupby("NEW_TEAM_ID_for")["GAME_DATE"]
          .transform(games_last_3_days)
    )

    df["3of4_for"] = (df["games_last_3_days_for"] >= 2).astype(int)

    # --- TEMPORAL FEATURES (AG TEAM) ----------------------------------------
    # Since each game appears twice, we can self-join on GAME_ID

    ag_features = df[[
        "GAME_ID",
        "NEW_TEAM_ID_for",
        "B2B_for",
        "3of4_for"
    ]].rename(columns={
        "NEW_TEAM_ID_for": "NEW_TEAM_ID_ag",
        "B2B_for": "B2B_ag",
        "3of4_for": "3of4_ag"
    })

    df = df.merge(
        ag_features,
        on=["GAME_ID", "NEW_TEAM_ID_ag"],
        how="left"
    )

    # --- SPATIAL FEATURES (TRAVEL) ------------------------------------------

    # Previous game info for for-team
    df["prev_IS_HOME_for"] = (
        df.groupby("NEW_TEAM_ID_for")["IS_HOME"]
          .shift(1)
    )

    df["prev_TEAM_ID_ag"] = (
        df.groupby("NEW_TEAM_ID_for")["NEW_TEAM_ID_ag"]
          .shift(1)
    )

    def travelled_for(row):
        if pd.isna(row["prev_IS_HOME_for"]):
            return 0
        # Home -> Away or Away -> Home
        if row["IS_HOME"] != row["prev_IS_HOME_for"]:
            return 1
        # Away -> Away, check opponent city
        if row["IS_HOME"] == 0:
            return int(row["NEW_TEAM_ID_ag"] != row["prev_TEAM_ID_ag"])
        # Home -> Home
        return 0

    df["travelled_for"] = df.apply(travelled_for, axis=1)

    # Mirror travelled_for for opponent
    ag_travel = df[[
        "GAME_ID",
        "NEW_TEAM_ID_for",
        "travelled_for"
    ]].rename(columns={
        "NEW_TEAM_ID_for": "NEW_TEAM_ID_ag",
        "travelled_for": "travelled_ag"
    })

    df = df.merge(
        ag_travel,
        on=["GAME_ID", "NEW_TEAM_ID_ag"],
        how="left"
    )

    # --- HOME FLAGS ----------------------------------------------------------

    df["IS_HOME_for"] = df["IS_HOME"].astype(int)
    df["IS_HOME_ag"] = (1 - df["IS_HOME"]).astype(int)

    # --- RETURN --------------------------------------------------------------

    return df[[
        "UNIQUE_ID",
        "IS_HOME_for",
        "IS_HOME_ag",
        "B2B_for",
        "B2B_ag",
        "3of4_for",
        "3of4_ag",
        "travelled_for",
        "travelled_ag"
    ]]
