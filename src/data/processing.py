# COMMON DF PROCESSING #
import pandas as pd
import numpy as np

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

#### NEW ####

def get_rolling_stats(
    game_data: pd.DataFrame,
    game_metadata: pd.DataFrame,
    game_data_cols=None,
    windows: list = [1],
    stats: tuple = ("mean", "std", "median", "IQR", "quantiles"),
    quantiles: tuple = (0.10, 0.90),
    need_opp: bool = True,
    set_na_to_0: bool = True,
) -> pd.DataFrame:
    """
    Compute rolling distributional statistics (leak-free) for game-level data.

    Rolling stats are computed per team per season, excluding the current game
    via shift(1).

    Supported stats:
        - mean        -> _mean_prev_k
        - std         -> _std_prev_k
        - median      -> _median_prev_k
        - IQR         -> _IQR_prev_k
        - quantiles   -> _Q{int(q*100)}_prev_k

    Parameters
    ----------
    windows : list
        Rolling window sizes. window <= 0 or > 82 -> treated as season window (prev_0)
    """

    # ------------------------
    # Prep inputs
    # ------------------------
    if game_data_cols is not None:
        game_data = game_data[["UNIQUE_ID"] + game_data_cols].copy()
    else:
        game_data = game_data.copy()

    metadata_cols = [
        "UNIQUE_ID",
        "SEASON_ID",
        "NEW_TEAM_ID_for",
        "NEW_TEAM_ID_ag",
        "GAME_DATE",
    ]
    game_metadata = game_metadata[metadata_cols].copy()

    cols_to_roll = [c for c in game_data.columns if c != "UNIQUE_ID"]

    # Merge metadata
    df = pd.merge(game_metadata, game_data, on="UNIQUE_ID")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    all_outputs = []

    # ------------------------
    # Helper: compute rolling stats
    # ------------------------
    def compute_rolling(group, window):
        rolled = group[cols_to_roll].shift(1).rolling(
            window=window, min_periods=1
        )

        out = []

        if "mean" in stats:
            out.append(rolled.mean().add_suffix("_mean"))

        if "std" in stats:
            out.append(rolled.std().add_suffix("_std"))

        if "median" in stats:
            out.append(rolled.median().add_suffix("_median"))

        if "IQR" in stats:
            q75 = rolled.quantile(0.75)
            q25 = rolled.quantile(0.25)
            out.append((q75 - q25).add_suffix("_IQR"))

        if "quantiles" in stats:
            for q in quantiles:
                q_df = rolled.quantile(q)
                suffix = f"_Q{int(q * 100)}"
                out.append(q_df.add_suffix(suffix))
                
        # ------------------------
        # GAME DENSITY 
        # ------------------------
        if "game_density" in stats:
            dates = group["GAME_DATE"]

            densities = []
            for d in dates:
                start = d - pd.Timedelta(days=window)
                count = ((dates < d) & (dates >= start)).sum()
                densities.append(count / window)

            out.append(
                pd.DataFrame(
                    {"game_density": densities},
                    index=group.index
                )
            )

        return pd.concat(out, axis=1)

    # ------------------------
    # Main loop over windows
    # ------------------------
    for window in windows:
        if window <= 0 or window > 82:
            window = 100
            window_str = "0"
        else:
            window_str = str(window)

        # -------- FOR TEAM --------
        df_for = df.sort_values(
            ["SEASON_ID", "NEW_TEAM_ID_for", "GAME_DATE"]
        )

        rolled_for = (
            df_for
            .groupby(
                ["SEASON_ID", "NEW_TEAM_ID_for"], 
                group_keys=False
            )
            .apply(
                lambda g: compute_rolling(g, window),
                include_groups=False
            )
        )

        rolled_for = rolled_for.set_index(df_for.index)
        rolled_for.columns = [
            f"{c}_prev_{window_str}" for c in rolled_for.columns
        ]
        rolled_for["UNIQUE_ID"] = df_for["UNIQUE_ID"].values

        # -------- OPP TEAM --------
        if need_opp:
            df_opp = df.sort_values(
                ["SEASON_ID", "NEW_TEAM_ID_ag", "GAME_DATE"]
            )

            rolled_opp = (
                df_opp
                .groupby(
                    ["SEASON_ID", "NEW_TEAM_ID_ag"], 
                    group_keys=False
                )
                .apply(
                    lambda g: compute_rolling(g, window),
                    include_groups=False
                )
            )

            rolled_opp = rolled_opp.set_index(df_opp.index)
            rolled_opp.columns = [
                f"{c}_prev_{window_str}" for c in rolled_opp.columns
            ]

            # Rename from for → opp perspective
            rolled_opp = rolled_opp.rename(columns=rename_for_rolled_opp)
            rolled_opp["UNIQUE_ID"] = df_opp["UNIQUE_ID"].values

            combined = pd.merge(
                rolled_for, rolled_opp, on="UNIQUE_ID", how="inner"
            )
        else:
            combined = rolled_for

        all_outputs.append(combined)

    # ------------------------
    # Final assembly
    # ------------------------
    final = pd.concat(all_outputs, axis=1)
    final = final.loc[:, ~final.columns.duplicated()]

    if set_na_to_0:
        final = final.fillna(0)

    return final


import re

def add_rolling_diffs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rolling diff features in a non-fragmenting, vectorized way.
    """

    new_cols = {}

    # -----------------------------
    # FOR-team diffs
    # -----------------------------
    for col in df.columns:
        m = re.match(r"(.+)_for_(.+)_prev_(.+)", col)
        if not m:
            continue

        base, stat, window = m.groups()
        opp_col = f"{base}_ag_opp_{stat}_prev_{window}"
        diff_col = f"{base}_for_{stat}_diff_prev_{window}"

        if opp_col in df.columns:
            new_cols[diff_col] = df[col] - df[opp_col]

    # -----------------------------
    # AG-team diffs
    # -----------------------------
    for col in df.columns:
        m = re.match(r"(.+)_ag_(.+)_prev_(.+)", col)
        if not m:
            continue

        base, stat, window = m.groups()
        opp_col = f"{base}_for_opp_{stat}_prev_{window}"
        diff_col = f"{base}_ag_{stat}_diff_prev_{window}"

        if opp_col in df.columns:
            new_cols[diff_col] = df[col] - df[opp_col]

    if not new_cols:
        return df

    # Single concat → no fragmentation
    diff_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, diff_df], axis=1)

def add_rolling_avg_diffs(df):
    """
    'STAT1_for_STAT2_diff_prev_0' is 'STAT1_for_STAT2_prev_0' - 'STAT1_ag_opp_STAT2_prev_0'
    'STAT1_ag_STAT2_diff_prev_0' is 'STAT1_ag_STAT2_prev_0' - 'STAT1_ag_opp_STAT2_prev_0'
    E.g., 'PLUS_MINUS_for_mean_diff_prev_0'
    """
    
    for_cols = df.filter(regex=r"_for_mean_prev_")
    ag_opp_cols = df.filter(regex=r"_ag_mean_opp_prev_")
    diff_for = for_cols.values - ag_opp_cols.values
    diff_for_cols = [
        c.replace("_for_mean_prev_", "_diff_for_mean_prev_")
        for c in for_cols.columns
    ]
    df[diff_for_cols] = diff_for

    ag_cols = df.filter(regex=r"_ag_mean_prev_")
    for_opp_cols = df.filter(regex=r"_for_mean_opp_prev_")

    diff_ag = ag_cols.values - for_opp_cols.values
    diff_ag_cols = [
        c.replace("_ag_mean_prev_", "_diff_ag_mean_prev_")
        for c in ag_cols.columns
    ]
    df[diff_ag_cols] = diff_ag

    return df

def get_temporal_spatial_features(
    game_metadata: pd.DataFrame,
    scale_0_1: bool = True
) -> pd.DataFrame:
    """
    Creates temporal and spatial features from game metadata.

    Always returns:
        - IS_HOME_for, IS_HOME_ag
        - B2B_for, B2B_ag
        - 3of4_for, 3of4_ag
        - travelled_for, travelled_ag

    Season progression features:
        If scale_0_1=True:
            *_scaled_in_szn
            *_scaled_x_szn
        If scale_0_1=False:
            unscaled versions only
    """

    df = game_metadata.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["SEASON_ID", "NEW_TEAM_ID_for", "GAME_DATE"])

    # ------------------------------------------------------------
    # TEMPORAL (FOR)
    # ------------------------------------------------------------
    df["days_since_last_for"] = (
        df.groupby(["SEASON_ID", "NEW_TEAM_ID_for"])["GAME_DATE"]
          .diff()
          .dt.days
    )

    df["B2B_for"] = (df["days_since_last_for"] == 1).astype(int)

    def games_last_3_days(dates):
        return [(dates >= d - pd.Timedelta(days=3)).sum() - 1 for d in dates]

    df["games_last_3_days_for"] = (
        df.groupby(["SEASON_ID", "NEW_TEAM_ID_for"])["GAME_DATE"]
          .transform(games_last_3_days)
    )

    df["3of4_for"] = (df["games_last_3_days_for"] >= 2).astype(int)

    # ------------------------------------------------------------
    # SEASON PROGRESSION (FOR)
    # ------------------------------------------------------------
    df["games_into_season_for"] = (
        df.groupby(["SEASON_ID", "NEW_TEAM_ID_for"]).cumcount() + 1
    )

    first_game = (
        df.groupby(["SEASON_ID", "NEW_TEAM_ID_for"])["GAME_DATE"]
          .transform("min")
    )

    df["days_into_season_for"] = (df["GAME_DATE"] - first_game).dt.days

    season_games = (
        df.groupby(["SEASON_ID", "NEW_TEAM_ID_for"])["GAME_DATE"]
          .transform("count")
    )

    season_days = (
        df.groupby(["SEASON_ID", "NEW_TEAM_ID_for"])["GAME_DATE"]
          .transform("max") - first_game
    ).dt.days

    df["games_left_in_season_for"] = season_games - df["games_into_season_for"]
    df["days_left_in_season_for"] = season_days - df["days_into_season_for"]

    # ------------------------------------------------------------
    # MIRROR TO AG
    # ------------------------------------------------------------
    ag_map = {
        "B2B_for": "B2B_ag",
        "3of4_for": "3of4_ag",
        "games_into_season_for": "games_into_season_ag",
        "days_into_season_for": "days_into_season_ag",
        "games_left_in_season_for": "games_left_in_season_ag",
        "days_left_in_season_for": "days_left_in_season_ag",
    }

    ag_features = (
        df[["GAME_ID", "NEW_TEAM_ID_for"] + list(ag_map.keys())]
        .rename(columns={"NEW_TEAM_ID_for": "NEW_TEAM_ID_ag", **ag_map})
    )

    df = df.merge(
        ag_features,
        on=["GAME_ID", "NEW_TEAM_ID_ag"],
        how="left"
    )

    # ------------------------------------------------------------
    # TRAVEL
    # ------------------------------------------------------------
    df["prev_IS_HOME_for"] = (
        df.groupby(["SEASON_ID", "NEW_TEAM_ID_for"])["IS_HOME"].shift(1)
    )

    df["prev_TEAM_ID_ag"] = (
        df.groupby(["SEASON_ID", "NEW_TEAM_ID_for"])["NEW_TEAM_ID_ag"].shift(1)
    )

    def travelled_for(row):
        if pd.isna(row["prev_IS_HOME_for"]):
            return 0
        if row["IS_HOME"] != row["prev_IS_HOME_for"]:
            return 1
        if row["IS_HOME"] == 0:
            return int(row["NEW_TEAM_ID_ag"] != row["prev_TEAM_ID_ag"])
        return 0

    df["travelled_for"] = df.apply(travelled_for, axis=1)

    df = df.merge(
        df[["GAME_ID", "NEW_TEAM_ID_for", "travelled_for"]]
          .rename(columns={
              "NEW_TEAM_ID_for": "NEW_TEAM_ID_ag",
              "travelled_for": "travelled_ag"
          }),
        on=["GAME_ID", "NEW_TEAM_ID_ag"],
        how="left"
    )

    # ------------------------------------------------------------
    # HOME FLAGS
    # ------------------------------------------------------------
    df["IS_HOME_for"] = df["IS_HOME"].astype(int)
    df["IS_HOME_ag"] = (1 - df["IS_HOME"]).astype(int)

    # ------------------------------------------------------------
    # SCALING (EXCLUSIVE)
    # ------------------------------------------------------------
    progression_cols = [
        "games_into_season",
        "games_left_in_season",
        "days_into_season",
        "days_left_in_season",
    ]

    if scale_0_1:
        max_games_x_szn = season_games.max()
        max_days_x_szn = season_days.max()

        for side in ["for", "ag"]:
            for col in progression_cols:
                base = f"{col}_{side}"

                if "games" in col:
                    df[f"{base}_scaled_in_szn"] = df[base] / season_games
                    df[f"{base}_scaled_x_szn"] = df[base] / max_games_x_szn
                else:
                    df[f"{base}_scaled_in_szn"] = df[base] / season_days
                    df[f"{base}_scaled_x_szn"] = df[base] / max_days_x_szn

    # ------------------------------------------------------------
    # FINAL COLUMN SELECTION (STRICT)
    # ------------------------------------------------------------
    base_feats = [
        "UNIQUE_ID",
        "IS_HOME_for", "IS_HOME_ag",
        "B2B_for", "B2B_ag",
        "3of4_for", "3of4_ag",
        "travelled_for", "travelled_ag",
    ]

    if scale_0_1:
        prog_feats = [
            c for c in df.columns
            if c.endswith("_scaled_in_szn") or c.endswith("_scaled_x_szn")
        ]
    else:
        prog_feats = [
            f"{col}_{side}"
            for col in progression_cols
            for side in ["for", "ag"]
        ]

    return df[base_feats + prog_feats]

def scale_data(df, means_stds):
    """
    Returns df with all features 'like' those seen in mean std scaled
    
    For example: 'PTS_mean', and 'PTS_std' exists in means_stds;
    so all columns in df containing the same stem 'PTS' will
    be scaled by subtracting 'PTS_mean', and dividing by 'PTS_std'
    (e.g. 'PTS_for_prev_0', 'PTS_ag_opp_prev_5' will be scaled like this)
    """
    df = df.copy()
    
    # Expect means_stds to be a single-row df (e.g. per season)
    means = means_stds.filter(like='_mean').iloc[0]
    stds  = means_stds.filter(like='_std').iloc[0]

    for mean_col in means.index:
        stem = mean_col.replace('_mean', '')
        std_col = f"{stem}_std"

        if std_col not in stds.index:
            continue

        mean_val = means[mean_col]
        std_val = stds[std_col]

        # Avoid division by zero or NaNs
        if not np.isfinite(std_val) or std_val == 0:
            continue

        # Scale all matching feature columns
        feature_cols = [
            c for c in df.columns
            if c.startswith(stem) and not c.endswith(('_mean', '_std'))
        ]

        if feature_cols:
            df[feature_cols] = (df[feature_cols] - mean_val) / std_val

    return df
