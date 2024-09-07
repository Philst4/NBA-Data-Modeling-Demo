import sqlite3
import numpy as np
import torch
from torch.utils.data import (
    Dataset,
)
import random


#### FUNCTIONALITY FOR FEEDING DATA TO MODEL

class SeasonSequenceDataset(Dataset):
    def __init__(
        self, 
        db_path,
        table_name,
        season_col='SEASON_ID',
        date_col='GAME_DATE',
        data_cols='*',
        meta_cols=['SEASON_ID', 'GAME_DATE', 'MATCHUP'],
        start_season=21983,
        end_season=22023,
        transform = None,
    ):
    
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.table_name = table_name
        self.season_col = season_col
        self.date_col = date_col
        self.data_cols = data_cols
        self.meta_cols = meta_cols
        self.ordering = f"""
        ORDER BY {date_col}
        """
        
        self.seasons = self._set_unique_seasons(
            table_name,
            season_col,
            start_season, 
            end_season
        )
    
    def _set_unique_seasons(self, table_name, season_col, start_season, end_season):
        table_name = self.table_name
        season_col = self.season_col
        query = f"""
        SELECT DISTINCT {season_col} 
        FROM {table_name} 
        WHERE {season_col} BETWEEN {start_season} AND {end_season} 
        ORDER BY {season_col} ASC
        """
        self.cursor.execute(query)
        return [row[0] for row in self.cursor.fetchall()]
        
    def _get_main_query(self, season, include_meta=False):
        table_name = self.table_name
        season_col = self.season_col
        if include_meta:
            cols = self.meta_cols + self.data_cols
        else:
            cols = self.data_cols
        col_str = ', '.join(cols)
        return f"""
        SELECT {col_str}
        FROM {table_name}
        WHERE {season_col} = {season} 
        AND IS_HOME_for = {1}"""
    
    def get_full_season_sequence(
        self, 
        season, 
        include_meta=False
    ):
        main_query = self._get_main_query(season, include_meta)
        query = ''.join((main_query, self.ordering, ';'))
        self.cursor.execute(query)
        return self.cursor.fetchall()
    
    def _get_season(self, date):
        # Finds season associated with date
        # If no games played on date, gives 
        # season of most recent game preceding date
        query = f"""
        SELECT {self.season_col}
        FROM my_table
        WHERE {self.date_col} <= '{date}'
        ORDER BY {self.date_col} DESC
        LIMIT 1;
        """
        self.cursor.execute(query)
        res = self.cursor.fetchall()
        if not res:
            return 21983
        return res[0][0]
    
    def get_partial_season_sequence(
        self, 
        date, 
        include_meta=False
    ):
        # Gets season sequence preceding date
        season = self._get_season(date)
        main_query = self._get_main_query(season, include_meta)
        date_condn = f"""
        AND {self.date_col} < '{date}'
        """
        query = ''.join((main_query, date_condn, self.ordering, ';'))
        self.cursor.execute(query)
        return self.cursor.fetchall(), season
    
    def get_team_id(self, team_abbr, season):
        query = f"""
        SELECT NEW_TEAM_ID
        FROM team_metadata
        WHERE SEASON_ID = {season}
        AND TEAM_ABBREVIATION = '{team_abbr}'
        LIMIT 1;
        """
        self.cursor.execute(query)
        res = self.cursor.fetchall()
        if not res:
            return None
        return res[0][0]

    def get_team_abbr(self, season, team_id=None):
        if not team_id:
            team_id = random.randint(0, 29)
        query = f"""
        SELECT TEAM_ABBREVIATION
        FROM team_metadata
        WHERE SEASON_ID = {season}
        AND NEW_TEAM_ID = {team_id}
        LIMIT 1;
        """
        self.cursor.execute(query)
        res = self.cursor.fetchall()
        if not res:
            return None
        return res[0][0]


    def __len__(self):
        return len(self.seasons)

    def __getitem__(self, idx):
        # Get the date for the current index
        season = self.seasons[idx]
        # Get the sequence of rows corresponding to that date
        season_sequence = self.get_full_season_sequence(season)
        return season_sequence
    
    



# Turns given sequence of features into one-hot encoded sequence of features
def make_ohe(sequence : torch.Tensor, ohe_size=32, stack=True) -> torch.Tensor:
    # NOTE: NumPy has better support for mapping
    n, T, d = sequence.shape
    ohe_sequence = []

    for i in range(d):
        # (1) Give each feature value an ID
        feature_sequence = torch.Tensor.numpy(sequence[:, :, i])
        unique_values = np.unique(feature_sequence)
        id_mapping = dict(zip(unique_values, range(len(unique_values))))
        vectorized_id_mapping = np.vectorize(id_mapping.get)
        
        # (2) Map features to ID's; get OHE according to ID's
        feature_id_sequence = vectorized_id_mapping(feature_sequence)
        feature_ohe_sequence = np.eye(ohe_size)[feature_id_sequence]
        ohe_sequence.append(torch.from_numpy(feature_ohe_sequence).to(torch.float))
    
    if stack:
        return torch.cat(ohe_sequence, dim=-1)
    return ohe_sequence
