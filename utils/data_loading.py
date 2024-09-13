import sqlite3
import numpy as np
import torch
from torch.utils.data import (
    Dataset,
)
from torch.nn.utils.rnn import pad_sequence
import random


#### FUNCTIONALITY FOR FEEDING DATA TO MODEL

class SeasonSequenceDataset(Dataset):
    def __init__(
        self, 
        db_path,
        table_name,
        ssd_config,
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
        self.season_col = season_col
        self.date_col = date_col
        self.table_name = table_name
        self.partitioned_data_cols = data_cols
        self.flattened_data_cols = self._flatten_partition(data_cols)
        self.meta_cols = meta_cols
        self.meta_len = len(meta_cols)
        self.ordering = f"""
        ORDER BY {date_col}
        """
        
        self.seasons = self._set_unique_seasons(
            table_name,
            season_col,
            start_season, 
            end_season
        )
    
    def _flatten_partition(self, partitioned_cols):
        # Flattens the list
        data_cols = [col for partition in partitioned_cols for col in partition]
        return data_cols
    
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
            cols = self.meta_cols + self.flattened_data_cols
        else:
            cols = self.flattened_data_cols
        col_str = ', '.join(cols)
        return f"""
        SELECT {col_str}
        FROM {table_name}
        WHERE {season_col} = {season} 
        AND IS_HOME_for = {1}"""
    
    def _separate_metadata(self, selection, include_meta):
        # Separate metadata
        if include_meta:
            metadata_len = self.meta_len
        else:
            metadata_len = 0
        metadata, data = [], []
        for row in selection:
            metadata.append(row[:metadata_len])
            data.append(row[metadata_len:])
        return metadata, torch.tensor(data)
    
    def _partition_data(self, data):
        partitioned_data_cols = self.partitioned_data_cols
        partitioned_data = []
        start = 0
        for p in partitioned_data_cols:
            end = start + len(p)
            partitioned_data.append(data[:, start : end])
            start = end
        return partitioned_data
    
    def _get_means_stds(self, season):
        data_cols = self.flattened_data_cols
        data_cols = [col.replace('_for', '') for col in data_cols]        
        data_cols = [col.replace('_ag', '') for col in data_cols]
        data_cols = list(set(data_cols)) # drop duplicates
        select_cols = [col + '_mean' for col in data_cols]
        select_cols += [col + '_std' for col in data_cols]
        print(data_cols)
        
        select_cols = ['PLUS_MINUS_mean', 'PLUS_MINUS_std']
        select_cols_str = ', '.join(select_cols)
        
        query = f"""
        SELECT {select_cols_str}
        FROM summary_stats
        WHERE SEASON_ID = ?
        LIMIT 1;
        """
        self.cursor.execute(query, (season,))
        res = self.cursor.fetchall()
        print(res)
        return
    
    def _process_data(self, partitioned_data):
        return partitioned_data
    
    def get_full_season_sequence(
        self, 
        season, 
        include_meta=False,
        normalize=False
    ):
        main_query = self._get_main_query(season, include_meta)
        query = ''.join((main_query, self.ordering, ';'))
        self.cursor.execute(query)
        selection  = self.cursor.fetchall()
        
        # Separate metadata
        metadata, data = self._separate_metadata(selection, include_meta)
        
        # Partition data
        data = self._partition_data(data)
        
        # Return tuple
        return metadata, data

    
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

    def __getitem__(self, idx, normalize=True):
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

#### COLLATOR
def make_padding_masks(batch):
    make_padding_mask = lambda sequence : torch.ones(sequence.shape[0])
    padding_masks = list(map(make_padding_mask, batch))
    padding_masks = pad_sequence(padding_masks, batch_first=True, padding_value=0)
    return padding_masks.unsqueeze(dim=-1)


def make_collate_fn(partition):
    pass


def collate_fn(batch, shuffle=False):    
    batch = list(map(torch.tensor, batch))
    padding_masks = make_padding_masks(batch)
    batch = pad_sequence(batch, batch_first=True, padding_value=0)
    
    kqs = make_ohe(batch[:, :, 0:2], shuffle)
    vs = batch[:, :, 2:3]
    targets = batch[:, :, 3:4]
    batch = (kqs, vs, targets, padding_masks)
    return batch    

