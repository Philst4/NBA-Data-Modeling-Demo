import sqlite3
import numpy as np
import torch
from torch.utils.data import (
    Dataset, 
    DataLoader
)
from torch.nn.utils.rnn import pad_sequence


#### FUNCTIONALITY FOR FEEDING DATA TO MODEL

# Finds all data in sequence
# This class only supports numerical data. At the moment
class SequenceDataset(Dataset):
    def __init__(
            self, 
            db_path, 
            table_name, 
            season_col, 
            date_col, 
            start_season, 
            end_season, 
            cols_to_select,
            transform=None
        ):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.table_name = table_name
        self.season_col = season_col
        self.date_col = date_col
        self.start_season = start_season
        self.end_season = end_season
        self.dates = self._get_unique_dates(table_name, season_col, start_season, end_season)
        self.partitioned_cols = cols_to_select
        self.main_query = self._get_main_query(table_name, season_col, date_col, cols_to_select)
        self.transform = transform # NOTE: should not include ToTensor.

    def _get_unique_dates(self, table_name, season_col, start_season, end_season):
        # Query to get all unique dates in the database
        query = f"SELECT DISTINCT {season_col} FROM {table_name} WHERE {season_col} BETWEEN {start_season} AND {end_season} AND IS_HOME_for = 1"
        self.cursor.execute(query)
        return [row[0] for row in self.cursor.fetchall()]
    
    def _get_main_query(self, table_name, season_col, date_col, cols_to_select):
        flatten = lambda nested_list : [item for sublist in nested_list for item in sublist]
        cols = ', '.join(flatten(cols_to_select))
        return f"SELECT {cols} FROM {table_name} WHERE {season_col} = ? ORDER BY {date_col}"

    def _get_season_sequence(self, season):
        # Query to get all rows for the given date, optionally ordered
        self.cursor.execute(self.main_query, (season,))
        return self.cursor.fetchall() # returns a list of tuples

    # Modify to support strings
    def _get_partitioned_selection(self, selection):
        selection = torch.tensor(selection)
        partitioned_columns = self.partitioned_cols
        partitioned_selection = []
        i = 0
        for partition in partitioned_columns:
            partition_len = len(partition)
            partitioned_selection.append(selection[:, i:i+partition_len])
            i += partition_len
        return tuple(partitioned_selection)

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        # Get the date for the current index
        date = self.dates[idx]
        # Get the sequence of rows corresponding to that date
        sequence = self._get_season_sequence(date)
        partitioned_selection = self._get_partitioned_selection(sequence)

        if self.transform is not None:
            partitioned_selection = tuple(map(self.transform, partitioned_selection))

        return partitioned_selection


# Pads sequences of given batch to match
def collate_fn(batch):
    partitions = zip(*batch)
    padder = lambda b : pad_sequence(b, batch_first=True, padding_value=0)
    padded_batch = tuple(map(padder, partitions))
    return padded_batch


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
