import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

class Gameset(Dataset):
    def __init__(self, df, feature_cols, target_cols, metadata_cols=['UNIQUE_ID', 'SEASON_ID', 'GAME_DATE', 'MATCHUP']):
        super(Gameset, self).__init__()

        assert 'UNIQUE_ID' in list(df.columns)
        assert [feature_col in list(df.columns) for feature_col in feature_cols]
        #keep_cols = features + ['UNIQUE_ID'] if 'UNIQUE_ID' not in features else features
        self.df = df.copy()
        self.idx_mapping = df['UNIQUE_ID'].unique() # In order that df was provided in
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.metadata_cols = metadata_cols

    def get_input_dim(self):
        return len(self.feature_cols)

    def get_output_dim(self):
        return len(self.target_cols)

    def __len__(self):
        return len(self.idx_mapping)

    def __getitem__(self, idx, as_df=False):
        """
        Returns tuple (X, y) at the specified index in given data.
        
        Use 'as_df' is for debugging.
        """
        if as_df:
            cols = self.metadata_cols + self.feature_cols + self.target_cols
            return self.df.loc[self.df['UNIQUE_ID'] == self.idx_mapping[idx], cols]
        else:
            # Return tuple of X, y
            game = self.df.loc[self.df['UNIQUE_ID'] == self.idx_mapping[idx]]
            X = game[self.feature_cols].values.squeeze(axis=0)
            targets = game[self.target_cols].values.squeeze(axis=0)
            return torch.tensor(X).float(), torch.tensor(targets).float()
