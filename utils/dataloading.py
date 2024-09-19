import sqlite3
import numpy as np
import torch
from torch.utils.data import (
    Dataset,
    DataLoader # DO NOT REMOVE
)
from torch.nn.utils.rnn import pad_sequence


#### FUNCTIONALITY FOR FEEDING DATA TO MODEL

class SeasonSequenceDataset(Dataset):
    def __init__(
        self, 
        db_path,
        blueprint,
        seasons=None,
    ):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        # List of table info
        self.table_names = self.extract_table_names(blueprint)
        self.cols_at_table = self.make_cols_at_table_map(blueprint)
        self.flattened_columns = self.extract_columns_flattened(blueprint)
        self.cols_at_transform = self.make_cols_at_transform_map(blueprint)
        self.cols_at_group = self.make_cols_at_group_map(blueprint)
        
        # NOTE: seasons is a tuple indicating interval of seasons to include
        if seasons is None:
            self.seasons = None
        else:
            self.seasons = [row[0] for row in self._query_unique_seasons(
                seasons[0], 
                seasons[-1]
            )]
    
    #### INITIALIZATION METHODS
    def extract_table_names(self, blueprint):
        return [entry['table_name'] for entry in blueprint]
    
    def make_cols_at_table_map(self, blueprint):
        res = {}
        for entry in blueprint:
            table_name = entry['table_name']
            res[table_name] = []
            for col in entry['channel']:
                res[table_name].append(col[0])
        return res
    
    def extract_columns_flattened(self, blueprint):
        columns = []
        for entry in blueprint:
            table_name = entry['table_name']
            for col in entry['channel']:
                col_name = col[0]
                columns.append('.'.join((table_name, col_name)))
        return columns
    
    def make_cols_at_transform_map(self, blueprint):
        cols_at_transform = {}
        for entry in blueprint:
            table_name = entry['table_name']
            for col, transform, _ in entry['channel']:
                if transform not in cols_at_transform:
                    cols_at_transform[transform] = []
                col = '.'.join((table_name, col))
                cols_at_transform[transform].append(col)
        return cols_at_transform
    
    def make_cols_at_group_map(self, blueprint):
        cols_at_grouping = {}
        for entry in blueprint:
            table_name = entry['table_name']
            for col, _, grouping in entry['channel']:
                if grouping not in cols_at_grouping:
                    cols_at_grouping[grouping] = []
                col = '.'.join((table_name, col))
                cols_at_grouping[grouping].append(col)
        return cols_at_grouping
    
    #### QUERIES
    def _query_unique_seasons(self, start_season : int, end_season : int):
        query = f"""
        SELECT DISTINCT SEASON_ID
        FROM team_metadata
        WHERE SEASON_ID BETWEEN {start_season} AND {end_season} 
        ORDER BY SEASON_ID ASC;
        """
        self.cursor.execute(query)
        return self.cursor.fetchall()
    
    def _query_full_season_data(self, season, include_cols=True):
        create_temp_cmd = f"""
        CREATE TABLE temp AS 
        SELECT UNIQUE_ID, GAME_DATE
        FROM game_metadata
        WHERE SEASON_ID = {season}
        AND IS_HOME = 1
        ORDER BY GAME_DATE ASC;
        """
        drop_temp_cmd = f"""
        DROP TABLE IF EXISTS temp;
        """
        self.cursor.execute(drop_temp_cmd)
        self.cursor.execute(create_temp_cmd)
        self.cursor.execute("SELECT * FROM temp;")
        unique_ids = [row[0] for row in self.cursor.fetchall()]
        cols_at_unique_id = {id : [] for id in unique_ids}
        for table_name in self.table_names:
            cols = self.cols_at_table[table_name]
            cols_to_select = ['.'.join((table_name, col)) for col in cols]
            cols_to_select_str =', '.join(cols_to_select)
            # Right join unsupported
            query = f"""
            SELECT temp.UNIQUE_ID, {cols_to_select_str}
            FROM {table_name}
            INNER JOIN temp
            ON temp.UNIQUE_ID = {table_name}.UNIQUE_ID
            ORDER BY temp.GAME_DATE ASC;
            """
            self.cursor.execute(query)
            selection = self.cursor.fetchall()
            for row in selection:
                id, data = row[0], row[1:]
                cols_at_unique_id[id] += data
        self.cursor.execute(drop_temp_cmd)
        # Keep rows with all fields
        selections = [
            tuple(cols_at_unique_id[id]) for id in unique_ids if len(cols_at_unique_id[id]) == len(self.flattened_columns)
        ]
        return selections
    
    def _query_season_by_date(self, cutoff_date : str):
        # Extrapolates last-played season wrt given date
        query = f"""
        SELECT SEASON_ID
        FROM game_metadata
        WHERE GAME_DATE < '{cutoff_date}'
        ORDER BY GAME_DATE DESC
        LIMIT 1;
        """
        self.cursor.execute(query)
        selection = self.cursor.fetchall()
        return selection
    
    def _query_partial_season_data(self, cutoff_date : str):
        latest_season_selection = self._query_season_by_date(cutoff_date)
        if not latest_season_selection:
            latest_season = 21983
        else:
            latest_season = latest_season_selection[0][0]
        create_temp_cmd = f"""        
        CREATE TABLE temp AS 
        SELECT UNIQUE_ID, GAME_DATE
        FROM game_metadata
        WHERE SEASON_ID = {latest_season}
        AND IS_HOME = 1
        AND GAME_DATE < '{cutoff_date}'
        ORDER BY GAME_DATE ASC;
        """
        drop_temp_cmd = f"""
        DROP TABLE IF EXISTS temp;
        """
        self.cursor.execute(drop_temp_cmd)
        self.cursor.execute(create_temp_cmd)
        self.cursor.execute("SELECT * FROM temp;")
        unique_ids = [row[0] for row in self.cursor.fetchall()]
        cols_at_unique_id = {id : [] for id in unique_ids}
        for table_name in self.table_names:
            cols = self.cols_at_table[table_name]
            cols_to_select = ['.'.join((table_name, col)) for col in cols]
            cols_to_select_str =', '.join(cols_to_select)
            # Right join unsupported
            query = f"""
            SELECT temp.UNIQUE_ID, {cols_to_select_str}
            FROM {table_name}
            INNER JOIN temp
            ON temp.UNIQUE_ID = {table_name}.UNIQUE_ID
            ORDER BY temp.GAME_DATE ASC;
            """
            self.cursor.execute(query)
            selection = self.cursor.fetchall()
            for row in selection:
                id, data = row[0], row[1:]
                cols_at_unique_id[id] += data
        self.cursor.execute(drop_temp_cmd)
        # Keep rows with all fields
        selections = [
            tuple(cols_at_unique_id[id]) for id in unique_ids if len(cols_at_unique_id[id]) == len(self.flattened_columns)
        ]
        return selections, latest_season
    
    def _query_season_means_stds(self, season : int, table, cols_to_select):
        # Select from previous season
        if season > 21983:
            season -= 1
        cols_to_select_str = ', '.join(cols_to_select) 
        query = f"""
        SELECT {cols_to_select_str}
        FROM {table}
        WHERE SEASON_ID = {season}
        LIMIT 1;
        """
        self.cursor.execute(query)
        return self.cursor.fetchall()
    
    def _query_team_id(self, season : int, team_abbr : str):
        query = f"""
        SELECT TEAM_ID
        FROM team_metadata
        WHERE SEASON_ID = {season}
        AND TEAM_ABBREVIATION = '{team_abbr}'
        LIMIT 1;
        """
        self.cursor.execute(query)
        res = self.cursor.fetchall()
        return res

    def _query_team_abbr(self, season : int, team_id : str):
        query = f"""
        SELECT TEAM_ABBREVIATION
        FROM team_metadata
        WHERE SEASON_ID = {season}
        AND TEAM_ID = {team_id}
        LIMIT 1;
        """
        self.cursor.execute(query)
        res = self.cursor.fetchall()
        return res
    
    #### GROUPING METHODS
    def group_by_col_from_raw(self, selection):
        # Transpose the list of tuples
        transposed_selection = list(zip(*selection))
        
        # Create a dictionary with the column names as keys
        selection_dict = {col: list(transposed_selection[i]) for i, col in enumerate(self.flattened_columns)}
        return selection_dict
    
    def group_for_model_from_col(self, selection_dict):
        new_dict = {}
        cols_at_group = self.cols_at_group
        for grouping, cols in cols_at_group.items():
            new_dict[grouping] = []
            for col in cols:
                new_dict[grouping].append(selection_dict[col])
        return new_dict
    
    #### TRANSFORMS
    def ohe(self, feature_sequence, ohe_size=32):
        # NOTE: NumPy has better support for mapping
        # (1) Get unique features
        unique_values = np.unique(feature_sequence)
        id_mapping = dict(zip(unique_values, range(len(unique_values))))
        vectorized_id_mapping = np.vectorize(id_mapping.get)
            
        # (2) Map features to ID's; get OHE according to ID's
        feature_id_sequence = vectorized_id_mapping(feature_sequence)
        feature_sequence_ohe = np.eye(ohe_size)[feature_id_sequence]
        feature_sequence_ohe = torch.tensor(feature_sequence_ohe)
        return feature_sequence_ohe.to(torch.float)
    
    
    def process_for_normalize(self, col):
        # Remove suffixes
        col = col.replace('_for', '')
        col = col.replace('_ag', '')
        # Fix table prefixes before '.'
        table, _, col = col.partition('.')
        table += '_summary'
        col = table + '.' + col
        # Add suffixes
        cols_to_select = [col + '_mean', col + '_std']
        # Define new table
        return table, cols_to_select
    
    def normalize(self, season, feature_sequence, col):
        feature_sequence = torch.tensor(feature_sequence)
        # Get info for query
        table, cols_to_select = self.process_for_normalize(col)
        # Query for mean, std
        selection = self._query_season_means_stds(season, table, cols_to_select)
        try:
            selection = torch.tensor(selection).squeeze(dim=0)
            mean, median = selection[0], selection[1]
            # Apply to sequence
            feature_sequence = (feature_sequence - mean) / median
        except RuntimeError:
            print(f" * Columns {cols_to_select} at season {season} is {selection}; not normalizing")
        return feature_sequence.to(torch.float)
        
        
    def transform(self, selection_dict, season):
        cols_at_transform = self.cols_at_transform
        for transform, cols in cols_at_transform.items():
            data_to_transform = [selection_dict[col] for col in cols]
            if transform is None:
                transformed_data = data_to_transform
            elif transform == 'normalize':
                # Call normalize
                normalize = lambda x : self.normalize(season, x[0], x[1])
                normalize_input = list(zip(data_to_transform, cols))
                transformed_data = list(map(normalize, normalize_input))
            elif transform == 'ohe':
                # Call ohe
                transformed_data = list(map(self.ohe, data_to_transform))
            for i, col in enumerate(cols):
                selection_dict[col] = transformed_data[i]
        return selection_dict
    
    #### GET METHODS/FUNCTIONALITY      
    def get_full_season_data(
        self, 
        season : int, 
        apply_transforms=True,
        group_for_model=True
    ):
        selection = self._query_full_season_data(season)
        selection_dict = self.group_by_col_from_raw(selection)
        if apply_transforms:
            selection_dict = self.transform(selection_dict, season)
        if group_for_model:
            selection_dict = self.group_for_model_from_col(selection_dict)
        return selection_dict
    
    def get_partial_season_data(
        self, 
        cutoff_date : str,
        apply_transforms=True,
        group_for_model=True
    ):
        selection, season = self._query_partial_season_data(cutoff_date)
        selection_dict = self.group_by_col_from_raw(selection)
        if apply_transforms:
            selection_dict = self.transform(selection_dict, season)
        if group_for_model:
            selection_dict = self.group_for_model_from_col(selection_dict)
        return selection_dict, season
    
    #### PYTORCH DATASET FUNCTIONALITY
    def __len__(self):
        return len(self.seasons)

    def __getitem__(self, idx):
        # Get the date for the current index
        season = self.seasons[idx]
        selection_dict = self.get_full_season_data(season)
        return selection_dict


#### COLLATOR (and padding)
def make_padding_masks(samples):
    not_padding = [torch.ones(len(sample)) for sample in samples]
    padding_masks = pad_sequence(not_padding, batch_first=True, padding_value=0)
    return padding_masks.unsqueeze(dim=-1)

def collate_fn(batch):    
    made_padding = False
    keys = batch[0].keys()
    dict_of_batch = {}
    for key in keys:
        samples_at_key = []
        if key == 'metadata':
            for dict_of_sample in batch:
                samples_at_key.append(list(zip(*dict_of_sample[key])))
            entry = samples_at_key
        else:
            for dict_of_sample in batch:
                samples_at_key.append(torch.stack(dict_of_sample[key], dim=1))
            entry = pad_sequence(samples_at_key, batch_first=True, padding_value=0)
            if not made_padding:
                dict_of_batch['padding_masks'] = make_padding_masks(samples_at_key)
                made_padding = True
        dict_of_batch[key] = entry 
    return dict_of_batch

