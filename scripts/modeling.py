print(" * Importing ... ")
import sqlite3
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np
from torchsummary import summary
import lightning as L
from lightning.pytorch import loggers as pl_loggers


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


##### MODEL ARCHITECTURE (PyTorch)
class TransformerLayerB(nn.Module):
    def __init__(self, 
                 input_dim_kq,  
                 embed_dim_kq,
                 input_dim_v,
                 embed_dim_v,
                 n_heads=1):
        super(TransformerLayerB, self).__init__()

        self.input_dim_kq = input_dim_kq
        self.embed_dim_kq = embed_dim_kq
        self.input_dim_v = input_dim_v
        self.embed_dim_v = embed_dim_v

        self.W_k = nn.Linear(input_dim_kq, embed_dim_kq)
        self.W_q = nn.Linear(input_dim_kq, embed_dim_kq)
        self.W_v = nn.Linear(input_dim_v, embed_dim_v)
        self.n_heads = n_heads
        self.ff_layer = nn.Sequential(
            nn.Linear(embed_dim_v, 4*embed_dim_v),
            nn.GELU(),
            nn.Linear(4*embed_dim_v, embed_dim_v)
        )

    def forward(self, kq, v) -> torch.Tensor:
        # FOR TIME BEING:
        #  * keys are OHE matchups
        #  * values are stats
        n, T, _ = kq.shape
        n_heads = self.n_heads

        keys = self.W_k(kq)
        queries = self.W_q(kq)
        values = self.W_v(v)

        keys = keys.view(n, T, n_heads, -1).transpose(1, 2).contiguous()
        queries = queries.view(n, T, n_heads, -1).transpose(1, 2).contiguous()
        values = values.view(n, T, n_heads, -1).transpose(1, 2).contiguous()

        attn_weights = queries @ keys.transpose(-2, -1)
        
        # Causal mask, cannot attend to diagonal
        # Test; allowing the first game to attend to itself
        mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=0)
        mask[0,0] = 0 # To get rid of NaN's
        attn_scores = F.softmax(attn_weights + mask, dim=-1)
        
        # v' = A @ v
        values = attn_scores @ values

        values = values.transpose(2, 1).contiguous().view(n, T, -1)
        values = self.ff_layer(values)
        
        return values


class Model(nn.Module):
    def __init__(self, 
                 input_dim_kq,  
                 embed_dim_kq,
                 input_dim_v,
                 embed_dim_v,
                 output_dim,
                 n_heads=1):
        super(Model, self).__init__()

        self.input_dim_kq = input_dim_kq
        self.embed_dim_kq = embed_dim_kq
        self.input_dim_v = input_dim_v
        self.embed_dim_v = embed_dim_v
        self.output_dim = output_dim

        self.n_heads = n_heads
        self.transformer_layer = TransformerLayerB(
            input_dim_kq, 
            embed_dim_kq,
            input_dim_v,
            embed_dim_v,
            n_heads)
        self.output_layer = nn.Linear(embed_dim_v, output_dim)

    def forward(self, kq, v):
        v = self.transformer_layer(kq, v)
        return self.output_layer(v)

    def predict(self, kq, v):
        return (self.forward(kq, v) > 0).to(int)


###### LIGHTNING MODULE
# Integrate lightning
class LightningModel(L.LightningModule):
    def __init__(self, model, loss_fn, optimizer_class, lr):
        super().__init__()
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_class = optimizer_class
        self.lr = lr

    def forward(self, kq, v):
        kq = make_ohe(kq)
        return self.model(kq, v)

    def training_step(self, batch, batch_idx):
        kq, v, targets = batch
        preds = self(kq, v)
        loss = self.loss_fn(preds, targets)
        loss = self.loss_fn(preds, targets)
        acc = (torch.sign(preds) == torch.sign(targets)).to(torch.float).mean().item()
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return loss    
    
    def validation_step(self, batch, batch_idx):
        kq, v, targets = batch
        preds = self(kq, v)
        loss = self.loss_fn(preds, targets)
        acc = (torch.sign(preds) == torch.sign(targets)).to(torch.float).mean().item()
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self.parameters(), 
            lr=self.lr
        )
        return {"optimizer" : optimizer}




if __name__ == '__main__':
    print(" * Running...")
    
    # Set configuration for PyTorch model
    input_dim_kq = 64
    embed_dim_kq = 64
    input_dim_v = 1
    embed_dim_v = 8
    output_dim = 1
    n_heads = 4
    
    # Set configuration for training model in lightning
    loss_fn = torch.nn.MSELoss()
    optimizer_class = optim.Adam
    lr = 0.001
    
    #### INSTANTIATIONS
    # Instantiate PyTorch model (with summary)
    model = Model(input_dim_kq, embed_dim_kq, input_dim_v, embed_dim_v, output_dim, 4)
    summary(model, [(1230, input_dim_kq), (1230, input_dim_v)], batch_size=1)
    
    # Instantiate lightning module, logger, trainer
    l_model = LightningModel(model, loss_fn, optimizer_class, lr)
    logger = pl_loggers.CSVLogger("../models/", name="model_0",version="version_0")
    trainer = L.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        logger=logger
    )
    
    # Set configuration for SequenceDataset
    kq_cols = ['NEW_TEAM_ID_for', 'NEW_TEAM_ID_ag']
    v_cols = ['PLUS_MINUS']
    t_cols = ['PLUS_MINUS']
    cols_to_select=[kq_cols, v_cols, t_cols]
    
    # Instantiate trainset + valset, trainloader + valloader
    trainset = SequenceDataset(
        db_path='../data/cleaned/my_database.db', 
        table_name='my_table',
        season_col='SEASON_ID',
        date_col='GAME_DATE',
        start_season=21983,
        end_season=22022,
        cols_to_select=cols_to_select
        )

    valset = SequenceDataset(
        db_path='../data/cleaned/my_database.db',
        table_name='my_table',
        season_col='SEASON_ID',
        date_col='GAME_DATE',
        start_season=22023,
        end_season=22023,
        cols_to_select=cols_to_select
    )
    
    trainloader = DataLoader(trainset, batch_size=2, collate_fn=collate_fn)
    valloader = DataLoader(valset, batch_size=2, collate_fn=collate_fn)
    
    
    # FIT THE MODEL
    trainer.fit(l_model, train_dataloaders=trainloader, val_dataloaders=valloader)
    
    # Check the logs
    log_path = "../models/model_0/version_0/metrics.csv"
    df = pd.read_csv(log_path)
    
    # Cleaning up dataframe
    # Fill NaN values in train columns with corresponding validation rows and vice versa
    df = df.groupby('epoch', as_index=False).apply(lambda x: x.ffill().bfill())

    # Drop duplicate rows (they might exist after filling)
    df = df.drop_duplicates(subset='epoch')

    # Drop 'step' column
    df.drop(['step'], axis=1, inplace=True)


    # Reset the index after cleaning
    df = df.reset_index(drop=True)

    print(" - - - Cleaned - - - ")
    print(df.values.shape)
    print(df)
    
    
    # Trace + save the model
    scripted_model = torch.jit.script(model)
    model_path = "../models/model_0/version_0/scripted_model.pt"
    scripted_model.save(model_path)