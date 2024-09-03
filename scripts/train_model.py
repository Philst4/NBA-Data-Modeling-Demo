import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)


# External imports 
# TODO: make it so this is handled in another file.
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import optim
from torchsummary import summary
import lightning as L
from pytorch_lightning import loggers as pl_loggers


# Local imports
from config import (
    CLEAN_DATA_DIR,
    MODEL_DIR
)


# Imports from modelign should be:
# * init_dataloader
# * init_model
# NOT the classes.

from utils.data_loading import (
    SequenceDataset,
    collate_fn
)


from utils.modeling import (
    Model,
    LightningModel,
)


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
    logger = pl_loggers.CSVLogger(MODEL_DIR, name="model_0",version="version_0")
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
    db_path = '/'.join((CLEAN_DATA_DIR, 'my_database.db'))
    trainset = SequenceDataset(
        db_path=db_path,
        table_name='my_table',
        season_col='SEASON_ID',
        date_col='GAME_DATE',
        start_season=21983,
        end_season=22022,
        cols_to_select=cols_to_select
        )

    valset = SequenceDataset(
        db_path=db_path,
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
    log_path = '/'.join((MODEL_DIR, 'model_0', 'version_0', 'metrics.csv'))
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
    model_path = '/'.join((MODEL_DIR, 'model_0', 'version_0', 'scripted_model.pt'))
    scripted_model.save(model_path)