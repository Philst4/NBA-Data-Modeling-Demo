#### STD IMPORTS
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

#### EXTERNAL IMPORTS
import torch
from torch.utils.data import DataLoader
from torch import optim
from torchsummary import summary
import lightning as L
from pytorch_lightning import loggers as pl_loggers

#### LOCAL IMPORTS
from config import (
    CLEAN_DATA_DIR,
    MODEL_DIR
)
from utils.data_loading import (
    SeasonSequenceDataset
)
from architectures.model_0 import (
    collate_fn,
    Model
)

from utils.modeling import (
    LightningModel
)
        
#### MAIN PROGRAM
if __name__ == '__main__':
    print(" * Running...")
    
    #### DATALOADING SETUP
    # Set configuration for SequenceDataset
    meta_cols = ['SEASON_ID', 'GAME_DATE', 'MATCHUP']
    kq_cols = ['NEW_TEAM_ID_for', 'NEW_TEAM_ID_ag']
    v_cols = ['PLUS_MINUS']
    t_cols = ['PLUS_MINUS']
    data_cols = kq_cols + v_cols + t_cols
    
    # Define path, table name
    db_path = '/'.join((CLEAN_DATA_DIR, 'my_database.db'))
    table_name = 'my_table'

    train_ssd = SeasonSequenceDataset(
        db_path=db_path,
        table_name=table_name,
        data_cols=data_cols,
        meta_cols=meta_cols,
        start_season=21983,
        end_season=22022
    )
    
    val_ssd = SeasonSequenceDataset(
        db_path=db_path,
        table_name=table_name,
        data_cols=data_cols,
        meta_cols=meta_cols,
        start_season=22022,
        end_season=22023
    )
    
    shuffling_collate_fn = lambda x : collate_fn(x, shuffle=True)
    trainloader = DataLoader(
        train_ssd, 
        batch_size=5, 
        collate_fn=shuffling_collate_fn, 
        shuffle=True
    ) 
    
    valloader = DataLoader(
        val_ssd, 
        batch_size=1, 
        collate_fn=shuffling_collate_fn
    )
    
    #### PYTORCH MODEL SETUP
    # Set configuration for PyTorch model
    input_dim_kq = 64
    input_dim_v = 1
    embed_dim_kq = 128 # make 256+
    embed_dim_v = 128 # make 256+
    output_dim = 1
    n_heads = 4
    
    # Set configuration for training model in lightning
    loss_fn_class = torch.nn.MSELoss
    optimizer_class = optim.Adam
    lr = 0.0025
    
    # Instantiate PyTorch model (with summary)
    model = Model(input_dim_kq, input_dim_v, embed_dim_kq, embed_dim_v, output_dim, 4)
    summary(model, [(1230, input_dim_kq), (1230, input_dim_v)], batch_size=1)
    
    #### LIGHTNING SETUP
    # Instantiate lightning module, logger, trainer
    l_model = LightningModel(
        model, 
        loss_fn_class, 
        optimizer_class, 
        lr,
        debug=False,
        normalize=True
    )
    logger = pl_loggers.CSVLogger(MODEL_DIR, name="model_0", version="version_0")
    trainer = L.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        logger=logger
    )
    
    #### FIT THE MODEL
    trainer.fit(l_model, train_dataloaders=trainloader, val_dataloaders=valloader)
    
    #### TRACE + SAVE MODEL
    scripted_model = torch.jit.script(model)
    model_path = '/'.join((MODEL_DIR, 'model_0', 'version_0', 'scripted_model.pt'))
    scripted_model.save(model_path)
    print(f'Model saved to {model_path}')