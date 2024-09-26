import sys
import os
import importlib
import argparse

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import yaml
import torch
from torch.nn import (
    MSELoss
)
from torch.optim import (
    Adam
)
from lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

# Local imports
from src.dataloading import (
    SeasonSequenceDataset,
    collate_fn,
    DataLoader # Base PyTorch DataLoader
)

from src.training import (
    LightningModel
)

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', required=True, type=int, help="Number of epochs to train model")
    args = parser.parse_args()
    epochs = args.epochs
    
    # Read in configuration
    with open('configs/old_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    CLEAN_DIR = config['clean_dir']
    DB_NAME = config['db_name']
    DB_PATH = os.path.join(CLEAN_DIR, DB_NAME)
    MODEL_SRC_DIR = config['model_src_dir']
    sys.path.append(MODEL_SRC_DIR)
    MODEL_SAVE_DIR = config['model_save_dir']
    
    # ---- BEGIN SETUP ---- 
    setup = config['setups']['A']
    
    # (A) SET UP DATALOADING
    dataloading_config = setup['dataloading_config']
    ssd_blueprint = dataloading_config['ssd_blueprint']
    train_seasons = dataloading_config['train_seasons']
    val_seasons = dataloading_config['val_seasons']
    batch_size = dataloading_config['batch_size']
    
    # Initialize train_ssd, val_ssd
    train_ssd  = SeasonSequenceDataset(
        db_path=DB_PATH,
        blueprint=ssd_blueprint,
        seasons=train_seasons
    )
    val_ssd = SeasonSequenceDataset(
        db_path=DB_PATH,
        blueprint=ssd_blueprint,
        seasons=val_seasons
    )
    
    # Initialize train_loader, val_loader
    train_loader = DataLoader(
        dataset=train_ssd, 
        collate_fn=collate_fn, 
        batch_size=batch_size
    )
    val_loader = DataLoader(
        dataset=val_ssd,
        collate_fn=collate_fn,
        batch_size=batch_size
    )
    
    # (B) SET UP PYTORCH MODEL
    model_config = setup['model_config']
    model_name = model_config['name']
    model_file = importlib.import_module(model_name)
    hyperparams = model_config['hyperparams']

    # Initialize model
    model = model_file.Model(**hyperparams)
    
    # (C) SET UP LIGHTNING
    training_config = setup['training_config']
    loss_fn_class = MSELoss
    optimizer_class = Adam
    lr = training_config['lr']
    #epochs = training_config['epochs']
    
    # Initialize LightningModule
    lightning_model = LightningModel(
        model=model,
        loss_fn_class=loss_fn_class,
        optimizer_class=optimizer_class,
        lr=lr
    )
    
    model_save_name = model_name + '.ckpt'
    model_save_path = os.path.join(MODEL_SAVE_DIR, model_save_name)
    
    # Initialize logger
    logger = CSVLogger("logs/", name="base_model")
    
    # Initialize trainer
    trainer = Trainer(max_epochs=epochs, logger=logger)
    
    # TRAINING
    print(f"Training for {epochs} epochs")
    trainer.fit(
        lightning_model, 
        train_loader,
        val_loader
    )
    print(f"Done training")
    model_save_name = model_name + '.ckpt'
    model_save_path = os.path.join(MODEL_SAVE_DIR, model_save_name)
    trainer.save_checkpoint(model_save_path)