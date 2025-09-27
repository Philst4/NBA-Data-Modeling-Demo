from joblib import load
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Internal imports
from src.model.dataloading import Gameset


def predict_sklearn(
    model,
    data,
    features
):  
    return model.predict(data[features])

def predict_torch(
    model,
    data,
    features,
    batch_size=-1,
    w_unique_ids=False,
):
    
    # Initialize DataLoader
    gameset = Gameset(data, features, [])
    if batch_size == -1:
        batch_size = 32
    gameloader = DataLoader(gameset, batch_size=batch_size, shuffle=False)
    
    # Initialize list to hold all predictions
    all_y_preds = []
    
    with torch.no_grad():
        # Iterate over batch
        for batch_idx, (X, _) in enumerate(gameloader):
            
            # Move tensors to proper device
            pass
        
            # Forward pass
            y_preds = model(X)
            
            # Add to list of results
            all_y_preds.append(y_preds)
    
    # Convert to numpy
    all_y_preds = torch.cat(all_y_preds, dim=0)
    
    # Return
    return all_y_preds