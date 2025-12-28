# External imports
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from lightgbm import LGBMRegressor
import lightgbm as lgb

# Internal imports
from src.model.dataloading import Gameset

def train_sklearn(
    model_class,
    model_hyperparams,
    training_data,
    features, 
    target,
    device=None,
    val_data=None
):
     
    assert len(training_data) > 0, f"No training data."
        
    # Get features/target
    X_tr = training_data[features]
    y_tr = training_data[target]
    
    # Add 'gpu' as hyperparam if device is cuda (for LGBM)
    if device.type == "cuda" and issubclass(model_class, LGBMRegressor):
        model_hyperparams['device'] = "gpu"
    
    # Instantiate model
    model = model_class(**model_hyperparams)
        
    # Fit the model on the training data
    if model_class == LGBMRegressor and not val_data is None:
        X_val = val_data[features]
        y_val = val_data[target]
        # Use early stopping for LGBM
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=50,
                    verbose=True
                )
            ]
        )
    else:
        model.fit(X_tr, y_tr)
    
    # Return model
    return model

def train_torch(
    model_class,
    model_hyperparams,
    training_data,
    features, 
    target,
    batch_size, # For Initializing DataLoader 
    optimizer_class, 
    optimizer_hyperparams,
    objective_fn, # For calculating loss/stepping w/ optimizer
    n_epochs, # How many passes over entire dataset to give model!
    device=None
    
):  
    # Initialize DataLoader
    trainset = Gameset(training_data, features, [target])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=device.type != "cpu")
    
    # Initialize model
    model = model_class(
        input_dim=trainset.get_input_dim(),
        output_dim=trainset.get_output_dim(),      
        **model_hyperparams
    )
    
    if device is not None:
        model = model.to(device)
    
    # Initialize optimizer
    optimizer = optimizer_class(params=model.parameters(), **optimizer_hyperparams)

    # Iteratively train model!
    for epoch in range(n_epochs):
        
        # Iterate over batch
        for batch_idx, (X, y) in enumerate(tqdm(trainloader, desc=f" -- Epoch {epoch + 1}/{n_epochs} -- ")):
            # Move tensors to proper device
            if device is not None:
                X = X.to(device)
                y = y.to(device)
        
            # Forward pass
            y_preds = model(X)
            loss = objective_fn(y_preds, y) # AKA loss_fn
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Print progress?
            pass

    # Return trained model instance
    return model