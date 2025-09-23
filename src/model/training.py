# External imports
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader

# Internal imports
from src.model.dataloading import Gameset

def train_sklearn(
    model_class,
    model_hyperparams,
    training_data,
    features, 
    target
):
     
    assert len(training_data) > 0, f"No training data."
        
    # Get features/target
    X_tr = training_data[features]
    y_tr = training_data[target]
    
    # Instantiate model
    model = model_class(**model_hyperparams)
        
    # Fit the model on the training data
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
    
):  
    # Initialize DataLoader
    trainset = Gameset(training_data, features, [target])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = model_class(
        input_dim=trainset.get_input_dim(),
        output_dim=trainset.get_output_dim(),      
        **model_hyperparams
    )
    
    # Initialize optimizer
    optimizer = optimizer_class(params=model.parameters(), **optimizer_hyperparams)

    # Iteratively train model!
    for epoch in range(n_epochs):
        
        # Iterate over batch
        for batch_idx, (X, y) in enumerate(tqdm(trainloader, desc=f" -- Epoch {epoch + 1}/{n_epochs} -- ")):
            # Move tensors to proper device
            pass
        
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