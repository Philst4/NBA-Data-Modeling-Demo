from sklearn.metrics import root_mean_squared_error
import torch
import torch.nn as nn

#### MODEL ####
class FFNN(nn.Module):
    """
    Regressive FFNN Model.
    """
    
    def __init__(
        self, 
        input_dim, 
        hidden_dim, 
        output_dim, 
        n_hidden_layers=1, 
        use_residual=True
    ):
        super(FFNN, self).__init__()

        self.use_residual = use_residual

        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_hidden_layers):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        X = self.input_layer(X)

        for layer in self.hidden_layers:
            residual = X
            X = layer(X)
            if self.use_residual:
                X = X + residual  # Residual connection

        # Round predictions away from 0?
        return self.output_layer(X)
    
    def fit(self, X, y):
        """
        Fits the model to the data.
        
        Meant to mimic sklearn's interface.
        """
        
        # Initialize dataset with data.
        
        pass

    def predict(self, X):
        """
        Gives the model sklearn-like predict functionality.
        
        Just makes (regressive) predictions and returns them as an np.ndarray.
        """
        return self.forward(X).numpy()
    
model_class = FFNN
model_hyperparam_space = {
    "hidden_dim" : 256, #("int", {"low" : 256, "high" : 256}),
    "n_hidden_layers" : 2, #("int", {"low" : 2, "high" : 2}),
}

# Need objective function to accept torch tensors
def rmse_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))
objective_fn = rmse_torch

val_seasons = [2020, 2021, 2022, 2023, 2024]

n_train_seasons_space = ("int", {"low" : 1, "high" : 10})
batch_size = 1024
optimizer_class = torch.optim.Adam
optimizer_hyperparam_space = {
    "lr" : ("float", {"low" : 1e-4, "high" : 1e-2, "log" : True})
}
n_epochs_space = ("int", {"low" : 1, "high" : 3})

# Names
model_name = "torch_ffnn"
study_name = "torch_ffnn_study"
model_filename = "torch_ffnn_weights.pth"