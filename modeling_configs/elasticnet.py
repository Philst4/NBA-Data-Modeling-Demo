from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import ElasticNet

model_class = ElasticNet

model_hyperparam_space = {
    "random_state": 42,
    # Overall regularization strength
    "alpha": ("float", {"low": 1e-4, "high": 1e-2, "log": True}),
    # Mix between L1 and L2
    # 1.0 = Lasso, 0.0 = Ridge
    "l1_ratio": ("float", {"low": 0.0, "high": 1.0}),
    "tol" : 1e-3,
    "max_iter" : 2000,
}

objective_fn = root_mean_squared_error
val_seasons = [2020, 2021, 2022, 2023, 2024]

n_train_seasons_space = ("int", {"low": 1, "high": 20})
w_norm_data = True

# Names
model_name = "elasticnet"
model_extension = "joblib"
