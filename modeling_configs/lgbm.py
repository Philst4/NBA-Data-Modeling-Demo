from sklearn.metrics import root_mean_squared_error
from lightgbm import LGBMRegressor
import os

model_class = LGBMRegressor
hyperparam_space = {
    "n_jobs": os.cpu_count() // 2,
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "boosting_type": "gbdt",  # "gbdt", "dart", "goss"
    "n_estimators": ("int", {"low" : 500, "high" : 2000}),
    "learning_rate": ("float", {"low" : 1e-3, "high" : 0.2}),

    # Tree complexity
    "num_leaves": ("int", {"low" : 31, "high" : 512}),
    "max_depth": ("int", {"low" : -1, "high" : 16}),  # -1 = no limit
    "min_child_samples": ("int", {"low" : 5, "high" : 100}),
    "min_child_weight": ("float", {"low" : 1e-3, "high" : 10, "log": True}),

    # Regularization
    "lambda_l1": ("float", {"low" : 1e-3, "high" : 10, "log": True}),
    "lambda_l2": ("float", {"low" : 1e-3, "high" : 10, "log": True}),

    # Feature & data subsampling
    "feature_fraction": ("float", {"low" : 0.5, "high" : 1.0}),
    "bagging_fraction": ("float", {"low" : 0.5, "high" : 1.0}),
    "bagging_freq": ("int", {"low" : 1, "high" : 10}),
}

objective_fn = root_mean_squared_error
val_seasons = [2020, 2021, 2022, 2023, 2024]
study_name = "lgbm_study"

n_train_seasons_suggestion = ("int", {"low" : 1, "high" : 50}) # Number of training seasons to use