from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import Lasso

model_class = Lasso
model_hyperparam_space = {
    "random_state" : 42,
    "alpha": ("float", {"low" : 1e-4, "high" : 1e-2, "log": True}),
    "tol" : 1e-3,
    "max_iter" : 2000
}
objective_fn = root_mean_squared_error
val_seasons = [2020, 2021, 2022, 2023, 2024]

n_train_seasons_space = ("int", {"low" : 1, "high" : 20}) # Number of training seasons to use
w_norm_data = True

# NOTE: hyperparam space will be dict[str, tuple],
# where the tuple is (type of hyperparam, key_word_args)

# Names
model_name = "lasso"
model_extension = "joblib"