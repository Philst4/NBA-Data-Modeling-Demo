from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import Lasso

model_class = Lasso
hyperparam_space = {
    "alpha": ("float", {"low" : 1e-4, "high" : 10, "log": True}),
}
objective_fn = root_mean_squared_error
val_seasons = [2020, 2021, 2022, 2023, 2024]
study_name = "lasso_study"

n_train_seasons_suggestion = ("int", {"low" : 1, "high" : 50}) # Number of training seasons to use

# NOTE: hyperparam space will be dict[str, tuple],
# where the tuple is (type of hyperparam, key_word_args)