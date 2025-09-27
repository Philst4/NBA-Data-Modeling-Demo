from sklearn.metrics import root_mean_squared_error
import numpy as np

class Baseline0():
    """
    Predicts '0'.
    """
    
    def __init__(self):
        pass

    def fit(self, X, y):
        pass
    
    def predict(self, X):
        return np.zeros(X.shape[0])

model_class = Baseline0
model_hyperparam_space = {}

objective_fn = root_mean_squared_error
val_seasons = [2020, 2021, 2022, 2023, 2024]

n_train_seasons_space = ("int", {"low" : 1, "high" : 1}) # Number of training seasons to use

# NOTE: hyperparam space will be dict[str, tuple],
# where the tuple is (type of hyperparam, key_word_args)

# Names
model_name = "baseline_0"
study_name = "baseline_0_study"
model_filename = "baseline_0_model.joblib"