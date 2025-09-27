from sklearn.metrics import root_mean_squared_error
import numpy as np

class Baseline1():
    """
    Predicts '1' if home, '-1' if away.
    """
    
    def __init__(self, home_feature='IS_HOME_for', C=1.):
        self.home_feature = home_feature
        self.C = C

    def fit(self, X, y):
        pass
    
    def predict(self, X):
        """
        Returns a numpy array of 1 (home) or -1 (away).
        """
        if self.home_feature not in X.columns:
            raise ValueError(f"Column '{self.home_feature}' not found in X")
        
        col = X[self.home_feature]
        # Ensure it works whether column is boolean, 0/1, or string flags
        # True/1/'home' => 1 ; False/0/other => -1
        return np.where(col > 0, self.C, -self.C)

model_class = Baseline1
model_hyperparam_space = {"C": ("float", {"low" : 0.0, "high" : 2.0}),}

objective_fn = root_mean_squared_error
val_seasons = [2020, 2021, 2022, 2023, 2024]

n_train_seasons_space = ("int", {"low" : 1, "high" : 1}) # Number of training seasons to use

# NOTE: hyperparam space will be dict[str, tuple],
# where the tuple is (type of hyperparam, key_word_args)

# Names
model_name = "baseline_1"
study_name = "baseline_1_study"
model_filename = "baseline_1_model.joblib"