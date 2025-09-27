import optuna

# Load in study
def extract_best_model_hyperparams_from_study(modeling_config, optuna_storage):    
    
    # Extract model class from modeling config
    model_class = modeling_config.model_class
        
    # Extract best hyperparams from study
    study_name = modeling_config.study_name
    study = optuna.load_study(
        storage=optuna_storage,
        study_name=study_name
    )
    
    print(f"Best hyperparams of '{model_class}' from '{study_name}':")
    print(study.best_trial)
    all_hyperparams = study.best_trial.params
        
    # Extract best model hyperperams from study
    model_hyperparams = {
        key: all_hyperparams[key] if key in all_hyperparams else val
        for key, val in modeling_config.model_hyperparam_space.items()
    }
    
    return model_hyperparams