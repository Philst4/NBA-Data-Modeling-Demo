import sys
from pathlib import Path
import importlib

# For loading in modeling configs
# Required variables that must exist in the config
REQUIRED_CONFIG_VARS = [
    "model_class",
    "model_hyperparam_space",
    "objective_fn",
    "val_seasons",
    "model_name"
]

TORCH_CONFIG_VARS = [
    "batch_size",
    "optimizer_class",
    "optimizer_hyperparam_space",
    "n_epochs_space",
]

def load_modeling_config(config_path):
    """Load a Python config file as a module."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found.")

    spec = importlib.util.spec_from_file_location("config", str(config_path))
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)

    # Validate required variables
    missing_vars = [var for var in REQUIRED_CONFIG_VARS if not hasattr(config, var)]
    if missing_vars:
        raise ValueError(f"Config file is missing required variables: {missing_vars}")

    # Add torch config variables as 'None' if missing
    for var in TORCH_CONFIG_VARS:
        if not hasattr(config, var):
            setattr(config, var, None) 

    return config


import importlib
from pathlib import Path

def load_modeling_config(modeling_config_path: str):
    """
    Load a modeling config as a proper Python module.

    Parameters
    ----------
    modeling_config_path : str
        e.g.
        - "modeling_configs/baseline_1.py"
        - "src/modeling_configs/baseline_1.py"
        - "modeling_configs/baseline_1"
    """
    path = Path(modeling_config_path)

    # Drop extension if present
    path = path.with_suffix("")

    # Build module path from path parts
    module_path = ".".join(path.parts)

    try:
        config = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Could not import modeling config '{module_path}'. "
            f"Ensure the project root is on PYTHONPATH and "
            f"'{path.parent}' is a Python package."
        ) from e

    # Validate required variables
    missing_vars = [v for v in REQUIRED_CONFIG_VARS if not hasattr(config, v)]
    if missing_vars:
        raise ValueError(
            f"Modeling config '{module_path}' is missing required variables: {missing_vars}"
        )

    # Add torch config variables as None if missing
    for v in TORCH_CONFIG_VARS:
        if not hasattr(config, v):
            setattr(config, v, None)

    return config
