# app_factory.py
import os
import yaml
from flask import Flask

from src.data.io import get_modeling_data
from datetime import date
from .prediction.loader import load_models


def create_app(args):
    app = Flask(__name__)
    app.secret_key = os.urandom(24)

    # --- Load config ---
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    CLEAN_DATA_DIR = config['clean_data_dir']
    DB_NAME = config['db_name']
    DB_PATH = os.path.join(CLEAN_DATA_DIR, DB_NAME)
    MODELING_CONFIG_DIR = config['modeling_config_dir']
    MODEL_STORAGE = config['model_storage']
    OPTUNA_STORAGE = config['optuna_storage']

    # --- Load data + models (heavy work) ---
    MIN_DATE = date(1983, 1, 1)
    MAX_DATE = date.today()
    DEFAULT_DATE = date(2025, 4, 13)

    _, features, _ = get_modeling_data(DB_PATH, config, DEFAULT_DATE)
    INPUT_DIM = len(features)

    # --------------------
    #   Load all models
    # --------------------
    MODELS = load_models(
        MODELING_CONFIG_DIR,
        OPTUNA_STORAGE,
        MODEL_STORAGE,
        INPUT_DIM,
        config,
    )

    # Make these accessible inside route functions:
    app.config["MODELS"] = MODELS
    app.config["DB_PATH"] = DB_PATH
    app.config["DEFAULT_DATE"] = DEFAULT_DATE
    app.config["MIN_DATE"] = MIN_DATE
    app.config["MAX_DATE"] = MAX_DATE
    app.config["CONFIG"] = config
    app.config["CACHE"] = {}

    # Import the routes AFTER initialization
    from .routes import register_routes
    register_routes(app)

    return app
