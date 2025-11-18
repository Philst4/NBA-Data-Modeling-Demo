from app import create_app
import os

CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/config.yaml")
app = create_app(CONFIG_PATH)
