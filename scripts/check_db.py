import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import yaml
import sqlite3

# Internal imports
from src.utils import check_db

if __name__ == '__main__':
    # Read configuration
    with open('configs/old_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    CLEAN_DIR = config['clean_dir']
    DB_NAME = config['db_name']
    DB_PATH = os.path.join(CLEAN_DIR, DB_NAME)
    
    print("--- CHECKING DATABASE ---")
    check_db(DB_PATH)