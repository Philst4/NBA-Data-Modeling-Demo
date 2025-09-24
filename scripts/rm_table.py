# Standard library imports
import sys
import os
import argparse

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import yaml
import sqlite3



def main(args):
    # (0) Read in configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    CLEAN_DIR = config['clean_data_dir']
    DB_NAME = config['db_name']
    DB_PATH = os.path.join(CLEAN_DIR, DB_NAME)
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Drop the table named "table_name"
    table_name = args.table_name
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
        (table_name,)
    )
    row = cursor.fetchone()

    if row is None:
        print(f"Table '{table_name}' does not exist in '{DB_PATH}'. Nothing to drop.")
    else:
        cursor.execute(f"DROP TABLE {table_name};")
        conn.commit()
        print(f"Table '{table_name}' dropped from '{DB_PATH}'.")

        conn.commit()
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('table_name', type=str, help="Name of table to remove from clean database.")
    args = parser.parse_args()
    main(args)