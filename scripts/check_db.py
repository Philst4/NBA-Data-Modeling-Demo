import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import yaml
import sqlite3

if __name__ == '__main__':
    print("--- CHECKING DATABASE ---")
    
    # Read configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    CLEAN_DIR = config['clean_dir']
    DB_NAME = config['db_name']
    DB_PATH = os.path.join(CLEAN_DIR, DB_NAME)
    
    assert(os.path.exists(DB_PATH)), f"Database not found at '{DB_PATH}'"
    
    # Connect to your SQLite database
    conn = sqlite3.connect(DB_PATH)

    # Create a cursor object to execute SQL queries
    cursor = conn.cursor()

    # Query to get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

    # Fetch and print table names
    tables = cursor.fetchall()

    for table_name in tables:
        print(f"\nTABLE: {table_name[0]}")
        
        cursor.execute(f"SELECT COUNT(*) FROM {table_name[0]}")
        row_count = cursor.fetchone()[0]
        
        # Query to get column names and types
        cursor.execute(f"PRAGMA table_info({table_name[0]});")
        columns = cursor.fetchall()

        print(f" * {row_count} rows, {len(columns)} columns")
        
        # Display columns information
        for col in columns:
            print(f" - - {col[1]} ({col[2]})")
    
    # Print number of tables 
    tables_str = ', '.join([name for (name,) in tables])
    print(f"\n{len(tables)} tables in database:", tables_str)
    
    # Get size of database
    cursor.execute("PRAGMA page_count;")
    page_count = cursor.fetchone()[0]

    cursor.execute("PRAGMA page_size;")
    page_size = cursor.fetchone()[0]

    db_size = page_count * page_size  # in bytes
    print(f"Size of database: {db_size} bytes => ~{db_size/1e9:.3f} GB")
