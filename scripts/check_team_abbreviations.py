import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports

# Internal imports
from src.utils import (
    query_db
)


if __name__ == "__main__":
    # Command line arguments (revisit)
    season_id = "22023"

    # Extract from config (revisit)
    db_path = os.path.join("data", "clean", "my_database.db")
    query = f"""SELECT TEAM_ABBREVIATION 
    FROM team_metadata 
    WHERE SEASON_ID = {season_id}
    """
    
    team_abbreviations = query_db(
        db_path=db_path,
        query=query
    )
    
    print(team_abbreviations['TEAM_ABBREVIATION'].sort_values(ignore_index=True))