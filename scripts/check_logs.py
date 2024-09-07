import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

#### EXTERNAL IMPORTS
import pandas as pd
import matplotlib.pyplot as plt
 
#### LOCAL IMPORTS
from config import (
    MODEL_DIR
)
  
if __name__ == '__main__':
    # Check the logs
    log_path = '/'.join((MODEL_DIR, 'model_0', 'version_0', 'metrics.csv'))
    df = pd.read_csv(log_path)
    
    # Cleaning up dataframe
    # Fill NaN values in train columns with corresponding validation rows and vice versa
    df = df.groupby('epoch', as_index=False).apply(lambda x: x.ffill().bfill())

    # Drop duplicate rows (they might exist after filling)
    df = df.drop_duplicates(subset='epoch')

    # Drop 'step' column
    df.drop(['step'], axis=1, inplace=True)

    # Reset the index after cleaning
    df = df.reset_index(drop=True)

    print(" - - - Cleaned - - - ")
    print(df.values.shape)
    
    for column in df.columns[1:]:
        plt.figure()  # Create a new figure for each plot
        plt.plot(df.index, df[column], label=column)
        plt.xlabel('Epoch')
        plt.ylabel(f'{column}')
        plt.title(f'{column} vs Epoch')
        plt.show()
    
    print(df)
    