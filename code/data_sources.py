#---- Data Sources Dictionary----#
# Providing sources for all datasets used in the project

import pandas as pd
import os

def load_data_sources():
    cwd = os.getcwd()
    data_path = "/ML/Data/data_sources.csv"
    file_path = str(cwd + data_path)
    data_source_df = pd.read_csv(file_path)
    
    return data_source_df
