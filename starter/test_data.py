import pandas as pd
import pickle
import os
  
def test_clean_data():
    """
    test if model exist and can be loaded
    """
    # load the model from disk
    try:
        data=pd.read_csv(os.path.abspath(os.getcwd()+'/'+os.path.join('data','census_cleaned.csv')))
        
    except FileNotFoundError as err:
        
        raise err
    assert ' ' not in data.iloc[0]['occupation']          
        
