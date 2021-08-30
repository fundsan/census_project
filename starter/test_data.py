import pandas as pd
import pickle
import os
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
def test_loading_model():
    """
    test if model exist and can be loaded
    """
    # load the model from disk
    try:
        open(os.path.join('model','model.pkl'), 'rb')
        
    except FileNotFoundError as err:
        
        raise err
    try:
        loaded_model = pickle.load(open(os.path.join('model','model.pkl'), 'rb'))
    except EOFError as err:
        raise err
        
        
    
def test_score():
    """
    test if model scores
    """
    loaded_model = pickle.load(open(os.path.join('model','model.pkl'), 'rb'))
    
    assert type(loaded_model) == type(GridSearchCV(SGDClassifier(),param_grid={'max_iter':(500,1000)}))