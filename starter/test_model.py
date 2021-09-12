import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

n_estimators = [100, 300, 500,1000]
max_depth = [5, 8, 15, 25]
min_samples_split = [2, 5]
min_samples_leaf = [1, 5] 
hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)
def test_train():
    """
    test if model scores
    """
    loaded_model = pickle.load(open(os.path.abspath(os.getcwd())+'/'+os.path.join('model','model.pkl'), 'rb'))
    loaded_encoder = pickle.load(open(os.path.abspath(os.getcwd())+'/'+os.path.join('model','encoder.pkl'), 'rb'))
    loaded_labeler = pickle.load(open(os.path.abspath(os.getcwd())+'/'+os.path.join('model','labeler.pkl'), 'rb'))

    
    gs_clf=GridSearchCV(RandomForestClassifier(),param_grid=hyperF)
    assert type(loaded_model) == type(gs_clf)
    assert type(loaded_encoder) == type(OneHotEncoder(sparse=False, handle_unknown="ignore"))
    assert type(loaded_labeler) == type(LabelBinarizer())        
        
    
def test_loaded_file_types():
    """
    test if model scores
    """
    loaded_model = pickle.load(open(os.path.abspath(os.getcwd())+'/'+os.path.join('model','model.pkl'), 'rb'))
    loaded_encoder = pickle.load(open(os.path.abspath(os.getcwd())+'/'+os.path.join('model','encoder.pkl'), 'rb'))
    loaded_labeler = pickle.load(open(os.path.abspath(os.getcwd())+'/'+os.path.join('model','labeler.pkl'), 'rb'))
    
    gs_clf=GridSearchCV(RandomForestClassifier(),param_grid=hyperF)
    assert type(loaded_model) == type(gs_clf)
    assert type(loaded_encoder) == type(OneHotEncoder(sparse=False, handle_unknown="ignore"))
    assert type(loaded_labeler) == type(LabelBinarizer())
    
def test_slice_inference():
    """
    test if model scores
    """
    loaded_model = pickle.load(open(os.path.abspath(os.getcwd())+'/'+os.path.join('model','model.pkl'), 'rb'))
    loaded_encoder = pickle.load(open(os.path.abspath(os.getcwd())+'/'+os.path.join('model','encoder.pkl'), 'rb'))
    loaded_labeler = pickle.load(open(os.path.abspath(os.getcwd())+'/'+os.path.join('model','labeler.pkl'), 'rb'))

    loaded_model = pickle.load(open(os.path.abspath(os.getcwd())+'/'+os.path.join('model','model.pkl'), 'rb'))
    loaded_encoder = pickle.load(open(os.path.abspath(os.getcwd())+'/'+os.path.join('model','encoder.pkl'), 'rb'))
    loaded_labeler = pickle.load(open(os.path.abspath(os.getcwd())+'/'+os.path.join('model','labeler.pkl'), 'rb'))
    
    gs_clf=GridSearchCV(RandomForestClassifier(),param_grid=hyperF)
    assert type(loaded_model) == type(gs_clf)
    assert type(loaded_encoder) == type(OneHotEncoder(sparse=False, handle_unknown="ignore"))
    assert type(loaded_labeler) == type(LabelBinarizer())
    