import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from starter.ml.data import process_data
from starter.ml.model import slice_inference
from sklearn.model_selection import train_test_split
from starter.train_model import just_do_tst

n_estimators = [100, 300, 500,1000]
max_depth = [5, 8, 15, 25]
min_samples_split = [2, 5]
min_samples_leaf = [1, 5] 
hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)
def test_model_files():
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
        
    
def test_testing():
    """
    test if model scores
    """
    loaded_model = pickle.load(open(os.path.abspath(os.getcwd())+'/'+os.path.join('model','model.pkl'), 'rb'))
    loaded_encoder = pickle.load(open(os.path.abspath(os.getcwd())+'/'+os.path.join('model','encoder.pkl'), 'rb'))
    loaded_labeler = pickle.load(open(os.path.abspath(os.getcwd())+'/'+os.path.join('model','labeler.pkl'), 'rb'))
    precision, recall, fbeta = just_do_tst()
    assert type(precision) == np.float64
    assert type(recall) == np.float64
    assert type(fbeta) == np.float64
    
    
    
def test_slice_inference():
    """
    test if model scores
    """    
    _, test = train_test_split(pd.read_csv(os.path.abspath(os.getcwd()+'/'+os.path.join('data','census_cleaned.csv'))), test_size=0.20)
    
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    loaded_model = pickle.load(open(os.path.join('model','model.pkl'), 'rb'))
    loaded_encoder = pickle.load(open(os.path.join('model','encoder.pkl'), 'rb'))
    loaded_lb = pickle.load(open(os.path.join('model','labeler.pkl'), 'rb'))
    
    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False,encoder=loaded_encoder, lb=loaded_lb
    )
    attr_xtab = slice_inference(loaded_model,X_test.copy().reset_index(drop = True), test.copy().reset_index(drop = True),y_test,cat_features, slice_feats='all',test=True)
    for feat in cat_features:
        assert (feat in attr_xtab['attribute_name'].unique().tolist()) == True
    