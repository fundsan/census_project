# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import train_model,inference,compute_model_metrics

import pickle
import os
import pytest
# Add code to load in the data.


def data():
    return pd.read_csv("data/census_cleaned.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
def train_and_test():
    train, test = train_test_split(data(), test_size=0.20)
    
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
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    
    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False,encoder=encoder, lb=lb
    )
    # Train and save a model.
    model=train_model(X_train,y_train)
    # save the model to disk
    filename = os.path.abspath(os.getcwd())+'/'+os.path.join('model','model.pkl')
    pickle.dump(model, open(filename, 'wb'))
    
    filename = os.path.abspath(os.getcwd())+'/'+os.path.join('model','encoder.pkl')
    pickle.dump(encoder, open(filename, 'wb'))
    
    filename = os.path.abspath(os.getcwd())+'/'+os.path.join('model','labeler.pkl')
    pickle.dump(lb, open(filename, 'wb'))
    
    y_test_hat = inference(model, X_test)
    precision, recall, fbeta=compute_model_metrics(y_test,y_test_hat)
    print("Test results: Precision: {} Recall: {} Fbeta: {}".format(precision, recall, fbeta))
    return model, encoder, lb
def just_do_tst():
    _, test = train_test_split(data(), test_size=0.20)
    
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

    y_test_hat = inference(loaded_model, X_test)
    precision, recall, fbeta=compute_model_metrics(y_test,y_test_hat)
    print("Test results: Precision: {} Recall: {} Fbeta: {}".format(precision, recall, fbeta))
    return precision, recall, fbeta

if __name__ == "__main__":
     
    train_and_test()