import pandas as pd

def test_data_length(data):
    """
    We test that we have enough data to continue
    """
    assert len(data) > 1000
    
def test_columns(data):

    sample1, sample2 = data

    columns = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]

    assert column in data.columns for column in columns