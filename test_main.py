from fastapi.testclient import TestClient
import json
from main import app

client = TestClient(app)


def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome to my model!"}


def test_post_infer_zero():
    r = client.post("/infer",json={'age': 39,
    'workclass': 'State-gov',
    'fnlgt' : 77516,
    'education' : 'Bachelors',
    'education-num' :13,
    'marital-status': 'Never-married',
    'occupation' : 'Adm-clerical',
    'relationship':'Not-in-family',
    'race':'White',
    'sex':'Male',
    'capital-gain':2147,
    'capital-loss':0,
    'hours-per-week':40,
    'native-country':'United-States'})
    assert r.status_code == 200
    assert r.json() == {'prediction':0}
def test_post_infer_one():
    r = client.post("/infer",json={'age': 43,
    'workclass': 'Private',
    'fnlgt' : 292175,
    'education' : 'Doctorate',
    'education-num' :16,
    'marital-status': 'Married-civ-spouse',
    'occupation' : 'Prof-specialty',
    'relationship':'Husband',
    'race':'White',
    'sex':'Male',
    'capital-gain':0,
    'capital-loss':0,
    'hours-per-week':45,
    'native-country':'United-States'})
    
    assert r.status_code == 200
    assert r.json() == {'prediction':1}
