from fastapi.testclient import TestClient
import json
from main import app

client = TestClient(app)


def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome to my model!"}


def test_post_infer():
    r = client.post("/infer",json={'age': 39,
    'workclass': 'State-gov',
    'fnlgt' : 77516,
    'education' : 'Bachelors',
    'education_num' :13,
    'marital_status': 'Never-married',
    'occupation' : 'Adm-clerical',
    'relationship':'Not-in-family',
    'race':'White',
    'sex':'Male',
    'capital_gain':2147,
    'capital_loss':0,
    'hour_per_week':40,
    'native_country':'United-States'})
    assert r.status_code == 200
    assert r.json() == {'prediction':[0]}


def test_get_malformed():
    r = client.get("/items")
    assert r.status_code != 200