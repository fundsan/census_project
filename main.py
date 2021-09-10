# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import os
from typing import List, Union
from starter.ml.model import inference
from starter.ml.data import process_data


app = FastAPI()
loaded_model = pickle.load(open(os.path.abspath(os.getcwd())+'/'+os.path.join('model','model.pkl'), 'rb'))
loaded_encoder = pickle.load(open(os.path.abspath(os.getcwd())+'/'+os.path.join('model','encoder.pkl'), 'rb'))
loaded_lb = pickle.load(open(os.path.abspath(os.getcwd())+'/'+os.path.join('model','labeler.pkl'), 'rb'))

class DataPoint(BaseModel):
    age: int
    workclass : str 
    fnlgt : int
    education : str
    education_num :int
    marital_status: str
    occupation : str
    relationship:str
    race:str
    sex:str
    capital_gain:int
    capital_loss:int
    hour_per_week:int
    native_country:str

class PredictionOutput(BaseModel):
    prediction: Union[list,int] 

@app.get("/")
async def say_hello():
    return {"greeting": "Welcome to my model!"}
@app.post("/infer")
async def infer_datapoint(datapoint: DataPoint,response_model=PredictionOutput):
    X= pd.DataFrame(datapoint.dict(by_alias=True),index=range(1))
    X.columns = [x.replace('_','-') for x in X.columns]
    loaded_model = pickle.load(open(os.path.abspath(os.getcwd())+'/'+os.path.join('model','model.pkl'), 'rb'))
    loaded_encoder = pickle.load(open(os.path.abspath(os.getcwd())+'/'+os.path.join('model','encoder.pkl'), 'rb'))
    loaded_lb = pickle.load(open(os.path.abspath(os.getcwd())+'/'+os.path.join('model','labeler.pkl'), 'rb'))
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
    X,y,enc, lb = process_data(X, categorical_features=cat_features, label=None, training=False, encoder=loaded_encoder, lb=loaded_lb)

    return {'prediction':inference(loaded_model, X).tolist()}