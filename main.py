# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import os
from typing import List, Union
from starter.ml.model import inference
from starter.ml.data import process_data

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    
    os.system("dvc config core.no_scm true")
    os.system("dvc config core.hardlink_lock true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")
    
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

@app.get("/jane")
async def happy_birthday():
    return {"Happy Birthday Message": "Happy Birthday! It's me Jake, I finally got this stupid thing working. I love you so much!"}
@app.get("/love/{love_points}")
async def love(love_points: int):
    jakes_love_for_jane = 100000
    janes_love_for_jake = love_points 
    if janes_love_for_jake > jakes_love_for_jane:
        return {"Message": "okay, you love me more."}
    else:
        return {"Message": "No, I love you more or equal."}

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