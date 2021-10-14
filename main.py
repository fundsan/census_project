# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field, PositiveInt
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
    workclass: str 
    fnlgt: int 
    education: str 
    education_num: int = Field(..., alias = "education-num")
    marital_status: str = Field(..., alias = "marital-status")
    occupation: str 
    relationship: str
    race: str 
    sex: str 
    capital_gain: int = Field(..., alias = "capital-gain")
    capital_loss: int = Field(..., alias = "capital-loss")
    hours_per_week: int = Field(..., alias = "hours-per-week")
    native_country: str = Field(..., alias = "native-country")
    
    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                'sex':"Male",
                "capital-gain": 2147,
                "capital-loss": 4,
                "hours-per-week": 40,
                "native-country":"United-States"
                
            }
        }

class PredictionOutput(BaseModel):
    prediction: int 


@app.get("/")
async def say_hello():
    return {"greeting": "Welcome to my model!"}
@app.post("/infer",response_model= PredictionOutput)
async def infer_datapoint(datapoint: DataPoint):
    
    X= pd.DataFrame(datapoint.dict(by_alias=True),index=range(1))
    #X.columns = [x.replace('_','-') for x in X.columns]
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

    return {'prediction':int(inference(loaded_model, X).tolist()[0])}
