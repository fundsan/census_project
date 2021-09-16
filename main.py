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
    
app = FastAPI(docs_url='/docs')
loaded_model = pickle.load(open(os.path.abspath(os.getcwd())+'/'+os.path.join('model','model.pkl'), 'rb'))
loaded_encoder = pickle.load(open(os.path.abspath(os.getcwd())+'/'+os.path.join('model','encoder.pkl'), 'rb'))
loaded_lb = pickle.load(open(os.path.abspath(os.getcwd())+'/'+os.path.join('model','labeler.pkl'), 'rb'))

class DataPoint(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example = "State-gov")
    fnlgt: int = Field(..., example = 77516)
    education: str = Field(..., example = "Bachelors")
    education_num: int = Field(..., alias = "education-num", example = 13)
    marital_status: str = Field(..., alias = "marital-status", example = "Never-married")
    occupation: str = Field(..., example = "Adm-clerical")
    relationship: str = Field(..., example = "Not-in-family")
    race: str = Field(..., example = "White")
    sex: str = Field(..., example = "Male")
    capital_gain: int = Field(..., alias = "capital-gain", example = 2147)
    capital_loss: int = Field(..., alias = "capital-loss", example = 4)
    hours_per_week: int = Field(..., alias = "hours-per-week", example = 40)
    native_country: str = Field(..., alias = "native-country", example = "United-States")

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
