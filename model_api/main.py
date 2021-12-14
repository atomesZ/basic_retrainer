from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import numpy as np
import os
import time
import sklearn

app = FastAPI()

model_joblib_name = "regression.joblib"

model_path = f"models/{model_joblib_name}"

model_info = {}
model_info["last_modified_date"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(model_path)))
model_info["model"] = joblib.load(model_path)

class Item_list(BaseModel):
    X: list

class Item_model(BaseModel):
    model_name: str

@app.get("/")
async def get_model_version():
    return {"Last modified date" : model_info["last_modified_date"]}

@app.post("/new_model")
async def get_new_model(item: Item_model):
    model_path = f"models/{item.model_name}"
    if not os.path.isfile(model_path):
        return "Model not found"

    model_info["last_modified_date"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(model_path)))
    model_info["model"] = joblib.load(model_path)

@app.post("/predict")
async def predict(item: Item_list):
    X = np.array(item.X)

    if len(X.shape) == 1:
        X = X.reshape(1, -1)

    y = model_info["model"].predict(X)
    y = y.tolist()

    return {'y': y}
