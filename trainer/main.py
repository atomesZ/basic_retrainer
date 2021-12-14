from sklearn.linear_model import LinearRegression
import numpy as np
import joblib
import time
import requests


model_name = "regression.joblib"


while True:

    X = np.random.randn(10, 2)
    y = 2 * X[:, 0] - 0.1 * X[:, 1]

    model = LinearRegression()

    model.fit(X, y)

    joblib.dump(model, f"models/{model_name}")

    data = {'model_name' : "regression.joblib"}
    requests.post(url="http://model_api:80/new_model", json=data)

    time.sleep(30)
