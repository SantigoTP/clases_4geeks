import logging
from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__) #App sera nuestro aplicativo flask
with open("../models/iris_model.pkl", "rb") as file:
    artifact = pickle.load(file) #Cargamos en tiempo real nuesto modelo creado

@app.route("/",methods=["GET","POST"]) #Creamos nuestra apy para get y post

def prediction():
    if request.method == "POST":
        val1 = float(request.form["val1"])
        val2 = float(request.form["val2"])
        val3 = float(request.form["val3"])
        val4 = float(request.form["val4"])

        df_predict = pd.DataFrame([[val1,val2,val3,val4]], columns=artifact["predictors"])
        result = artifact["model"].predict(df_predict)
        predicted_label = artifact["target_encoder"].inverse_transform(result)
    else:
        predicted_label = None
    
    return render_template("../templates/index.html", prediction=predicted_label)
