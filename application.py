from flask import Flask,request,render_template,app
import pandas as pd
import numpy as np
import pickle
import os,sys


application = Flask(__name__)
app=application

from src.pipeline.predict_pipeline import predictpipeline,customerdata


@app.route("/")
def hello_world():
    return render_template("index.html")

##route for  home page
@app.route('/predict',methods=["GET","POST"])
def predict_datapoint():
    result=""
    if request.method=="POST":
        data=customerdata(
        Pregnancies=int(request.form.get("Pregnancies")),
        Glucose = float(request.form.get('Glucose')),
        BloodPressure = float(request.form.get('BloodPressure')),
        SkinThickness = float(request.form.get('SkinThickness')),
        Insulin = float(request.form.get('Insulin')),
        BMI = float(request.form.get('BMI')),
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction')),
        Age = float(request.form.get('Age')),
        )

        finalnewdata=data.get_data_as_dataframe()
        predictdata=predictpipeline()
        predict=predictdata.predict(finalnewdata)
        if predict[0]==1:
            result="Diabetic"
        else:
            result="No-Diabetic"
        
        return render_template('single_prediction.html',result=result)
    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
