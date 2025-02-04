from flask import Flask, request, render_template, redirect
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, PredictionPipeline


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=['GET', 'POST'])
def predict_data():
    if request.method == 'GET':
        return render_template("home.html")
    
    elif request.method == 'POST':
        
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

        df_pred = data.get_data_as_dataframe()
        print(df_pred)

        pred_pipeline = PredictionPipeline()
        prediction = pred_pipeline.predict(df_pred)

        return render_template("home.html", results=prediction[0])
    

if __name__ == "__main__":
    app.run("0.0.0.0", debug=True)

