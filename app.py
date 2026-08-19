from flask import Flask,request,render_template

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import customData,predictPipeline,dataTransformation
from src.logger import logging

application = Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = customData(
            customerID = request.form.get('customerID'),
            gender=request.form.get('gender'),
            SeniorCitizen=request.form.get('SeniorCitizen'),
            Partner=request.form.get('Partner'),
            Dependents=request.form.get('Dependents'),
            tenure=request.form.get('tenure'),
            PhoneService=request.form.get('PhoneService'),
            MultipleLines=request.form.get('MultipleLines'),
            InternetService=request.form.get('InternetService'),
            OnlineSecurity=request.form.get('OnlineSecurity'),
            OnlineBackup=request.form.get('OnlineBackup'),
            DeviceProtection=request.form.get('DeviceProtection'),
            TechSupport=request.form.get('TechSupport'),
            StreamingTV=request.form.get('StreamingTV'),
            StreamingMovies=request.form.get('StreamingMovies'),
            Contract=request.form.get('Contract'),
            PaperlessBilling=request.form.get('PaperlessBilling'),
            PaymentMethod=request.form.get('PaymentMethod'),
            MonthlyCharges=request.form.get('MonthlyCharges'),
            TotalCharges=request.form.get('TotalCharges')
        )

        df = data.get_data_as_dataframe()

        print(df)

        transform = dataTransformation()
        transformed_df = transform.transformation(df=df)

        predict_pipeline = predictPipeline()

        result = predict_pipeline.predict(transformed_df)
        logging.info('Successfully made prediction on new data')
        print(result)

        return render_template('home.html',result=result[0])


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
