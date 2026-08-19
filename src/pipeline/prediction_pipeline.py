import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object

class predictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join('artifacts','model.pkl')
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            preprocessed_data = preprocessor.transform(features)
            prediction = model.predict(preprocessed_data)

            return prediction
        except Exception as e:
            raise CustomException(e,sys)

class dataTransformation:
    def __init__(self):
        self.numerical_feature = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']
        self.new_numerical_feature = ['SeniorCitizen','tenure','MonthlyCharges']
        self.categorical_feature = ['gender','Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity',
                                                        'OnlineBackup','DeviceProtection','TechSupport','PaperlessBilling',
                                                        'StreamingMovies','StreamingTV','InternetService','Contract','PaymentMethod']
        self.oe_cat_feature = ['InternetService','Contract','PaymentMethod']
        self.new_categorical_feature = ['gender','Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','PaperlessBilling','Is_streaming']
                

    def transformation(self,df):
        try:
            for col in self.numerical_feature:
                if df[col].dtype == 'str':
                    df[col] = pd.to_numeric(df[col],errors='coerce')

            for col in self.categorical_feature:
                df[col] =  df[col].str.replace(' ', '')
                df[col] =  df[col].str.replace('-', '')

            df['Is_streaming'] = np.where(
                        (df['StreamingMovies'] == 'Nointernetservice') | (df['StreamingTV'] == 'Nointernetservice'), 'Nointernetservice',
                            np.where(
                                (df['StreamingMovies'] == 'Yes') | (df['StreamingTV'] == 'Yes'), 'Yes', 'No'))
            
            df['Is_streaming'] = np.where((df['Is_streaming'] == 'Nointernetservice') | (df['Is_streaming'] == 'No'),'No',"Yes")

            drop_columns = ['StreamingMovies','StreamingTV','customerID','TotalCharges']
            for col in drop_columns:
                if col in df.columns:
                    df.drop(columns=[col],inplace=True)

            return df
            
        except Exception as e:
            raise CustomException(e,sys)

class customData:
    def __init__(self,
                    customerID,
                    gender,
                    SeniorCitizen,
                    Partner,
                    Dependents,
                    tenure,
                    PhoneService,
                    MultipleLines,
                    InternetService,
                    OnlineSecurity,
                    OnlineBackup,
                    DeviceProtection,
                    TechSupport,
                    StreamingTV,
                    StreamingMovies,
                    Contract,
                    PaperlessBilling,
                    PaymentMethod,
                    MonthlyCharges,
                    TotalCharges
                 ):
        self.customerID = customerID
        self.gender = gender
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = tenure
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'customerID' : [self.customerID],
                'gender': [self.gender],
                'SeniorCitizen': [self.SeniorCitizen],
                'Partner': [self.Partner],
                'Dependents': [self.Dependents],
                'tenure': [self.tenure],
                'PhoneService': [self.PhoneService],
                'MultipleLines': [self.MultipleLines],
                'InternetService': [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                'OnlineBackup': [self.OnlineBackup],
                'DeviceProtection': [self.DeviceProtection],
                'TechSupport': [self.TechSupport],
                'StreamingTV': [self.StreamingTV],
                'StreamingMovies': [self.StreamingMovies],
                'Contract': [self.Contract],
                'PaperlessBilling': [self.PaperlessBilling],
                'PaymentMethod': [self.PaymentMethod],
                'MonthlyCharges': [self.MonthlyCharges],
                'TotalCharges': [self.TotalCharges]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)