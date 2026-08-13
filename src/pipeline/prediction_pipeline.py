import sys
import os
import pandas as pd
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
        pass

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