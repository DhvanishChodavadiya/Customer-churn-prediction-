import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelBinarizer,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,feature_engineering,data_cleaning

@dataclass
class DataTransformationConfig:
    preprocessor_file_path: str=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation = DataTransformationConfig()
        self.numerical_feature = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']
        self.new_numerical_feature = ['SeniorCitizen','tenure','MonthlyCharges']
        self.categorical_feature = ['gender','Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity',
                                                'OnlineBackup','DeviceProtection','TechSupport','PaperlessBilling',
                                                'StreamingMovies','StreamingTV','InternetService','Contract','PaymentMethod']
        self.oe_cat_feature = ['InternetService','Contract','PaymentMethod']
        self.new_categorical_feature = ['gender','Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','PaperlessBilling','Is_streaming']
        

    def get_data_transformation_object(self):
        try:
            
            # self.numerical_feature = train_df.select_dtypes(exclude="str").columns
            # self.categorical_feature = train_df.select_dtypes(include="str").columns

            # for col in self.categorical_feature:
            #     if col == 'Churn':
            #         self.categorical_feature = self.categorical_feature.drop(col)

            num_pipeline = Pipeline(
                steps=[
                    ('Imputer',SimpleImputer(strategy='mean')),
                    ("Scaler",StandardScaler())
                ]
            )

            oe_cat_pipeline = Pipeline(
                steps=[
                    ('Imputer',SimpleImputer(strategy='most_frequent')),
                    ('OE',OrdinalEncoder(categories=[['No','DSL','Fiberoptic'],['Monthtomonth','Oneyear','Twoyear'],['Electroniccheck','Mailedcheck','Banktransfer(automatic)','Creditcard(automatic)']]))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy='most_frequent')),
                    ("OHE",OneHotEncoder(drop='first'))
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_columns',num_pipeline,self.new_numerical_feature),
                ('oe_cat_columns',oe_cat_pipeline,self.oe_cat_feature),
                ('cat_columns',cat_pipeline,self.new_categorical_feature)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Reading training and testing data is complete')

            cleaned_train_df,cleaned_test_df = data_cleaning(train_df,test_df,self.numerical_feature,self.categorical_feature)

            logging.info('TotalCharges is converted into numeric datatype and all blank spaces and - are removes from categorical columns')

            train_df_FE,test_df_FE = feature_engineering(cleaned_train_df,cleaned_test_df)

            logging.info('New feature are created')

            X_train = train_df_FE.drop(columns=['Churn'])
            y_train = train_df_FE['Churn']

            X_test = test_df_FE.drop(columns=['Churn'])
            y_test = test_df_FE['Churn']

            logging.info('Train and test dataset devided into X and y') 

            preprocessing_obj = self.get_data_transformation_object()
            logging.info('Got preprocessing object')           

            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)

            logging.info('Preprocessor object applied on X')

            le = LabelBinarizer()
            y_train_transformed = le.fit_transform(y_train)
            y_test_transformed = le.transform(y_test)

            logging.info('LableBinarizer applied on y')
            
            # train_arr = np.c_[X_train_transformed,np.array(y_train_transformed)]
            # test_arr = np.c_[X_test_transformed,np.array(y_test_transformed)]

            save_object(
                file_path = self.data_transformation.preprocessor_file_path,
                obj = preprocessing_obj
            )

            logging.info('Saved preprocessed objects')

            return(
                X_train_transformed,
                X_test_transformed,
                y_train_transformed,
                y_test_transformed
            )
        
        except Exception as e:
            raise CustomException(e,sys)

        