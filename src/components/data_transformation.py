import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelBinarizer
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

    def get_data_transformation_object(self):
        try:
            self.numerical_feature = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']
            self.categorical_feature = ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','Contract','PaperlessBilling','PaymentMethod','StreamingMovies','StreamingTV']

            self.new_categorical_feature = ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','Contract','PaperlessBilling','PaymentMethod','Is_streaming']

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

            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy='most_frequent')),
                    ("OHE",OneHotEncoder(drop='first'))
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_columns',num_pipeline,self.numerical_feature),
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

            preprocessing_obj = self.get_data_transformation_object()
            logging.info('Got preprocessing object')

            cleaned_train_df,cleaned_test_df = data_cleaning(train_df,test_df,self.numerical_feature,self.categorical_feature)

            logging.info('TotalCharges is converted into numeric datatype and all blank spaces and - are removes from categorical columns')

            train_df_FE,test_df_FE = feature_engineering(cleaned_train_df,cleaned_test_df)

            logging.info('New feature are created')

            X_train = train_df_FE.drop(columns=['Churn'])
            y_train = train_df_FE['Churn']

            X_test = test_df_FE.drop(columns=['Churn'])
            y_test = test_df_FE['Churn']

            logging.info('Train and test dataset devided into X and y')            

            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)

            logging.info('Preprocessor object applied on X')

            le = LabelBinarizer()
            y_train_transformed = le.fit_transform(y_train)
            y_test_transformed = le.transform(y_test)

            logging.info('LableBinarizer applied on y')
            
            # train_arr = np.c_[X_train_transformed,np.array(y_train_transformed)]
            # test_arr = np.c_[X_test_transformed,np.array(y_test_transformed)]
            logging.info('Saved preprocessed objects')

            save_object(
                file_path = self.data_transformation.preprocessor_file_path,
                obj = preprocessing_obj
            )

            return(
                X_train_transformed,
                X_test_transformed,
                y_train_transformed,
                y_test_transformed
            )
        
        except Exception as e:
            raise CustomException(e,sys)

        