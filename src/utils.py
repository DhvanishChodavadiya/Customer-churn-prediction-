import pandas as pd
import numpy as np
import os
import sys
import pickle
from src.exception import CustomException
from collections import defaultdict

from sklearn.metrics import roc_auc_score,f1_score

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train,X_test,y_train,y_test,models):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_roc_auc_score = roc_auc_score(y_train,y_train_pred)
            test_model_roc_auc_score = roc_auc_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_roc_auc_score

        return report
    
    except Exception as e:
        raise CustomException(e,sys)

def data_cleaning(train_df,test_df,numerical_feature,categorical_feature):
    try:
        for col in numerical_feature:
            if train_df[col].dtype == 'str':
                train_df[col] = pd.to_numeric(train_df[col],errors='coerce')
        for col in numerical_feature:
            if test_df[col].dtype == 'str':
                test_df[col] = pd.to_numeric(test_df[col],errors='coerce')

        for col in categorical_feature:
            train_df[col] =  train_df[col].str.replace(' ', '')
            train_df[col] =  train_df[col].str.replace('-', '')
        for col in categorical_feature:
            test_df[col] =  test_df[col].str.replace(' ', '')
            test_df[col] =  test_df[col].str.replace('-', '')

        return train_df,test_df
    except Exception as e:
        raise CustomException(e,sys)

def feature_engineering(train_df,test_df):
    try:
        train_df['Is_streaming'] = np.where(
            (train_df['StreamingMovies'] == 'Nointernetservice') | (train_df['StreamingTV'] == 'Nointernetservice'), 'Nointernetservice',
                np.where(
                    (train_df['StreamingMovies'] == 'Yes') | (train_df['StreamingTV'] == 'Yes'), 'Yes', 'No'))

        test_df['Is_streaming'] = np.where(
            (test_df['StreamingMovies'] == 'Nointernetservice') | (test_df['StreamingTV'] == 'Nointernetservice'), 'Nointernetservice',
                np.where(
                    (test_df['StreamingMovies'] == 'Yes') | (test_df['StreamingTV'] == 'Yes'), 'Yes', 'No'))

        drop_columns = ['StreamingMovies','StreamingTV']
        for col in drop_columns:
            if col in train_df.columns:
                train_df.drop(columns=[col],inplace=True)
            if col in test_df.columns:
                test_df.drop(columns=[col],inplace=True)

        return train_df,test_df
    except Exception as e:
        raise CustomException(e,sys)