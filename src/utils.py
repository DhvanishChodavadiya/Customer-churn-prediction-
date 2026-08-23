import pandas as pd
import numpy as np
import optuna
import os
import sys
import pickle
from src.exception import CustomException
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score,f1_score,precision_score,recall_score

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train,X_test,y_train,y_test):
    try:
        #THRESHOLD = 0.3

        #model = LogisticRegression(penalty='l1', solver='saga', C=0.0010422116168071938, max_iter= 228, class_weight= 'balanced', tol= 0.009876899486027357)
        # model = LogisticRegression(class_weight='balanced',max_iter=1000,C=0.1,penalty='l1',solver='liblinear')
       
        # model.fit(X_train,y_train)

        # y_pred = model.predict(X_test)
        # # proba = model.predict_proba(X_test)[:, 1]
        # # y_pred = (proba >= THRESHOLD).astype(int)   

        # f1 = f1_score(y_test,y_pred)
        # precision = precision_score(y_test,y_pred)
        # recall = recall_score(y_test,y_pred)

        param_grid = [
    {
        "penalty": ["l1", "l2"],
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "solver": ["liblinear"],
    },
    {
        "penalty": ["l2", None],
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs"],
    },
    {
        "penalty": ["elasticnet"],
        "C": [0.01, 0.1, 1, 10],
        "l1_ratio": [0.2, 0.5, 0.8],
        "solver": ["saga"],
    },
]

        grid_search = GridSearchCV(
            estimator=LogisticRegression(class_weight='balanced',max_iter=1000),
            param_grid=param_grid,
            scoring="recall",      # change to "f1", "roc_auc", etc. as needed
            cv=5,
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        model = grid_search.best_estimator_

        y_pred = model.predict(X_test)
        f1 = f1_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)


        return f1,model,precision,recall
    
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

        train_df['Is_streaming'] = np.where((train_df['Is_streaming'] == 'Nointernetservice') | (train_df['Is_streaming'] == 'No'),'No',"Yes")

        test_df['Is_streaming'] = np.where(
            (test_df['StreamingMovies'] == 'Nointernetservice') | (test_df['StreamingTV'] == 'Nointernetservice'), 'Nointernetservice',
                np.where(
                    (test_df['StreamingMovies'] == 'Yes') | (test_df['StreamingTV'] == 'Yes'), 'Yes', 'No'))

        test_df['Is_streaming'] = np.where((test_df['Is_streaming'] == 'Nointernetservice') | (test_df['Is_streaming'] == 'No'),'No',"Yes")

        drop_columns = ['StreamingMovies','StreamingTV','customerID','TotalCharges']
        for col in drop_columns:
            if col in train_df.columns:
                train_df.drop(columns=[col],inplace=True)
            if col in test_df.columns:
                test_df.drop(columns=[col],inplace=True)

        return train_df,test_df
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)