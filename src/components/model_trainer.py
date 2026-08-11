import sys
import os
from src.exception import CustomException
from src.logger import logging

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_auc_score

from src.utils import save_object,evaluate_model

from dataclasses import dataclass

@dataclass
class ModelTrainingConfig:
    trained_model_path: str = os.path.join('artifacts','model.pkl')

class ModelTraining:
    def __init__(self):
        self.model_training_config = ModelTrainingConfig()

    def initialize_model_training(self,X_train,X_test,y_train,y_test):
        try:
            models = {
                "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
                "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42),
                "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
                "LightGBM": LGBMClassifier(class_weight='balanced', random_state=42),
                "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "SVM": SVC(class_weight='balanced', probability=True, random_state=42),
                "KNN": KNeighborsClassifier(n_neighbors=5),
            }

            model_report:dict = evaluate_model(X_train,X_test,y_train,y_test,models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            logging.info(f'Best model is {best_model}')

            save_object(
                file_path=self.model_training_config.trained_model_path,
                obj=best_model
            )
            logging.info("Best model is saved")

            predict = best_model.predict(X_test)
            roc__auc_score = roc_auc_score(y_test,predict)

            return best_model,roc__auc_score


        except Exception as e:
            raise CustomException(e,sys)