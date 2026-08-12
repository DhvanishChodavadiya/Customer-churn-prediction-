import sys
import os
import pandas as pd
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTraining

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered in data ingestion components")
        try:
            df = pd.read_csv("notebook/data/WA_Fn-UseC_-Telco-Customer-Churn.xls")
            logging.info("Read the data as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train test split initiated')
            train_data,test_data = train_test_split(df,test_size=0.3,random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("train test split is complete")

            return ( 
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)


if __name__ == '__main__':
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    X_train,X_test,y_train,y_test = data_transformation.initiate_data_transformation(train_data,test_data)
    #print(train_arr)

    model_training = ModelTraining()
    best_model,roc,f1= model_training.initialize_model_training(X_train,X_test,y_train,y_test)
    print(f'Best model = {best_model} \n roc_auc_score = {roc} \n f1_score = {f1}')
    
