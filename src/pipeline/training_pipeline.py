import os,sys

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import Data_Intitation

from src.components.data_tranfermation import Data_Transfermation

from src.components.model_trainer import ModelTraining


if __name__=='__main__':
    try:
        logging.info("Training pipeline started")
        obj=Data_Intitation()
        x_train_path,X_test_path=obj.initiate_data_ingestion()
        print(x_train_path,X_test_path)
        data_transfermation=Data_Transfermation()
        train_data,test_data,processor_filepath=data_transfermation.Initate_data_transformation(x_train_path,X_test_path)

        print('preprocessing pickle file',processor_filepath)
        model=ModelTraining()
        model.Initiate_model_training(train_data,test_data)

    except Exception as e:
        raise CustomException(e,sys)