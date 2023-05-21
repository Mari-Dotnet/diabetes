from src.logger import logging
from src.exception import CustomException
import os,sys
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import Save_obj


from dataclasses import dataclass

@dataclass
class Data_Transfermation_config:
    preprocessor_filepath=os.path.join('artifacts','preprocessor.pkl')

class Data_Transfermation:
    def __init__(self):
        self.Data_tranfer_config=Data_Transfermation_config
    
    def get_data_trsnfermation_obj(self):
        try:
            logging.info('Get gata Tranfermation started')

            numric_colum=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
            ##numeric pipeline
            numeric_pipeline=Pipeline(
                steps=[
                 ('impute',SimpleImputer(strategy='median')),
                 ('scalling',StandardScaler())
                ]
            )

           
            logging.info("numeric pipeline")

            prepocessor=ColumnTransformer([
                ('num_pipeline',numeric_pipeline,numric_colum)
            ])

            logging.info('pipeline completed')


            return prepocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def Initate_data_transformation(self,train_path,test_path):
        try:
            logging.info("Data trnfromation initiate started")

            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info(f"train and test data path loaded and  train data {train_df.shape} and test data {test_df.shape}")

            target='Outcome'
            ## Split input and output feature
            input_feature_train_df=train_df.drop(columns=target)
            input_feature_train_target=train_df[target]

            input_feature_test_df=test_df.drop(columns=target)
            input_feature_test_target=test_df[target]

            logging.info("pre proecssing started")

            preprocessing=self.get_data_trsnfermation_obj()
            input_feature_train_df=preprocessing.fit_transform(input_feature_train_df)
            input_feature_test_df=preprocessing.transform(input_feature_test_df)

            train_array=np.c_[input_feature_train_df,np.array(input_feature_train_target)]
            test_array=np.c_[input_feature_test_df,input_feature_test_target]

            ##save the preprocessing object
            Save_obj(
                filepath=self.Data_tranfer_config.preprocessor_filepath,
                object=preprocessing
            )

            return(
                train_array,
                test_array,
                self.Data_tranfer_config.preprocessor_filepath
            )
            logging.info("preprocessing pickle file saved")

        except Exception as e:
            raise CustomException(e,sys)




            

            
            

    
        