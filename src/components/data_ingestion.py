import os, sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class Data_Ingestion_config:
    training_datapath=os.path.join('artifacts','train.csv')
    test_datapath=os.path.join('artifacts','test.csv')
    rawdata_datapath=os.path.join('artifacts','raw.csv')

class Data_Intitation:
    def __init__(self) -> None:
        self.ingsttion_config=Data_Ingestion_config()
    
    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            df=pd.read_csv(os.path.join('notebook\data','diabetes.csv'))
            logging.info('data read from pandas')

            os.makedirs(os.path.dirname(self.ingsttion_config.rawdata_datapath),exist_ok=True)
            df.to_csv(self.ingsttion_config.rawdata_datapath,index=False)

            df['Glucose']=df['Glucose'].replace(0,df['Glucose'].mean())
            df['BloodPressure']=df['BloodPressure'].replace(0,df['BloodPressure'].mean())
            df['SkinThickness']=df['SkinThickness'].replace(0,df['SkinThickness'].mean())
            df['Insulin']=df['Insulin'].replace(0,df['Insulin'].mean())
            df['BMI']=df['BMI'].replace(0,df['BMI'].mean())
            
            train_df,test_df=train_test_split(df,random_state=42,test_size=0.30)
            train_df.to_csv(self.ingsttion_config.training_datapath,index=False,header=True)
            test_df.to_csv(self.ingsttion_config.test_datapath,index=False,header=True)
            logging.info("trian and test data file saved")

            return (
                self.ingsttion_config.training_datapath,
                self.ingsttion_config.test_datapath
            )

        except Exception as e:
            logging.info("error in initiate_data_ingestion")
            raise CustomException(e,sys)
