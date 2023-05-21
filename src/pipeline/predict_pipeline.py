import os,sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.utils import load_object

class predictpipeline:
    def __init__(self) -> None:
        pass

    def predict(self,feature):
        try:
            logging.info('predict started')

            pre_processor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            
            print('dataframe',feature)
            pre_processor=load_object(pre_processor_path)
            model=load_object(model_path)

            print("model",model)
            print("pre_processor",pre_processor)
            datascaled=pre_processor.transform(feature)
            print('features',datascaled)
            print("model",model)
            predict=model.predict(datascaled)
            return predict
        except Exception as e:
            raise CustomException(e,sys)


class customerdata():
    def __init__(self,
                 Pregnancies,
                 Glucose,
                 BloodPressure,
                 SkinThickness,
                 Insulin,
                 BMI,
                 DiabetesPedigreeFunction,
                 Age) :
        self.Pregnancies=Pregnancies
        self.Glucose=Glucose
        self.BloodPressure=BloodPressure
        self.SkinThickness=SkinThickness
        self.Insulin=Insulin
        self.BMI=BMI
        self.DiabetesPedigreeFunction=DiabetesPedigreeFunction
        self.Age=Age

    def get_data_as_dataframe(self):
        try:
            customer_iuput_data={
                'Pregnancies':[self.Pregnancies],
                'Glucose':[self.Glucose],
                'BloodPressure':[self.BloodPressure],
                'SkinThickness':[self.SkinThickness],
                'Insulin':[self.Insulin],
                'BMI':[self.BMI],
                'DiabetesPedigreeFunction':[self.DiabetesPedigreeFunction],
                'Age':[self.Age]
            }

            df=pd.DataFrame(customer_iuput_data)
            print(df)
            return df
        except Exception as e:
            raise CustomException(e,sys)
        

