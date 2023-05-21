import os,sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from dataclasses import dataclass


from src.utils import evaluatemetric,Save_obj

@dataclass
class Datamodelconfig:
    Data_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTraining:
    def __init__(self) -> None:
        self.model_trainer_config=Datamodelconfig()

    def Initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("model training started and split the train and test data")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                'LogisticRegression':LogisticRegression(),
                'DecisionTreeClassifier':DecisionTreeClassifier(),
                 'RandomForestClassifier':RandomForestClassifier(),
                 'AdaBoostClassifier':AdaBoostClassifier(),
                 'GradientBoostingClassifier':GradientBoostingClassifier(),
                 'KNeighborsClassifier':KNeighborsClassifier()
            }

            result=evaluatemetric(X_train,y_train,X_test,y_test,models)
            print('resut',result)
            best_score=max(result.values())
            best_model=list(result.keys())[
                list(result.values()).index(best_score)
                ]

            logging.info(f"best model as {best_model}  and the score is  {best_score}")

            print(f"best model as {best_model}  and the score is  {best_score}")
            model=models[best_model]
           
            Save_obj(
                filepath= self.model_trainer_config.Data_model_file_path,
                object=model
                )
        except Exception as e:
            raise CustomException(e,sys)
