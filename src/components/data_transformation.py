import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
        #     yes_no_columns = ['Mood Swing', 'Suicidal thoughts', 'Anorxia', 'Authority Respect', 'Try-Explanation', 'Aggressive Response', 'Ignore & Move-On', 'Nervous Break-down', 'Admit Mistakes', 'Overthinking']
            categorical_columns = ['Sadness', 'Euphoric', 'Exhausted', 'Sleep dissorder']
            degree_out_of_ten_columns = ['Sexual Activity', 'Concentration', 'Optimisim']

            # yes_no_pipeline = Pipeline(
            #     steps=[
            #         ("label_encoder", LabelEncoder())
            #     ]
            # )

            categorical_columns_pipeline = Pipeline(
                steps=[
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler())
                ]
            )

            degree_out_of_ten_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=False),)
                ]
            )

            logging.info("Columns incoding complete")

            preprocessor=ColumnTransformer(
                [
                    # ('yes_no_pipeline', yes_no_pipeline, yes_no_columns),
                    ('categorical_pipeline', categorical_columns_pipeline, categorical_columns),
                    ('degree_out_of_ten_pipeline', degree_out_of_ten_pipeline, degree_out_of_ten_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading of train and test data complete")
            logging.info("Obtaining preproessing object")

            preprocessor_obj= self.get_data_transformer_obj()
            target_column_name='Expert Diagnose'
            
            input_feature_train_df=train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name].map({'Normal': 0,'Bipolar Type-1': 1,'Bipolar Type-2': 2,'Depression': 3}).astype(int)
           
            
            input_feature_test_df=test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df=test_df[target_column_name].map({'Normal': 0,'Bipolar Type-1': 1,'Bipolar Type-2': 2,'Depression': 3}).astype(int)
            
            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            inputfeature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            inputfeature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[inputfeature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[inputfeature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)