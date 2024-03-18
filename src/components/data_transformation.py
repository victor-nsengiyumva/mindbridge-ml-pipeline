import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, FunctionTransformer

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            yes_no_cols = ['Mood Swing', 'Suicidal thoughts', 'Anorxia', 'Authority Respect', 'Try-Explanation', 'Aggressive Response', 'Ignore & Move-On', 'Nervous Break-down', 'Admit Mistakes', 'Overthinking']
            frequency_columns = ['Sadness', 'Euphoric', 'Exhausted', 'Sleep dissorder']
            from_columns = ['Sexual Activity', 'Concentration', 'Optimisim']

            logging.info("Reading of train and test data complete")
            logging.info("Obtaining preproessing object")

            target_column_name='Expert Diagnose'
            encoder = OneHotEncoder(handle_unknown='ignore')

            input_feature_train_df=train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name].map({'Normal': 0,'Bipolar Type-1': 1,'Bipolar Type-2': 2,'Depression': 3}).astype(int)
            
            input_feature_test_df=test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df=test_df[target_column_name].map({'Normal': 0,'Bipolar Type-1': 1,'Bipolar Type-2': 2,'Depression': 3}).astype(int)
            
            for col in yes_no_cols:
                input_feature_train_df[col] = input_feature_train_df[col].map({'YES': 1, 'NO': 0}).astype(int)
                input_feature_test_df[col] = input_feature_test_df[col].map({'YES': 1, 'NO': 0}).astype(int)

            for column in frequency_columns:
                input_feature_train_df[column] = input_feature_train_df[column].map({'Seldom': 0, 'Sometimes': 1, 'Usually': 2, 'Most-Often': 3}).astype(int)
                input_feature_test_df[column] = input_feature_test_df[column].map({'Seldom': 0, 'Sometimes': 1, 'Usually': 2, 'Most-Often': 3}).astype(int)

            for column in from_columns:
                input_feature_train_df[column] = input_feature_train_df[column].astype(str).str.extract('(\d)')
                input_feature_train_df[column] = pd.to_numeric(input_feature_train_df[column])

                input_feature_test_df[column] = input_feature_test_df[column].astype(str).str.extract('(\d)')
                input_feature_test_df[column] = pd.to_numeric(input_feature_test_df[column])
            
            training_df = input_feature_train_df.drop(columns=['Patient Number'], axis=1)
            testing_df = input_feature_test_df.drop(columns=['Patient Number'], axis=1)

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            train_arr = np.c_[np.array(training_df), np.array(target_feature_train_df)]
            test_arr = np.c_[np.array(testing_df), np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            preprocessor = ColumnTransformer([
                    ('onehot', OneHotEncoder(handle_unknown='ignore'), yes_no_cols),
                    ('freq_encoding', LabelEncoder(), frequency_columns),
                    ('from_extract', FunctionTransformer(lambda x: x.str.extract('(\d)').astype(int)), from_columns)
                ])
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
    
    def manual_transformation(self, features: pd.DataFrame) -> pd.DataFrame:
        yes_no_cols = ['Mood Swing', 'Suicidal thoughts', 'Anorxia', 'Authority Respect', 'Try-Explanation', 'Aggressive Response', 'Ignore & Move-On', 'Nervous Break-down', 'Admit Mistakes', 'Overthinking']
        frequency_columns = ['Sadness', 'Euphoric', 'Exhausted', 'Sleep dissorder']
        from_columns = ['Sexual Activity', 'Concentration', 'Optimisim']

        logging.info('Starting Transformation')
        
        for col in yes_no_cols:
            features[col] = features[col].map({'YES': 1, 'NO': 0})

        for column in frequency_columns:
            features[column] = features[column].map({'Seldom': 0, 'Sometimes': 1, 'Usually': 2, 'Most-Often': 3})

        for column in from_columns:
            features[column] = features[column].astype(str).str.extract('(\d)')
            features[column] = pd.to_numeric(features[column])

        logging.info('Finish Transformation')
        print(features)
        return features
