import sys
import os
import pandas as pd
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.utils import load_obj

class Pipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            transform = DataTransformation()
            model_path=os.path.join("artifacts","model.pkl")
            # preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_obj(file_path=model_path)
            # preprocessor=load_obj(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=transform.manual_transformation(features)
            preds=model.predict(data_scaled)
            return preds
            
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self, 
                sadness:str,
                euphoric: str,
                exhausted:str,
                sleep_dissorder: str,
                mood_swing: str,
                suicidal_thoughts: str,
                anorxia: str,
                authority_respect: str,
                try_explanation: str,
                aggressive_response: str,
                ignore_and_move_on: str,
                nervous_break_down: str,
                admit_mistakes: str,
                overthinking: str,
                sexual_activity: str,
                concentration: str ,
                optimisim: str):
        self.sadness = sadness
        self.euphoric = euphoric
        self.exhausted = exhausted
        self.sleep_dissorder = sleep_dissorder   
        self.mood_swing = mood_swing
        self.suicidal_thoughts = suicidal_thoughts  
        self.anorxia = anorxia
        self.authority_respect = authority_respect
        self.try_explanation = try_explanation
        self.aggressive_response = aggressive_response
        self.ignore_and_move_on = ignore_and_move_on
        self.nervous_break_down = nervous_break_down
        self.admit_mistakes = admit_mistakes
        self.overthinking = overthinking
        self.sexual_activity = sexual_activity
        self.concentration = concentration
        self.optimisim = optimisim

    def get_data_as_df(self):
        try:
            input_dict = {
                'Sadness': [self.sadness],
                'Euphoric': [self.euphoric],
                'Exhausted': [self.exhausted],
                'Sleep dissorder': [self.sleep_dissorder],
                'Mood Swing': [self.mood_swing],
                'Suicidal thoughts': [self.suicidal_thoughts],
                'Anorxia': [self.anorxia],
                'Authority Respect': [self.authority_respect],
                'Try-Explanation': [self.try_explanation],
                'Aggressive Response': [self.aggressive_response],
                'Ignore & Move-On': [self.ignore_and_move_on],
                'Nervous Break-down': [self.nervous_break_down],
                'Admit Mistakes': [self.admit_mistakes],
                'Overthinking': [self.overthinking],
                'Sexual Activity': [self.sexual_activity],
                'Concentration': [self.concentration],
                'Optimisim': [self.optimisim]
            }
            return pd.DataFrame(input_dict)
        except Exception as e:
            raise CustomException(e, sys)
