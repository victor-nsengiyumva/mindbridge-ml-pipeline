import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            #Random Forest Classifier
            param_grid = {
                'n_estimators': [150, 250, 500],
                'max_depth': [None, 30, 60],
                'min_samples_split': [10, 20, 40],
                'min_samples_leaf': [5, 13, 20]
            }

            rf_classifier = RandomForestClassifier(random_state = 42)

            grid_search = GridSearchCV(estimator = rf_classifier, param_grid = param_grid)

            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_
            optimized_rf_classifier = RandomForestClassifier(**best_params, random_state = 42)

            optimized_rf_classifier.fit(X_train, y_train)

            y_pred_train = optimized_rf_classifier.predict(X_train)
            y_pred_test = optimized_rf_classifier.predict(X_test)

            train_accuracy = accuracy_score(y_test, y_pred_test)

            logging.info(train_accuracy)

            logging.info('Done Choosing the best model')
            
            rf_accuracy = classification_report(y_test, y_pred_test)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=optimized_rf_classifier
            )

            r2_square = r2_score(y_test, y_pred_test)

            return r2_square
                
        except Exception as e:
            CustomException(e, sys)