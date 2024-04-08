import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.ensemble import StackingClassifier

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

            ##Random Forest Classifier
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

            y_pred_test = optimized_rf_classifier.predict(X_test)

            train_accuracy = accuracy_score(y_test, y_pred_test)

            logging.info(train_accuracy)

            ##SVM
            svm_param_grid = {
                'C': [0.1, 1, 10],                   
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto'],          
                'degree': [2, 3, 4],                 
            }       

            svm = SVC(random_state=42)
            grid_search_svm = GridSearchCV(estimator = svm, param_grid = svm_param_grid)

            grid_search_svm.fit(X_train, y_train)

            best_params = grid_search_svm.best_params_
            optimized_svm_classifier = SVC(**best_params, random_state = 42)

            optimized_svm_classifier.fit(X_train, y_train)

            y_pred_test_svm = optimized_svm_classifier.predict(X_test)

            optimized_svm_accuracy = accuracy_score(y_test, y_pred_test_svm)
            logging.info(optimized_svm_accuracy)

            ##Ensemble Model
            np.random.seed(42)
            base_models = [
                # ('logreg', LogisticRegression()),
                ('svm', SVC()),
                ('rf', RandomForestClassifier())
            ]
            stacking_clf = StackingClassifier(estimators=base_models, final_estimator=SVC())

            stacking_clf.fit(X_train, y_train)

            y_pred_test_stacking = stacking_clf.predict(X_test)

            logging.info(accuracy_score(y_test, y_pred_test_stacking))

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=stacking_clf
            )
                
        except Exception as e:
            CustomException(e, sys)