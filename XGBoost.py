import platform
import numpy as np
import pandas as pd
from mlflow import (create_experiment,get_experiment_by_name,log_metric,log_param,set_tracking_uri,start_run)
from psutil import virtual_memory
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

''' 
TODOs:
1) Store MLFlow Data in SQLite oder PostgreSQL
2) Generalise to DRY
3) Unit Tests
4) Provide solution to efficiently & simple! test many different pipeline configs. Maybe YAMLs?
5) Use MLFlow monitoring instead of simply stupidly tracking static system parameters
'''


# Setting up the MLFlow Project:
set_tracking_uri(uri="http://127.0.0.1:5000")
experiment_description = "XGBoost"


# Hyperparameter-Settings for GridSearch for the logistic regression
param_grid = {
    'xgb__n_estimators': [100, 200, 300],
    'xgb__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.01, 0.1, 0.2],
    # Add more hypetpot_digits_pipeline.pyrparameters you want to tune
}

# Define preprocessing steps with a custom log transformer
def run_XGBoost_regression(experiment_id,
                           project_name,
                           case_name,
                           x_training,
                           x_test,
                           y_training,
                           y_test,
                           k_fold: int):

    y_training = le.fit_transform(y_training)
    xgb_model = XGBClassifier()

    pipeline = Pipeline(
            [
                ("xgb", xgb_model),
            ]
    )


    run_name = case_name + "_kFold=" + str(k_fold)

    with start_run(experiment_id=experiment_id, run_name=run_name):
        log_param("__Machine__Python Version", platform.python_version())
        log_param("__Machine__Operating System", platform.system())
        log_param("__Machine__Processor", platform.processor())
        log_param("__Machine__Memory_RAM",str(round(virtual_memory().total / (1024.0**3), 2)) + " GB",)

        search = GridSearchCV(pipeline, param_grid=param_grid, cv=k_fold, verbose=True, scoring='accuracy')

        search.fit(x_training, y_training)

        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        log_param("__Hyperparam__xgb__n_estimators", search.best_params_["xgb__n_estimators"])
        log_param("__Hyperparam__xgb__max_depth", search.best_params_["xgb__max_depth"])
        log_param("__Hyperparam__xgb__learning_rate", search.best_params_["xgb__learning_rate"])

        print(search.best_params_)
        best_estimator = search.best_estimator_
        y_pred = best_estimator.predict(x_test)


        if not y_test is None: # This only works in non-Kaggle Sets.
            y_test = y_test.to_numpy()
            Y_pred = y_pred.astype(np.int8)
            Y_test = y_test.astype(np.int8)

            log_metric("F1_Score",f1_score(y_true=Y_test, y_pred=Y_pred, average="binary", pos_label=1),)
            log_metric("Accuracy Score", accuracy_score(y_true=Y_test, y_pred=Y_pred))
            log_metric("Precision_Score",precision_score(y_true=Y_test, y_pred=Y_pred, pos_label=1),)
            log_metric("Recall_Score", recall_score(y_true=Y_test, y_pred=Y_pred, pos_label=1))

        else:
            df_ypred = pd.DataFrame(y_pred)
            df_ypred.to_csv('/Users/dominikpichler/Documents/Git/ml-assignments/ml-assignments/predictions/' + project_name  + '/' + case_name)


if __name__ == '__main__':

    # Define different testszenarios:
    parameters = [
        {"n": None, "use_log_trans": True, "use_scaler": True},
        {"n": None, "use_log_trans": False, "use_scaler": True},
        {"n": None, "use_log_trans": False, "use_scaler": False},
        {"n": None, "use_log_trans": True, "use_scaler": False}
    ]


    experiment_name = "XGBoost_Diabetes_Dataset"
    experiment_id = (
        create_experiment(experiment_name, experiment_description)
        if get_experiment_by_name(experiment_name) is None
        else get_experiment_by_name(experiment_name).experiment_id
    )


    breastcancer_umap_15_train = pd.read_csv("/Users/dominikpichler/Documents/Git/ml-assignments/ml-assignments/data/breastcancer/clean/umap/breast-cancer-diagnostic_umap_15.shuf.lrn.csv")
    breastcancer_umap_15_train_y = breastcancer_umap_15_train["class"]
    breastcancer_umap_15_train_X = breastcancer_umap_15_train.drop("class", axis=1)
    breastcancer_umap_15_test = pd.read_csv("/Users/dominikpichler/Documents/Git/ml-assignments/ml-assignments/data/breastcancer/clean/umap/breast-cancer-diagnostic_umap_15.shuf.tes.csv")

# Run the function with different parameters
    for param in parameters:

        cv_list = [2, 3, 4, 5, 10]

        for n in cv_list:
             param["n"] = n
             run_XGBoost_regression(experiment_id = experiment_id, case_name='XGBoost_breastcancer', project_name='breastcancer',
                                    x_training= breastcancer_umap_15_train_X,
                                    x_test= breastcancer_umap_15_test,
                                    y_training= breastcancer_umap_15_train_y,
                                    y_test= None,
                                    k_fold= param["n"],
                                    )


