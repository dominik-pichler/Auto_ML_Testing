import platform
from pathlib import Path
import numpy as np
import pandas as pd
from mlflow import (create_experiment, get_experiment_by_name, log_metric, log_param, set_tracking_uri,start_run)
from psutil import virtual_memory
from scipy.io.arff import loadarff
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


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
experiment_description = "Logistic Regression"

# Hyperparameter-Settings for GridSearch for the logistic regression
param_grid = [
    {
        "classifier": [LogisticRegression()],
        "classifier__penalty": ["l1", "l2"],
        "classifier__C": np.logspace(-4, 4, 20),
        "classifier__solver": ["liblinear", "saga"],
    },
]

# General
class GenericTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # TODO: Insert custom transformer if needed
        return X


# Define preprocessing steps with a custom log transformer
def log_transform(X):
    return np.log(X + 1e-10)  # Add a small constant to prevent taking log of zero or negative values


def run_log_regression(experiment_id,
                       case_name,
                       x_training,
                       x_test,
                       y_training,
                       y_test,
                       k_fold: int,
                       use_scaler=True, use_log_trans=True):

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", FunctionTransformer(log_transform), slice(0, 1),),
            ("imputer", SimpleImputer(strategy="mean"), slice(0, -1)),
        ],
        remainder="drop",  # Drop any remaining columns that were not specified in the transformers
    )

    # TODO: noch vereinfachen. AM besten über liste an pipeline steps die conditional mit weiteren Elementen befüllt wird.
    if use_log_trans:

        if use_scaler:
            pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression()),
                ]
            )

        else:
            pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("classifier", LogisticRegression()),
                ]
            )

    elif use_scaler:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression()),
            ]
        )

    else:
        pipeline = Pipeline(
            [
                ("classifier", LogisticRegression()),
            ]
        )

    run_name = case_name
    run_name = run_name + "_kFold=" + str(k_fold)
    if use_log_trans:
        run_name += "_with_log_trans"
    if use_scaler:
        run_name += "_with_scaler"

    with start_run(experiment_id=experiment_id, run_name=run_name):
        log_param("__Machine__Python Version",platform.python_version())  # TODO: Es gibt von MLFlow selbst noch Tools zum Monitoring der Auslastungen. Die sollten hier stattdessen verwendet werden!
        log_param("__Machine__Operating System", platform.system())
        log_param("__Machine__Processor", platform.processor())
        log_param("__Machine__Memory_RAM", str(round(virtual_memory().total / (1024.0 ** 3), 2)) + " GB", )

        search = GridSearchCV(pipeline, param_grid=param_grid, cv=[(slice(None), slice(None))], verbose=True, )

        search.fit(x_training, y_training)

        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        log_param("__Hyperparam__Classifier", search.best_params_["classifier"])
        log_param("__Hyperparam__C", search.best_params_["classifier__C"])
        log_param("__Hyperparam__Penalty", search.best_params_["classifier__penalty"])
        log_param("__Hyperparam__Solver", search.best_params_["classifier__solver"])

        print(search.best_params_)
        print(search.best_params_["classifier"])
        y_pred = search.predict(x_test)

        log_metric("F1_Score", f1_score(y_true=y_test, y_pred=y_pred, average="binary", pos_label=1), )
        log_metric("Accuracy Score", accuracy_score(y_true=y_test, y_pred=y_pred))
        log_metric("Precision_Score", precision_score(y_true=y_test, y_pred=y_pred, pos_label=1), )
        log_metric("Recall_Score", recall_score(y_true=y_test, y_pred=y_pred, pos_label=1))


if __name__ == '__main__':

    # ============= DIABETES =============

    # Define different testszenarios:
    parameters = [
        {"n": None, "use_crossvalidation": True, "use_log_trans": True, "use_scaler": True},
        {"n": None, "use_crossvalidation": True, "use_log_trans": False, "use_scaler": True},
        {"n": None, "use_crossvalidation": True, "use_log_trans": False, "use_scaler": False, },
        {"n": None, "use_crossvalidation": False, "use_log_trans": False, "use_scaler": False, },
        {"n": None, "use_crossvalidation": False, "use_log_trans": True, "use_scaler": False, },
        {"n": None, "use_crossvalidation": False, "use_log_trans": True, "use_scaler": True},
    ]

    experiment_name = "Logistische_Regression_Diabetes_Dataset"
    experiment_id = (
        create_experiment(experiment_name, experiment_description)
        if get_experiment_by_name(experiment_name) is None
        else get_experiment_by_name(experiment_name).experiment_id
    )

    path = Path("/Users/dominikpichler/Documents/Git/ml-assignments/ml-assignments/data/diabetes/raw/diabetes.arff")
    raw_data = loadarff(path)
    df_data = pd.DataFrame(raw_data[0])
    X = df_data.drop("Outcome", axis=1)  # Features
    Y = df_data["Outcome"].astype("|S80").str.decode("utf-8")  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    # Run the function with different parameters
    for param in parameters:
        cv_list = [2, 3, 4, 5, 10] #k in kFold CV
        for n in cv_list:
            param["n"] = n
            run_log_regression(experiment_id=experiment_id,
                               case_name='Logistische_regression',
                               x_training=X_train,
                               x_test=X_test,
                               y_training=y_train,
                               y_test=y_test,
                               k_fold=param["n"],
                               use_scaler=param["use_scaler"],
                               use_log_trans=param["use_log_trans"],
                               )


