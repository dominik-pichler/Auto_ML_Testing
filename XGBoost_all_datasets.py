import platform
from pathlib import Path
import numpy as np
import pandas as pd
from mlflow import (create_experiment,get_experiment_by_name,log_input,log_metric,log_param,set_tracking_uri,start_run)
from psutil import virtual_memory
from scipy.io.arff import loadarff
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Setting up the MLFlow Project:
set_tracking_uri(uri="http://127.0.0.1:5000")
experiment_description = "Testdescription"


# Hyperparameter-Settings for GridSearch for the logistic regression
param_grid = {
    'xgb__n_estimators': [100, 200, 300],
    'xgb__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.01, 0.1, 0.2],
    # Add more hypetpot_digits_pipeline.pyrparameters you want to tune
}

# Define preprocessing steps with a custom log transformer
def run_log_regression(experiment_id,
                       run_name_ml,
                       project_name,
                       case_name,
                       x_training,
                       x_test,
                       y_training,
                       y_test,
                       k_fold: int,
                       use_crossValidation=False):
    y_training = le.fit_transform(y_training)

    xgb_model = XGBClassifier()

    pipeline = Pipeline(
            [
                ("xgb", xgb_model),
            ]
    )

    run_name = case_name

    if not use_crossValidation:
        run_name = run_name + "without_Crossvalidation"
    else:
        run_name = run_name + "_kFold=" + str(k_fold)

    with start_run(experiment_id=experiment_id, run_name=run_name):
        log_param("__Machine__Python Version", platform.python_version())  #TODO: Es gibt von MLFlow selbst noch Tools zum Monitoring der Auslastungen. Die sollten hier stattdessen verwendet werden!
        log_param("__Machine__Operating System", platform.system())
        log_param("__Machine__Processor", platform.processor())
        log_param("__Machine__Memory_RAM",str(round(virtual_memory().total / (1024.0**3), 2)) + " GB",)

        if use_crossValidation:
            search = GridSearchCV(pipeline, param_grid=param_grid, cv=[(slice(None), slice(None))],verbose=True,scoring='accuracy')
        else:
            search = GridSearchCV(pipeline, param_grid=param_grid, cv=k_fold, verbose=True, scoring='accuracy')

        search.fit(x_training, y_training)

        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        log_param("__Hyperparam__xgb__n_estimators", search.best_params_["xgb__n_estimators"])
        log_param("__Hyperparam__xgb__max_depth", search.best_params_["xgb__max_depth"])
        log_param("__Hyperparam__xgb__learning_rate", search.best_params_["xgb__learning_rate"])

        print(search.best_params_)
        best_estimator = search.best_estimator_
        print("here")
        print(x_test)
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


# ============= DIABETES =============

    # Define different testszenarios:
    parameters = [
        {"n": None, "use_crossvalidation": True, "use_log_trans": True, "use_scaler": True},
        {"n": None, "use_crossvalidation": False, "use_log_trans": False, "use_scaler": True},
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

        cv_list = [2]
        if param["use_crossvalidation"]:
            cv_list = [2, 3, 4, 5, 10]

        for n in cv_list:
             param["n"] = n
             run_log_regression(experiment_id = experiment_id, run_name_ml=experiment_name,case_name= 'XGBoost_breastcancer_umap', project_name='breastcancer',
                 x_training= breastcancer_umap_15_train_X,
                 x_test= breastcancer_umap_15_test,
                 y_training= breastcancer_umap_15_train_y,
                 y_test= None,
                 k_fold= param["n"],
                 use_crossValidation=param["use_crossvalidation"]
             )


    path = Path("/Users/dominikpichler/Documents/Git/ml-assignments/ml-assignments/data/diabetes/raw/diabetes.arff")
    raw_data = loadarff(path)
    df_data = pd.DataFrame(raw_data[0])
    X = df_data.drop("Outcome", axis=1)  # Features
    Y = df_data["Outcome"].astype("|S80").str.decode("utf-8")  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    # Run the function with different parameters
    for param in parameters:

        cv_list = [2]
        if param["use_crossvalidation"]:
            cv_list = [2, 3, 4, 5, 10]

        for n in cv_list:
             param["n"] = n
             run_log_regression(experiment_id = experiment_id, run_name_ml=experiment_name,case_name= 'XGBoost_diabetes', project_name='diabetes',
                 x_training= X_train,
                 x_test= X_test,
                 y_training= y_train,
                 y_test= y_test,
                 k_fold= param["n"],
                 use_crossValidation=param["use_crossvalidation"]
             )


# ====== Loan ======

    experiment_name = "XGBoost_Loan_Dataset"
    experiment_id = (
        create_experiment(experiment_name, experiment_description)
        if get_experiment_by_name(experiment_name) is None
        else get_experiment_by_name(experiment_name).experiment_id
    )

    loan_pca_15_train = pd.read_csv("/Users/dominikpichler/Documents/Git/ml-assignments/ml-assignments/data/loan/clean/pca/loan-10k_pca_15.shuf.lrn.csv")
    loan_pca_15_train_y = loan_pca_15_train["class"]
    loan_pca_15_train_X = loan_pca_15_train.drop("class", axis=1)
    loan_pca_15_test = pd.read_csv(
    "/Users/dominikpichler/Documents/Git/ml-assignments/ml-assignments/data/loan/clean/pca/loan-10k_pca_15.shuf.tes.csv")


# Run the function with different parameters
    for param in parameters:

        cv_list = [2]
        if param["use_crossvalidation"]:
            cv_list = [2, 3, 4, 5, 10]

        for n in cv_list:
             param["n"] = n
             run_log_regression(experiment_id = experiment_id, run_name_ml=experiment_name,case_name= 'XGBoost_Loan_PCA', project_name='loan',
                 x_training= loan_pca_15_train_X,
                 x_test= loan_pca_15_test,
                 y_training= loan_pca_15_train_y,
                 y_test= None,
                 k_fold= param["n"],
                 use_crossValidation=param["use_crossvalidation"]
             )






    loan_tsne_15_train = pd.read_csv("/Users/dominikpichler/Documents/Git/ml-assignments/ml-assignments/data/loan/clean/tsne/loan-10k_tsne_15.shuf.lrn.csv")
    loan_tsne_15_train_y = loan_tsne_15_train["class"]
    loan_tsne_15_train_X = loan_tsne_15_train.drop("class", axis=1)
    loan_tsne_15_test = pd.read_csv(
    "/Users/dominikpichler/Documents/Git/ml-assignments/ml-assignments/data/loan/clean/tsne/loan-10k_tsne_15.shuf.tes.csv")
# Run the function with different parameters
    for param in parameters:

        cv_list = [2]
        if param["use_crossvalidation"]:
            cv_list = [2, 3, 4, 5, 10]

        for n in cv_list:
             param["n"] = n
             run_log_regression(experiment_id = experiment_id, run_name_ml=experiment_name,case_name= 'XGBoost_Loan_tsne', project_name='loan',
                 x_training= loan_tsne_15_train_X,
                 x_test= loan_tsne_15_test,
                 y_training= loan_tsne_15_train_y,
                 y_test= None,
                 k_fold= param["n"],
                 use_crossValidation=param["use_crossvalidation"]
             )


    loan_umap_15_train = pd.read_csv("/Users/dominikpichler/Documents/Git/ml-assignments/ml-assignments/data/loan/clean/umap/loan-10k_umap_15.shuf.lrn.csv")
    loan_umap_15_train_y = loan_umap_15_train["class"]
    loan_umap_15_train_X = loan_umap_15_train.drop("class", axis=1)
    loan_umap_15_test = pd.read_csv(
    "/Users/dominikpichler/Documents/Git/ml-assignments/ml-assignments/data/loan/clean/umap/loan-10k_umap_15.shuf.tes.csv")


# Run the function with different parameters
    for param in parameters:

        cv_list = [2]
        if param["use_crossvalidation"]:
            cv_list = [2, 3, 4, 5, 10]

        for n in cv_list:
             param["n"] = n
             run_log_regression(experiment_id = experiment_id, run_name_ml=experiment_name,case_name= 'XGBoost_loan_umap', project_name='loan',
                 x_training= loan_umap_15_train_X,
                 x_test= loan_umap_15_test,
                 y_training= loan_umap_15_train_y,
                 y_test= None,
                 k_fold= param["n"],
                 use_crossValidation=param["use_crossvalidation"]
             )
        

    loan_selection_15_train = pd.read_csv("/Users/dominikpichler/Documents/Git/ml-assignments/ml-assignments/data/loan/clean/selection/loan_selection.shuf.lrn.csv")
    loan_selection_15_train_y = loan_selection_15_train["class"]
    loan_selection_15_train_X = loan_selection_15_train.drop("class", axis=1)
    loan_selection_15_test = pd.read_csv(
    "/Users/dominikpichler/Documents/Git/ml-assignments/ml-assignments/data/loan/clean/selection/loan_selection.shuf.tes.csv")


# Run the function with different parameters
    for param in parameters:

        cv_list = [2]
        if param["use_crossvalidation"]:
            cv_list = [2, 3, 4, 5, 10]

        for n in cv_list:
             param["n"] = n
             run_log_regression(experiment_id = experiment_id, run_name_ml=experiment_name,case_name= 'XGBoost_loan_selection', project_name='loan',
                 x_training= loan_selection_15_train_X,
                 x_test= loan_selection_15_test,
                 y_training= loan_selection_15_train_y,
                 y_test= None,
                 k_fold= param["n"],
                 use_crossValidation=param["use_crossvalidation"])

# ====== BREASTCANER ======

    experiment_name = "XGBoost_BREASTCANCER_Dataset"
    experiment_id = (
        create_experiment(experiment_name, experiment_description)
        if get_experiment_by_name(experiment_name) is None
        else get_experiment_by_name(experiment_name).experiment_id
    )

    breastcancer_pca_15_train = pd.read_csv("/Users/dominikpichler/Documents/Git/ml-assignments/ml-assignments/data/breastcancer/clean/pca/breast-cancer-diagnostic_pca_15.shuf.lrn.csv")
    breastcancer_pca_15_train_y = breastcancer_pca_15_train["class"]
    breastcancer_pca_15_train_X = breastcancer_pca_15_train.drop("class", axis=1)
    breastcancer_pca_15_test = pd.read_csv(
    "/Users/dominikpichler/Documents/Git/ml-assignments/ml-assignments/data/breastcancer/clean/pca/breast-cancer-diagnostic_pca_15.shuf.tes.csv")


# Run the function with different parameters
    for param in parameters:

        cv_list = [2]
        if param["use_crossvalidation"]:
            cv_list = [2, 3, 4, 5, 10]

        for n in cv_list:
             param["n"] = n
             run_log_regression(experiment_id = experiment_id, run_name_ml=experiment_name,case_name= 'XGBoost_Breastcancer_PCA', project_name='breastcancer',
                 x_training= breastcancer_pca_15_train_X,
                 x_test= breastcancer_pca_15_test,
                 y_training= breastcancer_pca_15_train_y,
                 y_test= None,
                 k_fold= param["n"],
                 use_crossValidation=param["use_crossvalidation"]
             )






    breastcancer_tsne_15_train = pd.read_csv("/Users/dominikpichler/Documents/Git/ml-assignments/ml-assignments/data/breastcancer/clean/tsne/breast-cancer-diagnostic_tsne_15.shuf.lrn.csv")
    breastcancer_tsne_15_train_y = breastcancer_tsne_15_train["class"]
    breastcancer_tsne_15_train_X = breastcancer_tsne_15_train.drop("class", axis=1)
    breastcancer_tsne_15_test = pd.read_csv(
    "/Users/dominikpichler/Documents/Git/ml-assignments/ml-assignments/data/breastcancer/clean/tsne/breast-cancer-diagnostic_tsne_15.shuf.tes.csv")
# Run the function with different parameters
    for param in parameters:

        cv_list = [2]
        if param["use_crossvalidation"]:
            cv_list = [2, 3, 4, 5, 10]

        for n in cv_list:
             param["n"] = n
             run_log_regression(experiment_id = experiment_id, run_name_ml=experiment_name,case_name= 'XGBoost_Breastcancer_tsne', project_name='breastcancer',
                 x_training= breastcancer_tsne_15_train_X,
                 x_test= breastcancer_tsne_15_test,
                 y_training= breastcancer_tsne_15_train_y,
                 y_test= None,
                 k_fold= param["n"],
                 use_crossValidation=param["use_crossvalidation"]
             )


    

    breastcancer_selection_15_train = pd.read_csv("/Users/dominikpichler/Documents/Git/ml-assignments/ml-assignments/data/breastcancer/clean/selection/breast-cancer-diagnostic_selection.shuf.lrn.csv")
    breastcancer_selection_15_train_y = breastcancer_selection_15_train["class"]
    breastcancer_selection_15_train_X = breastcancer_selection_15_train.drop("class", axis=1)
    breastcancer_selection_15_test = pd.read_csv("/Users/dominikpichler/Documents/Git/ml-assignments/ml-assignments/data/breastcancer/clean/selection/breast-cancer-diagnostic_selection.shuf.tes.csv")

    # Run the function with different parameters
    for param in parameters:

        cv_list = [2]
        if param["use_crossvalidation"]:
            cv_list = [2, 3, 4, 5, 10]

        for n in cv_list:
            param["n"] = n
            run_log_regression(experiment_id=experiment_id, run_name_ml=experiment_name,
                               case_name='XGBoost_breastcancer_selection', project_name='breastcancer',
                               x_training=breastcancer_selection_15_train_X,
                               x_test=breastcancer_selection_15_test,
                               y_training=breastcancer_selection_15_train_y,
                               y_test=None,
                               k_fold=param["n"],
                               use_crossValidation=param["use_crossvalidation"]
                               )

