import pandas as pd
from mlflow import (create_experiment, get_experiment_by_name, log_metric, log_param, set_tracking_uri, start_run)
import psutil
import json
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import utils.custom_preProcessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline



class Experiment:
    def __init__(self, path_config_json):

        try:
            with open(path_config_json, "r") as f:
                config_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Faulty Path for Config JSON")

        self.train_x = config_data["data_paths"]["train_x"]
        self.train_y = config_data["data_paths"]["train_y"]
        self.test_y = config_data["data_paths"]["test_x"]
        self.test_x = config_data["data_paths"]["test_y"]

        self.preprocessing_steps = config_data["preprocessing_steps"]
        self.model = config_data["model"]
        self.HPOptimizer_type = config_data["HPOptimizer_type"]
        self.param_grid = [config_data["param_grid"]]
        self.pipeline = config_data["pipeline"]

        self.tracking_uri = config_data["tracking_uri"]
        self.experiment_description = config_data["experiment_description"]
        self.experiment_name = config_data["experiment_name"]

    def setup_preProcessor(self):
        preprocessor_transformers = []

        for step in self.preprocessing_steps:
            preprocessor_transformers.append((step["name"], (step["type"])))

        return ColumnTransformer(
            transformers=preprocessor_transformers,
            remainder="drop")  # Drop any remaining columns that were not specified in the transformers

    def run_experiment(self):
        # General logging for reproducability
        log_param("cpu_model", psutil.cpu_freq())
        log_param("cpu_count", psutil.cpu_count())
        log_param("ram_total_gb", round(psutil.virtual_memory().total / (1024.0 ** 3), 2))

        # Log system metrics during training
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent

        log_metric("cpu_percent", cpu_percent)
        log_metric("ram_percent", ram_percent)

        # Define classifier with default parameters
        match self.model:
            case 'kNN_Custom':
                classifer = None #TODO: here
            case 'kNN_benchmark':
                classifier = KNeighborsRegressor()
            case _ :
                raise Exception


        # Create a Pipeline
        pipeline = Pipeline([
            ("preprocessor", self.setup_preProcessor()),
            ('classifier', classifier)
        ])

        match self.HPOptimizer_type:
            case 'GridSearchCV':
                search = GridSearchCV(pipeline,
                                      param_grid=self.param_grid,
                                      cv=[(slice(None), slice(None))],
                                      verbose=True
                                      )
            case _:
                print("nothing")
            # TODO: extend

        search.fit(self.train_x, self.train_y)

        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        log_param("__Hyperparam__n_neighbors", search.best_params_["classifier__n_neighbors"])
        log_param("__Hyperparam__weights", search.best_params_["classifier__weights"])
        log_param("__Hyperparam__p", search.best_params_["classifier__p"])

        for param in self.param_grid:
            log_param(str(param["name"]), search.best_params_[str(param["name"])])

        print(search.best_params_)
        print(search.best_params_["classifier"])
        y_pred = search.predict(self.test_x)

        log_metric("R^2", r2_score(y_true=self.test_y, y_pred=y_pred))
        log_metric("MSE", mean_squared_error(y_true=self.test_y, y_pred=y_pred))
        log_metric("MAPE", mean_absolute_percentage_error(y_true=self.test_y, y_pred=y_pred, pos_label=1), )


if __name__ == '__main__':
    experiment = Experiment("templates/template_config.json")
    experiment.run_experiment()
