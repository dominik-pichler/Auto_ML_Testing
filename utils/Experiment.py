import pandas as pd
from mlflow import (log_metric, log_param, start_run)
import psutil
import json
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
from ..custom_implementations.custom_models.custom_kNN import custom_kNN

warnings.filterwarnings("ignore")  # TODO: Remove before release


class Experiment:
    def __init__(self, path_config_json,experiment_id):

        try:
            with open(path_config_json, "r") as f:
                config_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Faulty Path for Config JSON")

        self.train_x = pd.read_csv(config_data["data_paths"]["train_x"])
        self.train_y = pd.read_csv(config_data["data_paths"]["train_y"])
        self.test_y = pd.read_csv(config_data["data_paths"]["test_y"])
        self.test_x = pd.read_csv(config_data["data_paths"]["test_x"])

        self.preprocessing_steps = config_data["preprocessing_steps"]
        self.param_grid = [config_data["param_grid"]]

        self.model = config_data["model"]
        self.classifier = self._get_classifier()

        self.HPOptimizer_type = config_data["HPOptimizer_type"]
        self.pipeline = config_data["pipeline"]

        self.tracking_uri = config_data["tracking_uri"]
        self.experiment_description = config_data["experiment_description"]
        self.experiment_name = config_data["experiment_name"]
        self.experiment_id = experiment_id

    def _get_classifier(self):
        # Dictionary mapping model names to scikit-learn classes
        model_dict = {
            'kNN': KNeighborsClassifier,
            'logistic_regression': LogisticRegression,
            'svm': SVC,
            'random_forest': RandomForestClassifier,
            'decision_tree': DecisionTreeClassifier,
            'KNeighborsRegressor': KNeighborsRegressor,
            'custom_kNN': custom_kNN  # Use custom kNN class directly
            # TODO: Add more mappings as needed
        }

        # Get the classifier class from the dictionary
        classifier_class = model_dict.get(self.model)
        if classifier_class is None:
            raise ValueError(f"Model '{self.model}' is not supported.")

        # Initialize and return the classifier instance
        if self.model == 'custom_kNN':
            # Extract custom kNN parameters from param_grid
            custom_kNN_params = {param.split('__')[1]: value for param, value in self.param_grid[0].items() if
                                 'classifier__' in param}
            return classifier_class(**custom_kNN_params)
        else:
            return classifier_class()

    def setup_preProcessor(self):
        preprocessor_transformers = []

        for step in self.preprocessing_steps:
            preprocessor_transformers.append((step["name"], (step["type"])))

        return ColumnTransformer(
            transformers=preprocessor_transformers,
            remainder="drop")  # Drop any remaining columns that were not specified in the transformers

    def run_experiment(self):
        # Setting up the MLFlow Project:
        with start_run(experiment_id=self.experiment_id, run_name=self.experiment_name):

            # General logging for reproducability
            log_param("_cpu_model", psutil.cpu_freq())
            log_param("_cpu_count", psutil.cpu_count())
            log_param("_ram_total_gb", round(psutil.virtual_memory().total / (1024.0 ** 3), 2))

            # Create a Pipeline
            pipeline = Pipeline([
                # ("preprocessor", self.setup_preProcessor()), #TODO: Include Preprocessing in Pipeline
                ('classifier', self.classifier)
            ])

            match self.HPOptimizer_type:
                case 'GridSearch':
                    search = GridSearchCV(pipeline,
                                          param_grid=self.param_grid,
                                          cv=3,
                                          verbose=0
                                          )
                # TODO: Implement more searchers
                case _:
                    raise ValueError(f"Searcher '{self.model}' is not supported.")

            search.fit(self.train_x, self.train_y)

            for key in self.param_grid[0].keys():
                log_param(f"_Hyperparamter_{key}", search.best_params_[key])
            log_param("_HPOtimizer_type",self.HPOptimizer_type)
            y_pred = search.predict(self.test_x)

            # TODO: Add way to also parametrise the metrics  (CAUTION: Right now it will fail for Classification
            log_metric("R2", r2_score(y_true=self.test_y, y_pred=y_pred))
            log_metric("MSE", mean_squared_error(y_true=self.test_y, y_pred=y_pred))
            log_metric("MAPE", mean_absolute_percentage_error(y_true=self.test_y, y_pred=y_pred), )