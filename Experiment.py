import pandas as pd
from mlflow import (create_experiment, get_experiment_by_name, log_metric, log_param, set_tracking_uri,start_run)
import psutil
import json

class Experiment:
    def __init__(self,path_config_json):

        try:
            with open(path_config_json,"r") as f:
               config_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Faulty Path for Config JSON")

        self.train_x = config_data["data_paths"]["train_x"]
        self.train_y = config_data["data_paths"]["train_y"]
        self.test_y  = config_data["data_paths"]["test_x"]
        self.test_x  = config_data["data_paths"]["test_y"]

        self.preprocessing_steps = config_data["preprocessing_steps"]
        self.model = config_data["model"]
        self.HPOptimizer_type = config_data["HPOptimizer_type"]
        self. param_grid = [config_data["param_grid"]]

        self.tracking_uri = config_data["tracking_uri"]
        self.experiment_description = config_data["experiment_description"]
        self.experiment_name = config_data["experiment_name"]
        print("Init successfull")

    def run_experiment(self):
        # General logging for reproducability
        log_param("cpu_model", psutil.cpu_freq().model)
        log_param("cpu_count", psutil.cpu_count())
        log_param("ram_total_gb", round(psutil.virtual_memory().total / (1024.0 ** 3), 2))


        # Log system metrics during training
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent

        log_metric("cpu_percent", cpu_percent)
        log_metric("ram_percent", ram_percent)

if __name__ == '__main__':
    experiment = Experiment("templates/template_config.json")


        ##search = GridSearchCV(pipeline, param_grid=param_grid, cv=[(slice(None), slice(None))], verbose=True, )



# TODO: Generatisches Template!
