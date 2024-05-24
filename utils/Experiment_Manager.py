from mlflow import set_tracking_uri, create_experiment, get_experiment_by_name
from Auto_ML_Testing.utils.Experiment import Experiment
from tqdm import tqdm
from uuid import uuid4
import memory_profiler
from datetime import datetime
import mlflow

# Function to create a colored string
# ANSI escape codes for colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"

log_file_name = f"logs/memory_profiler{str(uuid4())[:8]}_{datetime.now().strftime("%d_%m_%Y__%H_%M_%S")}.log"
fp=open(log_file_name,'w+')

def colored_text(color, text):
    return f"{color}{text}{GREEN}"


class Experiment_Manager:
    def __init__(self, experiment_name: str,
                 experiment_description="No description",
                 experiment_creator="Admin",
                 add_UID=True):

        self.experiment_description = experiment_description

        if add_UID:
            # TODO: Not ideal, should be improved in the future
            self.experiment_name = experiment_name + "_" + datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
            self.experiment_id = create_experiment(self.experiment_name)


        else:
            self.experiment_name = experiment_name

            if get_experiment_by_name(experiment_name) is None:
                self.experiment_id = create_experiment(experiment_name)
            else:
                self.experiment_id = dict(get_experiment_by_name(experiment_name))['experiment_id']

        self.experiment_creator = experiment_creator
        self.experiments = []

    def add_experiments(self, list_of_configs: list):
        for config in list_of_configs:
            self.experiments.append(Experiment(config, self.experiment_id))

    @memory_profiler.profile(stream=fp)
    def run_experiments(self):
        log_name = f"========== MEMORY_PROFILE FOR {str(self.experiments)} ========== \n "
        fp.write(log_name)

        print(f"Your {len(self.experiments)} experiments in the {self.experiment_name}-context will now be executed")
        for experiment in tqdm(self.experiments, desc=colored_text(GREEN, "Experimenting in progress")):
            experiment.run_experiment()
        print("All your experiments have been successfully executed. The results can be found in the MLFlow UI")


