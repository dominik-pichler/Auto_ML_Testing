from mlflow import set_tracking_uri, create_experiment, get_experiment_by_name


class Experiment_Manager:
    def __init__(self,experiment_name:str, experiment_description="No description",experiment_creator="Admin"):
        self.experiment_name = experiment_name
        self.experiment_description = experiment_description
        self.experiment_id = (
            create_experiment(experiment_name, experiment_description)
            if get_experiment_by_name(experiment_name) is None
            else get_experiment_by_name(experiment_name).experiment_id
        )
        self.experiment_creator = experiment_creator


if __name__ == '__main__':
    testing_experiment = Experiment_Manager("Test")
    print(testing_experiment.experiment_id)