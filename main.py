from utils.Experiment_Manager import Experiment_Manager

if __name__ == '__main__':
    experiment_paths = ["templates/template_config.json", "templates/template_config.json",
                        "templates/template_config.json", "templates/template_config.json"]

    experiment_manager = Experiment_Manager("test")
    experiment_manager.add_experiments(experiment_paths)
    experiment_manager.run_experiments()
