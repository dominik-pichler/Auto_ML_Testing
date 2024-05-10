import json
from pytest import raises

@staticmethod
def read_json_file(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

class TestExperiment():

    def faultyPath_readJSON_shouldThrowException(self):
        # Provide a non-existent JSON file path
        json_file = "nonexistent_file.json"

        # Call the function and expect it to raise FileNotFoundError
        with raises(FileNotFoundError):
            read_json_file(json_file)
    def test_jsonFull_init_shouldFindEverything(self):
        # Path to your JSON file
        json_file = "test_template_config.json"

        # Call the function to read JSON file
        data = read_json_file(json_file)

        # Assert the expected values
        assert data["data_paths"]["train_x"] == "path/to/train_x.csv"
        assert data["data_paths"]["train_y"] == "path/to/train_y.csv"
        assert data["data_paths"]["test_x"] == "path/to/test_x.csv"
        assert data["data_paths"]["test_y"] == "path/to/test_y.csv"
        assert data["preprocessing_steps"] == []
        assert data["model"] is None
        assert data["HPOptimizer_type"] is None
        assert data["param_grid"] == []
        assert data["tracking_uri"] == "http://127.0.0.1:5000"
        assert data["experiment_description"] == ""
        assert data["experiment_name"] == "Description"

    def test_emptyJson_init_initShouldFail(self):
        json_file = "test_empty_template_config.json"

        # Call the function to read the empty JSON file
        data = read_json_file(json_file)

        # Assert that the data is None (indicating failure to read JSON)
        assert data is None