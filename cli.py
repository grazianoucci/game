import os
import json

from game.prepare import stat_library
from models import FilesConfig, LabelsConfig, Game


def read_config(file_path):
    with open(file_path, 'r') as reader:
        data = reader.read()  # read normally
        data = json.loads(data)  # then parse json
        return data


def get_config(file_path='config.json'):
    data = read_config(file_path)
    input_folder = data['input']
    output_folder = data['output']
    labels = data['labels']
    additional_files = data['additional files']
    n_repetitions = data['n repetitions']
    n_estimators = data['n estimators']

    files_config = FilesConfig(
        os.path.join(input_folder, 'lines.dat'),
        os.path.join(input_folder, 'errors.dat'),
        os.path.join(input_folder, 'labels.dat'),
        output_folder,
        additional_files
    )
    labels_config = LabelsConfig(
        labels
    )

    return files_config, labels_config, n_repetitions, n_estimators


def main():
    this_folder = os.path.dirname(os.path.realpath(__file__))
    stat_library(this_folder)  # searches for library files in this folder
    files_config, labels_config, n_repetitions, n_estimators = get_config()  # parses config

    n_cores = 4
    driver = Game(files, n_cores, n_repetitions, n_estimators, labels)
    driver.debug_params()
    # driver.run()


if __name__ == "__main__":
    main()
