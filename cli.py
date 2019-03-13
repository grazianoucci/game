# -*- coding: utf-8 -*-

import json
import os

from game.prepare import stat_library
from models import FilesConfig, LabelsConfig, Game


def read_config(file_path):
    with open(file_path, 'r') as reader:
        data = reader.read()  # read normally
        data = json.loads(data)  # then parse json
        return data


def get_config(file_path='config.json'):
    data = read_config(file_path)
    additional_files = data['OptionalFiles']
    n_repetitions = data['nRepetitions']
    n_estimators = data['nEstimators']

    files_config = FilesConfig(
        data['InputFile'],
        data['ErrorFile'],
        data['LabelsFile'],
        data['OutputFolder'],
        additional_files
    )
    labels_config = LabelsConfig(
        data['labels']
    )

    return files_config, labels_config, n_repetitions, n_estimators


def main():
    this_folder = os.path.dirname(os.path.realpath(__file__))
    stat_library(this_folder)  # searches for library files in this folder
    files_config, labels_config, n_repetitions, n_estimators = get_config()

    n_cores = 12
    driver = Game(files_config, n_cores, n_repetitions, n_estimators,
                  labels_config)
    driver.run()


if __name__ == "__main__":
    main()
