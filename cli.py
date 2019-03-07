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
    stat_library('/opt/game/game')
    input_folder = '/opt/game/games/slack_22_02_2019/'
    output_folder = '/opt/game/game/output/out-test-10/'

    files = FilesConfig(
        os.path.join(input_folder, 'lines.dat'),  # todo was 'inputs'
        os.path.join(input_folder, 'errors.dat'),
        os.path.join(input_folder, 'labels.dat'),
        output_folder,
        additional_files=True
    )
    labels = LabelsConfig(
        ["g0", "n", "NH", "U", "Z", "Av", "fesc"],
    )

    driver = Game(files, 4, 5, labels)
    driver.run()


if __name__ == "__main__":
    main()
