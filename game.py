import os

from game.setup import stat_library
from models import FilesConfig, LabelsConfig, Game


def main():
    stat_library(os.getcwd())
    input_folder = '/home/stefano/Work/sns/game/tests/input/small'
    output_folder = os.path.join(os.getcwd(), 'output/')

    files = FilesConfig(
        os.path.join(input_folder, 'inputs.dat'),
        os.path.join(input_folder, 'errors.dat'),
        os.path.join(input_folder, 'labels.dat'),
        output_folder,
        True
    )
    labels = LabelsConfig(
        ["G0", "n", "NH", "U", "Z"],
        ["Av", "fesc"]
    )

    driver = Game(files, 2, 10000, labels)
    driver.run()


if __name__ == "__main__":
    main()
