import os

from game.prepare import stat_library
from models import FilesConfig, LabelsConfig, Game


def main():
    stat_library('/opt/game/')  # todo config
    input_folder = '/opt/game/games/slack_22_02_2019/'  # todo config ??
    output_folder = os.path.join(os.getcwd(), 'output', 'out-test-1/')

    files = FilesConfig(
        os.path.join(input_folder, 'lines.dat'),  # todo was 'inputs'
        os.path.join(input_folder, 'errors.dat'),
        os.path.join(input_folder, 'labels.dat'),
        output_folder,
        True
    )
    labels = LabelsConfig(
        ["G0", "n", "NH", "U", "Z"],
        ["Av", "fesc"]
    )

    driver = Game(files, 3, 10000, labels)
    driver.run()


if __name__ == "__main__":
    main()
