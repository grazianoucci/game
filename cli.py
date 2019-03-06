import os

from game.prepare import stat_library
from models import FilesConfig, LabelsConfig, Game


def main():
    stat_library('/opt/game/game')
    input_folder = '/opt/game/games/slack_22_02_2019/'

    # stat_library('/home/stefano/Work/sns/game/code/game')
    # input_folder = '/home/stefano/Downloads/slack/'

    output_folder = '/opt/game/game/output/out-test-5'

    files = FilesConfig(
        os.path.join(input_folder, 'lines.dat'),  # todo was 'inputs'
        os.path.join(input_folder, 'errors.dat'),
        os.path.join(input_folder, 'labels.dat'),
        output_folder,
        additional_files=True
    )
    labels = LabelsConfig(
        ["g0", "n", "NH", "U", "Z", "AV", "fesc"],
    )

    driver = Game(files, 4, 10, labels)
    driver.run()


if __name__ == "__main__":
    main()
