# !/usr/bin/python3
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


""" GAME (GAlaxy Machine learning for Emission lines) """

import os

from game.models import Game
from benchmark.benchmark import simple_benchmark


def main():
    driver = Game(
        ["g0", "n", "NH", "U", "Z"],
        output_filename="output_ml.dat",
        manual_input=False,
        verbose=True
    )

    driver.run()
    driver.run_additional_labels(
        additional_features=["AV", "fesc"],
        labels_file=os.path.join(
            os.getcwd(),
            "library",
            "additional_labels.dat"
        ),
        output_filename="output_ml_additional.dat"
    )


if __name__ == "__main__":

    main()

    simple_benchmark()

