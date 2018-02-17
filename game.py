# !/usr/bin/python3
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


""" GAME (GAlaxy Machine learning for Emission lines) """

import os

from game.models import Game
from benchmark.benchmark import simple_benchmark,check_precision


def main(features, additional_features, output_folder):
    driver = Game(
        features,
        "inputs/big/inputs.dat",
        "inputs/big/errors.dat",
        "inputs/big/labels.dat",
        output_folder,
        verbose=True
    )
    driver.run()

    if additional_features:
        output_filename =\
            os.path.join(output_folder, "output_ml_additional.dat")
        driver.run_additional_labels(
            additional_features,
            output_filename,
            os.path.join(
                os.getcwd(),
                "library",
                "additional_labels.dat"
            )
        )


if __name__ == "__main__":
    main(["g0", "n", "NH", "U"], ["AV", "fesc"], os.getcwd())

    simple_benchmark()

    stat = check_precision()
    stat += check_precision(f_in="output_ml_additional.dat", log_data=False)
