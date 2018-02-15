# !/usr/bin/python3
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


""" GAME (GAlaxy Machine learning for Emission lines) """

import os

from game.models import Game
from benchmark.benchmark import simple_benchmark,check_precision


def main(labels, additional_features, output_folder):
    output_filename = os.path.join(output_folder, "output_ml.dat")
    driver = Game(
        labels,
        inputs_file="input_big/inputs.dat",
        errors_file="input_big/errors.dat",
        labels_file="input_big/labels.dat",
        output_filename=output_filename,
        manual_input=False,
        verbose=True
    )
    driver.run()

    output_filename = os.path.join(output_folder, "output_ml_additional.dat")
    driver.run_additional_labels(
        additional_features=additional_features,
        labels_file=os.path.join(
            os.getcwd(),
            "library",
            "additional_labels.dat"
        ),
        output_filename=output_filename
    )


if __name__ == "__main__":
    main(["g0", "n", "NH", "U"], ["AV", "fesc"], os.getcwd())

    simple_benchmark()

    stat = check_precision()
    stat += check_precision(f_in="output_ml_additional.dat", log_data=False)
