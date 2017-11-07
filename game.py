# !/usr/bin/python3
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


""" GAME (GAlaxy Machine learning for Emission lines) """

import os

from game.models import Game


def main():
    driver = Game(
        ["g0", "n", "NH", "U", "Z"],
        output_header="id_model mean[Log(G0)] median[Log(G0)]"
                      "sigma[Log(G0)] mean[Log(n)] median[Log(n)]"
                      "sigma[Log(n)] mean[Log(NH)] median[Log(NH)]"
                      "sigma[Log(NH)] mean[Log(U)] median[Log(U)]"
                      "sigma[Log(U)] mean[Log(Z)] median[Log(Z)]"
                      "sigma[Log(Z)]",
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
        output_header="id_model mean[Av] median[Av] sigma[Av] mean[fesc] "
                      "median[fesc] sigma[fesc]",
        output_filename="output_ml_additional.dat"
    )


if __name__ == "__main__":
    main()
