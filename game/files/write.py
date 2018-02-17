# !/usr/bin/python3
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


""" GAME write tools """

import os

import numpy as np

from game.files.read import FMT_PRINT


def write_optional_files(dir_path, labels, data):
    """
    :param dir_path: str
        Path to output folder
    :param labels: [] of str
        Features to write
    :param data: {} str -> matrix
        Dict like
        {
            "pred": predictions,
            "trues": trues,
            "pdf": matrix_ml
        }
    :return: void
        Saves .txt files with output data
    """

    for i, label in enumerate(labels):
        for key in data:
            file_name = os.path.join(
                dir_path,
                "output_" + str(key) + "_" + str(label) + ".dat"
            )
            np.savetxt(
                file_name,
                data[key][i::5, :],
                fmt=FMT_PRINT
            )


def write_importances_files(dir_path, labels, data, importances):
    """
    :param dir_path: str
        Path to output folder
    :param labels: [] of str
        Features to write
    :param data: matrix
        Data
    :param importances: matrix
        Importances matrix
    :return: void
        Saves .txt files with importances data
    """

    for label in labels:
        file_name = os.path.join(
            dir_path,
            "output_feature_importances_" + str(label) + ".dat"
        )
        np.savetxt(
            file_name,
            np.vstack((data[0], importances[0::5, :])),
            fmt=FMT_PRINT
        )


def write_models_info(dir_path, labels, data, list_of_lines):
    """
    :param dir_path: str
        Path to output folder
    :param labels: [] of str
        Features to write
    :param data: [] of {} str -> matrix
        List of dicts like
        {
            "lst": sigmas,
            "str": "Standard deviation of"
        }
    :param list_of_lines: []
        List of lines
    :return: void
        Saves to .dat file info about models used
    """

    with open(os.path.join(dir_path, "model_ids.dat"), "w+") as out_file:
        tot_rows = len(data[0]["lst"])
        for i in xrange(tot_rows):
            out_file.write("##############################\n")
            out_file.write("Id model: %d\n" % (i + 1))

            for d in data:
                for j, label in enumerate(labels):
                    out_file.write(
                        d["str"] + label + ": %.3f\n" % d["lst"][j]
                    )

            out_file.write("List of input lines:\n")
            out_file.write("%s\n" % list_of_lines[i])
        out_file.write("##############################\n")
