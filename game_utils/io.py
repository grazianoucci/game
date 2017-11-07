# !/usr/bin/python3
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


""" GAME read/write utilities """

import os

import numpy as np


def get_files_from_user():
    """
    :return: tuple (str, str, str)
        Files for (line intensities, errors, labels)
    """

    line = raw_input(
        "Insert input file name (line intensities): "
    ).strip()
    errors = raw_input(
        "Insert input file name (errors on line intensities): "
    ).strip()
    labels = raw_input(
        "Insert name of file containing the labels: "
    ).strip()

    return line, errors, labels


def get_write_output(model_ids, matrix_ml):
    """ TODO add docs
    :param model_ids:
    :param matrix_ml:
    :return: np.vstack
    """

    return np.vstack(
        (
            model_ids, np.log10(np.mean(10 ** matrix_ml[:, 0], axis=1)),
            np.log10(np.median(10 ** matrix_ml[:, 0], axis=1)),
            np.std(matrix_ml[:, 0], axis=1),
            np.log10(np.mean(10 ** matrix_ml[:, 1], axis=1)),
            np.log10(np.median(10 ** matrix_ml[:, 1], axis=1)),
            np.std(matrix_ml[:, 1], axis=1),
            np.log10(np.mean(10 ** matrix_ml[:, 2], axis=1)),
            np.log10(np.median(10 ** matrix_ml[:, 2], axis=1)),
            np.std(matrix_ml[:, 2], axis=1),
            np.log10(np.mean(10 ** matrix_ml[:, 3], axis=1)),
            np.log10(np.median(10 ** matrix_ml[:, 3], axis=1)),
            np.std(matrix_ml[:, 3], axis=1),
            np.log10(np.mean(10 ** matrix_ml[:, 4], axis=1)),
            np.log10(np.median(10 ** matrix_ml[:, 4], axis=1)),
            np.std(matrix_ml[:, 4], axis=1)
        )
    ).T  # transpose


def get_additional_labels(labels, limit,
                          filename="library/additional_labels.dat"):
    """
    :param labels: matrix
        Initial labels
    :param limit: int
        Limit
    :param filename: str
        Path to input file
    :return: tuple (matrix, matrix, matrix)
        Definition of additional labels for Machine Learning
    """

    labels[:, -2:] = np.loadtxt(filename)

    # This code is inserted in order to work with logarithms!
    # If there is a zero, we substitute it with 1e-9
    labels[labels[:, -2] == 0, -2] = 1e-9
    labels[labels[:, -1] == 0, -1] = 1e-9
    labels[:, -2] = np.log10(labels[:, -2])
    labels[:, -1] = np.log10(labels[:, -1])

    # Reading labels in the library corresponding to the line
    labels_train = labels[:limit, :]
    labels_test = labels[limit:, :]

    return labels, labels_train, labels_test


def write_output_files(dir_path, labels, data):
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
                fmt="%.5f"
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
            fmt="%.5f"
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
                        d["str"] + label + ": %.3f\n" % d["lst"][i, j]
                    )

            out_file.write("List of input lines:\n")
            out_file.write("%s\n" % list_of_lines[i])
        out_file.write("##############################\n")
