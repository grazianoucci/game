# !/usr/bin/python3
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


""" GAME read/write utilities """

import os

import numpy as np

FMT_PRINT = "%.5f"


def get_output_header(features):
    """
    :param features:
      features list
    :return header: string
      standard header for output
    """

    log_list = ["g0", "n", "NH", "U", "Z"]
    out_list = ['mean', 'median', 'sigma']

    out_head = ['id_model']
    for k in features:
        var_st = k
        if var_st in log_list:
            var_st = 'Log(' + var_st + ')'

        for out_st in out_list:
            out_head.append(out_st + '[' + var_st + ']')

    out_head = ' '.join(out_head)
    return out_head


def get_input_files():
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


def get_output(model_ids, matrix_ml, n_features, optional_files):
    """
    :param model_ids: []
        IDs of models
    :param matrix_ml: matrix
        Output matrix
    :param n_features: int
        Number of features to include
    :param optional_files: bool
        True iff you want to enable optional files generation
    :return: matrix
        Output matrix ready to be written to file
    """

    if optional_files:
        out = [model_ids]
        for i in range(n_features):
            out += [
                np.log10(np.mean(10 ** matrix_ml[:, i], axis=1)),
                np.log10(np.median(10 ** matrix_ml[:, i], axis=1)),
                np.std(matrix_ml[:, i], axis=1)
            ]
        return np.vstack(tuple(out)).T  # transpose
    else:
        return np.column_stack(
            (model_ids, matrix_ml)  # add model IDs to output matrix
        )


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
