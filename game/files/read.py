# !/usr/bin/python3
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


""" GAME read tools """

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
