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
        'Insert input file name (line intensities): '
    ).strip()
    errors = raw_input(
        'Insert input file name (errors on line intensities): '
    ).strip()
    labels = raw_input(
        'Insert name of file containing the labels: '
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
                          filename='library/additional_labels.dat'):
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


def write_output_files(dir_path, preds, trues, matrix_ml):
    """
    :param dir_path: str
        Path to output folder
    :param preds: matrix
        Predictions
    :param trues: matrix
        True values
    :param matrix_ml: matrix
        ML matrix
    :return: void
        Saves .txt files with output data
    """

    # This writes down the output relative to the predicted and true
    # value of the library
    np.savetxt(dir_path + 'output_pred_G0.dat', preds[0::5, :], fmt='%.5f')
    np.savetxt(dir_path + 'output_pred_n.dat', preds[1::5, :], fmt='%.5f')
    np.savetxt(dir_path + 'output_pred_NH.dat', preds[2::5, :], fmt='%.5f')
    np.savetxt(dir_path + 'output_pred_U.dat', preds[3::5, :], fmt='%.5f')
    np.savetxt(dir_path + 'output_pred_Z.dat', preds[4::5, :], fmt='%.5f')
    np.savetxt(dir_path + 'output_true_G0.dat', trues[0::5, :], fmt='%.5f')
    np.savetxt(dir_path + 'output_true_n.dat', trues[1::5, :], fmt='%.5f')
    np.savetxt(dir_path + 'output_true_NH.dat', trues[2::5, :], fmt='%.5f')
    np.savetxt(dir_path + 'output_true_U.dat', trues[3::5, :], fmt='%.5f')
    np.savetxt(dir_path + 'output_true_Z.dat', trues[4::5, :], fmt='%.5f')

    # This writes down the output relative to the PDFs of the physical
    # properties
    np.savetxt(dir_path + 'output_pdf_G0.dat', matrix_ml[:, 0], fmt='%.5f')
    np.savetxt(dir_path + 'output_pdf_n.dat', matrix_ml[:, 1], fmt='%.5f')
    np.savetxt(dir_path + 'output_pdf_NH.dat', matrix_ml[:, 2], fmt='%.5f')
    np.savetxt(dir_path + 'output_pdf_U.dat', matrix_ml[:, 3], fmt='%.5f')
    np.savetxt(dir_path + 'output_pdf_Z.dat', matrix_ml[:, 4], fmt='%.5f')


def write_optional_files(dir_path, preds, trues, matrix_ml):
    """
    :param dir_path: str
        Path to output folder
    :param preds: matrix
        Predictions
    :param trues: matrix
        True values
    :param matrix_ml: matrix
        ML matrix
    :return: void
        Saves .txt files with optional data
    """

    # This writes down the output relative to the predicted and true
    # value of the library
    np.savetxt(dir_path + 'output_pred_Av.dat', preds[0::2, :], fmt='%.5f')
    np.savetxt(dir_path + 'output_pred_fesc.dat', preds[1::2, :],
               fmt='%.5f')
    np.savetxt(dir_path + 'output_true_Av.dat', trues[0::2, :], fmt='%.5f')
    np.savetxt(dir_path + 'output_true_fesc.dat', trues[1::2, :],
               fmt='%.5f')
    # This writes down the output relative to the PDFs of the physical
    # properties
    np.savetxt(dir_path + 'output_pdf_Av.dat', matrix_ml[:, 0], fmt='%.5f')
    np.savetxt(dir_path + 'output_pdf_fesc.dat', matrix_ml[:, 1],
               fmt='%.5f')


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


def write_models_info(dir_path, sigmas, scores, list_of_lines):
    """
    :param dir_path: str
        Path to output folder
    :param sigmas: matrix
        Sigmas
    :param scores: matrix
        Scores
    :param list_of_lines: []
        List of lines
    :return: void
        Saves to .dat file info about models used
    """

    with open(dir_path + 'model_ids.dat', 'w+') as out_file:
        for i in xrange(len(sigmas)):
            out_file.write('##############################\n')
            out_file.write('Id model: %d\n' % (i + 1))
            out_file.write('Standard deviation of log(G0): %.3f\n' % sigmas[i, 0])
            out_file.write('Standard deviation of log(n):  %.3f\n' % sigmas[i, 1])
            out_file.write('Standard deviation of log(NH): %.3f\n' % sigmas[i, 2])
            out_file.write('Standard deviation of log(U):  %.3f\n' % sigmas[i, 3])
            out_file.write('Standard deviation of log(Z):  %.3f\n' % sigmas[i, 4])
            out_file.write('Cross-validation score for G0: %.3f +- %.3f\n' % (
                scores[i, 1], 2. * scores[i, 2]))
            out_file.write('Cross-validation score for n:  %.3f +- %.3f\n' % (
                scores[i, 3], 2. * scores[i, 4]))
            out_file.write('Cross-validation score for NH: %.3f +- %.3f\n' % (
                scores[i, 5], 2. * scores[i, 6]))
            out_file.write('Cross-validation score for U:  %.3f +- %.3f\n' % (
                scores[i, 7], 2. * scores[i, 8]))
            out_file.write('Cross-validation score for Z:  %.3f +- %.3f\n' % (
                scores[i, 9], 2. * scores[i, 10]))
            out_file.write('List of input lines:\n')
            out_file.write('%s\n' % list_of_lines[i])
        out_file.write('##############################\n')
