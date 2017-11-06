# !/usr/bin/python3
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


""" GAME tools """

import os
import tarfile
import urllib

import numpy as np
from sklearn.preprocessing import Normalizer


def download_library(
        download_file,
        url="http://cosmology.sns.it/library_game/library.tar.gz"):
    """
    :param download_file: str
        Path where to download file
    :param url: str
        Url of library
    :return: void
        Downloads library to file
    """

    try:
        urllib.urlretrieve(
            url,
            filename=download_file
        )
    except Exception:
        if os.path.exists(download_file):
            os.remove(download_file)

        raise Exception("Cannot download library .tar file")


def create_library():
    """
    :return: void
        Creates necessary  library directory if not existing
    """

    lib_folder = os.path.join(
        os.getcwd(),
        "library/"
    )
    lib_file = os.path.join(
        lib_folder,
        "library.tar.gz"
    )
    check_file = os.path.join(
        lib_folder,
        "library.csv"
    )

    if not os.path.exists(check_file):
        if not os.path.exists(lib_folder):
            print "Creating library folder ..."
            os.makedirs(lib_folder)  # create necessary folders

        if not os.path.exists(lib_file):
            print "Downloading library ..."
            download_library(lib_file)  # download library

        if not os.path.exists(check_file):
            print "Extracting library files ..."
            tar = tarfile.open(lib_file)  # extract library
            tar.extractall()
            tar.close()


def create_output_directory(dir_path):
    """
    :param dir_path: str
        Path to output folder
    :return: void
        Creates folder if not existent
    """

    directory = os.path.dirname(dir_path)
    if not os.path.exists(directory):
        os.mkdir(directory)


def read_emission_line_file(filename_int):
    """
    :param filename_int: str
        Path to input file
    :return: tuple (matrix, min, max)
        Reading file containing
        emission line intensities (with normalization with respect to the
        maximum)
    """

    data = np.loadtxt(filename_int)
    mms = Normalizer(norm='max')
    data[1:, :] = mms.fit_transform(data[1:, :])
    lower = np.min(data[0, :])
    upper = np.max(data[0, :])
    return data, lower, upper


def read_library_file(filename_library):
    """
    :param filename_library: str
        Path to library file
    :return: tuple (array, numpy array)
        Reads file containing the library
    """

    # Reading the labels in the first row of the library
    lines = np.array(open('library/library.csv').readline().split(','))

    # Read the file containing the user-input labels
    input_labels = open(filename_library).read().splitlines()
    columns = []
    for element in input_labels:
        columns.append(np.where(lines == element)[0][0])

    # Add the labels indexes to columns
    columns.append(-5)  # Habing flux
    columns.append(-4)  # density
    columns.append(-3)  # column density
    columns.append(-2)  # ionization parameter
    columns.append(-1)  # metallicity
    array = np.loadtxt('library/library.csv', skiprows=2, delimiter=',',
                       usecols=columns)

    # Normalization of the library for each row with respect to the maximum
    # Be careful: do not normalize the labels!
    mms = Normalizer(norm='max')
    array[0:, :-5] = mms.fit_transform(array[0:, :-5])

    return array, np.array(input_labels)


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


def write_importances_files(dir_path, data, importances):
    """
    :param dir_path: str
        Path to output folder
    :param data: matrix
        Data
    :param importances: matrix
        Importances matrix
    :return: void
        Saves .txt files with importances data
    """
    np.savetxt(dir_path + 'output_feature_importances_G0.dat',
               np.vstack((data[0], importances[0::5, :])), fmt='%.5f')
    np.savetxt(dir_path + 'output_feature_importances_n.dat',
               np.vstack((data[0], importances[1::5, :])), fmt='%.5f')
    np.savetxt(dir_path + 'output_feature_importances_NH.dat',
               np.vstack((data[0], importances[2::5, :])), fmt='%.5f')
    np.savetxt(dir_path + 'output_feature_importances_U.dat',
               np.vstack((data[0], importances[3::5, :])), fmt='%.5f')
    np.savetxt(dir_path + 'output_feature_importances_Z.dat',
               np.vstack((data[0], importances[4::5, :])), fmt='%.5f')


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
