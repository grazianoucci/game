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


def create_library_folder():
    """
    :return: void
        Creates necessary  library directory if not existing
    """

    dir_path = 'library/'
    directory = os.path.dirname(dir_path)
    if not os.path.exists(directory):
        urllib.urlretrieve(
            "http://cosmology.sns.it/library_game/library.tar.gz",
            filename="library.tar.gz")
        tar = tarfile.open("library.tar.gz")
        tar.extractall()
        tar.close()
        os.remove("library.tar.gz")


def create_output_directory(dir_path):
    """
    :param dir_path: str
        Path to output folder
    :return: void
        Creates folder if not existent
    """

    directory = os.path.dirname(dir_path)
    if not os.path.exists(directory):
        os.mkdir(directory)  # TODO why not mkdirs ??


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
