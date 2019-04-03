# -*- coding: utf-8 -*-

import os
import re
import tarfile
import urllib

import numpy as np
from sklearn.preprocessing import Normalizer

LIB_URL = "http://cosmology.sns.it/library_game/library.tar.gz"
JUST_ALPHANUM = re.compile("[^a-zA-Z0-9.]*")


def stat_library(folder):
    dir_path = os.path.join(folder, 'library/')
    library_file = os.path.join(dir_path, 'library.csv')

    if not os.path.exists(library_file):
        directory = os.path.dirname(dir_path)

        try:
            os.stat(directory)
        except:
            urllib.urlretrieve(LIB_URL, filename="library.tar.gz")
            tar = tarfile.open("library.tar.gz")
            tar.extractall()
            tar.close()
            os.remove("library.tar.gz")


def read_emission_line_file(filename_int):
    data = np.loadtxt(filename_int)
    mms = Normalizer(norm='max')
    data[1:, :] = mms.fit_transform(data[1:, :])
    lower = np.min(data[0, :])
    upper = np.max(data[0, :])
    return data, lower, upper


def read_library_file(library_csv, filename_library):
    # Reading the labels in the first row of the library
    lines = np.array(open(library_csv).readline().split(','))
    lines = [
        JUST_ALPHANUM.sub('', line).lower()
        for line in lines
    ]
    lines = np.array(lines)  # re-convert to numpy array

    # Read the file containing the user-input labels
    input_labels = open(filename_library).read().splitlines()

    columns = []
    for element in input_labels:
        element = ' '.join(element.split(' ')[:-1])
        element = JUST_ALPHANUM.sub('', element).lower()
        columns.append(np.where(lines == element)[0][0])

    # Add the labels indexes to columns
    columns.append(-5)  # Habing flux
    columns.append(-4)  # density
    columns.append(-3)  # column density
    columns.append(-2)  # ionization parameter
    columns.append(-1)  # metallicity
    array = np.loadtxt(library_csv, delimiter=',', skiprows=2, usecols=columns)

    # Normalization of the library for each row with respect to the maximum
    # Be careful: do not normalize the labels!
    mms = Normalizer(norm='max')
    array[0:, :-5] = mms.fit_transform(array[0:, :-5])
    return array, np.array(input_labels)
