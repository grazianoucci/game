# !/usr/bin/python3
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


""" GAME machine learning utilities """

import numpy as np
from sklearn.preprocessing import Normalizer


def realization(filename_int, filename_err, n_rep, mask):
    """
    :param filename_int: str
        Path to .txt input file
    :param filename_err: str
        Path to .txt error file
    :param n_rep: int
        Number of repetitions to generate
    :param mask: matrix
        Mask matrix
    :return: []
        Builds the array useful for PDFs
    """

    data = np.loadtxt(filename_int)[1:, :][mask]
    errors = np.loadtxt(filename_err)[1:, :][mask]

    # Be careful: in the first row of "errors" there are wavelenghts!
    # Find the position where error = -99 (i.e. the upper limits)
    mask_upper = [errors == -99]

    # Find the position where error = 0 (i.e. the errors associated to the
    # missing values)
    mask_miss = [errors == 0.0]

    # Assign a positive value where errors = -99 or 0 JUST TO COMPUTE
    # REPETITION WITHOUT ERRORS
    errors[errors == -99] = 0.1
    errors[errors == 0.0] = 0.1

    # Compute the "repetition matrix"
    repetition = np.random.normal(loc=np.tile(data, (n_rep, 1)),
                                  scale=np.tile(errors, (n_rep, 1)))

    # For upper limits assign to repetition a random number between 0 and
    # the value itself
    tiled_mask_upper = np.tile(mask_upper, (n_rep, 1))
    tiled_mask_miss = np.tile(mask_miss, (n_rep, 1))
    tiled_data = np.tile(data, (n_rep, 1))
    repetition[tiled_mask_upper[0]] = np.random.uniform(
        0, tiled_data[tiled_mask_upper[0]]
    )
    repetition[tiled_mask_miss[0]] = 0.0
    mms = Normalizer(norm="max")
    repetition = mms.fit_transform(repetition)
    return repetition
