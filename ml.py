# !/usr/bin/python3
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


""" GAME machine learning utilities """

import copy

import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Normalizer


def realization(filename_int, filename_err, n_rep, mask):
    """
    :param filename_int: str
        Path to .txt input file
    :param filename_err: str
        Path to .txt error file
    :param n_rep: int
        Number of repetitions to generate
    :param mask: TODO find type
        TODO find purpose
    :return: []
        Builds the array useful for PDFs
    """

    data = np.loadtxt(filename_int)[1:, :][mask]
    errors = np.loadtxt(filename_err)[1:, :][mask]

    # Be careful: in the first row of 'errors' there are wavelenghts!
    # Find the position where error = -99 (i.e. the upper limits)
    mask_upper = [errors == -99]

    # Find the position where error = 0 (i.e. the errors associated to the
    # missing values)
    mask_miss = [errors == 0.0]

    # Assign a positive value where errors = -99 or 0 JUST TO COMPUTE
    # REPETITION WITHOUT ERRORS
    errors[errors == -99] = 0.1
    errors[errors == 0.0] = 0.1

    # Compute the 'repetition matrix'
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
    mms = Normalizer(norm='max')
    repetition = mms.fit_transform(repetition)
    return repetition


def determination_models(data):
    """
    :param data: []
        Data list
    :return: TODO find types
        Determination of number of different models
    """

    initial = [data != 0][0]
    models = np.zeros(len(initial))
    mask = np.where((initial == initial[0]).all(axis=1))[0]
    models[mask] = 1
    check = True
    i = 2

    while check:
        if len(models[models == 0]) == 0:
            check = False
        else:
            mask = np.where(
                (
                    initial == initial[np.argmax(models == 0)]
                ).all(axis=1))[0]
            models[mask] = i
            i += 1
    return initial, models, np.unique(models)


def error_estimation(feat_train, feat_test, lab_train, lab_test, ml_regressor):
    """
    :param feat_train: matrix
        Fit train
    :param feat_test: array
        Fit data
    :param lab_train: matrix
        Train data
    :param lab_test: array
        Test data
    :param ml_regressor: TODO find type
        Regressor used
    :return: tuple (float, TODO find type, float)
        Estimate error of ML
    """

    ml_regressor.fit(feat_train, lab_train)
    y = ml_regressor.predict(feat_test)
    sigma = np.std(np.double(lab_test) - y)

    return np.double(lab_test), y, sigma


def machine_learning(feat, lab, physical_p, ml_regressor):
    """
    :param feat: TODO
        TODO
    :param lab: TODO
        TODO
    :param physical_p: TODO
        TODO
    :param ml_regressor: TODO
        TODO
    :return:
        Function for Machine Learning
    """

    model = ml_regressor.fit(feat, lab[:, physical_p])  # Model
    importances = model.feature_importances_  # Feature importances

    # Cross-validation score
    score = cross_val_score(ml_regressor, feat, lab[:, physical_p], cv=5)
    return copy.copy(model), importances, np.mean(score), np.std(score)


def find_features_from_dict(to_predict):
    """
    :param to_predict: {}
        Dict with values to predict
    :return: []
        List with features to predict
    """

    output = []
    if "AV" and "fesc" in to_predict:
        keys = ["AV", "fesc"]
    else:
        keys = ["g0", "n", "NH", "U", "Z"]

    for k in keys:
        if k in to_predict in to_predict:
            output.append(to_predict[k])

    for k in keys:
        if ("importances_" + k) in to_predict in to_predict:
            output.append(to_predict[("importances_" + k)])

    return output


def create_importances_from_data(data):
    """
    :param data: matrix
        Data input
    :return: (generator of) numpy array
        Arrays filled with zeros
    """

    length = len(data[0])
    for _ in range(len(["g0", "n", "NH", "U", "Z"])):
        yield np.zeros(length)


def initialize_arrays_from_data(data, n_repetition):
    """
    :param data: matrix
        Data input
    :param n_repetition: int
        Number of repetition
    :return: (generator of) numpy array
        Arrays filled with zeros
    """

    length = len(data[1:])
    for _ in range(len(["g0", "n", "NH", "U", "Z"])):
        yield np.zeros(shape=(length, n_repetition))
