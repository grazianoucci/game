# !/usr/bin/python3
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


""" GAME classes models to predict and machine learn data """

import copy

import numpy as np
from sklearn.model_selection import cross_val_score


class Prediction(object):
    """ General prediction of model """

    def __init__(self, features, data, regressor):
        """
        :param features: [] of {}
            List of names of features
        :param data: matrix
            Data input
        :param regressor: sklearn regressor
            Regressor to predict
        """

        self.features = features
        self.data = data
        self.regr = regressor

    def generate_features(self):
        return [
            FeaturePrediction(
                feature["name"],
                feature["index"],
                copy.copy(self.regr)
            ) for feature in self.features
        ]

    def generate_importances_arrays(self):
        """
        :return: (generator of) numpy array
            Arrays filled with zeros
        """

        length = len(self.data[0])
        return [
            np.zeros(length) for _ in self.features
        ]


class FeaturePrediction(object):
    """ Prediction of single feature """

    def __init__(self, label, index_in_matrix, regressor):
        """
        :param label: str
            Identify feature with this literal
        :param index_in_matrix: int
            Respective index in matrix
        :param regressor: sklearn regressor
            Regressor to predict
        """

        self.label = label
        self.matrix_index = index_in_matrix
        self.regr = regressor

    def error_estimate(self, features_train, features_test, labels_train,
                       labels_test):
        """
        :param features_train: matrix
            X input
        :param features_test: matrix
            X test
        :param labels_train: matrix
            Y input
        :param labels_test: matrix
            Y test
        :return: float, [], float
            score, predictions, sigma
        """

        self.regr.fit(features_train, labels_train[:, self.matrix_index])
        prediction = self.regr.predict(features_test)
        sigma = np.std(
            np.double(labels_test[:, self.matrix_index]) - prediction
        )
        return {
            "true": np.double(labels_test),
            "pred": prediction,
            "sigma": sigma
        }

    def train(self, x_input, y_input):
        """
        :param x_input: matrix
            X input
        :param y_input: matrix
            Y input
        :return: model, [], float, float
            Copy of fit model, importances, score, std
        """

        model = self.regr.fit(x_input, y_input[:, self.matrix_index])  # Model
        score = cross_val_score(
            self.regr, x_input, y_input[:, self.matrix_index], cv=5
        )
        return {
            "model": copy.copy(model),
            "importance": model.feature_importances_,
            "score": np.mean(score),
            "std": np.std(score)
        }

    def predict(self, x_input):
        """
        :param x_input: matrix
            New data to base new prediction on
        :return: float
            Prediction
        """

        return self.regr.predict(x_input)
