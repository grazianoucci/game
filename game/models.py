# !/usr/bin/python3
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


""" GAME classes models to predict and machine learn data """

import copy
import os
import time
from functools import partial

import numpy as np
from sklearn import tree
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Normalizer

import game.utils as utils
from game.alg import game
from game.rw import write_optional_files, write_importances_files, \
    write_models_info, get_input_files, get_output, FMT_PRINT, \
    get_output_header


class Prediction(object):
    """ General prediction of model """

    def __init__(self, features, data, regressor):
        """
        :param features: {} str -> int
            Identify feature with this literal and index in matrix
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
                feature,
                self.features[feature],
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


class Game(object):
    """ GAlaxy Machine learning for Emission lines """

    REGRESSOR = AdaBoostRegressor(
        tree.DecisionTreeRegressor(
            criterion="mse",
            splitter="best",
            max_features=None
        ),
        n_estimators=2,
        random_state=0
    )  # algorithm for Machine Learning (ref1:
    # http://adsabs.harvard.edu/abs/2017MNRAS.465.1144U(

    # Testing, test_size is the percentage of the library to use as testing
    # set to determine the PDFs
    TEST_SIZE = 0.10

    INTRO = "--------------------------------------------------------\n" + \
            "--- GAME (GAlaxy Machine learning for Emission lines) --\n" + \
            "------- see Ucci G. et al. (2017a,b) for details -------\n" + \
            "--------------------------------------------------------\n\n" + \
            "ML Algorithm: AdaBoost with Decision Trees as base learner."

    LIBRARY_FOLDER = os.path.join(
        os.getcwd(),
        "library/"
    )
    LABELS_FILE = os.path.join(
        LIBRARY_FOLDER,
        "library.csv"
    )

    def __init__(self, features, manual_input, verbose,
                 output_filename, n_repetition=10, input_folder=os.getcwd(),
                 output_folder=os.path.join(os.getcwd(), "output")):
        """
        :param features: [] of str
            List of features to predict
        """

        self.features = features

        # user interaction
        self.user_input = manual_input
        self.verbose = verbose
        self.output_folder = output_folder
        self.optional_files = False

        # input files
        self.filename_int = os.path.join(
            input_folder,
            "input/inputs_game_test.dat"
        )
        self.filename_err = os.path.join(
            input_folder,
            "input/errors_game_test.dat"
        )
        self.filename_library = os.path.join(
            input_folder,
            "input/labels_game_test.dat"
        )

        # data
        self.n_repetition         = n_repetition
        self.data                 = None
        self.labels               = None
        self.prediction_features  = None
        self.test_size_limit      = 0
        self.output               = None
        self.line_labels          = None
        self.n_processes          = 2
        self.results              = None
        self.output_sigmas_header = "Standard deviation of log "
        self.output_scores_header = "Cross-validation score for "
        # self.output_header        = get_output_header(self.features)
        self.output_header = get_output_header(features)
        self.output_filename      = os.path.join(
            self.output_folder,
            output_filename
        )

    def start(self):
        """
        :return: void
            Prints to stdout intro and asks for which files to use
        """

        utils.create_library(self.LIBRARY_FOLDER, self.LABELS_FILE)
        if not os.path.exists(self.output_folder):  # prepare output folder
            os.makedirs(self.output_folder)

        if self.verbose:
            print self.INTRO

        if self.user_input:
            self.filename_int, self.filename_err, self.filename_library = \
                get_input_files()

            self.optional_files = str(raw_input(
                "Do you want to create the optional files [y/n]?: "
            )).strip() == "y"  # optional files

            self.n_processes = int(
                raw_input("Choose the number of processors: ")
            )

    def parse_input_files(self):
        """
        :return: void
            Parses input files and saves data in object
        """

        self.data = np.loadtxt(self.filename_int)
        mms = Normalizer(norm="max")
        self.data[1:, :] = mms.fit_transform(self.data[1:, :])
        self.output, self.line_labels = \
            self.parse_library_file()

    def parse_library_file(self):
        """
        :return: tuple (array, numpy array)
            Reads file containing the library
        """

        # Reading the labels in the first row of the library
        lines = np.array(open(self.LABELS_FILE).readline().split(","))

        # Read the file containing the user-input labels
        input_labels = open(self.filename_library).read().splitlines()
        columns = []
        for element in input_labels:
            columns.append(np.where(lines == element)[0][0])

        # Add the labels indexes to columns
        columns.append(-5)  # Habing flux
        columns.append(-4)  # density
        columns.append(-3)  # column density
        columns.append(-2)  # ionization parameter
        columns.append(-1)  # metallicity
        array = np.loadtxt(
            self.LABELS_FILE, skiprows=2, delimiter=",", usecols=columns
        )

        # Normalization of the library for each row with respect to the maximum
        # Be careful: do not normalize the labels!
        mms = Normalizer(norm="max")
        array[0:, :-5] = mms.fit_transform(array[0:, :-5])

        return array, np.array(input_labels)

    def determine_models(self):
        """
        :return: tuple
            Determination of unique models based on
            the missing data. In this case missing data are values with zero
            intensities. Be careful because the first row in data there are
            wavelengths!
        """

        initial = [self.data[1:] != 0][0]
        models = np.zeros(len(initial))
        mask = np.where((initial == initial[0]).all(axis=1))[0]
        models[mask] = 1
        check = True
        i = 2

        while check:
            if not models[models == 0]:
                check = False
            else:
                mask = np.where(
                    (
                        initial == initial[np.argmax(models == 0)]
                    ).all(axis=1))[0]
                models[mask] = i
                i += 1
        return initial, models, np.unique(models)

    def run(self, additional_labels_file=None):
        """
        :param additional_labels_file: str
            Path to file containing additional labels
        :return: void
            Runs predictions and writes results
        """

        self.start()

        if self.verbose:
            print "\nProgram started..."

        if additional_labels_file:
            print ""
            print "Running GAME with additional labels...\n"
        else:
            print ""
            print "Running GAME with default labels...\n"

        self.parse_input_files()
        initial, models, unique_id = self.determine_models()

        if self.verbose:
            print "# of input  models                     :", \
                len(self.data[1:])
            print "# of unique models for Machine Learning:", int(
                np.max(unique_id))
            print "\nStarting Machine Learning algorithm for " \
                  + ", ".join(self.features) + " labels... "

        timer = time.time()  # TIMER start

        # Definition of features and labels for Machine Learning. Searching
        # for values of the physical properties (for metallicity logarithm)
        if self.labels is None:
            self.prediction_features = self.output[:, : -len(self.features)]
            self.labels = np.double(
                self.output[:,
                len(self.output[0]) - len(self.features): len(self.output[0])
                ]
            )
            self.labels[:, -1] = np.log10(self.labels[:, -1])
            self.test_size_limit = int(
                (1. - self.TEST_SIZE) * len(self.prediction_features)
            )

        if additional_labels_file:
            self.labels[:, -2:] = np.loadtxt(additional_labels_file)

            # This code is inserted in order to work with logarithms!
            # If there is a zero, we substitute it with 1e-9
            self.labels[self.labels[:, -2] == 0, -2] = 1e-9
            self.labels[self.labels[:, -1] == 0, -1] = 1e-9
            self.labels[:, -2] = np.log10(self.labels[:, -2])
            self.labels[:, -1] = np.log10(self.labels[:, -1])
            to_predict = Prediction(
                {
                    feature: i + 3 for i, feature in enumerate(self.features)
                },  # additional labels have +3 offset
                self.data,
                self.REGRESSOR
            )
        else:
            to_predict = Prediction(
                {
                    feature: i for i, feature in enumerate(self.features)
                },
                self.data,
                self.REGRESSOR
            )

        labels_train = self.labels[:self.test_size_limit, :]
        labels_test = self.labels[self.test_size_limit:, :]

        algorithm = partial(
            game,
            models=models, unique_id=unique_id, initial=initial,
            limit=self.test_size_limit,
            features=self.prediction_features, labels_train=labels_train,
            labels_test=labels_test, labels=self.labels,
            line_labels=self.line_labels, filename_int=self.filename_int,
            filename_err=self.filename_err, n_repetition=self.n_repetition,
            optional_files=self.optional_files,
            to_predict=to_predict
        )
        self.results = utils.run_parallel(
            algorithm, self.n_processes, unique_id
        )
        self.results = list(self.results[0])
        timer = time.time() - timer  # TIMER end
        if self.verbose:
            print "Elapsed seconds for ML:", timer
            print "\nWriting output files..."

        self.write_results()

    def run_additional_labels(self, additional_features, labels_file,
                              output_filename):
        """
        :param additional_features: [] of str
            List of features to predict
        :param labels_file: str
            Path to file containing additional labels
        :param output_filename: str
            Name of output file
        :return: void
            Runs predictions and writes results
        """

        self.features = additional_features
        self.output_header = get_output_header(self.features)
        self.output_filename = output_filename
        self.run(additional_labels_file=labels_file)

    def parse_results(self):
        """
        :return: tuple of []
            Rearrange based on the find_ids indexes
        """

        sigmas = np.array(self.results[0])
        scores = np.array(self.results[1])
        list_of_lines = np.array(self.results[2])
        find_ids = list(self.results[3])
        temp_model_ids = list(self.results[4])
        tmp_matrix_ml = np.array(self.results[5])
        importances = np.array(self.results[6])
        trues = np.array(self.results[7])
        predictions = np.array(self.results[8])

        # Rearrange the matrix based on the find_ids indexes
        if not self.optional_files:
            tmp_matrix_ml = tmp_matrix_ml.reshape(
                len(self.data[1:]), 3 * len(self.features)
            )

        matrix_ml = np.zeros(shape=tmp_matrix_ml.shape)
        for i in xrange(len(matrix_ml)):
            matrix_ml[find_ids[i], :] = tmp_matrix_ml[i, :]

        model_ids = np.zeros(len(temp_model_ids))
        for i in xrange(len(matrix_ml)):
            model_ids[find_ids[i]] = temp_model_ids[i]

        return sigmas, scores, list_of_lines, model_ids, matrix_ml, \
               importances, predictions, trues

    def write_results(self):
        sigmas, scores, list_of_lines, model_ids, matrix_ml, \
        importances, predictions, trues = self.parse_results()

        # Write information on different models
        write_models_info(
            self.output_folder, self.features, [
                {
                    "lst": sigmas,
                    "str": self.output_sigmas_header
                },
                {
                    "lst": scores,
                    "str": self.output_scores_header
                },
            ], list_of_lines
        )

        np.savetxt(
            os.path.join(
                self.output_folder,
                self.output_filename
            ),
            get_output(
                model_ids, matrix_ml, len(self.features), self.optional_files
            ),  # Outputs relative to the Machine Learning determination
            header=self.output_header,
            fmt=FMT_PRINT
        )

        # Outputs with the feature importances
        write_importances_files(
            self.output_folder, self.features, self.data, importances
        )

        # Optional files
        if self.optional_files:
            write_optional_files(
                self.output_folder, self.features,
                {
                    "pred": predictions,
                    "trues": trues,
                    "pdf": matrix_ml
                }
            )
