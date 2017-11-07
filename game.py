# !/usr/bin/python3
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


""" GAME (GAlaxy Machine learning for Emission lines) """

import multiprocessing
import os
import tarfile
import time
import urllib
from functools import partial
from itertools import chain

import numpy as np
from sklearn import tree
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import Normalizer

from game_utils.io import write_optional_files, write_importances_files, \
    write_models_info, get_input_files, get_output
from game_utils.ml import realization, error_estimate, machine_learn


class Prediction(object):
    """ General prediction of model """

    def __init__(self, features, data):
        """
        :param features: [] of str
            Data to predict
        :param data: matrix
            Data input
        """

        self.features = features
        self.data = data

    def are_additional_labels(self):
        """
        :return: bool
            True iff "AV" and "fesc" features to be predicted
        """

        return ("AV" in self.features) and ("fesc" in self.features)

    def generate_features_arrays(self, n_repetition):
        """
        :param n_repetition: int
            Number of repetition
        :return: (generator of) numpy array
            Arrays filled with zeros
        """

        length = len(self.data[1:])
        for _ in self.features:
            yield np.zeros(shape=(length, n_repetition))

    def generate_importances_arrays(self):
        """
        :return: (generator of) numpy array
            Arrays filled with zeros
        """

        length = len(self.data[0])
        for _ in self.features:
            yield np.zeros(length)


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

    def __init__(self, features, manual_input, verbose, output_header,
                 output_filename, n_repetition=10000, input_folder=os.getcwd(),
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
        self.n_repetition = n_repetition
        self.data = None
        self.labels = None
        self.prediction_features = None
        self.test_size_limit = 0
        self.output = None
        self.line_labels = None
        self.n_processes = 2
        self.results = None
        self.output_sigmas_header = "Standard deviation of log "
        self.output_scores_header = "Cross-validation score for "
        self.output_header = output_header
        self.output_filename = os.path.join(
            self.output_folder,
            output_filename
        )

    def start(self):
        """
        :return: void
            Prints to stdout intro and asks for which files to use
        """

        self.create_library()

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
        :return: tuple (TODO types) Determination of unique models based on
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

        labels_train = self.labels[:self.test_size_limit, :]
        labels_test = self.labels[self.test_size_limit:, :]
        to_predict = Prediction(
            self.features,
            self.data
        )

        algorithm = partial(
            game,
            i=1,
            models=models, unique_id=unique_id, initial=initial,
            limit=self.test_size_limit,
            features=self.prediction_features, labels_train=labels_train,
            labels_test=labels_test, labels=self.labels,
            regr=self.REGRESSOR, line_labels=self.line_labels,
            filename_int=self.filename_int,
            filename_err=self.filename_err,
            n_repetition=self.n_repetition, optional_files=self.optional_files,
            to_predict=to_predict
        )
        # self.results = self.run_parallel(algorithm, self.n_processes,
        #                                  unique_id)
        self.results = [algorithm()]

        timer = time.time() - timer  # TIMER end
        if self.verbose:
            print "Elapsed seconds for ML:", timer
            print "\nWriting output files for the default labels..."

        self.write_results(unique_id)

    def run_additional_labels(self, additional_features, labels_file,
                              output_header, output_filename):
        """
        :param additional_features: [] of str
            List of features to predict
        :param labels_file: str
            Path to file containing additional labels
        :param output_header: str
            Header of output file
        :param output_filename: str
            Name of output file
        :return: void
            Runs predictions and writes results
        """

        self.features = additional_features
        self.output_header = output_header
        self.output_filename = output_filename
        self.run(additional_labels_file=labels_file)

    def parse_results(self, unique_id):
        """
        :param unique_id:
        :return: tuple of []
            Rearrange based on the find_ids indexes
        """

        sigmas = np.array(
            list(chain.from_iterable(np.array(self.results)[:, 0]))
        ).reshape(len(unique_id.astype(int)), len(self.features))

        scores = np.array(
            list(chain.from_iterable(np.array(self.results)[:, 1]))
        ).reshape(len(unique_id.astype(int)), len(self.features) * 2 + 1)

        importances = np.array(
            list(
                chain.from_iterable(np.array(self.results)[:, 6])
            )
        )
        trues = np.array(
            list(
                chain.from_iterable(np.array(self.results)[:, 7])
            )
        )
        predictions = np.array(
            list(
                chain.from_iterable(np.array(self.results)[:, 8])
            )
        )
        list_of_lines = np.array(self.results)[:, 2]

        # find_ids are useful to reorder the matrix with the ML determinations
        find_ids = list(chain.from_iterable(np.array(self.results)[:, 3]))
        temp_model_ids = list(
            chain.from_iterable(np.array(self.results)[:, 4]))

        tmp_matrix_ml = np.array(
            list(chain.from_iterable(np.array(self.results)[:, 5]))
        )

        matrix_ml = np.zeros(shape=tmp_matrix_ml.shape)
        for i in xrange(len(matrix_ml)):
            matrix_ml[find_ids[i], :] = tmp_matrix_ml[i, :]

        if self.optional_files:
            tmp_matrix_ml = np.array(
                list(chain.from_iterable(
                    np.array(self.results)[:, 5])
                )
            )
        else:
            tmp_matrix_ml = np.array(
                list(chain.from_iterable(
                    np.array(self.results)[:, 5])
                )
            ).reshape(len(self.data[1:]), len(self.features) * 3)

        # Rearrange the matrix based on the find_ids indexes
        matrix_ml = np.zeros(shape=tmp_matrix_ml.shape)
        for i in xrange(len(matrix_ml)):
            matrix_ml[find_ids[i], :] = tmp_matrix_ml[i, :]

        # Rearrange the model_ids based on the find_ids indexes
        model_ids = np.zeros(len(temp_model_ids))
        for i in xrange(len(temp_model_ids)):
            model_ids[find_ids[i]] = temp_model_ids[i]

        return sigmas, scores, list_of_lines, model_ids, matrix_ml, \
               importances, predictions, trues, matrix_ml

    def write_results(self, unique_id):
        sigmas, scores, list_of_lines, model_ids, matrix_ml, \
        importances, predictions, trues, matrix_ml = \
            self.parse_results(unique_id)

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

        # Outputs relative to the Machine Learning determination
        if self.optional_files:
            write_output = get_output(
                model_ids, matrix_ml, len(self.features)
            )
        else:
            write_output = np.column_stack((model_ids, matrix_ml))

        np.savetxt(
            os.path.join(
                self.output_folder,
                self.output_filename
            ),
            write_output,
            header=self.output_header,
            fmt="%.5f"
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

        if self.verbose:
            print ""

    @staticmethod
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

    @staticmethod
    def download_library(
            download_file,
            url="http://cosmology.sns.it/library_game/library.tar.gz"
    ):
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

    @staticmethod
    def create_library():
        """
        :return: void
            Creates necessary  library directory if not existing
        """

        lib_file = os.path.join(
            Game.LIBRARY_FOLDER,
            "library.tar.gz"
        )

        if not os.path.exists(Game.LABELS_FILE):
            if not os.path.exists(Game.LIBRARY_FOLDER):
                print "Creating library folder ..."
                os.makedirs(Game.LIBRARY_FOLDER)  # create necessary folders

            if not os.path.exists(lib_file):
                print "Downloading library ..."
                Game.download_library(lib_file)  # download library

            if not os.path.exists(Game.LABELS_FILE):
                print "Extracting library files ..."
                tar = tarfile.open(lib_file)  # extract library
                tar.extractall()
                tar.close()

    @staticmethod
    def run_parallel(algorithm, n_processes, unique_id):

        pool = multiprocessing.Pool(processes=n_processes)
        results = pool.map(
            algorithm,
            np.arange(1, np.max(unique_id.astype(int)) + 1, 1)
        )
        pool.close()
        pool.join()
        return results


def game(
        i, models, unique_id, initial, limit, features,
        labels_train, labels_test, labels, regr, line_labels,
        filename_int, filename_err, n_repetition, optional_files, to_predict
):
    predicting_additional_labels = to_predict.are_additional_labels()
    if predicting_additional_labels:
        AV, fesc = list(to_predict.generate_features_arrays(n_repetition))
        importances_AV, importances_fesc = list(
            to_predict.generate_importances_arrays())
    else:
        g0, n, NH, U, Z = list(
            to_predict.generate_features_arrays(n_repetition))
        importances_g0, importances_n, importances_NH, \
        importances_U, importances_Z = list(
            to_predict.generate_importances_arrays())

    mask = np.where(models == unique_id[i - 1])
    matrix_mms = []  # matrix_mms is useful to save physical properties
    index_find = []  # index_find helps to keep trace of the indexes
    id_model = []

    # Indexes for the labels:
    # G/G0: 0, n: 1, NH: 2, U: 3, Z: 4
    # In case of AV AV: 3, fesc: 4
    # Definition of training / testing
    features_train = features[:, initial[mask][0]][:limit, :]
    features_test = features[:, initial[mask][0]][limit:, :]

    # ML error estimation
    if predicting_additional_labels:
        [AV_true, AV_pred, sigma_AV] = error_estimate(
            features_train,
            features_test,
            labels_train[:, 3],
            labels_test[:, 3],
            regr
        )
        [fesc_true, fesc_pred, sigma_fesc] = error_estimate(
            features_train,
            features_test,
            labels_train[:, 4],
            labels_test[:, 4],
            regr
        )
    else:
        [g0_true, g0_pred, sigma_g0] = error_estimate(
            features_train,
            features_test,
            labels_train[:, 0],
            labels_test[:, 0],
            regr)
        [n_true, n_pred, sigma_n] = error_estimate(
            features_train, features_test,
            labels_train[:, 1],
            labels_test[:, 1], regr
        )
        [NH_true, NH_pred, sigma_NH] = error_estimate(
            features_train,
            features_test,
            labels_train[:, 2],
            labels_test[:, 2], regr
        )
        [U_true, U_pred, sigma_U] = error_estimate(
            features_train, features_test,
            labels_train[:, 3],
            labels_test[:, 3], regr
        )
        [Z_true, Z_pred, sigma_Z] = error_estimate(
            features_train, features_test,
            labels_train[:, 4],
            labels_test[:, 4], regr
        )

    # Function calls for the machine learning routines

    if predicting_additional_labels:
        [model_AV, imp_AV, score_AV, std_AV] = machine_learn(
            features[:, initial[mask][0]], labels, 3, regr)
        [model_fesc, imp_fesc, score_fesc, std_fesc] = machine_learn(
            features[:, initial[mask][0]], labels, 4, regr)
    else:
        [model_g0, imp_g0, score_g0, std_g0] = machine_learn(
            features[:, initial[mask][0]], labels, 0, regr)
        [model_n, imp_n, score_n, std_n] = machine_learn(
            features[:, initial[mask][0]], labels, 1, regr)
        [model_NH, imp_NH, score_NH, std_NH] = machine_learn(
            features[:, initial[mask][0]], labels, 2, regr)
        [model_U, imp_U, score_U, std_U] = machine_learn(
            features[:, initial[mask][0]], labels, 3, regr
        )
        [model_Z, imp_Z, score_Z, std_Z] = machine_learn(
            features[:, initial[mask][0]], labels, 4, regr
        )

    # Bootstrap
    new_data = realization(
        filename_int, filename_err, n_repetition, mask
    )[:, initial[mask][0]]

    # Prediction of the physical properties
    if optional_files:
        for el in xrange(len(mask[0])):
            if predicting_additional_labels:
                AV[mask[0][el], :] = model_AV.predict(
                    new_data[el::len(mask[0])])
                fesc[mask[0][el], :] = model_fesc.predict(
                    new_data[el::len(mask[0])])
            else:
                g0[mask[0][el], :] = model_g0.predict(
                    new_data[el::len(mask[0])])
                n[mask[0][el], :] = model_n.predict(new_data[el::len(mask[0])])
                NH[mask[0][el], :] = model_NH.predict(
                    new_data[el::len(mask[0])])
                U[mask[0][el], :] = model_U.predict(new_data[el::len(mask[0])])
                Z[mask[0][el], :] = model_Z.predict(new_data[el::len(mask[0])])

            # Model ids
            id_model.append(i)
            index_find.append(mask[0][el])
            matrix_mms.append([g0[mask[0][el], :],
                               n[mask[0][el], :],
                               NH[mask[0][el], :],
                               U[mask[0][el], :],
                               Z[mask[0][el], :]])
    else:
        for el in xrange(len(mask[0])):
            if predicting_additional_labels:
                result = np.zeros((len(new_data[el::len(mask[0])]), 2))
                result[:, 0] = model_AV.predict(new_data[el::len(mask[0])])
                result[:, 1] = model_fesc.predict(new_data[el::len(mask[0])])

                # Model ids
                id_model.append(i)
                index_find.append(mask[0][el])
                vector_mms = np.zeros(6)
                vector_mms[0::3] = np.mean(result, axis=0)
                vector_mms[1::3] = np.median(result, axis=0)
                vector_mms[2::3] = np.std(result, axis=0)
                matrix_mms.append(vector_mms)
            else:
                result = np.zeros((len(new_data[el::len(mask[0])]), 5))
                result[:, 0] = model_g0.predict(new_data[el::len(mask[0])])
                result[:, 1] = model_n.predict(new_data[el::len(mask[0])])
                result[:, 2] = model_NH.predict(new_data[el::len(mask[0])])
                result[:, 3] = model_U.predict(new_data[el::len(mask[0])])
                result[:, 4] = model_Z.predict(new_data[el::len(mask[0])])

                # Model ids
                id_model.append(i)
                index_find.append(mask[0][el])
                vector_mms = np.zeros(15)
                vector_mms[0::3] = np.log10(np.mean(10 ** result, axis=0))
                vector_mms[1::3] = np.log10(np.median(10 ** result, axis=0))
                vector_mms[2::3] = np.std(result, axis=0)
                matrix_mms.append(vector_mms)

    # Importance matrices
    if predicting_additional_labels:
        importances_AV[initial[mask][0]] = imp_AV
        importances_fesc[initial[mask][0]] = imp_fesc
    else:
        importances_g0[initial[mask][0]] = imp_g0
        importances_n[initial[mask][0]] = imp_n
        importances_NH[initial[mask][0]] = imp_NH
        importances_U[initial[mask][0]] = imp_U
        importances_Z[initial[mask][0]] = imp_Z

    # Print message
    print "Model", str(int(i)) + "/" + str(
        int(np.max(unique_id))), "completed..."

    # Returns for the parallelization
    if predicting_additional_labels:
        return [sigma_AV, sigma_fesc], \
               [i, score_AV, std_AV, score_fesc, std_fesc], \
               line_labels[initial[mask][0]], \
               index_find, id_model, matrix_mms, \
               [importances_AV, importances_fesc], \
               [np.array(AV_true), np.array(fesc_true)], \
               [np.array(AV_pred), np.array(fesc_pred)]
    else:
        return [sigma_g0, sigma_n, sigma_NH, sigma_U, sigma_Z], \
               [i, score_g0, std_g0, score_n, std_n, score_NH, std_NH, score_U,
                std_U, score_Z, std_Z], \
               line_labels[initial[mask][0]], \
               index_find, id_model, matrix_mms, \
               [importances_g0, importances_n, importances_NH, importances_U,
                importances_Z], \
               [np.array(g0_true), np.array(n_true), np.array(NH_true),
                np.array(U_true), np.array(Z_true)], \
               [np.array(g0_pred), np.array(n_pred), np.array(NH_pred),
                np.array(U_pred), np.array(Z_pred)]


def main():
    driver = Game(
        ["g0", "n", "NH", "U", "Z"],
        output_header="id_model mean[Log(G0)] median[Log(G0)]"
                      "sigma[Log(G0)] mean[Log(n)] median[Log(n)]"
                      "sigma[Log(n)] mean[Log(NH)] median[Log(NH)]"
                      "sigma[Log(NH)] mean[Log(U)] median[Log(U)]"
                      "sigma[Log(U)] mean[Log(Z)] median[Log(Z)]"
                      "sigma[Log(Z)]",
        output_filename="output_ml.dat",
        manual_input=False,
        verbose=True
    )

    driver.run()
    driver.run_additional_labels(
        additional_features=["AV", "fesc"],
        labels_file=os.path.join(
            os.getcwd(),
            "library",
            "additional_labels.dat"
        ),
        output_header="id_model mean[Av] median[Av] sigma[Av] mean[fesc] "
                      "median[fesc] sigma[fesc]",
        output_filename="output_ml_additional.dat"
    )


if __name__ == "__main__":
    main()
