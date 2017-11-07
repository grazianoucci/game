# !/usr/bin/python3
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


""" GAME (GAlaxy Machine learning for Emission lines) """

import time
from functools import partial
from itertools import chain

import numpy as np
from sklearn import tree
from sklearn.ensemble import AdaBoostRegressor

from ml import realization, determine_models, error_estimate, machine_learn, \
    get_write_output
from utils import create_library, create_output_directory, \
    read_emission_line_file, read_library_file, write_output_files, \
    write_importances_files, write_models_info, get_files_from_user

YES, NO = "y", "n"
INTRO = '--------------------------------------------------------\n' + \
        '--- GAME (GAlaxy Machine learning for Emission lines) --\n' + \
        '------- see Ucci G. et al. (2017a,b) for details -------\n' + \
        '--------------------------------------------------------\n\n' + \
        'ML Algorithm: AdaBoost with Decision Trees as base learner.'

# algorithm for Machine Learning
REGRESSOR = AdaBoostRegressor(
    tree.DecisionTreeRegressor(
        criterion='mse',
        splitter='best',
        max_features=None
    ),
    n_estimators=2,
    random_state=0
)  # ref1: http://adsabs.harvard.edu/abs/2017MNRAS.465.1144U

# Testing, test_size is the percentage of the library to use as testing
# set to determine the PDFs
TEST_SIZE = 0.10
NUMBER_OF_PROCESSES = 2


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

    def __init__(self):
        pass

    def run_parallel(self, number_of_processes=2):
        """
        :param number_of_processes: int
            Number of processes to use when running parallel algorithm
        :return: TODO
            TODO
        """

        pass


def game(
        i, models, unique_id, initial, limit, features,
        labels_train, labels_test, labels, regr, line_labels,
        filename_int, filename_err, n_repetition, choice_rep, to_predict
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
    if choice_rep == YES:
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
    print 'Model', str(int(i)) + '/' + str(
        int(np.max(unique_id))), 'completed...'

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


def run_parallel_game():
    pass


def main(
        manual_input=False,
        filename_int='input/inputs_game_test.dat',
        filename_err='input/errors_game_test.dat',
        filename_library='input/labels_game_test.dat',
        choice_rep=YES, n_processes=NUMBER_OF_PROCESSES, n_repetition=10000,
        dir_path='output/', verbose=True
):
    create_library()

    if verbose:
        print INTRO

    # Input file reading
    if manual_input:
        filename_int, filename_err, filename_library = get_files_from_user()

    create_output_directory(
        dir_path
    )  # Create output directory if not existing

    if manual_input:
        choice_rep = raw_input(
            'Do you want to create the optional files [y/n]?: '
        )  # optional files

        n_processes = raw_input('Choose the number of processors: ')

    if verbose:
        print '\nProgram started...'

    # Number of repetition for the PDFs determination
    # Input file reading
    data, lower, upper = read_emission_line_file(filename_int)
    output, line_labels = read_library_file(filename_library)

    # Determination of unique models based on the missing data
    # In this case missing data are values with zero intensities
    # Be careful because the first row in data there are wavelengths!
    initial, models, unique_id = determine_models(data[1:])

    if verbose:
        print '# of input  models                     :', len(data[1:])
        print '# of unique models for Machine Learning:', int(
            np.max(unique_id))
        print '\nStarting Machine Learning algorithm for the default ' \
              'labels... '

    start_time = time.time()

    # Definition of features and labels for Machine Learning
    # (for metallicity logarithm has been used)
    features = output[:, :-5]
    labels = np.double(output[:, len(output[0]) - 5:len(output[0])])
    labels[:, -1] = np.log10(labels[:, -1])
    limit = int((1. - TEST_SIZE) * len(features))
    labels_train = labels[:limit, :]
    labels_test = labels[limit:, :]

    to_predict = Prediction(
        ["g0", "n", "NH", "U", "Z"],
        data
    )

    # Searching for values of the physical properties
    algorithm = partial(
        game,
        i=0,
        models=models, unique_id=unique_id, initial=initial, limit=limit,
        features=features, labels_train=labels_train,
        labels_test=labels_test, labels=labels,
        regr=REGRESSOR, line_labels=line_labels,
        filename_int=filename_int,
        filename_err=filename_err,
        n_repetition=n_repetition, choice_rep=choice_rep,
        to_predict=to_predict
    )
    results = [algorithm()]
    end_time = time.time()

    if verbose:
        print 'Elapsed time for ML:', (end_time - start_time)
        print '\nWriting output files for the default labels...'

    # Rearrange based on the find_ids indexes
    sigmas = np.array(
        list(chain.from_iterable(np.array(results)[:, 0]))).reshape(
        len(unique_id.astype(int)), 5)

    scores = np.array(
        list(chain.from_iterable(np.array(results)[:, 1]))).reshape(
        len(unique_id.astype(int)), 11)

    importances = np.array(list(chain.from_iterable(np.array(results)[:, 6])))
    trues = np.array(list(chain.from_iterable(np.array(results)[:, 7])))
    preds = np.array(list(chain.from_iterable(np.array(results)[:, 8])))
    list_of_lines = np.array(results)[:, 2]

    # find_ids are useful to reorder the matrix with the ML determinations
    find_ids = list(chain.from_iterable(np.array(results)[:, 3]))
    temp_model_ids = list(chain.from_iterable(np.array(results)[:, 4]))

    temp_matrix_ml = np.array(
        list(chain.from_iterable(np.array(results)[:, 5]))
    )

    # Rearrange the matrix based on the find_ids indexes
    matrix_ml = np.zeros(shape=temp_matrix_ml.shape)

    for i in xrange(len(matrix_ml)):
        matrix_ml[find_ids[i], :] = temp_matrix_ml[i, :]

    if choice_rep == NO:
        temp_matrix_ml = np.array(
            list(chain.from_iterable(np.array(results)[:, 5]))).reshape(
            len(data[1:]), 15)

        # Rearrange the matrix based on the find_ids indexes
        matrix_ml = np.zeros(shape=temp_matrix_ml.shape)
        for i in xrange(len(matrix_ml)):
            matrix_ml[find_ids[i], :] = temp_matrix_ml[i, :]

    # Rearrange the model_ids based on the find_ids indexes
    model_ids = np.zeros(len(temp_model_ids))
    for i in xrange(len(temp_model_ids)):
        model_ids[find_ids[i]] = temp_model_ids[i]

    # Write information on different models
    write_models_info(dir_path, sigmas, scores, list_of_lines)

    # Outputs relative to the Machine Learning determination
    if choice_rep == YES:
        write_output = get_write_output(model_ids, matrix_ml)
    else:
        write_output = np.column_stack((model_ids, matrix_ml))

    np.savetxt(
        dir_path + 'output_ml.dat',
        write_output,
        header="id_model mean[Log(G0)] median[Log(G0)] sigma[Log(G0)] "
               "mean[Log(n)] median[Log(n)] sigma[Log(n)] mean[Log("
               "NH)] median[Log(NH)] sigma[Log(NH)] mean[Log(U)] "
               "median[Log(U)] sigma[Log(U)] mean[Log(Z)] median[Log("
               "Z)] sigma[Log(Z)]",
        fmt='%.5f'
    )

    # Outputs with the feature importances
    write_importances_files(dir_path, data, importances)

    # Optional files
    if choice_rep == YES:
        write_output_files(dir_path, preds, trues, matrix_ml)

    if verbose:
        print ''


if __name__ == "__main__":
    main()
