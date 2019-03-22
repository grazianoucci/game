# -*- coding: utf-8 -*-

import multiprocessing
import os
import time
import math
from functools import partial
from itertools import chain

import numpy as np
from sklearn import tree
from sklearn.ensemble import AdaBoostRegressor

from errors import GameException
from game_alg import main_algorithm_to_pool
from ml import determination_models
from prepare import read_emission_line_file, read_library_file

MODELS_FORMAT = '# of input  models: {}\n# of unique models: {}'
END_RUN_FORMAT = 'Elapsed time for ML: {}'

MATRIX_TO_GB = 0.000000008
GAME_MAX_CORES = 30
TOTAL_MEM = 256  # gb
MAX_MEM_PERC = 90  # 90%
MAX_MEM = MAX_MEM_PERC / 100.0 * TOTAL_MEM
TOO_MUCH_MEMORY_REQUIRED_FORMAT = 'Memory required is {} GB: too much!'

OUTPUT_PRED_FORMAT = 'output_pred_{}.dat'
OUTPUT_TRUE_FORMAT = 'output_pred_{}.dat'
OUTPUT_PDF_FORMAT = 'output_pred_{}.dat'
OUTPUT_IMPORTANCES_FORMAT = 'output_feature_importances_{}.dat'
OUTPUT_ML = 'output_ml.dat'
OUTPUT_MODELS_IDS = 'model_ids.dat'
NUM_FORMAT = '%.5f'

STD_FORMAT = 'Standard deviation of log({}): {:.3f}\n'
CROSS_VAL_FORMAT = 'Cross-validation score for: {:.3f} +- {:.3f}\n'
OUT_HEADER_FORMAT = ' mean[Log({0})] median[Log({0})] sigma[Log({0})]'


def raise_if_mem_per_process_is_too_high(mem_required_per_process):
    total_worst_weight = mem_required_per_process * GAME_MAX_CORES
    print('{} GB VS {} GB'.format(total_worst_weight, MAX_MEM))
    if total_worst_weight >= MAX_MEM:  # critical memory saturation
        message = TOO_MUCH_MEMORY_REQUIRED_FORMAT.format(total_worst_weight)
        exception = GameException.build_too_much_memory_exception(message)
        raise exception

    raise ValueError('should proceed')

def raise_if_matrix_size_is_too_high(matrix_size, n_repetitions, n_labels=7):
    all_matrices_size = matrix_size * n_labels
    all_matrices_weight = all_matrices_size * n_repetitions * MATRIX_TO_GB  # GB
    raise_if_mem_per_process_is_too_high(all_matrices_weight)


def raise_if_matrix_per_process_is_too_high(n_rows, n_cols, n_repetitions, n_labels=7):
    matrix_size = n_rows * n_cols
    print(matrix_size)
    raise_if_matrix_size_is_too_high(matrix_size, n_repetitions, n_labels)

def raise_if_mem_is_too_high(input_rows, input_cols, n_repetitions, additional_files, models, unique_id):
    if additional_files:
        raise_if_matrix_per_process_is_too_high(input_rows, input_cols, n_repetitions)
    else:  # no optional files -> check big matrix used for statistics
        all_i = np.arange(1, np.max(unique_id.astype(int)) + 1, 1)
        sub_matrix_max = 0
        sub_matrix_min = 9999999999
        all_rows = input_rows
 
        for i in all_i:
            mask = np.where(models == unique_id[i - 1])
            n_rows = len(mask[0])
            n_cols = input_cols
            matrix_size = n_cols * n_rows
            print('model #{} -> matrix size: {} (rows = {})'.format(i, matrix_size, n_rows))
            if matrix_size > sub_matrix_max:
                sub_matrix_max = matrix_size

            if matrix_size < sub_matrix_min:
                sub_matrix_min = matrix_size

        try:
            raise_if_matrix_size_is_too_high(sub_matrix_max, n_repetitions)
        except Exception as e:
            try:
                # raise_if_matrix_size_is_too_high(sub_matrix_min, n_repetitions)
                n_chunks = math.ceil((all_rows * n_cols) / sub_matrix_max)
                print('should chunkize files into {}'.format(n_chunks))
                # todo send email
            except Exception as e:
                raise e
            

def game(
        filename_int
        , filename_err
        , filename_library
        , additional_files
        , n_proc
        , n_repetitions
        , n_estimators
        , output_folder
        , verbose
        , out_labels
        , lib_folder
):
    tree_regr = tree.DecisionTreeRegressor(criterion='mse', splitter='best',
                                           max_features=None)
    regr = AdaBoostRegressor(
        tree_regr,
        n_estimators=n_estimators,
        random_state=0
    )
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    data, lower, upper = read_emission_line_file(filename_int)

    # Library file reading
    library_file = os.path.join(lib_folder, 'library.csv')
    additional_labels_lib_file = os.path.join(lib_folder,
                                              'additional_labels.dat')

    output, line_labels = read_library_file(library_file, filename_library)
    # Determination of unique models based on the missing data
    # In this case missing data are values with zero intensities
    # Be careful because the first row in data there are wavelengths!
    initial, models, unique_id = determination_models(data[1:])

    test_size = 0.10
    n_unique_models = int(np.max(unique_id))

    if verbose:
        print MODELS_FORMAT.format(len(data[1:]), n_unique_models)

    start_time = time.time()

    features = output[:, :-5]
    labels = np.double(output[:, len(output[0]) - 5:len(output[0])])
    labels[:, -1] = np.log10(labels[:, -1])

    labels_rows = labels.shape[0]
    additional_columns = 2
    labels = np.append(
        labels,
        np.zeros(shape=(labels_rows, additional_columns)),
        axis=1
    )  # make room for additional labels
    labels[:, -2:] = np.loadtxt(additional_labels_lib_file)

    # This code is inserted in order to work with logarithms!
    # If there is a zero, we substitute it with 1e-9
    labels[labels[:, -2] == 0, -2] = 1e-9
    labels[labels[:, -1] == 0, -1] = 1e-9
    labels[:, -2] = np.log10(labels[:, -2])
    labels[:, -1] = np.log10(labels[:, -1])

    # Limit data
    limit = int((1. - test_size) * len(features))
    labels_train = labels[:limit, :]
    labels_test = labels[limit:, :]

    n_rows = len(data)
    n_cols = len(data[0])
    raise_if_mem_is_too_high(n_rows, n_cols, n_repetitions, additional_files,
                             models, unique_id)

    try:
        raise_if_mem_is_too_high(n_rows, n_cols, n_repetitions, additional_files,
                             models, unique_id)
    except Exception as e:
        if additional_files:
            additional_files = False
            # send email to notify of no output of additional
        else:
            raise e

    matrix_shape = len(data[1:]), n_repetitions
    if 'g0' in out_labels:
        if additional_files:
            g0 = np.zeros(shape=matrix_shape)
        else:
            g0 = None

        importances_g0 = np.zeros(len(data[0]))
    else:
        g0, importances_g0 = None, None

    if 'n' in out_labels:
        if additional_files:
            n = np.zeros(shape=matrix_shape)
        else:
            n = None

        importances_n = np.zeros(len(data[0]))
    else:
        n, importances_n = None, None

    if 'NH' in out_labels:
        if additional_files:
            NH = np.zeros(shape=matrix_shape)
        else:
            NH = None

        importances_NH = np.zeros(len(data[0]))
    else:
        NH, importances_NH = None, None

    if 'U' in out_labels:
        if additional_files:
            U = np.zeros(shape=matrix_shape)
        else:
            U = None

        importances_U = np.zeros(len(data[0]))
    else:
        U, importances_U = None, None

    if 'Z' in out_labels:
        if additional_files:
            Z = np.zeros(shape=matrix_shape)
        else:
            Z = None

        importances_Z = np.zeros(len(data[0]))
    else:
        Z, importances_Z = None, None

    if 'Av' in out_labels:
        if additional_files:
            Av = np.zeros(shape=matrix_shape)
        else:
            Av = None

        importances_Av = np.zeros(len(data[0]))
    else:
        Av, importances_Av = None, None

    if 'fesc' in out_labels:
        if additional_files:
            fesc = np.zeros(shape=matrix_shape)
        else:
            fesc = None

        importances_fesc = np.zeros(len(data[0]))
    else:
        fesc, importances_fesc = None, None

    main_algorithm = partial(main_algorithm_to_pool,
                             models=models, unique_id=unique_id,
                             initial=initial, limit=limit
                             , features=features, labels_train=labels_train,
                             labels_test=labels_test
                             , labels=labels, regr=regr, line_labels=line_labels
                             , g0=g0, n=n, NH=NH, U=U, Z=Z, Av=Av, fesc=fesc
                             , importances_g0=importances_g0
                             , importances_n=importances_n
                             , importances_NH=importances_NH
                             , importances_U=importances_U
                             , importances_Z=importances_Z
                             , importances_Av=importances_Av
                             , importances_fesc=importances_fesc
                             , filename_int=filename_int
                             , filename_err=filename_err
                             , n_repetition=n_repetitions
                             , additional_files=additional_files
                             , out_labels=out_labels
                             )

    pool = multiprocessing.Pool(processes=n_proc)
    results = pool.map(main_algorithm,
                       np.arange(1, np.max(unique_id.astype(int)) + 1, 1))
    pool.close()
    pool.join()

    end_time = time.time()
    if verbose:
        print END_RUN_FORMAT.format(end_time - start_time)

    sigmas = np.array(
        list(chain.from_iterable(np.array(results)[:, 0]))).reshape(
        len(unique_id.astype(int)), 7)
    scores = np.array(
        list(chain.from_iterable(np.array(results)[:, 1]))).reshape(
        len(unique_id.astype(int)), 15)
    importances = np.array(list(chain.from_iterable(np.array(results)[:, 6])))
    trues = np.array(list(chain.from_iterable(np.array(results)[:, 7])))
    preds = np.array(list(chain.from_iterable(np.array(results)[:, 8])))
    list_of_lines = np.array(results)[:, 2]

    # find_ids are useful to reorder the matrix with the ML determinations
    find_ids = list(chain.from_iterable(np.array(results)[:, 3]))
    temp_model_ids = list(chain.from_iterable(np.array(results)[:, 4]))

    if additional_files:
        temp_matrix_ml = np.array(
            list(chain.from_iterable(np.array(results)[:, 5])))

        # Rearrange the matrix based on the find_ids indexes
        matrix_ml = np.zeros(shape=temp_matrix_ml.shape)

        for i in xrange(len(matrix_ml)):
            matrix_ml[find_ids[i], :] = temp_matrix_ml[i, :]
    else:
        temp_matrix_ml = np.array(
            list(chain.from_iterable(np.array(results)[:, 5]))).reshape(
            len(data[1:]), 21)
        # Rearrange the matrix based on the find_ids indexes
        matrix_ml = np.zeros(shape=temp_matrix_ml.shape)
        for i in xrange(len(matrix_ml)):
            matrix_ml[find_ids[i], :] = temp_matrix_ml[i, :]

    # Rearrange the model_ids based on the find_ids indexes
    model_ids = np.zeros(len(temp_model_ids))
    for i in xrange(len(temp_model_ids)):
        model_ids[find_ids[i]] = temp_model_ids[i]

    f = open(os.path.join(output_folder, OUTPUT_MODELS_IDS), 'w+')

    for i in xrange(len(sigmas)):
        f.write('##############################\n')
        f.write('Id model: %d\n' % (i + 1))

        if 'g0' in out_labels:
            f.write(STD_FORMAT.format('g0', sigmas[i, 0]))
        if 'n' in out_labels:
            f.write(STD_FORMAT.format('n', sigmas[i, 1]))
        if 'NH' in out_labels:
            f.write(STD_FORMAT.format('NH', sigmas[i, 2]))
        if 'U' in out_labels:
            f.write(STD_FORMAT.format('U', sigmas[i, 3]))
        if 'Z' in out_labels:
            f.write(STD_FORMAT.format('Z', sigmas[i, 4]))
        if 'Av' in out_labels:
            f.write(STD_FORMAT.format('Av', sigmas[i, 5]))
        if 'fesc' in out_labels:
            f.write(STD_FORMAT.format('fesc', sigmas[i, 6]))

        if 'g0' in out_labels:
            f.write(
                CROSS_VAL_FORMAT.format('g0', scores[i, 1], 2. * scores[i, 2]))
        if 'n' in out_labels:
            f.write(
                CROSS_VAL_FORMAT.format('n', scores[i, 3], 2. * scores[i, 4]))
        if 'NH' in out_labels:
            f.write(
                CROSS_VAL_FORMAT.format('NH', scores[i, 5], 2. * scores[i, 6]))
        if 'U' in out_labels:
            f.write(
                CROSS_VAL_FORMAT.format('U', scores[i, 7], 2. * scores[i, 8]))
        if 'Z' in out_labels:
            f.write(
                CROSS_VAL_FORMAT.format('Z', scores[i, 9], 2. * scores[i, 10]))
        if 'Av' in out_labels:
            f.write(CROSS_VAL_FORMAT.format('Av', scores[i, 11],
                                            2. * scores[i, 12]))
        if 'fesc' in out_labels:
            f.write(CROSS_VAL_FORMAT.format('fesc', scores[i, 13],
                                            2. * scores[i, 14]))

        f.write('List of input lines:\n')
        f.write('%s\n' % list_of_lines[i])

    out_header = 'id_model'
    out_ml = [model_ids]

    if 'g0' in out_labels:
        out_header += OUT_HEADER_FORMAT.format('g0')
        np.savetxt(os.path.join(output_folder,
                                OUTPUT_IMPORTANCES_FORMAT.format('g0')),
                   np.vstack((data[0], importances[0::7, :])), fmt=NUM_FORMAT)

        if additional_files:
            out_ml += [
                np.log10(np.mean(10 ** matrix_ml[:, 0], axis=1)),
                np.log10(np.median(10 ** matrix_ml[:, 0], axis=1)),
                np.std(matrix_ml[:, 0], axis=1),
            ]
    if 'n' in out_labels:
        out_header += OUT_HEADER_FORMAT.format('n')
        np.savetxt(os.path.join(output_folder,
                                OUTPUT_IMPORTANCES_FORMAT.format('n')),
                   np.vstack((data[0], importances[1::7, :])), fmt=NUM_FORMAT)
        if additional_files:
            out_ml += [
                np.log10(np.mean(10 ** matrix_ml[:, 1], axis=1)),
                np.log10(np.median(10 ** matrix_ml[:, 1], axis=1)),
                np.std(matrix_ml[:, 1], axis=1)
            ]
    if 'NH' in out_labels:
        out_header += OUT_HEADER_FORMAT.format('NH')
        np.savetxt(os.path.join(output_folder,
                                OUTPUT_IMPORTANCES_FORMAT.format('NH')),
                   np.vstack((data[0], importances[2::7, :])), fmt=NUM_FORMAT)
        if additional_files:
            out_ml += [
                np.log10(np.mean(10 ** matrix_ml[:, 2], axis=1)),
                np.log10(np.median(10 ** matrix_ml[:, 2], axis=1)),
                np.std(matrix_ml[:, 2], axis=1)
            ]
    if 'U' in out_labels:
        out_header += OUT_HEADER_FORMAT.format('U')
        np.savetxt(os.path.join(output_folder,
                                OUTPUT_IMPORTANCES_FORMAT.format('U')),
                   np.vstack((data[0], importances[3::7, :])), fmt=NUM_FORMAT)
        if additional_files:
            out_ml += [
                np.log10(np.mean(10 ** matrix_ml[:, 3], axis=1)),
                np.log10(np.median(10 ** matrix_ml[:, 3], axis=1)),
                np.std(matrix_ml[:, 3], axis=1)
            ]
    if 'Z' in out_labels:
        out_header += OUT_HEADER_FORMAT.format('Z')
        np.savetxt(os.path.join(output_folder,
                                OUTPUT_IMPORTANCES_FORMAT.format('Z')),
                   np.vstack((data[0], importances[4::7, :])), fmt=NUM_FORMAT)
        if additional_files:
            out_ml += [
                np.log10(np.mean(10 ** matrix_ml[:, 4], axis=1)),
                np.log10(np.median(10 ** matrix_ml[:, 4], axis=1)),
                np.std(matrix_ml[:, 4], axis=1)
            ]
    if 'Av' in out_labels:
        out_header += OUT_HEADER_FORMAT.format('Av')
        np.savetxt(os.path.join(output_folder,
                                OUTPUT_IMPORTANCES_FORMAT.format('Av')),
                   np.vstack((data[0], importances[5::7, :])), fmt=NUM_FORMAT)

        if additional_files:
            out_ml += [
                np.mean(matrix_ml[:, 5], axis=1),
                np.median(matrix_ml[:, 5], axis=1),
                np.std(matrix_ml[:, 5], axis=1)
            ]

    if 'fesc' in out_labels:
        out_header += OUT_HEADER_FORMAT.format('fesc')
        np.savetxt(os.path.join(output_folder,
                                OUTPUT_IMPORTANCES_FORMAT.format('fesc')),
                   np.vstack((data[0], importances[6::7, :])), fmt=NUM_FORMAT)

        if additional_files:
            out_ml += [
                np.mean(matrix_ml[:, 6], axis=1),
                np.median(matrix_ml[:, 6], axis=1),
                np.std(matrix_ml[:, 6], axis=1)
            ]

    f.close()

    if additional_files:
        write_output = np.vstack(tuple(out_ml)).T
    else:
        write_output = np.column_stack((model_ids, matrix_ml))

    output_ml_file = os.path.join(output_folder, OUTPUT_ML)
    np.savetxt(output_ml_file, write_output, header=out_header, fmt=NUM_FORMAT)

    if additional_files:
        if 'g0' in out_labels:
            np.savetxt(os.path.join(output_folder,
                                    OUTPUT_PRED_FORMAT.format('g0')),
                       preds[0::7, :],
                       fmt=NUM_FORMAT)
            np.savetxt(os.path.join(output_folder,
                                    OUTPUT_TRUE_FORMAT.format('g0')),
                       trues[0::7, :],
                       fmt=NUM_FORMAT)
            np.savetxt(os.path.join(output_folder,
                                    OUTPUT_PDF_FORMAT.format('g0')),
                       matrix_ml[:, 0],
                       fmt=NUM_FORMAT)

        if 'n' in out_labels:
            np.savetxt(
                os.path.join(output_folder, OUTPUT_PRED_FORMAT.format('n')),
                preds[1::7, :],
                fmt=NUM_FORMAT)
            np.savetxt(
                os.path.join(output_folder, OUTPUT_TRUE_FORMAT.format('n')),
                trues[1::7, :],
                fmt=NUM_FORMAT)
            np.savetxt(
                os.path.join(output_folder, OUTPUT_PDF_FORMAT.format('n')),
                matrix_ml[:, 1],
                fmt=NUM_FORMAT)

        if 'NH' in out_labels:
            np.savetxt(
                os.path.join(output_folder, OUTPUT_PRED_FORMAT.format('NH')),
                preds[2::7, :],
                fmt=NUM_FORMAT)
            np.savetxt(
                os.path.join(output_folder, OUTPUT_TRUE_FORMAT.format('NH')),
                trues[2::7, :],
                fmt=NUM_FORMAT)
            np.savetxt(
                os.path.join(output_folder, OUTPUT_PDF_FORMAT.format('NH')),
                matrix_ml[:, 2],
                fmt=NUM_FORMAT)

        if 'U' in out_labels:
            np.savetxt(
                os.path.join(output_folder, OUTPUT_PRED_FORMAT.format('U')),
                preds[3::7, :],
                fmt=NUM_FORMAT)
            np.savetxt(
                os.path.join(output_folder, OUTPUT_TRUE_FORMAT.format('U')),
                trues[3::7, :],
                fmt=NUM_FORMAT)
            np.savetxt(
                os.path.join(output_folder, OUTPUT_PDF_FORMAT.format('U')),
                matrix_ml[:, 3],
                fmt=NUM_FORMAT)

        if 'Z' in out_labels:
            np.savetxt(
                os.path.join(output_folder, OUTPUT_PRED_FORMAT.format('Z')),
                preds[4::7, :],
                fmt=NUM_FORMAT)
            np.savetxt(
                os.path.join(output_folder, OUTPUT_TRUE_FORMAT.format('Z')),
                trues[4::7, :],
                fmt=NUM_FORMAT)
            np.savetxt(
                os.path.join(output_folder, OUTPUT_PDF_FORMAT.format('Z')),
                matrix_ml[:, 4],
                fmt=NUM_FORMAT)

        if 'Av' in out_labels:
            np.savetxt(
                os.path.join(output_folder, OUTPUT_PRED_FORMAT.format('Av')),
                preds[5::2, :],
                fmt=NUM_FORMAT)
            np.savetxt(
                os.path.join(output_folder, OUTPUT_TRUE_FORMAT.format('Av')),
                trues[5::2, :],
                fmt=NUM_FORMAT)
            np.savetxt(
                os.path.join(output_folder, OUTPUT_PDF_FORMAT.format('Av')),
                matrix_ml[:, 5],
                fmt=NUM_FORMAT)

        if 'fesc' in out_labels:
            np.savetxt(
                os.path.join(output_folder, OUTPUT_PRED_FORMAT.format('fesc')),
                preds[6::7, :],
                fmt=NUM_FORMAT)
            np.savetxt(
                os.path.join(output_folder, OUTPUT_TRUE_FORMAT.format('fesc')),
                trues[6::7, :],
                fmt=NUM_FORMAT)
            np.savetxt(
                os.path.join(output_folder, OUTPUT_PDF_FORMAT.format('fesc')),
                matrix_ml[:, 6],
                fmt=NUM_FORMAT)
