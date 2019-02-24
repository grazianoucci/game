import multiprocessing
import os
import time
from functools import partial
from itertools import chain

import numpy as np
from sklearn import tree
from sklearn.ensemble import AdaBoostRegressor

from game_alg import main_algorithm_to_pool
from game_additional import main_algorithm_additional_to_pool
from ml import determination_models
from prepare import read_emission_line_file, read_library_file


def game(
        filename_int
        , filename_err
        , filename_library
        , additional_files
        , n_proc
        , n_repetition
        , output_folder
        , verbose
        , out_labels
        , out_additional_labels
):
    regr = AdaBoostRegressor(tree.DecisionTreeRegressor(criterion='mse',
                                                        splitter='best',
                                                        max_features=None),
                             n_estimators=50,
                             random_state=0)

    ###########################################
    # Create output directory if not existing #
    ###########################################
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    ###################################################
    # Number of repetition for the PDFs determination #
    ###################################################
    # Input file reading
    data, lower, upper = read_emission_line_file(filename_int)
    # Library file reading
    library_file = os.path.join(os.getcwd(), 'library', 'library.csv')
    output, line_labels = read_library_file(library_file, filename_library)
    # Determination of unique models based on the missing data
    # In this case missing data are values with zero intensities
    # Be careful because the first row in data there are wavelengths!
    initial, models, unique_id = determination_models(data[1:])
    # This creates arrays useful to save the output for the feature importances
    importances_g0 = np.zeros(len(data[0]))
    importances_n = np.zeros(len(data[0]))
    importances_NH = np.zeros(len(data[0]))
    importances_U = np.zeros(len(data[0]))
    importances_Z = np.zeros(len(data[0]))

    ###################################################################################################
    # Testing, test_size is the percentage of the library to use as testing set to determine the PDFs #
    ###################################################################################################
    test_size = 0.10
    if verbose:
        print '# of input  models                     :', len(data[1:])
        print '# of unique models for Machine Learning:', int(np.max(unique_id))
        print ''
        print 'Starting of Machine Learning algorithm for the default labels...'
    start_time = time.time()

    ##########################################################
    # Definition of features and labels for Machine Learning #
    #      (for metallicity logarithm has been used)         #
    ##########################################################
    features = output[:, :-5]
    labels = np.double(output[:, len(output[0]) - 5:len(output[0])])
    labels[:, -1] = np.log10(labels[:, -1])
    limit = int((1. - test_size) * len(features))
    labels_train = labels[:limit, :]
    labels_test = labels[limit:, :]

    ######################################
    # Initialization of arrays and lists #
    ######################################
    if additional_files:
        g0 = np.zeros(shape=(len(data[1:]), n_repetition))
        n = np.zeros(shape=(len(data[1:]), n_repetition))
        NH = np.zeros(shape=(len(data[1:]), n_repetition))
        U = np.zeros(shape=(len(data[1:]), n_repetition))
        Z = np.zeros(shape=(len(data[1:]), n_repetition))

    ################
    # Pool calling #
    ################
    main_algorithm = partial(main_algorithm_to_pool,
                             models=models, unique_id=unique_id,
                             initial=initial, limit=limit
                             , features=features, labels_train=labels_train,
                             labels_test=labels_test
                             , labels=labels, regr=regr, line_labels=line_labels
                             , g0=g0, n=n, NH=NH, U=U, Z=Z
                             , importances_g0=importances_g0,
                             importances_n=importances_n,
                             importances_NH=importances_NH,
                             importances_U=importances_U,
                             importances_Z=importances_Z
                             , filename_int=filename_int,
                             filename_err=filename_err,
                             n_repetition=n_repetition,
                             additional_files=additional_files
                             )
    pool = multiprocessing.Pool(processes=n_proc)
    results = pool.map(main_algorithm,
                       np.arange(1, np.max(unique_id.astype(int)) + 1, 1))
    pool.close()
    pool.join()
    end_time = time.time()

    if verbose:
        print 'Elapsed time for ML:', (end_time - start_time)
        print ''
        print 'Writing output files for the default labels...'

    ###########################################
    # Rearrange based on the find_ids indexes #
    ###########################################
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

    # find_ids are usefult to reorder the matrix with the ML determinations
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
            len(data[1:]), 15)
        # Rearrange the matrix based on the find_ids indexes
        matrix_ml = np.zeros(shape=temp_matrix_ml.shape)
        for i in xrange(len(matrix_ml)):
            matrix_ml[find_ids[i], :] = temp_matrix_ml[i, :]

    # Rearrange the model_ids based on the find_ids indexes
    model_ids = np.zeros(len(temp_model_ids))
    for i in xrange(len(temp_model_ids)):
        model_ids[find_ids[i]] = temp_model_ids[i]
    #########################################
    # Write information on different models #
    #########################################
    f = open(os.path.join(output_folder, 'model_ids.dat'), 'w+')
    for i in xrange(len(sigmas)):
        f.write('##############################\n')
        f.write('Id model: %d\n' % (i + 1))

        if "G0" in out_labels:
            f.write('Standard deviation of log(G0): %.3f\n' % sigmas[i, 0])
        if "n" in out_labels:
            f.write('Standard deviation of log(n):  %.3f\n' % sigmas[i, 1])
        if "NH" in out_labels:
            f.write('Standard deviation of log(NH): %.3f\n' % sigmas[i, 2])
        if "U" in out_labels:
            f.write('Standard deviation of log(U):  %.3f\n' % sigmas[i, 3])
        if "Z" in out_labels:
            f.write('Standard deviation of log(Z):  %.3f\n' % sigmas[i, 4])

        f.write('List of input lines:\n')
        f.write('%s\n' % list_of_lines[i])
    f.write('##############################\n')

    out_header = "id_model"
    out_ml = [model_ids]

    if "G0" in out_labels:
        out_header += " mean[Log(G0)] median[Log(G0)] sigma[Log(G0)]"
        f.write('Cross-validation score for G0: %.3f +- %.3f\n' % (
            scores[i, 1], 2. * scores[i, 2]))
        np.savetxt(os.path.join(output_folder,
                                'output_feature_importances_G0.dat'),
                   np.vstack((data[0], importances[0::5, :])), fmt='%.5f')

        if additional_files:
            out_ml += [
                np.log10(np.mean(10 ** matrix_ml[:, 0], axis=1)),
                np.log10(np.median(10 ** matrix_ml[:, 0], axis=1)),
                np.std(matrix_ml[:, 0], axis=1),
            ]
    if "n" in out_labels:
        out_header += " mean[Log(n)] median[Log(n)] sigma[Log(n)]"
        f.write('Cross-validation score for n:  %.3f +- %.3f\n' % (
            scores[i, 3], 2. * scores[i, 4]))
        np.savetxt(os.path.join(output_folder,
                                'output_feature_importances_n.dat'),
                   np.vstack((data[0], importances[1::5, :])), fmt='%.5f')
        if additional_files:
            out_ml += [
                np.log10(np.mean(10 ** matrix_ml[:, 1], axis=1)),
                np.log10(np.median(10 ** matrix_ml[:, 1], axis=1)),
                np.std(matrix_ml[:, 1], axis=1)
            ]
    if "NH" in out_labels:
        out_header += " mean[Log(NH)] median[Log(NH)] sigma[Log(NH)]"
        f.write('Cross-validation score for NH: %.3f +- %.3f\n' % (
            scores[i, 5], 2. * scores[i, 6]))
        np.savetxt(os.path.join(output_folder,
                                'output_feature_importances_NH.dat'),
                   np.vstack((data[0], importances[2::5, :])), fmt='%.5f')
        if additional_files:
            out_ml += [
                np.log10(np.mean(10 ** matrix_ml[:, 2], axis=1)),
                np.log10(np.median(10 ** matrix_ml[:, 2], axis=1)),
                np.std(matrix_ml[:, 2], axis=1)
            ]
    if "U" in out_labels:
        out_header += " mean[Log(U)] median[Log(U)] sigma[Log(U)]"
        f.write('Cross-validation score for U:  %.3f +- %.3f\n' % (
            scores[i, 7], 2. * scores[i, 8]))
        np.savetxt(os.path.join(output_folder,
                                'output_feature_importances_U.dat'),
                   np.vstack((data[0], importances[3::5, :])), fmt='%.5f')
        if additional_files:
            out_ml += [
                np.log10(np.mean(10 ** matrix_ml[:, 3], axis=1)),
                np.log10(np.median(10 ** matrix_ml[:, 3], axis=1)),
                np.std(matrix_ml[:, 3], axis=1)
            ]
    if "Z" in out_labels:
        out_header += " mean[Log(Z)] median[Log(Z)] sigma[Log(Z)]"
        f.write('Cross-validation score for Z:  %.3f +- %.3f\n' % (
            scores[i, 9], 2. * scores[i, 10]))
        np.savetxt(os.path.join(output_folder,
                                'output_feature_importances_Z.dat'),
                   np.vstack((data[0], importances[4::5, :])), fmt='%.5f')
        if additional_files:
            out_ml += [
                np.log10(np.mean(10 ** matrix_ml[:, 4], axis=1)),
                np.log10(np.median(10 ** matrix_ml[:, 4], axis=1)),
                np.std(matrix_ml[:, 4], axis=1)
            ]

    f.close()

    ##########################################################
    # Outputs relative to the Machine Learning determination #
    ##########################################################
    if additional_files:
        write_output = np.vstack(tuple(out_ml)).T
    else:
        write_output = np.column_stack((model_ids, matrix_ml))

    np.savetxt(os.path.join(output_folder, 'output_ml.dat'), write_output,
               header=out_header,
               fmt='%.5f')

    ##################
    # Optional files #
    ##################
    if additional_files:
        if "G0" in out_labels:
            np.savetxt(os.path.join(output_folder, 'output_pred_G0.dat'),
                       preds[0::5, :],
                       fmt='%.5f')
            np.savetxt(os.path.join(output_folder, 'output_true_G0.dat'),
                       trues[0::5, :],
                       fmt='%.5f')
            np.savetxt(os.path.join(output_folder, 'output_pdf_G0.dat'),
                       matrix_ml[:, 0],
                       fmt='%.5f')

        if "n" in out_labels:
            np.savetxt(os.path.join(output_folder, 'output_pred_n.dat'),
                       preds[1::5, :],
                       fmt='%.5f')
            np.savetxt(os.path.join(output_folder, 'output_true_n.dat'),
                       trues[1::5, :],
                       fmt='%.5f')
            np.savetxt(os.path.join(output_folder, 'output_pdf_n.dat'),
                       matrix_ml[:, 1],
                       fmt='%.5f')

        if "NH" in out_labels:
            np.savetxt(os.path.join(output_folder, 'output_pred_NH.dat'),
                       preds[2::5, :],
                       fmt='%.5f')
            np.savetxt(os.path.join(output_folder, 'output_true_NH.dat'),
                       trues[2::5, :],
                       fmt='%.5f')
            np.savetxt(os.path.join(output_folder, 'output_pdf_NH.dat'),
                       matrix_ml[:, 2],
                       fmt='%.5f')

        if "U" in out_labels:
            np.savetxt(os.path.join(output_folder, 'output_pred_U.dat'),
                       preds[3::5, :],
                       fmt='%.5f')
            np.savetxt(os.path.join(output_folder, 'output_true_U.dat'),
                       trues[3::5, :],
                       fmt='%.5f')
            np.savetxt(os.path.join(output_folder, 'output_pdf_U.dat'),
                       matrix_ml[:, 3],
                       fmt='%.5f')

        if "Z" in out_labels:
            np.savetxt(os.path.join(output_folder, 'output_pred_Z.dat'),
                       preds[4::5, :],
                       fmt='%.5f')
            np.savetxt(os.path.join(output_folder, 'output_true_Z.dat'),
                       trues[4::5, :],
                       fmt='%.5f')
            np.savetxt(os.path.join(output_folder, 'output_pdf_Z.dat'),
                       matrix_ml[:, 4],
                       fmt='%.5f')

    #####################
    # Additional labels #
    #####################
    # This creates arrays useful to save the output for the feature importances of the 'additional labels'
    importances_AV = np.zeros(len(data[0]))
    importances_fesc = np.zeros(len(data[0]))
    if verbose:
        print 'Starting of Machine Learning algorithm for the additional labels...'

    start_time = time.time()

    ########################################################
    # Definition of additional labels for Machine Learning #
    #         (just change the last two of them)           #
    ########################################################
    labels[:, -2:] = np.loadtxt('library/additional_labels.dat')
    # This code is inserted in order to work with logarithms!
    # If there is a zero, we substitute it with 1e-9
    labels[labels[:, -2] == 0, -2] = 1e-9
    labels[labels[:, -1] == 0, -1] = 1e-9
    labels[:, -2] = np.log10(labels[:, -2])
    labels[:, -1] = np.log10(labels[:, -1])
    labels_train = labels[:limit, :]
    labels_test = labels[limit:, :]

    ######################################
    # Initialization of arrays and lists #
    ######################################
    if additional_files:
        AV = np.zeros(shape=(len(data[1:]), n_repetition))
        fesc = np.zeros(shape=(len(data[1:]), n_repetition))
    ##############################################################
    # Searching for values of the additional physical properties #
    ##############################################################

    pool = multiprocessing.Pool(processes=n_proc)
    main_algorithm_additional = partial(main_algorithm_additional_to_pool,
                                        models=models, unique_id=unique_id,
                                        initial=initial, limit=limit
                                        , features=features,
                                        labels_train=labels_train,
                                        labels_test=labels_test
                                        , labels=labels, regr=regr,
                                        line_labels=line_labels
                                        , AV=AV, fesc=fesc
                                        , importances_AV=importances_AV,
                                        importances_fesc=importances_fesc
                                        , filename_int=filename_int,
                                        filename_err=filename_err,
                                        n_repetition=n_repetition,
                                        additional_files=additional_files
                                        )
    results = pool.map(main_algorithm_additional,
                       np.arange(1, np.max(unique_id.astype(int)) + 1, 1))
    pool.close()
    pool.join()
    end_time = time.time()
    if verbose:
        print 'Elapsed time for ML:', (end_time - start_time)
        print ''
        print 'Writing output files for the additional labels...'

    ###########################################
    # Rearrange based on the find_ids indexes #
    ###########################################
    sigmas = np.array(
        list(chain.from_iterable(np.array(results)[:, 0]))).reshape(
        len(unique_id.astype(int)), 2)
    scores = np.array(
        list(chain.from_iterable(np.array(results)[:, 1]))).reshape(
        len(unique_id.astype(int)), 5)
    importances = np.array(list(chain.from_iterable(np.array(results)[:, 6])))
    trues = np.array(list(chain.from_iterable(np.array(results)[:, 7])))
    preds = np.array(list(chain.from_iterable(np.array(results)[:, 8])))
    list_of_lines = np.array(results)[:, 2]
    # find_ids are usefult to reorder the matrix with the ML determinations
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
            len(data[1:]), 6)
        # Rearrange the matrix based on the find_ids indexes
        matrix_ml = np.zeros(shape=temp_matrix_ml.shape)
        for i in xrange(len(matrix_ml)):
            matrix_ml[find_ids[i], :] = temp_matrix_ml[i, :]
    # Rearrange the model_ids based on the find_ids indexes
    model_ids = np.zeros(len(temp_model_ids))
    for i in xrange(len(temp_model_ids)):
        model_ids[find_ids[i]] = temp_model_ids[i]

    #########################################
    # Write information on different models #
    #########################################
    f = open(os.path.join(output_folder, 'model_ids_additional.dat'), 'w+')
    for i in xrange(len(sigmas)):
        f.write('##############################\n')
        f.write('Id model: %d\n' % (i + 1))

        if "Av" in out_additional_labels:
            f.write('Standard deviation of Av:        %.3f\n' % sigmas[i, 0])
            f.write('Cross-validation score for Av:   %.3f +- %.3f\n' % (
                scores[i, 1], 2. * scores[i, 2]))

        if "fesc" in out_additional_labels:
            f.write('Standard deviation of fesc:      %.3f\n' % sigmas[i, 1])
            f.write('Cross-validation score for fesc: %.3f +- %.3f\n' % (
                scores[i, 3], 2. * scores[i, 4]))

        f.write('List of input lines:\n')
        f.write('%s\n' % list_of_lines[i])
    f.write('##############################\n')

    ##########################################################
    # Outputs relative to the Machine Learning determination #
    ##########################################################
    out_header = "id_model"
    out_ml = [model_ids]

    if "Av" in out_additional_labels:
        out_header += " mean[Av] median[Av] sigma[Av]"
        out_ml += [np.mean(matrix_ml[:, 0], axis=1),
                   np.median(matrix_ml[:, 0], axis=1),
                   np.std(matrix_ml[:, 0], axis=1)]
        np.savetxt(os.path.join(output_folder,
                                'output_feature_importances_Av.dat'),
                   np.vstack((data[0], importances[0::2, :])), fmt='%.5f')
    if "fesc" in out_additional_labels:
        out_header += " id_model mean[fesc] median[fesc]"
        out_ml += [np.mean(matrix_ml[:, 1], axis=1),
                   np.median(matrix_ml[:, 1], axis=1),
                   np.std(matrix_ml[:, 1], axis=1)]
        np.savetxt(os.path.join(output_folder,
                                'output_feature_importances_fesc.dat'),
                   np.vstack((data[0], importances[1::2, :])), fmt='%.5f')
    if additional_files:
        write_output = np.vstack(tuple(out_ml)).T
    else:
        write_output = np.column_stack((model_ids, matrix_ml))

    f.close()

    np.savetxt(os.path.join(output_folder, 'output_ml_additional.dat'),
               write_output,
               header=" sigma[fesc]",
               fmt='%.5f')

    ##################
    # Optional files #
    ##################
    if additional_files:
        if "Av" in out_additional_labels:
            np.savetxt(os.path.join(output_folder, 'output_pred_Av.dat'),
                       preds[0::2, :],
                       fmt='%.5f')
            np.savetxt(os.path.join(output_folder, 'output_true_Av.dat'),
                       trues[0::2, :],
                       fmt='%.5f')
            np.savetxt(os.path.join(output_folder, 'output_pdf_Av.dat'),
                       matrix_ml[:, 0],
                       fmt='%.5f')

        if "fesc" in out_additional_labels:
            np.savetxt(os.path.join(output_folder, 'output_pred_fesc.dat'),
                       preds[1::2, :],
                       fmt='%.5f')
            np.savetxt(os.path.join(output_folder, 'output_true_fesc.dat'),
                       trues[1::2, :],
                       fmt='%.5f')
            np.savetxt(os.path.join(output_folder, 'output_pdf_fesc.dat'),
                       matrix_ml[:, 1],
                       fmt='%.5f')
