import copy
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
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Normalizer


def stat_library():
    dir_path = 'library/'
    directory = os.path.dirname(dir_path)
    try:
        os.stat(directory)
    except:
        urllib.urlretrieve(
            "http://cosmology.sns.it/library_game/library.tar.gz",
            filename="library.tar.gz")
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


def realization(filename_int, filename_err, n_rep, mask):
    data = np.loadtxt(filename_int)[1:, :][mask]
    errors = np.loadtxt(filename_err)[1:, :][mask]
    # Be careful: in the first row of 'errors' there are wavelenghts!
    # Find the position where error = -99 (i.e. the upper limits)
    mask_upper = [errors == -99]
    # Find the position where error = 0 (i.e. the errors associated to the missing values)
    mask_miss = [errors == 0.0]
    # Assign a positive value where errors = -99 or 0 JUST TO COMPUTE REPETITION WITHOUT ERRORS
    errors[errors == -99] = 0.1
    errors[errors == 0.0] = 0.1
    # Compute the 'repetition matrix'
    repetition = np.random.normal(loc=np.tile(data, (n_rep, 1)),
                                  scale=np.tile(errors, (n_rep, 1)))
    # For upper limits assign to repetition a random number between 0 and the value itself
    tiled_mask_upper = np.tile(mask_upper, (n_rep, 1))
    tiled_mask_miss = np.tile(mask_miss, (n_rep, 1))
    tiled_data = np.tile(data, (n_rep, 1))
    repetition[tiled_mask_upper[0]] = np.random.uniform(0, tiled_data[
        tiled_mask_upper[0]])
    repetition[tiled_mask_miss[0]] = 0.0
    mms = Normalizer(norm='max')
    repetition = mms.fit_transform(repetition)
    return repetition


def read_library_file(filename_library):
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


def determination_models(data):
    initial = [data != 0][0]
    models = np.zeros(len(initial))
    mask = np.where((initial == initial[0]).all(axis=1))[0]
    models[mask] = 1
    check = True;
    i = 2
    while check:
        if (len(models[models == 0]) == 0):
            check = False
        else:
            mask = \
                np.where(
                    (initial == initial[np.argmax(models == 0)]).all(axis=1))[
                    0]
            models[mask] = i
            i += 1
    return initial, models, np.unique(models)


def error_estimation(feat_train, feat_test, lab_train, lab_test, ml_regr):
    ml_regr.fit(feat_train, lab_train)
    y = ml_regr.predict(feat_test)
    sigma = np.std(np.double(lab_test) - y)
    return np.double(lab_test), y, sigma


def machine_learning(feat, lab, physical_p, ml_regr):
    # Model
    model = ml_regr.fit(feat, lab[:, physical_p])
    # Feature importances
    importances = model.feature_importances_
    # Cross-validation score
    score = cross_val_score(ml_regr, feat, lab[:, physical_p], cv=5)
    return copy.copy(model), importances, np.mean(score), np.std(score)


def main_algorithm_to_pool(i
                           , models, unique_id, initial, limit
                           , features, labels_train, labels_test
                           , labels, regr, line_labels
                           , g0, n, NH, U, Z
                           , importances_g0, importances_n, importances_NH,
                           importances_U, importances_Z
                           , filename_int, filename_err, n_repetition,
                           additional_files
                           ):
    mask = np.where(models == unique_id[i - 1])
    # matrix_mms is useful to save physical properties
    matrix_mms = []
    # index_find helps to keep trace of the indexes
    index_find = []
    id_model = []
    # Indexes for the labels:
    # G/G0: 0, n: 1, NH: 2, U: 3, Z: 4
    # Definition of training / testing
    features_train = features[:, initial[mask][0]][:limit, :]
    features_test = features[:, initial[mask][0]][limit:, :]
    # ML error estimation
    [g0_true, g0_pred, sigma_g0] = error_estimation(features_train,
                                                    features_test,
                                                    labels_train[:, 0],
                                                    labels_test[:, 0], regr)
    [n_true, n_pred, sigma_n] = error_estimation(features_train, features_test,
                                                 labels_train[:, 1],
                                                 labels_test[:, 1], regr)
    [NH_true, NH_pred, sigma_NH] = error_estimation(features_train,
                                                    features_test,
                                                    labels_train[:, 2],
                                                    labels_test[:, 2], regr)
    [U_true, U_pred, sigma_U] = error_estimation(features_train, features_test,
                                                 labels_train[:, 3],
                                                 labels_test[:, 3], regr)
    [Z_true, Z_pred, sigma_Z] = error_estimation(features_train, features_test,
                                                 labels_train[:, 4],
                                                 labels_test[:, 4], regr)
    # Function calls for the machine learning routines
    [model_g0, imp_g0, score_g0, std_g0] = machine_learning(
        features[:, initial[mask][0]], labels, 0, regr)
    [model_n, imp_n, score_n, std_n] = machine_learning(
        features[:, initial[mask][0]], labels, 1, regr)
    [model_NH, imp_NH, score_NH, std_NH] = machine_learning(
        features[:, initial[mask][0]], labels, 2, regr)
    [model_U, imp_U, score_U, std_U] = machine_learning(
        features[:, initial[mask][0]], labels, 3, regr)
    [model_Z, imp_Z, score_Z, std_Z] = machine_learning(
        features[:, initial[mask][0]], labels, 4, regr)
    # Bootstrap
    new_data = realization(filename_int, filename_err, n_repetition, mask)[:,
               initial[mask][0]]
    # Prediction of the physical properties
    if additional_files == 'y':
        for el in xrange(len(mask[0])):
            g0[mask[0][el], :] = model_g0.predict(new_data[el::len(mask[0])])
            n[mask[0][el], :] = model_n.predict(new_data[el::len(mask[0])])
            NH[mask[0][el], :] = model_NH.predict(new_data[el::len(mask[0])])
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
    if additional_files == 'n':
        for el in xrange(len(mask[0])):
            result = np.zeros((len(new_data[el::len(mask[0])]), 5))
            result[:, 0] = model_g0.predict(new_data[el::len(mask[0])])
            result[:, 1] = model_n.predict(new_data[el::len(mask[0])])
            result[:, 2] = model_NH.predict(new_data[el::len(mask[0])])
            result[:, 3] = model_U.predict(new_data[el::len(mask[0])])
            result[:, 4] = model_Z.predict(new_data[el::len(mask[0])])
            # Model ids
            id_model.append(i)
            index_find.append(mask[0][el])
            #
            vector_mms = np.zeros(15)
            vector_mms[0::3] = np.log10(np.mean(10 ** result, axis=0))
            vector_mms[1::3] = np.log10(np.median(10 ** result, axis=0))
            vector_mms[2::3] = np.std(result, axis=0)
            matrix_mms.append(vector_mms)

    # Importance matrixes
    importances_g0[initial[mask][0]] = imp_g0
    importances_n[initial[mask][0]] = imp_n
    importances_NH[initial[mask][0]] = imp_NH
    importances_U[initial[mask][0]] = imp_U
    importances_Z[initial[mask][0]] = imp_Z

    # Print message
    print 'Model', str(int(i)) + '/' + str(
        int(np.max(unique_id))), 'completed...'

    # Returns for the parallelization
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


def main_algorithm_additional_to_pool(i
                                      , models, unique_id, initial, limit
                                      , features, labels_train, labels_test
                                      , labels, regr, line_labels
                                      , AV, fesc
                                      , importances_AV, importances_fesc
                                      , filename_int, filename_err,
                                      n_repetition, additional_files
                                      ):
    mask = np.where(models == unique_id[i - 1])

    # matrix_mms is useful to save physical properties
    matrix_mms = []

    # index_find helps to keep trace of the indexes
    index_find = []
    id_model = []

    # Indexes for the additional labels:
    # AV: 3, fesc: 4
    # Definition of training / testing
    features_train = features[:, initial[mask][0]][:limit, :]
    features_test = features[:, initial[mask][0]][limit:, :]

    # ML error estimation
    [AV_true, AV_pred, sigma_AV] = error_estimation(features_train,
                                                    features_test,
                                                    labels_train[:, 3],
                                                    labels_test[:, 3], regr)
    [fesc_true, fesc_pred, sigma_fesc] = error_estimation(features_train,
                                                          features_test,
                                                          labels_train[:, 4],
                                                          labels_test[:, 4],
                                                          regr)
    # Function calls for the machine learning routines
    [model_AV, imp_AV, score_AV, std_AV] = machine_learning(
        features[:, initial[mask][0]], labels, 3, regr)
    [model_fesc, imp_fesc, score_fesc, std_fesc] = machine_learning(
        features[:, initial[mask][0]], labels, 4, regr)
    # Bootstrap
    new_data = realization(filename_int, filename_err, n_repetition, mask)[:,
               initial[mask][0]]
    # Prediction of the physical properties
    if additional_files == 'y':
        for el in xrange(len(mask[0])):
            AV[mask[0][el], :] = model_AV.predict(new_data[el::len(mask[0])])
            fesc[mask[0][el], :] = model_fesc.predict(
                new_data[el::len(mask[0])])
            # Model ids
            id_model.append(i)
            index_find.append(mask[0][el])
            matrix_mms.append([AV[mask[0][el], :],
                               fesc[mask[0][el], :]])
    else:
        for el in xrange(len(mask[0])):
            result = np.zeros((len(new_data[el::len(mask[0])]), 2))
            result[:, 0] = model_AV.predict(new_data[el::len(mask[0])])
            result[:, 1] = model_fesc.predict(new_data[el::len(mask[0])])
            # Model ids
            id_model.append(i)
            index_find.append(mask[0][el])
            #
            vector_mms = np.zeros(6)
            vector_mms[0::3] = np.mean(result, axis=0)
            vector_mms[1::3] = np.median(result, axis=0)
            vector_mms[2::3] = np.std(result, axis=0)
            matrix_mms.append(vector_mms)

    # Importance matrixes
    importances_AV[initial[mask][0]] = imp_AV
    importances_fesc[initial[mask][0]] = imp_fesc

    # Print message
    print 'Model', str(int(i)) + '/' + str(
        int(np.max(unique_id))), 'completed...'

    # Returns for the parallelization
    return [sigma_AV, sigma_fesc], \
           [i, score_AV, std_AV, score_fesc, std_fesc], \
           line_labels[initial[mask][0]], \
           index_find, id_model, matrix_mms, \
           [importances_AV, importances_fesc], \
           [np.array(AV_true), np.array(fesc_true)], \
           [np.array(AV_pred), np.array(fesc_pred)]


def run_game(
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
                             n_estimators=2,
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
    output, line_labels = read_library_file(filename_library)
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
    if additional_files == 'y':
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
    if additional_files == 'y':
        temp_matrix_ml = np.array(
            list(chain.from_iterable(np.array(results)[:, 5])))

    # Rearrange the matrix based on the find_ids indexes
    matrix_ml = np.zeros(shape=temp_matrix_ml.shape)
    for i in xrange(len(matrix_ml)):
        matrix_ml[find_ids[i], :] = temp_matrix_ml[i, :]
    if additional_files == 'n':
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
    f = open(output_folder + 'model_ids.dat', 'w+')
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
    f.close()

    out_header = "id_model"
    out_ml = [model_ids]

    if "G0" in out_labels:
        out_header += " mean[Log(G0)] median[Log(G0)] sigma[Log(G0)]"
        f.write('Cross-validation score for G0: %.3f +- %.3f\n' % (
            scores[i, 1], 2. * scores[i, 2]))
        np.savetxt(output_folder + 'output_feature_importances_G0.dat',
                   np.vstack((data[0], importances[0::5, :])), fmt='%.5f')
        out_ml += [
            np.log10(np.mean(10 ** matrix_ml[:, 0], axis=1)),
            np.log10(np.median(10 ** matrix_ml[:, 0], axis=1)),
            np.std(matrix_ml[:, 0], axis=1),
        ]
    if "n" in out_labels:
        out_header += " mean[Log(n)] median[Log(n)] sigma[Log(n)]"
        f.write('Cross-validation score for n:  %.3f +- %.3f\n' % (
            scores[i, 3], 2. * scores[i, 4]))
        np.savetxt(output_folder + 'output_feature_importances_n.dat',
                   np.vstack((data[0], importances[1::5, :])), fmt='%.5f')
        out_ml += [
            np.log10(np.mean(10 ** matrix_ml[:, 1], axis=1)),
            np.log10(np.median(10 ** matrix_ml[:, 1], axis=1)),
            np.std(matrix_ml[:, 1], axis=1)
        ]
    if "NH" in out_labels:
        out_header += " mean[Log(NH)] median[Log(NH)] sigma[Log(NH)]"
        f.write('Cross-validation score for NH: %.3f +- %.3f\n' % (
            scores[i, 5], 2. * scores[i, 6]))
        np.savetxt(output_folder + 'output_feature_importances_NH.dat',
                   np.vstack((data[0], importances[2::5, :])), fmt='%.5f')
        out_ml += [
            np.log10(np.mean(10 ** matrix_ml[:, 2], axis=1)),
            np.log10(np.median(10 ** matrix_ml[:, 2], axis=1)),
            np.std(matrix_ml[:, 2], axis=1)
        ]
    if "U" in out_labels:
        out_header += " mean[Log(U)] median[Log(U)] sigma[Log(U)]"
        f.write('Cross-validation score for U:  %.3f +- %.3f\n' % (
            scores[i, 7], 2. * scores[i, 8]))
        np.savetxt(output_folder + 'output_feature_importances_U.dat',
                   np.vstack((data[0], importances[3::5, :])), fmt='%.5f')
        out_ml += [
            np.log10(np.mean(10 ** matrix_ml[:, 3], axis=1)),
            np.log10(np.median(10 ** matrix_ml[:, 3], axis=1)),
            np.std(matrix_ml[:, 3], axis=1)
        ]
    if "Z" in out_labels:
        out_header += " mean[Log(Z)] median[Log(Z)] sigma[Log(Z)]"
        f.write('Cross-validation score for Z:  %.3f +- %.3f\n' % (
            scores[i, 9], 2. * scores[i, 10]))
        np.savetxt(output_folder + 'output_feature_importances_Z.dat',
                   np.vstack((data[0], importances[4::5, :])), fmt='%.5f')
        out_ml += [
            np.log10(np.mean(10 ** matrix_ml[:, 4], axis=1)),
            np.log10(np.median(10 ** matrix_ml[:, 4], axis=1)),
            np.std(matrix_ml[:, 4], axis=1)
        ]

    ##########################################################
    # Outputs relative to the Machine Learning determination #
    ##########################################################
    if additional_files == 'y':
        write_output = np.vstack(tuple(out_ml)).T
    else:
        write_output = np.column_stack((model_ids, matrix_ml))

    np.savetxt(output_folder + 'output_ml.dat', write_output,
               header=out_header,
               fmt='%.5f')

    ##################
    # Optional files #
    ##################
    if additional_files == 'y':
        if "G0" in out_labels:
            np.savetxt(output_folder + 'output_pred_G0.dat', preds[0::5, :],
                       fmt='%.5f')
            np.savetxt(output_folder + 'output_true_G0.dat', trues[0::5, :],
                       fmt='%.5f')
            np.savetxt(output_folder + 'output_pdf_G0.dat', matrix_ml[:, 0],
                       fmt='%.5f')

        if "n" in out_labels:
            np.savetxt(output_folder + 'output_pred_n.dat', preds[1::5, :],
                       fmt='%.5f')
            np.savetxt(output_folder + 'output_true_n.dat', trues[1::5, :],
                       fmt='%.5f')
            np.savetxt(output_folder + 'output_pdf_n.dat', matrix_ml[:, 1],
                       fmt='%.5f')

        if "NH" in out_labels:
            np.savetxt(output_folder + 'output_pred_NH.dat', preds[2::5, :],
                       fmt='%.5f')
            np.savetxt(output_folder + 'output_true_NH.dat', trues[2::5, :],
                       fmt='%.5f')
            np.savetxt(output_folder + 'output_pdf_NH.dat', matrix_ml[:, 2],
                       fmt='%.5f')

        if "U" in out_labels:
            np.savetxt(output_folder + 'output_pred_U.dat', preds[3::5, :],
                       fmt='%.5f')
            np.savetxt(output_folder + 'output_true_U.dat', trues[3::5, :],
                       fmt='%.5f')
            np.savetxt(output_folder + 'output_pdf_U.dat', matrix_ml[:, 3],
                       fmt='%.5f')

        if "Z" in out_labels:
            np.savetxt(output_folder + 'output_pred_Z.dat', preds[4::5, :],
                       fmt='%.5f')
            np.savetxt(output_folder + 'output_true_Z.dat', trues[4::5, :],
                       fmt='%.5f')
            np.savetxt(output_folder + 'output_pdf_Z.dat', matrix_ml[:, 4],
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
    if additional_files == 'y':
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
    if additional_files == 'y':
        temp_matrix_ml = np.array(
            list(chain.from_iterable(np.array(results)[:, 5])))
        # Rearrange the matrix based on the find_ids indexes
        matrix_ml = np.zeros(shape=temp_matrix_ml.shape)
        for i in xrange(len(matrix_ml)):
            matrix_ml[find_ids[i], :] = temp_matrix_ml[i, :]
    if additional_files == 'n':
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
    f = open(output_folder + 'model_ids_additional.dat', 'w+')
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
    f.close()

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
        np.savetxt(output_folder + 'output_feature_importances_Av.dat',
                   np.vstack((data[0], importances[0::2, :])), fmt='%.5f')
    if "fesc" in out_additional_labels:
        out_header += " id_model mean[fesc] median[fesc]"
        out_ml += [np.mean(matrix_ml[:, 1], axis=1),
                   np.median(matrix_ml[:, 1], axis=1),
                   np.std(matrix_ml[:, 1], axis=1)]
        np.savetxt(output_folder + 'output_feature_importances_fesc.dat',
                   np.vstack((data[0], importances[1::2, :])), fmt='%.5f')
    if additional_files == 'y':
        write_output = np.vstack(tuple(out_ml)).T
    if additional_files == 'n':
        write_output = np.column_stack((model_ids, matrix_ml))

    np.savetxt(output_folder + 'output_ml_additional.dat', write_output,
               header=" sigma[fesc]",
               fmt='%.5f')

    ##################
    # Optional files #
    ##################
    if additional_files == 'y':
        if "Av" in out_additional_labels:
            np.savetxt(output_folder + 'output_pred_Av.dat', preds[0::2, :],
                       fmt='%.5f')
            np.savetxt(output_folder + 'output_true_Av.dat', trues[0::2, :],
                       fmt='%.5f')
            np.savetxt(output_folder + 'output_pdf_Av.dat', matrix_ml[:, 0],
                       fmt='%.5f')

        if "fesc" in out_additional_labels:
            np.savetxt(output_folder + 'output_pred_fesc.dat', preds[1::2, :],
                       fmt='%.5f')
            np.savetxt(output_folder + 'output_true_fesc.dat', trues[1::2, :],
                       fmt='%.5f')
            # This writes down the output relative to the PDFs of the physical properties
            np.savetxt(output_folder + 'output_pdf_fesc.dat', matrix_ml[:, 1],
                       fmt='%.5f')


def main():
    input_folder = '/home/stefano/Work/sns/game/tests/input/small'
    output_folder = os.path.join(os.getcwd(), 'output')

    stat_library()
    run_game(
        filename_int=os.path.join(input_folder, 'inputs.dat')
        , filename_err=os.path.join(input_folder, 'errors.dat')
        , filename_library=os.path.join(input_folder, 'labels.dat')
        , additional_files='y'
        , n_proc=2
        , n_repetition=10000
        , output_folder=output_folder
        , verbose=True
        , out_labels=["G0", "n", "NH", "U", "Z"]
        , out_additional_labels=["Av", "fesc"]
    )


if __name__ == "__main__":
    main()
