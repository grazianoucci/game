import numpy as np

from ml import error_estimation, machine_learning, realization


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
