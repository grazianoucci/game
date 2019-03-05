import numpy as np

from ml import error_estimation, machine_learning, realization


def main_algorithm_to_pool(i
                           , models, unique_id, initial, limit
                           , features, labels_train, labels_test
                           , labels, regr, line_labels
                           , g0, n, NH, U, Z, AV, fesc
                           , importances_g0, importances_n, importances_NH,
                           importances_U, importances_Z, importances_AV,
                           importances_fesc
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
    # G/G0: 0, n: 1, NH: 2, U: 3, Z: 4, AV: 5, fesc: 6
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
    [AV_true, AV_pred, sigma_AV] = error_estimation(features_train,
                                                    features_test,
                                                    labels_train[:, 5],
                                                    labels_test[:, 5], regr)
    [fesc_true, fesc_pred, sigma_fesc] = error_estimation(features_train,
                                                          features_test,
                                                          labels_train[:, 6],
                                                          labels_test[:, 6],
                                                          regr)
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
    [model_AV, imp_AV, score_AV, std_AV] = machine_learning(
        features[:, initial[mask][0]], labels, 5, regr)
    [model_fesc, imp_fesc, score_fesc, std_fesc] = machine_learning(
        features[:, initial[mask][0]], labels, 6, regr)

    # Bootstrap
    new_data = realization(filename_int, filename_err, n_repetition, mask)[:,
               initial[mask][0]]

    # Prediction of the physical properties
    if additional_files:
        for el in xrange(len(mask[0])):
            g0[mask[0][el], :] = model_g0.predict(new_data[el::len(mask[0])])
            n[mask[0][el], :] = model_n.predict(new_data[el::len(mask[0])])
            NH[mask[0][el], :] = model_NH.predict(new_data[el::len(mask[0])])
            U[mask[0][el], :] = model_U.predict(new_data[el::len(mask[0])])
            Z[mask[0][el], :] = model_Z.predict(new_data[el::len(mask[0])])
            AV[mask[0][el], :] = model_AV.predict(new_data[el::len(mask[0])])
            fesc[mask[0][el], :] = model_fesc.predict(new_data[el::len(mask[0])])

            # Model ids
            id_model.append(i)
            index_find.append(mask[0][el])
            matrix_mms.append([g0[mask[0][el], :],
                               n[mask[0][el], :],
                               NH[mask[0][el], :],
                               U[mask[0][el], :],
                               Z[mask[0][el], :],
                               AV[mask[0][el], :],
                               fesc[mask[0][el], :]])
    else:
        for el in xrange(len(mask[0])):
            result = np.zeros((len(new_data[el::len(mask[0])]), 7))
            result[:, 0] = model_g0.predict(new_data[el::len(mask[0])])
            result[:, 1] = model_n.predict(new_data[el::len(mask[0])])
            result[:, 2] = model_NH.predict(new_data[el::len(mask[0])])
            result[:, 3] = model_U.predict(new_data[el::len(mask[0])])
            result[:, 4] = model_Z.predict(new_data[el::len(mask[0])])
            result[:, 5] = model_AV.predict(new_data[el::len(mask[0])])
            result[:, 6] = model_fesc.predict(new_data[el::len(mask[0])])

            # Model ids
            id_model.append(i)
            index_find.append(mask[0][el])

            vector_mms = np.zeros(21)
            vector_mms[0::3] = np.log10(np.mean(10 ** result, axis=0))
            vector_mms[1::3] = np.log10(np.median(10 ** result, axis=0))
            vector_mms[2::3] = np.std(result, axis=0)

            print 'result', result.size
            print 'result[:, 5]', result[:, 5].size
            print 'result[:, 6]', result[:, 6].size
            print 'vector_mms', vector_mms.size

            # AV and fesc do NOT require log
            # todo proposed solution
            # vector_mms[15::3] = np.log10(np.mean(10 ** result[:, 5], axis=0))
            # vector_mms[16::3] = np.log10(np.median(10 ** result[:, 5],
            # axis=0))

            # todo should have been
            # vector_mms = np.zeros(6)
            # vector_mms[0::3] = np.mean(result, axis=0)
            # vector_mms[1::3] = np.median(result, axis=0)
            # vector_mms[2::3] = np.std(result, axis=0)
            # end todo

            matrix_mms.append(vector_mms)

    # Importance matrixes
    importances_g0[initial[mask][0]] = imp_g0
    importances_n[initial[mask][0]] = imp_n
    importances_NH[initial[mask][0]] = imp_NH
    importances_U[initial[mask][0]] = imp_U
    importances_Z[initial[mask][0]] = imp_Z
    importances_AV[initial[mask][0]] = imp_AV
    importances_fesc[initial[mask][0]] = imp_fesc

    print 'Model', str(int(i)) + '/' + str(
        int(np.max(unique_id))), 'completed...'

    return [sigma_g0, sigma_n, sigma_NH, sigma_U, sigma_Z, sigma_AV, sigma_fesc], \
           [i, score_g0, std_g0, score_n, std_n, score_NH, std_NH, score_U,
            std_U, score_Z, std_Z, score_AV, std_AV, score_fesc, std_fesc], \
           line_labels[initial[mask][0]], \
           index_find, id_model, matrix_mms, \
           [importances_g0, importances_n, importances_NH, importances_U,
            importances_Z, importances_AV, importances_fesc], \
           [np.array(g0_true), np.array(n_true), np.array(NH_true),
            np.array(U_true), np.array(Z_true), np.array(AV_true), np.array(fesc_true)], \
           [np.array(g0_pred), np.array(n_pred), np.array(NH_pred),
            np.array(U_pred), np.array(Z_pred), np.array(AV_pred), np.array(fesc_pred)]
