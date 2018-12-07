import numpy as np

from ml import error_estimation, machine_learning, realization


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
