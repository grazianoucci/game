import numpy as np

from ml import error_estimation, machine_learning, realization


def main_algorithm_to_pool(i
                           , models, unique_id, initial, limit
                           , features, labels_train, labels_test
                           , labels, regr, line_labels
                           , g0, n, NH, U, Z, Av, fesc
                           , importances_g0, importances_n, importances_NH,
                           importances_U, importances_Z, importances_Av,
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
    # G/G0: 0, n: 1, NH: 2, U: 3, Z: 4, Av: 5, fesc: 6
    # Definition of training / testing
    features_train = features[:, initial[mask][0]][:limit, :]
    features_test = features[:, initial[mask][0]][limit:, :]

    # ML
    if g0 is not None:
        g0_true, g0_pred, sigma_g0 = error_estimation(features_train,
                                                        features_test,
                                                        labels_train[:, 0],
                                                        labels_test[:, 0], regr)
        model_g0, imp_g0, score_g0, std_g0 = machine_learning(
            features[:, initial[mask][0]], labels, 0, regr)
    else:
        g0_true, g0_pred, sigma_g0, model_g0, imp_g0, score_g0, std_g0 = \
            None, None, None, None, None, None, None

    if n is not None:
        n_true, n_pred, sigma_n = error_estimation(features_train, features_test,
                                                     labels_train[:, 1],
                                                     labels_test[:, 1], regr)
        model_n, imp_n, score_n, std_n = machine_learning(
            features[:, initial[mask][0]], labels, 1, regr)
    else:
        n_true, n_pred, sigma_n, model_n, imp_n, score_n, std_n = \
            None, None, None, None, None, None, None

    if NH is not None:
        NH_true, NH_pred, sigma_NH = error_estimation(features_train,
                                                        features_test,
                                                        labels_train[:, 2],
                                                        labels_test[:, 2], regr)
        model_NH, imp_NH, score_NH, std_NH = machine_learning(
            features[:, initial[mask][0]], labels, 2, regr)
    else:
        NH_true, NH_pred, sigma_NH, model_NH, imp_NH, score_NH, std_NH = \
            None, None, None, None, None, None, None

    if U is not None:
        U_true, U_pred, sigma_U = error_estimation(features_train, features_test,
                                                     labels_train[:, 3],
                                                     labels_test[:, 3], regr)
        model_U, imp_U, score_U, std_U = machine_learning(
            features[:, initial[mask][0]], labels, 3, regr)
    else:
        U_true, U_pred, sigma_U, model_U, imp_U, score_U, std_U = \
            None, None, None, None, None, None, None

    if Z is not None:
        Z_true, Z_pred, sigma_Z = error_estimation(features_train, features_test,
                                                     labels_train[:, 4],
                                                     labels_test[:, 4], regr)
        model_Z, imp_Z, score_Z, std_Z = machine_learning(
            features[:, initial[mask][0]], labels, 4, regr)
    else:
        Z_true, Z_pred, sigma_Z, model_Z, imp_Z, score_Z, std_Z = \
            None, None, None, None, None, None, None

    if Av is not None:
        Av_true, Av_pred, sigma_Av = error_estimation(features_train,
                                                        features_test,
                                                        labels_train[:, 5],
                                                        labels_test[:, 5], regr)
        model_Av, imp_Av, score_Av, std_Av = machine_learning(
            features[:, initial[mask][0]], labels, 5, regr)
    else:
        Av_true, Av_pred, sigma_Av, model_Av, imp_Av, score_Av, std_Av = \
            None, None, None, None, None, None, None

    if fesc is not None:
        fesc_true, fesc_pred, sigma_fesc = error_estimation(features_train,
                                                              features_test,
                                                              labels_train[:, 6],
                                                              labels_test[:, 6],
                                                              regr)
        model_fesc, imp_fesc, score_fesc, std_fesc = machine_learning(
            features[:, initial[mask][0]], labels, 6, regr)
    else:
        fesc_true, fesc_pred, sigma_fesc, model_fesc, imp_fesc, score_fesc, std_fesc = \
            None, None, None, None, None, None, None

    # Bootstrap
    new_data = realization(filename_int, filename_err, n_repetition, mask)[:,
               initial[mask][0]]

    # Prediction of the physical properties
    if additional_files:
        for el in xrange(len(mask[0])):
            if model_g0 is not None:
                g0[mask[0][el], :] = model_g0.predict(new_data[el::len(mask[0])])
            if model_n is not None:
                n[mask[0][el], :] = model_n.predict(new_data[el::len(mask[0])])
            if model_NH is not None:
                NH[mask[0][el], :] = model_NH.predict(new_data[el::len(mask[0])])
            if model_U is not None:
                U[mask[0][el], :] = model_U.predict(new_data[el::len(mask[0])])
            if model_Z is not None:
                Z[mask[0][el], :] = model_Z.predict(new_data[el::len(mask[0])])
            if model_Av is not None:
                Av[mask[0][el], :] = model_Av.predict(new_data[el::len(mask[0])])
            if model_fesc is not None:
                fesc[mask[0][el], :] = model_fesc.predict(new_data[el::len(mask[0])])

            # Model ids
            id_model.append(i)
            index_find.append(mask[0][el])

            new_row = [np.zeros(n_repetition)] * 7  # default

            if g0 is not None:
                new_row[0] = g0[mask[0][el], :]
            if n is not None:
                new_row[1] = n[mask[0][el], :]
            if NH is not None:
                new_row[2] = NH[mask[0][el], :]
            if U is not None:
                new_row[3] = U[mask[0][el], :]
            if Z is not None:
                new_row[4] = Z[mask[0][el], :]
            if Av is not None:
                new_row[5] = Av[mask[0][el], :]
            if fesc is not None:
                new_row[6] = fesc[mask[0][el], :]

            matrix_mms.append(new_row)
    else:
        for el in xrange(len(mask[0])):
            result = np.zeros((len(new_data[el::len(mask[0])]), 7))

            if model_g0 is not None:
                result[:, 0] = model_g0.predict(new_data[el::len(mask[0])])
            if model_n is not None:
                result[:, 1] = model_n.predict(new_data[el::len(mask[0])])
            if model_NH is not None:
                result[:, 2] = model_NH.predict(new_data[el::len(mask[0])])
            if model_U is not None:
                result[:, 3] = model_U.predict(new_data[el::len(mask[0])])
            if model_Z is not None:
                result[:, 4] = model_Z.predict(new_data[el::len(mask[0])])
            if model_Av is not None:
                result[:, 5] = model_Av.predict(new_data[el::len(mask[0])])
            if model_fesc is not None:
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

            # Av and fesc do NOT require log
            # todo proposed solution
            vector_mms[15::3] = np.log10(np.mean(10 ** result[:, 5], axis=0))
            vector_mms[16::3] = np.log10(np.median(10 ** result[:, 5], axis=0))

            # todo should have been
            # vector_mms = np.zeros(6)
            # vector_mms[0::3] = np.mean(result, axis=0)
            # vector_mms[1::3] = np.median(result, axis=0)
            # vector_mms[2::3] = np.std(result, axis=0)
            # end todo

            matrix_mms.append(vector_mms)

    # Importance matrixes
    if imp_g0 is not None:
        importances_g0[initial[mask][0]] = imp_g0
    if imp_n is not None:
        importances_n[initial[mask][0]] = imp_n
    if imp_NH is not None:
        importances_NH[initial[mask][0]] = imp_NH
    if imp_U is not None:
        importances_U[initial[mask][0]] = imp_U
    if imp_Z is not None:
        importances_Z[initial[mask][0]] = imp_Z
    if imp_Av is not None:
        importances_Av[initial[mask][0]] = imp_Av
    if imp_fesc is not None:
        importances_fesc[initial[mask][0]] = imp_fesc

    sigmas = [0.0] * 7
    scores = [i] + [0.0] * 7 * 2  # will contain also std

    len_arrays = features.shape[0] / 10
    trues = [np.zeros(len_arrays)] * 7
    preds = [np.zeros(len_arrays)] * 7

    len_importances = line_labels.shape[0]
    importances = [np.zeros(len_importances)] * 7

    if g0_true is not None and g0_pred is not None and importances_g0 is not \
            None:
        importances[0] = importances_g0
        trues[0] = np.array(g0_true)
        preds[0] = np.array(g0_pred)
        sigmas[0] = sigma_g0
        scores[1] = score_g0
        scores[2] = std_g0

    if n_true is not None and n_pred is not None and importances_n is not None:
        importances[1] = importances_n
        trues[1] = np.array(n_true)
        preds[1] = np.array(n_pred)
        sigmas[1] = sigma_n
        scores[3] = score_n
        scores[4] = std_n

    if NH_true is not None and NH_pred is not None and importances_NH is not \
            None:
        importances[2] = importances_NH
        trues[2] = np.array(NH_true)
        preds[2] = np.array(NH_pred)
        sigmas[2] = sigma_NH
        scores[5] = score_NH
        scores[6] = std_NH

    if U_true is not None and U_pred is not None and importances_U is not None:
        importances[3] = importances_U
        trues[3] = np.array(U_true)
        preds[3] = np.array(U_pred)
        sigmas[3] = sigma_U
        scores[7] = score_U
        scores[8] = std_U
    else:  # todo g0 -> U
        trues[3] = np.array(g0_true)
        preds[3] = np.array(g0_pred)
        sigmas[3] = sigma_g0
        scores[7] = score_g0
        scores[8] = std_g0

    if Z_true is not None and Z_pred is not None and importances_Z is not None:
        importances[4] = importances_Z
        trues[4] = np.array(Z_true)
        preds[4] = np.array(Z_pred)
        sigmas[4] = sigma_Z
        scores[9] = score_Z
        scores[10] = std_Z

    if Av_true is not None and Av_pred is not None and importances_Av is not None:
        importances[5] = importances_Av
        trues[5] = np.array(Av_true)
        preds[5] = np.array(Av_pred)
        sigmas[5] = sigma_Av
        scores[11] = score_Av
        scores[12] = std_Av

    if fesc_true is not None and fesc_pred is not None and importances_fesc \
            is not None:
        importances[6] = importances_fesc
        trues[6] = np.array(fesc_true)
        preds[6] = np.array(fesc_pred)
        sigmas[6] = sigma_fesc
        scores[13] = score_fesc
        scores[14] = std_fesc

    print 'Model', str(int(i)) + '/' + str(int(np.max(unique_id))), 'completed...'

    return sigmas, \
           scores, \
           line_labels[initial[mask][0]], \
           index_find, id_model, matrix_mms, \
           importances, \
           trues, \
           preds
