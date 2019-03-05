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

    # ML
    if g0:
        [g0_true, g0_pred, sigma_g0] = error_estimation(features_train,
                                                        features_test,
                                                        labels_train[:, 0],
                                                        labels_test[:, 0], regr)
        [model_g0, imp_g0, score_g0, std_g0] = machine_learning(
            features[:, initial[mask][0]], labels, 0, regr)
    else:
        g0_true, g0_pred, sigma_g0, model_g0, imp_g0, score_g0, std_g0 = \
            None, None, None, None, None, None, None

    if n:
        [n_true, n_pred, sigma_n] = error_estimation(features_train, features_test,
                                                     labels_train[:, 1],
                                                     labels_test[:, 1], regr)
        [model_n, imp_n, score_n, std_n] = machine_learning(
            features[:, initial[mask][0]], labels, 1, regr)
    else:
        n_true, n_pred, sigma_n, model_n, imp_n, score_n, std_n = \
            None, None, None, None, None, None, None

    if NH:
        [NH_true, NH_pred, sigma_NH] = error_estimation(features_train,
                                                        features_test,
                                                        labels_train[:, 2],
                                                        labels_test[:, 2], regr)
        [model_NH, imp_NH, score_NH, std_NH] = machine_learning(
            features[:, initial[mask][0]], labels, 2, regr)
    else:
        NH_true, NH_pred, sigma_NH, model_NH, imp_NH, score_NH, std_NH = \
            None, None, None, None, None, None, None

    if U:
        [U_true, U_pred, sigma_U] = error_estimation(features_train, features_test,
                                                     labels_train[:, 3],
                                                     labels_test[:, 3], regr)
        [model_U, imp_U, score_U, std_U] = machine_learning(
            features[:, initial[mask][0]], labels, 3, regr)
    else:
        U_true, U_pred, sigma_U, model_U, imp_U, score_U, std_U = \
            None, None, None, None, None, None, None

    if Z:
        [Z_true, Z_pred, sigma_Z] = error_estimation(features_train, features_test,
                                                     labels_train[:, 4],
                                                     labels_test[:, 4], regr)
        [model_Z, imp_Z, score_Z, std_Z] = machine_learning(
            features[:, initial[mask][0]], labels, 4, regr)
    else:
        Z_true, Z_pred, sigma_Z, model_Z, imp_Z, score_Z, std_Z = \
            None, None, None, None, None, None, None

    if AV:
        [AV_true, AV_pred, sigma_AV] = error_estimation(features_train,
                                                        features_test,
                                                        labels_train[:, 5],
                                                        labels_test[:, 5], regr)
        [model_AV, imp_AV, score_AV, std_AV] = machine_learning(
            features[:, initial[mask][0]], labels, 5, regr)
    else:
        AV_true, AV_pred, sigma_AV, model_AV, imp_AV, score_AV, std_AV = \
            None, None, None, None, None, None, None

    if fesc:
        [fesc_true, fesc_pred, sigma_fesc] = error_estimation(features_train,
                                                              features_test,
                                                              labels_train[:, 6],
                                                              labels_test[:, 6],
                                                              regr)
        [model_fesc, imp_fesc, score_fesc, std_fesc] = machine_learning(
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
            if model_g0:
                g0[mask[0][el], :] = model_g0.predict(new_data[el::len(mask[0])])
            if model_n:
                n[mask[0][el], :] = model_n.predict(new_data[el::len(mask[0])])
            if model_NH:
                NH[mask[0][el], :] = model_NH.predict(new_data[el::len(mask[0])])
            if model_U:
                U[mask[0][el], :] = model_U.predict(new_data[el::len(mask[0])])
            if model_Z:
                Z[mask[0][el], :] = model_Z.predict(new_data[el::len(mask[0])])
            if model_AV:
                AV[mask[0][el], :] = model_AV.predict(new_data[el::len(mask[0])])
            if model_fesc:
                fesc[mask[0][el], :] = model_fesc.predict(new_data[el::len(mask[0])])

            # Model ids
            id_model.append(i)
            index_find.append(mask[0][el])

            new_row = [None] * 7
            if g0:
                new_row[0] = g0[mask[0][el], :]
            if n:
                new_row[1] = n[mask[0][el], :]
            if NH:
                new_row[2] = NH[mask[0][el], :]
            if U:
                new_row[3] = U[mask[0][el], :]
            if Z:
                new_row[4] = Z[mask[0][el], :]
            if AV:
                new_row[5] = AV[mask[0][el], :]
            if fesc:
                new_row[6] = fesc[mask[0][el], :]

            matrix_mms.append(new_row)
    else:
        for el in xrange(len(mask[0])):
            result = np.zeros((len(new_data[el::len(mask[0])]), 7))
            if model_g0:
                result[:, 0] = model_g0.predict(new_data[el::len(mask[0])])
            if model_n:
                result[:, 1] = model_n.predict(new_data[el::len(mask[0])])
            if model_NH:
                result[:, 2] = model_NH.predict(new_data[el::len(mask[0])])
            if model_U:
                result[:, 3] = model_U.predict(new_data[el::len(mask[0])])
            if model_Z:
                result[:, 4] = model_Z.predict(new_data[el::len(mask[0])])
            if model_AV:
                result[:, 5] = model_AV.predict(new_data[el::len(mask[0])])
            if model_fesc:
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
    if imp_g0:
        importances_g0[initial[mask][0]] = imp_g0
    if imp_n:
        importances_n[initial[mask][0]] = imp_n
    if imp_NH:
        importances_NH[initial[mask][0]] = imp_NH
    if imp_U:
        importances_U[initial[mask][0]] = imp_U
    if imp_Z:
        importances_Z[initial[mask][0]] = imp_Z
    if imp_AV:
        importances_AV[initial[mask][0]] = imp_AV
    if imp_fesc:
        importances_fesc[initial[mask][0]] = imp_fesc

    sigmas = [None] * 7
    scores = [i] + [None] * 7 * 2  # will contain also std
    trues = [None] * 7
    preds = [None] * 7

    if g0_true and g0_pred:
        trues[0] = np.array(g0_true)
        preds[0] = np.array(g0_pred)
        sigmas[0] = sigma_g0
        scores[1] = score_g0
        scores[2] = std_g0
    if n_true and n_pred:
        trues[1] = np.array(n_true)
        preds[1] = np.array(n_pred)
        sigmas[1] = sigma_n
        scores[3] = score_n
        scores[4] = std_n
    if NH_true and NH_pred:
        trues[2] = np.array(NH_true)
        preds[2] = np.array(NH_pred)
        sigmas[2] = sigma_NH
        scores[5] = score_NH
        scores[6] = std_NH
    if U_true and U_pred:
        trues[3] = np.array(U_true)
        preds[3] = np.array(U_pred)
        sigmas[3] = sigma_U
        scores[7] = score_U
        scores[8] = std_U
    if Z_true and Z_pred:
        trues[4] = np.array(Z_true)
        preds[4] = np.array(Z_pred)
        sigmas[4] = sigma_Z
        scores[9] = score_Z
        scores[10] = std_Z
    if AV_true and AV_pred:
        trues[5] = np.array(AV_true)
        preds[5] = np.array(AV_pred)
        sigmas[5] = sigma_AV
        scores[11] = score_AV
        scores[12] = std_AV
    if fesc_true and fesc_pred:
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
           [importances_g0, importances_n, importances_NH, importances_U,
            importances_Z, importances_AV, importances_fesc], \
           trues, \
           preds
