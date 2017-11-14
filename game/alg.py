# !/usr/bin/python3
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


""" GAME pooling algorithm """

import numpy as np

from game.ml import realization


def game(
        i, models, unique_id, initial, limit, features,
        labels_train, labels_test, labels, line_labels,
        filename_int, filename_err, n_repetition, optional_files, to_predict
):
    features_to_predict = to_predict.generate_features()
    mask = np.where(models == unique_id[i - 1])
    matrix_mms = []  # matrix_mms is useful to save physical properties
    index_find = []  # index_find helps to keep trace of the indexes
    id_model = []

    # Definition of training / testing
    features_train = features[:, initial[mask][0]][:limit, :]
    features_test = features[:, initial[mask][0]][limit:, :]
    importances = to_predict.generate_importances_arrays()
    sigmas = [
        feature.error_estimate(
            features_train,
            features_test,
            labels_train,
            labels_test
        ) for feature in features_to_predict
    ]  # ML error estimation
    models = [
        feature.train(
            features[:, initial[mask][0]],
            labels
        ) for feature in features_to_predict
    ]  # ML find models

    # Bootstrap
    new_data = realization(
        filename_int, filename_err, n_repetition, mask
    )[:, initial[mask][0]]

    for k in range(len(mask[0])):
        id_model.append(i)  # Model ids
        index_find.append(mask[0][k])

        if optional_files:
            predictions = [
                feature.predict(new_data[k::len(mask[0])])
                for feature in features_to_predict
            ]
            matrix_mms.append([
                prediction[mask[0][k], :] for prediction in predictions
            ])
        else:
            results = np.zeros(
                (len(new_data[k::len(mask[0])]), len(features_to_predict))
            )
            for j, feature in enumerate(features_to_predict):
                results[:, j] = feature.predict(new_data[k::len(mask[0])])

            # result vector
            vector_mms = np.zeros(3 * len(features_to_predict))
            vector_mms[0::3] = np.log10(np.mean(10 ** results, axis=0))
            vector_mms[1::3] = np.log10(np.median(10 ** results, axis=0))
            vector_mms[2::3] = np.std(results, axis=0)

            matrix_mms.append(vector_mms)

    # Importance matrices
    for j in range(len(importances)):
        importances[j][initial[mask][0]] = models[j]["importance"]

    print "Model", str(int(i)) + "/" + str(
        int(np.max(unique_id))), "completed..."

    scores = []
    for model in models:
        scores += [model["score"], model["std"]]
    scores = [i] + scores

    errors = [
        error["sigma"] for error in sigmas
    ]
    trues = [
        np.array(error["true"]) for error in sigmas
    ]
    predictions = [
        np.array(error["pred"]) for error in sigmas
    ]

    return errors, scores, line_labels[initial[mask][0]], index_find, \
           id_model, matrix_mms, importances, trues, predictions
