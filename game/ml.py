import copy

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Normalizer


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


def determination_models(data):
    initial = [data != 0][0]
    models = np.zeros(len(initial))
    mask = np.where((initial == initial[0]).all(axis=1))[0]
    models[mask] = 1
    check = True
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
