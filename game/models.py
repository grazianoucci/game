# -*- coding: utf-8 -*-
import time

from behaviour import ok_status, GameBehaviour
from core import check_input, game


class FilesConfig:
    def __init__(self, filename_int, filename_err, filename_library,
                 output_folder, additional_files):
        self.filename_int = filename_int
        self.filename_err = filename_err
        self.filename_library = filename_library
        self.output_folder = output_folder
        self.additional_files = additional_files


class LabelsConfig:
    def __init__(self, output):
        self.output = output


class Game:
    DEFAULT_LIB_FOLDER = '/opt/game/game/library'

    def __init__(self, filename_config, n_proc, n_repetitions, n_estimators,
                 labels_config):
        self.filename_config = filename_config
        self.n_proc = n_proc
        self.n_repetitions = n_repetitions
        self.n_estimators = n_estimators
        self.labels_config = labels_config
        self.lib_folder = self.DEFAULT_LIB_FOLDER

    def run(self):
        status, labels, limit, data, line_labels, models, \
        unique_id, initial, features, additional_files = check_input(
            self.filename_config.filename_int,
            self.filename_config.filename_library,
            self.filename_config.additional_files,
            self.n_repetitions,
            self.filename_config.output_folder,
            True,
            self.lib_folder
        )

        if status.is_error():
            return status

        timer = time.time()

        try:
            game(
                labels,
                limit,
                data,
                line_labels,
                self.n_estimators,
                self.n_repetitions,
                self.labels_config.out_labels,
                additional_files,
                models,
                unique_id,
                initial,
                features,
                self.filename_config.filename_int,
                self.filename_config.filename_err,
                self.filename_config.output_folder,
                self.n_proc
            )
        except Exception as e:
            return GameBehaviour.from_exception(e)

        timer = time.time() - timer
        return ok_status(str(timer))
