from core import game


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

    def __init__(self, filename_config, n_proc, n_repetitions, n_estimators, labels_config):
        self.filename_config = filename_config
        self.n_proc = n_proc
        self.n_repetitions = n_repetitions
        self.n_estimators = n_estimators
        self.labels_config = labels_config
        self.lib_folder = self.DEFAULT_LIB_FOLDER

    def debug_params(self):
        print 'labels', self.labels_config.output
        print 'inputs', self.filename_config.filename_int
        print 'errors', self.filename_config.filename_err
        print 'library', self.filename_config.filename_library
        print 'output', self.filename_config.output_folder
        print 'additional files?', self.filename_config.additional_files
        print 'n processors', self.n_proc
        print 'n repetitions', self.n_repetitions
        print 'n estimators', self.n_estimators

    def run(self):
        game(
            filename_int=self.filename_config.filename_int,
            filename_err=self.filename_config.filename_err,
            filename_library=self.filename_config.filename_library,
            additional_files=self.filename_config.additional_files,
            n_proc=self.n_proc,
            n_repetitions=self.n_repetitions,
            n_estimators=self.n_estimators,
            output_folder=self.filename_config.output_folder,
            verbose=True,  # debug
            out_labels=self.labels_config.output,
            lib_folder=self.lib_folder
        )
