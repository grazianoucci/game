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
    def __init__(self, output, additional):
        self.output = output
        self.additional = additional


class Game:
    def __init__(self, filename_config, n_proc, n_repetition, labels_config):
        self.filename_config = filename_config
        self.n_proc = n_proc
        self.n_repetition = n_repetition
        self.labels_config = labels_config

    def run(self):
        game(
            filename_int=self.filename_config.filename_int,
            filename_err=self.filename_config.filename_err,
            filename_library=self.filename_config.filename_library,
            additional_files=self.filename_config.additional_files,
            n_proc=self.n_proc,
            n_repetition=self.n_repetition,
            output_folder=self.filename_config.output_folder,
            verbose=True,  # todo debug only
            out_labels=self.labels_config.output,
            out_additional_labels=self.labels_config.additional
        )
