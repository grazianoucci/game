# -*- coding: utf-8 -*-


from enum import Enum


class GameErrorsCode(Enum):
    # files: 210 - 219
    FILES = 210  # general
    FILE_NOT_FOUND = 211
    FILE_NOT_READ = 212
    FILE_NOT_WRITE = 213

    # system: 220 - 229
    SYSTEM = 220  # general

    # cpu
    SYSTEM_CPU = 221

    # memory
    SYSTEM_MEM = 225


class GameException(Exception):
    OUTPUT_FORMAT = '{} (code {})'

    def __init__(self, message, ref_code):
        """
        :param message: explain code
        :param ref_code: reference code of error. Use the following table:
        - codes between (including) 210 and 219: files errors
        - codes between (including) 220 and 229: system (cpu, memory) errors
        """

        super(GameException, self).__init__(message)
        self.ref_code = int(ref_code)

    def __str__(self):
        return self.OUTPUT_FORMAT.format(self.message, self.ref_code)

    @staticmethod
    def build(message, code):
        return GameException(message, code)

    @staticmethod
    def build_files_exception(message):
        return GameException(message, GameErrorsCode.FILES)

    @staticmethod
    def build_system_exception(message):
        return GameException(message, GameErrorsCode.SYSTEM)

    @staticmethod
    def build_too_much_memory_exception(message):
        return GameException(message, GameErrorsCode.SYSTEM_MEM)
