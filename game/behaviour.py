# -*- coding: utf-8 -*-


import abc

from enum import Enum


class GameErrorsCode(Enum):
    OK = 200

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

    # unknown
    UNKNOWN = 201


class GameBehaviour(Exception):
    OUTPUT_FORMAT = '{} (code {})'

    def __init__(self, message, code):
        """
        :param message: explain code
        :param code: reference code of error. Use the following table:
        - code 200: OK
        - codes between (including) 210 and 219: files errors
        - codes between (including) 220 and 229: system (cpu, memory) errors
        """

        if message is None:
            message = ''

        super(GameBehaviour, self).__init__(message)
        self.code = int(code)

    def __str__(self):
        return self.OUTPUT_FORMAT.format(self.message, self.code)

    @staticmethod
    @abc.abstractmethod
    def build(message, code):
        return GameBehaviour(message, code)

    @staticmethod
    def from_exception(exception):
        message = str(exception)
        code = GameErrorsCode.UNKNOWN

        if isinstance(exception, OSError):
            code = GameErrorsCode.FILES  # todo other codes

        return GameBehaviour(message, code)

    def is_error(self):
        return self.code != GameErrorsCode.OK


class GameStatus(GameBehaviour):
    @staticmethod
    def build(message, code):
        return GameStatus(message, code)


class GameError(GameBehaviour):
    @staticmethod
    @abc.abstractmethod
    def build(message, code):
        return GameError(message, code)

    @staticmethod
    def build_files_exception(message):
        return GameError(message, GameErrorsCode.FILES)

    @staticmethod
    def build_system_exception(message):
        return GameError(message, GameErrorsCode.SYSTEM)

    @staticmethod
    def build_too_much_memory_exception(message):
        return GameError(message, GameErrorsCode.SYSTEM_MEM)


def ok_status(message=None):
    return GameStatus(message, GameErrorsCode.OK)


def game_error(message, code):
    return GameError(message, code)
