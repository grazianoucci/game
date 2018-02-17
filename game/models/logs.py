# !/usr/bin/python3
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


""" Log stuff """

from datetime import datetime

from game.utils import get_actual_class_name

LOG_TIME_FORMAT = "%m-%d %H:%M:%S"


class Logger:
    """ Logs itself """

    def __init__(self, verbose):
        self.class_name = get_actual_class_name(self)
        self.verbose = bool(verbose)

    def log(self, *content):
        """
        :param content: *
            Data to print to stdout
        :return: void
            Prints log
        """

        if self.verbose:
            now = datetime.now().strftime(LOG_TIME_FORMAT)
            content = " ".join([str(x) for x in content])
            print now, self.class_name, ">>>", content, "\n"
