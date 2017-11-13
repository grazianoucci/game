# !/usr/bin/python3
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


""" GAME utilities """

import multiprocessing
import os
import tarfile
import urllib

import numpy as np


def download_library(
        download_file,
        url="http://cosmology.sns.it/library_game/library.tar.gz"
):
    """
    :param download_file: str
        Path where to download file
    :param url: str
        Url of library
    :return: void
        Downloads library to file
    """

    try:
        urllib.urlretrieve(
            url,
            filename=download_file
        )
    except Exception:
        if os.path.exists(download_file):
            os.remove(download_file)

        raise Exception("Cannot download library .tar file")


def create_library(folder, check_file):
    """
    :return: void
        Creates necessary  library directory if not existing
    """

    lib_file = os.path.join(
        folder,
        "library.tar.gz"
    )

    if not os.path.exists(check_file):
        os.makedirs(folder)

        if not os.path.exists(lib_file):
            download_library(lib_file)  # download library

        if not os.path.exists(check_file):
            tar = tarfile.open(lib_file)  # extract library
            tar.extractall()
            tar.close()


def run_parallel(algorithm, n_processes, unique_id):
    pool = multiprocessing.Pool(processes=n_processes)
    results = pool.map(
        algorithm,
        range(1, int(np.max(unique_id)) + 1)
    )
    pool.close()
    pool.join()
    return results
