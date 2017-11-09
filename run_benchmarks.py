# !/usr/bin/python3
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


""" Runs empty GAME models on input files of various size to test GAME
performances """

import argparse
import os
import time
import datetime
import numpy as np
import pylab

from game.models import Game

GENERATE_CHARTS = True
INPUT_FOLDER = os.path.join(
    os.getcwd(),
    "benchmarks",
    "inputs"
)
OUTPUT_FOLDER = os.path.join(
    os.getcwd(),
    "benchmarks",
    "outputs"
)


def create_and_parse_args():
    """
    :return: {}
        User args parsed by cmd
    """

    parser = argparse.ArgumentParser(
        usage="-i <folder to search for input files> -o <folder where to "
              "save output files> -c <whether to generate charts>\n"
              "-help for help and usage"
    )

    parser.add_argument(
        "-i",
        dest="ins",
        help="Folder to search for input files",
        default=INPUT_FOLDER,
        required=False,
        type=str
    )
    parser.add_argument(
        "-o", dest="outs",
        help="Folder where to save output files",
        default=OUTPUT_FOLDER,
        required=False,
        type=str
    )
    parser.add_argument(
        "-c", dest="charts",
        help="Whether to generate charts",
        default=GENERATE_CHARTS,
        required=False,
        type=bool
    )

    args = parser.parse_args()  # parse args

    return {
        "input": args.ins,
        "output": args.outs,
        "charts": args.charts
    }


def get_just_folders(folder):
    """
    :param folder: str
        Path to folder to search for folders
    :return: [] of str
        List of folders (full path) just inside specified directory
    """

    folders = [
        os.path.join(
            folder,
            f
        ) for f in os.listdir(folder) if os.path.isdir(
            os.path.join(
                folder,
                f
            )
        )
    ]
    return folders


def get_just_files(folder):
    """
    :param folder: str
        Path to folder to search for files
    :return: [] of str
        List of files (full path) just inside specified directory
    """

    files = [
        os.path.join(
            folder,
            f
        ) for f in os.listdir(folder) if os.path.isfile(
            os.path.join(
                folder,
                f
            )
        )
    ]
    return files


def get_name_of(folder):
    """
    :param folder: str
        Full path of folder
    :return: str
        Name of folder
    """

    return folder.split(os.path.sep)[-1]


def is_valid_test_folder(folder):
    """
    :param folder: str
        Path to folder to check
    :return: bool
        True iff folder contains right files to run tests
    """

    content = os.listdir(folder)
    file_names = ["errors.dat", "inputs.dat", "labels.dat"]
    for file_name in file_names:
        if file_name not in content:
            return False
    return True


def get_test_set(folder):
    """
    :param folder: str
        Path to folder to search
    :return: {}
        Dict with path to test set files
    """

    files = get_just_files(folder)
    keys = ["inputs", "errors", "labels"]
    test_set = {}
    for key in keys:
        test_set[key] = [
            f for f in files if get_name_of(f) == key + ".dat"
        ][0]

    return test_set


def discover_tests(folder):
    """
    :param folder: str
        Path to folder to search for input files
    :return: [] of {}
        List of test suites
    """

    folders = get_just_folders(folder)
    for fold in folders:
        if is_valid_test_folder(fold):
            test_set = get_test_set(fold)
            test_set["size"] = int(get_name_of(fold))
            test_set["root"] = fold
            yield test_set


def run_test(input_folder):
    """
    :param input_folder: str
        Path where to find input files
    :return: void
        Runs GAME algorithm with inputs
    """

    stopwatch = time.time()

    try:
        driver = Game(
            ["g0", "n", "NH", "U", "Z"],
            input_folder=input_folder,
            output_folder="/dev/null",  # TODO change when in Windows
            output_header="",
            output_filename="",
            manual_input=False,
            verbose=False
        )
        driver.run()
        successfully_run = True
    except:
        successfully_run = False

    stopwatch = time.time() - stopwatch
    return stopwatch, successfully_run


def show_plot(x_data, y_data, trend_line_exp=3):
    """
    :param x_data: [] of float
        x
    :param y_data: [] of float
        y
    :param trend_line_exp: int
        Max degree of fitter polynomial
    :return: void
        Shows plot
    """

    pylab.plot(x_data, y_data, "-x")
    trend = np.polyfit(x_data, y_data, trend_line_exp)
    poly_line = np.poly1d(trend)
    pylab.plot(x_data, poly_line(x_data), "r--")
    pylab.title("Test size (number of features) VS Time taken (seconds)")
    pylab.show()


def main():
    """
    :return: void
        Creates test input files with user args
    """

    args = create_and_parse_args()
    tests = discover_tests(args["input"])
    test_times = []  # x, y data to plot
    for test in tests:
        test_size = test["size"]

        print datetime.datetime.now()
        print "Running test with", test_size, "features"
        time_taken, successully_run = run_test(test["root"])

        if successully_run:
            print "Successfully completed test with", test_size, "features"
        else:
            print "Aborted test with", test_size, "features"
        print "\tTime taken:", time_taken, "seconds\n"

        test_times.append(
            (test_size, time_taken)
        )

    print "Done all tests, displaying chart..."
    test_times.sort(key=lambda tup: tup[0])  # sort based on test size
    show_plot(
        [t[0] for t in test_times],
        [t[1] for t in test_times]
    )


if __name__ == '__main__':
    main()
