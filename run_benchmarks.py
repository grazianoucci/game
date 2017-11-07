# !/usr/bin/python3
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


""" Runs empty GAME models on input files of various size to test GAME
performances """

import argparse
import os

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
    print files
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
            yield test_set


def main():
    """
    :return: void
        Creates test input files with user args
    """

    args = create_and_parse_args()
    tests = discover_tests(args["input"])
    for test in tests:
        print test


if __name__ == '__main__':
    main()
