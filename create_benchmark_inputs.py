# !/usr/bin/python3
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


""" Creates input files of various size to test GAME performances """

import argparse
import os
import random

import numpy as np

from game.utils import create_directory

MIN_SIZE = 2
MAX_SIZE = 100
STEP_SIZE = 10
INPUT_SIZE = 500
OUTPUT_FOLDER = os.path.join(
    os.getcwd(),
    "benchmarks",
    "inputs"
)
LABELS_FILE = os.path.join(
    os.getcwd(),
    "library",
    "library_labels.dat"
)


def create_and_parse_args():
    """
    :return: {}
        User args parsed by cmd
    """

    parser = argparse.ArgumentParser(
        usage="-m <minimum qty of features> -M <maximum qty of features> -s "
              "<specifies the increment of features size> -q <quantity of "
              "input to generate> -f <file containing allowed labels>"
              " -o <folder to store outputs>\n"
              "-help for help and usage"
    )

    parser.add_argument(
        "-m",
        dest="min_inputs",
        help="Minimum qty of features",
        default=MIN_SIZE,
        required=False,
        type=int
    )
    parser.add_argument(
        "-M", dest="max_inputs",
        help="Maximum qty of features",
        default=MAX_SIZE,
        required=False,
        type=int
    )
    parser.add_argument(
        "-s", dest="step_inputs",
        help="Specifies the increment of features size",
        default=STEP_SIZE,
        required=False,
        type=int
    )
    parser.add_argument(
        "-q", dest="qty_inputs",
        help="Quantity of input to generate",
        default=INPUT_SIZE,
        required=False,
        type=int
    )
    parser.add_argument(
        "-f", dest="features_data",
        help="File containing allowed labels",
        default=LABELS_FILE,
        required=False,
        type=str
    )
    parser.add_argument(
        "-o", dest="output_path",
        help="Folder to store outputs",
        default=OUTPUT_FOLDER,
        required=False,
        type=str
    )

    args = parser.parse_args()  # parse args

    return {
        "min": args.min_inputs,
        "max": args.max_inputs,
        "step": args.step_inputs,
        "qty": args.qty_inputs,
        "lab": args.features_data,
        "out": args.output_path
    }


def get_allowed_labels(labels_file):
    """
    :param labels_file: str
        File containing allowed labels
    :return: [] of str
        List of allowed labels from file
    """

    with open(labels_file, "r") as in_file:
        lines = in_file.readlines()
        data = [l.split("\" ")[0].replace("\"", "") for l in lines]
        return data


def create_input_set(n_features, qty_to_generate, labels, low=1e-9, high=1e9):
    """
    :param n_features: int
        Number of features to generate
    :param qty_to_generate: int
        Size of features to generate
    :param labels: [] of str
        Allowed labels
    :param low: float
        Min number allowed in matrices
    :param high: float
        Max number allowd in matrices
    :return: {}
        Dict with inputs, errors and labels matrices
    """

    inputs = np.random.uniform(
        low,
        high,
        size=(qty_to_generate, n_features)  # rows, columns
    )
    errors = np.random.uniform(
        low,
        high,
        size=(qty_to_generate, n_features)  # rows, columns
    )
    labels = np.array([
        random.choice(labels) for _ in range(n_features)
    ])

    return {
        "inputs": inputs,
        "errors": errors,
        "labels": labels
    }


def save_inputs_set(input_set, output_folder):
    """
    :param input_set: {}
        Dict with inputs, errors and labels matrices
    :param output_folder: str
        Path to output folder
    :return: void
        Saves files to output folder
    """

    create_directory(output_folder)
    for key in input_set:
        file_path = os.path.join(
            output_folder,
            key + ".dat"
        )
        np.savetxt(
            file_path,
            input_set[key],
            fmt="%s"
        )


def main():
    """
    :return: void
        Creates test input files with user args
    """

    args = create_and_parse_args()
    labels = get_allowed_labels(args["lab"])
    for feature_size in range(args["min"], args["max"], args["step"]):
        print "Generating inputs with", feature_size, "features"

        inputs_set = create_input_set(feature_size, args["qty"], labels)
        output_folder = os.path.join(
            args["out"],
            str(feature_size) + os.path.sep  # make sure to create a folder
        )
        save_inputs_set(
            inputs_set,
            output_folder
        )


if __name__ == '__main__':
    main()
