# !/usr/bin/python3
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


""" Creates input files of various size to test GAME performances """

import argparse
import os

from game.utils import create_directory

MIN_SIZE = 10
MAX_SIZE = 100000
STEP_SIZE = 500
INPUT_SIZE = 1000
OUTPUT_FOLDER = os.path.join(
    os.getcwd(),
    "output"
)


def create_and_parse_args():
    parser = argparse.ArgumentParser(
        usage="-m <minimum qty of features> -M <maximum qty of features> -s "
              "<specifies the increment of features size> -q <quantity of "
              "input to generate> -o <folder to store outputs>\n"
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
        "out": args.output_path
    }


def create_input_set(features_size, qty_to_generate):
    """
    :param features_size: int
        Number of features to generate
    :param qty_to_generate: int
        Size of features to generate
    :return: 
    """

    return {
        "inputs": None,
        "errors": None,
        "labels": None
    }


def save_inputs_set(input_set, output_folder):
    """
    :param input_set: 
    :param output_folder: 
    :return: 
    """

    create_directory(output_folder)
    for key in input_set:
        file_path = os.path.join(
            output_folder,
            key + ".dat"
        )
        with open(file_path, "w") as out_file:
            out_file.write(input_set[key])


def main():
    """
    :return: void
        Creates test input files with user args
    """

    args = create_and_parse_args()
    for feature_size in range(args["min"], args["max"], args["step"]):
        print "Generating inputs with", feature_size, "features"

        inputs_set = create_input_set(feature_size, args["qty"])
        output_folder = os.path.join(
            args["out"],
            str(feature_size)
        )
        save_inputs_set(
            inputs_set,
            output_folder
        )


if __name__ == '__main__':
    main()
