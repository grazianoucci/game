#!/usr/bin/env bash
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


out_path="c_code"
file_name="game"

echo "Generating .c code ..."
cython --embed -o ${file_name}".c" ${file_name}".py"  # generate .c code

echo "Compiling executable ..."
gcc -Os -I /usr/include/python2.7 -o "c_"${file_name} ${file_name}".c" \
-lpython2.7 -lpthread -lm -lutil -ldl  # generate executable

echo "Moving to output folder ..."
rm -rf ${out_path}  # clean folder
mkdir ${out_path}
mv ${file_name}".c" "c_"${file_name} ${out_path}  # move files
EXE_PATH=$(pwd)"/"${out_path}"/"${file_name}
echo "Done! The executable can be found here: "${EXE_PATH}
