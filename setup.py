# !/usr/bin/python3
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


""" Setups librarys and install dependencies """


from setuptools import setup, find_packages


DESCRIPTION = \
        '--------------------------------------------------------\n' + \
        '--- GAME (GAlaxy Machine learning for Emission lines) --\n' + \
        '------- see Ucci G. et al. (2017a,b) for details -------\n' + \
        '--------------------------------------------------------\n\n' + \
        'ML Algorithm: AdaBoost with Decision Trees as base learner.'


setup(
    name="GAME",
    version="1.1",
    description="GAlaxy Machine learning for Emission lines",
    long_description=DESCRIPTION,
    keywords="machine learning",
    url="https://github.com/grazianoucci/game",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "sklearn", 'PIL', 'psutil',
    ],
    test_suite="tests"
)
