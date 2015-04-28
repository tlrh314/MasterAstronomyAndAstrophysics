#!/usr/bin/python
"""
lofar.py
Radio Astronomy
Function to set up LOFAR telescope
Timo Halbesma, 6125661
April 28, 2015. Version 1.0.
"""

from numpy.random import standard_normal


def set_up_lofar():
    """
    Generate a list of 3-tuples with (x, y, z) position of antennas for LOFAR.
    """

    telescope = []
    number_of_antennas = 96

    # Add three to avoid generating negative numbers
    x = standard_normal(number_of_antennas) + 3
    y = standard_normal(number_of_antennas) + 3
    z = standard_normal(number_of_antennas) + 3

    telescope = [(x[i], y[i], z[i]) for i in range(len(x))]

    return telescope
