#!/usr/bin/python
# -* coding: utf-8 -*

# BLAC_ex6_Friday_6126561_sobel.py

# Basic Linux and Coding for AA homework 6 (Friday week 4)
# Usage: import into BLAC_ex6_Friday_6126561.py
# TLR Halbesma, 6126561, september 26, 2015. Version 1.0; implemented

from scipy import signal as sg
import numpy as np


def sobel_filtered(gray_luminosity):
    # https://en.wikipedia.org/wiki/Edge_detection
    # First order Sobel method
    sobel_operator_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_operator_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # From Scipy convolve2d documentation.
    l_x = sg.convolve2d(gray_luminosity, sobel_operator_x, 'same')
    l_y = sg.convolve2d(gray_luminosity, sobel_operator_y, 'same')

    # Gradient magnutide according to Wikipedia.
    magnitude_gradient = np.sqrt(l_x**2, l_y**2)

# https://stackoverflow.com/questions/7185655/applying-the-sobel-filter-using-scipy
    # One might have to normalize according to this stack overflow answer.
    # magnitude_gradient *= 255. / np.max(magnitude_gradient)

    return magnitude_gradient
