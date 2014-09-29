#!/usr/bin/python
# -* coding: utf-8 -*

# BLAC_ex6_Friday_6126561_gaussian_blur.py

# Basic Linux and Coding for AA homework 6 (Friday week 4)
# Usage: import into BLAC_ex6_Friday_6126561.py
# TLR Halbesma, 6126561, september 26, 2015. Version 1.0; implemented

from scipy import signal as sg
import numpy as np


def two_dim_gauss(x, y, sigma):
    return 1. / (2*np.pi*sigma**2) * np.exp((x**2 + y**2) / -2*sigma**2)


def gaussian_matrix(radius, sigma):
    kernel = np.array([[two_dim_gauss(x, y, sigma) for x in range(radius)]
                      for y in range(radius)])

    normalization_cst = 1. / np.sum(kernel)
    kernel *= normalization_cst

    return kernel


# https://en.wikipedia.org/wiki/Gaussian_blur
def gaussian_blur(red, green, blue, radius, sigma):
    kernel = gaussian_matrix(radius, sigma)

    # From Scipy convolve2d documentation.
    blurred_red = sg.convolve2d(red, kernel, 'same')
    blurred_green = sg.convolve2d(green, kernel, 'same')
    blurred_blue = sg.convolve2d(blue, kernel, 'same')

    return np.dstack((blurred_red, blurred_green, blurred_blue))
