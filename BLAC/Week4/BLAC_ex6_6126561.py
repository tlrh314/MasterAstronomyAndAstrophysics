#!/usr/bin/python
# -* coding: utf-8 -*

# BLAC_ex6_6126561.py

# Basic Linux and Coding for AA homework 6 (week 4)
# Usage: python BLAC_ex6_6126561.py
# TLR Halbesma, 6126561, september 21, 2015. Version 1.0; implemented

import numpy as np
import matplotlib.pyplot as plt


# https://stackoverflow.com/questions/11364745/how-can-i-turn-a-numpy-array-into-a-matplotlib-colormap
# http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps
def plot_colormap(data, style):
    plt.gca().pcolormesh(data, cmap=style)

    cb = plt.cm.ScalarMappable(norm=None, cmap=style)
    cb.set_array(data)
    cb.set_clim((-1., 1.))
    plt.gcf().colorbar(cb)
    plt.show()


# http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html
def plot_histogram(data, mu, sigma):
    count, bins, ignored = plt.hist(data, 100, normed=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - mu)**2 / (2 * sigma**2)),
             linewidth=2, color='r')
    plt.show()

    return


def create_gauss(x, y, mu, sigma):
    return np.random.normal(mu, sigma, (x, y))


def create_sincos(i, j):
    sincos = np.zeros((i, j))

    for x in range(i):
        for y in range(j):
            sincos[x][y] = np.sin(x/100.)*np.cos(y/100.)
            # print '(' + str(x) + ',' + str(y) + ') =', sincos[x][y]

    return sincos


def remove_negative(array):
    for x in range(len(array)):
        for y in range(len(array[x])):
            if array[x][y] < 0:
                array[x][y] = 0
            # print '(' + str(x) + ',' + str(y) + ') =', array[x][y]


def main():
    # step 1
    mu, sigma = 0, 0.42
    gauss = create_gauss(100, 200, mu, sigma)

    # step 2
    plot_colormap(gauss, 'gist_rainbow')

    # step 3
    plot_colormap(gauss, 'binary')

    # step 4
    x, y = gauss.shape
    plot_histogram(gauss.reshape(x*y, 1), mu, sigma)

    # step 5
    sincos = create_sincos(2*314, 2*314)
    plot_colormap(sincos, 'gist_rainbow')

    # step 6
    # remove_negative(sincos)
    plot_colormap(sincos.clip(0, 1), 'gist_rainbow')


if __name__ == "__main__":
    main()
