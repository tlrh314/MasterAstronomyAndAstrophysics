#!/usr/bin/python
# -* coding: utf-8 -*

# BLAC_ex6_Friday_6126561.py

# Basic Linux and Coding for AA homework 6 (Friday week 4)
# Usage: python BLAC_ex6_6126561.py
# TLR Halbesma, 6126561, september 26, 2015. Version 1.0; implemented

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# http://matplotlib.org/users/image_tutorial.html
def plot_individual_channels(pngimage):
    # Step 2
    img = mpimg.imread(pngimage)

    # Step 3: three dimensional array (x, y, N), where x and y is the number
    # of pixels and N the number of channels (either 3 RGB, or 4 RGB alpha).
    print img.shape, type(img)

    fig = plt.figure()
    ax1 = fig.add_subplot(3, 2, 3)
    ax2 = fig.add_subplot(3, 2, 2)
    ax3 = fig.add_subplot(3, 2, 4)
    ax4 = fig.add_subplot(3, 2, 6)
    ax5 = fig.add_subplot(3, 2, 5)

    ax1.imshow(img)
    ax1.set_ylabel('Original')

    red_img = img[:, :, 0]
    ax2.imshow(red_img, cmap='Reds')
    ax2.set_ylabel('Red')

    green_img = img[:, :, 1]
    ax3.imshow(green_img, cmap='Greens')
    ax3.set_ylabel('Green')

    blue_img = img[:, :, 2]
    ax4.imshow(blue_img, cmap='Blues')
    ax4.set_ylabel('Blue')

    try:
        alpha = img[:, :, 3]
        # 0 is transparant, 255 is totally saturated.
    except IndexError:
        print 'No alpha channel present'
    else:
        ax5.imshow(alpha, cmap='binary')
        ax5.set_ylabel('Alpha')

    fig.suptitle('RGB image and its separate channels')

    plt.show()


def lightness(img):
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return (np.fmax(np.fmax(R, G), B) + np.minimum(np.minimum(R, G), B)) / 2.


def average(img):
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return (R + G + B) / 3


def luminosity(img):
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return 0.21*R + 0.72*G + 0.07*B


def plot_greyscale_images(pngimage):
    plt.imshow(lightness(mpimg.imread(pngimage)), cmap='binary')
    plt.show()
    plt.imshow(average(mpimg.imread(pngimage)), cmap='binary')
    plt.show()
    plt.imshow(luminosity(mpimg.imread(pngimage)), cmap='binary')
    plt.show()


def main():
    # Step 1
    inputfile = './colormaps.png'

    # Step 4
    plot_individual_channels(inputfile)

    # Step 5
    plot_greyscale_images(inputfile)


if __name__ == "__main__":
    main()
