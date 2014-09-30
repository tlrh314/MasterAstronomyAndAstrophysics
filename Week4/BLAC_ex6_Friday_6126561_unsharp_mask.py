#!/usr/bin/python
# -* coding: utf-8 -*

# BLAC_ex6_Friday_6126561_unsharp_mask.py

# Basic Linux and Coding for AA homework 6 (Friday week 4)
# Usage: import into BLAC_ex6_Friday_6126561.py
# TLR Halbesma, 6126561, september 29, 2015. Version 1.0; implemented

import numpy as np


# https://en.wikipedia.org/wiki/Unsharp_masking
def unsharp_mask(rgb, edges, blur, threshold, amount):
    edges_rgb = np.dstack((edges, edges, edges))
    return rgb - blur * threshold + edges_rgb * amount
