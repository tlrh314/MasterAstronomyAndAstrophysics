"""
File: helper_functions.py
Author: Timo L. R. Halbesma <timo.halbesma@student.uva.nl>
Version: 0.01 (Initial)
Date created: Fri Mar 04, 2016 03:46 pm
Last modified: Fri Mar 04, 2016 04:48 pm

Description: Helper functions for Cygnus A merger

"""

import sys

from amuse.units import units


# From AMUSE code
def smart_length_units_for_vector_quantity(quantity):
    length_units = [units.Mpc, units.kpc, units.parsec, units.AU, units.RSun, units.km]
    total_size = max(quantity) - min(quantity)
    for length_unit in length_units:
        if total_size > (1 | length_unit):
            return length_unit
    return units.m


def print_progressbar(i, tot):
        bar_width = 42  # obviously
        progress = float(i)/tot
        block = int(round(bar_width * progress))
        sys.stdout.write(
            "\r[{0}{1}] {2:.2f}% \t{3}/{4}"
            .format('#'*block, ' '*(bar_width - block),
                    progress*100, i, tot))
        sys.stdout.flush()
