#!/usr/bin/python
"""
vla.py
Radio Astronomy
Function to set up VLA telescope
Timo Halbesma, 6125661
April 29, 2015. Version 1.1.
"""


from numpy import sqrt


def set_up_vla(setup='default'):
    """
    Generate a list of 3-tuples with (x, y, z) position of antennas for VLA.

    @param setup: string, determine where last four movable dishes are located.
    If setup is not specified, the antennas are evenly spaced in a y-shape
    where the angle between arms is 120 degrees.

    see science.nrao.edu/facilities/vla/docs/manuals/oss2015A/ant_positions.ps
    for the actual antenna positions.

    """

    telescope = []

    if setup == 'default':
        # All of these parameters are not necessary, but they increase
        # readability a lot.
        number_of_antennas = 27
        arm_length = 21000  # meters
        number_of_arms = 3

        antennas_per_arm = number_of_antennas / number_of_arms
        antenna_separation = arm_length / antennas_per_arm

        # Here, the chosen center happens to equal the arm_length.
        center = arm_length

        for i in range(1, antennas_per_arm+1):
            # 'arm straight up'
            telescope.append((center,
                             center+antenna_separation*i, 0))

            # 'arm downwards to the right'
            telescope.append((center+antenna_separation*i/2,
                             center-antenna_separation*i*sqrt(3)/2, 0))
            # 'arm downward to the left'
            telescope.append((center-antenna_separation*i/2,
                             center-antenna_separation*i*sqrt(3)/2, 0))
    elif setup == 'A':
        print 'not implemented'

    elif setup == 'B':
        print 'not implemented'
    elif setup == 'C':
        print 'not implemented'
    elif setup == 'D':
        print 'not implemented'
    elif setup == 'DnC' or setup == 'CnB' or setup == 'BnA':
        print 'Hybrid configurations are not implemented.'

    return telescope
