#!/usr/bin/python
"""
westerbork.py
Radio Astronomy
Function to set up Westerbork telescope
Timo Halbesma, 6125661
April 20, 2015. Version 1.0.
"""


def set_up_westerbork(setup='default'):
    """
    Generate a list of 3-tuples with (x, y, z) position of antennas for
    Westerbork.
    See http://www.astron.nl/radio-observatory/public/public-0

    @param setup: string, determine where last four movable dishes are located.
    If setup is not specified, only the inner 10 dishes are used.

    """

    telescope = []
    number_of_antennas = 10  # There are ten fixed dishes

    for i in range(number_of_antennas):
        # Dishes are 25m big and lie 144 meter apart.
        # Dishes are positioned in array from West to East.
        # x = 1 but could have been any constant value
        # z = 0 because we do not use z.
        telescope.append((1, i*144, 0))

    # For information regarding possible setups, see:
    # http://www.astron.nl/radio-observatory/astronomers/wsrt-guide-observations/3-telescope-parameters-and-array-configuration
    if setup == 'maxi-short':
        telescope.append((1, 9*144+36, 0))  # RT9 -> RTA = 36
        telescope.append((1, 9*144+90, 0))  # RTA -> RTB = 54
        telescope.append((1, 9*144+1332, 0))  # RTB -> RTC = 9*144+36
        telescope.append((1, 9*144+1404, 0))  # RTC -> RTD = 72
    elif setup[:-2] == 'traditional':
        # Get distance from RT9 -> RTA, which is RTC-> RTD
        try:
            from_9_to_A = int(setup[-2:])
        except ValueError:
            print 'Error: usage of traditional ends with two ' + \
                'integers giving the distance from telescope 9 to telescope ' + \
                'A, e.g. \'traditional36\', or \'traditional54\''

        telescope.append((1, 9*144+from_9_to_A, 0))
        telescope.append((1, 9*144+72+from_9_to_A, 0))
        telescope.append((1, 9*144+9*144+from_9_to_A, 0))
        telescope.append((1, 9*144+9*144+72+from_9_to_A, 0))
    elif setup == '2x48':
        telescope.append((1, 9*144+48, 0))
        telescope.append((1, 9*144+96, 0))
        telescope.append((1, 9*144+1332, 0))
        telescope.append((1, 9*144+1404, 0))
    elif setup == 'mini-short':
        telescope.append((1, 9*144+96, 0))
        telescope.append((1, 9*144+192, 0))
        telescope.append((1, 9*144+1332, 0))
        telescope.append((1, 9*144+1428, 0))
    elif setup == '2x96':
        telescope.append((1, 9*144+96, 0))
        telescope.append((1, 9*144+192, 0))
        telescope.append((1, 9*144+1332, 0))
        telescope.append((1, 9*144+1404, 0))

    return telescope
