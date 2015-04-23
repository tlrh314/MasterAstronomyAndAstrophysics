#!/usr/bin/python
"""
simulateyourowninterferometer.py
Radio Astronomy
Create Your Own Interferometer
Timo Halbesma, 6125661
April 20, 2015. Version 1.0.
"""

import numpy
from matplotlib import pyplot


verbose = False
show_plots = False


def set_up_westerbork(setup=''):
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
        # x = 1 to be able to scale with grid size later on.
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

    if verbose:
        print "Generated Westerbork:\n", telescope

    return telescope


def set_up_VLA():
    """
    Description
    """

    # Only 1, 4, 7, allowed for symmetry
    # Add telescope in middle of 100x100 grid
    telescope.append((50, 50, 0))
    for i in range(1, number_of_antennas):
        telescope.append((50+5*i, 50, 0))
        telescope.append((50-5*i, 50+5*i, 0))
        telescope.append((50-5*i, 50-5*i, 0))

    if verbose:
        print "VLA:\n", telescope

    return telescope


def calculate_baselines(telescope):
    """
    Given the telescope positions, calculate the baselines.
    Return the baselines as a numpy array for each dimension.
    """

    number_of_antennas = len(telescope)
    number_of_baselines = number_of_antennas*(number_of_antennas - 1)/2

    # Calculate baselines Bx, By, Bz and save in (separate) arrays.
    baselines_x = numpy.zeros(2*number_of_baselines)
    baselines_y = numpy.zeros(2*number_of_baselines)
    baselines_z = numpy.zeros(2*number_of_baselines)

    baseline_count = 0
    # Calculate all baselines
    for i in range(number_of_antennas):
        for j in range(i+1, number_of_antennas):
            x1, y1, z1 = telescope[i][0], telescope[i][1], telescope[i][2]
            x2, y2, z2 = telescope[j][0], telescope[j][1], telescope[j][2]
            # Since we consider only unique combinations of antenna's, we
            # obtain a symmetrical problem. Hence, we use both antenna1 -
            # antenna2, and antenna1 - antenna2
            baselines_x[baseline_count] = x1 - x2
            baselines_x[baseline_count+1] = x2  - x1
            baselines_y[baseline_count] = y1 - y2
            baselines_y[baseline_count+1] = y2 - y1
            baselines_z[baseline_count] = z1 - z2
            baselines_z[baseline_count+1] = z2 - z1

            # Counter to save baseline to array
            baseline_count += 2

    if verbose:
        print baselines_x, "\n", baselines_y, "\n", baselines_z
    return baselines_x, baselines_y, baselines_z


def calculate_max_baseline(baselines_xyz):
    baselines_x, baselines_y, baselines_z = baselines_xyz

    return max(max(abs(baselines_x)),
               max(abs(baselines_y)),
               max(abs(baselines_z)))


def plot_ground_grid(telescope):
    """
    Given a list of 3-tuple with antenna positions, plot the ground matrix.
    """

    # Here, we assume one pixel correspons to the size of the dish.
    dish_size = 25  # WRT
    grid_multiplicationfactor = 3.
    baselines_xyz = calculate_baselines(telescope)
    maximum_baseline = calculate_max_baseline(baselines_xyz)
    grid_size = maximum_baseline / dish_size + 1
    grid_size = grid_size * grid_multiplicationfactor

    ground_matrix = numpy.zeros([grid_size, grid_size])

    for i, j, k in telescope:
        ground_matrix[i*grid_size/2, j/dish_size+grid_size/grid_multiplicationfactor] = 1

    # i are x values, but are written horizontally in matrix.
    # j are y values, but are written vertically. => Transpose
    # ground_matrix = ground_matrix.T

    if show_plots:
        pyplot.figure()
        pyplot.title("Ground Matrix")
        # pyplot.xlabel("X - North South")
        # pyplot.ylabel("Y- East West")
        pyplot.imshow(ground_matrix, cmap='binary')
        pyplot.colorbar()

    # return ground_matrix


def generate_synthesized_beam(baselines, source_HA=0, source_DEC=numpy.pi/2):
    """

    """

    baselines_x, baselines_y, baselines_z = baselines

    converter = numpy.array(
        [[numpy.sin(source_HA), numpy.cos(source_HA), 0],
         [-1*numpy.sin(source_DEC)*numpy.cos(source_HA), numpy.sin(source_DEC)*numpy.sin(source_HA), numpy.cos(source_DEC)],
         [numpy.cos(source_DEC)*numpy.cos(source_HA), -1*numpy.cos(source_DEC)*numpy.sin(source_HA), numpy.sin(source_DEC)]])
    if verbose:
        print converter

    uvw_matrix = numpy.zeros((len(baselines_x), 3))
    for i in range(len(baselines_x)):
        uvw_matrix[i] = numpy.inner(converter, numpy.array([baselines_x[i], baselines_y[i], baselines_z[i]]))

    u_array = uvw_matrix[:, 0]
    v_array = uvw_matrix[:, 1]
    w_array = uvw_matrix[:, 2]

    dish_size = 25  # WRT
    grid_multiplicationfactor = 3.
    maximum_baseline = calculate_max_baseline(baselines)
    grid_size = 2 * maximum_baseline / dish_size
    grid_size = grid_size * grid_multiplicationfactor

    uv_grid = numpy.zeros([grid_size, grid_size])

    for i in range(len(u_array)):
        uv_grid[v_array[i]/dish_size + grid_size/2,
                u_array[i]/dish_size + grid_size/2] = 1

    if show_plots:
        pyplot.figure()
        pyplot.title("UV-plane")
        pyplot.xlabel("u - East-West")
        pyplot.ylabel("v - Noth-South")
        pyplot.imshow(uv_grid, cmap='binary')
        pyplot.colorbar()

    uv_fourier = numpy.absolute(numpy.fft.fft2(uv_grid))
    beam = numpy.roll(uv_fourier, uv_fourier.shape[0]/2, axis=0)
    beam = numpy.roll(beam, beam.shape[1]/2, axis=1)

    if show_plots:
        pyplot.figure()
        pyplot.title("Fourier transformed UV-plane, rotated to get main beam in center")
        pyplot.imshow(beam)
        pyplot.colorbar()

    return uv_grid, beam


def rotate_synthesized_beam(telescope):
    """
    Due to the Earths rotation, the telescope will rotate.
    Here we simulate this rotation and its effect on the beam.
    """

    baselines = calculate_baselines(telescope)

    # HA ranges from -180 to 180. Dec is fixed at 90.
    shape_known = False
    for HA in numpy.arange(-numpy.pi, numpy.pi, 0.01):
        uv_plane, beam = generate_synthesized_beam(baselines, HA)
        if not shape_known:
            dim = beam.shape[0]
            sum_of_beams = numpy.zeros([dim, dim])
            sum_of_uv_plane = numpy.zeros([dim, dim])
            shape_known = True
        sum_of_beams += beam
        sum_of_uv_plane += uv_plane

    if not show_plots:
        pyplot.figure()
        pyplot.title("Westerbork uv-plane over 12h period")
        pyplot.xlabel("u")
        pyplot.ylabel("v")
        pyplot.imshow(sum_of_uv_plane, cmap='binary')
        pyplot.colorbar()

        pyplot.figure()
        pyplot.title("Westerbork synthesized beam over 12h period")
        pyplot.imshow(sum_of_beams)
        pyplot.colorbar()



if __name__ in '__main__':
    for setup in ['maxi-short', 'traditional36', '2x48', 'mini-short', '2x96']:
    # antenna_positions = set_up_westerbork('traditional90')
        antenna_positions = set_up_westerbork(setup)
        plot_ground_grid(antenna_positions)
        rotate_synthesized_beam(antenna_positions)
        pyplot.show()
        break

    # print set_up_uv_plane(antenna_positions)
