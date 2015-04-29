#!/usr/bin/python
"""
simulateyourowninterferometer.py
Radio Astronomy
Create Your Own Interferometer
Timo Halbesma, 6125661
April 29, 2015. Version 1.1.
"""

import numpy
from matplotlib import pyplot

# Shove in different files because actual layout gets lengthy
from westerbork import set_up_westerbork
from vla import set_up_vla
from lofar import set_up_lofar


VERBOSE = True
SHOWPLOTS = True


class Telescope(object):
    """
    Class that holds the information about the telescope used.
    """
    def __init__(self, name, setup='default'):
        self.name = name
        self.setup = setup

        if VERBOSE:
            print "Initializing telescope '{0}' in setup '{1}'\n" \
                .format(self.name, self.setup)

        # Generate antenna positions dependent on which telescope we build.
        if name == 'Westerbork':
            self.antennas = set_up_westerbork(self.setup)
            self.dish_size = 25  # meter -> 1 pixel = 25 meter
        elif name == 'VLA':
            self.antennas = set_up_vla(self.setup)
#
            self.dish_size = 250  # dish is 25 meter. Here we use 250 because
            # otherwise calculation takes ages...
        elif name == 'LOFAR':
            self.antennas = set_up_lofar()
            # No idea how big the dish is. This just gives a good
            # image...
            self.dish_size = 0.25

        if VERBOSE:
            print "Generated antenna positions:\n", self.antennas

            pyplot.figure()
            for x, y, z in self.antennas:
                pyplot.scatter(x, y)
            pyplot.title("Scatterplot of antenna positions for '{0}'"
                         .format(self.name))
            pyplot.savefig("Results/{0}/{0}_{1}_uv_plane_scatter.png"
                            .format(self.name, self.setup))

        # Generate baselines for the telescope.
        self.baselines_x, self.baselines_y, self.baselines_z =\
            self.calculate_baselines()

        if VERBOSE:
            print "\n\nCalculated the baselines.\n"
            print "x-direction\n{0}\n".format(self.baselines_x)
            print "y-direction\n{0}\n".format(self.baselines_y)
            print "z-direction\n{0}\n".format(self.baselines_z)

    def calculate_baselines(self):
        """
        Given the antenna positions, calculate the baselines.
        """
        number_of_antennas = len(self.antennas)

        number_of_baselines = number_of_antennas \
            * (number_of_antennas - 1)/2
        baselines_x = numpy.zeros(2*number_of_baselines)
        baselines_y = numpy.zeros(2*number_of_baselines)
        baselines_z = numpy.zeros(2*number_of_baselines)

        # Calculate baselines Bx, By, Bz and save in (separate) arrays.

        baseline_count = 0
        # Calculate all baselines
        for i in range(number_of_antennas):
            for j in range(i+1, number_of_antennas):
                x1, y1, z1 = self.antennas[i][0], self.antennas[i][1],\
                    self.antennas[i][2]
                x2, y2, z2 = self.antennas[j][0], self.antennas[j][1],\
                    self.antennas[j][2]

                # Since we consider only unique combinations of antenna's, we
                # obtain a symmetrical problem. Hence, we use both antenna1 -
                # antenna2, and antenna1 - antenna2 to decrease # iterations.
                baselines_x[baseline_count] = x1 - x2
                baselines_x[baseline_count+1] = x2 - x1
                baselines_y[baseline_count] = y1 - y2
                baselines_y[baseline_count+1] = y2 - y1
                baselines_z[baseline_count] = z1 - z2
                baselines_z[baseline_count+1] = z2 - z1

                # Counter to save baseline to array. Add 2 because symmetry.
                baseline_count += 2

        return baselines_x, baselines_y, baselines_z

    def calculate_max_baseline(self):
        return max(max(abs(self.baselines_x)),
                   max(abs(self.baselines_y)),
                   max(abs(self.baselines_z)))


    def plot_ground_grid(self):
        """
        Greate a ground grid, and plot the ground grid.
        """

        if self.name == 'Westerbork':
            # Here, we assume one pixel correspons to the size of the dish.
            grid_multiplicationfactor = 3.
            maximum_baseline = self.calculate_max_baseline()
            grid_size = maximum_baseline / self.dish_size + 1
            grid_size = grid_size * grid_multiplicationfactor
            ground_matrix = numpy.zeros([grid_size, grid_size])

            for i, j, k in self.antennas:
                # Scale ground matrix with respect to dish size.
                # This way one pixel represents 25 meters.
                ground_matrix[i*grid_size/2,
                    j/self.dish_size+grid_size/grid_multiplicationfactor] = 1

        elif self.name == 'VLA':
            grid_size = 2*21000 / self.dish_size
            ground_matrix = numpy.zeros([grid_size, grid_size])

            for i, j, k in self.antennas:
                ground_matrix[i/self.dish_size, j/self.dish_size] = 1

        elif self.name == "LOFAR":
            grid_multiplicationfactor = 3.
            maximum_baseline = self.calculate_max_baseline()
            grid_size = 2 * maximum_baseline / self.dish_size
            grid_size = grid_size * grid_multiplicationfactor

            ground_matrix = numpy.zeros([grid_size, grid_size])

            for i, j, k in self.antennas:
                ground_matrix[i/self.dish_size+grid_size/grid_multiplicationfactor,
                    j/self.dish_size+grid_size/grid_multiplicationfactor] = 1

        # i are x values, but are written horizontally in matrix.
        # j are y values, but are written vertically. => Transpose
        ground_matrix = ground_matrix.T

        pyplot.figure()
        pyplot.imshow(ground_matrix, cmap='binary')
        pyplot.title("Ground Matrix for '{0}'".format(self.name))
        pyplot.xlabel("X [North South]")
        pyplot.ylabel("Y [East West]")
        pyplot.colorbar()

        pyplot.savefig("Results/{0}/{0}_{1}_ground_matrix.png"
                       .format(self.name, self.setup))

    def generate_synthesized_beam(self, source_HA=0, source_DEC=numpy.pi/2):
        """
        Given the baselines of the telescope, we can calculate the synthesized
        beam for a specific combination of the source hour hangle and the
        source declination. By default, we assume the hour angle is 0, and the
        declination is 90 degrees.
        """

        # Matrix given in lecture 6, slide 64.
        # Blegh, the formatting looks ugly as hell, but PEP8....
        converter = numpy.array([[numpy.sin(source_HA),
                                  numpy.cos(source_HA),
                                  0],
                                 [-1*numpy.sin(source_DEC)*numpy.cos(source_HA),
                                  numpy.sin(source_DEC)*numpy.sin(source_HA),
                                  numpy.cos(source_DEC)],
                                 [numpy.cos(source_DEC)*numpy.cos(source_HA),
                                  -1*numpy.cos(source_DEC)*numpy.sin(source_HA),
                                  numpy.sin(source_DEC)]])

        uvw_matrix = numpy.zeros((len(self.baselines_x), 3))
        for i in range(len(self.baselines_x)):
            uvw_matrix[i] = numpy.inner(converter,
                                        numpy.array([self.baselines_x[i],
                                                     self.baselines_y[i],
                                                     self.baselines_z[i]]))

        u_array = uvw_matrix[:, 0]
        v_array = uvw_matrix[:, 1]
        w_array = uvw_matrix[:, 2]

        grid_multiplicationfactor = 3.
        maximum_baseline = self.calculate_max_baseline()
        grid_size = 2 * maximum_baseline / self.dish_size
        grid_size = grid_size * grid_multiplicationfactor

        uv_grid = numpy.zeros([grid_size, grid_size])

        for i in range(len(u_array)):
            uv_grid[v_array[i]/self.dish_size + grid_size/2,
                    u_array[i]/self.dish_size + grid_size/2] = 1

        # pyplot.figure()
        # pyplot.title("UV-plane")
        # pyplot.xlabel("u - East-West")
        # pyplot.ylabel("v - Noth-South")
        # pyplot.imshow(uv_grid, cmap='binary')
        # pyplot.colorbar()

        uv_fourier = numpy.absolute(numpy.fft.fft2(uv_grid))
        beam = numpy.roll(uv_fourier, uv_fourier.shape[0]/2, axis=0)
        beam = numpy.roll(beam, beam.shape[1]/2, axis=1)

        # pyplot.figure()
        # pyplot.title("Fourier transformed UV-plane, rotated to get main beam in center")
        # pyplot.imshow(beam)
        # pyplot.colorbar()

        return uv_grid, beam


    def rotate_synthesized_beam(self, ha_min=-numpy.pi, ha_max=numpy.pi, precision=0.1):
        """
        Due to the Earths rotation, the telescope will rotate.
        Here we simulate this rotation and its effect on the beam.
        """

        # HA ranges from -180 to 180. Dec is fixed at 90.
        shape_known = False
        hour_angle = ha_min
        while hour_angle <= ha_max:
            uv_plane, beam = self.generate_synthesized_beam(hour_angle)
            if not shape_known:
                dim = beam.shape[0]
                sum_of_beams = numpy.zeros([dim, dim])
                sum_of_uv_plane = numpy.zeros([dim, dim])
                shape_known = True
            sum_of_beams += beam
            sum_of_uv_plane += uv_plane

            hour_angle += precision

        # 2pi <--> 24h
        total_angle = 24 * (ha_max - ha_min) / (2*numpy.pi)

        pyplot.figure()
        pyplot.title("{0} uv-plane over {1}h period"
                     .format(self.name,total_angle))
        pyplot.xlabel("u")
        pyplot.ylabel("v")
        pyplot.imshow(sum_of_uv_plane, cmap='binary')
        pyplot.colorbar()
        pyplot.savefig("Results/{0}/{0}_{1}_uv_plane_{2}h.png"
                       .format(self.name, self.setup, total_angle))

        pyplot.figure()
        pyplot.title("{0} synthesized beam over {1}h period"
                     .format(self.name, total_angle))
        pyplot.xlabel("l")
        pyplot.ylabel("m")
        pyplot.imshow(sum_of_beams)
        pyplot.colorbar()
        pyplot.savefig("Results/{0}/{0}_{1}_synthesized_beam_{2}h.png"
                       .format(self.name, self.setup, total_angle))


if __name__ in '__main__':
    # Westerbork setup = 'default' gives only the inner ten antennas. Question 1-1
    westerbork = Telescope("Westerbork")
    westerbork.plot_ground_grid()
    # Rotating with ha_min and ha_max set to zero gives the instantaneous uv-coverage. Question 1-2
    westerbork.rotate_synthesized_beam(0, 0)

    # Rotate over 12 hour period. Question 1-3
    westerbork.rotate_synthesized_beam(-numpy.pi, 0)

    # Add the four outer dishes in traditional setup with 36 meter separation
    # between RT9 -> RTA (thus RTC -> RTD is also 36 meter). Question 1-4.
    westerbork = Telescope("Westerbork", "traditional36")
    westerbork.rotate_synthesized_beam(-numpy.pi, 0)

    vla = Telescope("VLA")
    # vla.plot_ground_grid()
    # Question 2-2
    vla.rotate_synthesized_beam(0, 0)
    vla.rotate_synthesized_beam(0, numpy.pi)

    # Question 2-3
    vla = Telescope("VLA", "A")
    vla.rotate_synthesized_beam(0, numpy.pi)

    vla = Telescope("VLA", "D")
    vla.rotate_synthesized_beam(0, numpy.pi)

    # Question 3-2
    lofar = Telescope("LOFAR")
    lofar.rotate_synthesized_beam(0, 0)

    if SHOWPLOTS:
        pyplot.show()

    import sys; sys.exit(0)

    for setup in ['maxi-short', 'traditional36', '2x48', 'mini-short', '2x96']:
        westerbork_instance = Telescope("Westerbork", setup)
        # westerbork.plot_ground_grid()
        westerbork.rotate_synthesized_beam()
