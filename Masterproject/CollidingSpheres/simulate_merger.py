"""
File: simulate_merger.py
Author: Timo L. R. Halbesma <timo.halbesma@student.uva.nl>
Version: 0.01 (Initial)
Date created: Fri Mar 04, 2016 03:46 pm
Last modified: Sat Mar 05, 2016 05:16 PM

Description: Main routine for Cygnus A merger

"""

from matplotlib import pyplot
pyplot.rcParams.update({'font.size': 22})

from amuse.units import units
from amuse.units.quantities import VectorQuantity
from amuse.plot import plot
from amuse.plot import scatter
from amuse.plot import xlabel
from amuse.plot import ylabel
from amuse.io import write_set_to_file

from initial_subcluster import SubCluster
from initial_world import ClusterMerger
from helper_functions import print_progressbar


class CygAMergerSimulation(object):
    def __init__(self):
        self.merger = ClusterMerger()

        timesteps = VectorQuantity.arange(0 | units.Myr, 1 | units.Gyr, 50 | units.Myr)
        tot = len(timesteps)
        end_time = timesteps[-1]

        print "Starting Simulation :-)"
        print "Generating plots on the fly :-)"

        gasA_vel_list = [] | (units.km/units.s)
        dmA_vel_list = [] | (units.km/units.s)
        gasB_vel_list = [] | (units.km/units.s)
        dmB_vel_list = [] | (units.km/units.s)
        time_list = [] | units.Gyr
        for i, time in enumerate(timesteps):
            print_progressbar(i, tot)
            self.merger.code.evolve_model(time)
            self.merger.dm_gas_sph_subplot(i)
            gasA_vel_list.append(self.merger.gasA.center_of_mass_velocity())
            dmA_vel_list.append(self.merger.dmA.center_of_mass_velocity())
            gasB_vel_list.append(self.merger.gasB.center_of_mass_velocity())
            dmB_vel_list.append(self.merger.dmB.center_of_mass_velocity())
            time_list.append(time)
            write_set_to_file(self.merger.code.particles,
                'out/{0}/data/cluster_{1}.amuse'.format(self.merger.timestamp, i),
                "amuse")

        print "Plotting velocity as function of time"
        fig = pyplot.figure(figsize=(12, 10), dpi=50)
        plot(time_list.number, gasA_vel_list.number, label="gasA", c='r', ls='solid')
        plot(time_list.number, dmA_vel_list.number, label="dmA", c='r', ls='dashed')
        plot(time_list.number, gasB_vel_list.number, label="gasB", c='g', ls='solid')
        plot(time_list.number, dmB_vel_list.number, label="dmB", c='g', ls='dashed')
        xlabel("Time")
        ylabel("Velocity")
        pyplot.legend()
        pyplot.show()

        print "Generating gif :-)"
        self.merger.create_gif()

        print "Stopping the code. End of pipeline :-)"
        self.merger.code.stop()


if __name__ == '__main__':
    merger = CygAMergerSimulation()
