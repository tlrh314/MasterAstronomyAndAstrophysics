"""
File: initial_world.py
Author: Timo L. R. Halbesma <timo.halbesma@student.uva.nl>
Date created: Fri Mar 04, 2016 03:45 pm
Last modified: Sat Mar 05, 2016 05:18 PM

Description: Initial conidtions for Cygnus A cluster.

"""

import os
import glob
from datetime import datetime
import pickle

import numpy

from matplotlib import pyplot
from matplotlib import animation
import matplotlib.gridspec as gridspec
pyplot.rcParams.update({'font.size': 22})

from PIL import Image
import images2gif # https://raw.githubusercontent.com/rec/echomesh/master/code/python/external/images2gif.py

from amuse.units import units
from amuse.units import nbody_system
from amuse.community.gadget2.interface import Gadget2
from amuse.plot import scatter
from amuse.plot import xlabel
from amuse.plot import ylabel

from initial_subcluster import SubCluster
from helper_functions import smart_length_units_for_vector_quantity

class ClusterMerger(object):
    def __init__(self, massratio=1./3):
        # Set up directories to store data in
        self.timestamp = datetime.today().strftime('%Y%m%dT%H%M')
        if not os.path.exists('out/{0}'.format(self.timestamp)):
            os.mkdir('out/{0}'.format(self.timestamp))
        if not os.path.exists('out/{0}/plots'.format(self.timestamp)):
            os.mkdir('out/{0}/plots'.format(self.timestamp))
        if not os.path.exists('out/{0}/data'.format(self.timestamp)):
            os.mkdir('out/{0}/data'.format(self.timestamp))

        # Set up sub clusters
        self.subClusterA = SubCluster(name="Sub Cluster A")
        self.subClusterB = SubCluster(name="Sub Cluster B",
            Mtot=massratio*(1e15 | units.MSun), Rvir=(200 | units.kpc))
        self.converter = self.subClusterA.converter

        # self.subClusterA.dm_rvir_gas_sph_3dsubplot()
        # self.subClusterA.dm_rvir_gas_sph_subplot()

        # Set up world and gravity/hydro solvers
        self.place_clusters_in_world()

        # Write simulation parameters to text file
        # pickle.dump(self, open('{0}/data/merger.dat'.format(self.timestamp), 'wb'))

        # Set up simulation codes
        self.setup_codes()


    def place_clusters_in_world(self):
        self.subClusterA.dm.rotate(0.0, numpy.pi/4, 0.0)
        self.subClusterA.dm.position += [0.0, 0.0, 0.0] | units.kpc
        self.subClusterA.dm.velocity -= [0.0, 0.0, 0.0] | units.km/units.s
        self.subClusterA.gas.rotate(0.0, numpy.pi/4, 0.0)
        self.subClusterA.gas.position += [0.0, 0.0, 0.0] | units.kpc
        self.subClusterA.gas.velocity -= [0.0, 0.0, 0.0] | units.km/units.s

        self.subClusterB.dm.rotate(numpy.pi/6, 0.0, 0.0)
        self.subClusterB.dm.position += [5.0, 0.0, 0.0] | units.Mpc
        self.subClusterB.dm.velocity -= [2000.0, 0.0, 0.0] | units.km/units.s
        self.subClusterB.gas.rotate(numpy.pi/6, 0.0, 0.0)
        self.subClusterB.gas.position += [5.0, 0.0, 0.0] | units.Mpc
        self.subClusterB.gas.velocity -= [2000.0, 0.0, 0.0] | units.km/units.s

    def setup_codes(self):
        # 1 code is created and started
        self.code = Gadget2(self.converter, redirection='none', number_of_workers=4)
        # 2 parameters are set
        self.code.parameters.epsilon_squared = 0.0000001 | nbody_system.length**2
        self.dmA = self.code.dm_particles.add_particles(self.subClusterA.dm)
        self.gasA = self.code.gas_particles.add_particles(self.subClusterA.gas)
        self.dmB = self.code.dm_particles.add_particles(self.subClusterB.dm)
        self.gasB = self.code.gas_particles.add_particles(self.subClusterB.gas)
        self.code.commit_particles()

        self.Etot_init = self.code.kinetic_energy + self.code.potential_energy + self.code.thermal_energy
        print "Ekin:", self.code.kinetic_energy
        print "Epot:", self.code.potential_energy
        print "Eth:", self.code.thermal_energy

    def dm_gas_sph_subplot(self, i=0):
        fig = pyplot.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 4])
        # gs.update(left=0.05, right=0.48, wspace=0.05)

        ax_text = pyplot.subplot(gs[0, :])
        ax_text.axis('off')
        time_text = ax_text.text(0.02, 1.0, '', transform=ax_text.transAxes, fontsize=42)
        time_text.set_text('Time: {0:.1f} Myr'.format(self.code.model_time.value_in(units.Myr)))
        energy_text = ax_text.text(0.02, -0.2, '', transform=ax_text.transAxes, fontsize=42)

        Ekin = self.code.kinetic_energy.value_in(units.erg)
        Epot = self.code.potential_energy.value_in(units.erg)
        Eth = self.code.thermal_energy.value_in(units.erg)

        energy_text.set_text('Ekin: {0:.3e} erg\nEpot: {1:.3e} erg\nEth: {2:.3e} erg'
            .format(Ekin, Epot, Eth))


        # lim = max(abs((self.dmA.center_of_mass() - self.dmB.center_of_mass()).value_in(units.Mpc)))
        lim = 2.5
        ax_dm = pyplot.subplot(gs[1, 0], xlim=(-4*lim, 4*lim), ylim=(-4*lim, 4*lim))
        ax_gas = pyplot.subplot(gs[1, 1], aspect='equal',
            sharex=ax_dm, sharey=ax_dm, xlim=(-4*lim, 4*lim), ylim=(-4*lim, 4*lim))
        # ax_dm = fig.add_subplot(121, aspect='equal')
        # ax_gas = fig.add_subplot(122, aspect='equal',
        #    sharex=ax_dm, sharey=ax_dm)

        # plot dark matter
        pyplot.gcf().sca(ax_dm)
        x = self.dmA.x.as_quantity_in(units.Mpc)
        y = self.dmA.y.as_quantity_in(units.Mpc)
        scatter(x, y, c='red', edgecolor='red', label=str(self.subClusterA.name))
        x = self.dmB.x.as_quantity_in(units.Mpc)
        y = self.dmB.y.as_quantity_in(units.Mpc)
        scatter(x, y, c='green', edgecolor='green', label=str(self.subClusterB.name))
        xlabel(r'$x$')
        ylabel(r'$y$')
        pyplot.legend()

        # plot gas as sph plot
        def plot_sph(gas):
            # Adjusted code from amuse.plot.sph_particles_plot
            pyplot.gcf().sca(ax_gas)
            min_size = 100
            max_size = 10000
            alpha = 0.1
            x = gas.x
            y = gas.y
            z = gas.z
            z, x, y, us, h_smooths = z.sorted_with(x, y, gas.u, gas.h_smooth)
            u_min, u_max = min(us), max(us)

            log_u = numpy.log((us / u_min)) / numpy.log((u_max / u_min))
            clipped_log_u = numpy.minimum(numpy.ones_like(log_u), numpy.maximum(numpy.zeros_like(log_u), log_u))

            red = 1.0 - clipped_log_u**4
            blue = clipped_log_u**4
            green = numpy.minimum(red, blue)

            colors = numpy.transpose(numpy.array([red, green, blue]))
            n_pixels = pyplot.gcf().get_dpi() * pyplot.gcf().get_size_inches()

            ax_gas.set_axis_bgcolor('#101010')
            ax_gas.set_aspect("equal", adjustable="datalim")
            length_unit = smart_length_units_for_vector_quantity(x)
            phys_to_pix2 = n_pixels[0]*n_pixels[1] / ((max(x)-min(x))**2 + (max(y)-min(y))**2)
            sizes = numpy.minimum(numpy.maximum((h_smooths**2 * phys_to_pix2), min_size), max_size)

            scatter(x.as_quantity_in(length_unit), y.as_quantity_in(length_unit),
                    color=colors, s=sizes, edgecolors="none", alpha=alpha)

        plot_sph(self.gasA)
        plot_sph(self.gasB)
        xlabel(r'$x$')
        ylabel(r'$y$')

        pyplot.tight_layout()
        pyplot.savefig('out/{0}/plots/dm_gas_sph_subplot_{1}.png'
            .format(self.timestamp, i), dpi=50)
        # pyplot.show()
        pyplot.close()


    def animate(self):
        # Set up figure, axes, and PathCollection instances
        fig = pyplot.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 4])

        self.ax_text = pyplot.subplot(gs[0, :])
        self.ax_text.axis('off')
        self.time_text = self.ax_text.text(0.02, 1.0, '', transform=self.ax_text.transAxes, fontsize=42)
        self.energy_text = self.ax_text.text(0.02, -0.2, '', transform=self.ax_text.transAxes, fontsize=42)

        lim = max(abs((self.dmA.center_of_mass() - self.dmB.center_of_mass()).value_in(units.Mpc)))
        self.ax_dm = pyplot.subplot(gs[1, 0], xlim=(-4*lim, 4*lim), ylim=(-4*lim, 4*lim))
        self.ax_dm.set_xlabel(r'$x$')
        self.ax_dm.set_ylabel(r'$y$')
        self.subAdm_scat, = self.ax_dm.plot([], [], 'ro', ms=6, label="A")
        self.subBdm_scat, = self.ax_dm.plot([], [], 'go', ms=6, label="B")
        pyplot.legend()

        self.ax_gas = pyplot.subplot(gs[1, 1], aspect='equal',
            sharex=self.ax_dm, sharey=self.ax_dm, xlim=(-4*lim, 4*lim), ylim=(-4*lim, 4*lim))
        self.ax_gas.set_xlabel(r'$x$')
        self.ax_gas.set_ylabel(r'$y$')
        self.ax_gas.set_axis_bgcolor('#101010')
        self.subAgas_scat = self.ax_gas.scatter([], [], alpha=0.1, edgecolors="none")
        self.subBgas_scat = self.ax_gas.scatter([], [], alpha=0.1, edgecolors="none")

        # 20 fps at 5 Myr interval --> 1 second in animation is 100 Myr
        self.timesteps = VectorQuantity.arange(0.0 | units.Myr, 100 | units.Myr, 5 | units.Myr)

        # pyplot.show()

        self.anim = animation.FuncAnimation(fig, self.update,
            frames=self.timesteps, blit=True, init_func=self.clear)

    def clear(self):
        self.time_text.set_text('')
        self.energy_text.set_text('')
        self.subAdm_scat.set_data([], [])
        self.subBdm_scat.set_data([], [])
        self.subAgas_scat.set_offsets([[], []])
        self.subBgas_scat.set_offsets([[], []])

        return self.time_text, self.energy_text, self.subAdm_scat, self.subBdm_scat, \
            self.subAgas_scat, self.subBgas_scat

    def update(self, time):
        i = numpy.where(self.timesteps == time)[0][0]
        tot = len(self.timesteps)
        end_time = self.timesteps[-1]
        print_progressbar(i, tot, end_time)

        self.code.evolve_model(time.as_quantity_in(units.Myr))

        Ekin = self.code.kinetic_energy.value_in(units.erg)
        Epot = self.code.potential_energy.value_in(units.erg)
        Eth = self.code.thermal_energy.value_in(units.erg)

        self.time_text.set_text('Time: {0:.1f} Gyr'.format(self.code.model_time.value_in(units.Myr)))
        self.energy_text.set_text('Ekin: {0:.3e} erg\nEpot: {1:.3e} erg\nEth: {2:.3e} erg'
            .format(Ekin, Epot, Eth))

        # plot dark matter
        # pyplot.gcf().sca(self.ax_dm)
        x = self.dmA.x.value_in(units.Mpc)
        y = self.dmA.y.value_in(units.Mpc)
        self.subAdm_scat.set_data(x, y)
        x = self.dmB.x.value_in(units.Mpc)
        y = self.dmB.y.value_in(units.Mpc)
        self.subBdm_scat.set_data(x, y)

        def plot_sph(gas, scat):
            # Adjusted code from amuse.plot.sph_particles_plot
            # pyplot.gcf().sca(self.ax_gas)
            min_size = 100
            max_size = 10000
            alpha = 0.1
            x = gas.x
            y = gas.y
            z = gas.z
            z, x, y, us, h_smooths = z.sorted_with(x, y, gas.u, gas.h_smooth)

            u_min, u_max = min(us), max(us)

            log_u = numpy.log((us / u_min)) / numpy.log((u_max / u_min))
            clipped_log_u = numpy.minimum(numpy.ones_like(log_u), numpy.maximum(numpy.zeros_like(log_u), log_u))

            red = 1.0 - clipped_log_u**4
            blue = clipped_log_u**4
            green = numpy.minimum(red, blue)

            colors = numpy.transpose(numpy.array([red, green, blue]))
            n_pixels = pyplot.gcf().get_dpi() * pyplot.gcf().get_size_inches()

            # self.ax_gas.set_aspect("equal", adjustable="datalim")
            phys_to_pix2 = n_pixels[0]*n_pixels[1] / ((max(x)-min(x))**2 + (max(y)-min(y))**2)
            sizes = numpy.minimum(numpy.maximum((h_smooths**2 * phys_to_pix2), min_size), max_size)

            scat.set_offsets([x.value_in(units.Mpc), y.value_in(units.Mpc)])
            scat.set_sizes(sizes)
            scat.set_color(colors)

        plot_sph(self.gasA, self.subAgas_scat)
        plot_sph(self.gasB, self.subBgas_scat)

        return self.time_text, self.energy_text, self.subAdm_scat, self.subBdm_scat, \
            self.subAgas_scat, self.subBgas_scat

    def create_gif(self):
        # Convert matplotlib generated plots to a gif using imges2gif (requires Pillow!)
        images = []
        sorted_plots = sorted(glob.glob('out/{0}/plots/*.png'.format(self.timestamp)),
                              key=os.path.getmtime)
        for plot in sorted_plots:
            images.append(Image.open(plot))

        #with NamedTemporaryFile(suffix='.gif') as f:
            #print f.name
        self.gifname = 'out/{0}/plots/clustermerger.gif'.format(self.timestamp)
        images2gif.writeGif(self.gifname, images)

    def gif_to_html(self):
        IMG_TAG = """<img src="data:image/gif;base64,{0}"></img>"""

        # convert gif to html
        data = open(self.gifname, 'rb').read()
        data = data.encode('base64')

        return IMG_TAG.format(data)
