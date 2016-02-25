"""
File: initial_conditions.py
Author: Timo L. R. Halbesma <timo.halbesma@student.uva.nl>
Date created: Thu Feb 25, 2016 10:03 am
Last modified: Thu Feb 25, 2016 05:19 pm

Description: Initial conidtions for two subclusters in Cygnus A cluster.

"""

import numpy

from matplotlib import pyplot
import matplotlib.gridspec as gridspec
pyplot.rcParams.update({'font.size': 22})
from mpl_toolkits.mplot3d import Axes3D

from amuse.units import units
from amuse.units import constants
from amuse.units import nbody_system
from amuse.units.quantities import VectorQuantity

# TODO: change Plummer sphere to.. something else?
from amuse.ic.plummer import new_plummer_model
from amuse.ic.gasplummer import new_plummer_gas_model

from amuse.community.gadget2.interface import Gadget2

from amuse.plot import plot
from amuse.plot import scatter
from amuse.plot import xlabel
from amuse.plot import ylabel
from amuse.plot import xlim
from amuse.plot import ylim


def smart_length_units_for_vector_quantity(quantity):
    length_units = [units.Mpc, units.kpc, units.parsec, units.AU, units.RSun, units.km]
    total_size = max(quantity) - min(quantity)
    for length_unit in length_units:
        if total_size > (1 | length_unit):
            return length_unit
    return units.m


class SubCluster(object):
    """ Set up a virialized cluster of virial radius Rvir, and total mass Mtot,
        where gas_fraction of the mass is in the cluster gas and 1-gas_fraction
        of the mass is dark matter. Both the gas and dark matter are initially
        at rest and located at the center. For now we assume Plummer spheres. """
    def __init__(self, name, Mtot=(1e15 | units.MSun), Rvir=(500 | units.kpc)):
        self.name = name
        self.converter = nbody_system.nbody_to_si(Mtot, Rvir)
        number_of_dm_particles = 1e3
        number_of_gas_particles = 1e4

        # Set up numerical smoothing fractions
        dm_smoothing_fraction = 0.001
        gas_smoothing_fraction = 0.05
        dm_epsilon = dm_smoothing_fraction * Rvir
        gas_epsilon = gas_smoothing_fraction * Rvir

        # Setup up gas and dark matter fractions and gas/dm mass
        gas_fraction = 0.9
        dm_fraction = 1.0 - gas_fraction
        dm_mass = dm_fraction * Mtot
        gas_mass = gas_fraction * Mtot

        # Set up dark matter particles
        # TODO: probably not use Plummer sphere
        self.dm = new_plummer_model(number_of_dm_particles, convert_nbody=self.converter)
        self.dm.radius = dm_epsilon
        self.dm.mass = (1.0/number_of_dm_particles) * dm_mass
        self.dm.move_to_center()

        # Set up virialized gas sphere of N gas particles, Plummer sphere
        # TODO: probably not use Plummer sphere
        self.gas = new_plummer_gas_model(number_of_gas_particles, convert_nbody=self.converter)
        self.gas.h_smooth = gas_epsilon
        self.gas.move_to_center()
        self.gas.mass = (1.0/number_of_gas_particles) * gas_mass

        print "Created ", str(self)

    def __str__(self):
        tmp = "{0}\n".format(self.name)
        tmp += "Mtot: {0}\n".format((self.dm.mass.sum() + self.gas.mass.sum()).as_quantity_in(units.MSun))
        tmp += "Rvir dm: {0}\n".format(self.dm.virial_radius().as_quantity_in(units.kpc))
        tmp += "Rvir gas: {0}\n".format(self.gas.virial_radius().as_quantity_in(units.kpc))

        return tmp

    def dm_rvir_gas_sph_3dsubplot(self):
        fig = pyplot.figure(figsize=(20, 10))
        ax_dm = fig.add_subplot(121, aspect='equal', projection='3d')
        ax_gas = fig.add_subplot(122, aspect='equal', projection='3d',
            sharex=ax_dm, sharey=ax_dm)

        # plot dark matter
        center_of_mass = self.dm.center_of_mass()
        virial_radius = self.dm.virial_radius().as_quantity_in(units.kpc)
        innersphere = self.dm.select(lambda r: (center_of_mass-r).length()<virial_radius,["position"])
        outersphere = self.dm.select(lambda r: (center_of_mass-r).length()>= virial_radius,["position"])
        pyplot.gcf().sca(ax_dm)
        x = outersphere.x.as_quantity_in(units.kpc)
        y = outersphere.y.as_quantity_in(units.kpc)
        z = outersphere.z.as_quantity_in(units.kpc)
        plot(x, y, z, 'o', c='red', label=r'$r \geq r_{\rm vir}$')
        x = innersphere.x.as_quantity_in(units.kpc)
        y = innersphere.y.as_quantity_in(units.kpc)
        z = innersphere.z.as_quantity_in(units.kpc)
        plot(x, y, z, 'o', c='green', label=r'$r < r_{\rm vir}$')
        xlabel(r'$x$')
        ylabel(r'$y$')
        ax_dm.set_zlabel(r'$z$ [{0}]'.format(virial_radius.unit))
        pyplot.legend()

        # plot gas as sph plot
        # Adjusted code from amuse.plot.sph_particles_plot
        pyplot.gcf().sca(ax_gas)
        min_size = 100
        max_size = 10000
        alpha = 0.1
        x = self.gas.x.as_quantity_in(units.kpc)
        y = self.gas.y.as_quantity_in(units.kpc)
        z = self.gas.z.as_quantity_in(units.kpc)
        z, x, y, us, h_smooths = z.sorted_with(x, y, self.gas.u, self.gas.h_smooth)
        u_min, u_max = min(us), max(us)

        log_u = numpy.log((us / u_min)) / numpy.log((u_max / u_min))
        clipped_log_u = numpy.minimum(numpy.ones_like(log_u), numpy.maximum(numpy.zeros_like(log_u), log_u))

        red = 1.0 - clipped_log_u**4
        blue = clipped_log_u**4
        green = numpy.minimum(red, blue)

        colors = numpy.transpose(numpy.array([red, green, blue]))
        n_pixels = pyplot.gcf().get_dpi() * pyplot.gcf().get_size_inches()

        ax_gas.set_axis_bgcolor('#101010')
        ax_gas.set_aspect("equal", adjustable = "datalim")
        phys_to_pix2 = n_pixels[0]*n_pixels[1] / ((max(x)-min(x))**2 + (max(y)-min(y))**2)
        sizes = numpy.minimum(numpy.maximum((h_smooths**2 * phys_to_pix2), min_size), max_size)

        ax_gas.scatter(x.number, y.number, z.number, color=colors, s=sizes, edgecolors="none", alpha=alpha)
        xlabel(r'$x$')
        ylabel(r'$y$')
        ax_gas.set_zlabel(r'$z$ [{0}]'.format(virial_radius.unit))

        xlim(-2.*virial_radius, 2*virial_radius)
        ylim(-2.*virial_radius, 2*virial_radius)
        ax_dm.set_zlim(-2.*virial_radius.number, 2*virial_radius.number)
        ax_gas.set_zlim(-2.*virial_radius.number, 2*virial_radius.number)
        pyplot.tight_layout()
        pyplot.show()

    def dm_rvir_gas_sph_subplot(self):
        fig = pyplot.figure(figsize=(20, 10))
        ax_dm = fig.add_subplot(121, aspect='equal')
        ax_gas = fig.add_subplot(122, aspect='equal',
            sharex=ax_dm, sharey=ax_dm)

        # plot dark matter
        center_of_mass = self.dm.center_of_mass()
        virial_radius = self.dm.virial_radius().as_quantity_in(units.kpc)
        innersphere = self.dm.select(lambda r: (center_of_mass-r).length()<virial_radius,["position"])
        outersphere = self.dm.select(lambda r: (center_of_mass-r).length()>= virial_radius,["position"])
        pyplot.gcf().sca(ax_dm)
        x = outersphere.x.as_quantity_in(units.kpc)
        y = outersphere.y.as_quantity_in(units.kpc)
        scatter(x, y, c='red', edgecolor='red', label=r'$r \geq r_{\rm vir}$')
        x = innersphere.x.as_quantity_in(units.kpc)
        y = innersphere.y.as_quantity_in(units.kpc)
        scatter(x, y, c='green', edgecolor='green', label=r'$r < r_{\rm vir}$')
        xlabel(r'$x$')
        ylabel(r'$y$')
        pyplot.legend()

        # plot gas as sph plot

        # Adjusted code from amuse.plot.sph_particles_plot
        pyplot.gcf().sca(ax_gas)
        min_size = 100
        max_size = 10000
        alpha = 0.1
        x = self.gas.x
        y = self.gas.y
        z = self.gas.z
        z, x, y, us, h_smooths = z.sorted_with(x, y, self.gas.u, self.gas.h_smooth)
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
        xlabel(r'$x$')
        ylabel(r'$y$')

        xlim(-2.*virial_radius, 2*virial_radius)
        ylim(-2.*virial_radius, 2*virial_radius)
        pyplot.tight_layout()
        pyplot.show()


class ClusterMerger(object):
    def __init__(self, massratio=1./3):
        self.subClusterA = SubCluster(name="Sub Cluster A")
        self.subClusterB = SubCluster(name="Sub Cluster B",
            Mtot=massratio*(1e15 | units.MSun), Rvir=(200 | units.kpc))
        self.converter = self.subClusterA.converter

        # self.subClusterA.dm_rvir_gas_sph_3dsubplot()
        # self.subClusterA.dm_rvir_gas_sph_subplot()

        self.place_clusters_in_world()
        self.setup_codes()

    def place_clusters_in_world(self):
        self.subClusterA.dm.rotate(0.0, numpy.pi/4, 0.0)
        self.subClusterA.dm.position += [0.0, 0.0, 0.0] | units.kpc
        self.subClusterA.dm.velocity -= [0.0, 0.0, 0.0] | units.km/units.s
        self.subClusterA.gas.rotate(0.0, numpy.pi/4, 0.0)
        self.subClusterA.gas.position += [0.0, 0.0, 0.0] | units.kpc
        self.subClusterA.gas.velocity -= [0.0, 0.0, 0.0] | units.km/units.s

        self.subClusterB.dm.rotate(numpy.pi/6, 0.0, 0.0)
        self.subClusterB.dm.position += [1.0, 0.0, 0.0] | units.Mpc
        self.subClusterB.dm.velocity += [1000.0, 0.0, 0.0] | units.km/units.s
        self.subClusterB.gas.rotate(numpy.pi/6, 0.0, 0.0)
        self.subClusterB.gas.position += [1.0, 0.0, 0.0] | units.Mpc
        self.subClusterB.gas.velocity += [1000.0, 0.0, 0.0] | units.km/units.s

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

        self.code.evolve_model(1 | units.yr)

        # self.code.stop()

    def dm_gas_sph_subplot(self):
        fig = pyplot.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 4])
        # gs.update(left=0.05, right=0.48, wspace=0.05)

        ax_text = pyplot.subplot(gs[0, :])
        ax_text.axis('off')
        time_text = ax_text.text(0.02, 1.0, '', transform=ax_text.transAxes, fontsize=42)
        time_text.set_text('Time: {0:.1f} Gyr'.format(self.code.model_time.value_in(units.Myr)))
        energy_text = ax_text.text(0.02, 0.7, '', transform=ax_text.transAxes, fontsize=42)
        energy_text.set_text('Etot: {0:.3e} erg'.format(self.Etot_init.value_in(units.erg)))

        ax_dm = pyplot.subplot(gs[1, 0])
        ax_gas = pyplot.subplot(gs[1, 1], aspect='equal',
            sharex=ax_dm, sharey=ax_dm)
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
        pyplot.show()


if __name__ == '__main__':
    merger = ClusterMerger()
