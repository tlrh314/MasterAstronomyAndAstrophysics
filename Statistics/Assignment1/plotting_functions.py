"""
File: plotting_functions.py
Authors: Timo L. R. Halbesma <timo.halbesma@student.uva.nl>
         Sarah Brands <saaaaaaaar@hotmail.com>
         Geert Raaijmakers <raaymakers8@hotmail.com>

Version: 1.0 (Final Draft)
Date created: Thu Nov 19, 2015 02:53 PM
Last modified: Thu Nov 19, 2015 03:45 PM

Description:
    Statistical Methods for the Physical Sciences (5214SMFA3Y) --
    Group mini-project --
    Sloan Digital Sky Survey Data Release 7 (SDSS DR7) --
    Catalog of Quasar Properties (CoQP) -- Data from Yue Shen (2011)
    See http://adsabs.harvard.edu/abs/2011ApJS..194...45S

Usage:
    See Statistical Methods Group Assignment TLRH, SAB, GR notebook
"""

import sys
import time

import numpy
import scipy
import pandas
from scipy import stats
import matplotlib
from matplotlib import pyplot

# from parse_dataset import parse_dataset
from find_intersection import find_intersection


def progressbar(current, total):
    # Print readout progress bar =)
    bar_width = 42  # obviously
    if ((current == total-1) or (current % 100 is 0)):

        progress = float(current)/total
        block = int(round(bar_width * progress))
        sys.stdout.write(
            "\r[{0}{1}] {2:.2f}%"
            .format('#'*block,
                    ' '*(bar_width - block),
                    progress*100))
        sys.stdout.flush()


def plot_colour_versus_mass(quasars, use_arrays=True):
    """ Find out whether colour is correlated with mass """

    if use_arrays:
        start_time = time.time()
        LOGL3000, LOGL3000_ERR, LOGL5100, LOGL5100_ERR, LOGBH, LOGBH_ERR =\
            find_intersection(quasars, 'LOGL3000', 'LOGL5100', 'LOGBH')
        print "Finding intersection takes {0} seconds." \
              .format(time.time() - start_time)

    start_time = time.time()

    pyplot.figure()

    total = len(quasars)
    plotted_count = 0
    if use_arrays:
        # LOGBH[(numpy.where(LOGBH_ERR < 0.01*LOGBH))]
        pyplot.scatter(LOGBH, LOGL5100 - LOGL3000)
        plotted_count = len(LOGL3000)
    else:
        for i, quasar in enumerate(quasars):
            if quasar and quasar.LOGBH and quasar.LOGL3000 and quasar.LOGL5100:
                    # and quasar.LOGBH_ERR < 0.1*quasar.LOGBH:
                pyplot.scatter(quasar.LOGBH,
                               (quasar.LOGL5100 - quasar.LOGL3000))
                progressbar(i, total)
                plotted_count += 1

    sys.stdout.write('\n')

    pyplot.title("Colour - mass diagram")
    pyplot.ylabel("Colour [log(luminosity_5100) - log(luminosity_3000)]")
    pyplot.xlabel("Log of Black Hole mass [Msun]")

    print "Plotting {0} quasars takes {1} seconds." \
          .format(plotted_count, time.time() - start_time)
    pyplot.show()

    n = plotted_count
    rho, pval = scipy.stats.spearmanr(LOGL5100 - LOGL3000, LOGBH)
    t = rho * numpy.sqrt((n-2)/(1-rho**2))
    print "Spearman's Rho = {0}, p-value = {1}\nt = {2}".format(rho, pval, t)

    r, pval = scipy.stats.pearsonr(LOGL5100 - LOGL3000, LOGBH)
    t = r * numpy.sqrt((n-2)/(1-r**2))
    print "\nPearsons's r = {0}, p-value = {1}\nt = {2}".format(r, pval, t)

    slope, intercept, r_value, p_value, std_err =\
        stats.linregress(LOGL5100 - LOGL3000, LOGBH)
    print "\nLinear Regression"
    print "slope={0:.2f}\tintercept={1:.2f}".format(slope, intercept),
    print "\tr_value={0:.2f}\tp_value={1:.2f}".format(r_value, p_value),
    print "\tstd_err={0:.2f}".format(std_err)


def plot_loglbol_bhmass_redshift(quasars):
    """ Create Figure 2 from Meusinger & Weiss (2013) """

    LOGLBOL, LOGLBOL_ERR, LOGBH, LOGBH_ERR, REDSHIFT, REDSHIFT_ERR =\
        find_intersection(quasars, 'LOGLBOL', 'LOGBH', 'REDSHIFT')

    pyplot.scatter(LOGBH, LOGLBOL, lw=0.01, s=1, c=REDSHIFT,
                   norm=matplotlib.colors.LogNorm(vmin=REDSHIFT.min(),
                                                  vmax=REDSHIFT.max()),
                   cmap=pyplot.get_cmap('afmhot_r'))

    pyplot.title('Mean redshift (colour)')
    pyplot.ylabel('log luminosity [erg/s]')
    pyplot.xlabel('log BH mass [MSun]')
    pyplot.xlim(numpy.min(LOGBH), numpy.max(LOGBH))
    pyplot.ylim(numpy.min(LOGLBOL), numpy.max(LOGLBOL))

    myticks = numpy.logspace(numpy.log10(REDSHIFT.min()),
                             numpy.log10(REDSHIFT.max()), 10)
    ticklabels = numpy.round(myticks, decimals=2)
    cbar = pyplot.colorbar(ticks=myticks)
    cbar.set_ticklabels(ticklabels)
    pyplot.show()


def show_eddington_limit(bolometric_luminosity, bolometric_luminosity_err,
                         blackhole_mass, blackhole_mass_err,
                         number_of_quasars):
    """ Generate plow of Eddington Limit """

    # TODO: justify 0.1 cutoff
    threshold = 0.1

    if type(bolometric_luminosity) is not list:
        blackhole_mass_err = list(blackhole_mass_err)
        blackhole_mass = list(blackhole_mass)
        bolometric_luminosity = list(bolometric_luminosity)
        bolometric_luminosity_err = list(bolometric_luminosity_err)

    for i in range(number_of_quasars):
        if blackhole_mass_err[i] > threshold*blackhole_mass[i]:
            del blackhole_mass_err[i]
            del blackhole_mass[i]
            del bolometric_luminosity[i]
            del bolometric_luminosity_err[i]

    # FIXME: idea on how to do this for numpy arrays, but this does not work.
    # indices = numpy.where(blackhole_mass_err < threshold*blackhole_mass)
    # blackhole_mass_err = blackhole_mass_err[indices]
    # blackhole_mass = blackhole_mass[indices]
    # bolometric_luminosity[indices]
    # bolometric_luminosity_err[indices]

    x = numpy.linspace(numpy.nanmin(blackhole_mass[0:number_of_quasars])-0.5,
                       numpy.nanmax(blackhole_mass[0:number_of_quasars]), 100)
    y = 38.1 + x

    pyplot.figure(figsize=(10, 10), dpi=100)
    pyplot.plot(x, y, color='red')
    pyplot.errorbar(blackhole_mass[0:number_of_quasars],
                    bolometric_luminosity[0:number_of_quasars],
                    xerr=blackhole_mass_err[0:number_of_quasars],
                    yerr=bolometric_luminosity_err[0:number_of_quasars],
                    ls='none', color='grey')
    pyplot.xlabel('Blackhole Mass (Logarithmic and in solar mass units)',
                  fontsize=15)
    pyplot.ylabel('Luminosity (Logarithmic in ergs/s)', fontsize=15)
    pyplot.tick_params(axis='both', which='major', labelsize=15)
    pyplot.title('Eddington Limit', fontsize=17)
    pyplot.show()


def plot_matrixplot(quasars, number_quasars=None):

    if number_quasars is None:
        number_quasars = len(quasars)

    start_time = time.time()

    continuum_300 = []
    continuum_510 = []
    Hb_narrow = []
    OIII_5007 = []
    MGII = []

    for i in range(1, number_quasars):
        if quasars[i]:
            continuum_300.append(quasars[i].LOGL3000)
            continuum_510.append(quasars[i].LOGL5100)
            OIII_5007.append(quasars[i].LOGL_OIII_5007)
            Hb_narrow.append(quasars[i].LOGL_NARROW_HB)
            MGII.append(quasars[i].LOGL_MGII)

    rand_data = numpy.array([continuum_510, continuum_300,
                             Hb_narrow, MGII]).transpose()
    ndims = rand_data.shape[1]
    fig, axes = pyplot.subplots(4, 4, figsize=(14, 14))
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    axes[0, 0].set_title('Continuum 510 nm')
    axes[0, 1].set_title('Continuum 300 nm')
    axes[0, 2].set_title('H-Beta narrow line')
    axes[0, 3].set_title('Magnesium II line')

    for i in xrange(ndims):
        for j in xrange(ndims):
            if i == j:
                axes[i, j].hist(
                    rand_data[:, i][rand_data[:, i] != numpy.array(None)],
                    bins=15, facecolor='green', alpha=0.5)
                axes[i, j].set_xlabel('Log(L)')
                axes[i, j].set_ylabel('Frequency')
                axes[i, j].locator_params(nbins=7)

            else:
                axes[i, j].scatter(rand_data[:, i], rand_data[:, j], alpha=0.5)
                axes[i, j].set_xlabel('Log(L)')
                axes[i, j].set_ylabel('Log(L)')
                axes[i, j].locator_params(nbins=7)

    print "Plotting takes {0} seconds.".format(time.time() - start_time)
    pyplot.show()


def plot_sky_map(quasars, fraction=1, use_arrays=True):
    """ Create 2D sky map showing locations of all quasars. """

    if use_arrays:
        start_time = time.time()
        RA, RA_err, DEC, DEC_err =\
            find_intersection(quasars, 'RA', 'DEC')
        print "Finding intersection takes {0} seconds." \
              .format(time.time() - start_time)

    start_time = time.time()

    fig = pyplot.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="hammer")

    total = len(quasars)
    plotted_count = 0
    if use_arrays:
        pyplot.scatter(RA*(numpy.pi/180) - numpy.pi, DEC*(numpy.pi/180),
                       marker='.', color='red', s=4)
        plotted_count = len(RA)
    else:
        # Unacceptably slow: 190s for all datapoints.
        for i, quasar in enumerate(quasars):
            if quasar and quasar.RA and quasar.DEC and i % fraction == 0:
                ax.scatter(quasar.RA*(numpy.pi/180) - numpy.pi,
                           quasar.DEC*(numpy.pi/180),
                           marker='.', color='red', s=4)
                progressbar(i, total)
                plotted_count += 1
                # print quasar.RA, quasar.DEC

        sys.stdout.write('\n')

    ax.set_xticklabels(['02h', '04h', '06h', '08h', '10h',
                        '12h', '14h', '16h', '18h', '20h', '22h'])
    ax.grid(True)
    ax.set_xlabel('Right Ascension (Hours)', labelpad=150)
    ax.set_ylabel('Declination (Degrees)')
    ax.set_title("Distribution of Quasars", y=1.08)

    print "Plotting {0} quasars takes {1} seconds." \
          .format(plotted_count, time.time() - start_time)
    pyplot.show()


def radio_loudness_vs_colour(quasars, attr1, attr2,
                             describe=False, describe_subset=False):
    R_6CM_2500A, R_6CM_2500A_ERR, LOGL_1, LOGL_1_ERR, LOGL_2, LOGL_2_ERR\
        = find_intersection(quasars, 'R_6CM_2500A', attr1, attr2)

    if describe:
        s = pandas.Series(R_6CM_2500A)
        print(s.describe())
        return

    colourlist = []
    for i in xrange(len(LOGL_1)):
        colour = LOGL_1[i] - LOGL_2[i]
        colourlist.append(colour)

    radio_loud = []
    radio_quiet = []
    radio_loud_colour = []
    radio_quiet_colour = []

    for i in xrange(len(R_6CM_2500A)):
        if R_6CM_2500A[i] > 10:
            radio_loud.append(R_6CM_2500A[i])
            radio_loud_colour.append(colourlist[i])
        else:
            radio_quiet.append(R_6CM_2500A[i])
            radio_quiet_colour.append(colourlist[i])

    radio_loud = numpy.array(radio_loud)
    radio_quiet = numpy.array(radio_quiet)
    radio_loud_colour = numpy.array(radio_loud_colour)
    radio_quiet_colour = numpy.array(radio_quiet_colour)

    if describe_subset:
        loud = pandas.Series(radio_loud_colour)
        quiet = pandas.Series(radio_quiet_colour)

        print "Radio loud quasars:"
        print(loud.describe())
        print " "
        print "Radio quiet quasars:"
        print(quiet.describe())
        return

    return colourlist, radio_loud, radio_quiet,\
        radio_loud_colour, radio_quiet_colour


def plot_histogram_radioloud_radioquiet(radio_loud, radio_quiet,
                                        radio_loud_colour, radio_quiet_colour):
    """ Create histograms of two subsets: radio-loud and radio-quiet """
    import seaborn as sns

    fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(12, 6))
    sns.set(style="white", palette="muted")
    b = sns.color_palette("muted")[2]
    b2 = sns.color_palette("muted")[0]

    sns.distplot(radio_quiet_colour, bins=25, kde=True, kde_kws={"shade": True},
                 color=b, ax=ax1)
    ax1.set_xlabel("colour")
    ax1.set_ylabel("Density")
    ax1.set_title("colour (radio quiet quasars)")

    sns.set(style="white", palette="muted")

    sns.distplot(radio_loud_colour, bins=30, kde=True, kde_kws={"shade": True},
                 color=b2, ax=ax2)
    ax2.set_xlabel("colour")
    ax2.set_ylabel("Density")
    ax2.set_title("colour (radio loud quasars)")



def plot_radioloudness_vs_colour(colourlist, radio_loud, radio_quiet,
                                 radio_loud_colour, radio_quiet_colour):

    fig, (ax2, ax1) = pyplot.subplots(1, 2, figsize=(12, 6))

    ax1.scatter(radio_quiet, radio_quiet_colour, c="blue", label="Radio-quiet")
    ax1.scatter(radio_loud, radio_loud_colour, c="red", label="Radio-loud")
    ax1.set_xlim(0, 100)
    # ax1.set_ylim(16,-3)
    ax1.set_xlabel("Radio Loudness")
    ax1.set_ylabel("colour: Lbol(300nm) - Lbol(135nm)")
    ax1.legend()
    ax1.set_title("Radio Loudness vs colour (ZOOM)")

    ax2.scatter(radio_quiet, radio_quiet_colour, c="blue", label="Radio-quiet")
    ax2.scatter(radio_loud, radio_loud_colour, c="red", label="Radio-loud")
    # ax1.set_ylim(16,-3)
    ax2.set_xlabel("Radio Loudness")
    ax2.set_ylabel("colour: Lbol(300nm) - Lbol(135nm)")
    ax2.legend()
    ax2.set_title("Radio Loudness vs colour")

    pyplot.show()


def create_boxplot_radioloudness(radio_loud_colour, radio_quiet_colour):
    fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(8, 4))
    ax1.boxplot(radio_loud_colour)
    ax1.set_ylabel("Colour", fontsize=10)
    ax1.set_xlabel("Radio Loud")
    ax1.set_ylim(-0.6, 0.5)

    ax2.boxplot(radio_quiet_colour)
    ax2.set_xlabel("Radio Quiet")
    ax2.set_ylim(-0.6, 0.5)
    pyplot.show()


def visualise_a_bias(quasars, histograms_only=False):
    import seaborn as sns

    REDSHIFT, REDSHIFT_ERR, LOGLBOL, LOGLBOL_ERR\
        = find_intersection(quasars, 'REDSHIFT', 'LOGLBOL')

    if histograms_only:
        fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(12, 6))
        sns.set(style="white", palette="muted")
        b = sns.color_palette("muted")[0]
        sns.distplot(REDSHIFT, bins=50, kde=True, kde_kws={"shade": True},
                     color=b, ax=ax1)
        ax1.set_xlabel("Redshift")
        ax1.set_ylabel("Density")

        sns.distplot(LOGLBOL, bins=50, kde=True, kde_kws={"shade": True},
                     color=b, ax=ax2)
        ax2.set_xlabel("Log Lbol")
        ax2.set_ylabel("Density")
    else:
        # Since we use the first quartile of the errors as a boundary,
        # our new subset will contain 25% of the points in our original set,
        # only the ones which have a small error.
        first_quartile = scipy.stats.mstats.mquantiles(LOGLBOL_ERR, 0.05)

        LOGLBOL_small_error = []
        REDSHIFT_small_loglbol_error = []

        for i in xrange(len(LOGLBOL)):
            if LOGLBOL_ERR[i] < first_quartile:
                LOGLBOL_small_error.append(LOGLBOL[i])
                REDSHIFT_small_loglbol_error.append(REDSHIFT[i])

        # We do the same with the quasars with the largest errors:
        last_quartile = scipy.stats.mstats.mquantiles(LOGLBOL_ERR, 0.95)

        LOGLBOL_large_error = []
        REDSHIFT_large_loglbol_error = []

        for i in xrange(len(LOGLBOL)):
            if LOGLBOL_ERR[i] > last_quartile:
                LOGLBOL_large_error.append(LOGLBOL[i])
                REDSHIFT_large_loglbol_error.append(REDSHIFT[i])

        # Convert the lists to numpy arrays
        LOGLBOL = numpy.array(LOGLBOL)
        REDSHIFT = numpy.array(REDSHIFT)
        LOGLBOL_small_error = numpy.array(LOGLBOL_small_error)
        REDSHIFT_small_loglbol_error =\
            numpy.array(REDSHIFT_small_loglbol_error)
        LOGLBOL_large_error = numpy.array(LOGLBOL_large_error)
        REDSHIFT_large_loglbol_error =\
            numpy.array(REDSHIFT_large_loglbol_error)

        xm = numpy.mean(REDSHIFT)
        ym = numpy.mean(LOGLBOL)
        x2m = numpy.mean(REDSHIFT**2.)
        xym = numpy.mean(LOGLBOL*REDSHIFT)
        b = (xym - xm*ym)/(x2m - xm**2.)
        a = ym - b*xm

        xm = numpy.mean(REDSHIFT_small_loglbol_error)
        ym = numpy.mean(LOGLBOL_small_error)
        x2m = numpy.mean(REDSHIFT_small_loglbol_error**2.)
        xym = numpy.mean(LOGLBOL_small_error*REDSHIFT_small_loglbol_error)
        b_err = (xym - xm*ym)/(x2m - xm**2.)
        a_err = ym - b_err*xm

        xm = numpy.mean(REDSHIFT_large_loglbol_error)
        ym = numpy.mean(LOGLBOL_large_error)
        x2m = numpy.mean(REDSHIFT_large_loglbol_error**2.)
        xym = numpy.mean(LOGLBOL_large_error*REDSHIFT_large_loglbol_error)
        b_err2 = (xym - xm*ym)/(x2m - xm**2.)
        a_err2 = ym - b_err2*xm

        fig, ax = pyplot.subplots(1, figsize=(8, 8))
        c1 = '#FEE090'
        c2 = '#525252'
        c3 = '#BF812D'
        c4 = '#8C510A'
        c5 = '#35978F'
        c6 = '#01665E'
        ax.plot(REDSHIFT, LOGLBOL, "o", color=c1,
                label='all available data points')
        ax.plot(REDSHIFT_small_loglbol_error, LOGLBOL_small_error,
                "o", color=c3, label='data with small errors')
        ax.plot(REDSHIFT_large_loglbol_error, LOGLBOL_large_error,
                "o", color=c5, label='data with large errors')

        ax.plot(REDSHIFT, a+b*REDSHIFT, lw=2, color=c2,
                label='linear reg all data')
        ax.plot(REDSHIFT_small_loglbol_error,
                a_err+b_err*REDSHIFT_small_loglbol_error, color=c4, lw=2,
                label='linear reg small errors')
        ax.plot(REDSHIFT_large_loglbol_error,
                a_err2+b_err2*REDSHIFT_large_loglbol_error, lw=2, color=c6,
                label='linear reg large errors')

        pyplot.xlabel('Redshift')
        pyplot.ylabel('Log Lbol')
        pyplot.title('Correlation between redshift and bolometric luminosity')
        pyplot.legend(loc=2)
        pyplot.show()

        (cor, pval) = scipy.stats.pearsonr(REDSHIFT, LOGLBOL)
        (cor2, pval2) = scipy.stats.pearsonr(REDSHIFT_small_loglbol_error,
                                             LOGLBOL_small_error)

        print "Correlation for the total data set: ", cor  # , pval
        print "Correlation for the subset with small errors: ", cor2  # , pval2
