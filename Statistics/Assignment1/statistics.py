"""
File: statistics.py
Authors: Timo L. R. Halbesma <timo.halbesma@student.uva.nl>
         Sarah Brands <saaaaaaaar@hotmail.com>
         Geert Raaijmakers <raaymakers8@hotmail.com>

Version: 1.0 (Final Draft)
Date created: Fri Nov 13, 2015 12:07 pm
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

import numpy
import pandas
import scipy
from scipy import stats
import matplotlib
from matplotlib import pyplot

# from parse_dataset import parse_dataset
from find_intersection import find_intersection


def do_linear_regression(x_data, x_data_err, y_data, y_data_err,
                         number_quasars):
    """ Calculate and plot linear regression """

    x_data = numpy.array(x_data[0:number_quasars])
    x_data_err = x_data_err[0:number_quasars]
    y_data = numpy.array(y_data[0:number_quasars])
    y_data_err = y_data_err[0:number_quasars]
    func = lambda x, a, b: a*x + b

    r, pcov = scipy.optimize.curve_fit(func, x_data, y_data,
                                       sigma=y_data_err, p0=(1, 1))

    fig, ax = pyplot.subplots(1, 1, figsize=(10, 6))
    ax.errorbar(x_data, y_data, xerr=x_data_err, yerr=y_data_err,
                ls='none', color='grey')
    ax.plot(x_data, r[0]*x_data + r[1], color='red')
    ax.set_xlabel('Blackhole Mass (Logarithmic and in solar mass units)',
                  fontsize=15)
    ax.set_ylabel('Luminosity (Logarithmic in ergs/s)', fontsize=15)
    ax.set_title('Linear Regression Blackhole Mass-Luminosity Relation',
                 fontsize=17)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.locator_params(nbins=5)
    pyplot.show()

    err_a = numpy.sqrt(pcov[0, 0])
    err_b = numpy.sqrt(pcov[1, 1])

    print "a={0:.2f}, and b={1:.2f}".format(r[0], r[1]),
    print "Standard deviation of a, s_a={0:.2f}, and of b, s_b={1:.2f}."\
        .format(err_a, err_b)


def bootstrap(x_data, y_data, y_data_err, number_quasars):
    """ Bootstrapping method """

    params = []

    data = numpy.transpose([x_data, y_data, y_data_err])
    n_boot = int(number_quasars * (numpy.log(number_quasars))**2)
    idx = numpy.random.randint(0, number_quasars, (n_boot, number_quasars))
    samples = data[idx]

    func = lambda x, a, b: x*a+b

    for i in range(n_boot):
        r, pcov = scipy.optimize.curve_fit(
            func, samples[i][:, 0], samples[i][:, 1],
            sigma=samples[i][:, 2], p0=(1, 1))

        params.append(r.tolist())

    params = numpy.array(params)

    f, (ax1, ax2) = pyplot.subplots(1, 2, sharey=True, figsize=(10, 6))

    ax1.hist(params[:, 0], histtype='step')
    ax1.set_xlabel('Values for parameter a', fontsize=15)
    ax1.set_ylabel('Frequency', fontsize=15)
    ax2.hist(params[:, 1], histtype='step')
    ax2.set_xlabel('Values for parameter b', fontsize=15)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    f.suptitle('Bootstrapping Linear Regression Parameters', fontsize=17)

    pyplot.show()

    stata = numpy.sort(params[:, 0])
    statb = numpy.sort(params[:, 1])
    low_a, high_a = (stata[int((0.318/2.0)*n_boot)],
                     stata[int((1-0.318/2.0)*n_boot)])
    low_b, high_b = (statb[int((0.318/2.0)*n_boot)],
                     statb[int((1-0.318/2.0)*n_boot)])

    mean_a = numpy.mean(params[:, 0])
    mean_b = numpy.mean(params[:, 1])

    print "Mean value of a is {0:.2f}, and mean value of b is {1:.2f}"\
        .format(mean_a, mean_b)
    print "The 68.2 percent confidence interval for a is [{0:.2f}, {1:.2f}]."\
        .format(low_a, high_a)
    print "The 68.2 percent confidence interval for b is [{0:.2f}, {1:.2f}]."\
        .format(low_b, high_b)


def give_correlation_coefficient(quasars, number_quasars=None):

    if number_quasars is None:
        number_quasars = len(quasars)

    correlation_coefficient = []
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

    for i in xrange(ndims):
        for j in xrange(ndims):
            correlation = []

            for k, item in enumerate(rand_data):
                if rand_data[k, i] and rand_data[k, j] is not None:
                    correlation.append([rand_data[k, i], rand_data[k, j]])
            correlation = numpy.array(correlation)

            (cor, pval) = scipy.stats.pearsonr(correlation[:, 0],
                                               correlation[:, 1])
            correlation_coefficient.append(cor)

    correlation_coefficient = numpy.array(
        correlation_coefficient).reshape(-1, 4)
    correlation_coefficient = numpy.around(correlation_coefficient, decimals=2)
    names_list = ["Cont 510 nm", "Cont 300 nm", "Hb narrow line", "Mg II line"]
    print "The correlation coefficients are:"
    print pandas.DataFrame(correlation_coefficient, names_list, names_list)


def spearmanr_sarah(x, y):
    """ Implementation of Spearman Rho as Sarah has Scipy 0.13 which
        does not yet contain spearmanr method """
    n = 1.0 * len(x)

    x_rank = scipy.stats.rankdata(x)
    y_rank = scipy.stats.rankdata(y)

    d = x_rank-y_rank
    rho = 1. - (6.*sum(d**2))/(n*(n**2 - 1))

    return rho


def pearsonr_sarah(x, y):
    """ Implementation of Pearson's r as Sarah has Scipy 0.13 which
        does not yet contain pearsonr method """
    n = 1.0 * len(x)

    mean_x = numpy.mean(x)
    mean_y = numpy.mean(y)
    stdev_x = numpy.sqrt(numpy.var(x, ddof=1))
    stdev_y = numpy.sqrt(numpy.var(y, ddof=1))

    x_part = []
    y_part = []

    for i in xrange(int(n)):
        x_part.append((x[i] - mean_x)/stdev_x)
        y_part.append((y[i] - mean_y)/stdev_y)

    x_part = numpy.array(x_part)
    y_part = numpy.array(y_part)

    r = (1./(n - 1.))*sum(x_part*y_part)

    return r


def linear_regression(x_data, y_data):
    """ Implementation of linear regression by Sarah.
        FIXME: merge with Geert's do_linear_regression method """

    import seaborn as sns

    x_data = numpy.array(x_data)
    y_data = numpy.array(y_data)

    c5 = '#35978F'
    c6 = '#01665E'

    xm = numpy.mean(x_data)
    ym = numpy.mean(y_data)
    x2m = numpy.mean(x_data**2.)
    xym = numpy.mean(y_data*x_data)
    b = (xym - xm*ym)/(x2m - xm**2.)
    a = ym - b*xm

    pyplot.plot(x_data, y_data, "o", color=c5, label='data points')
    pyplot.plot(x_data, a+b*x_data, lw=2, color=c6, label='linear regression')
    pyplot.title('Correlatioin redshift Line-luminosity')
    pyplot.show()

    return "y = {0:.2f} + {1:.2f}*x".format(a, b)


def correlate_redshift_emission_line(quasars, line, subset=False):
    """ Plot correlation between redshift and emission line strength.
        Provide Spearman's rho, pearson's R and linear regression. """

    # Get datapoints for which the redshift and line luminosity is known.
    # The error array of the redshift will be empty.
    REDSHIFT, REDSHIFT_ERR, LINE, LINE_ERR\
        = find_intersection(quasars, 'REDSHIFT', line)

    if subset:
        REDSHIFT_range = []
        LINE_range = []

        for i in xrange(len(REDSHIFT)):
            if REDSHIFT[i] > 1.5 and REDSHIFT[i] < 2.2:
                REDSHIFT_range.append(REDSHIFT[i])
                LINE_range.append(LINE[i])

        REDSHIFT = REDSHIFT_range
        LINE = LINE_range

    # Then look at the correlation coefficients and plot the linear regression
    print
    print "Spearman's Rho\nSelf-written method:\t{0}\nScipy method:\t\t{1}"\
        .format(spearmanr_sarah(REDSHIFT, LINE),
                scipy.stats.spearmanr(REDSHIFT, LINE)[0]),
    print
    print "Spearman's r\nSelf-written method:\t{0}\nScipy method:\t\t{1}"\
        .format(pearsonr_sarah(REDSHIFT, LINE),
                scipy.stats.pearsonr(REDSHIFT, LINE)[0])
    print
    print "Linear regression\nSarah: {0}\nGeert{1}"\
        .format(linear_regression(REDSHIFT, LINE), "Not implemented")


def ks_test(radio_loud_colour, radio_quiet_colour):

    print "Mean color of radio quiet quasars:", numpy.mean(radio_quiet_colour)
    print "Mean color of radio loud quasars:", numpy.mean(radio_loud_colour)

    (D_value, p_value) = scipy.stats.ks_2samp(radio_quiet_colour,
                                              radio_loud_colour)

    print "The K-S test gives us a D-value of {0}".format(D_value),
    print "corresponding to a p-value of {0}.".format(p_value)

    certain = 1.0 - p_value

    print "This means that with {0} certainty we can say".format(certain),
    print "that the two samples are drawn from a different distribution."


def t_statistic(x, y):
    # Means, variance and n of each sample is calculated
    mean_x = numpy.mean(x)
    mean_y = numpy.mean(y)
    vr_x = numpy.var(x, ddof=1)
    vr_y = numpy.var(y, ddof=1)
    n_x = float(len(x))
    n_y = float(len(y))

    # denominator of the t-value is caculated:
    standard_error_diff_of_means = numpy.sqrt(vr_x/n_x + vr_y/n_y)
    # numerator of the t-value is calculated:
    difference_of_means = mean_x - mean_y
    # to make sure the t-value is always positive:
    abs_dom = abs(difference_of_means)

    # The t-value is calculated
    t = abs_dom / standard_error_diff_of_means

    # We now calculate the degrees of freedom, which is needed to calculate
    # the p-value corresponding to the calculated t-value.
    sn_x = vr_x/n_x
    sn_y = vr_y/n_y
    num = (sn_x + sn_y)**2
    denom = (sn_x**2 / (n_x - 1.)) + (sn_y**2 / (n_y - 1.))
    dof = int(num/denom)

    return t, dof


def print_t_statistic(radio_quiet_colour, radio_loud_colour):

    # t-statistic for mean
    (tvalue, ddof) = t_statistic(radio_quiet_colour, radio_loud_colour)

    print "The t-statistic is:", tvalue
    print "Degrees of freedom:", ddof

    # two-sided pvalue for a t-statistic of tvalue with ddof degrees of freedom
    pval = stats.t.sf(numpy.abs(tvalue), ddof)*2

    print "The corresponding p-value is:", pval
    certain = 1.0 - pval

    print "We can thus say that the chance that the difference in color",
    print "between radio loud and radio quiet quasars is significant is {0}."\
        .format(certain)
