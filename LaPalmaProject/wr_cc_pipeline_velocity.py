#!/usr/bin/env python
"""
    wr_cc_pipeline_stable.py:

    pipeline for the cross correlation of [WR] CSPNe spectra

    Author: Ziggy Pleunis (ziggypleunis@gmail.com)
    Collaborator: Timo Halbesma (timo.halbesma@student.uva.nl)
    Version: 2015/12/23 post 14h42
"""

import glob
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import astropy.io.fits as pf
try:
    import pyraf
    HAS_PYRAF = True
except ImportError:
    HAS_PYRAF = False

VERBOSE = True
SAVEPLOTS = False
SHOWPLOTS = False
SAVEFITS = False

# Rest wavelengths of specific stellar elements.
# Can be obtained from NIST
stellar_lines = ['C III', 'He II', 'He II', 'C IV', 'C III',
                 'C IV', 'C IV', 'He II', 'C III', 'O V']
stellar_lines_wav = [4650.25, 4685.804, 5411.521, 5471, 5696.92,
                     5801, 5812, 6564, 6730, 6740]

# Rest wavelenghts of specific nebular elements
# Meinel,A.B., Aveni,A.F., Stockton,M.W.: 1975, Catalog of Emission Lines
#     in Astrophysical Objects, Optical Sciences Center (Univ. of Arizona),
#     Technical Report, 27, 2nd edition.
#
# Acker,A., Koppen,J., Stenholm,B., Jasniewicz,G. : 1989,
#     Astron.Astrophys.Sup.Ser., 80, 201.
nebular_lines = [r'H$\beta$', '[O III]', '[O III]', '[O I]', 'He I',
                 '[N II]', r'H$\alpha$', '[N II]', '[S II]', '[S II]']
nebular_lines_wav = [4861.332, 4958.9, 5006.8, 5577.4, 5875.621,
                     6548.1, 6562.852, 6583.6, 6717.0, 6731.3]

# from examaning plt.show()
ngc40_guess = [[4860.4, 4861.2], [4958.0, 4959.0], [5005.9, 5006.9],
               [5577.5], [5875.6], [6546.8, 6548.0], [6561.6, 6562.7],
               [6582.2, 6583.4], [6715.2, 6716.4], [6729.6, 6730.8]]


def nfd2mjd(date_str):
    """ Calculate MJD for date string, format: <YYYY-MM-DDTHH:MM:SS>"""

    date, time = date_str.split('T')
    year, month, day = map(float, date.split('-'))
    hour, minute, sec = map(float, time.split(':'))

    JD = 367 * year - int(7 * (year + int((month + 9) / 12)) / 4) - int(3 * (int((year + (month - 9) / 7) / 100) + 1) / 4) + \
        int(275 * month / 9) + day + 1721028.5 + (hour + minute / 60 + sec / 3600) / 24
    MJD = JD - 2400000.5

    return MJD


def log_to_wav(start, step, left, right):
    """
    Convert a pixel log scale to a wavelength scale (for plotting).

    start -- wavelength in angstrom of first pixel
    step -- wavelength resolution in angstrom
    left -- left pixel of pixel window you want to use
    right -- right pixel of pixel window you want to use
    """

    # exp because the data is in log scale
    wav = np.exp(np.arange(start + left * step, start + right * step, step))

    return wav


def find_xvalues(edges, start, step):
    """
    Given *round* numbers for x locations, give existing x locations.

    edges -- list of lists with [xmin, xmax] values in angstrom
    start -- wavelength in angstrom of first pixel
    step -- wavelength resolution in angstrom
    """

    loc_edge = []
    for edge in edges:
        loc = []
        # use start and step information to find value near preferred value
        loc.append(int(np.floor((np.log(edge[0]) - start) / step)))
        loc.append(int(np.floor((np.log(edge[1]) - start) / step)))
        loc_edge.append(loc)

    return loc_edge


def obtain_fits_paths (data_dir):
    """
    Return a list of all fits files to analyse.

    data_dir -- string with the path to the data directory.
    """

    # Linear and log are HermesDRS output files.
    # Linear is not barycentric corrected; log is
    # linear = '*_HRF_OBJ_ext_CosmicsRemoved_wavelength_merged_c.fits'
    log = '*_HRF_OBJ_ext_CosmicsRemoved_log_merged_c.fits'

    fnames = []
    for name in glob.glob(data_dir+'/'+log):
        fnames.append(name)

    NGC246 = False
    if NGC246:
        for name in glob.glob(data_dir+'/ngc246/'+log):
            fnames.append(name)

    if not fnames:
        print 'Warning: no inputfiles found!'

    return fnames


def generate_LaTeX_observationlog(observations):
    """
    Generate LaTeX-ready observation tabular.

    observations -- list containing all HermesObservation instances
    """

    print 'Object & UNSEQ & Date & Start time & End time & ',
    print 'Exposure Time (s) & SNR (blue) & SNR (red) \\\\'
    for obs in observations:
        print obs.obj_name.upper().replace(' ', '~') + ' &',
        print obs.unseq + ' &',
        date = obs.header['DATE-OBS'][0:10]
        print date + ' &',
        begintime = obs.header['DATE-OBS'][11:19]
        print begintime + ' &',
        endtime = obs.header['DATE-END'][11:19]
        print endtime + ' &',
        print '{0:.0f} &'.format(obs.exp_time),
        print '{0:.1f} &'.format(obs.snr_blue),
        print '{0:.1f} \\\\'.format(obs.snr_red)


def velocity(line, restline, vrad):
    """
    Calculate the velocity of a spectral line.

    line -- observed wavelength of line
    restline -- rest wavelength of line
    """

    clight = 299792.458  # km/s

    return clight * (line / restline - 1) - vrad


def line_velocity(wav, spectrum, vrad, unseq):
    """
    Calculate the velocities of given emission lines.

    spectrum -- a spectrum
    vrad -- radial velocity of the star in km/s
    This also needs a list emission lines, but here that list is global.
    """

    print_LaTeX = False

    # print LaTeX ready table
    if print_LaTeX:
        print 'Line & $\lambda_0$ & $\lambda_\\text{obs}$ & Intensity & '+\
               'Velocity (max) & Velocity (Gaussian fit) \\\\'
        print ' & (\AA) & (\AA) & & (km/s) & (km/s) \\\\'

    plot_number = 0
    for line_nr in range(len(nebular_lines_wav)):
        element = nebular_lines[line_nr]
        rest_wav = nebular_lines_wav[line_nr]

        obs_wavs = ngc40_guess[line_nr]

        for obs_wav in obs_wavs:
            plot_number += 1
            # fit Gaussians
            line = np.intersect1d(np.where(wav >= obs_wav-0.3),
                                  np.where(wav <= obs_wav+0.3))

            # Obtain (binned!!) signal to noise ratio to use as error
            nbins = 4  # because very narrow line
            dx = (line.max() - line.min()) / nbins  # int because indices
            bins = [(line.min() + i*dx) for i in xrange(nbins)]
            snr_binned = np.zeros(nbins)
            noise_binned = np.zeros(nbins)
            for i in xrange(nbins):
                xmin = line.min() + i*dx
                xmax = line.min() + (i+1)*dx
                snr_binned[i] = (np.max(spectrum[xmin:xmax]) /
                                 np.std(spectrum[xmin:xmax]))
                noise_binned[i] = np.std(spectrum[xmin:xmax])

            # NB snr_binned has different length than subset of wav, norm_flux we use!
            # Assume that the SNR in the bin is valid over the entire bin interval
            snr = np.zeros(len(line))
            noise = np.zeros(len(line))
            for i in xrange(len(line)):
                index = np.where(line[i] <= bins)
                if len(index[0]) is 0:
                    index = 3
                else:
                    index = index[0][0]
                snr[i] = snr_binned[index]
                noise[i] = noise_binned[index]

            fit_values = fit_gaussian(wav[line], spectrum[line], noise,
                                      unseq, plot_number, 0.1, False, False, False)

            if not fit_values:
                gauss_velocity = np.nan
            else:
                gauss_velocity = velocity(fit_values["x"][1], rest_wav, vrad)

            # And calculate using estimate of peak
            left = wav > obs_wav - 0.3
            right = wav < obs_wav + 0.3
            both = np.logical_and(left, right)
            peak = np.max(spectrum[both])
            peak_index = np.where(spectrum == peak)
            shift = wav[peak_index][0]


            # brackets so LaTeX doesn't think '[' belongs to '\\'
            if print_LaTeX:
                print '{' + element + '} &',
                print '{0:.1f} &'.format(rest_wav),
                print '{0:.1f} &'.format(obs_wav),
                print '{0:.2f} &'.format(peak),
                print '${0:.2f}$ &'.format(velocity(shift, rest_wav, vrad)),
                print '${0:.2f}$ \\\\'.format(gauss_velocity)

            if SHOWPLOTS:
                plt.axvline(shift, c='r')

    if not SHOWPLOTS:
        return

    plt.clf()
    plt.plot(wav, spectrum)

    plt.xlim([4000, 7000])
    plt.ylim([0, 30])

    plt.show()


def sum_spectra(observations, vrad, plot=False):
    """
    Make a weighted summed spectrum of all the observations.

    observations -- list containing all HermesObservation instances
    vrad -- radial velocity of the star in km/s.
    """

    # the 167297 is *hardcoded* the length of the shortest wavelength,
    # throwing away the last part is fine, since we are not even plotting it

    first = True
    var_sum = 0

    for obs in observations:
        var = (obs.var_blue + obs.var_red) / 2.
        if first is True:
            sum_flux = np.copy(obs.norm_flux[:167297]) / var
            first = False
        else:
            sum_flux += obs.norm_flux[:167297] / var
        var_sum += 1 / var

    sum_flux = sum_flux / var_sum

    line_velocity(obs.wav[:167297], sum_flux, vrad, obs.unseq)

    if (not SHOWPLOTS and not SAVEPLOTS) and not plot:
        return

    plt.figure(figsize=(18, 10))
    plt.rcParams.update({'font.size': 20})

    plt.plot(obs.wav[:167297], sum_flux, c='k')

    plt.xlabel(r'Wavelength ($\AA$)')
    plt.ylabel('Normalised flux')
    # plt.title('NGC 40 - weighted sum of {0} spectra'
    #          .format(len(observations)))

    plt.xlim([4000, 7000])
    plt.ylim([0, np.max(sum_flux[obs.wav[:167297] > 4000])])

    plt.minorticks_on()

    # print names of the lines
#    for i in range(len(stellar_lines)):
#        #plt.axvline(stellar_lines_wav[i], color='red', linestyle='--')
#        x = find_xvalues([[stellar_lines_wav[i]-10, stellar_lines_wav[i]+10]],
#                         obs.start, obs.step)
#        plt.text(stellar_lines_wav[i], np.max(sum_flux[x[0][0]:x[0][1]])+3,
#                 stellar_lines[i], fontsize=12,
#                 horizontalalignment='center', verticalalignment='center')
#    for j in range(len(nebular_lines)):
#        #plt.axvline(nebular_lines_wav[j], color='blue', linestyle='--')
#        x = find_xvalues([[nebular_lines_wav[j]-10, nebular_lines_wav[j]+10]],
#                         obs.start, obs.step)
#        plt.text(nebular_lines_wav[j], np.max(sum_flux[x[0][0]:x[0][1]])+3,
#                 nebular_lines[j], fontsize=12,
#                 horizontalalignment='center', verticalalignment='center')

    if SHOWPLOTS or plot:
        plt.show()
    if SAVEPLOTS:
        plt.savefig('out/spectrum.png', dpi=300)


def calculate_template_2(self):
    """ Explain what T2 is """

    # TODO function that generates average template spectrum T2
    return


def gaussian(x, mu, sigma):
    ''' Normal distribution '''
    return np.exp(-1.*(x-mu)**2/(2*sigma**2)) / (sigma*np.sqrt(2*np.pi))


# https://stackoverflow.com/questions/23828226/scipy-curve-fit-does-not-seem-to-change-the-initial-parameters
def gauss(parms, x):
    A = parms[0]
    mu = parms[1]
    sigma = parms[2]
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))+1


def chisq(parms, x, y, dy):
    ymod = gauss(parms, x)
    return np.sum((y-ymod)**2/dy**2)


def fit_gaussian(wav_values, norm_flux, error, unseq, line_nr,
                 sigma_guess=7, verbose=False, plot=False, save=False):
    # p0 = [np.max(norm_flux), wav_values[np.argmax(norm_flux)], 0.05]
    # popt, pcov = scipy.optimize.curve_fit(gaussFunction, wav_values, norm_flux, p0, error)
    # curve_fit returns fit parameter values and errors
    # A_fit, mu_fit, sigma_fit = popt[0], popt[1], popt[2]
    # A_fit_stdev, mu_fit_stdev, sigma_fit_stdev = np.sqrt(np.diag(pcov))

    # Xsquared_min = calculate_chi_squared(norm_flux,
    #     gaussFunction(wav_values, A_fit, mu_fit, sigma_fit), error)

    # Here we have 3 fit parameters. Ddof is delta degrees of freedom.
    # ddof = len(norm_flux)-3
    # chisquared = scipy.stats.chi2(ddof)
    # p_value = 1. - chisquared.cdf(Xsquared_min)

    # print "The fit parameters are as follows."
    # print "mu = {0:.2f} (stdev = {1:.2f})".format(mu_fit, mu_fit_stdev)
    # print "sigma = {0:.2f} (stdev = {1:.2f})".format(sigma_fit, sigma_fit_stdev)
    # print "The p-value for this fit is {0:.2f}".format(p_value)

    parms = [np.max(norm_flux), wav_values[np.argmax(norm_flux)], sigma_guess]
    gauss_res = scipy.optimize.minimize(
        chisq, parms, args=(wav_values, norm_flux, error),
        method='Nelder-Mead')

    if gauss_res["success"] is not True:
        print "Error! The fit failed!"

    ml_gauss = gauss_res["fun"]
    dof = len(norm_flux) - 3

    gauss_pars = gauss_res["x"]

    # Sketchy calculation. This 3 should have been dof
    chi2 = scipy.stats.chi2(3)
    # ml_gaus/dof should have been just ml_gaus
    # But in that case the p-values are all exactly zero...
    p_gauss = 1.0 - chi2.cdf(ml_gauss/dof)

    if verbose:
        print "Reduced chi-squared is {0:.6f}, with\nA={1}\nmu={2}\nsigma={3}"\
            .format(ml_gauss/dof, gauss_pars[0], gauss_pars[1], gauss_pars[2])
        print "p-value for this fit is {0:.15f}".format(p_gauss)

    if plot or save:
        param_string = '\n'
        param_string += r'$\chi^2_{{\rm reduced}}${1: <3}={0:.3f}'.format(ml_gauss/dof, "")
        param_string += '\n'
        param_string += r'$A${1: <11}={0:.3f}'.format(gauss_pars[0], "")
        param_string += '\n'
        param_string += r'$\mu${1: <11}={0:.3f}'.format(gauss_pars[1], "")
        param_string += '\n'
        param_string += r'$\sigma${1: <11}={0:.3f}'.format(gauss_pars[2], "")
        param_string += '\n'
        param_string += r'p-value{1: <1}={0:.4f}'.format(p_gauss, "")
        param_string += '\n'

        plt.subplots(2, 1, figsize=(10, 8))
        gs1 = gridspec.GridSpec(3, 3)
        gs1.update(hspace=0)
        ax1 = plt.subplot(gs1[:-1,:])
        ax2 = plt.subplot(gs1[-1,:])

        lmbda = gauss_pars[1]  # To remove later.

        ax1.errorbar(wav_values-lmbda, norm_flux, yerr=error,
                     ls='', c="black", marker='o', ms=3, label="data")
        ax2.errorbar(wav_values-lmbda, (norm_flux-(gauss(gauss_pars, wav_values))),
                     yerr=error, ls='', c="red", marker='o', ms=3)

        ax1.tick_params(labelsize=15)
        ax2.tick_params(labelsize=15)

    # Build in offset because we miss the tails. NB this is only for plotting.
    if sigma_guess <= 1:
        wav_range = np.linspace(wav_values[0], wav_values[-1], 100)
    else:
        wav_range = np.linspace(wav_values[0]-42, wav_values[-1]+42, 10000)
    fit_values = gauss(gauss_pars, wav_range)

    if plot or save:
        ax1.plot(wav_range-lmbda, fit_values, c="red", lw=3, label="fit values:"+param_string)
        ax2.axhline(y=0, lw=2, ls='dashed', c="black")

        ax2.set_xlabel(r'Wavelength - {0:.2f} ($\AA$)'.format(lmbda), fontsize=18)
        ax2.set_ylabel("Residuals", fontsize=18)
        ax1.set_ylabel('Normalised Counts', fontsize=18)
        ax1.legend()
        ax1.tick_params(labelbottom='off')
        nbins = len(ax2.get_yticklabels())
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper'))

    if SHOWPLOTS or plot:
        plt.show()
    if SAVEPLOTS or save:
        plt.savefig('out/{0}_GaussianFit_Nebular-line-{1}.png'
                    .format(unseq, line_nr),
                        dpi=300, bbox_inches='tight')
        # plt.close()

    # Dealing with the nebular emission lines
    if sigma_guess < 1 and p_gauss < 0.6:
        # print "Error: fit failed"
        return None
    return gauss_res


def calculate_center_of_gravity_velocity(all_observations,
        xmin, xmax, vrad, LaTeX=False):
    """
    Calculate the center of gravity stellar emission line wavelength
    To do so, one has to find emission lines, calculate lambda_cog.
    We also fit a Gaussian profile to the emission line and then calculate
    lambda_cog. The value of lambda_cog can be turned into velocity in the
    line of sight.

    Returns list of modified julian dates, list of v_los, list of v_los_gauss
    """


    mjd_list = []
    v_los_list = []
    v_los_gauss_list = []
    lambda_nod_list = []

    show_plots_of_linefinding = False
    print "\nCalculating center of gravity line velocities."

    print_header = True
    for obs in all_observations:
        if show_plots_of_linefinding:
            plt.rcParams.update({'font.size': 20})
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(18, 10))

        # plot of the flux
        # ax1.plot(obs.wav, obs.flux, c='red', label="obs.flux")
        # ax1.set_ylim([0, np.max(obs.flux[obs.wav > xmin])])
        # ax1.set_ylabel('Counts')
        # ax1.legend()

        # Design decision (after inspecting plot of both
        # obs.flux and obs.norm_flux: continue with normalised flux.

        # Now find stellar emission lines
        # i) cut away wavelengths below 4500 because no look nice, visually selected :(
        demand_i = np.where(obs.wav > 4500)
        # ii) find counts 2 * mean counts
        demand_ii = np.where(obs.norm_flux > 2*np.mean(obs.norm_flux))
        # iii) neglect thin (nebular) lines, visually selected :(
        demand_iii = np.where(obs.wav < 6500)

        # intersection of three arrays
        from functools import reduce
        stellar_line_index = reduce(np.intersect1d,
            (demand_i, demand_ii, demand_iii))

        # Now how to find individual lines in array with all star em. lines?
        if show_plots_of_linefinding:
            ax1.scatter(obs.wav[stellar_line_index],
                        obs.norm_flux[stellar_line_index],
                        c='k', label="stellar emission lines")
            ax1.set_ylim([1, np.max(obs.norm_flux[obs.wav > xmin])])
            ax1.set_ylabel('Counts')
            ax1.legend()

        # We find groups of consecitives indices of at least length 42.
        def consecutive(data, stepsize=1, max_gap=1):
            return np.split(data,
                np.where(np.diff(data) != stepsize)[0]+max_gap)

        consecutives = consecutive(stellar_line_index)
        individual_lines = []
        for candidate in consecutives:
            if len(candidate) > 42:  # Arbitrary
                individual_lines.append(candidate)

        if show_plots_of_linefinding:
            colourpicker = {0: 'red', 1: 'orange', 2: 'yellow', 3: 'green',
                            4: 'blue', 5: 'purple', 6: 'pink'}
            for i, line in enumerate(individual_lines):
                ax2.scatter(obs.wav[line.min():line.max()],
                            obs.norm_flux[line.min():line.max()],
                            c=colourpicker.get(i, 'black'),
                            label="line {0}".format(i))
            ax2.set_ylim([1, np.max(obs.norm_flux[obs.wav > xmin])])
            ax2.set_ylabel('Counts')
            ax2.legend()

            # sharex=True, thus, only set ax2 lim and xlabel.
            ax2.set_xlabel(r'Wavelength ($\AA$)')
            ax2.set_xlim([xmin, xmax])

            plt.minorticks_on()
            plt.show()

        if LaTeX and print_header:
            print '& & & & \\multicolumn{2}{c}{Data} & \\multicolumn{2}{c}{Fit} \\\\'
            print 'Object & MJD & line name & $\\lambda_0$ (\\AA) & $\\lambda_{\\rm COG}$ (\\AA) & $v_{\\rm LOS}$ (km/s) & $\\lambda_{\\rm COG}$ (\\AA) & $v_{\\rm LOS}$ (km/s) \\\\'
            print '\\hline'
            print_header = False

        if LaTeX:
            print 'HD~826',
        for line_nr, line in enumerate(individual_lines):
            # An issue might be that the tails of the lines have fallen off.
            # See plot, and see demand_ii. Fix this by adding offset.

            # Add offset to encount for tails of line
            offset = 500 # arbitrary
            line = np.arange(line.min()-offset, line.max()+offset)

            # Obtain (binned!!) signal to noise ratio to use as error
            nbins = 42
            dx = (line.max() - line.min()) / nbins  # int because indices
            bins = [(line.min() + i*dx) for i in xrange(nbins)]
            snr_binned = np.zeros(nbins)
            noise_binned = np.zeros(nbins)
            for i in xrange(nbins):
                xmin = line.min() + i*dx
                xmax = line.min() + (i+1)*dx
                snr_binned[i] = (np.max(obs.norm_flux[xmin:xmax]) /
                                 np.std(obs.norm_flux[xmin:xmax]))
                noise_binned[i] = np.std(obs.norm_flux[xmin:xmax])

            # NB snr_binned has different length than subset of wav, norm_flux we use!
            # Assume that the SNR in the bin is valid over the entire bin interval
            snr = np.zeros(len(line)-1)
            noise = np.zeros(len(line)-1)
            for i in xrange(len(line)-1):
                index = np.where(line[i] <= bins)
                if len(index[0]) is 0:
                    index = 41
                else:
                    index = index[0][0] - 1
                snr[i] = snr_binned[index]
                noise[i] = noise_binned[index]

            # Now snr has same length as the line, and the snr values can
            # be used as error for the fit and chi-squared calculation.
            # NB len(line) = len of subset of wav + 1

            # plt.figure()
            # plt.plot(line, snr)
            # plt.scatter(bins, snr_binned)
            # plt.show()

            # The idea was to calculate the center of gravity for
            # the Gaussian fit, but the fits have high reduced chi^2 compared
            # to the number of fit parameters, and have low p-values.
            # Therefore we do not use the Gaussian fit, but the data.

            # Fit a Gaussian
            gauss_res = fit_gaussian(obs.wav[line.min():line.max()],
                    obs.norm_flux[line.min():line.max()], noise,
                    obs.unseq, line_nr)

            x_values = obs.wav[line.min():line.max()]
            fit_values = gauss(gauss_res["x"], x_values)

            # Calculate center of gravity (units of \AA)
            lambda_cog_gauss = (np.sum(x_values * (1 - fit_values)) /
                np.sum((1-fit_values)))
            lambda_cog = (np.sum(obs.wav[line.min():line.max()] *
                (1-obs.norm_flux[line.min():line.max()])) /
                np.sum((1-obs.norm_flux[line.min():line.max()])))

            # Convert to velocity center of gravity
            # Find nearest value in stellar_lines_wav, where lambda_0 is given
            lambda_0_values = np.array(stellar_lines_wav)
            index = np.abs(stellar_lines_wav - lambda_cog).argmin()
            lambda_0 = lambda_0_values.flat[index]
            line_name = stellar_lines[index]

            speed_of_light = 299792.458  # km/s
            v_los = speed_of_light * (lambda_0 - lambda_cog)/lambda_0
            v_los_gauss = speed_of_light * (lambda_0 - lambda_cog_gauss)/lambda_0
            # Correct for radial velocity
            v_los -= vrad
            v_los_gauss -= vrad

            mjd = nfd2mjd(obs.header['DATE-OBS'])

            if LaTeX:
                print ' & {0:.2f} & {1} & '.format(mjd, line_name),
                print '{0:.2f} & '.format(lambda_0),
                print '{0:.2f} & {1:.2f} & '.format(lambda_cog, v_los),
                print '{0:.2f} & {1:.2f} '.format(lambda_cog_gauss, v_los_gauss),
                print '\\\\'.format(v_los)

            mjd_list.append(mjd)
            v_los_list.append(v_los)
            v_los_gauss_list.append(v_los_gauss)
            lambda_nod_list.append(lambda_0)

        # Another issue might be that I have assumed we can use the normalised
        # continuum, so I_cont in the formula is equal to one.
        # Furthermore, we sum; (not) integrate. Here, dlambda != infinitesimal


    return mjd_list, v_los_list, v_los_gauss_list, lambda_nod_list


def run_iraf(self):
    """ Depends on pyraf """

    # TODO: see if we can run IRAF functions from Python
    #   -> xcsao and fxcor for cross-correlation
    #   -> check whether cross-correlation works as expected with our
    #      *gapped* spectra
    #   -> shift spectra with IRAF our with own function?
    return


def create_mjd_plot((mjd_list, v_los_list, v_los_gauss_list, lambda_nod_list)):

    plt.figure(figsize=(18, 10))
    plt.rcParams.update({'font.size': 20})

    mjd = np.array(mjd_list)
    v_los = np.array(v_los_list)
    v_los_gauss = np.array(v_los_gauss_list)
    lambda_nod = np.array(lambda_nod_list)

    line1 = np.where(lambda_nod == 4650.25)
    line2 = np.where(lambda_nod == 5696.92)
    line3 = np.where(lambda_nod == 5812.0)

    plt.errorbar(mjd[line1]-57000, v_los[line1], yerr=v_los[line1]*0.1,
                 ls='', marker='o', ms=3, c='red', label=r'4650.25 $\AA$')
    plt.errorbar(mjd[line2]-57000, v_los[line2], yerr=v_los[line2]*0.1,
                 ls='', marker='o', ms=3, c='green', label=r'5696.92 $\AA$')
    plt.errorbar(mjd[line3]-57000, v_los[line3], yerr=v_los[line3]*0.1,
                 ls='', marker='o', ms=3, c='blue', label=r'5812.0 $\AA$')

    plt.xlabel("MJD - 57000", fontsize=18)
    plt.ylabel(r'$v_{\rm LOS}$ (km/s)', fontsize=22)

    mean1 = np.mean(v_los[line1])
    std1 = np.std(v_los[line1])
    mean2 = np.mean(v_los[line2])
    std2 = np.std(v_los[line2])
    mean3 = np.mean(v_los[line3])
    std3 = np.std(v_los[line3])

    plt.axhline(y=mean1, linewidth=2, linestyle='dashed', c='red')
    plt.axhline(y=mean1-std1, linewidth=2, linestyle='dotted', c='red')
    plt.axhline(y=mean1+std1, linewidth=2, linestyle='dotted', c='red')
    plt.axhspan(mean1-std1, mean1+std1, facecolor='red', alpha=0.2)

    plt.axhline(y=mean2, linewidth=2, linestyle='dashed', c='green')
    plt.axhline(y=mean2-std2, linewidth=2, linestyle='dotted', c='green')
    plt.axhline(y=mean2+std2, linewidth=2, linestyle='dotted', c='green')
    plt.axhspan(mean2-std2, mean2+std2, facecolor='green', alpha=0.2)

    plt.axhline(y=mean3, linewidth=2, linestyle='dashed', c='blue')
    plt.axhline(y=mean3-std3, linewidth=2, linestyle='dotted', c='blue')
    plt.axhline(y=mean3+std3, linewidth=2, linestyle='dotted', c='blue')
    plt.axhspan(mean3-std3, mean3+std3, facecolor='red', alpha=0.2)

    plt.legend(loc=2)

    plt.savefig('out/{0}.png'.format("vlos_vs_mjd"),
                dpi=300, bbox_inches='tight')
    plt.show()




class HermesObservation(object):
    """ HERMES observation data analysis pipeline """
    def __init__(self, fname):
        """ Open fits file, return data and relevant header information. """

        fitsfile = pf.open(fname)
        self.header = fitsfile[0].header
        self.data = fitsfile[0].data
        self.unseq = fname[8:16]  # unique sequence number @ Mercator
        self.obj_name = self.header['OBJECT']
        self.exp_time = self.header['EXPTIME']
        self.start = self.header['CRVAL1']
        self.step = self.header['CDELT1']
        self.snr_blue = 0
        self.snr_red = 0
        self.var_blue = 0
        self.var_red = 0

        self.ndata = len(self.data)
        self.wav_raw = log_to_wav(self.start, self.step, 0, self.ndata)

        # get rid of NaNs
        self.flux = self.data[np.isnan(self.data) == False]
        self.wav = self.wav_raw[np.isnan(self.data) == False]

        self.masked_flux = np.zeros(len(self.wav))
        self.norm_flux_mask = np.zeros(len(self.wav))

        fit_edges = [[4500, 6800]]
        loc_edge = find_xvalues(fit_edges, self.start, self.step)
        self.normalise(4000, 7000, loc_edge, 10, show_plot=False)
# This value 10 was 5 in the previous code

        self.calculate_snr()

        if VERBOSE:
            self.print_observation_info()

    def print_observation_info(self):
        """ Verbose information """
        # print important header information (\t is for tab)
        print 'I have read your .fits file!'
        print '---'
        print 'Object:\t\t', self.obj_name
        print 'Observation ID:\t', self.unseq
        print 'Exposure time:\t', self.exp_time, 's'
        print 'Start:\t\t', self.start, u'\u212B'.encode('utf-8')  # angstrom
        print 'Step size:\t', self.step, u'\u212B'.encode('utf-8')  # angstrom
        print 'SNR (blue):\t', self.snr_blue
        print 'SNR (red):\t', self.snr_red


    def print_entire_header(self):
        """ Print header fields, values """
        for header_fieldname in self.header:
            print header_fieldname
            print self.header[header_fieldname]
            print

    def plot_hermes(self, xmin=4000, xmax=7000, regions=None, plot=1):
        """
        Plot Hermes spectrum.

        xmin = 4000 -- plot range in angstrom
        xmax = 7000 -- plot range in angstrom
        regions = None -- list of lists with regions to plot shaded
        plot = 1 -- 1: raw spectrum, 2: masked spectrum, 3: normed spectrum
        """

        if not SHOWPLOTS and not SAVEPLOTS:
            return

        plt.figure(figsize=(18, 10))
        plt.rcParams.update({'font.size': 20})
        filename = {0: 'ERROR', 1: 'flux',
                    2: 'masked_flux', 3: 'norm_flux_mask'}

        if plot == 1:
            plt.plot(self.wav, self.flux, c='k')
        elif plot == 2:
            plt.plot(self.wav, self.masked_flux, c='k')
        elif plot == 3:
            plt.plot(self.wav, self.norm_flux_mask, c='k')
        plt.xlabel(r'Wavelength ($\AA$)')
        plt.ylabel('Counts')
        # plt.title(self.obj_name + r', texp = ' + str(self.exp_time) + ' s')

        # wavelength range of Hermes: 377-900 nm, so 3770-9000 A
        plt.xlim([xmin, xmax])
        if plot == 1:
            plt.ylim([0, np.max(self.flux[self.wav > xmin])])
        if plot == 2:
            plt.ylim([0, np.max(self.masked_flux[self.wav > xmin])])
        elif plot == 3:
            plt.ylim([1, np.max(self.norm_flux[self.wav > xmin])])

        filename_regions = ''
        if regions:
            for reg in regions:
                plt.axvspan(reg[0], reg[1], alpha=0.2, color='red', lw=0)
                filename_regions = '_with_regions'

        # for i in [4550, 4755, 5290, 5620, 5990, 6220, 6450, 6825]:
        #     plt.axvline(i, c='r', lw=4)

        plt.minorticks_on()
        if SHOWPLOTS:
            plt.show()
        if SAVEPLOTS:
            plt.savefig('out/{0}_{1}{2}.png'
                        .format(self.unseq, filename.get(plot, 0),
                                filename_regions
                                ),
                        dpi=300, bbox_inches='tight')

    def calculate_snr(self, snr_step=2000, show_plot=False):
        """ Calculate the S/N ratio for a spectrum. """

        # Use bins of size snr_step. Obtain array with middle of each bin.
        bins = np.arange(snr_step/2, len(self.flux), snr_step)
        snr = []
        sigma = []
        for bin_middle in bins:
            # peak flux / RMS
            bin_lower = bin_middle - snr_step/2
            bin_upper = bin_middle + snr_step/2
            sigma.append(np.std(self.flux[bin_lower:bin_upper]))
            snr.append(np.max(self.flux[bin_lower:bin_upper]) /
                       np.std(self.flux[bin_lower:bin_upper]))

        # According to Manick et al. 2015
        # The S/N was measured as an average of four different S/N at
        # wavelengths 4220, 4755, 5290 and 5620 Ansgstrom for the blue spectra
        # and 5990, 6415, 6950 and 7370 Angstrom for the red.

        # Find index of wavelength nearest to Manick values, then obtain snr.
        bin_central_wav = self.wav[bins]
        snr_blue = 0
        snr_red = 0
        var_blue = 0
        var_red = 0
        for manick_wav in [4550, 4755, 5290, 5620, 5990, 6220, 6450, 6825]:
            nearest_wav = bin_central_wav.flat[np.abs(bin_central_wav
                                                      - manick_wav).argmin()]
            nearest_wav_index = np.where(bin_central_wav == nearest_wav)[0][0]
            var_value = sigma[nearest_wav_index] ** 2
            snr_value = snr[nearest_wav_index]
            if manick_wav < 5620:
                var_blue += var_value
                snr_blue += snr_value
            else:
                var_red += var_value
                snr_red += snr_value

        self.var_blue = var_blue / 4
        self.var_red = var_red / 4
        self.snr_blue = snr_blue / 4
        self.snr_red = snr_red / 4

        if SHOWPLOTS and show_plot:
            plt.clf()
            plt.plot(self.wav[bins], np.asarray(snr), 'o')
            plt.xlim([4000, 7000])
            plt.ylim([0, 100])
            plt.show()

    def snr_region(self, xmin, xmax):
        """ Calculate the S/N ratio for a region between xmin and xmax. """

        # bit ugly to fit in find_xvalues(), but fine this way
        edge = [[xmin, xmax]]
        pixmin, pixmax = find_xvalues(edge, self.start, self.step)[0]
        snr = np.max(self.flux[pixmin:pixmax]) / \
                np.std(self.flux[pixmin:pixmax])

        return snr

    def get_regions(self):
        """ Find spectral regions with prominent stellar emission lines """
        # TODO? find regions with prominent stellar lines automagically?

        if self.obj_name == 'ngc 40':
            # for NGC 40, prominant stellar emission lines (by eye).
            # 4610-4710 A
            # 53560-5910 A
            regions = [[4610, 4710], [5385, 5513], [5645, 5925]]
        elif self.obj_name == 'NGC 246':
            print 'Warning: NGC 246 regions should still be determined by eye!'
            regions = [[5500, 6000]]
        else:
            print 'Invalid object name --> regions = None'
            return None

        return regions

    def mask_spectrum(self, regions):
        """Make a masked spectrum, given a list of regions to include."""

        # make a list with the length of the data set, with True on the places
        # you want to keep in the masked spectrum
        self.mask = [False] * len(self.wav)
        for reg in regions:
            self.mask = np.logical_or(self.mask, np.logical_and(self.wav >
                                      reg[0], self.wav < reg[1]))

        # make new data list; False -> 0, True -> value from raw spectrum
        masked_flux = []
        for i, j in enumerate(self.mask):
            if j:
                masked_flux.append(self.flux[i])
            else:
                masked_flux.append(0.)

        self.masked_flux = np.asarray(masked_flux)

    def normalise(self, xmin, xmax, loc, fitwidth, show_plot=False):
        """Fit 1st order polynomial to spectrum regions and normalise."""

        self.norm_flux = np.copy(self.flux)

        # in here are some tricks to account for pixel/wavelength conversions
        x = range(loc[0][0] - fitwidth, loc[0][0] + fitwidth) + \
            range(loc[0][1] - fitwidth, loc[0][1] + fitwidth)
        y = self.flux[x]
        b, a = np.polyfit(x, y, 1)
        fit = a + b * self.wav

        # normalise
        self.norm_flux = self.flux / fit

        if (SHOWPLOTS or SAVEPLOTS) and show_plot:
            plt.figure(figsize=(18, 10))
            plt.rcParams.update({'font.size': 20})

            # plot the fit on top of the masked spectrum
            plt.plot(self.wav, self.flux, c='k')
            plt.xlabel(r'Wavelength ($\AA$)')
            plt.ylabel('Counts')
            # plt.title(self.obj_name + r', texp = ' + str(self.exp_time) + ' s')

            # wavelength range of Hermes: 377-900 nm, so 3770-9000 A
            plt.xlim([xmin, xmax])
            plt.ylim([0, np.max(self.flux[self.wav > xmin])])

            # plot the linear fit
            plt.plot(self.wav, fit, c='#AD1737')

            if SHOWPLOTS:
                plt.show()
            if SAVEPLOTS:
                plt.savefig('out/{0}_normalised.png'.format(self.unseq),
                            dpi=300, bbox_inches='tight')

    def normalise_mask(self, xmin, xmax, loc_edge, fit):
        """ Fit 1st order polynomial to spectrum regions and normalise. """
        if SHOWPLOTS or SHOWPLOTS:
            plt.figure(figsize=(18, 10))
            plt.rcParams.update({'font.size': 20})

            # plot the fit on top of the masked spectrum
            plt.plot(self.wav, self.masked_flux, c='k')
            plt.xlabel(r'Wavelength ($\AA$)')
            plt.ylabel('Counts')
            # plt.title(self.obj_name + r', texp = ' + str(self.exp_time) + ' s')

            # wavelength range of Hermes: 377-900 nm, so 3770-9000 A
            plt.xlim([xmin, xmax])
            plt.ylim([0, np.max(self.masked_flux[self.wav > xmin])])
        self.norm_flux_mask = np.copy(self.masked_flux)

        # in here are some tricks to account for pixel/wavelength conversions
        for loc in loc_edge:
            x = range(loc[0] - fit, loc[0] + fit) + \
                range(loc[1] - fit, loc[1] + fit)
            y = self.flux[x]
            b, a = np.polyfit(x, y, 1)
            cont_x = np.arange(loc[0], loc[1] + 1, 1.)
            cont_y = a + b * cont_x
            wav_x = log_to_wav(self.start, self.step, loc[0], loc[1])

            if SHOWPLOTS or SAVEPLOTS:
                # plot the linear fit
                plt.plot(wav_x, cont_y, c='#AD1737', lw=4)

            # normalise
            self.norm_flux_mask[loc[0]:loc[1]+1] = \
                self.masked_flux[loc[0]:loc[1]+1] / cont_y

        if SHOWPLOTS:
            plt.show()
        if SAVEPLOTS:
            plt.savefig('out/{0}_normalised_masked_fit.png'.format(self.unseq),
                        dpi=300, bbox_inches='tight')

    def pipeline(self, xmin, xmax, fit, plot=False):
        """Run the whole analysis pipeline."""

        self.plot_hermes(xmin, xmax)

        regions = self.get_regions()
        if not regions:
            return
        loc_edge = find_xvalues(regions, self.start, self.step)

        self.plot_hermes(xmin, xmax, regions=regions)
        self.mask_spectrum(regions)
        self.plot_hermes(xmin, xmax, plot=2)
        self.normalise_mask(xmin, xmax, loc_edge, fit)
        self.plot_hermes(xmin, xmax, plot=3)


def main():
    """
    Read all fits files in directory, run pipeline
    """

    vrad = -20.5  # km/s -> NGC 40
    xmin = 4000
    xmax = 7000

    all_observations = []
    for fname in obtain_fits_paths('../Data'):
        obs = HermesObservation(fname)
        # if obs.unseq == '00671212':
        obs.pipeline(xmin, xmax, 5)
        all_observations.append(obs)
        obs.header['VRAD'] = vrad  # TODO: add proper ref
        # obs.print_entire_header()
        # break


        if SAVEFITS and True:
            # save normed spectrum as a .fits file as input for
            #   cross-correlation with logical file name
            filename = {0: 'ERROR', 1: 'masked', 2: 'norm_flux_mask'}
            for i, tosave in enumerate([obs.masked_flux, obs.norm_flux_mask]):
                hdu = pf.PrimaryHDU(tosave)
                hdulist = pf.HDUList([hdu])
                hdulist.writeto('fits_out/{0}_{1}_{2:.0f}_{3:.0f}_{4}.fits'
                        .format(obs.obj_name.upper().replace(' ', ''),
                            obs.unseq, obs.exp_time, obs.snr_red,
                            filename.get(i+1, 0)
                            ),
                        clobber=True
                        )

    # generate_LaTeX_observationlog(all_observations)

    # print "\n\n\n\n\n\n"

    # sum_spectra(all_observations, vrad, plot=False)

    # print "\n\n\n\n\n\n"

    cog_lists = calculate_center_of_gravity_velocity(all_observations,
        xmin, xmax, vrad, LaTeX=False)

    create_mjd_plot(cog_lists)

    if HAS_PYRAF:
        run_iraf()  # Rename to something with cross-correlate?
    else:
        print 'End of pipeline. Pyraf could not be imported. Use IRAF.'


if __name__ == '__main__':
    main()
