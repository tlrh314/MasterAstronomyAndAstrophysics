#!/usr/bin/python
"""
RA_Pulsar_Practicum_TLRH_6126561.py
Radio Astronomy
Pulsar Timing Practicum
Timo Halbesma, 6125661
May 18, 2015. Version 1.0
"""

# In this "Radio Astronomy Pulsar Practicum" you will determine the
# spin period and pulse profile shape of the very first pulsar ever
# discovered by Jocelyn Bell: PSR B1919+21.

# Please fill in this .py file with the necessary code to produce the
# plots and answer the imbedded questions in the comments.  The total
# assignment is worth 10 points + 1 bonus point question.

# Note that some of the necessary code has been left in in order to
# help you along.

# The goal is to produce a plot like the one provided.  You can make a
# single plot with subplots or you can make 4 individual plots.  It
# doesn't matter.  Please label the axes and titles of the plots
# properly.

# 1) Import the necessary python modules

import numpy
from matplotlib import pyplot

# 2) Load the data into a numpy array

# Note: the data is:

# a) L197621_SAP0_BEAM1_DM12.437.dat : a simple binary file of 32-bit
# floats.  This file is a timeseries of the data.  The raw data were
# already dedispersed and the bandwidth summed together.  As such,
# this just represents the total signal strength over the whole band
# at a dispersion measure of 12.437pc/cc.  This is the data you need
# to read-in to a numpy array.

# b) L197621_SAP0_BEAM1_DM12.437.inf : a simple text file that lists
# the metadata associated with the .dat file (you can open this with
# any text editor).  For this exercise, you just need the "Width of
# each time series bin (sec)".  That's all you need from this file,
# and you can just copy-paste that value into your code.

data_file = open('L197621_SAP0_BEAM1_DM12.437.dat', 'rb')
timeseries = numpy.fromfile(file=data_file, dtype=numpy.float32)
data_file.close()

# Determine number of bins in the time series (simply the number of
# floats in the binary file

nbins = len(timeseries)

# Set the time per sample (this can be found in the ".inf" file)

time_samp = 0.00262143998406827

# 3) Plot the timeseries.  This plot is worth 1 point.

# QUESTION: Why are there apparently data missing and why does the
# level of the baseline change?  (1 point)

# ANSWER: It looks like there are data missing at the sudden increase - or
# decrease in pulse amplitude and background. This thing you call baseline I
# call the background. I would think someone just switched a button on the
# telescope causing the recording to stop and the background level to change.

# Note that I divide by the minimum value to reduce the size of the
# numbers on the y-axis

pyplot.figure(figsize=(13.5, 10)) # increase figsize to prevent overlapping titles
time = numpy.arange(nbins)*time_samp  # in seconds
ax = pyplot.subplot(4, 1, 1)
pyplot.plot(time, timeseries/1e6)
pyplot.xlabel("Observation Time (s)")
pyplot.ylabel("Amplitude\n(arbitrary)")
pyplot.title("Timeseries")

# 4) Calculate and plot the power spectrum.  This plot is worth 1 point.

# QUESTION: Why are there multiple peaks visible in the power
# spectrum?  What is their relationship, and which one corresponds to
# the pulsar's spin frequency?  (1 point)

# ANSWER: The Fourier power spectrum finds the pulse and the 'overtones'. If
# something is spinning with frequency nu, then there is also a periodicity at
# 2*nu, 3*nu, and so on. The first peak is the frequency, but the second peak is
# thrice the frequency. But the frequency is also the time between the second
# and third peak, between the third and fourth peak, and so on.

power_spectrum = numpy.abs(numpy.fft.fft(timeseries))**2

# Note: you'll want to cut off 10 bins at the beginning and end of the
# power spectrum.  These have very large values and will suppress the
# visibility of the pulsar signal.

# Note: you'll want to plot just the spectrum from ~0.1 - 5 Hz, but
# you can also investigate a broader spectral range if you like.

# Note: to get the x-axis in the right units you need to calculate the
# frequency width of each bin in the power spectrum.  This is given by
# delta_f = 1./(nbins*time_samp) , where nbins is the total number of
# bins in the timeseries and time_samp is the sampling time of the
# data.

ax = pyplot.subplot(4, 1, 2)
delta_f = 1./(nbins*time_samp)
lo_f_bin = 10
hi_f_bin = nbins-10

frequency = numpy.arange(nbins)[lo_f_bin: hi_f_bin] * delta_f  # Hz
power_spectrum_slice = power_spectrum[lo_f_bin: hi_f_bin]

pyplot.plot(frequency, power_spectrum_slice)
pyplot.xlim(0.1, 5)
pyplot.xlabel("Spin Frequency (Hz)")
pyplot.ylabel("Fourier Power\n(arbitrary)")
pyplot.title("Power Spectrum")

# 5) Using the power spectrum peaks, determine the pulse period
# (inverse of the spin frequency).  You can use the higher harmonics
# and divide by the harmonic number to get higher precision on the
# fundamental spin frequency.  Note that in the matplotlib window you
# can zoom in on one of the peaks and then use the cursor so that the
# x,y coordinates are displayed in the bottom left of the plot window.
# That's the easiest way to measure the frequencies of the peaks.

# QUESTION: what are the pulse period and spin frequency (these are
# the inverse of each other)?  (1 point)

maximum_power = max(power_spectrum_slice)
pulse = {}

i, j = 0, 0
pulseFound = False
while i < len(frequency):
    if power_spectrum_slice[i] > 0.5 * maximum_power:
        pulse[j] = frequency[i], power_spectrum_slice[i]
        pulseFound = True

    if pulseFound:
        pulseFound = False
        j += 1

    i += 1

    if frequency[i] > 5:
        break

pulse_periods = []
for i in range(len(pulse)-1):
        pulse_periods.append(1. / (pulse[i+1][0] - pulse[i][0]))

print pulse_periods
print "Pulse period is {0} seconds".format(
    sum(pulse_periods)/len(pulse_periods))

# ANSWER: Pulse period is 1.33695212898 seconds. Spin frequency is 0.74796993

# Here's the value that I determined from the data and then tweaked by
# hand.

# We have limited frequency bin precision, but we could alter the algorithm
# above to a while loop that continues as long as the peak is above a treshold
# value, then obtains the maximum for that peak to more acurately obtain the
# spin period. This method does rely on having high frequency precision.
spin_period = sum(pulse_periods) / len(pulse_periods)
spin_period = 1.337302088331  # http://adsabs.harvard.edu/abs/1994ApJ...422..671A

# 6) Make a plot of the signal strength versus pulse number (time) and
# rotational phase.

# This plot is worth 3 points.  It will be judged on how well the
# pulses line-up in times (i.e. form a vertical line).

# BONUS QUESTION (1 point extra): is it surprising that the pulses
# aren't all of the same intensity?  Pulsars have stable pulse
# profiles, which are the basis for doing precision pulsar timing,
# right?  So how can that work if each pulse is different?  Discuss...

# ANSWER: The atmospheric/ionospheric conditions differ as a function of time,
# the noise levels in the receiver differ as well. However, here we observe
# amplitude changes on the order of seconds, which is somewhat too fast for
# atmospheric changes.

# Note: this requires chopping the timeseries into chunks equal to the
# pulsar period, which you determined in the previous step.

# Note: a tricky thing is that the pulse period will not be an integer
# number of time samples.  You need to drop non-integer number of bins
# for each pulse period (see example code).

# Note: the timeseries also ends on a non-integer pulse phase.  So
# drop the last partial pulse in the data set as a whole.

# Note: if your pulsar period is incorrect then the pulses won't stack
# on top of each other nicely.  You can tweak the period by hand until
# they do.

# Note: I divide the profile in each pulse period by the median in
# that pulse period in order to take out jumps in the baseline
# (cf. first plot of the raw timeseries).

# Number of bins (time samples) across the pulse period
spin_period_bins = spin_period/time_samp

# Create an empty list to store the individual pulse profiles
stacked_profiles = []

# Calculate the bin number that each new pulse starts at
lo_phase_bin = numpy.round(numpy.arange(0, len(timeseries), spin_period_bins))

# Chop the data into chunks of the pulse period
for phase in lo_phase_bin:
    profile = timeseries[numpy.int(phase):numpy.int(phase)+numpy.int(spin_period_bins)]
    profile = profile/numpy.median(profile)
    stacked_profiles.append(profile)

# Convert this to a 2-D numpy array (note that we're chopping off the
# last incomplete pulse).
stacked_profiles = numpy.asarray(stacked_profiles[:-1])

ax = pyplot.subplot(4, 1, 3)
#ax1.set_ylim(0, 120)
#pyplot.ylim(0, 120)
pyplot.imshow(stacked_profiles, origin='lower', aspect=0.4)
pyplot.xlabel("Rotational Phase Bin")
pyplot.ylabel("Pulse Number")
pyplot.title("Pulse Strength vs. Time")

# 7) Plot the cumulative pulse profile.  This just requires squashing
# the 2-D array of pulse profiles to a single dimension.

# This plot is worth 1 point.

# QUESTION: what is the approximate pulse width in seconds and how
# does this compare to the duration of the pulse period?  (1 point)

# ANSWER: The FWHM is roughly 11. If we multiply this with the time bin size we
# end up with 0.0288 seconds. The peaks are roughly 1.33 seconds apart, and the
# pulse duration is the 0.0288 seconds. What is the 'duration of the pulse
# period'? Should this either be the 'pulse period', or 'the duration of the
# pulse'?

squash = {}
for i in range(len(stacked_profiles)):
    for j in range(len(stacked_profiles[i])):
        try:
            if not squash[j]:
                squash[j] = 0
            squash[j] += stacked_profiles[i][j]
        except KeyError:
            squash[j] = 0

ax = pyplot.subplot(4, 1, 4)
pyplot.plot(squash.keys(), squash.values())
pyplot.xlim(0, 500)
pyplot.xlabel("Rotational Phase Bin")
pyplot.ylabel("Pulse Strength\n(arbitrary)")
pyplot.title("Cumulative Pulse Profile")

# This just helps ensure that the plots aren't on top of each other.
pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.75)

# pyplot.tight_layout()
pyplot.savefig("RA_Practicum3_PulsarTiming_TLRH_6126561.png")
pyplot.show()
