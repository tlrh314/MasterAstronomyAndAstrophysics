"""
File: parse_dataset.py
Authors: Timo L. R. Halbesma <timo.halbesma@student.uva.nl>
         Sarah Brands <saaaaaaaar@hotmail.com>
         Geert Raaijmakers <raaymakers8@hotmail.com>

Version: 1.0 (Final Draft)
Date created: Wed Nov 11, 2015 02:45 PM
Last modified: Thu Nov 19, 2015 03:45 PM

Description:
    Statistical Methods for the Physical Sciences (5214SMFA3Y) --
    Group mini-project --
    Sloan Digital Sky Survey Data Release 7 (SDSS DR7) --
    Catalog of Quasar Properties (CoQP) -- Data from Yue Shen (2011)
    See http://adsabs.harvard.edu/abs/2011ApJS..194...45S

Usage:
    call function 'parse_dataset' from file 'parse_dataset'.
    e.g.:
        from parse_dataset import parse_dataset
        quasars = parse_dataset('sdss_dr7_qsos.dat')

Keyword arguments:
    filename  -- filename of datafile, defaults to 'sdss_dr7_qsos.dat'.
    verbose   -- give verbose output while reading data, defaults to True.
    debug     -- print additional information for development, default False.
"""


import sys
import time
import csv

import numpy

from QuasarObservation import QuasarObservation


def parse_dataset(filename='sdss_dr7_qsos.dat', verbose=True, debug=False):
    """ Open and parse dataset. If possible return numpy array, else list """

    # http://www.cyberciti.biz/faq/python-run-external-command-and-get-output/
    if verbose:
        print "Reading dataset '{0}' started.".format(filename)

    use_array, i = False, 1  # Have i start at 1 such that index equals
    skipped, skipped_list = 0, []
    try:
        import subprocess
        p = subprocess.Popen("wc -l {0}".format(filename),
                             stdout=subprocess.PIPE, shell=True)
        output, err = p.communicate()
        p_status = p.wait()

        file_length = output.split()[0]
    # Very bad practice, but just in case anything breaks at TA's machine..
    except:
        if verbose:
            print 'WARN: Failed to obtain file length. Reading into lists.'
        quasars = []
        quasars.append(None)
    else:
        try:
            file_length = int(file_length)
        except ValueError:
            if verbose:
                print 'WARN: Failed to obtain file length. Reading into lists.'
            quasars = []
            quasars.append(None)
        else:
            if p_status is 0 and type(file_length) is int:
                if verbose:
                    print 'Reading into numpy arrays :-)'
                qo_dtype = numpy.dtype(QuasarObservation)
                quasars = numpy.empty(file_length, dtype=qo_dtype)
                use_array = True

    if verbose:
        start_time = time.time()

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        reader.next()  # Skip first (header) row
        for row in reader:
            # row list -> string, strip whitespace, split into list
            # FIXME: this can most likely be optimized hugely
            column_list = row[0].strip().split()

            Row = int(column_list[0])
            RA = float(column_list[1])
            DEC = float(column_list[2])
            REDSHIFT = float(column_list[3])
            UNIFORM_TARGET = float(column_list[4])
            LOGLBOL = float(column_list[5])  # log_10(L_bol) [erg/s]
            LOGLBOL_ERR = float(column_list[6])

            # radio intensity (6cm) / optical intensity (250 nm)
            R_6CM_2500A = float(column_list[7])

            LOGL5100 = float(column_list[8])  # Optical
            LOGL5100_ERR = float(column_list[9])
            LOGL3000 = float(column_list[10])  # Near_UV
            LOGL3000_ERR = float(column_list[11])
            LOGL1350 = float(column_list[12])  # Far-UV
            LOGL1350_ERR = float(column_list[13])

            LOGL_BROAD_HA = float(column_list[14])
            LOGL_BROAD_HA_ERR = float(column_list[15])
            LOGL_NARROW_HA = float(column_list[16])
            LOGL_NARROW_HA_ERR = float(column_list[17])
            LOGL_BROAD_HB = float(column_list[18])
            LOGL_BROAD_HB_ERR = float(column_list[19])
            LOGL_NARROW_HB = float(column_list[20])
            LOGL_NARROW_HB_ERR = float(column_list[21])
            LOGL_OIII_5007 = float(column_list[22])
            LOGL_OIII_5007_ERR = float(column_list[23])
            LOGL_MGII = float(column_list[24])
            LOGL_MGII_ERR = float(column_list[25])
            LOGL_CIV = float(column_list[26])
            LOGL_CIV_ERR = float(column_list[27])

            LOGBH = float(column_list[28])  # log_10(M/M_Sun)
            LOGBH_ERR = float(column_list[29])

            quasar = QuasarObservation(
                Row, RA, DEC,
                REDSHIFT, UNIFORM_TARGET, LOGLBOL,
                LOGLBOL_ERR, R_6CM_2500A,
                LOGL5100, LOGL5100_ERR,
                LOGL3000, LOGL3000_ERR,
                LOGL1350, LOGL1350_ERR,
                LOGL_BROAD_HA, LOGL_BROAD_HA_ERR,
                LOGL_NARROW_HA, LOGL_NARROW_HA_ERR,
                LOGL_BROAD_HB, LOGL_BROAD_HB_ERR,
                LOGL_NARROW_HB, LOGL_NARROW_HB_ERR,
                LOGL_OIII_5007, LOGL_OIII_5007_ERR,
                LOGL_MGII, LOGL_MGII_ERR,
                LOGL_CIV, LOGL_CIV_ERR,
                LOGBH, LOGBH_ERR
                )

            quasar.clean()
            if use_array:
                if quasar.UNIFORM_TARGET != 2:
                    quasars[i] = quasar
                else:
                    skipped += 1
                    skipped_list.append(quasar.Row)

                # Print readout progress bar =)
                bar_width = 42  # obviously
                if verbose and ((i == file_length-1) or (i % 100 is 0)):
                    progress = float(i)/file_length
                    block = int(round(bar_width * progress))
                    sys.stdout.write(
                        "\r[{0}{1}] {2:.2f}%"
                        .format('#'*block,
                                ' '*(bar_width - block),
                                progress*100))
                    sys.stdout.flush()

                i = i+1
            else:
                # file length is not known, so no progress bar :-(
                if quasar.UNIFORM_TARGET != 2:
                    quasars.append(quasar)
                else:
                    skipped += 1
                    skipped_list.append(quasar.Row)
                    quasars.append(None)

    if verbose:
        sys.stdout.write('\n')
        print "Read {0} quasars in {1} seconds." \
            .format(len(quasars)-1, time.time() - start_time)
        print "Skipped {0} of the {1} quasars because uniformity is 2."\
            .format(skipped, len(quasars)-1)
        print "Continuing statistical anaylsis with {0} quasars.\n"\
            .format(len(quasars)-skipped-1)
    if debug:
        print "Skipped the following quasars:"
        print skipped_list
    return quasars
