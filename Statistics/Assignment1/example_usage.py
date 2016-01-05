"""
File: example_usage.py
Authors: Timo L. R. Halbesma <timo.halbesma@student.uva.nl>
         Sarah Brands <saaaaaaaar@hotmail.com>
         Geert Raaijmakers <raaymakers8@hotmail.com>

Version: 1.0 (Final Draft)
Date created: Wed Nov 11, 2015 02:45 PM
Last modified: Thu Nov 19, 2015 04:08 PM

Description:
    Statistical Methods for the Physical Sciences (5214SMFA3Y) --
    Group mini-project --
    Sloan Digital Sky Survey Data Release 7 (SDSS DR7) --
    Catalog of Quasar Properties (CoQP) -- Data from Yue Shen (2011)
    See http://adsabs.harvard.edu/abs/2011ApJS..194...45S

Usage:
    call function to see example usage of parse_dataset and QuasarObservation
"""


from parse_dataset import parse_dataset
from find_intersection import find_intersection


def main():
    """ Parse and clean dataset, then do statistics """

    quasars = parse_dataset('sdss_dr7_qsos.dat')

    print 'Example usage'
    quasars = parse_dataset('sdss_dr7_qsos.dat')
    for quasar in [quasars[1], quasars[1337], quasars[13370]]:
        si = quasar.to_si_for_Sarah()  # nieuwe quasar
        print quasar.RA, "=>", si.RA
        print quasar.DEC, "=>", si.DEC
        print '\n\n'


    attr1_list, attr1_err_list, attr2_list, attr2_err_list =\
        find_intersection(quasars, 'LOGL1350', 'LOGL5100')

    attr1_list, attr1_err_list, attr2_list, attr2_err_list,\
        attr3_list, attr3_err_list =\
        find_intersection(quasars, 'LOGL3000', 'LOGL5100', 'LOGBH')

    attr1_list, attr1_err_list, attr2_list, attr2_err_list,\
        attr3_list, attr3_err_list, attr4_list, attr4_err_list =\
        find_intersection(quasars, 'LOGL3000', 'LOGL5100', 'LOGBH', 'LOGLBOL')


if __name__ == '__main__':
    main()
