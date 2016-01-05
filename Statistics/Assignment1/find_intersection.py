"""
File: find_intersection.py
Authors: Timo L. R. Halbesma <timo.halbesma@student.uva.nl>
         Sarah Brands <saaaaaaaar@hotmail.com>
         Geert Raaijmakers <raaymakers8@hotmail.com>

Version: 1.0 (Final Draft)
Date created: Thu Nov 12, 2015 02:00 PM
Last modified: Thu Nov 19, 2015 04:08 PM

Description:
    Statistical Methods for the Physical Sciences (5214SMFA3Y) --
    Group mini-project --
    Sloan Digital Sky Survey Data Release 7 (SDSS DR7) --
    Catalog of Quasar Properties (CoQP) -- Data from Yue Shen (2011)
    See http://adsabs.harvard.edu/abs/2011ApJS..194...45S

Usage:
    call function 'find_intersection' from file 'find_intersection'.
    e.g.:
        from find_intersection import find_intersection
        attr1_list, attr1_err_list, attr2_list, attr2_err_list =\
        find_intersection(quasars, 'LOGL1350', 'LOGL5100')


Keyword arguments:
    quasars -- numpy array/list with QuasarObservation instances
               NB, bad observations are denoted as None in the array/list
    attr1   -- String of attribute, see QuasarObservation class
    attr2   -- String of seocnd attribute. NB at least two attr required!
    attr3   -- Optional, if given: return intersection of three attributes
    attr4   -- Optional, if given: return intersection of four attributes

@return: tuple with lists of attr1 values, lists of attr1 errors,
         attr2 values attr2 errors, optional 3 and 4.
         NB, if attribute has no error, an empty list is returned!
"""


import numpy


def find_intersection(quasars, attr1, attr2, attr3=None, attr4=None):
    """ Find intersection of two, three or four attributes. """
    # FIXME: change attr[1-4] to a list? Make generic for n attributes?

    # Attributes in has_no_error list have no attr+'_STR'.
    has_no_error = ['RA', 'DEC', 'REDSHIFT', 'UNIFORM_TARGET', 'R_6CM_2500A']

    attr1_list = []
    attr1_err_list = []
    attr2_list = []
    attr2_err_list = []
    if attr3:
        attr3_list = []
        attr3_err_list = []
    if attr4:
        attr4_list = []
        attr4_err_list = []

    i = 0
    for quasar in quasars:
        # We have None in list/array for pulsars with UNIFORM_TARGET == 2
        if quasar:  # Filters out None
            attr1_value = getattr(quasar, attr1, None)

            if attr1 not in has_no_error:
                attr1_error = getattr(quasar, attr1+'_ERR', None)
            attr2_value = getattr(quasar, attr2, None)

            if attr2 not in has_no_error:
                attr2_error = getattr(quasar, attr2+'_ERR', None)
            if attr3:
                attr3_value = getattr(quasar, attr3, None)
                if attr3 not in has_no_error:
                    attr3_error = getattr(quasar, attr3+'_ERR', None)
            else:
                attr3_value = True
            if attr4:
                attr4_value = getattr(quasar, attr4, None)
                if attr4 in has_no_error:
                    attr4_error = getattr(quasar, attr4+'_ERR', None)
            else:
                attr4_value = True

            if attr1_value and attr2_value and attr3_value and attr4_value:
                i += 1
                attr1_list.append(attr1_value)
                if attr1 not in has_no_error:
                    attr1_err_list.append(attr1_error)
                attr2_list.append(attr2_value)
                if attr2 not in has_no_error:
                    attr2_err_list.append(attr2_error)
                if attr3:
                    attr3_list.append(attr3_value)
                    if attr3 not in has_no_error:
                        attr3_err_list.append(attr3_error)
                if attr4:
                    attr4_list.append(attr4_value)
                    if attr4 not in has_no_error:
                        attr4_err_list.append(attr4_error)

    if attr4:
        print "Found {0} quasars in intersection of"\
            .format(i),
        print " '{0}', '{1}', '{2}', and '{3}'"\
            .format(attr1, attr2, attr3, attr4)
        return numpy.asarray(attr1_list), numpy.asarray(attr1_err_list),\
            numpy.asarray(attr2_list), numpy.asarray(attr2_err_list),\
            numpy.asarray(attr3_list), numpy.asarray(attr3_err_list),\
            numpy.asarray(attr4_list), numpy.asarray(attr4_err_list)
    elif attr3:
        print "Found {0} quasars in intersection of '{1}', '{2}', and '{3}'"\
            .format(i, attr1, attr2, attr3)
        return numpy.asarray(attr1_list), numpy.asarray(attr1_err_list),\
            numpy.asarray(attr2_list), numpy.asarray(attr2_err_list),\
            numpy.asarray(attr3_list), numpy.asarray(attr3_err_list)
    else:
        print "Found {0} quasars in intersection of '{1}' and '{2}'"\
            .format(i, attr1, attr2)
        return numpy.asarray(attr1_list), numpy.asarray(attr1_err_list),\
            numpy.asarray(attr2_list), numpy.asarray(attr2_err_list)
