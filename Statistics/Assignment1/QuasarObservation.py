"""
File: QuasarObservation.py
Authors: Timo L. R. Halbesma <timo.halbesma@student.uva.nl>
         Sarah Brands <saaaaaaaar@hotmail.com>
         Geert Raaijmakers <raaymakers8@hotmail.com>

Version: 0.01 (Initial)
Date created: Fri Nov 06, 2015 02:35 pm
Last modified: Thu Nov 12, 2015 04:19 PM

Description:
    Statistical Methods for the Physical Sciences (5214SMFA3Y) --
    Group mini-project --
    Sloan Digital Sky Survey Data Release 7 (SDSS DR7) --
    Catalog of Quasar Properties (CoQP) -- Data from Yue Shen (2011)
    See http://adsabs.harvard.edu/abs/2011ApJS..194...45S

Usage:
    use class 'QuasarObservation' from file 'QuasarObservation' to store
    dataset obtained as part of the assignment.

Functions:
    __init__: create new instance of QuasarObservation class.

    __str__: print quasar observation data (all attributes of the class).

    clean: change missing data from '-1' or '0' to Python's Nonetype.
        keyword arguments:
            verbose       -- Print bad measurement (could be used)
            debug -- Print every bad datapoint (bad idea to use!)

    to_si_for_Sarah: returns new quasar with attributes in SI, not cgs.
"""


class QuasarObservation(object):
    """ SDSS DR7 CoQP Quasar Observation """
    def __init__(self,
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
                 ):

        self.Row = Row
        self.RA = RA
        self.DEC = DEC
        self.REDSHIFT = REDSHIFT

        """
            Uniformity flag - an integer denoting the sample the quasar is
            drawn from. Not so relevant here but Shen advises that flag=2
            objects are not suitable for statistical studies so you should
            try to exclude them.
        """
        self.UNIFORM_TARGET = UNIFORM_TARGET

        # In log_10(L_bol) [erg/s]
        self.LOGLBOL = LOGLBOL
        self.LOGLBOL_ERR = LOGLBOL_ERR

        # radio intensity (6cm) / optical intensity (250 nm)
        self.R_6CM_2500A = R_6CM_2500A

        """ 'Continuum' luminosities and errors obtained at three
            separate rest-frame  wavelengths: 510 nm (optical), 300 nm
            (near-UV) and 135 nm (far-UV).  This emission  is caused by
            thermal blackbody emission from material falling towards the
            black hole in  an 'accretion disc'.
        """
        self.LOGL5100 = LOGL5100  # Optical
        self.LOGL5100_ERR = LOGL5100_ERR
        self.LOGL3000 = LOGL3000  # Near_UV
        self.LOGL3000_ERR = LOGL3000_ERR
        self.LOGL1350 = LOGL1350  # Far-UV
        self.LOGL1350_ERR = LOGL1350_ERR

        """ Atomic emission line luminosities and errors associated with
            Halpha (broad and  narrow line components given separately)
            Hbeta (also broad and narrow lines), [OIII] (500.7 nm), MgII
            and CIV. The emission lines are produced by gas clouds at
            various  differences from the black hole, and illuminated by
            the accretion disc.
        """
        self.LOGL_BROAD_HA = LOGL_BROAD_HA
        self.LOGL_BROAD_HA_ERR = LOGL_BROAD_HA_ERR
        self.LOGL_NARROW_HA = LOGL_NARROW_HA
        self.LOGL_NARROW_HA_ERR = LOGL_NARROW_HA_ERR
        self.LOGL_BROAD_HB = LOGL_BROAD_HB
        self.LOGL_BROAD_HB_ERR = LOGL_BROAD_HB_ERR
        self.LOGL_NARROW_HB = LOGL_NARROW_HB
        self.LOGL_NARROW_HB_ERR = LOGL_NARROW_HB_ERR
        self.LOGL_OIII_5007 = LOGL_OIII_5007
        self.LOGL_OIII_5007_ERR = LOGL_OIII_5007_ERR
        self.LOGL_MGII = LOGL_MGII
        self.LOGL_MGII_ERR = LOGL_MGII_ERR
        self.LOGL_CIV = LOGL_CIV
        self.LOGL_CIV_ERR = LOGL_CIV_ERR

        """ Black hole mass (expressed as log10 of the mass in units
            of solar mass) and its  error.
        """
        self.LOGBH = LOGBH  # log_10(M/M_Sun)
        self.LOGBH_ERR = LOGBH_ERR

    def __str__(self):
        """ Printing an instance of QuasarObservation class
            calls this function """
        str = "" + \
            "Quasar {0}\n".format(self.Row) + \
            "RA \t\t\t {0}\n".format(self.RA) + \
            "DEC \t\t\t {0}\n".format(self.DEC) + \
            "REDSHIFT \t\t {0}\n".format(self.REDSHIFT) + \
            "UNIFORM_TARGET \t\t {0}\n".format(self.UNIFORM_TARGET) + \
            "LOGLBOL \t\t {0}\n".format(self.LOGLBOL) + \
            "LOGLBOL_ERR \t\t {0}\n".format(self.LOGLBOL_ERR) + \
            "R_6CM_2500A \t\t {0}\n".format(self.R_6CM_2500A) + \
            "LOGL5100 \t\t {0}\n".format(self.LOGL5100) + \
            "LOGL5100_ERR \t\t {0}\n".format(self.LOGL5100_ERR) + \
            "LOGL3000 \t\t {0}\n".format(self.LOGL3000) + \
            "LOGL3000_ERR \t\t {0}\n".format(self.LOGL3000_ERR) + \
            "LOGL1350 \t\t {0}\n".format(self.LOGL1350) + \
            "LOGL1350_ERR \t\t {0}\n".format(self.LOGL1350_ERR) + \
            "LOGL_BROAD_HA \t\t {0}\n".format(self.LOGL_BROAD_HA) + \
            "LOGL_BROAD_HA_ERR \t {0}\n".format(self.LOGL_BROAD_HA_ERR) + \
            "LOGL_NARROW_HA \t\t {0}\n".format(self.LOGL_NARROW_HA) + \
            "LOGL_NARROW_HA_ERR \t {0}\n".format(self.LOGL_NARROW_HA_ERR) + \
            "LOGL_BROAD_HB \t\t {0}\n".format(self.LOGL_BROAD_HB) + \
            "LOGL_BROAD_HB_ERR \t {0}\n".format(self.LOGL_BROAD_HB_ERR) + \
            "LOGL_NARROW_HB \t\t {0}\n".format(self.LOGL_NARROW_HB) + \
            "LOGL_NARROW_HB_ERR \t {0}\n".format(self.LOGL_NARROW_HB_ERR) + \
            "LOGL_OIII_5007 \t\t {0}\n".format(self.LOGL_OIII_5007) + \
            "LOGL_OIII_5007_ERR \t {0}\n".format(self.LOGL_OIII_5007_ERR) + \
            "LOGL_MGII \t\t {0}\n".format(self.LOGL_MGII) + \
            "LOGL_MGII_ERR \t\t {0}\n".format(self.LOGL_MGII_ERR) + \
            "LOGL_CIV \t\t {0}\n".format(self.LOGL_CIV) + \
            "LOGL_CIV_ERR \t\t {0}\n".format(self.LOGL_CIV_ERR) + \
            "LOGBH \t\t\t {0}\n".format(self.LOGBH) + \
            "LOGBH_ERR \t\t {0}".format(self.LOGBH_ERR)

        return str

    # FIXME: neat method, but doubles data parse time. Could be done in parse_dataset ?
    def clean(self, verbose=False, debug=False):
        """
        i) If quasar uniformity is 2 it should be excluded
        from the statistical sample, according to Yue Shen
        ii) Missing values are denoted as -1. Here, change it to None.
        iii) Missing luminosities, however, are denoted as zero. Change to None.
        """

        if self.UNIFORM_TARGET == 2:
            if debug:
                print 'WARN: {0} should be excluded' \
                    .format(self.Row),
                print 'from the statistical sample. Uniformity = 2'

        for attr in [a for a in dir(self) if not a.startswith('__')]:
            attr_value = getattr(self, attr)
            if ('LOGL' in attr and attr_value == 0) or\
                    ('LOGL' not in attr and attr_value == -1):
                setattr(self, attr, None)
                if debug:  # bad idea to print this :-)...
                    print "WARN: '{0}' misses '{1}', was '{2}', now set to '{3}'." \
                        .format(self.Row, attr, attr_value,
                                getattr(self, attr))


    def to_si_for_Sarah(self):
        """ Return new instance of QuasarObservation with attribtues in SI """
        def lum_to_si(log_10_of_lum):
            """ Convert log(L) in erg/s to L in W """
            if log_10_of_lum:  # Prevent error when data misses.
                return 10**(log_10_of_lum) * 1e-7  # erg/s => W
            else:
                return None
        def MSun_to_si(solarmass):
            """ Convert solar mass to SI mass """
            if solarmass:  # Prevent error when data misses.
                return solarmass*1.988435e30,  # MSun => kg
            else:
                return None

        def ra_to_hour(RA):
            """ Convert right ascension from degrees to hour, min, sec """
            ra_hour = int(RA/15)
            ra_min = int(60*((RA/15) % 1))
            ra_sec = 60*((60*((RA/15) % 1)) % 1)
            return " {0:02d}h {1:02d}m {2:.2f}s"\
                .format(ra_hour, ra_min, ra_sec)

        def dec_to_hour(DEC):
            """ Convert declination from degrees to deg, min, sec """
            dec_hour = int(abs(DEC))
            dec_min = int(60*(abs(DEC) % 1))  # might have precision issues
            dec_sec = 60*((60*(abs(DEC) % 1)) % 1)
            return "{0}{1:02d}d {2:02d}' {3:.2f}\""\
                .format("+" if DEC > 0 else "-", dec_hour, dec_min, dec_sec)

        return QuasarObservation(
            self.Row,
            ra_to_hour(self.RA), dec_to_hour(self.DEC),
            # self.RA, self.DEC,
            self.REDSHIFT, self.UNIFORM_TARGET,
            lum_to_si(self.LOGLBOL),
            lum_to_si(self.LOGLBOL_ERR),
            self.R_6CM_2500A,
            lum_to_si(self.LOGL5100),
            lum_to_si(self.LOGL5100_ERR),
            lum_to_si(self.LOGL3000),
            lum_to_si(self.LOGL3000_ERR),
            lum_to_si(self.LOGL1350),
            lum_to_si(self.LOGL1350_ERR),
            lum_to_si(self.LOGL_BROAD_HA),
            lum_to_si(self.LOGL_BROAD_HA_ERR),
            lum_to_si(self.LOGL_NARROW_HA),
            lum_to_si(self.LOGL_NARROW_HA_ERR),
            lum_to_si(self.LOGL_BROAD_HB),
            lum_to_si(self.LOGL_BROAD_HB_ERR),
            lum_to_si(self.LOGL_NARROW_HB),
            lum_to_si(self.LOGL_NARROW_HB_ERR),
            lum_to_si(self.LOGL_OIII_5007),
            lum_to_si(self.LOGL_OIII_5007_ERR),
            lum_to_si(self.LOGL_MGII),
            lum_to_si(self.LOGL_MGII_ERR),
            lum_to_si(self.LOGL_CIV),
            lum_to_si(self.LOGL_CIV_ERR),
            MSun_to_si(self.LOGBH),
            MSun_to_si(self.LOGBH_ERR)
            )
