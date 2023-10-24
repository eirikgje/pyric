import numpy as np
from scipy.interpolate import CubicSpline

"""
Module defining the general Band class, meant to represent a single detector in
a CMB experiment. It also defines several Bandpass types, which must be input
to a Band instance, to define what kind of bandpass it uses. Currently, only
the DeltaBandpass and GeneralBandpass classes are implemented, where
GeneralBandpass can take a tabulated response per frequency and interpolate.
"""


class Band:

    def __init__(self, bandpass, invn, data):
        """ General Band class to be used to characterize a detector

        bandpass: Bandpass instance, defining the frequency range over which
            the detector is defined.
        invn: invn.Invn instance, defining the inverse noise properties of the
            band.
        data: array of (nmaps, npix) values, representing the IQU measurements
            per pixel.
        """
        self.bandpass = bandpass
        self.invn = invn
        self.data = data


class Bandpass:
    """ Base class to represent bandpasses"""

    def __init__(self):
        pass


class DeltaBandpass(Bandpass):
    def __init__(self, frequency):
        """ DeltaBandpass represents a bandpass defined only at a single
        frequency.

        The normalization is such that when integrating a spectral
        response over this bandpass, the differential response at the frequency
        represented by the bandpass is returned.

        frequency (float): The frequency (in GHz) at which the bandpass is
            defined.
        """
        super().__init__()
        self.frequency = frequency

    def response(self, frequencies):
        """ Returns the bandpass response over an array of frequencies.

        The return value will be an array of the same shape as the frequencies
        which is zero everywhere except for the frequency at which the bandpass
        is defined.
        """
        try:
            response = np.zeros(frequencies.shape)
            response[np.where(frequencies == self.frequency)] = 1
            return response
        except AttributeError:
            if frequencies == self.frequency:
                return 1
            return 0


class GeneralBandpass(Bandpass):
    def __init__(self, frequencies, response):
        """
        GeneralBandpass represents a bandpass with tabulated responses at given
            frequencies.

        To calculate the response at other frequencies than those given, the
        tabulated responses are interpolated, and the interpolation evaluated
        at the input frequencies.

        frequencies (array): Frequency values (in GHz) at which the bandpass
            response is tabulated.
        response (array): The response corresponding to each frequency value.
        """
        super().__init__()
        self.frequencies = frequencies
        self.tabulated_responses = response
        self.spline = CubicSpline(frequencies, response)

    def response(self, frequencies):
        """ Interpolates the bandpass at the given frequencies.

        This has not been tested and may not work as intended yet.
        """
        return self.spline(frequencies)
