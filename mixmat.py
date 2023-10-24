import numpy as np

"""
Defines the MixingMatrix class, which is meant to be a interface to the
mixing matrix which abstracts away the 'heavy lifting' involved in calculating
this quantity.
"""


class MixingMatrix:
    def __init__(self, bands, components, npix, polarization=False):
        """ Defines a mixing matrix.

        The mixing matrix of a context that involves n detectors and m
        components is defined as a (npix x nmaps x n x m) matrix, where npix is
        the number of pixels in the map and nmaps is 1 if polarization is not
        included in the analysis, 3 otherwise. Each element is defined as the
        integrated bandpass response of the given component over the given
        band, in the pixel in question.

        bands: Iterable of band.Band instances defining the bands involved. The
            array order is important, and use of get_band_comp_mixmat will
            assume the same order.
        components: Iterable of component.Component instances defining the
            components involved. The array order is important, and use of
            get_band_comp_mixmat will assume the same order.
        npix (int) : The number of pixels per map.
        polarization (bool) : Whether polarization is included in the analysis.
        """

        self.bands = bands
        self.components = components
        self.polarization = polarization
        self.npix = npix
        self.nmaps = 1
        if self.polarization:
            self.nmaps = 3

    def get_band_comp_mixmat(self, band_ind, comp_ind):
        """ Get the mixing matrix for a band-component combination.

        band_ind (int): The index of the initialization band array
            corresponding to the desired band.
        comp_ind (int): The index of the initialization component array
            corresponding to the desired component.

        Returns: An (npix x nmaps) array consisting of the component in
            question integrated over the bandpass in question for every pixel.
        """
        element = self.components[comp_ind].integrate_band_noamp(
                self.bands[band_ind].bandpass)
        try:
            if len(element) == self.npix * self.nmaps:
                return element
        except TypeError:
            return np.array([element]*self.npix*self.nmaps)
