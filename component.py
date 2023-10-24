import numpy as np
import band
from scipy.integrate import quad
from astropy import constants
import healpy


"""
Module for representing a general sky component. Currently contains a general
abstract class ("ModelComponent") as well as specific classes of physical sky
components.  They are mainly meant to be used as representations in a sampling
process, representing which sky components we're including in our data model,
but can be used in simulating a sky map as well.

Currently, we represent all components in K_rj (though there are good arguments
to be in Mjy/sr).

"""


def simulate_cmb(
        polarization=True, nside=16,
        powspec_path='/home/eirik/data/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt'):
    """ Utility function for simulating a CMB amplitude map.

    Assumes the Cls to be in the format l, TT, TE, EE, BB (or a format that
    synfast can understand)
    """

    cls = np.loadtxt(powspec_path)
    cls[:, 1:] = cls[:, 1:] / (cls[:, 0:1] * (cls[:, 0:1] + 1))
    if polarization:
        cmb_map = healpy.synfast(np.append(cls[:, 1:4],
                                           np.zeros((len(cls[:, 0]), 1)),
                                           axis=1).T, nside=nside)
    else:
        npix = 12 * nside ** 2
        cmb_map = healpy.synfast(cls[:, 1], nside=nside).reshape((1, npix))
    return cmb_map


def read_353_map(polarization=True, nside=16,
                 map_path='/home/eirik/data/vincent_forecast/HFI_SkyMap_353-psb-field-IQU_2048_R3.00_full.fits'):
    """ Utility function for creating a 'realistic' dust simulation using the
    353 GHz Planck amplitude map.
    """
    inmap = healpy.ud_grade(
            healpy.read_map(map_path, field=None), nside) * 1000
    if polarization:
        dustamps = inmap
    else:
        npix = 12 * nside ** 2
        dustamps = np.zeros((1, npix))
        dustamps[0, :] = inmap[0, :]
    return dustamps


def read_synch_init(polarization=True, nside=16):
    """ Utility function for creating a 'realistic' synch simulation using a
    Commander input file, with a reference frequency of 30 GHz.

    This isn't really well balanced and I think 30 GHz is not a good reference
    frequency for this, so I will try to find a better template.)
    """

    synch_template = '/home/eirik/data/vincent_forecast/synch_init_30Ghz.fits'
    inmap = healpy.ud_grade(healpy.read_map(synch_template, field=None), nside)
    # I have no idea about what the proper amplitude of this should be
    inmap /= 100
    if polarization:
        synchamp = inmap
    else:
        npix = 12 * nside ** 2
        synchamp = np.zeros((1, npix))
        synchamp[0, :] = inmap[0, :]
    return synchamp


class ModelComponent:
    """ Abstract class for a sky model.

    Per now, all sky models are assumed to have an amplitude, as well as a flag
    indicating whether the component is polarized, and whether spectral index
    sampling is possible (and should be performed, if general spectral index
    sampling is turned on) for this component.
    """

    def __init__(self, amplitude, polarization=False,
                 spectral_index_sampling=None):
        """

        Amplitude is expected to be in the format (nmaps, npix), where nmaps is
        1 if polarization=False, 3 otherwise, and npix=12*nside**2.

        spectral_index_sampling (str or None):
            If set to 'uniform', indicates sampling one common spectral index
            for the whole map (and the same value for temperature and
            polarization maps).
            If None, sampling spectral indices will not include this component.
        """
        self.amplitude = amplitude
        self.polarization = polarization
        self.spectral_index_sampling = spectral_index_sampling

    def integrate_band(self, bandpass):
        """ Integrates the component over a bandpass.

        bandpass: Instance of the Bandpass class.

        Returns: The component amplitude multiplied by the integrated spectral
            response of the component convolved with the bandpass response over
            the bandpass frequencies. It will thus typically be an array of the
            form (nmaps, npix).
        """

        if isinstance(bandpass, band.DeltaBandpass):
            return self.emission(bandpass.frequency)
        else:
            return quad(lambda x: bandpass.response(x) * self.emission(x), )

    def integrate_band_flattened(self, bandpass):
        """ Integrates the component over a bandpass, returning a flattened
        array.

        bandpass: Instance of the Bandpass class.

        Returns: The flattened component amplitude multiplied by the integrated
            spectral response of the component convolved with the bandpass
            response over the bandpass frequencies. It will thus typically be
            an array with length nmaps * npix.
        """

        if isinstance(bandpass, band.DeltaBandpass):
            return self.emission_flattened(bandpass.frequency)
        else:
            raise NotImplementedError

    def integrate_band_noamp(self, bandpass):
        """ Integrates the component spectral response (without amplitude) over
        a bandpass.

        bandpass: Instance of the Bandpass class.

        Returns: The integrated spectral response of the component convolved
            with the bandpass response over the bandpass frequencies, *not*
            multiplied by the amplitude. The return value will have a shape
            that depends on the type of spectral index sampling. For uniform
            spectral index sampling, it will be a single number.
        """

        if isinstance(bandpass, band.DeltaBandpass):
            return self.emission_noamp(bandpass.frequency)
        else:
            return quad(lambda x: bandpass.response(x) *
                        self.emission_noamp(x), )

    def set_amplitude(self, new_amplitude):
        """ Set a new amplitude for the component

        The input amplitude can be flattened or (nmaps, npix), but it needs to
        have nmaps*npix elements in total.
        """
        if new_amplitude.ndim < 2:
            if self.polarization:
                self.amplitude = np.reshape(new_amplitude,
                                            (3, int(len(new_amplitude) / 3)))
            else:
                self.amplitude = np.reshape(new_amplitude,
                                            (1, len(new_amplitude)))
        else:
            self.amplitude = new_amplitude

    def emission(self, freqs):
        """ Calculate emission at a frequency or set of frequencies.

        freqs (float, array): The set of frequencies (in GHz) to evaluate the
            component at.

        Returns: The current component amplitudes multiplied by the spectral
            emission at the given frequencies. Will have shape (nmaps, npix).
        """
        return self.amplitude * self.emission_noamp(freqs)

    def emission_flattened(self, freqs):
        """ Calculate emission at a frequency or set of frequencies, returned
        as a flattened array

        freqs (float, array): The set of frequencies (in GHz) to evaluate the
            component at.

        Returns: The current component amplitudes, flattened, multiplied by the
        spectral emission at the given frequencies. Will have length (nmaps *
        npix).
        """

        return self.get_flattened_amplitude() * self.emission_noamp(freqs)

    def get_flattened_amplitude(self):
        """ Return the current amplitude of the component, flattened into a
        1d-array of length nmaps*npix.
        """
        return self.amplitude.flatten()


class PowerLawComponent(ModelComponent):
    """ Class representing a component that follows a power-law spectral
    behavior:

    I(nu)[K_RJ] = a * (nu/nu_ref) ** beta

    where nu_ref is the reference frequency, a is the reference amplitude, and
    beta is the spectral index.
    """

    def __init__(self, reference_freq, spectral_index, polarization=False,
                 init_amp=read_synch_init, nside=16,
                 spectral_index_range=None, spectral_index_sampling=None):
        """
        reference_freq (float): The reference frequency of the powerlaw.
        spectral_index (float, array of floats): The spectral index (or set of
            indices) characterising the sky component. If a float, it's assumed
            it's a common spectral index for all pixels and all polarization
            components.
        init_amp (func): A function that can generate the initial amplitude of
            this component. Should return a (nmaps, npix) array.
        spectral_index_range (tuple or list of two numbers): The range over
            which spectral index sampling will take place.
        """
        amplitude = read_synch_init(polarization, nside)
        super().__init__(amplitude, polarization=polarization,
                         spectral_index_sampling=spectral_index_sampling)
        self.spectral_index_range = spectral_index_range
        self.reference_freq = reference_freq
        self.spectral_index = spectral_index

    def emission_noamp(self, freqs):
        """ The powerlaw emission without an amplitude (i.e. normalized to one
            at the reference frequency) for a set of frequencies """

        return (freqs / self.reference_freq) ** self.spectral_index


class CMBComponent(ModelComponent):
    """ Class representing a component that follows the blackbody behavior with
        CMB temperature T_CMB:

    I(nu)[K_RJ] = x ** 2 * exp(x) / (exp(x) - 1) ** 2

    where

    x = h * nu / (k_b * T_CMB)

    """

    def __init__(self, polarization=False, init_amp=simulate_cmb, nside=16):
        """
        init_amp (func): A function that can generate the initial amplitude of
                         this component. Should return a (nmaps, npix) array.
        """
        amplitude = init_amp(polarization, nside)
        super().__init__(amplitude, polarization=polarization)
        self.CMB_TEMP = 2.7255

    def emission_noamp(self, freqs):
        """ The CMB emission without an amplitude for a set of frequencies """
        x = (constants.h.value * freqs * 1e9 /
             (constants.k_B.value * self.CMB_TEMP))
        return x ** 2 * np.exp(x) / (np.exp(x) - 1) ** 2


class MBBComponent(ModelComponent):
    """ Class representing a component that follows a modified blackbody (MBB)
    behavior (like thermal dust):

    I(nu)[K_RJ] = x ** 2 * exp(x) / (exp(x) - 1) ** 2

    where

    x = h * nu / (k_b * T_CMB)
    """

    def __init__(self, reference_freq, spectral_index, temperature,
                 polarization=False, init_amp=read_353_map, nside=16,
                 spectral_index_range=None, spectral_index_sampling=None):
        amplitude = read_353_map(polarization, nside)
        super().__init__(amplitude, polarization=polarization,
                         spectral_index_sampling=spectral_index_sampling)
        self.spectral_index_range = spectral_index_range
        self.reference_freq = reference_freq
        self.spectral_index = spectral_index
        self.temperature = temperature

    def emission_noamp(self, freqs):
        h = constants.h.value
        k = constants.k_B.value
        t = self.temperature
        return (freqs/self.reference_freq) ** self.spectral_index * (
            (np.exp(h*self.reference_freq * 1e9 / (k * t)) - 1) /
            (np.exp(h*freqs * 1e9 / (k * t)) - 1))
