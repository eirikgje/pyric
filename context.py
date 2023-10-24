import numpy as np


"""
Module defining the Context class, which is the catch-all class that contains
all relevant information about the data, the model, the sampling method,
etc. It is the object that is passed around and updated."""


class Context:
    def __init__(self, bands, components, nside, noise_model='uniform',
                 polarization=True, gibbs_mode='sample'):
        """ The current context.


        bands (Band array): The bands involved in the analysis. Each band
            contains an observation which is what is used in the general
            likelihood evaluation.
        components (Component array): The components we include in our data
            model.
        nside (int): The general nside of the data (currently only a common
            nside is supported).
        noise_model (str): Can be 'uniform', 'diagonal', or 'full_matrix' to
            indicate the type of noise used. Should correspond to the noise
            model used in the bands. Currently, all bands should be of the same
            type.
        polarization (bool): Whether the analysis includes polarization data.
        gibbs_mode (str): Currently only 'sample' is supported.
        """
        self.bands = bands
        self.gibbs_mode = gibbs_mode
        self.num_bands = len(bands)
        self.components = components
        self.num_components = len(components)
        self.nside = nside
        self.npix = 12 * nside ** 2
        self.noise_model = noise_model
        self.polarization = polarization
        if self.noise_model == 'full_matrix':
            if self.nside > 16:
                raise ValueError("Can't run full noise matrix with nsides "
                                 "higher than 16")

    def get_bands(self):
        "Return all bands."
        return self.bands

    def get_band(self, band_idx):
        """
        Return a specific band, specified by the index corresponding to the
        band's position in the Band array that was given upon initialization.
        """

        return self.bands[band_idx]

    def get_components(self):
        "Return all components."
        return self.components

    def get_component(self, cmp_idx):
        """
        Return a specific component, specified by the index corresponding to
        the component's position in the Component array that was given upon
        initialization.
        """
        return self.components[cmp_idx]

    def get_data(self):
        """
        Return all data, gathered into one array.

        Returns a flattened data array of shape (npix * nmaps * num_bands),
        where npix = 12 * nside ** 2, nmaps is 3 if polarization is enabled, 1
        otherwise, and num_bands is the length of the input Band array given
        upon initialization. If nmaps is 3, all three maps will be put
        consecutively before a new band is started.
        """
        npix = self.npix
        nmaps = 3 if self.polarization else 1
        num_bands = self.num_bands
        data = np.zeros(npix * nmaps * num_bands)
        for i, band in enumerate(self.get_bands()):
            for j in range(nmaps):
                data[i*nmaps*npix + j*npix:i*nmaps*npix+(j+1)*npix] = (
                        band.data[j, :])
        return data

    def update_signal_components(self, new_signal):
        """
        Set new values for the signal part of the sky model components.

        new_signal (1d array): Array of length (npix * nmaps * n_comp), where
            npix = 12 * nside ** 2 where nside is the nside set in the
            initialization, nmaps is 1 if polarization=False and 3 otherwise,
            and n_comp is the number of components that the context contains.
        """
        nmaps = 3 if self.polarization else 1
        for i, component in enumerate(self.components):
            component.set_amplitude(
                    new_signal[i*self.npix*nmaps:(i+1)*self.npix*nmaps])
