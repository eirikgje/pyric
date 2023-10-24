import numpy as np

""" Defines the InvN class, meant to be a general inverse noise covariance
matrix representation.
"""


class InvN:
    def __init__(self, values, npix, polarization=False, noise_type='uniform'):
        """ General class representing an inverse noise covariance matrix.

        values (float or np.ndarray): The actual values of the inverse noise
            covariance matrix.
                If noise_type='uniform', this should be a single value (float
                    or array) if nmaps=1, an array of length 3 otherwise.
                If noise_type='diagonal', this should be an array of length
                    (npix*nmaps), where nmaps is 3 if polarization is enabled,
                    1 otherwise.
                If noise_type='matrix', this should be an array of shape
                    (npix*nmaps, npix*nmaps).
        npix (int): The number of pixels (12 * nside ** 2).
        polarization (bool): Whether to include polarization in the analysis.
        noise_type (str):
            'uniform' if there is one value for the whole sky
                (but potentially different values for I, Q, U)
            'diagonal' if each pixel has a noise variance, but no covariance
                between pixels.
            'matrix' if there are both within-and-between variations per pixel.
        """
        self.values = values
        self.polarization = polarization
        self.noise_type = noise_type
        self.npix = npix
        self.nmaps = 1
        if polarization:
            self.nmaps = 3
        if self.noise_type == 'uniform':
            if self.polarization:
                assert len(self.values) == self.nmaps
            else:
                try:
                    if len(self.values) != 1:
                        raise ValueError("Uniform noise with temperature only "
                                         "should have length 1")
                except TypeError:
                    self.values = [self.values]
        elif self.noise_type == 'diagonal':
            assert self.values.ndim == 1
            assert len(self.values) == self.npix * self.nmaps
        elif self.noise_type == 'matrix':
            assert self.values.ndim == 2
            assert len(self.values) == self.npix * self.nmaps
            assert len(self.values[0]) == self.npix * self.nmaps

    def get_invn_as_vector(self):
        "For uniform or diagonal noise matrices, return it as a vector."

        if self.noise_type == 'matrix':
            raise ValueError("Trying to get a noise matrix as a vector")
        if self.noise_type == 'uniform':
            return_val = []
            for i in range(self.nmaps):
                return_val.extend([self.values[i]]*self.npix)
            return np.array(return_val)
        elif self.noise_type == 'diagonal':
            return np.array(self.values)

    def get_invn_as_matrix(self):
        raise NotImplementedError
