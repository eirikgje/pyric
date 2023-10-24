import numpy as np
import mixmat

"""General utility functions that don't belong elsewhere.
Most of the sampling related routines are currently here.
"""


def run_gibbs_sampler(context, callback=None, num_samples=1000):
    """Runs a Gibbs sampling loop given a context.

    Currently, the context *data* is assumed to be immutable, whereas the
    *model* is meant to reflect the latest update of the Gibbs loop. Hence,
    e.g. the model amplitudes and spectral indices in the Context object  will
    change after having been called by this function.

    context (context.Context): Describes the context (the model and data of the
        situation). Will be changed as described above.
    callback (function): A function to be called after every Gibbs loop. Must
        take two parameters: An integer representing the current Gibbs
        iteration, and the Context object (updated with the latest results).
    num_samples (int, default=1000): The number of Gibbs samples to draw.

    Returns: The context after the Gibbs loop
    """

    for i in range(num_samples):
        print(f"Running Gibbs sample {i}")
        context = sample_signal_components(context)
        context = sample_spectral_params(context)
        if callback is not None:
            callback(i, context)

    return context


def sample_signal_components(context):
    """Sample the signal coomponents of a model by finding the lsq solution.

    This function will solve the equation

    (A^T)N^{-1}Ax = A^TN^{-1}d (+N^{-1/2}eta)

    for x, where d is the observed data (for all bands, put into one vector),
    N^{-1} is the inverse noise matrix (for all bands put into one matrix),
    and A is the mixing matrix (see mixmat.Mixingmatrix). eta is a random
    vector, and will be added only if the sampling mode is 'sample' - this
    represents the statistical fluctuation term.

    context (context.Context): The data/model context. Will be updated with the
        sampled signal component.

    Returns: The updated context.
    """
    ata, rhs = get_lsq_matrices(context)
    print("Found lsq matrices")
    sol = np.linalg.solve(ata, rhs)
    print("Solved for the signal")
    context.update_signal_components(sol)
    return context


def sample_spectral_params(context):
    """
    Samples the spectral index of each component (that has a spectral index).

    Currently, only spectral indices that are uniform on the whole sky is
        supported.

    context (context.Context): The data/model context. Will be updated with the
        sampled spectral index component.

    Returns: The updated context.
    """

    for comp_ind, component in enumerate(context.components):
        if component.spectral_index_sampling is None:
            continue
        if component.spectral_index_sampling == 'uniform':
            specind = sample_uniform_spectral_index(comp_ind, context)
            context.components[comp_ind].spectral_index = specind
        else:
            raise NotImplementedError
    return context


def sample_uniform_spectral_index(component_index, context):
    """ Samples a uniform spectral index for a component.

    The sampling is done using inversion sampling by evaulating the chisquared
    on a grid (given in the context). The grid range will potentially be
    updated as part of the sampling: If more than 25% of the current grid
    points returns a probability of 0, the grid will be updated to be 1.2 times
    the distance from the probability maximum to the first zero.

    component_index (int): The index corresponding to the desired component
        (see component.get_component).
    context (context.Context): The data/model context.

    Returns: The context, with a potentially modified range for the spectral
        index sampling.
    """
    component = context.get_component(component_index)
    spec_range = np.linspace(component.spectral_index_range[0],
                             component.spectral_index_range[1],
                             1000)
    offset = np.sum(calc_chisq(context))
    def prob(specind, context=context, component_index=component_index,
             offset=offset):
        context.components[component_index].spectral_index = specind
        chisq_maps = calc_chisq(context)
        return np.exp(-0.5 * (np.sum(chisq_maps)-offset))

    specind, probs = inversion_sampling(spec_range, prob)
    if np.count_nonzero(probs) < 0.75 * len(probs):
        max_argument = np.argmax(probs)
        zeros = np.where(probs[:max_argument] == 0)[0]
        last_zero_ind = zeros[-1]
        distance = spec_range[max_argument] - spec_range[last_zero_ind]
        new_range = [spec_range[max_argument] - 1.2 * distance,
                     spec_range[max_argument] + 1.2 * distance]
        context.components[component_index].spectral_index_range = new_range
    print(f"Specind: {specind}")
    return specind


def inversion_sampling(x_range, fun):
    """ General function to perform inversion sampling.

    x_range (float array): The range over which the function to be sampled is
        evaluated.
    fun (function): The function to sample from.

    Returns: A sample from the input function over the given x_range.

    """
    funarr = []
    for x in x_range:
        funarr.append(fun(x))
    cumulative_dist = np.cumsum(funarr)
    cumulative_dist /= cumulative_dist[-1]
    rand = np.random.rand()
    x_interp = np.interp(rand, cumulative_dist, x_range)
    return x_interp, np.array(funarr)


def get_lsq_matrices(context):
    """
    Specialized auxilliary function to find the matrices needed for
    sample_signal_components.

    context (context.Context): The data/model context.

    Returns: A (lhs, rhs) tuple where lhs and rhs are given by the equation to
        be solved in sample_signal_components. lhs is a square matrix of shape
        (num_components*nmaps*npix, num_components*nmaps*npix) where
        num_components is the number of components involved, nmaps is 1 if
        polarization is not included, 3 otherwise, and npix=12*nside**2.
        Similarly, rhs is a vector of (num_components*nmaps*npix) elements.
    """
    nmaps = 3 if context.polarization else 1
    npix = context.npix
    ata = np.zeros((context.num_components*nmaps*npix,
                    context.num_components*nmaps*npix))
    rhs = np.zeros((context.num_components*nmaps*npix))
    data = context.get_data()
    mixing_matrix = mixmat.MixingMatrix(context.bands, context.components,
                                        context.npix,
                                        polarization=context.polarization)
    for band_ind in range(context.num_bands):
        invn = context.bands[band_ind].invn
        if invn.noise_type == 'matrix':
            raise NotImplementedError
        if context.gibbs_mode == 'sample':
            fluct_term = np.random.randn(nmaps*npix)
        for comp_ind1 in range(context.num_components):
            curr_mixmat1 = mixing_matrix.get_band_comp_mixmat(band_ind,
                                                              comp_ind1)
            at = curr_mixmat1 * invn.get_invn_as_vector()
            for comp_ind2 in range(context.num_components):
                curr_mixmat2 = mixing_matrix.get_band_comp_mixmat(
                        band_ind, comp_ind2)
                for nmap in range(nmaps):
                    for pix in range(npix):
                        ata[comp_ind1*nmaps*npix + nmap*npix + pix,
                            comp_ind2*nmaps*npix + nmap*npix + pix] += (
                                    at[nmap*npix + pix] *
                                    curr_mixmat2[nmap*npix + pix])
            rhs[comp_ind1*nmaps*npix:(comp_ind1+1)*nmaps*npix] += (
                    at * data[band_ind*nmaps*npix:(band_ind+1)*nmaps*npix])
            if context.gibbs_mode == 'sample':
                rhs[comp_ind1*nmaps*npix:(comp_ind1+1)*nmaps*npix] += (
                        curr_mixmat1 * np.sqrt(invn.get_invn_as_vector()) *
                        fluct_term)
    return ata, rhs


def calc_chisq(context):
    """Calculates the chi-squared per band given the current context.

    context (context.Context): the data/model context.

    Returns: An array of shape (num_bands, nmaps, npix) where num_bands is the
        number of bands included in the analysis, nmaps is 1 if polarization is
        not involved in the analysis, 3 otherwise. The array contains the
        quantity (d-m)^TN^{-1}(d-m) where d is the data for a given band,
        N^{-1} is the inverse noise covariance matrix for the band, and m is
        the current model evaluated at that band.
    """
    data = context.get_data()
    nmaps = 3 if context.polarization else 1
    npix = context.npix
    chisq_maps = np.zeros((context.num_bands, nmaps, npix))
    for band_ind in range(context.num_bands):
        invn = context.bands[band_ind].invn
        if invn.noise_type == 'matrix':
            raise NotImplementedError
        res = data[band_ind*nmaps*npix:(band_ind+1)*nmaps*npix]
        for component in context.components:
            res -= component.integrate_band_flattened(
                    context.bands[band_ind].bandpass)
        curr_chisq = res**2 * invn.get_invn_as_vector()
        for nmap in range(nmaps):
            chisq_maps[band_ind, nmap] = curr_chisq[nmap*npix:(nmap+1)*npix]
    return chisq_maps
