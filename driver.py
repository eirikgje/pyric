import utils
import context
import component as comp
import band as bd
import numpy as np
import healpy
import matplotlib.pyplot as plt
import invn

"""
An example driver script for running a Gibbs sampling loop. Simulates a map
using uniform noise, and then samples the posterior.
"""


# A section of defining constants etc.

# nside=2
# nside = 32
# nside = 8
nside = 16
npix = 12 * nside ** 2
polarization = True
include_CMB = True
include_dust = True
include_synch = True

components = []
freqs = [5.0, 40.0, 80.0, 100.0, 120.0, 200.0, 400.0]
dust_reference_freq = 353
dust_spectral_index = 1.6
dust_specind_range = [1.0, 2.5]
dust_temperature = 20.0
synch_reference_freq = 30.0
synch_spectral_index = -3.1
synch_specind_range = [-3.4, -2.8]
plotfreqs = np.arange(freqs[0], freqs[-1], 1/100)
noise_base_sigma_temp = 2 * (nside / 8) ** 2
noise_base_sigma_pol = 0.05 * (nside / 8) ** 2

# Define uniform noise terms
if polarization:
    noise_invsigma = [1/noise_base_sigma_temp, 1/noise_base_sigma_pol,
                      1/noise_base_sigma_pol]
else:
    noise_invsigma = [1/noise_base_sigma_temp]


# Define the various components to be included in both the simulated map and
# the analysis. The component.Component class is mainly meant to be used in
# defining a data/model context, but can also be used to simulate a map (as
# shown further down).
if include_CMB:
    true_cmb_component = comp.CMBComponent(polarization=polarization,
                                           nside=nside)
    healpy.write_map('input_cmb_map.fits', true_cmb_component.amplitude,
                     overwrite=True, dtype=np.float64)
    components.append(true_cmb_component)
    print("Created CMB component")
    cmb_plot_signal = true_cmb_component.emission_noamp(plotfreqs)
    plt.plot(plotfreqs, cmb_plot_signal)

if include_dust:
    if not include_CMB:
        raise ValueError("We don't do that here")
    true_dust_component = comp.MBBComponent(
        dust_reference_freq, dust_spectral_index, dust_temperature,
        polarization=polarization, nside=nside,
        spectral_index_sampling='uniform',
        spectral_index_range=dust_specind_range)
    healpy.write_map('input_dust_map.fits', true_dust_component.amplitude,
                     overwrite=True, dtype=np.float64)
    components.append(true_dust_component)
    dust_plot_signal = true_dust_component.emission_noamp(plotfreqs)
    plt.plot(plotfreqs, dust_plot_signal)
    print("Created dust component")

if include_synch:
    if not include_CMB:
        raise ValueError("We don't do that here")
    true_synch_component = comp.PowerLawComponent(
        synch_reference_freq, synch_spectral_index, polarization=polarization,
        nside=nside, spectral_index_sampling='uniform',
        spectral_index_range=synch_specind_range)
    healpy.write_map('input_synch_map.fits', true_synch_component.amplitude,
                     overwrite=True, dtype=np.float64)
    components.append(true_synch_component)
    synch_plot_signal = true_synch_component.emission_noamp(plotfreqs)
    plt.plot(plotfreqs, synch_plot_signal)
    print("Created synch component")


# Just plot the spectral behaviors
plt.ylim(0, 2)
plt.savefig("specinds.png", bbox_inches='tight')


# Define the observational bands to be included.
bands = []
for freq in freqs:
    bp = bd.DeltaBandpass(freq)
    nmaps = 1
    if polarization:
        nmaps = 3
    freqmap = np.zeros((nmaps, npix))
    for i in range(nmaps):
        # We loop over the already defined components and add them to the data
        # map to simulate.
        for component in components:
            freqmap[i, :] += (component.amplitude[i, :] *
                              component.integrate_band_noamp(bp))
        # Add noise to the simulation.
        freqmap[i, :] += np.random.randn(npix) * 1/noise_invsigma[i]
    noise = invn.InvN([n**2 for n in noise_invsigma],
                      npix, polarization, noise_type='uniform')
    bands.append(bd.Band(bp, noise, freqmap))
    healpy.write_map(f'Freqmap_{int(freq)}Ghz.fits', freqmap, overwrite=True,
                     dtype=np.float64)
print("Created frequency maps")

# Create the data/model context.
context = context.Context(bands, components, nside,
                          noise_model='uniform_noise',
                          polarization=polarization,
                          gibbs_mode='sample')

print("Created context")


# This is the callback function we'll use to handle the result of one Gibbs
# sample.
def write_gibbs_sample_output(gibbs_it, context):
    for i in range(context.num_components):
        fname = f'output_component_{i}_it{gibbs_it}.fits'
        ampmap = context.components[i].amplitude
        if polarization:
            ampmap = np.reshape(ampmap, (3, npix))
        healpy.write_map(fname, ampmap, overwrite=True, dtype=np.float64)

    chisq_maps = utils.calc_chisq(context)
    for i in range(context.num_bands):
        fname = f'chisq_band_{i}_it{gibbs_it}.fits'
        healpy.write_map(fname, chisq_maps[i], overwrite=True,
                         dtype=np.float64)
    fname = f'chisq_it{gibbs_it}.fits'
    sum_chisq = np.sum(chisq_maps, axis=0) / context.num_bands
    healpy.write_map(fname, sum_chisq, overwrite=True, dtype=np.float64)


# final_context = utils.run_gibbs_sampler(context,
#                                         callback=write_gibbs_sample_output,
#                                         num_samples=10)
final_context = utils.run_gibbs_sampler(context,
                                        callback=write_gibbs_sample_output,
                                        num_samples=10)
