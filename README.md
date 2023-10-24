# pyric

PYthon library for Rapid Investigation of (foreground) Components.

### Structure/philosophy

The pyric library is meant to allow for users to 'rapidly' do semi-realistic Bayesian component separation in a non-prohibitive time-scale. In practice, this means that currently, the library is limited to
* Diagonal noise matrices,
* Low and common resolutions (nside=16 is the recommended resolution),
* Uniform spectral indices.

The intended flow of analysis is as follows:
* Create a context using the `Context` class found in `context.py`. The context should contain
    * A list of bands, created using the `Band` class found in `band.py`. Each of these represent a detector with a given bandpass and a given set of observations.
    * A list of components to include in the analysis, using the `Component` class in the `component.py` module. Currently three kinds of components are possible: CMB components, modified blackbody components, and powerlaw components. An analysis can contain several instances of the same type of component - this means that the sampler will try to fit those components simultaneously.
* Once the context has been created, you can run a Gibbs loop using that context by calling `run_gibbs_sampler` within `utils.py`.

See `driver.py` for a working example. In order for this script to work, you must either pass a proper filename to the initialization routines found in `component.py` so that they match files on your computer, or you must change these filenames in the code itself.

The data currently used to simulate a sky is
* A best-fit powerspectrum file (default is found here: `https://pla.esac.esa.int/pla-sl/data-action?COSMOLOGY.COSMOLOGY_OID=2800)`
* A dust template, currently the Planck 353 GHz map: `https://pla.esac.esa.int/pla-sl/data-action?MAP.MAP_OID=14613`
* A 'synch' template (really just the 30 GHz Cosmoglobe DR1 map): `http://sdc.uio.no/vol/cosmoglobe-data/cosmoglobe_DR1/CG_030_IQU_n0512_v1.fits`

### Dependencies

Currently, pyric depends on `numpy`, `scipy`, `astropy`, `healpy` and `matplotlib` and `imageio` (for plotting results).
