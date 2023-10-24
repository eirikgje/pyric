import healpy
import imageio
import matplotlib.pyplot as plt

""" Utility module for plotting the output of a Gibbs chain. Creates pngs and
gifs. This module expects that files are written as is currently done in the
callback function defined in driver.py, so if that is changed, this script will
not necessarily work."""


# Defining parameters for the chain (this probably should be rewritten to take
# a Context or something like that as input.

num_components = 3
ranges = [[[-50, 50], [-0.2, 0.2], [-0.2, 0.2]],
          [[0, 10], [0, 0.2], [0, 0.2]],
          [[0, 1], [0, 0.5], [0, 0.5]]]
num_bands = 7
freqs = [5.0, 40.0, 80.0, 100.0, 120.0, 200.0, 400.0]
polarization = True

iterations = 10
gif_duration = 400


for comp_ind in range(num_components):
    images_I = []
    images_Q = []
    images_U = []
    for it in range(iterations):
        base = f'output_component_{comp_ind}_it{it}'
        try:
            currmap = healpy.read_map(base+'.fits', field=None)
        except FileNotFoundError:
            break

        healpy.mollview(currmap[0, :], min=ranges[comp_ind][0][0],
                        max=ranges[comp_ind][0][1], title=f'Iteration {it}')
        plt.savefig(base + '_I.png', bbox_inches='tight')
        plt.close()
        if polarization:
            healpy.mollview(currmap[1, :], min=ranges[comp_ind][1][0],
                            max=ranges[comp_ind][1][1],
                            title=f'Iteration {it}')
            plt.savefig(base + '_Q.png', bbox_inches='tight')
            plt.close()
            healpy.mollview(currmap[2, :], min=ranges[comp_ind][2][0],
                            max=ranges[comp_ind][2][1],
                            title=f'Iteration {it}')
            plt.savefig(base + '_U.png', bbox_inches='tight')
            plt.close()
        images_I.append(imageio.imread(base + '_I.png'))
        if polarization:
            images_Q.append(imageio.imread(base + '_Q.png'))
            images_U.append(imageio.imread(base + '_U.png'))
    imageio.mimsave(f'output_component_{comp_ind}_I.gif', images_I,
                    duration=gif_duration, loop=0)
    if polarization:
        imageio.mimsave(f'output_component_{comp_ind}_Q.gif', images_Q,
                        duration=gif_duration, loop=0)
        imageio.mimsave(f'output_component_{comp_ind}_U.gif', images_U,
                        duration=gif_duration, loop=0)

# Plotting the chi-squareds per band
for band in range(num_bands):
    chisq_I = []
    chisq_Q = []
    chisq_U = []
    for it in range(iterations):
        base = f'chisq_band_{band}_it{it}'
        currmap = healpy.read_map(base+'.fits', field=None)
        healpy.mollview(currmap[0, :], min=0, max=1, title=f'Iteration {it}')
        plt.savefig(base+'_I.png', bbox_inches='tight')
        plt.close()
        chisq_I.append(imageio.imread(base + '_I.png'))
        if polarization:
            healpy.mollview(currmap[1, :], min=0, max=1,
                            title=f'Iteration {it}')
            plt.savefig(base+'_Q.png', bbox_inches='tight')
            plt.close()
            healpy.mollview(currmap[2, :], min=0, max=1,
                            title=f'Iteration {it}')
            plt.savefig(base+'_U.png', bbox_inches='tight')
            plt.close()
            chisq_Q.append(imageio.imread(base + '_Q.png'))
            chisq_U.append(imageio.imread(base + '_U.png'))
    imageio.mimsave(f'chisq_band_{band}_I.gif', chisq_I, duration=gif_duration,
                    loop=0)
    if polarization:
        imageio.mimsave(f'chisq_band_{band}_Q.gif', chisq_Q,
                        duration=gif_duration, loop=0)
        imageio.mimsave(f'chisq_band_{band}_U.gif', chisq_U,
                        duration=gif_duration, loop=0)


# Total chisquared
chisq_I = []
chisq_Q = []
chisq_U = []
for it in range(iterations):
    base = f'chisq_it{it}'
    currmap = healpy.read_map(base+'.fits', field=None)
    healpy.mollview(currmap[0, :], min=0, max=1, title=f'Iteration {it}')
    plt.savefig(base+'_I.png', bbox_inches='tight')
    plt.close()
    chisq_I.append(imageio.imread(base + '_I.png'))
    if polarization:
        healpy.mollview(currmap[1, :], min=0, max=1, title=f'Iteration {it}')
        plt.savefig(base+'_Q.png', bbox_inches='tight')
        plt.close()
        healpy.mollview(currmap[2, :], min=0, max=1, title=f'Iteration {it}')
        plt.savefig(base+'_U.png', bbox_inches='tight')
        plt.close()
        chisq_Q.append(imageio.imread(base + '_Q.png'))
        chisq_U.append(imageio.imread(base + '_U.png'))
imageio.mimsave('chisq_I.gif', chisq_I, duration=gif_duration, loop=0)
if polarization:
    imageio.mimsave('chisq_Q.gif', chisq_Q, duration=gif_duration, loop=0)
    imageio.mimsave('chisq_U.gif', chisq_U, duration=gif_duration, loop=0)


# Frequency map plotting
for freq in freqs:
    base = f'Freqmap_{int(freq)}Ghz'
    currmap = healpy.read_map(base+'.fits', field=None)
    healpy.mollview(currmap[0, :], min=-40, max=40)
    plt.savefig(base+'_I.png', bbox_inches='tight')
    plt.close()
    if polarization:
        healpy.mollview(currmap[1, :], min=-5, max=5)
        plt.savefig(base+'_Q.png', bbox_inches='tight')
        plt.close()
        healpy.mollview(currmap[2, :], min=-5, max=5)
        plt.savefig(base+'_U.png', bbox_inches='tight')
        plt.close()
