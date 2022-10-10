'''Generates plots of all the test cases for the new haze feature(s) and
opacities.
'''
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
import astropy.constants as c
from astropy.visualization import quantity_support

import glob

import pylustrator
#pylustrator.start()


# Baseline case vs. thick and thin haze case
# Implemented with soot_haze testing and the following cases:
# THICK HAZE 
# === === CLOUDS === ===
# 
# number of cloud decks =                               1                             
# path to Mie files =                                   ./input/cloud_files/test_soot/
# aerosol radius mode [micron] =                        10   20                       
# aerosol radius geometric std dev =                    2    1.5                      
# cloud mixing ratio =                                  manual                        
#   file   --> path to file with cloud data =           ./input/cloud_file.txt        
#   file   --> cloud file format =                      1 Pressure cgs                
#   file   --> aerosol name =                           Aerosol1  Aerosol2            
#   manual --> cloud bottom pressure [10^-6 bar] =      1e5       1e3                 
#   manual --> cloud bottom mixing ratio =              1e-10     1e-10               
#   manual --> cloud to gas scale height ratio =        0.5       0.5                 
# 
# === === HAZES === ===
# 
# haze opacity file =                                   ./input/haze_opac.dat         
# haze profile file =                                   ./input/haze_profile.dat      
def plot_tp_from_file(filename, ax, name="", label=None, plt_kwargs={}):
    data = {
            'temp':  [],
            'press': [],
            'alt':   [],
            'effective_temperature': 0.
            }

    with open(filename) as infile:
        # Skip two lines
        infile.readline()
        infile.readline()

        for i, line in enumerate(infile):
            cols = line.strip().split()
            data['temp'].append(float(cols[1]))
            data['press'].append(float(cols[2]))
            data['alt'].append(float(cols[3]))

            # The first row has the effetive temperature.
            if i == 0:
                data['effective_temperature'] = float(cols[-1])

    data['temp'] = np.array(data['temp']) * u.Kelvin
    data['press'] = np.array(data['press']) * 1e-6 * u.bar
    data['alt'] = np.array(data['alt']) * u.cm

    if label is None:
        # Label will be the effective temperature and any name given by the
        # name argument.
        label = f"{name} " if name else ""
        label += f"({int(data['effective_temperature'])} K)"

        plt_kwargs['label'] = label

    elif 'label' not in plt_kwargs:
        plt_kwargs['label'] = label

    line, = ax.plot(data['temp'], data['press'], **plt_kwargs)

    return line, data


def plot_cloud_mr(filename, ax, name="", label=None, plt_kwargs={}):
    data = {
            'press':        [],
            'mixing_ratio': []
            }

    with open(filename, 'r') as infile:
        # Skip first three lines
        infile.readline()
        infile.readline()

        for line in infile:
            _, p, mr = [float(x) for x in line.split()]

            data['press'].append(p)
            data['mixing_ratio'].append(mr)

        data['press'] = np.array(data['press']) * 1e-6 * u.bar

    if label is None and 'label' not in plt_kwargs:
        # Label will be the effective temperature and any name given by the
        # name argument.
        label = f"{name} " if name else ""
        label += f"({int(data['effective_temperature'])} K)"

        plt_kwargs['label'] = label

    elif 'label' not in plt_kwargs:
        plt_kwargs['label'] = label

    line, = ax.plot(data['mixing_ratio'], data['press'], **plt_kwargs)

    return line, data


def plot_eclipse_spec(filename, ax, name="", label=None, plt_kwargs={}):
    data = {
            'wavelength':   [],
            'fp_fs':        []
            }

    with open(filename, 'r') as infile:
        # Skip first two lines
        infile.readline()
        infile.readline()
        infile.readline()

        for line in infile:
            cols = [float(x) for x in line.split()]

            data['wavelength'].append(cols[1])
            data['fp_fs'].append(cols[-1])

        data['wavelength'] = data['wavelength'] * u.micron
        data['fq_fs'] = data['fp_fs'] * u.dimensionless_unscaled

    if label is None and 'label' not in plt_kwargs:
        # Label will be the effective temperature and any name given by the
        # name argument.
        label = f"{name} " if name else ""
        plt_kwargs['label'] = label.strip()

    elif 'label' not in plt_kwargs:
        plt_kwargs['label'] = label

    line, = ax.plot(data['wavelength'], data['fp_fs'], **plt_kwargs)

    return line, data


def main():
    # Plot the TPs I want to plot
    dirs = {"output/thick_haze_testcase": "Thick, .5 ",
            "output/mod_haze_testcase":  "Mid, .6",
            "output/thin_haze_testcase":  "Thin, .7",
            "output/baseline_case":       "Baseline"
            }

    colors = ['orange', 'g', 'b', 'k']

    assert len(dirs) == len(colors), "Must have same num of colors and dirs"

    # Plot the tp profiles.
    tp_fig, tp_ax = plt.subplots(figsize=(6, 4))

    tp_ax.invert_yaxis()

    tp_ax.set(
            yscale='log',
            ylabel='Pressure',
            xlabel='Temperature'
        )

    # Also plot the cloud profile
    cld_ax = tp_ax.twiny()
    cld_ax.set_yscale('log')
    cld_ax.set_xscale('log')

    cld_ax.set_xlabel('Cloud Volume Mixing Ratio')


    # Plot the eclipse spectra
    spec_fig, spec_ax = plt.subplots(figsize=(6, 4))

    spec_ax.set(
            xscale='log',
            xlabel='Wavlength',
            ylabel='Planet/Star Flux Ratio'
            )

    for (d, name), c in zip(dirs.items(), colors):
        # TP and clouds
        d = d.rstrip()
        filename = f"{d}/{d.split('/')[-1]}_tp.dat"

        l, tp_data = plot_tp_from_file(filename,
                                       tp_ax,
                                       name=name,
                                       plt_kwargs={'color': c}
                                       )

        plt_kwargs = {
                'color': c,
                'ls': '--',
                'label': None
                }

        filename = f"{d}/{d.split('/')[-1]}_cloud_mixing_ratio.dat"
        plot_cloud_mr(filename,
                     cld_ax,
                     name=None,
                     plt_kwargs=plt_kwargs
                     )

        # Spectrum
        filename = f"{d}/{d.split('/')[-1]}_TOA_flux_eclipse.dat"
        plot_eclipse_spec(filename,
                          spec_ax,
                          name=name,
                          plt_kwargs={'color': c}
                          )

    tp_ax.legend(loc='best')
    spec_ax.legend(loc='best')

    # Let pylustrator do its thing
    tp_fig.tight_layout()
    spec_fig.tight_layout()
    #% start: automatic generated code from pylustrator
    plt.figure(2).ax_dict = {ax.get_label(): ax for ax in plt.figure(2).axes}
    import matplotlib as mpl
    plt.figure(2).axes[0].set_xlim(0.16849534309836586, 723.2700054643944)
    plt.figure(2).axes[0].set_xticks([1.0, 10.0, 100.0])
    plt.figure(2).axes[0].set_xticklabels(["1", "10", "100"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="DejaVu Sans", horizontalalignment="center")
    plt.figure(2).axes[0].get_xaxis().get_label().set_text("Wavelength (micron)")
    #% end: automatic generated code from pylustrator
    #% start: automatic generated code from pylustrator
    plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
    import matplotlib as mpl
    plt.figure(1).axes[0].get_xaxis().get_label().set_text("Temperature (Kelvin)")
    plt.figure(1).axes[0].get_yaxis().get_label().set_text("Pressure (bar)")
    plt.figure(1).axes[1].set_position([0.128897, 0.145694, 0.845377, 0.704611])
    plt.figure(1).axes[1].xaxis.labelpad = -4.313498
    plt.figure(1).axes[1].xaxis.labelpad = 7.448476
    #% end: automatic generated code from pylustrator
    plt.show()

    # Save the figures
    tp_fig.savefig("tp_clouds.png", dpi=200)
    spec_fig.savefig("eclipse_spec.png", dpi=200)

if __name__ == "__main__":
    main()
