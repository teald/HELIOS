import matplotlib.pyplot as plt
import sys
import os
import glob
import argparse


sys.path.append(os.path.dirname(__file__).rstrip('/') + "/..")
from source import tools as tls


# Parse command line arguments
if os.path.dirname(__file__) == os.getcwd():
    default_dir = "../output/0/"

else:
    default_dir = "output/0/"

parser = argparse.ArgumentParser()
parser.add_argument(
        "-p", "--path", type=str, default=default_dir,
        help="Directory where the spectrum file is located. Assumes "
        "the file uses the default HELIOS file naming scheme."
        )
parser.add_argument(
        "-l", "--label", type=str, default="Emission spectrum",
        help="Label for the plot."
        )

cmd_line = parser.parse_args()

if not os.path.exists(cmd_line.path):
    raise ValueError(f"Could not find file: {cmd_line.path}")

elif os.path.isdir(cmd_line.path):
    file_dir = cmd_line.path.rstrip('/')
    HELIOS_filename = glob.glob(f"{file_dir}/*TOA_flux_eclipse.dat")[0]

    # Deleting here just in case a variable has the same name later on.
    del file_dir


#################################### FUNCTIONS ####################################

def read_and_plot(ax,
                  path,
                  label="",
                  width=1,
                  style="-",
                  color="blue",
                  alpha=0.6,
                  rebin=0
                  ):

    lamda, spec = tls.read_helios_spectrum(path, type='emission')

    if rebin > 0:
        lamda, spec = tls.rebin_spectrum_to_resolution(lamda, spec, resolution=rebin, w_unit='micron')

    line, = ax.plot(lamda, spec, color=color,linewidth=width, linestyle=style, label=label, alpha=alpha)

    return line

########################################### READ & PLOT ##########################################

fig, ax = plt.subplots()

read_and_plot(
        ax,
        HELIOS_filename,
        label=cmd_line.label
        )

ax.set(
        yscale='log',
        xlim=[0.25, 20],
        xscale='log',
        xlabel='wavelength ($\mu$m)',
        ylabel='flux (erg s$^{-1}$ cm$^{-3}$)'
        )

ax.set_xticks([0.5, 1, 2, 3, 5, 10, 20])
ax.set_xticklabels(['0.5', '1', '2', '3', '5', '10', '20'])

ax.legend(loc='best', frameon=True)

plt.savefig("spectrum.pdf")
plt.show()
