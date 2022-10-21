import matplotlib.pyplot as plt
import sys
import os
import argparse
import glob

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
        "-l", "--label", type=str, default="TP Profile",
        help="Label for the plot."
        )

cmd_line = parser.parse_args()

if not os.path.exists(cmd_line.path):
    raise ValueError(f"Could not find file: {cmd_line.path}")

elif os.path.isdir(cmd_line.path):
    file_dir = cmd_line.path.rstrip('/')
    HELIOS_filename = glob.glob(f"{file_dir}/*_tp.dat")[0]

    # Deleting here just in case a variable has the same name later on.
    del file_dir


#################################### FUNCTIONS ####################################

def read_and_plot(ax, path, color='blue', shade='darkorange', label='', width=2, style='-'):

    press, temp, cpress0, ctemp0, cpress1, ctemp1, cpress2, ctemp2, cpress3, ctemp3 = tls.read_helios_tp(path)

    ax.plot(ctemp0, cpress0, color=shade,linewidth=8, alpha=0.7)
    ax.plot(ctemp1, cpress1, color=shade,linewidth=8, alpha=0.7)
    ax.plot(ctemp2, cpress2, color=shade,linewidth=8, alpha=0.7)
    ax.plot(ctemp3, cpress3, color=shade,linewidth=8, alpha=0.7)

    line, = ax.plot(temp, press, color=color, linewidth=width, linestyle=style, alpha=1, label=label)

    return line

########################################### PLOT ##########################################

fig, ax = plt.subplots()

read_and_plot(
        ax,
        HELIOS_filename,
        label=cmd_line.label
        )

ax.set(
        ylim=[1e3, 1e-6],
        yscale='log',
        xlabel=r'temperature (K)',
        ylabel=r'pressure (bar)'
        )

ax.legend(loc='best', frameon=True)

plt.savefig("tp.pdf")
plt.show()
