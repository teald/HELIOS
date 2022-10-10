'''Converts a set of haze property files and turns them into HELIOS-readable
format.
'''
import numpy as np
import glob
import os
import pdb
import shutil
import astropy.units as u
from scipy.interpolate import interp2d


# Inputs
new_haze_folder_name = "test_soot"
haze_files = "../soot_props/out*out"


# Process inputs
new_haze_folder_name = new_haze_folder_name.strip('/')

def get_all_data(file_names: str) ->dict:
    if isinstance(file_names, str):
        all_files = glob.glob(file_names)

    data = {}
    for f in all_files:
        with open(f, 'r') as infile:
            # Get all the header data
            # First line is the partictle radius (effective spherical radius)
            line = infile.readline()
            rpar = float(line.split()[2]) * u.micron

            # Second line can be ignored.
            infile.readline()

            # Third line is monomer size
            line = infile.readline()
            rmon = float(line.split()[2]) * u.micron

            # Fourth line is df (no clue)
            line = infile.readline()
            df = float(line.split()[2]) * u.dimensionless_unscaled

            # Fifth line is monomer number
            nmon = float(line.split()[2]) * u.dimensionless_unscaled

        # Get the actual data using genfromtxt
        arr_data = np.genfromtxt(f, skip_header=6)

        wl = arr_data[:, 0] * u.micron
        qext = arr_data[:, 1]
        qsca = arr_data[:, 2]
        gf = arr_data[:, 3]

        rdata = {
                "PARTICLE RADIUS"      : rpar.cgs.value,
                "MONOMER RADIUS"       : rmon.cgs.value,
                "DF"                   : df.value,
                "NMON"                 : nmon.value,
                "WAVELENGTH"           : wl.cgs.value,
                "Q_EXTINCTION"         : qext,
                "Q_SCATTERING"         : qsca,
                "ASYMMETRY PARAMETER"  : gf
                }

        radius_str = f"{rpar.cgs.value:1.5e}"

        data[radius_str] = rdata

    return data

def cross_section(radius, q, g0=0):
    return np.pi * radius**2 * q * (1 - g0**2)

def gen_helios_data(haze_data: dict) -> dict:
    helios_data = {}

    conv_u = lambda d, u1, u2: (d * u1).to(u2).value

    for radius_label, data in haze_data.items():
        hdata = {}

        wl = data['WAVELENGTH']
        hdata['wavelengths (mu)'] = conv_u(wl, u.cm, u.micron)

        # Can leave size parameter as zeros
        hdata['size parameter'] = np.zeros_like(wl)

        # Can put in extinction cross section since we have it.
        radius = data['PARTICLE RADIUS']
        qext = data['Q_EXTINCTION']
        g0 = data['ASYMMETRY PARAMETER']
        xsec_ext = cross_section(radius, qext, g0)

        # Radius is in micron I think
        xsec_ext = conv_u(xsec_ext, u.micron**2, u.cm**2)
        hdata['extinction cross section (cm^2)'] = xsec_ext

        # Need scattering cross section
        qsca = data['Q_SCATTERING']
        xsec_sca = cross_section(radius, qsca, g0)
        xsec_sca = conv_u(xsec_sca, u.micron**2, u.cm**2)
        hdata['scattering cross section (cm^2)'] = xsec_sca

        # Need absorption cross section
        qabs = qext - qsca
        xsec_abs = cross_section(radius, qabs, g0)
        xsec_abs = conv_u(xsec_abs, u.micron**2, u.cm**2)
        hdata['absorption cross section (cm^2)'] = xsec_abs

        # Single scattering albedo
        ssa = qsca / qext
        hdata['single scattering albedo'] = ssa

        # Asymmetry parameter
        hdata['asymmetry parameter'] = g0

        helios_data[radius_label] = hdata

    return helios_data

def make_helios_files(haze_folder: str, helios_haze_data: dict):
    # These are the hardcoded radii for HELIOS that need to be interpolated
    # onto. Taken straight from the HELIOS source Oct. 6 2022.
    r_values = 10 ** np.arange(-2, 3.1, 0.1)
    delta_r = r_values * (10**0.05 - 10**-0.05)

    # Create the array of lables.
    labels = [f"r{_r:.6f}.dat" for _r in r_values]
    helios_files = {l: r for l, r in zip(labels, r_values)}

    # Interpolate onto the radius/wavelength grid.
    keys = helios_haze_data.keys()
    sorted_radii_keys = sorted(keys, key=lambda x: float(x))
    radii = [float(x) for x in sorted_radii_keys]
    wl_grid = helios_haze_data[list(keys)[0]]['wavelengths (mu)']

    files = {}

    r_filename = lambda r: f"r{r:.6f}.dat"

    for radius in r_values:
        if radius <= radii[0]:
            neighbors = (None, 0)

        elif radius >= radii[-1]:
            neighbors = (len(radii) - 1, None)

        else:
            # Find the two nearest radii.
            lneighbor = 0
            while radii[lneighbor] > radius:
                lneighbor += 1

            neighbors = (lneighbor, lneighbor + 1)

            del lneighbor

        # If either is None, just use the data.
        if None in neighbors:
            irad = neighbors[0] if neighbors[1] is None else neighbors[1]
            rkey = sorted_radii_keys[irad]
            dat = helios_haze_data[rkey]
            files[r_filename(radius)] = build_cloudfile_string(dat);
            continue

        # Interpolate between the two values.
        def interp(x, x1, x2, y1, y2):
            slope = (y2 - y1) / (x2 - x1)
            return slope * (x - x1) + y1

        # this assumes the wavelength grids are already the same.
        r1 = radii[neighbors[0]]
        r2 = radii[neighbors[1]]
        rkey1 = sorted_radii_keys[neighbors[0]]
        rkey1 = sorted_radii_keys[neighbors[1]]
        dat1 = helios_haze_data[rkey1]
        dat2 = helios_haze_data[rkey2]

        interp_dat = {}

        for key in dat1:
            d1 = dat1[key]
            d2 = dat2[key]

            interp_dat[key] = interp(radius, r1, r2, d1, d2)

        files[r_filename(radius)] = build_cloudfile_string(interp_dat)

    dirstr = f"input/cloud_files/{haze_folder}"
    if not os.path.isdir(dirstr):
        os.mkdir(dirstr)

    # Write each of the files.
    for file, content in files.items():
        with open(f"{dirstr}/{file}", "w+") as outfile:
            outfile.write(content)

def build_cloudfile_string(data: dict):
    '''Ensures proper formatting for cloud files.'''
    order = [
            "wavelengths (mu)",
            "size parameter",
            "extinction cross section (cm^2)",
            "scattering cross section (cm^2)",
            "absorption cross section (cm^2)",
            "single scattering albedo",
            "asymmetry parameter"
            ]

    lines = ['#' + '\t'.join(order)]

    for i in range(len(data['wavelengths (mu)'])):
        line = []
        for key in order:
            line.append(f"{data[key][i]:1.9e}")

        lines.append('\t'.join(line))

    return '\n'.join(lines)

def main():
    data = get_all_data(haze_files)

    helios_data = gen_helios_data(data)

    make_helios_files(new_haze_folder_name, helios_data)

if __name__ == "__main__":
    main()
