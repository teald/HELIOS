'''Converts an out.aersol or out.hcaer-style file from the Atmos photochemistry
model into HELIOS.
'''
import numpy as np
import astropy.units as u


# inputs
folder = "../gj1214b/gj1214b_1000x_solar"
haze_filename = "out.hcaer"
ptz_profile_file = "PTZ_mixingratios_out.dist"

output_file = "input/haze_profile.dat"

# Process inputs
folder = folder.rstrip('/')
haze_filename = f"{folder}/{haze_filename}"
ptz_profile_file = f"{folder}/{ptz_profile_file}"


# Get all the vertical profiles
def load_ptz_dict(filename: str) -> dict:
    with open(filename, 'r') as infile:
        cols = infile.readline().strip().split()
        data = {c: [] for c in cols}

        for line in infile:
            row = [x for x in line.split()]

            # Need to do this one at a time for overflow fortran values.
            for i, d in enumerate(row):
                try:
                    row[i] = float(d)

                except ValueError:
                    # This is a very large or very small number
                    row[i] = 1e-99 if '-' in d else 1e+99

            for c, d in zip(cols, row):
                data[c].append(d)

    data = {c: np.array(d) for c, d in data.items()}
    return data


profile_data = load_ptz_dict(ptz_profile_file)


# Get the haze data
# columns for out.hcaer are:
#   + pressure
#   + temperature
#   + altitude
#   + ndens aer1
#   ...
#   + ndens aeri
#   + rpar aer1
#   ...
#   + rpar aeri
def get_haze_data(filename: str) -> dict:
    data = np.genfromtxt(filename)

    cols = ['pressure', 'temperature', 'altitude']

    remaining_cols = data.shape[1] - len(cols)

    if remaining_cols % 2 == 1:
        raise ValueError(f"Expected an odd number of columns in file "
                         f"{filename}\n\t(got {data.shape[1]})"
                         )

    n_hazes = remaining_cols // 2

    cols += [f"density{i}" for i in range(n_hazes)]
    cols += [f"radius{i}" for i in range(n_hazes)]

    datadict = {}

    for i, name in enumerate(cols):
        datadict[name] = data[:, i]

    return datadict

haze_data = get_haze_data(haze_filename)

# HELIOS wants this, all in CGS, in columns:
# pressure number_density particle_radius
# FOR THIS TESTING, just doing the first haze. BE AWARE TKREFACTOR.
def dump_haze_file(filename: str, profile_data: dict, haze_data: dict):
    '''Outputs the haze input file in the format expected by HELIOS.'''
    header = "# pressure number_density particle_radius (cgs)"
    pressure = profile_data['PRESS']
    number_density = haze_data['density0']  # TKREFACTOR this is hardcoded
    particle_radius = haze_data['radius0']  # TKREFACTOR this is hardcoded

    pressure = (pressure * u.bar).cgs.value
    number_density = (number_density * u.cm**-3).cgs.value
    particule_radius = (particle_radius * u.micron).cgs.value

    datastr = []

    for p, n, r in zip(pressure, number_density, particle_radius):
        datastr.append(f"{p:1.6e}    {n:1.6e}    {r:1.6e}")

    with open(filename, 'w+') as outfile:
        outfile.write(header + '\n')
        outfile.write('\n'.join(datastr))

dump_haze_file(output_file, profile_data, haze_data)


import pdb; pdb.set_trace()
