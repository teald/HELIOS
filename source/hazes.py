# ==============================================================================
# Module adding aerosol extinction to HELIOS
# Copyright (C) 2020 - 2022 Matej Malik
# ==============================================================================
# This file is part of HELIOS.
#
#     HELIOS is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     HELIOS is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You find a copy of the GNU General Public License in the main
#     HELIOS directory under <license.txt>. If not, see
#     <http://www.gnu.org/licenses/>.
# ==============================================================================
import numpy as np
from scipy.interpolate import interpn as interpolate
import json
from source import clouds


class Haze(clouds.Cloud):
    """ class that reads in haze parameters and data to be used in the HELIOS
    code
    """
    # The default input files for when it's instantiated raw.
    _default_haze_profiles = "input/haze_profile.dat"
    _default_haze_datafiles = "input/soot_haze.json"

    # Note: not initializing the Cloud class itself; we don't want the data or
    # attrs, just the methods.
    def __init__(self,
                 haze_profiles:  str | None = None,
                 haze_datafiles: str | None = None
                 ):
        '''Initiailizes the Haze object with profiles and datafiles.

        This does NOT initialize the inherited Cloud object; we only want some
        of its functionality.
        '''
        if not haze_profiles:
            haze_profiles = self._default_haze_profile

        if not haze_datafiles:
            haze_datafiles = self._default_haze_datafiles

        self.profile_input = haze_profiles
        self.haze_datafiles = haze_datafiles

        # Retrieve the data
        self.read_haze_data()
        self.read_profile_data()

    @property
    def input_files(self):
        return {
                'profiles': self.profile_input,
                'cross sections': self.haze_datafiles
                }

    def read_profile_data(self):
        '''Read in the pressure, number density, radius profile'''
        self.pressure = []
        self.number_density = []
        self.radius_profile = []

        with open(self.profile_input, 'r') as infile:
            for line in infile:
                if '#' in line:
                    line = line[:line.index('#')].strip()

                if not line:
                    continue

                cols = line.split()
                self.pressure.append(cols[0])
                self.number_density.append(cols[1])
                self.radius_profile.append(cols[2])

        self.pressure = np.array(self.pressure)
        self.number_density = np.array(self.number_density)
        self.radius_profile = np.array(self.radius_profile)

    def read_haze_data(self):
        '''Read in data about the optical properties of the haze particles.'''
        # This is hardcoded using .json, which isn't ideal since the community
        # generally uses different formating.
        with open(self.haze_datafiles, 'r') as infile:
            data = json.load(infile)

        # Each json column gets its own space for now.
        self.particle_radius = []
        self.monomer_radius = []
        self.df = []
        self.nmon = []
        self.wavelength = []
        self.q_extinction = []
        self.q_scattering = []
        self.assym_param = []

        for _, d in data.items():
            self.particle_radius.append(d['PARTICLE RADIUS'])
            self.monomer_radius.append(d['MONOMER RADIUS'])
            self.df.append(d['DF'])
            self.nmon.append(d['NMON'])
            self.wavelength.append(d['WAVELENGTH'])
            self.q_extinction.append(d['Q_EXCTINTION'])
            self.q_scattering.append(d['Q_SCATTERING'])
            self.assym_param.append(d['ASYMMETRY PARAMETER'])

        self.particle_radius = np.array(self.particle_radius)
        self.monomer_radius = np.array(self.monomer_radius)
        self.df = np.array(self.df)
        self.nmon = np.array(self.nmon)
        self.wavelength = np.array(self.wavelength)
        self.q_extinction = np.array(self.q_extinction)
        self.q_scattering = np.array(self.q_scattering)
        self.assym_param = np.array(self.assym_param)

    @property
    def helios_spec(self):
        '''Interpolates the data onto the HELIOS wavelength grid, returning a
        new Haze object.
        '''
        raise NotImplementedError

    @property
    def n_wl(self) -> int:
        return self.wavelength.shape[1]

    @property
    def n_radii(self) -> int:
        return self.particle_radius.shape[0]

    @property
    def nz(self) -> int:
        return self.pressure.shape[0]
