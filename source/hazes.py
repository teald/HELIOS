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
import astropy.constants as const
from scipy import interpolate as itp
from scipy.interpolate import interp2d as interpolate
import json
from source import clouds, quantities
from source import tools as tls
from source import read


class Haze(clouds.Cloud):
    """class that reads in haze parameters and data to be used in the HELIOS
    code
    """

    def __init__(self):
        # Parent class does not need any arguments, either.
        super().__init__()

    def haze_pre_processing(self, quant):
        # If haze processing isn't turned on, just return
        if not self.haze_turned_on:
            return

        # just putting a reference to quant into the object.
        if "quant" in self.__dict__:
            raise ValueError(
                "Assigning attribute 'quant' to this Haze object would "
                "override an existing 'quant' attribute."
            )

        self.quant = quant

        opac_dir = self.haze_opacity_data_dir
        profile_file = self.haze_profile_data_file

        self.get_profile_data(profile_file)

        self.convert_density_to_mixing_ratio()

        self.calc_weighted_cross_sections_with_pdf_and_interpolate_wavelengths()

        self.add_individual_hazes_to_total()

    def get_profile_data(self, profile_file):
        """Retrieves and stores the vertical haze density and radius profiles
        to the Haze object.
        """
        helios_press = self.quant.p_lay

        out = read.Read.read_haze_vert_density_and_interpolate_to_helios_press_grid(
            profile_file, helios_press
        )

        self.haze_density, self.haze_radius = out

        return self.haze_density, self.haze_radius

    def convert_density_to_mixing_ratio(self):
        """Since helios' calculations generally use mixing ratios, need
        to convert this it a mixing ratio profile.

        Just doing this by taking the
            haze density / (total density + haze density)
        """
        quant = self.quant

        # Get the total density of the atmosphere.
        ndens = (quant.p_lay / (const.k_B.cgs * quant.T_lay[1:])).cgs.value

        self.haze_mixing_ratio = self.haze_density / ndens
        f_cloud_orig = self.haze_mixing_ratio

        log_press_orig = [np.log10(p) for p in quant.p_lay]
        log_p_lay = [np.log10(p) for p in quant.p_lay]
        log_p_int = [np.log10(p) for p in quant.p_int]

        cloud_interpol_function = itp.interp1d(
            log_press_orig,
            f_cloud_orig,
            kind="linear",
            bounds_error=False,
            fill_value=np.amin(f_cloud_orig)
        )

        self.f_one_cloud_lay = cloud_interpol_function(log_p_lay)
        self.f_one_cloud_int = cloud_interpol_function(log_p_int)

    def calc_weighted_cross_sections_with_pdf_and_interpolate_wavelengths(
        self,
    ):
        """Overrides the clouds.Cloud object's method of the same name to
        account for vertical haze profiles.
        """
        quant = self.quant

        # reset for each cloud
        weighted_abs_cross_mie = []
        weighted_scat_cross_mie = []
        weighted_g_0_mie = []

        # WARNING: hardcoded particle sizes that match source/clouds.py
        # Uses microns, NOT CGS
        r_values = 10 ** np.arange(-2, 3.1, 0.1)
        delta_r = r_values * (10**0.05 - 10**-0.05)

        # get lambda values
        filename = f"{self.haze_opacity_data_dir}/r{r_values[0]:.6f}.dat"

        self.lambda_mie = self.read_mie_file(filename)[0]

        abs_cross_per_r = np.zeros((len(r_values), len(self.lambda_mie)))
        scat_cross_per_r = np.zeros((len(r_values), len(self.lambda_mie)))
        g_0_per_r = np.zeros((len(r_values), len(self.lambda_mie)))

        # Reading in the cross sections for *all* radii here.
        for r, radius in enumerate(r_values):  # range(len(r_values)):
            filename = f"{self.haze_opacity_data_dir}/r{radius:.6f}.dat"

            data = self.read_mie_file(filename)

            scat_cross_per_r[r, :] = data[1]
            abs_cross_per_r[r, :] = data[2]
            g_0_per_r[r, :] = data[3]

        # Create interpolators for each of these.
        sxsec_fxn = interpolate(
            self.lambda_mie, r_values, scat_cross_per_r, fill_value=1e-50
        )

        axsec_fxn = interpolate(
            self.lambda_mie, r_values, abs_cross_per_r, fill_value=1e-50
        )

        g_0_fxn = interpolate(
            self.lambda_mie, r_values, g_0_per_r, fill_value=1e-50
        )

        lambda_helios = quant.opac_wave
        lambda_helios_int = quant.opac_interwave

        hzradii, hzdensity = self.haze_radius, self.haze_density

        int_hzradii = itp.interp1d(
            quant.p_lay, hzradii, fill_value=1e-50, bounds_error=False
        )(quant.p_int)

        # for l in range(len(self.lambda_mie)):
        #     abs_cross = sum(abs_cross_per_r[:, l] * pdf * delta_r)
        #     scat_cross = sum(scat_cross_per_r[:, l] * pdf * delta_r)
        #     g_0 = sum(scat_cross_per_r[:, l] * pdf * delta_r)

        #     abs_cross = axsec_fxn(lambda_helios, radius)

        #     weighted_abs_cross_mie.append(abs_cross)
        #     weighted_scat_cross_mie.append(scat_cross)
        #     weighted_g_0_mie.append(g_0)

        self.vert_abs_cross_mie = axsec_fxn(quant.opac_wave, hzradii)
        self.vert_scat_cross_mie = sxsec_fxn(quant.opac_wave, hzradii)
        self.vert_g_0_mie = g_0_fxn(quant.opac_wave, hzradii)

        self.vert_abs_cross_mie_int = axsec_fxn(
            quant.opac_interwave, int_hzradii
        )
        self.vert_scat_cross_mie_int = sxsec_fxn(
            quant.opac_interwave, int_hzradii
        )
        self.vert_g_0_mie_int = g_0_fxn(quant.opac_interwave, int_hzradii)

        # # interpolate to HELIOS wavelength grid
        # self.abs_cross_one_cloud = tls.convert_spectrum(
        #         self.lambda_mie,
        #         weighted_abs_cross_mie,
        #         quant.opac_wave,
        #         int_lambda=quant.opac_interwave,
        #         type='log'
        #         )

        # self.scat_cross_one_cloud = tls.convert_spectrum(
        #         self.lambda_mie,
        #         weighted_scat_cross_mie,
        #         quant.opac_wave,
        #         int_lambda=quant.opac_interwave,
        #         type='log'
        #         )

        # self.g_0_one_cloud = tls.convert_spectrum(
        #         self.lambda_mie,
        #         weighted_g_0_mie,
        #         quant.opac_wave,
        #         int_lambda=quant.opac_interwave,
        #         type='linear'
        #         )

    def add_individual_hazes_to_total(self):
        # For now, folding this into the cloud calculation.
        quant = self.quant

        for i in range(quant.nlayer):
            quant.f_all_clouds_lay[i] += self.f_one_cloud_lay[i]
            abs_xsec = self.vert_abs_cross_mie[i, :]
            scat_xsec = self.vert_scat_cross_mie[i, :]
            g_0 = self.vert_g_0_mie[i, :]

            for j in range(quant.nbin):
                quant.abs_cross_all_clouds_lay[j + quant.nbin * i] += (
                    self.f_one_cloud_lay[i] * abs_xsec[j]
                )
                quant.scat_cross_all_clouds_lay[j + quant.nbin * i] += (
                    self.f_one_cloud_lay[i] * scat_xsec[j]
                )
                quant.g_0_all_clouds_lay[j + quant.nbin * i] += (
                    g_0[j] * self.f_one_cloud_lay[i] * scat_xsec[j]
                )

        if quant.iso == 0:
            for i in range(quant.ninterface):
                quant.f_all_clouds_int[i] += self.f_one_cloud_int[i]
                abs_xsec = self.vert_abs_cross_mie_int[i, :]
                scat_xsec = self.vert_scat_cross_mie_int[i, :]
                g_0 = self.vert_g_0_mie_int[i, :]

                for x in range(quant.nbin):

                    quant.abs_cross_all_clouds_int[x + quant.nbin * i] += (
                        self.f_one_cloud_int[i] * abs_xsec[x]
                    )

                    quant.scat_cross_all_clouds_int[x + quant.nbin * i] += (
                        self.f_one_cloud_int[i] * scat_xsec[x]
                    )

                    quant.g_0_all_clouds_int[x + quant.nbin * i] += (
                        g_0[x] * self.f_one_cloud_int[i] * scat_xsec[x]
                    )

    # # The default input files for when it's instantiated raw.
    # _default_haze_profiles = "input/haze_profile.dat"
    # _default_haze_datafiles = "input/soot_haze.json"

    # # Note: not initializing the Cloud class itself; we don't want the data or
    # # attrs, just the methods.
    # def __init__(self,
    #              haze_profiles:  str = '',
    #              haze_datafiles: str = ''
    #              ):
    #     '''Initiailizes the Haze object with profiles and datafiles.

    #     This does NOT initialize the inherited Cloud object; we only want some
    #     of its functionality.
    #     '''
    #     if not haze_profiles:
    #         haze_profiles = self._default_haze_profiles

    #     if not haze_datafiles:
    #         haze_datafiles = self._default_haze_datafiles

    #     self.profile_input = haze_profiles
    #     self.haze_datafiles = haze_datafiles

    #     # Retrieve the data
    #     self.read_haze_data()
    #     self.read_profile_data()

    # def haze_pre_processing(self, quant: quantities.Store):
    #     '''This goes through a similar process to what the clouds.Cloud object
    #     goes through during pre-processing, but without some of the
    #     cloud-specific handling.
    #     '''
    #     # This protects issues that might arise from clouds not pre-processing
    #     # (though they should still be preprocessed as of right now, could be
    #     # unexpected behavior...
    #     if quant.f_all_clouds_lay is None:
    #         quant.f_all_clouds_lay = np.zeros(quant.nlayer)
    #         quant.f_all_clouds_int = np.zeros(quant.ninterface)

    #         quant.abs_cross_all_clouds_lay = np.zeros(quant.nlayer * quant.nbin)
    #         quant.abs_cross_all_clouds_int = np.zeros(quant.ninterface * quant.nbin)

    #         quant.scat_cross_all_clouds_lay = np.zeros(quant.nlayer * quant.nbin)
    #         quant.scat_cross_all_clouds_int = np.zeros(quant.ninterface * quant.nbin)

    #         quant.g_0_all_clouds_lay = np.zeros(quant.nlayer * quant.nbin)
    #         quant.g_0_all_clouds_int = np.zeros(quant.ninterface * quant.nbin)

    #     # Calculate the absorption and scattering cross sections.

    #     # Process arrays to look like cloud inputs.

    # @property
    # def input_files(self):
    #     return {
    #             'profiles': self.profile_input,
    #             'cross sections': self.haze_datafiles
    #             }

    # def read_profile_data(self):
    #     '''Read in the pressure, number density, radius profile.

    #     Remember that the units are pure cgs.
    #     '''
    #     self.pressure = []
    #     self.number_density = []
    #     self.radius_profile = []

    #     with open(self.profile_input, 'r') as infile:
    #         for line in infile:
    #             if '#' in line:
    #                 line = line[:line.index('#')].strip()

    #             if not line:
    #                 continue

    #             cols = line.split()
    #             self.pressure.append(cols[0])
    #             self.number_density.append(cols[1])
    #             self.radius_profile.append(cols[2])

    #     self.pressure = np.array(self.pressure, dtype=float)
    #     self.number_density = np.array(self.number_density, dtype=float)
    #     self.radius_profile = np.array(self.radius_profile, dtype=float)

    # def read_haze_data(self):
    #     '''Read in data about the optical properties of the haze particles.

    #     Remember that the units are all pure cgs.
    #     '''
    #     # This is hardcoded using .json, which isn't ideal since the community
    #     # generally uses different formating.
    #     with open(self.haze_datafiles, 'r') as infile:
    #         data = json.load(infile)

    #     # Each json column gets its own space for now.
    #     self.particle_radius = []
    #     self.monomer_radius = []
    #     self.df = []
    #     self.nmon = []
    #     self.wavelength = []
    #     self.q_extinction = []
    #     self.q_scattering = []
    #     self.assym_param = []

    #     for _, d in data.items():
    #         self.particle_radius.append(d['PARTICLE RADIUS'])
    #         self.monomer_radius.append(d['MONOMER RADIUS'])
    #         self.df.append(d['DF'])
    #         self.nmon.append(d['NMON'])
    #         self.wavelength.append(d['WAVELENGTH'])
    #         self.q_extinction.append(d['Q_EXTINCTION'])
    #         self.q_scattering.append(d['Q_SCATTERING'])
    #         self.assym_param.append(d['ASYMMETRY PARAMETER'])

    #     self.particle_radius = np.array(self.particle_radius, dtype=float)
    #     self.monomer_radius = np.array(self.monomer_radius, dtype=float)
    #     self.df = np.array(self.df, dtype=float)
    #     self.nmon = np.array(self.nmon, dtype=float)
    #     self.wavelength = np.array(self.wavelength, dtype=float)
    #     self.q_extinction = np.array(self.q_extinction, dtype=float)
    #     self.q_scattering = np.array(self.q_scattering, dtype=float)
    #     self.assym_param = np.array(self.assym_param, dtype=float)

    # @property
    # def n_wl(self) -> int:
    #     return self.wavelength.shape[1]

    # @property
    # def n_radii(self) -> int:
    #     return self.particle_radius.shape[0]

    # @property
    # def nz(self) -> int:
    #     return self.pressure.shape[0]
