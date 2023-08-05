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
            fill_value=(f_cloud_orig[-1], f_cloud_orig[0]),
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

        # Need to convert radii to microns
        hzradii = 1e5 * hzradii

        int_hzradii = itp.interp1d(
            quant.p_lay, hzradii, fill_value=1e-50, bounds_error=False
        )(quant.p_int)

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
