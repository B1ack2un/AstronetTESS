# Copyright 2018 Liang Yu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for reading TESS data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
from astropy.io import fits
import numpy as np

from tensorflow import io


def define_aperture(ap, get_optval, app_size):
    ap_c_rowpos = ap.shape[0]//2
    ap_c_colpos = ap.shape[1]//2
    rangeval = app_size//2

    if get_optval:
        bitmask_arr = np.array(list(map(np.binary_repr, ap.flatten())))
        opttest = np.array([binval[-2] for binval in bitmask_arr])
        bitmask2_set = np.where(opttest == '1', 1, 0)
        bitmask2_set = bitmask2_set.reshape(ap.shape)

    if not get_optval:
        bitmask2_set = np.zeros(ap.shape)
        bitmask2_set[ap_c_rowpos-rangeval:ap_c_rowpos+rangeval+1, ap_c_colpos-rangeval: ap_c_colpos+rangeval+1] = 1

    return bitmask2_set

def flux_aperture(app_array, iflux_arr):
    app_array = app_array.flatten()
    iflux_arr = iflux_arr.flatten()
    flux_array = np.where(app_array == 1, iflux_arr, 0)
    flux_val = np.sum(flux_array, axis=None)

    return flux_val

def tess_filenames(tic,
                     base_dir='D:/ExoplanetMLFiles/Alt_Exoplanet/TESSExoplanetData/Astronet-Vetting-master/Astronet-Vetting-master/astronet/tess',
                     sector=43,
                     injected=False,
                     inject_dir='/sector-43',
                     check_existence=True):
    """Returns the light curve filename for a TESS target star.

    Args:
      tic: TIC of the target star. May be an int or a possibly zero-
          padded string.
      base_dir: Base directory containing TESS data.
      sector: Int, sector number of data.
      cam: Int, camera number of data.
      ccd: Int, CCD number of data.
      injected: Bool, whether target also has a light curve with injected planets.
      injected_dir: Directory containing light curves with injected transits.
      check_existence: If True, only return filenames corresponding to files that
          exist.

    Returns:
      filename for given TIC.
    """
    tic = str(tic).rjust(16, '0')

    if not injected:
        # modify this as needed
        #bdir = os.path.join(base_dir, 'sector-' + str(sector)+'-tpfiles')
        base_name = "tess2021258175143-s00"+str(sector)+"-"+str(tic)+"-0214-s_tp.fits"
        filename = os.path.join(base_dir, base_name)
    else:
        filename = os.path.join(inject_dir, tic + '.fits')

    if not check_existence or io.gfile.exists(filename):
        return filename
    return


def read_tess_light_curve(filename, flux_key='KSPMagnitude', invert=True):
    """Reads time and flux measurements for a TESS target star.

    Args:
      filename: str name of fits file containing light curve.
      flux_key: Key of fits column containing detrended flux.
      invert: Whether to invert the flux measurements by multiplying by -1.

    Returns:
      time: Numpy array; the time values of the light curve.
      mag: Numpy array corresponding to magnitudes at each time step (optimal aperture).
      mag_small: Numpy array corresponding to magnitudes at each time step (aperture).
    """
    with fits.open(io.gfile.GFile(filename, mode="rb")) as hdu_list:
        apgroup = hdu_list[2].data
        flux_array = hdu_list[1].data['FLUX']
        api = define_aperture(apgroup, True, 0)

        small_ap = 3
        big_ap = 5

        ap_small = define_aperture(apgroup, False, small_ap)
        ap_big = define_aperture(apgroup, False, big_ap)

        cad = flux_array.shape[0]
        mag = np.zeros(cad)
        mag_small = np.zeros(cad)
        mag_big = np.zeros(cad)

        time = np.array(hdu_list[1].data['TIME'])
        for d in range(cad):
            mag[d] = flux_aperture(api, flux_array[d,:,:])
            mag_small[d] = flux_aperture(ap_small, flux_array[d,:,:])
            mag_big[d] = flux_aperture(ap_big, flux_array[d,:,:])

        if 'QUALITY' in fits.getdata(filename, ext=1).columns.names:
            quality_flag = np.where(np.array(hdu_list[1].data['QUALITY']) == 0)

            # Remove outliers
            time = time[quality_flag]
            mag = mag[quality_flag]
            mag_small = mag_small[quality_flag]
            mag_big = mag_big[quality_flag]

            # Remove NaN flux values.
            valid_indices = np.where(np.isfinite(mag) & np.isfinite(mag_small) & np.isfinite(mag_big)
                                    & (mag < 50) & (mag_small < 50) & (mag_big < 50))
            if len(valid_indices) > 1:
                time = time[valid_indices]
                mag = mag[valid_indices]
                mag_small = mag_small[valid_indices]
                mag_big = mag_big[valid_indices]

        else:
            # manually remove sector 1 outliers
            bad = np.array([0, 1, 2, 31, 49, 88, 121, 152, 186, 188, 199,
                224, 225, 228, 241, 340, 359, 361, 463, 464, 465, 481,
                482, 483, 546, 547, 583, 584, 598, 599, 600, 601, 602,
                631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641,
                642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652,
                653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663,
                664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674,
                675, 676, 677, 678, 723, 726, 727, 730, 748, 749, 752,
                753, 754, 755, 756, 817, 819, 839, 853, 854, 855, 866,
                872, 873, 874, 875, 969, 971, 977, 987, 992, 993, 994,
                995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1005, 1006,
                1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017,
                1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028,
                1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039,
                1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050,
                1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061,
                1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072,
                1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083,
                1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094,
                1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1108,
                1112, 1113, 1114, 1141, 1175, 1180, 1183, 1191, 1193, 1195, 1196,
                1208, 1209, 1210, 1214, 1225, 1226, 1231, 1232, 1233, 1235, 1258,
                1278, 1279, 1280])

            bad = bad[bad < len(time)]
            mask = np.ones(len(time))
            mask[bad] = 0
            mask = mask.astype(bool)
            time = time[mask]
            mag = mag[mask]
            mag_small = mag_small[mask]
            mag_big = mag_big[mask]

            valid_indices = np.where(np.isfinite(mag) & np.isfinite(mag_small) & np.isfinite(mag_big))
            time = time[valid_indices]
            mag = mag[valid_indices]
            mag_small = mag_small[valid_indices]
            mag_big = mag_big[valid_indices]


    if invert:
        mag *= -1

    return time, mag, mag_small, mag_big
