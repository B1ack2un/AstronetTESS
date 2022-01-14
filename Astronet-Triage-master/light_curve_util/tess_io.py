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

from light_curve_util import util
from tensorflow import io


def tess_filenames(tic,
                     base_dir='D:/ExoplanetMLFiles/Alt_Exoplanet/TESSExoplanetData/Astronet-Triage-master/Astronet-Triage-master/astronet/tess',
                     sector=43,
                     injected=False,
                     inject_dir='/sector-43',
                     check_existence=True):
    """Returns the light curve filename for a TESS target star.

    Args:
      tic: TIC of the target star. May be an int or a possibly zero-
          padded string.
      base_dir: Base directory containing Kepler data.
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
        dir = os.path.join(base_dir, 'sector-' + str(sector))
        base_name = "tess2021258175143-s00"+str(sector)+"-"+str(tic)+"-0214-s_lc.fits"
        filename = os.path.join(dir, base_name)
    else:
        filename = os.path.join(inject_dir, tic + '.lc')

    if check_existence or io.gfile.exists(filename):
        return filename
    return


def read_tess_light_curve(filename, flux_key='KSPMagnitude', invert=True):
    """Reads time and flux measurements for a TESS target star.

    Args:
      filename: str name of .fits file containing time and flux measurements.
      invert: Whether to reflect flux values around the median flux value. This is
        performed separately for each .fits file.

    Returns:
      time: The time values of the light curve.
      flux: The flux values of the light curve.
    """
    with fits.open(io.gfile.GFile(filename, mode="rb")) as hdu_list:
        time = hdu_list[1].data['TIME']
        flux = hdu_list[1].data['PDCSAP_FLUX']

        if 'QUALITY' in fits.getdata(filename, ext=1).columns.names:
            quality_flag = np.where(np.array(hdu_list[1].data['QUALITY']) == 0)

            # Remove outliers
            time = time[quality_flag]
            flux = flux[quality_flag]

            # Remove NaN flux values.
            valid_indices = np.where(np.isfinite(flux))
            time = time[valid_indices]
            flux = flux[valid_indices]

    if invert:
        flux *= -1

    return time, flux
