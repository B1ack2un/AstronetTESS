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

"""Functions for creating catalog of TESS TCEs, containing TIC IDs, BLS information, stellar params, sector, camera and
CCD number."""

import numpy as np
import pandas as pd
import os
from astroquery.mast import Catalogs
from astroquery.mast import Tesscut
import argparse
import logging
import sys


parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_worker_processes",
    type=int,
    default=1,
    help="Number of subprocesses for processing the TCEs in parallel.")

parser.add_argument(
    '--input',
    nargs='+',
    help='CSV file(s) containing the TCE table(s) for training. Must contain '
         'columns: TIC, TCE planet number, final disposition',
    required=True)

parser.add_argument(
    '--tcestatfile',
    nargs='+',
    help='CSV file containing the TCE statistics from MAST for analysis.',
    required=True)

parser.add_argument(
    "--base_dir",
    type=str,
    default='/pdo/users/yuliang/',
    help="Directory where TCE lists are located, and where the output will be saved.")

parser.add_argument(
    "--out_name",
    type=str,
    default='tces.csv',
    help="Name of output file.")


def read_tce_file(infile):
    infile=infile[0]
    filename = os.path.join(str(FLAGS.base_dir), str(infile))
    bls_info = pd.read_csv(filename, index_col=1)
    return bls_info

def star_query(tic):
    """

    :param tic:  TIC of the target star. May be an int or a possibly zero-
          padded string.
    :param ra: RA of target star. Float.
    :param dec: Dec of target star. Float.

    :return: dict containing stellar parameters.
    """

    field_list = ["id", "mass", "ra", "dec", "rad", "e_rad", "teff", "e_teff", "logg", "e_logg", "tmag", "e_tmag"]
    result = Catalogs.query_object("TIC "+str(tic), radius=".02 deg", catalog="TIC")

    dtype = [(field_list[k], float) for k in range(len(field_list))]
    t = np.array(result)

    starparam = {}
    starparam["mass"] = result["mass"][0]
    starparam["rad"] = result["rad"][0]
    starparam["e_rad"] = result["e_rad"][0]
    starparam["teff"] = result["Teff"][0]
    starparam["e_teff"] = result["e_Teff"][0]
    starparam["logg"] = result["logg"][0]
    starparam["e_logg"] = result["e_logg"][0]
    starparam["tmag"] = result["Tmag"][0]
    starparam["e_tmag"] = result["e_Tmag"][0]
    starparam["ra"] = result["ra"][0]
    starparam["dec"] = result["dec"][0]

    result = Catalogs.query_object(str(starparam["ra"]) + " " + str(starparam["dec"]), catalog="GAIA")
    if result is not None:
        if not np.isnan(float(result["radius_val"][0])):
            starparam["rad"] = float(result["radius_val"][0])
            starparam["e_rad"] = np.sqrt(
                float(result["radius_percentile_lower"][0]) * float(result["radius_percentile_upper"][0]))
        if not np.isnan(float(result["teff_val"][0])):
            starparam["teff"] = float(result["teff_val"][0])
            starparam["e_teff"] = np.sqrt(
                float(result["teff_percentile_lower"][0]) * float(result["teff_percentile_upper"][0]))

    return starparam

def _process_tce(tce_table):
    """
    Find camera, ccd number and stellar params of target given TIC and sector, using catalogs under /scratch/tmp
    :param tce_table: Pandas dataframe containing TIC ID, RA and Dec
    :param sector: Int, sector number of data
    :return: tce with stellar params, camera and ccd columns filled
    """

    if FLAGS.num_worker_processes > 1:
        current = multiprocessing.current_process()

    total = len(tce_table)
    tce_table['camera'] = 0
    tce_table['ccd'] = 0
    tce_table['star_rad'] = np.nan
    tce_table['star_mass'] = np.nan
    tce_table['teff'] = np.nan
    tce_table['logg'] = np.nan
    tce_table['SN'] = np.nan
    tce_table['Qingress'] = np.nan

    cnt = 0

    bls = read_tce_file(FLAGS.tcestatfile)

    for index, tce in tce_table.iterrows():
        cnt += 1
        if FLAGS.num_worker_processes == 1:
            if cnt % 10 == 0:
                print('Processed %s/%s TCEs' % (cnt, total))
        else:
            if cnt % 10 == 0:
                logger.info('Process %s: processing TCE %s/%s ' %(current.name, cnt, total))

        sect_table = Tesscut.get_sectors(objectname="TIC "+str(tce_table['tic_id'][index]))
        starparam = star_query(tce_table['tic_id'][index])

        tce_table.camera.loc[index] = sect_table["camera"][0]
        tce_table.ccd.loc[index] = sect_table["ccd"][0]

        if (starparam["tmag"] <= 12) and (tce_table.camera.loc[index] > 0):
            tce_table.Epoc.loc[index] = bls['tce_time0bt'].iloc[index]
            tce_table.Period.loc[index] = bls['tce_period'].iloc[index]
            tce_table.Duration.loc[index] = bls['tce_duration'].iloc[index]
            tce_table['Transit Depth'].loc[index] = bls['tce_depth'].iloc[index]
            tce_table.SN.loc[index] = bls['tce_model_snr'].iloc[index]
            tce_table.Qingress.loc[index] = bls['tce_ingress'].iloc[index]
            tce_table.star_rad.loc[index] = starparam['rad']
            tce_table.star_mass.loc[index] = starparam['mass']
            tce_table.teff.loc[index] = starparam['teff']
            tce_table.logg.loc[index] = starparam['logg']
            if np.isnan(tce['RA']):
                tce_table.RA.loc[index] = starparam['ra']
                tce_table.Dec.loc[index] = starparam['dec']
                tce_table.Tmag.loc[index] = starparam['tmag']

    tce_table = tce_table[np.isfinite(tce_table['Period'])]
    return tce_table


def parallelize(data):
    partitions = FLAGS.num_worker_processes
    data_split = np.array_split(data, partitions)

    pool = multiprocessing.Pool(processes=partitions)
    df = pd.concat(pool.map(_process_tce, data_split))
    pool.close()
    pool.join()

    return df


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()

    # tce_table_names = ['../sector-1-earlylook.csv', '../sector-2-bright.csv', '../sector-3-01.csv',
                       # '../sector-3-02.csv']
    tce_table_names = [os.path.join(FLAGS.base_dir, csv) for csv in FLAGS.input]

    tce_table = pd.DataFrame()
    for table in tce_table_names:
        tces = pd.read_csv(table, header=0, usecols=[1,2,3,4,5,6,8,10,12,14,16])
        tce_table = pd.concat([tce_table, tces], ignore_index=True)

    if FLAGS.num_worker_processes == 1:
        tce_table = _process_tce(tce_table)
    else:
        logger = multiprocessing.get_logger()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.setLevel(logging.INFO)
        logger.info('Process started')
        tce_table = parallelize(tce_table)
    tce_table.to_csv(os.path.join(FLAGS.base_dir, FLAGS.out_name))
