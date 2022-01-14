import numpy as np
import pandas as pd
import csv

from more_itertools import unique_everseen

def pc_id():
    with open('prediction_yyy_invert.txt') as data_file:
        data_array = data_file.readlines()

    data_array = np.array(data_array)
    data_array = np.char.split(data_array)

    reduced_tce_list = []

    for tce in data_array:
        if float(tce[1]) > 0.4:
            reduced_tce_list.append(int(tce[0]))

    #Adding false negative results to code here
    reduced_tce_list.append(int('33807061'))
    reduced_tce_list.append(int('38586438'))
    reduced_tce_list.append(int('150637078'))

    reduced_tce_list.sort()
    reduced_tce_list = [str(x) for x in reduced_tce_list]

    return reduced_tce_list

def tic_filteredfile(list):
    tcefilterfile = open('tces_filter_file.txt', 'w')
    list=map(lambda x:x+'\n', list)
    tcefilterfile.writelines(list)
    tcefilterfile.close()

    print("Finished writing tces_filter_file.txt to folder.")

def main():
    true_list = pc_id()
    tic_filteredfile(true_list)

    tcesfile = pd.read_csv('tess_s39data_edit.csv')

    header=['', 'tic_id', 'toi_id', 'Disposition', 'RA', 'Dec', 'Tmag', 'Epoc', 'Period', 'Duration',
            'Transit Depth', 'Sectors', 'camera', 'ccd', 'star_rad', 'star_mass', 'teff', 'logg',
            'SN', 'Qingress'
            ]

    newfile = 'tess_s39data_pclist.csv'
    tlistid=0

    validtcedata=[]
    for x in range(len(tcesfile['tic_id'])):
        for y in range(len(true_list)):
            if tcesfile['tic_id'][x] == int(true_list[y]):
                validtcedata.append(tcesfile.iloc[x])

    with open(newfile, 'w', newline='') as clean_data:
        writer = csv.writer(clean_data)
        writer.writerow(header)
        writer.writerows(validtcedata)

main()
