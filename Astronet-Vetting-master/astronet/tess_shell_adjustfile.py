"""
Rewrites shell file commands to download target pixel files only for relevant
targets on planetary candidate list.
"""

import argparse

import pandas as pd
import numpy as np


def main(infile, inputshell, outputshell):
    """
    Args:
        infile: Name of planetary candidate .csv file
        inputshell: Name of .sh file to rewrite
        outputshell: Name of output .sh file with relevant commmands

    Throws:
        None

    Returns:
        A .sh file with relevant commands to download target pixel files
    """

    #Read TIC ID from .csv file containing planetary candidates
    pc = pd.read_csv(infile, index_col = 0)
    ticids = np.array(pc['tic_id'])

    #Read target pixel .sh file
    shell_file = str(inputshell)
    with open(shell_file) as sf:
        commands = sf.read().splitlines()

    #Parse commands for downloading target pixel files for TIC IDs listed on
    #.csv file.
    necessary_commands = np.array([])
    for i in range(len(ticids)):
        necessary_commands = np.append(necessary_commands,
                                        list(filter(lambda line: str(ticids[i])
                                        in line, commands)))

    #Write relevant shell commands to a new shell script.
    new_shell_file = str(outputshell)
    with open(new_shell_file, 'w', newline='') as new_file:
        new_file.writelines("%s\n" % n for n in necessary_commands)
    new_file.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv_file",
        type=str,
        required=True,
        help="CSV file containing the TESS TCE table for planetary candidates."
    )

    parser.add_argument(
        "--input_sh_file",
        type=str,
        required=True,
        help="Name of shell file containing the commands to download target"
             "pixel files for relevant sector."
    )

    parser.add_argument(
        "--output_sh_file",
        type=str,
        required=True,
        help="Name of shell file which contains commands to download target"
             "pixel files for relevant targets only."
    )

    FLAGS = parser.parse_args()

    main(FLAGS.input_csv_file, FLAGS.input_sh_file, FLAGS.output_sh_file)
    print("Completed!")
