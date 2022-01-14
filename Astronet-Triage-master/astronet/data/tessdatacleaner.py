"""
Eliminates undefined values for Transit Depth column in TESS TCES csv files and writes acceptable rows to a new file.
"""

import argparse
import csv


def main(infile, outfile):
    """
    Args:
        infile: Name of the input .csv file
        outfile: Name of the output .csv file

    Throws:
        None

    Returns:
        A .csv file containing TESS TCE data.
    """


    #Get nonempty rows.
    with open(infile, "r") as inp:
        rows = []
        for row in csv.reader(inp):
            if row[9] != "":
                rows.append(row)

    #Write nonempty rows in new file.
    with open(outfile, 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerows(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv_file",
        type=str,
        required=True,
        help="CSV file containing the TESS TCE table. Must contain "
        "columns: row_id, tic_id, toi_id, Period, Duration, Transit Depth, "
        "Epoc (t0)."
    )

    parser.add_argument(
        "--output_csv_file",
        type=str,
        required=True,
        help="Name of CSV file containing nonempty Transit Depth values."
    )

    FLAGS = parser.parse_args()

    main(FLAGS.input_csv_file, FLAGS.output_csv_file)
    print("Completed!")
