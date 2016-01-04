# knmi-1-6126561.py <-- Assignment 2

# Python script for Basic Linux and Coding for AA homework 3 (week 2).
# Usage: python knmi-1-6126561.py
# TLR Halbesma, 6126561, september 9, 2014. Version 1.0; implemented

# NB All functions in this program require the entire dataset as input.
# This behavior could be altered such that main() subsets the dataset and feeds
# it to the functions. I might change this later on for aesthetic reasons.

import sys

INPUTFILE = './KNMI_20000101.txt'


def update_progress(progress):
    """
    https://stackoverflow.com/questions/3160699/python-progress-bar

    update_progress() : Displays or updates a console progress bar

    Accepts a float between 0 and 1. Any int will be converted to a float.
    A value under 0 represents a 'halt'.
    A value at 1 or bigger represents 100%
    """

    # Modify length to change the length of the progress bar
    length = 42
    status = ""

    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"

    block = int(round(length * progress))
    text = "\rread_dataset(): [{0}] {1:.2f}% {2}".\
        format("#" * block + "-" * (length - block), progress * 100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


def read_data(raw_knmi_data, end_line):  # <-- Assignment 5
    """
    Function to read KNMI dataset obtained from
    http://www.knmi.nl/climatology/daily_data/selection.cgi

    raw_knmi_data : list containing the entire dataset including header

    returns a list containing a list of all datapoints per station per date.
    """

    lines = []

    # Assignment 3
    # Skip first 85 lines because that is the header. Very ugly solution :-(
    # NB this breaks down if the header size changed. Be cautious!
    # Header: first 85 lines. Read end_line lines.
    for i in range(85, end_line):
        line = raw_knmi_data[i].strip().split(',')  # strip to remove '\n'
        cleaned_line = []
        for entry in line:
            # entry.strip() removes the whitespace around the datapoint.
            # entry.strip() returns False if len(x.strip()) == 0 (missing..)
            if entry.strip():
                cleaned_line.append(int(entry.strip()))
            else:
                # Assignment 4. Use None for missing data entries.
                cleaned_line.append(None)
        lines.append(cleaned_line)
        # Inform user of progress because loading is annoyingly long :-).
        update_progress(float(i) / end_line)

    return lines


def read_stationid(raw_knmi_data):  # <-- Assignment 8
    """
    Function to read KNMI dataset header, in particular the station info.

    raw_knmi_data : list containing the entire dataset including header.

    returns a list containing one list for each station.
    """

    all_stations = raw_knmi_data[3:41]
    all_stations_cleaned = list()

    for station in all_stations[1:]:  # First line = column info, remove.
        # Remove leading '#', split and unpack first four columns.
        stationID, lon, lat, alt = \
            station.replace(':', '').strip('#').split()[:4]
        # The name may contain spaces. Take sublist until last element.
        name = ' '.join(station.strip('#').split()[4:])

        # Create list containing one list for each station.
        # That list contains for each station 5 entries
        # Expliciet typecasts to sensible datatypes. Trivial choices.
        # StationID is a natural number, thus, an integer.
        # long, lat and alt are rational numbers, thus, floats.
        # The name consists of multiple characters, thus, stored in string.
        all_stations_cleaned.append([int(stationID), float(lon),
                                    float(lat), float(alt), str(name)])

    return all_stations_cleaned


def read_column_description(raw_knmi_data):  # <-- Assignment 9
    """
    Function to read KNMI datasetheader , in particular column descriptions

    raw_knmi_data : list containing the entire dataset including header.

    returns a dictionary mapping the column name to its description
    NB dictionaries may be printed in random order.
    """

    column_description = raw_knmi_data[42:82]
    column_description_cleaned = dict()

    for entry in column_description:
        abbreviation = ''.join(entry.strip('#').split('=')[:1]).strip()
        description = ' '.join(entry.strip('#').split('=')[1:])
        column_description_cleaned[abbreviation] = description

    return column_description_cleaned


def read_column_header(raw_knmi_data):  # <-- Assignment 10
    """
    Function to read header from KNMI dataset, in particular column header.

    raw_knmi_data : list containing the entire dataset including header.

    returns list of column names.
    """

    column_header = raw_knmi_data[83:84]

    return ''.join(column_header).strip('#').strip().\
        replace(' ', '').split(',')


def main():
    # Assignment 1
    f = open(INPUTFILE, 'r')
    raw_knmi_data = f.readlines()
    f.close()

    # Assignment 12
    print read_data(raw_knmi_data, 500)[0], '\n\n'  # Read until line 500.
    print read_stationid(raw_knmi_data), '\n\n'
    print read_column_description(raw_knmi_data), '\n\n'
    print read_column_header(raw_knmi_data)
    # NB there is one entry more in the list returned by read_column_header
    # STN is in the line with column headers but it has no description.

# This codeblock is executed from CLI, but not upon import.
if __name__ == '__main__':  # <-- Assignment 6
    main()
