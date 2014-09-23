#!/usr/bin/python
# -* coding: utf-8 -*

# knmi-6126561.py <-- Step 1 (hw4)

# Basic Linux and Coding for AA homework 4 (week 2) and homework 5 (week 3).
# Usage: python knmi-6126561.py
# TLR Halbesma, 6126561, september 21, 2014. Version 2.0; added hw5.

# An instance of defaultdict(dict) enables obtaining values as
# name_of_instance[var1][var2]. e.g. for matrix of month and decade.
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# Import methods and variables from homework 3 (week 2).
from knmi_1_6126561 import *  # <-- Step 2 (hw4)

# Override INPUTFILE with dataset that does not include 20000101!
# NB this is a slightly different dataset than used for homework 3.
INPUTFILE = './KNMI_19991231.txt'

# Make data available troughout all methods (global variables).
# Perhaps in the future implement a class that holds the data?
knmi_data = list()
knmi_station_ids = list()
knmi_column_description = dict()
knmi_column_header = list()


def read_dataset(max_lines=None):
    """
    Read the KNMI dataset, save to global variables.

    max_lines : int/None. if None, entire dataset is read.
               else: max_lines is the maximum number of lines to read.

    knmi_data: list containing a list with all datapoints.
    knmi_station_ids: list containing station_id's parameters.
    knmi_column_description: dict mapping column name to description.
    knmi_column_header: list of column names

    See knmi_1_6126561.py for full details.
    """

    global knmi_data
    global knmi_station_ids
    global knmi_column_description
    global knmi_column_header

    f = open(INPUTFILE, 'r')
    raw_knmi_data = f.readlines()
    f.close()

    if max_lines is None:
        max_lines = len(raw_knmi_data)
    # The header is 85 lines so the program fails if max_lines < 85!
    elif max_lines < 85:
        max_lines = 85

    # Obtain data and entries using homework3's methods.
    print "read_dataset(): start. Be patient, may take a while."
    knmi_data = read_data(raw_knmi_data, max_lines)
    knmi_station_ids = read_stationid(raw_knmi_data)
    knmi_column_description = read_column_description(raw_knmi_data)
    knmi_column_header = read_column_header(raw_knmi_data)

    print "\nread_dataset(): done. Success :-)!\n"


def find_column_number(keyword):
    """
    Function to obtain the number of a column given a (unique) identifier.
    This functions searches keyword in ColumnDiscription header, finds
    its abbreviation and looks for that abbreviation in the column_header.

    keyword : string. e.g. 'Maximum temperature', 'precipitation', etc.

    returns an integer. Data entry list number for keyword string.
    """

    column_abbreviation = None
    # Loop trough ColumnDescription, find given string in value (description).
    for key, value in knmi_column_description.items():
        if keyword in value:
            # Now get the key (abbreviation) and find it in the ColumnHeader.
            column_abbreviation = key
            break
    if column_abbreviation:  # Check if column_abbreviation is found.
        return knmi_column_header.index(column_abbreviation)
    else:
        return None


def find_station_name(station_id):
    for station in knmi_station_ids:
        if station[0] == station_id:
            return station[-1]

    return None


def find_station_id(station_name):
    for station in knmi_station_ids:
        if ''.join(station[4:]) == station_name:
            return station[0]

    return None


# Step 3 (hw4), Question 1 (hw5)
def find_max(dataset, column_number, to_reverse):
    """
    Find the maximum value in the data set given a column_number to sort on.
    Found sorting a matrix on http://xahlee.info/perl-python/sort_list.html

    dataset : nested list. Contains the dataset that is sorted.
    column_number : int. Specify which column is  sorted on.
    to_reverse : boolean. True => reverse (max -> min); False => (min -> max)

    returns a list containing the entry of the max (or min) in the dataset.
    """

    # Initlial implementation with sort and lambda. This, however, alters
    # the order of my global variable. Therefore changed the implementation.
    # dataset.sort(key=lambda x: x[column_number], reverse=to_reverse)
    # return dataset[0]

    biggest = 0
    biggest_entry = list()

    if to_reverse:
        for entry in dataset:
            # NB biggest now is smallest!
            if entry[column_number] < biggest:
                biggest = entry[column_number]
                biggest_entry = entry
    else:
        for entry in dataset:
            if entry[column_number] > biggest:
                biggest = entry[column_number]
                biggest_entry = entry

    return biggest_entry


# Step 4 (hw4), Question 2 (hw5)
def create_time_series(dataset, column_number, station_id, year):
    """
    function to create a time-series.

    dataset : nested list. Contains the dataset a subset is made for.
    column_number : int. Specify which column is in the subset.
    station_id : int. ID number of Station the averages is obtained for.
    year : int. Specify which year is in the subset.

    returns a nested list containing the time-series subset.
    """

    subset = list()

    for entry in dataset:
        # entry[1] is the date YYYYMMDD as integer. So div by 1e5 will
        # result in YYYY. As it is int-int division it is truncated.
        if entry[0] == station_id and entry[1]/10000 == year:
            subset.append(list((entry[0], entry[1], entry[column_number])))

    return subset


def month_average_time_series(time_series):
    month_average = dict()
    month_count = dict()

    # Set initial values to zero
    for i in range(1, 13):
        month_average[i] = int()
        month_count[i] = int()

    # Sum per-month, keep track of number of entries in dataset.
    for entry in time_series:
        month = (entry[1]/100) % 100
        if entry[2] and entry[2] is not 0:
            month_count[month] += 1
            month_average[month] += entry[2]

    # Devide per-month sum by number of entries.
    for i in range(1, 13):
        if month_count[i]:
            month_average[i] /= float(month_count[i])

    return month_average


# A stepped line plot.. that is just a histogram, right?!
def plot_time_series(time_series, chosen_column):
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                   'Sep', 'Oct', 'Nov', 'Dec']
    y_label = {'TX': r'Maximum temperature TX ($^\circ$C)',
               'TN': r'Minimum temperature TN ($^\circ$C)',
               'RH': r'Daily precipitation RH (mm)'}

    first_entry = time_series[0]
    station_id = first_entry[0]
    station_name = find_station_name(station_id)
    year = first_entry[1]/10000

    month_average = month_average_time_series(time_series)

    index = np.arange(1, 13)
    width = 0.55
    fig, ax = plt.subplots()
    ax.bar(index, [month_average[x]/10. for x in range(1, 13)],
           width, color='r')
    ax.set_xlabel('Month of ' + str(year))
    ax.set_ylabel(y_label.get(chosen_column, 'KeyError'))
    ax.set_title(chosen_column + ' for ' + station_name +
                 ' in ' + str(year))
    plt.xticks(range(1, 13), month_names, rotation=45)
    plt.savefig('BLAC_hw5_TLRH_6126561_' + str(station_id) + '_' +
                chosen_column + '_' + str(year) + '.pdf')


# Question 4 (hw5)
def plot_five_year_series(station_id1, station_id2, year_start,
                          chosen_column, temp_west, temp_east):
    """ Documenting this function should have been added... """

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                   'Sep', 'Oct', 'Nov', 'Dec']
    y_label = {'TX': r'Maximum temperature TX ($^\circ$C)',
               'TN': r'Minimum temperature TN ($^\circ$C)'}
    pick_color = {0: 'r', 1: 'y', 2: 'g', 3: 'b', 4: 'c'}

    station_name1 = find_station_name(station_id1)
    station_name2 = find_station_name(station_id2)

    width = 0.19
    fig, ax = plt.subplots()
    for year in range(year_start, year_start+5):
        ax.bar([x + (year - year_start) * width for x in range(3)],
               [temp_west[month]/10. for month in
                range(year - year_start, year - year_start + 3)],
               width, color=pick_color.get(year - year_start, 'k'),
               alpha=0.3, label='West '+str(year))
        ax.bar([x + (year - year_start) * width for x in range(3)],
               [temp_east[month]/10. for month in
                range(year - year_start, year - year_start + 3)],
               width, color=pick_color.get(year - year_start, 'k'),
               label='East '+str(year))
    ax.set_xlabel('Month')
    ax.set_ylabel(y_label.get(chosen_column, 'KeyError'))
    # ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.set_xlim(-0.5, 4.5)
    ax.legend()
    if chosen_column == 'TX':
        ax.set_title(chosen_column + ' for ' + station_name1 + '(west) vs '
                     + station_name2 + '(east) in the summer of ' +
                     str(year_start) + ' - ' + str(year_start+5))
        plt.xticks([0.5, 1.5, 2.5], month_names[6:9], rotation=45)
        plt.savefig('BLAC_hw5_TLRH_6126561_' + str(station_id1) + 'vs' +
                    str(station_id2) + '_' + chosen_column + '_summer_' +
                    str(year_start) + '.pdf')
    elif chosen_column == 'TN':
        ax.set_title(chosen_column + ' for ' + station_name1 + '(west) vs '
                     + station_name2 + '(east) in the winter of ' +
                     str(year_start) + ' - ' + str(year_start+5))
        plt.xticks([0.5, 1.5, 2.5], month_names[0:3], rotation=45)
        plt.savefig('BLAC_hw5_TLRH_6126561_' + str(station_id1) + 'vs' +
                    str(station_id2) + '_' + chosen_column + '_winter_' +
                    str(year_start) + '.pdf')


def compare_two_stations(station_west, station_east):
    """
    Function to compare two stations, one at the North Sea, one in East.

    station_west : int. Station at the North Sea (West coast)
    station_east : int. Station in the East of the Netherlands.

    Creates plots and saves them to file.
    """

    tx_column_number = find_column_number('Maximum temperature')
    tn_column_number = find_column_number('Minimum temperature')

    summer_five_years_west = list()
    summer_five_years_east = list()
    winter_five_years_west = list()
    winter_five_years_east = list()

    # NB since we only work with month-averages, we ignore that seasons
    # change 21/22th of month. Summer := Jul/Aug/Sep; Winter := Jan/Feb/Mar
    for year in range(1991, 1996):
        # The maximum temperature (for the hottest summer).
        max_temp_west = create_time_series(knmi_data, tx_column_number,
                                           station_west, year)
        max_temp_west_avg = month_average_time_series(max_temp_west)
        max_temp_east = create_time_series(knmi_data, tn_column_number,
                                           station_east, year)
        max_temp_east_avg = month_average_time_series(max_temp_east)
        for month in range(7, 10):  # summer
            print 'west', year, month, 'max', max_temp_west_avg[month]
            print 'east', year, month, 'max', max_temp_east_avg[month]
            summer_five_years_west.append(max_temp_west_avg[month])
            summer_five_years_east.append(max_temp_east_avg[month])

        # And for the minimum temperature (for the coldest winter).
        min_temp_west = create_time_series(knmi_data, tx_column_number,
                                           station_west, year)
        min_temp_west_avg = month_average_time_series(min_temp_west)
        # plot_time_series(min_temp_west, 'TN')
        min_temp_east = create_time_series(knmi_data, tn_column_number,
                                           station_east, year)
        min_temp_east_avg = month_average_time_series(min_temp_east)
        for month in range(1, 4):  # winter
            print 'west', year, month, 'min', min_temp_west_avg[month]
            print 'east', year, month, 'min', min_temp_east_avg[month]
            winter_five_years_west.append(min_temp_west_avg[month])
            winter_five_years_east.append(min_temp_east_avg[month])

    print summer_five_years_west
    print summer_five_years_east
    plot_five_year_series(station_west, station_east, 1991, 'TX',
                          summer_five_years_west, summer_five_years_east)

    print winter_five_years_west
    print winter_five_years_east
    plot_five_year_series(station_west, station_east, 1991, 'TN',
                          winter_five_years_west, winter_five_years_east)

def monthly_decade_average(dataset, station_id, column_number):
    """
    Function to calculate monthly averages per decade.
    NB, this functions requires a dataset from 1950 until (excluding) 2000.
    This is because I use integer indices representing month and decade
    in the range(1,13) for month, and range(5,10) for decade.

    dataset : list containing the entire dataset including header
    station_id : int. ID number of station the averages is obtained for.
    column_number : int. Number of column the averages is obtained for.

    returns a dictionary. The keys are 4-tuples (station_id, column_number,
        month, decade). The values are the averages as a float.
    """

    decade_average = dict()
    number_of_entries = defaultdict(dict)

    # All variables must be zero initialy. Otherwise the first += fails.
    for month in range(1, 13):
        for decade in range(5, 10):
            number_of_entries[month][decade] = int()
            decade_average[(station_id, column_number, month, decade)] =\
                int()

    for entry in dataset:
        if entry[0] == station_id:
            # entry[1] is the date YYYYMMDD as integer. So (div by 100)%100
            # will result in MM. As it is int-int division it is truncated.
            month = (entry[1]/100) % 100
            # split decade up in blocks of 10
            # Note that the dataset must not include 2000!!
            decade = (entry[1]/100000) % 10

            # Missing data has value None in dataset. NB bool(0) --> False!
            if entry[column_number] and entry[column_number] is not 0:
                number_of_entries[month][decade] += 1
                decade_average[(station_id, column_number, month, decade)] \
                    += entry[column_number]

    # Now divide the monthly decade sums over the number of entries.
    for month in range(1, 13):
        for decade in range(5, 10):
            if decade_average[(station_id, column_number, month, decade)] \
                    != 0:
                decade_average[(station_id, column_number, month, decade)]\
                    /= float(number_of_entries[month][decade])

    return decade_average


# Step 5 (hw4)
# Compare the summers in “De kooy” with those in “Valkenburg”. Calculate
# monthly averages for min, max temperature and the amount of precipitation
# on a 10 yearly basis. Where are the summers warmer, where are they
# wetter?
def compare_dekooy_valkenburg(dataset):
    rh_column_number = find_column_number('precipitation amount')
    tx_column_number = find_column_number('Maximum temperature')
    tn_column_number = find_column_number('Minimum temperature')

    dekooy = find_station_id('DE KOOY')
    valkenburg = find_station_id('VALKENBURG')

    dekooyRHAverage = \
        monthly_decade_average(knmi_data, dekooy, rh_column_number)
    valkenburgRHAverage =\
        monthly_decade_average(knmi_data, valkenburg, rh_column_number)
    dekooyTXAverage = \
        monthly_decade_average(knmi_data, dekooy, tx_column_number)
    valkenburgTXAverage =\
        monthly_decade_average(knmi_data, valkenburg, tx_column_number)
    dekooyTNAverage = \
        monthly_decade_average(knmi_data, dekooy, tn_column_number)
    valkenburgTNAverage =\
        monthly_decade_average(knmi_data, valkenburg, tn_column_number)

    for k, v in dekooyRHAverage.items():
        print k, v
    for k, v in valkenburgRHAverage.items():
        print k, v
    print
    for k, v in dekooyTXAverage.items():
        print k, v
    for k, v in valkenburgTXAverage.items():
        print k, v
    print
    for k, v in dekooyTNAverage.items():
        print k, v
    for k, v in valkenburgTNAverage.items():
        print k, v


def plot_comparison(valkenburgData, dekooyData, s):
    # http://matplotlib.org/examples/api/barchart_demo.html
    title = {'TX': 'maximum temperature', 'TN': 'minimum temperature',
             'RH': 'daily precipitation'}
    month_names = ['Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug',
                   'Sep', 'Okt', 'Nov', 'Dec']
    ind = np.arange(1, 13)
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, tuple([valkenburgData.get((s, i, 5))
                    for i in range(1, 13)]), width, color='r')
    rects2 = ax.bar(ind + width, tuple([dekooyData.get((s, i, 5))
                    for i in range(1, 13)]), width, color='y')

    ax.legend((rects1[0], rects2[0]), ('Valkenburg', 'DeKooy'))
    ax.set_ylabel(s)
    ax.set_title('Plot of ' + title[s])
    plt.xticks(range(1, 13), month_names, rotation=45)
    plt.show()
    # plt.close()


# Step 6 (hw4)
# Using the monthly averages (averaged over 10 year blocks), is the weather
# getting warmer or wetter?
def warmer_or_wetter():
    # To implement this function requires rewriting the very crappy
    # implementation of step 5
    return None


def main():
    read_dataset()

    rh_column_number = find_column_number('precipitation amount')
    tx_column_number = find_column_number('Maximum temperature')

    wettest = find_max(knmi_data, rh_column_number,  False)
    print "The wettest day was at {0} in {1}({2}).".\
        format([1], find_station_name(wettest[0]), wettest[0]),
    print "The precipitation amount was {} mm.\n"\
        .format(wettest[rh_column_number]/10.0)

    hottest = find_max(knmi_data, tx_column_number, False)
    print "The hottest day was at {0} in {1}({2}).".\
        format(hottest[1], find_station_name(hottest[0]), hottest[0]),
    print "The temperature was {} degrees Centigrade.\n"\
        .format(hottest[tx_column_number]/10.0)

    hottest_time_series =\
        create_time_series(knmi_data, tx_column_number, 260, 1968)
    print "Maximum temperature for station 260 in 1968 has the following",
    print "first ten entries:\n{0}\n".format(hottest_time_series[0:10])
    plot_time_series(hottest_time_series, 'TX')

    compare_two_stations(210, 283)

    # compare_dekooy_valkenburg(knmi_data)

    # valkenburg, dekooy =compare_dekooy_valkenburg(knmi_data)
    # print 'valkenburg'
    # for k, v in valkenburg.items():
    #     print k, v
    # print 'dekooy'
    # for k, v in dekooy.items():
    #     print k, v

if __name__ == '__main__':
    main()
