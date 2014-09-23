# -* coding: utf-8 -*
#!/usr/bin/python
# knmi-6126561.py <-- Step 1 (hw4)

# Basic Linux and Coding for AA homework 4 (week 2) and homework 5 (week 3).
# Usage: python knmi-6126561.py
# TLR Halbesma, 6126561, september 21, 2014. Version 2.0; added hw5.

import math
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
# An instance of defaultdict(dict) enables obtaining values as
# name_of_instance[var1][var2]. e.g. for matrix of month and decade.

# Import methods and variables from homework 3 (week 2).
from knmi_1_6126561 import * # <-- Step 2 (hw4)

# Override INPUTFILE with dataset that does not include 20000101!
# NB this is a slightly different dataset than used for homework 3.
INPUTFILE = './KNMI_19991231.txt'

# Make data available troughout all methods (global variables).
# Perhaps in the future implement a class that holds the data?
knmiData = list()
knmiStationIDs = list()
knmiColumnDescription = dict()
knmiColumnHeader = list()

def readDataset(maxLines=None):
    """
    Read the KNMI dataset, save to global variables.

    maxLines : int/None. if None, entire dataset is read.
               else: maxLines is the maximum number of lines to read.

    knmiData: list containing a list with all datapoints.
    knmiStationIDs: list containing stationID's parameters.
    knmiColumnDescription: dict mapping column name to description.
    knmiColumnHeader: list of column names

    See knmi_1_6126561.py for full details.
    """

    f = open(INPUTFILE, 'r')
    datasetKNMI = f.readlines()
    f.close()

    if maxLines is None:
        maxLines = len(datasetKNMI)
    # The header is 85 lines so the program fails if maxLines < 85!
    elif maxLines < 85:
        maxLines = 85

    global knmiData
    global knmiStationIDs
    global knmiColumnDescription
    global knmiColumnHeader

    # Obtain data and entries using homework3's methods.
    print "readDataset(): start. Be patient, may take a while."
    knmiData = read_data(datasetKNMI, maxLines)
    knmiStationIDs = read_StationID(datasetKNMI)
    knmiColumnDescription = read_ColumnDescription(datasetKNMI)
    knmiColumnHeader = read_ColumnHeader(datasetKNMI)

    print "\nreadDataset(): done. Success :-)!\n"

def findColumnNumber(myIdentifier):
    """
    Function to obtain the number of a column given a (unique) identifier.
    This functions searches myIdentifier in ColumnDiscription header, finds
    its abbreviation and looks for that abbreviation in the columnHeader.

    myIdentifier : string. e.g. 'Maximum temperature', 'precipitation', etc.

    returns an integer. Data entry list number for myIdentifier string.
    """

    ColumnAbbreviation = None
    # Loop trough ColumnDescription, find given string in value (description).
    for key,value in knmiColumnDescription.items():
        if myIdentifier in value:
            # Now get the key (abbreviation) and find it in the ColumnHeader.
            ColumnAbbreviation = key
            break
    if ColumnAbbreviation: # Check if ColumnAbbreviation is found.
        return knmiColumnHeader.index(ColumnAbbreviation)
    else:
        return None

def findStationName(myStationID):
    for station in knmiStationIDs:
        if station[0] == myStationID:
            return station[-1]

    return None

def findStationID(myStationName):
    for station in knmiStationIDs:
        if ''.join(station[4:]) == myStationName:
            return station[0]

    return None

# Step 3 (hw4), Question 1 (hw5)
def findMax(myDataSet, columnNumber, toReverse):
    """
    Find the maximum value in the data set given a columnNumber to sort on.
    Found sorting a matrix on http://xahlee.info/perl-python/sort_list.html

    myDataSet : nested list. Contains the dataset that should be sorted.
    columnNumber : int. Specify which column should be sorted on.
    toReverse : boolean. True => reverse (max -> min); False => (min -> max)

    returns a list containing the entry of the max (or min) in the dataset.
    """

    myDataSet.sort(key=lambda x:x[columnNumber], reverse=toReverse)
    return myDataSet[0]

# Step 4 (hw4), Question 2 (hw5)
def createTimeSeries(myDataSet, columnNumber, stationID, year):
    """
    function to create a time-series.

    myDataSet : nested list. Contains the dataset a subset should be made for.
    columnNumber : int. Specify which column is in the subset.
    stationID : int. ID number of Station the averages should be obtained for.
    year : int. Specify which year should be in the subset.

    returns a nested list containing the time-series subset.
    """

    subset = list()

    for entry in myDataSet:
        # entry[1] is the date YYYYMMDD as integer. So div by 1e5 will
        # result in YYYY. As it is int-int division it is truncated.
        if entry[0] == stationID and entry[1]/10000 == year:
            print entry
            subset.append(list((entry[0], entry[1], entry[columnNumber])))

    return subset

def monthAverageTimeSeries(myTimeSeries):
    monthAverage = dict()
    monthCount = dict()

    # Set initial values to zero
    for i in range(1,13):
        monthAverage[i] = int()
        monthCount[i] = int()

    # Sum per-month, keep track of number of entries in dataset.
    for entry in myTimeSeries:
        month = (entry[1]/100)%100
        if entry[2] and entry[2] is not 0:
            monthCount[month] += 1
            monthAverage[month] += entry[2]

    # Devide per-month sum by number of entries.
    for i in range(1,13):
        monthAverage[i] /= float(monthCount[i])

    return monthAverage

# A stepped line plot.. that is just a histogram, right?!
def plotTimeSeries(myTimeSeries, whichData):
    monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',\
            'Sep', 'Oct', 'Nov', 'Dec']
    y_label = {'TX': r'Maximum temperature TX ($^\circ$C)',\
            'TN': r'Minimum temperature TN ($^\circ$C)',\
            'RH': r'Daily precipitation RH (mm)'}

    firstEntry = myTimeSeries[0]
    stationID = firstEntry[0]
    stationName = findStationName(stationID)
    year = firstEntry[1]/10000

    monthAverage = monthAverageTimeSeries(myTimeSeries)

    index = np.arange(1,13)
    width = 0.55
    fig, ax = plt.subplots()
    histogram = ax.bar(index, [monthAverage[x]/10. for x in range(1,13)],\
            width, color='r')
    ax.set_xlabel('Month of '+str(year))
    ax.set_ylabel(y_label.get(whichData, 'KeyError'))
    ax.set_title(whichData+' for '+stationName+' in '+str(year))
    plt.xticks(range(1,13), monthNames, rotation=45)
    plt.savefig('BLAC_hw5_TLRH_6126561_'+str(stationID)+'_'+whichData\
            +'_'+str(year)+'.pdf')

def monthlyDecadeAverage(myDataSet, stationID, columnNumber):
    """
    Function to calculate monthly averages per decade.
    NB, this functions requires a dataset from 1950 until (excluding) 2000.
    This is because I use integer indices representing month and decade
    in the range(1,13) for month, and range(5,10) for decade.

    myDataSet : list containing the entire dataset including header
    stationID : int. ID number of Station the averages should be obtained for.
    columnNumber : int. Number of column the averages should be obtained for.

    returns a dictionary. The keys are 4-tuples (stationID, columnNumber,
        month, decade). The values are the averages as a float.
    """

    decadeAverage = dict()
    numberOfEntries = defaultdict(dict)

    # All variables must be zero initialy. Otherwise the first += fails.
    for month in range(1,13):
        for decade in range(5,10):
            numberOfEntries[month][decade] = int()
            decadeAverage[(stationID,columnNumber,month,decade)] = int()

    for entry in myDataSet:
        if entry[0] == stationID:
            # entry[1] is the date YYYYMMDD as integer. So (div by 100)%100
            # will result in MM. As it is int-int division it is truncated.
            month = (entry[1]/100)%100
            # split decade up in blocks of 10
            # Note that the dataset must not include 2000!!
            decade = (entry[1]/100000)%10

            # Missing data has value None in dataset. NB bool(0) -> False!
            if entry[columnNumber] and entry[columnNumber] is not 0:
                numberOfEntries[month][decade] += 1
                decadeAverage[(stationID,columnNumber,month,decade)] \
                        += entry[columnNumber]

    # Now divide the monthly decade sums over the number of entries.
    for month in range(1,13):
        for decade in range(5, 10):
            if decadeAverage[(stationID, columnNumber,month,decade)] != 0:
                decadeAverage[(stationID,columnNumber,month,decade)]\
                        /= float(numberOfEntries[month][decade])

    return decadeAverage

# Step 5 (hw4)
# Compare the summers in “De kooy” with those in “Valkenburg”. Calculate
# monthly averages for min, max temperature and the amount of precipitation
# on a 10 yearly basis. Where are the summers warmer, where are they
# wetter?
def compareDeKooyValkenburg(myDataSet):
    precipitationNumber = findColumnNumber('precipitation amount')
    hottestNumber = findColumnNumber('Maximum temperature')
    coldestNumber = findColumnNumber('Minimum temperature')

    deKooy = findStationID('DE KOOY')
    valkenburg = findStationID('VALKENBURG')

    deKooyRHAverage = \
            monthlyDecadeAverage(knmiData, deKooy, precipitationNumber)
    valkenburgRHAverage =\
            monthlyDecadeAverage(knmiData, valkenburg, precipitationNumber)
    deKooyTXAverage = \
            monthlyDecadeAverage(knmiData, deKooy, hottestNumber)
    valkenburgTXAverage =\
            monthlyDecadeAverage(knmiData, valkenburg, hottestNumber)
    deKooyTNAverage = \
            monthlyDecadeAverage(knmiData, deKooy, coldestNumber)
    valkenburgTNAverage =\
            monthlyDecadeAverage(knmiData, valkenburg, coldestNumber)

    for k,v in deKooyRHAverage.items():
        print k,v
    for k,v in  valkenburgRHAverage.items():
        print k,v
    print
    for k,v in deKooyTXAverage.items():
        print k,v
    for k,v in  valkenburgTXAverage.items():
        print k,v
    print
    for k,v in deKooyTNAverage.items():
        print k,v
    for k,v in  valkenburgTNAverage.items():
        print k,v

def plotComparison(valkenburgData, deKooyData, s):
    """
    REPLACEREPLACE
    """

    # http://matplotlib.org/examples/api/barchart_demo.html
    title = {'TX': 'maximum temperature', 'TN':'minimum temperature',\
            'RH': 'daily precipitation'}
    monthNames = ['Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug',\
            'Sep', 'Okt', 'Nov', 'Dec']
    ind = np.arange(1,13)
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, tuple([valkenburgData.get((s,i, 5))\
            for i in range(1,13)]), width, color='r')
    rects2 = ax.bar(ind+width, tuple([deKooyData.get((s,i, 5))\
            for i in range(1,13)]), width, color='y')

    ax.legend((rects1[0], rects2[0]), ('Valkenburg', 'DeKooy'))
    ax.set_ylabel(s)
    ax.set_title('Plot of '+title[s])
    plt.xticks(range(1,13), monthNames, rotation=45)
    plt.show()
    #plt.close()

# Step 6 (hw4)
# Using the monthly averages (averaged over 10 year blocks), is the weather
# getting warmer or wetter?
def warmerOrWetter():
    # To implement this function requires rewriting the very crappy
    # implementation of step 5
    return None

def main():
    readDataset()

    precipitationNumber = findColumnNumber('precipitation amount')
    hottestNumber = findColumnNumber('Maximum temperature')
    coldestNumber = findColumnNumber('Minimum temperature')

    wettestDay = findMax(knmiData, precipitationNumber,  True)
    print "The wettest day was at {0} in {1}({2}).".format(\
            wettestDay[1],findStationName(wettestDay[0]),wettestDay[0]),
    print "The precipitation amount was {} mm.\n"\
            .format(wettestDay[precipitationNumber]/10.0)

    hottestDay = findMax(knmiData, hottestNumber, True)
    print "The hottest day was at {0} in {1}({2}).".format(\
            hottestDay[1],findStationName(hottestDay[0]),hottestDay[0]),
    print "The temperature was {} degrees Centigrade.\n"\
            .format(hottestDay[hottestNumber]/10.0)


    hottestTimeSeries = createTimeSeries(knmiData, hottestNumber, 260, 1968)
    print "Maximum temperature for station 260 in 1968 has the following",
    print "first ten entries:\n{0}\n".format(hottestTimeSeries[0:10])
    plotTimeSeries(hottestTimeSeries, 'TX')

    #compareDeKooyValkenburg(knmiData)

    #valkenburg, deKooy =compareDeKooyValkenburg(knmiData)
    #print 'valkenburg'
    #for k,v in valkenburg.items():
    #    print k,v
    #print 'deKooy'
    #for k,v in deKooy.items():
    #    print k,v

if __name__ == '__main__':
    main()
