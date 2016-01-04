#!/bin/bash
# blac_hw1.sh

# Shell script for Basic Linux and Coding for AA homework 1 (week 1).
# Usage: Place script and Thijs' rfi.tar.gz in the same directory. Make script
# exectuable and run the script.
# TLR Halbesma, 6126561, september 5, 2014. Version 1.0; implemented

set -o errexit

# Assignment 2

# Check if Thijs' tarball has already been unpacked.
unpacked=false
for dir in ./L26281_RSP{0,2,3,4,5,6,7,8};
do
    if [ ! -d "$dir" ]; then
        unpacked=false && break
    else
        unpacked=true
    fi
done

if ! "$unpacked"; then
    echo 'Unpacking rfi.tar.gz'
    # Only extracts if the file exists.
    find . -name 'rfi.tar.gz' -exec tar -xzvf {} \;
fi
echo "All files are unpacked"

# Assignment 3

# Find path to all png files, then take only the first result.
toOpen=$(find . -name "*.png" | head -n 1)
echo "Using file: $toOpen"
# Open the file with the application Preview (on Mac)
open -a /Applications/Preview.app/ "$toOpen"

# Assignment 4
metadata=$(find . -name "*.bestprof" | head -n 1)

# The bestprof files contain the character '=' in the header for each entrie.
# Counting the number of occurences of '=' in the bestprof gives the number of
# entries in the header.
entriesInHeader=$(cat $metadata | grep -c "=")
echo "There are $entriesInHeader entries in the header" # 25

# Assignment 5

# First find all bestprof files. We assume that for each dataset a bestprof file
# exists. The bestprof file might contain a pulsar, so the maximum number of
# detected pulsars is equal to the maximum number of bestprof files in the full
# dataset.
maxDetections=$(find . -name "*.bestprof" | grep -c "bestprof")
echo "There are $maxDetections pulsar detections at most" # 1586

# Assignment 6

# First we will find all bestprof files, then we grep the "Reduces chi-sqr"
# from the header. Sort by default has order low-high, so the last entry (tail)
# is the highest chi-sqr. We need to sort on the third column to sort on chi-sqr values.

# grep -r, recursief to obtain path to highest Reduces chi-sqr. 
# sort -k 3 to sort on the third column, because now we have the path
# predeceding the '# Reduced chi-sqr' so just sorting no longer works.
echo -n "The brightest pulsar is: "
find . -name "*.bestprof" -exec grep -r "Reduced chi-sqr" {} \; | sort -k 3 | tail -n 1

# Assignment 7

# Same as previous question, now only tail the last 100 instances found.
echo -n "The 100th brightest pulsar is: "
find . -name "*.bestprof" -exec grep -r "Reduced chi-sqr" {} \; | sort -k 3 | tail -n 100 | head -n 1

exit 0 # Exit with success. Strictly not necessary though.
