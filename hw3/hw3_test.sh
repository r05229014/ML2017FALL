#!/bin/bash

if [ -f "weights-improvement-20-0.65.hdf5" ]; then
    # exist
    echo "file /weights-improvement-20-0.65.hdf5 exists"
else
    # not exist
    echo "file /weights-improvement-20-0.65.hdf5 not exist"
    wget --no-check-certificate "https://www.dropbox.com/s/l5e6tc3596e7sdh/weights-improvement-20-0.65.hdf5?dl=1" -O weights-improvement-20-0.65.hdf5 

 
fi

python hw3_test.py $1 $2

