#!/bin/sh
file1=../mcpr-cuda/alphas.csv.bz2
#
file_out=../mcpr-cuda/alphas.csv
#
if [ ! -e $file_out ]; then
    bunzip2 $file1
fi
