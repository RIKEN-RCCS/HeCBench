#!/bin/sh
file1=../haversine-cuda/locations.txt.gz
file_out=../haversine-cuda/locations.txt
#
if [ ! -e $file_out ]; then
    gunzip $file1
fi
