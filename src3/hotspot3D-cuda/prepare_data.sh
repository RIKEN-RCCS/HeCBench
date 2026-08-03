#!/bin/sh
file1=../data/hotspot3D/power_512x8.bz2
file_out=../data/hotspot3D/power_512x8
#
if [ ! -e $file_out ]; then
    bunzip2 $file1
fi

file1=../data/hotspot3D/temp_512x8.bz2
file_out=../data/hotspot3D/temp_512x8
#
if [ ! -e $file_out ]; then
    bunzip2 $file1
fi

