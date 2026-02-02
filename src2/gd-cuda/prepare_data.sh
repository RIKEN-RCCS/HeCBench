#!/bin/sh
file1=../gd-cuda/gisette_scale.bz2
#
file_out=../gd-cuda/gisette_scale
#
if [ ! -e $file_out ]; then
    bunzip2 $file1
fi
