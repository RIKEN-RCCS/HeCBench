#!/bin/sh
file1=../data/kmeans/kdd_cup.bz2
#
file_out=../data/kmeans/kdd_cup
#
if [ ! -e $file_out ]; then
    bunzip2 $file1
fi
