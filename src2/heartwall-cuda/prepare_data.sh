#!/bin/sh
file1=../data/heartwall/test.avi.bz2
#
file_out=../data/heartwall/test.avi
#
if [ ! -e $file_out ]; then
    bunzip2 $file1
fi
