#!/bin/sh
file1=../svd3x3-cuda/Dataset_1M.txt.bz2
#
file_out=../svd3x3-cuda/Dataset_1M.txt
#
if [ ! -e $file_out ]; then
    bunzip2 $file1
fi
