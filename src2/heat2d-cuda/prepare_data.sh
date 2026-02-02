#!/bin/sh
file1=../heat2d-cuda/data.txt.bz2
#
file_out=../heat2d-cuda/data.txt
#
if [ ! -e $file_out ]; then
    bunzip2 $file1
fi
