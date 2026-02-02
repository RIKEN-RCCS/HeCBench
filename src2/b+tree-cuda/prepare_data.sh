#!/bin/sh

file1=../data/b+tree/command.txt.bz2
file_out=../data/b+tree/command.txt
#
if [ ! -e $file_out ]; then
    bunzip2 $file1
fi

file1=../data/b+tree/mil.txt.bz2
file_out=../data/b+tree/mil.txt
#
if [ ! -e $file_out ]; then
    bunzip2 $file1
fi

