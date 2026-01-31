#!/bin/sh
echo "# build の状況"
echo ""
echo "|" name "|" sycl "|" omp_nvc  "|"
echo "|" "--" "|" "--" "|" "--" "|"
#
SRCDIR=../tmp4
#SRCDIR=./
#

######################
sum0=0
sum1=0
sum2=0

######################
for n in `cat List_full`
#for n in `cat List_part`
do
    num=0
    while [ $num -lt 2 ];
    do
	success="---"
        if [ $num = 0 ]; then dir=$SRCDIR/$n-sycl; fi
        if [ $num = 1 ]; then dir=$SRCDIR/$n-omp_nvc; fi
	
	if [ $num = 0 ]; then comment="| "$n" |"; fi

	if [ -d $dir ]; then
	    c1=""
	    c2=""
	    c3=""
	    if [ -e $dir/log.build ]; then
		c1=`grep Error $dir/log.build | grep make`
		c2=`grep -i stop $dir/log.build`
		c3=`grep -i "No files to process" $dir/log.build`

		if [ "$c1" = "" -a "$c2" = "" -a "$c3" = "" ]; then
		    success="yes"
		    if [ $num = 0 ]; then let sum0="$sum0 +1"; fi
		    if [ $num = 1 ]; then let sum1="$sum1 +1"; fi
		    if [ $num = 2 ]; then let sum2="$sum2 +1"; fi
		fi
	    fi
	fi
	#
#        comment=$comment" "$mem" |"
	comment=$comment" "$success" |"
	let num="$num +1"
    done
    echo $comment
done
######################
echo "|"           "|"        "|"       "|"
echo "|" "success" "|"  $sum0 "|" $sum1 "|"
