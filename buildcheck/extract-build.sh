#!/bin/sh
echo "# build の状況"
echo ""
echo "|" name "|" sycl "|" acc  "|"
echo "|" "--" "|" "--" "|" "--" "|"
#
SRCDIR=../tmp2
#SRCDIR=./
#
for n in `cat List_full`
#for n in `cat List_part`
do
    num=0
    while [ $num -lt 2 ];
    do
	success="---"
        if [ $num = 0 ]; then dir=$SRCDIR/$n-sycl; fi
        if [ $num = 1 ]; then dir=$SRCDIR/$n-acc; fi
	
	if [ $num = 0 ]; then comment="| "$n" |"; fi

	if [ -d $dir ]; then
	    c1=""
	    c2=""
	    if [ -e $dir/log.build ]; then
		c1=`grep Error $dir/log.build | grep make`
		c2=`grep -i stop $dir/log.build`

		if [ "$c1" = "" -a "$c2" = "" ]; then
		    success="yes"
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
