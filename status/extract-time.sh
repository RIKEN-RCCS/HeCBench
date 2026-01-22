#!/bin/sh
echo "|" name "|" sycl "|" acc  "|"
echo "|" "--" "|" "--" "|" "--" "|"
#
SRCDIR=../src2
#
for n in `cat List_full`
#for n in `cat List_a`
do
    num=0
    while [ $num -lt 2 ];
    do
	if [ $num = 0 ]; then dir=$SRCDIR/$n-sycl; fi
	if [ $num = 1 ]; then dir=$SRCDIR/$n-acc; fi
	
	if [ $num = 0 ]; then comment="| "$n" |"; fi

	if [ -d $dir ]; then
	    c1=`grep Error $dir/log_run_bench.err | grep make`
	    c1a=`echo $c1 | awk '{print $NF}'`
	    if [ -e $dir/log.build ]; then
		c1=`grep Error $dir/log.build | grep make`
	    fi

	    if [ "$c1"  = "" ]; then
		c2=`grep -i error $dir/log.err`
                c3=`grep "core dumped" $dir/log.time`

		c4=`grep Terminated $dir/log_run_bench.err`
		c5=`grep "timeout was set" $dir/log_run_bench.err`

		if [ "$c4" != "" ]; then
		    if [ "$c5" != "" ]; then
			tmp1=`echo $c5 | awk '{print $NF}'`
			time="over "$tmp1
		    else
			time="TLE error"
		    fi
		else
                    if [ "$c2"  = "" -a "$c3" = "" ]; then
			time=`grep real $dir/log.time | awk '{print $NF}'`
		    else
			time="exe err"
		    fi
		fi
	    else
		if [ $c1a -gt 100 ]; then
		    time="exe err"
		else
		    time="build err"
		fi
	    fi
	else
	    time="--"
	fi
	#
	comment=$comment" "$time" |"
	let num="$num +1"
    done
    echo $comment
done
