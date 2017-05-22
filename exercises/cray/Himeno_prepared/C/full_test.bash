#!/bin/bash

source ../../Tools/XC_setup.bash cray

#for MYPE in cray pgi
#do

     ACC=yes
#     for PRECISION in single double
     for PRECISION in double
     do

        for VERSION in 00 01 02 03
        do
            
            make clean
            make VERSION=${VERSION} ACC=${ACC} PRECISION=${PRECISION} 
	    if [ $? != 0 ]
	    then
		echo "Error when building this code"
		continue
	    fi
            
            bash submit.bash himeno_v${VERSION}.x

        done

        make clean
        make VERSION=00 OMP=yes ACC=no PRECISION=${PRECISION} 
	if [ $? != 0 ]
	then
	    echo "Error when building this code"
	    continue
	fi
#        for NTHREADS in 1 2 3 4 6 8 10 12 14 16
        for NTHREADS in 1 16
        do
            
            bash submit.bash himeno_v00.x $NTHREADS
	    sleep 2

        done
     done
#    done
#done

make clean
