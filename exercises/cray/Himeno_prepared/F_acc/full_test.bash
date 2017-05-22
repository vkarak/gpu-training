#!/bin/bash

source ../../Tools/XK_setup.bash cray

#for MYPE in cray pgi
#do


#     for PRECISION in single double
     for PRECISION in double
     do
        ACC=yes
        OMP_DEV=no
        #for VERSION in 00 01 02 03 # OMP (no device) and ACC version
        for VERSION in 00 01 02 03  # OMP device versions
        do

            make clean
            make VERSION=${VERSION} ACC=${ACC} OMP_DEV=${OMP_DEV} PRECISION=${PRECISION}
            if [ $? != 0 ]
            then
                echo "Error when building this code"
                continue
            fi

            bash submit.bash himeno_v${VERSION}.x

        done

        ACC=no
        OMP_DEV=yes
        #for VERSION in 00 01 02 03 # OMP (no device) and ACC version
        for VERSION in 00 11 12 13  # OMP device versions
        do
            
            make clean
            make VERSION=${VERSION} ACC=${ACC} OMP_DEV=${OMP_DEV} PRECISION=${PRECISION} 
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
        for NTHREADS in 1 6 12
        do
            
            bash submit.bash himeno_v00.x $NTHREADS
	    sleep 2

        done
     done
#    done
#done

make clean
