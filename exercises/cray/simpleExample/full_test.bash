#!/bin/bash

MODEL=ACC
#for MYPE in cray pgi
#do

    for TARGET in Fstatic Fdynamic Cstatic Cdynamic
    do

        for VERSION in 00 01 02
        do

            bash build_submit.bash $TARGET $VERSION $MODEL

        done
    done
#done

make clean
