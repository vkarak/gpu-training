#!/bin/bash

if [ "$OMPI_COMM_WORLD_RANK" == "0" ] ; then
  exec nvprof --analysis-metrics -fo metrics.%q{OMPI_COMM_WORLD_RANK}.nvprof $*
else
  exec $*
fi
