#!/bin/bash

module load daint-gpu
module load PyExtensions/2.7-CrayGNU-17.08

python $(dirname $0)/plotting.py
