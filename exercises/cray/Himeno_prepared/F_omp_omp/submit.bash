#!/bin/bash

# Use this script to write a submission script and then submit the job

# Make sure there are the right amount of arguments given
case $# in
    1)
	echo "Defaulting to NTHREADS=1"
	NTHREADS=1;;
    2)
	NTHREADS=$2;;
    *)
	echo "Usage: $0 EXECUTABLE [NTHREADS]"
	exit
esac

EXECUTABLE=$1
app=$(basename $EXECUTABLE)
app_name=$(echo $app | sed 's/\.x//;s/\+pat//')

# Set the environment
MYPE=cray
TOOLSDIR=$(dirname $(dirname $(pwd -P)))/Tools
source $TOOLSDIR/XC_setup.bash $MYPE

# A directory on the lustre in which to run the code
case $HOST in
    todi*)
	LUSDIR=/scratch/todi/$USER;;
    *)
	LUSDIR=/lus/scratch/$USER;;
esac
LUSBASE=${LUSDIR}/Cray_GPU_training/Practical2
LUS=${LUSBASE}/F/cray_${app}_$(date +%y%m%d_%H%M%S)
# Tell me where it is, for future reference
echo "Running in directory:"
echo $LUS
# Make the directory (and parents) if it isn't already there
mkdir -p $LUS
# Copy the application to the working directory
cp $EXECUTABLE $LUS
# Move the setup script there as well
cp $TOOLSDIR/XC_setup.bash $LUS
# Move the compiler feedback file there for book-keeping
cp *.lst $LUS
# Move to the working directory
cd $LUS

# Write a WLM submission script
#   We will use one node (mppwidth), with one thread per node (mppdepth) and
#   one MPI process per node (mppnppn).
jobscript=submit.wlm

# Write a personalised header first
sed "s/APPLICATION/$app_name/g;s/mppdepth=1/mppdepth=$NTHREADS/;s/cpus-per-task=1/cpus-per-task=$NTHREADS/" $TOOLSDIR/$WLM_HEADER > $jobscript

# Now write the remainder of the jobscript
cat <<EOF >> $jobscript

# Make sure we are in the correct directory
cd $LUS

# Load appropriate programming environment
#   So appropriate runtime libraries are found
. XC_setup.bash $MYPE

# Uncomment this line for runtime commentary
#export CRAY_ACC_DEBUG=2
# Uncomment this line for basic profiling
#export COMPUTE_PROFILE=1

# Execute the code, inheriting the job parameters from PBS
export OMP_NUM_THREADS=$NTHREADS
/usr/bin/time -p aprun -n 1 -N 1 $app

EOF

# Submit the job (WLM_SUBMIT defined in XC_setup.bash)
echo "Submitting job: $WLM_SUBMIT $jobscript"
$WLM_SUBMIT $jobscript

#EOF
