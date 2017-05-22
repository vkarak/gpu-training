#!/bin/bash

# Use this subscript to choose a programming environment and code version,
# build the code, write a submission script and then submit the job

# Make sure there are the right amount of arguments given
if [ $# -ne 3 ]
then
  echo "Usage: $0 TARGET VERSION MODEL"
  echo "       where TARGET  can be Fstatic, Fdynamic, Cstatic, Cdynamic"
  echo "             VERSION can be 00, 01, 02"
  echo "             MODEL   can be ACC, OMP"
  exit
fi

MYPE=cray  # the Programming Environment to use
TARGET=$1
VERSION=$2
MODEL=$3

# Sanity checking for TARGET and VERSION
case $TARGET in
    Fstatic|Fdynamic|Cstatic|Cdynamic) ;;
    *)
	echo "What is TARGET=$TARGET ?"
	exit -1;;
esac
case $VERSION in
    00|01|02) ;;
    *)
	echo "What is VERSION=$VERSION ?"
	exit -1;;
esac
case $MODEL in
    ACC|OMP) ;;
    *)
        echo "What is MODEL=$VERSION ?"
        exit -1;;
esac


# You can't do TARGET=Cdynamic and VERSION=01
if [ $TARGET == "Cdynamic" ] && [ $VERSION == "00" ]
then
    echo "There is no TARGET=Cdynamic, VERSION=00"
    exit -1
fi

# Set up the programming environment
TOOLSDIR=$(dirname $(pwd -P))/Tools
source $TOOLSDIR/XC_setup.bash $MYPE

# Build the code
make clean
make $TARGET VERSION=$VERSION MODEL=$MODEL

if [ $? != 0 ]
then 
    echo "Error detected when building code"
    exit -1
fi

app=./first_example_${TARGET}_v${VERSION}_${MODEL}.x
app_name=$(echo first_example_${TARGET}_v${VERSION}_${MODEL} | cut -c1-15)
lst_file=first_example_${TARGET}_v${VERSION}.lst

# A directory on the lustre in which to run the code
case $HOST in
    todi*)
	LUSDIR=/scratch/todi/$USER;;
    *)
	LUSDIR=/lus/scratch/$USER;;
esac
LUSBASE=${LUSDIR}/Cray_OpenACC_training/Practical1
LUS=${LUSBASE}/$(date +%y%m%d_%H%M%S)_${MYPE}_${TARGET}_v${VERSION}_${MODEL}
# Tell me where it is, for future reference
echo "Running in directory:"
echo $LUS
# Make the directory (and parents) if it isn't already there
mkdir -p $LUS
# Copy the application to the working directory
cp $app $LUS
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
sed "s/APPLICATION/$app_name/g" $TOOLSDIR/$WLM_HEADER > $jobscript

# Now write the remainder of the jobscript
cat <<EOF >> $jobscript

# Make sure we are in the correct directory
cd $LUS

# Load appropriate programming environment
#   So appropriate runtime libraries are found (if needed)
. XC_setup.bash $MYPE

# Uncomment this line for debugging with CCE
#export CRAY_ACC_DEBUG=2
# Uncomment this line for basic profiling
#export COMPUTE_PROFILE=1

# Execute the code, inheriting the job parameters from WLM
/usr/bin/time -p aprun -B $app

EOF

# Submit the job (WLM_SUBMIT defined in XC_setup.bash)
echo "Submitting job: $WLM_SUBMIT $jobscript"
$WLM_SUBMIT $jobscript

#EOF

