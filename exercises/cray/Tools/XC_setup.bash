#!/bin/bash
#set -x
# This is a bash script that will ensure that you have the correct
#   modules loaded to compile and submit the exercises.
#   There are no versions for other shells, but you are free to
#   implement one.

# You should source this file rather than execute it in a new shell, 
#   i.e. type "source XK_setup.bash" or ". XK_setup.bash" 
#   rather than "bash XK_setup.bash" or "./XK_setup.bash"

# Load the module setup stuff
. ${MODULESHOME}/init/sh

echo "Setting Programming Environment:"
# Select MYPE from command line arguments
case $# in
    0)
	MYPE=cray
	echo "  defaulting to MYPE=$MYPE";;
    1)
	MYPE=$1
	echo "  using MYPE=$MYPE";;
    *)
	echo "Usage: source XC_setup.bash [MYPE]"
	return -1;;
esac
    
# Sanity check MYPE
case $MYPE in
    cray|pgi) ;;
    gnu|intel|pathscale)
	echo "No OpenACC support with MYPE=$MYPE"
	return -1;;
    *)
	echo "Unknown PrgEnv MYPE=$MYPE"
	return -1;;
esac

#   This is the one already loaded, transcribed to lower case
pe_env=$(echo $PE_ENV | tr '[:upper:]' '[:lower:]')
#   Compare it to what is already loaded
if [ ${pe_env} == ${MYPE} ] 
then
    echo "  module PrgEnv-$MYPE already loaded"
else
    echo "  swapping from PrgEnv-${pe_env} to PrgEnv-$MYPE"
    module swap PrgEnv-${pe_env} PrgEnv-${MYPE}
fi

# Swap to the most recent compiler, for best support
#module swap cce cce/8.3.9

#################################################################
# Make sure the accelerator module is loaded
this_module="craype-accel-nvidia35"
echo "Selecting $this_module:"
if module li 2>&1 | grep -q "$this_module"
then
    echo "  module $this_module already loaded"
else
    echo "  loading module $this_module"
    module switch craype-broadwell craype-ivybridge
    module load $this_module
#    module switch cudatoolkit/7.5.18-1.0502.10743.2.1 cudatoolkit/7.0.28-1.0502.10280.4.1
    module unload cray-libsci_acc
fi
#################################################################

#################################################################
# We need to make sure the WLM module is loaded
echo "Detecting WLM:"
#   First, look at available modules to identify which WLM is being used
if module -l avail moab 2>&1 | grep -q "^moab"
then
    WLM="moab"
elif module -l avail pbs 2>&1 | grep -q "^pbs"
then 
    WLM="pbs"
elif module -l avail slurm 2>&1 | grep -q "^slurm" 
then
    WLM="slurm"
else
    echo "Not sure which WLM is available"
    exit -1
fi
echo "  detected WLM=$WLM module"

#   Now make sure the WLM module is loaded
this_module=$WLM
echo "Selecting $this_module:"
if module li 2>&1 | grep -q "$this_module"
then
    echo "  module $this_module already loaded"
else
    echo "  loading module $this_module"
    module load $this_module
fi

# Here set some WLM-specific stuff
case $WLM in
    pbs|moab)
	WLM_HEADER=header.pbs
	WLM_SUBMIT=qsub;;
    slurm)
	WLM_HEADER=header.slurm
	WLM_SUBMIT=sbatch;;
    *)
	echo "Not sure which WLM to use"
	exit -1;;
esac

case $HOST in
    vista|mork)
	WLM_SUBMIT="$WLM_SUBMIT -lmppnodes=\"$(cnselect 'subtype="nVidia_Kepler"')\""
	;;
esac

#################################################################

# Finally, print a list of modules, redirected from STDERR to STDOUT
module list 2>&1

return 0

#EOF
