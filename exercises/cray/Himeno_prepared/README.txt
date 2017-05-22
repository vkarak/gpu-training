Cray_GPU_training/Practical2:

PURPOSE
=======

The aim of this practical is to develop an OpenACC/OpenMP device constructs port of a simple
application

* You will follow the steps taken in the Worked Example lecture
 *  You can either do these steps yourself
 *  Or you can use the prepared answer versions

The code considered is the scalar Himeno code. There are both C and Fortran
versions (both with static memory allocation). You should choose a language
and work in the appropriate directory.

In each case, there are 4 versions of the code
* Version 00 is (nearly) as downloaded
 -  a few language tweaks and a new timer is used
* Version 01 has the first OpenMP/OpenACC kernel
* Version 01a adds data scoping, preventing compiler over-cautious
* Version 02 has a data region in the jacobi() subprogram
* Version 03 outer data region in the main routine
* Version 03a is the full port

Unlike Practical1, you will do more of the building and submitting steps
yourself here.

To make things easier, many of the compiler flag options have been written
into the Makefile as options. This is not to hide them; all compilation lines
are written to the screen and you are free to type them yourself if you wish.


CHOOSE THE LANGUAGE VERSION
===========================

cd into either directory F/ or directory C/


SETTING THE ENVIRONMENT
=======================

You must do this before building (in every new terminal or login)

Type:

. ../../Tools/XC_setup.bash


BUILDING THE CODE
=================

Type:

make clean
make VERSION=[00|01|02|03]

Choose the VERSION to build. 

An executable is created called himeno_vNN.x where NN is the version.


WRITE JOB SCRIPT AND SUBMIT
===========================

Type:

bash submit.bash himeno_vNN.x

Specifying the executable that is to be run.

This script:

* creates directory on lustre (and reports where it is)
* submits it with the reported command
* check job status with command: 
*    PBS:   qstat -u $USER
*    SLURM: squeue -u $USER
* when job finishes, output written in log file in specified directory


PROFILING THE CODE
==================

To profile the code with CrayPAT, you need to first load the correct module:

module load perftools-base 
# and one perftools lite module, presumably
module load perftools-lite-gpu 
# OR the full flexible
module load perftools

For the LITE versions simply rebuild and rerun the code, the profile is written to the stdout file.

In the following the usage of the full CrayPAT using pat_build for instrumentation and pat_report for profile gerneration is described.

Then rebuild the chosen version of the code:

make clean
make VERSION=NN

Then instrument the code (-u traces all user functions):

pat_build -f -u himeno_vNN.x

Then run the instrumented version:

bash submit.bash himeno_vNN.x+pat

and move to the directory where the script reports the code will run

cd <lustre directory>

When the job finishes, there is a file whose name ends with ".xf" in the
lustre directory. Process this file to obtain a report:

qstat -u $USER  # or squeue -u $USER for SLURM
pat_report <.xf file>

Another useful report is:

pat_report -Ocalltree <.xf file>

In all cases, CrayPAT will by default not show entries that take less than 1%
of the profile. To see these, add option "-T" to the pat_report command.


For the OpenMP/OpenACC regions (versions 01 onwards), the asynchronous kernel
launches mean most of the time is spent in synchronisation, which makes the
profile harder to interpret. At the risk of distorting the profile, you can
switch off the asynchronous behaviour using the CCE flag
-hacc_model=auto_async_none (see "man openacc") and profile as before:

make clean
make VERSION=NN ACC_MODEL=auto_async_none
pat_build -f -u himeno_vNN.x
bash submit.bash himeno_vNN.x+pat
cd <lustre directory>
qstat -u $USER  # or squeue -u $USER for SLURM
pat_report <.xf file>

This is profiling at the routine level. For accelerator porting, profiling at
the loop level is often more useful. To obtain this with CCE, you should
recompile the CPU version (VERSION=00) with the -hprofile_generate flag and
profile as before:

module load perftools 
make clean
make VERSION=00 PROFILE_GENERATE=yes
pat_build -f -u himeno_v00.x
bash submit.bash himeno_v00.x+pat
cd <lustre directory>
qstat -u $USER  # or squeue -u $USER for SLURM
pat_report <.xf file>


WHAT TO DO NEXT
===============

The Makefile has a number of options that you can explore by adding one or
more of these argument to the make command, e.g.

make clean
make VERSION=03 PRECISION=double NTPB=512

PROBLEM_SIZE=[1|2|3|4]
----------------------

Controls size of arrays in the problem. 
default: 3

PRECISION=[single|double] 
-------------------------

Controls the precision of the data arrays. Accumulation variables are always
double precision to prevent rounding errors.
default: single

NTPB=[64|128|256|512|1024]
--------------------------

Vector-length (analagous to number of threads per block in CUDA).
default: 128

<EOF>
