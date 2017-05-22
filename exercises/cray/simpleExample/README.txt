Cray_OpenACC_training/Practical1:

PURPOSE
=======

The purpose of this exercise is to become familiar with compiling and
executing OpenACC codes.

The example codes used here are to demonstrate functionality rather than
performance.


OVERVIEW OF CODES
=================

Each source file is a different implementation of the same, simple code.
A three-dimensional array a is initialised. The elements of array b are
then set to be twice those of a. Finally the elements of b are summed and
the value printed out.

These operations are carried out on the accelerator using three OpenACC
kernels.

Four different programming languages and models are used. The codes are either
Fortran or C, and the arrays are either defined statically at compile time or
dynamically allocated at runtime.

For each programming model there are (usually) three different code
versions. The first (version 00) has all three kernels in the same main
program. There is no attempt to keep data on the GPU between the kernels.

The second (version 01) uses a data region to avoid unnecessary copying of
data.

The third version (version 02) does the same, but with a more-complicated
call-tree involving calling a subroutine that contains an OpenACC kernel. This
kernel contains a function call.


NOTE
----

The only exception here is that there is no version 00 for the case of dynamic
memory allocation in C. Multidimensional, dynamically-allocated arrays are
usually implemented as a pointer chain that must be correctly rebuilt on the
accelerator (so-called "deep copying"). At the moment there is no robust way
to achieve this using OpenACC v1.0. The Cray compiler offers an extended
runtime API that can be used to manually rebuild the pointer chain, replacing
the use of a data region. This is used in the Cdynamic example codes here.

This is, however, an advanced topic and most C users should start by looking
at the statically-allocated examples.



AUTOMATED BUILDING AND SUBMISSION
=================================

For convenience, you can compile and execute a given version of the code using
a script.

Type:

bash build_submit.bash TARGET VERSION

TARGET should be Fstatic or Fdynamic or Cstatic [or Cdynamic, if you must]
VERSION should be 00 or 01 or 02

This will:

* load the correct modules using script ../Tools/XK_setup.bash
* build the code using the Makefile
* create a directory on the lustre filesystem
* write and submit a WLM jobscript in this directory.
  -  Type "qstat -u $USER" or "squeue -u $USER" to check progress and completion.

The directory name is reported on the screen. 

You can then cd to this directory and look at the output. The output will be
in a file with filename ending ".log".  "Result: Correct!" denotes the code
executed with the correct checksum value.


If you want to build the code yourself, step-by-step, follow the steps listed
at the end of this file.


WHAT YOU SHOULD DO NEXT
=======================

Check that:

* Did the code compile correctly? (no errors printed to the screen)
* Did the job execute? (a log file was produced)
 -   Type "qstat -u $USER" to check progress and completion
* Was the answer correct? ("Result: Correct!" should appear in the log file)

Next, understand what did the compiler did:

* examine and understand the compiler feedback
 -   look at the .lst file
* did it compile for the accelerator?
 -   look for "G" and "g" in the loopmark
* what data did it plan to move and when?
 -   look for the informational messages
* how were the loop iterations scheduled?
 -   where were the "g"-s and what did the messages say?

Did we actually run on the accelerator?

* We can ask the runtime for some feedback
* cd to run directory, edit jobscript
 -  filename ends with ".wlm"
* Uncomment appropriate line (remove initial "#")
 -  set environment variable CRAY_ACC_DEBUG to 1 or 2 or 3
   +  1 gives least detailed information 
   +  2 is most useful for developers
   +  3 gives most detailed information 
 -  save the edited file
* Resubmit the job: "qsub <jobscript name>" or "sbatch <jobscript name>"
 -  type "qstat -u $USER" or "squeue -u $USER" to check for completion
* Examine commentary (in the log file) and make sure you understand it

Profiling the code

* A quick way of profiling is to use the Nvidia compute profiler
 -  CCE compiles to PTX (as does nvcc), so this will work for all
* Edit the jobscript and uncomment the profiling line
* Resubmit the job
* Examine the profile (in file cuda_profile_0.log)
 *  Can change location with env. var. COMPUTE_PROFILE_LOG
* This is an event-by-event account
 *  Larger codes need a more aggregated report

Warnings!

* Both runtime commentary and profiling will impact performance
 -  Enable neither for performance testing
 -  Don't enable CRAY_ACC_DEBUG when profiling


Building the code yourself (for more advanced users)

SETTING THE ENVIRONMENT
=======================

Type:

source ../Tools/XK_setup.bash

Don't type "bash ../Tools/XK_setup.bash", as this will not change your current
environment


BUILDING THE CODE
=================

Having set the environment, type:

make <TARGET> VERSION=<VERSION>

<TARGET> should be Fstatic or Fdynamic or Cstatic or Cdynamic
       (omitting TARGET will build all four)

where <VERSION> is 00, 01 or 02
       Note that there is no version 00 for target Cdynamic

To run the code by hand you will need to create your own job script
(use build_submit.bash as a template).

Copy the script and executable to a directory on the lustre filesystem
and qsub/sbatch the job from there.

<EOF>
