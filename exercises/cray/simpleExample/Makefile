# You can override these with command line arguments when calling
# the Makefile, e.g. "make ACC=yes"
VERSION=00
#MODEL=ACC

# You should not need to alter anything below this line
# The Cray wrappers for the different compilers
FC=ftn
CC=cc

FFLAGS=
CFLAGS=

# Options for the Cray compiler
# Write a compiler commentary in separate files with extension .lst
FFLAGS += -hlist=a
CFLAGS += -hlist=a

# select the recognition of OpenACC OpenMP directives
ifeq ($(MODEL),ACC)
  FFLAGS += -hacc -Dopenacc
  CFLAGS += -hpragma=acc -Dopenacc
else
  FFLAGS += -xacc 
  CFLAGS += -hnopragma=acc
endif
ifeq ($(MODEL),OMP)
  FFLAGS += -homp -Dopenmp
  CFLAGS += -hpragma=omp -Dopenmp
else
  FFLAGS += -xomp
  CFLAGS += -hnopragma=omp
endif


ifeq ($(VERSION),00)
# There is no Cdynamic version 00 (see README.txt for details)
all: Fstatic Fdynamic Cstatic
else
all: Fstatic Fdynamic Cstatic Cdynamic
endif

Fstatic: first_example_Fstatic_v$(VERSION).f90
	@echo "Executing: $(FC) $(FFLAGS) -o first_example_Fstatic_v$(VERSION)_$(MODEL).x $^"
	$(FC) $(FFLAGS) -o first_example_Fstatic_v$(VERSION)_$(MODEL).x $^

Fdynamic: first_example_Fdynamic_v$(VERSION).f90
	@echo "Executing: $(FC) $(FFLAGS) -o first_example_Fdynamic_v$(VERSION)_$(MODEL).x $^"
	$(FC) $(FFLAGS) -o first_example_Fdynamic_v$(VERSION)_$(MODEL).x $^

Cstatic: first_example_Cstatic_v$(VERSION).c
	@echo "Executing: $(CC) $(CFLAGS) -o first_example_Cstatic_v$(VERSION)_$(MODEL).x $^"
	$(CC) $(CFLAGS) -o first_example_Cstatic_v$(VERSION)_$(MODEL).x $^

Cdynamic: first_example_Cdynamic_v$(VERSION).c
	@echo "Executing: $(CC) $(CFLAGS) -o first_example_Cdynamic_v$(VERSION)_$(MODEL).x $^"
	$(CC) $(CFLAGS) -o first_example_Cdynamic_v$(VERSION)_$(MODEL).x $^

clean:
	@$(RM) *.cub *.ptx *.lst *.gpu *.mod *.x *~
