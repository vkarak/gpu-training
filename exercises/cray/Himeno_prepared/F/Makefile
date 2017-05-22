# Some default options
# These can be overridden via the command line when calling make
# e.g. "make VERSION=01 ACC=yes"
# Do not set both OMP and ACC to yes, as compilation will fail
VERSION=00
OMP=no
NTPB=128

PROBLEM_SIZE=3
PRECISION=double

# CCE compilation options
PROFILE_GENERATE=no
ASYNC=yes

# You should not need to change anything below this line
FC = ftn

FFLAGS=-DPROBLEM_SIZE=$(PROBLEM_SIZE) -DNTPB=$(NTPB)

ifeq ($(PRECISION),double)
 FFLAGS += -DDOUBLE_PRECISION
 ifeq ($(PROBLEM_SIZE),4)
  FFLAGS += -fpic
 endif
endif

ifeq ($(PE_ENV),CRAY)
 FFLAGS += -hlist=ad

ifneq ($(OMP),yes)
  ifneq ($(OMP_DEV),yes)
    FFLAGS += -hnoomp 
  endif
endif

 ifeq ($(ASYNC),no)
  FFLAGS += -hacc_model=auto_async_none
 endif

 ifeq ($(PROFILE_GENERATE),yes)
  FFLAGS += -hprofile_generate
 endif

endif

ifeq ($(PE_ENV),PGI)
 FFLAGS += -Minfo=all,ftn -Mpreprocess -fpic 

 ifeq ($(OMP),yes) 
  FFLAGS += -mp
 endif
endif

%.o: %.F90
	$(FC) $(FFLAGS) -c $<

himeno_v$(VERSION).x: himeno_F_v$(VERSION).o
	$(FC) $(FFLAGS) -o $@ $<

clean:
	@$(RM) himeno_v*.x himeno_v*.x+pat 
	@$(RM) *.lst *.i *.ptx *.cub *.gpu *.cg *.opt *.o *~

#EOF
