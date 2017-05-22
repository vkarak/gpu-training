# Some default options
# These can be overridden via the command line when calling make
# e.g. "make VERSION=01 ACC=yes"
# Do not set both OMP and ACC to yes, as compilation will fail
VERSION=00
OMP=yes
NTPB=128

PROBLEM_SIZE=3
PRECISION=double

# CCE compilation options
PROFILE_GENERATE=no
ASYNC=yes

# You should not need to change anything below this line
CC = cc

CFLAGS=-DPROBLEM_SIZE=$(PROBLEM_SIZE) -DNTPB=$(NTPB)

ifeq ($(PRECISION),double)
 CFLAGS += -DDOUBLE_PRECISION
 ifeq ($(PROBLEM_SIZE),4)
  CFLAGS += -fpic
 endif
endif

ifeq ($(PE_ENV),CRAY)
 CFLAGS += -hlist=ad

 ifneq ($(OMP),yes) 
  CFLAGS += -hnoomp
 endif

 ifeq ($(ASYNC),no)
  FFLAGS += -hacc_model=auto_async_none
 endif

 ifeq ($(PROFILE_GENERATE),yes)
  CFLAGS += -hprofile_generate
 endif
endif

ifeq ($(PE_ENV),PGI)
 CFLAGS += -Minfo=all -Mpreprocess -fpic 

 ifeq ($(OMP),yes) 
  CFLAGS += -mp
 endif
endif

%.o: %.c
	$(CC) $(CFLAGS) -c $<

himeno_v$(VERSION).x: himeno_C_v$(VERSION).o
	$(CC) $(CFLAGS) -o $@ $<

clean:
	@$(RM) himeno_v*.x himeno_v*.x+pat
	@$(RM) *.lst *.i *.ptx *.cub *.gpu *.cg *.opt *.o *~

#EOF
