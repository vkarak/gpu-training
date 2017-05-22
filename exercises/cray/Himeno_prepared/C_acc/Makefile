# Some default options
# These can be overridden via the command line when calling make
# e.g. "make VERSION=01 ACC=yes"
# Do not set both OMP and ACC to yes, as compilation will fail
VERSION=00
OMP=no
ACC=yes
NTPB=128

PROBLEM_SIZE=3
PRECISION=single

# CCE compilation options
PROFILE_GENERATE=no
ACC_MODEL=default

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

 ifneq ($(ACC),yes)
  CFLAGS += -hnopragma=acc
 endif

 ifeq ($(PROFILE_GENERATE),yes)
  CFLAGS += -hprofile_generate
 endif

 ifneq ($(ACC_MODEL),default)
  CFLAGS += -hacc_model=$(ACC_MODEL)
 endif
endif

ifeq ($(PE_ENV),PGI)
 CFLAGS += -Minfo=all -Mpreprocess -fpic 

 ifeq ($(OMP),yes) 
  CFLAGS += -mp
 endif

 ifeq ($(ACC),yes)
# craype-accel-* module sets the correct cuda level
# keepgpu keeps the intermediate CUDA (not very readable)
# keepptx keeps the ptx files
  CFLAGS += -ta=nvidia,keepgpu,keepptx
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
