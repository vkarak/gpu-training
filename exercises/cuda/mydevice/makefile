# --- daint:
# module swap PrgEnv-cray PrgEnv-gnu
# module load craype-accel-nvidia60
# module list -t
# ---
HH 	    := $(shell uname -n |cut -c1-3 )
CC      := cc
NVCC    := nvcc
CFLAGS  := -g
CUFLAGS := -G
LDFLAGS := $(CFLAGS)
OBJ     := mpic.o mpicu.o
EXECUTABLE := exe.$(HH)

ifneq ($(PE_ENV),GNU)
    $(error module swap PrgEnv-cray PrgEnv-gnu # PE_ENV=$(PE_ENV))
endif

ifeq ($(CUDATOOLKIT_HOME),)
    $(error module load craype-accel-nvidia60)
endif

ifeq ($(CRAY_ACCEL_TARGET),nvidia35)
    CUDAFLAGS += -arch=sm_35
endif

ifeq ($(CRAY_ACCEL_TARGET),nvidia60)
    CUDAFLAGS += -arch=sm_60
endif

SRCc   = mpic.c
SRCcu  = mpicu.cu
OBJC  := $(SRCc:.c=.o)
OBJCU := $(SRCcu:.cu=.o)

%.o: %.c
	$(CC) $(CFLAGS) -c $< 

%.o: %.cu
	$(NVCC) $(CUDAFLAGS) $(CFLAGS) -c $<

$(EXECUTABLE): $(OBJC) $(OBJCU)
	$(CC) $(LDFLAGS) $(OBJCU) $(OBJC) -o $@

clean:
	rm -f $(OBJC) $(OBJCU) $(EXECUTABLE)

