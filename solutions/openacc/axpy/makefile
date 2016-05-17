.PHONY: all clean

CRAY ?= 1
PGI  ?= 0
GCC  ?= 0
VERBOSE ?= 0

ifeq ($(CRAY), 1)
CXX = CC
FC = ftn
CXXFLAGS = -hstd=c++11 -O3 -homp -hacc -dynamic
FCFLAGS  = -O3 -homp -hacc -dynamic
ifeq ($(VERBOSE), 1)
CXXFLAGS += -hmsgs
FCFLAGS  += -hmsgs
endif
endif

ifeq ($(PGI), 1)
CXX = pgc++
CXXFLAGS = -std=c++11 -O3 -acc
endif

ifeq ($(GCC), 1)
CXX = g++
CXXFLAGS = -std=c++11 -O3 -fopenacc
endif

OPENACC_BENCHMARKS= axpy.openacc axpy.openacc.fort

BENCHMARKS=$(OPENACC_BENCHMARKS)

all: $(BENCHMARKS)

axpy.openacc: axpy_openacc.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

axpy.openacc.fort: axpy_openacc.f90 util.f03
	$(FC) -c $(FCFLAGS) util.f03
	$(FC) $(FCFLAGS) $< util.o -o $@

clean:
	rm -f $(BENCHMARKS) *.cub *.ptx *.o *.mod
