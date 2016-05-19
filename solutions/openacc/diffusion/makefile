.PHONY: all clean

CRAY ?= 1
PGI  ?= 0
GCC  ?= 0
VERBOSE ?= 0

ifeq ($(CRAY), 1)
CXX = CC
FC  = ftn
CXXFLAGS = -hstd=c++11 -O3 -hacc -dynamic
FCFLAGS  = -O3 -homp -hacc -dynamic
ifeq ($(VERBOSE), 1)
CXXFLAGS += -hmsgs
FCFLAGS  += -hmsgs
endif
endif

ifeq ($(PGI), 1)
CXX = pgc++
CXXFLAGS = -std=c++11 -O3 -acc -Minfo=accel
endif

ifeq ($(GCC), 1)
CXX = g++
CXXFLAGS = -std=c++11 -O3 -fopenacc
endif

OPENACC_BENCHMARKS= diffusion2d.openacc diffusion2d.openacc.cuda diffusion2d.omp
BENCHMARKS=$(OPENACC_BENCHMARKS)

all: $(BENCHMARKS)

util.mod: util.f03
	$(FC) -c $(FCFLAGS) util.f03

diffusion2d.openacc: diffusion2d_openacc.cpp diffusion2d.hpp
	$(CXX) $(CXXFLAGS) -DOPENACC_DATA $< -o $@ $(LDFLAGS)

diffusion2d.openacc.cuda: diffusion2d_openacc.cpp diffusion2d.hpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

diffusion2d.omp: diffusion2d_omp.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

diffusion2d.openacc.fort: diffusion2d_openacc.f90 util.mod
	$(FC) $(FCFLAGS) $< util.o -o $@

clean:
	rm -f $(BENCHMARKS) *.cub *.ptx *.o *.mod
