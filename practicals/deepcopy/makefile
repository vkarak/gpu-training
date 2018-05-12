.PHONY: all clean


FC = pgfortran
FCFLAGS = -g -O3 -acc -Minfo=accel -ta=tesla:cc60 -Mcuda
CXXFLAGS = -g -O3 -std=c++11 -acc -Minfo=accel -ta=tesla:cc60 -Mcuda
CXX = pgc++
LDLFAGS =

PROGRAMS = deepcopy.openacc.fort deepcopy.openacc

all: $(PROGRAMS)

deepcopy.openacc.fort: deepcopy.f90
	$(FC) $(FCFLAGS) $(LDFLAGS) $< -o $@

deepcopy.openacc: deepcopy.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $< -o $@

clean:
	rm -f $(PROGRAMS) *.cub *.ptx *.o *.mod output.*
