CC=pgcc
CFLAGS=-fast -mp -ta=tesla:pinned,cc60,cuda8 -Minfo=accel
CFLAGS=-fast -mp -Minfo=all
CFLAGS=-fast -mp -ta=tesla,cc60,cuda8 -Minfo=all

CXX=pgc++
CXXFLAGS=-fast -mp -ta=tesla:pinned,cc60,cuda8 -Minfo=accel
CXXFLAGS=-fast -mp -Minfo=all
CXXFLAGS=-fast -mp -ta=tesla,cc60,cuda8 -Minfo=all
MPICXX=mpic++

LDFLAGS=-L$(EBROOTOPENCV)/lib


all: filter.x

filter.x: filter.o main.o
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -lm -o $@ -lopencv_highgui -lopencv_core -lopencv_imgcodecs

filter_mpi.x: filter_mpi.o main_mpi.o
	$(MPICXX) $(CXXFLAGS) $^ $(LDFLAGS) -lm -o $@ -lopencv_highgui -lopencv_core -lopencv_imgcodecs

main_mpi.o: main_mpi.cpp
	$(MPICXX) $(CXXFLAGS) -c -o $@ $<

.SUFFIXES: .o .cpp .c
.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $<
.c.o:
	$(CC) $(CFLAGS) -c -o $@ $<
.PHONY: clean
clean:
	-rm -f *.o core filter.x filter_mpi.x
