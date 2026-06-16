CXX = g++
KOKKOS_INSTALL = /home/copilot/kokkos-install
CXXFLAGS = -std=c++17 -I$(KOKKOS_INSTALL)/include -O3
LDFLAGS = -L$(KOKKOS_INSTALL)/lib -lkokkoscore -lkokkoscontainers -lpthread -ldl

all: main

main: main.o
	$(CXX) $(CXXFLAGS) main.o -o main $(LDFLAGS)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

clean:
	rm -f main.o main
