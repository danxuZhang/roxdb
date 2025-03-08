CXX ?= g++
LD ?= $(CXX)

CXXFLAGS = -std=c++20 -Wall -Wextra -O3 -g

AVX2_CXXFLAGS = $(CXXFLAGS) -mavx2 -mfma
AVX512F_CXXFLAGS = $(CXXFLAGS) -mavx512f 
SCALAR_CXXFLAGS = $(CXXFLAGS) -fno-tree-vectorize 

all: avx2 avx512f scalar main

avx2:
	$(CXX) -c $(AVX2_CXXFLAGS) -o avx2.o avx2.cc

avx512f:
	$(CXX) -c $(AVX512F_CXXFLAGS) -o avx512.o avx512.cc

scalar:
	$(CXX) -c $(SCALAR_CXXFLAGS) -o scalar.o scalar.cc

main:
	$(CXX) $(CXXFLAGS) -fopenmp -o main main.cc scalar.o avx2.o avx512.o

clean:
	rm -f *.o main
