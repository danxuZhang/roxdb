CXX ?= g++
LD ?= $(CXX)

CXXFLAGS = -std=c++20 -Wall -Wextra -O3 -g

AVX2_CXXFLAGS = $(CXXFLAGS) -mavx2 -mfma -D__AVX2__ 
AVX512F_CXXFLAGS = $(CXXFLAGS) -mavx512f  -D__AVX512F__
SCALAR_CXXFLAGS = $(CXXFLAGS) -fno-tree-vectorize 

all: avx2 avx512f scalar main

avx2:
	$(CXX) -c $(AVX2_CXXFLAGS) -o avx2.o avx2.cc

avx512f:
	$(CXX) -c $(AVX512F_CXXFLAGS) -o avx512f.o avx512.cc

scalar:
	$(CXX) -c $(SCALAR_CXXFLAGS) -o scalar.o scalar.cc

main:
	$(CXX) $(CXXFLAGS) -fopenmp -o main main.cc scalar.o avx2.o avx512f.o

perf: scalar avx2 avx512f
	$(CXX) $(CXXFLAGS) -o perf_scalar perf.cc scalar.o
	$(CXX) $(CXXFLAGS) -D__AVX512F__ -o perf_avx512f perf.cc avx512f.o 
	$(CXX) $(CXXFLAGS) -D__AVX2__ -o perf_avx2 perf.cc avx2.o

run_perf: 
	perf stat -e cache-references,cache-misses,cycles,instructions,branches,branch-misses,L1-dcache-load-misses,L1-dcache-loads,LLC-load-misses,LLC-loads \
		./perf_scalar
	perf stat -e cache-references,cache-misses,cycles,instructions,branches,branch-misses,L1-dcache-load-misses,L1-dcache-loads,LLC-load-misses,LLC-loads \
		./perf_avx2
	perf stat -e cache-references,cache-misses,cycles,instructions,branches,branch-misses,L1-dcache-load-misses,L1-dcache-loads,LLC-load-misses,LLC-loads \
		./perf_avx512f

clean:
	rm -f *.o main perf_*
