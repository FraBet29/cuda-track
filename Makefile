#CXX=gcc
CXX=nvcc
#CXXFLAGS=-no-pie -O3 -g -pg -std=c++11 -Wall -Wno-sign-compare -Wno-unused-variable -Wno-unknown-pragmas
CXXFLAGS=-O3
LDFLAGS=-lm -lstdc++

CXXFILES=src/gcn.cpp src/optim.cpp src/module.cu src/variable.cpp src/parser.cpp src/rand.cpp src/timer.cpp src/cuda_check.cu
HFILES=include/gcn.h include/optim.h include/module.h include/variable.h include/sparse.h include/parser.h include/rand.h include/timer.h include/cuda_check.h


all: gcn-seq

gcn-seq: src/main.cpp $(CXXFILES) $(HFILES)
	mkdir exec
	$(CXX) $(CXXFLAGS) -o exec/gcn-seq $(CXXFILES) src/main.cpp $(LDFLAGS)

clean:
	rm exec/*

