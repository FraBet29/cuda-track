CXX=nvcc
CXXFLAGS=-O3
LDFLAGS=-lm -lstdc++

CXXFILES=src/gcn.cu src/optim.cu src/module.cu src/parser.cpp src/timer.cpp src/cuda_check.cu src/cuda_rand.cu src/cuda_variable.cu src/cuda_sparse.cu
HFILES=include/gcn.h include/optim.h include/module.h include/sparse.h include/parser.h include/timer.h include/cuda_check.h include/cuda_rand.h include/cuda_variable.h include/cuda_sparse.h


all: gcn-seq

gcn-seq: src/main.cu $(CXXFILES) $(HFILES)
	mkdir exec
	$(CXX) $(CXXFLAGS) -o exec/gcn-seq $(CXXFILES) src/main.cu $(LDFLAGS)

clean:
	rm exec/*

