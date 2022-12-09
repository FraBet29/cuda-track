#ifndef CUDA_SPARSE_H

class CudaSparseIndex {
public:
    int *indices;
    int *indptr;
};


#define CUDA_SPARSE_H
#endif