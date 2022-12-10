#ifndef CUDA_SPARSE_H

struct CudaSparseIndex {
    int *indices;
    int *indptr;
};


#define CUDA_SPARSE_H
#endif