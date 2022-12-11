#ifndef CUDA_SPARSE_H

class CudaSparseIndex {
public:
    CudaSparseIndex(int *indices, int *indptr, int indices_size, int indptr_size);
    ~CudaSparseIndex();
    int *indices;
    int *indptr;
    int indices_size;
    int indptr_size;
};


#define CUDA_SPARSE_H
#endif