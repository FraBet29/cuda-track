#ifndef CUDA_SPARSE_H

class CudaSparseIndex {
public:
    int *indices;
    int *indptr;
    int indices_size;
    int indptr_size;
    CudaSparseIndex() {};
    CudaSparseIndex(int *indices, int *indptr, int indices_size, int indptr_size);
    ~CudaSparseIndex();
};

#define CUDA_SPARSE_H
#endif