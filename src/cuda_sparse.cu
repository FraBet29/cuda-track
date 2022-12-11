#include "../include/cuda_sparse.h"
#include "../include/cuda_check.h"

CudaSparseIndex::CudaSparseIndex(int *indices, int *indptr, int indices_size, int indptr_size):
        indices_size(indices_size), indptr_size(indptr_size) {
            check_call(cudaMalloc(&this->indptr, indptr_size * sizeof(int)));
            check_call(cudaMalloc(&this->indices, indices_size * sizeof(int)));
            check_call(cudaMemcpy(this->indptr, indptr, indptr_size * sizeof(int), cudaMemcpyHostToDevice));
            check_call(cudaMemcpy(this->indices, indices, indices_size * sizeof(int), cudaMemcpyHostToDevice));
}