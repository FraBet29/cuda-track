#ifndef CUDA_CHECK_H
#include <iostream>

void check_call(cudaError_t call);
void check_kernel_call();

#define CUDA_CHECK_H
#endif
