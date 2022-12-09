#include "../include/cuda_variable.h"
#include "../include/rand.h"
#include "../include/cuda_check.h"
#include <cmath>
#include <cstdio>
#include <algorithm>

#define MAX_NUM_THREADS 1024


CudaVariable::CudaVariable(int size, bool requires_grad, bool thread_local_grad): size(size) {
    check_call(cudaMalloc(&data, size * sizeof(float)));
    if (requires_grad)
        check_call(cudaMalloc(&grad, size * sizeof(float)));
}

CudaVariable::~CudaVariable() {
    check_call(cudaFree(data));
    check_call(cudaFree(grad));
    check_call(cudaFree(local_grad));
}

/**
 * Glorot (Xavier) method for weights initialization
 * CAN BE PARALLELIZED (CUDA RANDOM GENERATOR NEEDED)
*/
void CudaVariable::glorot(int in_size, int out_size) {
    float range = sqrtf(6.0f / (in_size + out_size));
    float *temp_data = (float *) malloc(size * sizeof(float));
    for(int i = 0; i < size; ++i)
        temp_data[i] = (float(RAND()) / MY_RAND_MAX - 0.5) * range * 2;
    check_call(cudaMemcpy(data, temp_data, size * sizeof(float), cudaMemcpyHostToDevice));
    free(temp_data);
}

__global__ void zero(float *data, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size)
        data[i] = 0.0f;
}

void CudaVariable::zero() {
    zero<<<(size + MAX_NUM_THREADS - 1) / MAX_NUM_THREADS, MAX_NUM_THREADS>>>(data, size);
    check_kernel_call();
    cudaDeviceSynchronize();
}

void CudaVariable::zero_grad() {
    zero<<<(size + MAX_NUM_THREADS - 1) / MAX_NUM_THREADS, MAX_NUM_THREADS>>>(grad, size);
    check_kernel_call();
    cudaDeviceSynchronize();
}

// Reduction via warps
__global__ void grad_norm(int *in, int *out, int size) {
    int warp_size = 32;
    float sum = float(0);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        sum += in[i] * in[i];
    sum = warp_reduce(sum);
    if ((threadIdx.x & (warp_size - 1)) == 0)
        atomicAdd(out, sum);
}

float CudaVariable::grad_norm() {
    float norm;
    float *cuda_norm;
    check_call(cudaMalloc(&cuda_norm, sizeof(float)));
    grad_norm<<<(size + MAX_NUM_THREADS - 1) / MAX_NUM_THREADS, MAX_NUM_THREADS>>>(grad, cuda_norm, size);
    check_kernel_call();
    cudaDeviceSynchronize();
    check_call(cudaMemcpy(norm, cuda_norm, sizeof(float), cudaMemcpyDeviceToHost));
    check_call(cudaFree(cuda_norm));
    return sqrtf(norm);
}