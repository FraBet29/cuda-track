#include "../include/cuda_variable.h"
#include "../include/cuda_check.h"
#include "../include/cuda_rand.h"
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
    //check_call(cudaFree(data));
    //check_call(cudaFree(grad));
    //check_call(cudaFree(local_grad));
}

/**
 * Glorot (Xavier) method for weights initialization
 * WHAT IS WRONG WITH THE PARALLEL VERSION? CHANGE SEED FOR EVERY ALLOCATED WEIGHT!
*/

__global__ void glorot_parallel(float *data, float range, int size, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size)
        data[i] = (curand_uniform(&rand_state[i]) / MY_RAND_MAX - 0.5) * range * 2;
}

void CudaVariable::glorot(int in_size, int out_size) {
    curandState *cuda_rand_state;
    // Initialize CUDA random
    check_call(cudaMalloc(&cuda_rand_state, size * sizeof(curandState)));
    rand_setup_kernel<<<(size + MAX_NUM_THREADS - 1) / MAX_NUM_THREADS, MAX_NUM_THREADS>>>(cuda_rand_state, size, in_size + out_size);
    check_kernel_call();
    cudaDeviceSynchronize();
    float range = sqrtf(6.0f / (in_size + out_size)); 
    glorot_parallel<<<(size + MAX_NUM_THREADS - 1) / MAX_NUM_THREADS, MAX_NUM_THREADS>>>(data, range, size, cuda_rand_state);
    check_kernel_call();
    cudaDeviceSynchronize();
    /*
    float range = sqrtf(6.0f / (in_size + out_size));
    float *temp_data = (float *) malloc(size * sizeof(float));
    for(int i = 0; i < size; ++i)
        temp_data[i] = (float(RAND()) / MY_RAND_MAX - 0.5) * range * 2;
    check_call(cudaMemcpy(data, temp_data, size * sizeof(float), cudaMemcpyHostToDevice));
    free(temp_data);
    */
}

__global__ void zero_parallel(float *data, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size)
        data[i] = 0.0f;
}

void CudaVariable::zero() {
    zero_parallel<<<(size + MAX_NUM_THREADS - 1) / MAX_NUM_THREADS, MAX_NUM_THREADS>>>(data, size);
    check_kernel_call();
    cudaDeviceSynchronize();
}

void CudaVariable::zero_grad() {
    zero_parallel<<<(size + MAX_NUM_THREADS - 1) / MAX_NUM_THREADS, MAX_NUM_THREADS>>>(grad, size);
    check_kernel_call();
    cudaDeviceSynchronize();
}

// Reduction via warps
__device__ float warp_reduce(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void grad_norm_parallel(float *in, float *out, int size) {
    int warp_size = 32;
    float sum = 0.0f;
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
    grad_norm_parallel<<<(size + MAX_NUM_THREADS - 1) / MAX_NUM_THREADS, MAX_NUM_THREADS>>>(grad, cuda_norm, size);
    check_kernel_call();
    cudaDeviceSynchronize();
    check_call(cudaMemcpy(&norm, cuda_norm, sizeof(float), cudaMemcpyDeviceToHost));
    check_call(cudaFree(cuda_norm));
    return sqrtf(norm);
}