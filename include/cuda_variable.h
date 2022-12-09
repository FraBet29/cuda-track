#ifndef CUDA_VARIABLE_H

__global__ void zero_parallel(float *data, int size);
__device__ float warp_reduce(float val);
__global__ void grad_norm_parallel(int *in, int *out, int size);

struct CudaVariable {
    float *data, *grad;
    int size;
    float *local_grad; // ?
    CudaVariable(int size, bool requires_grad=true, bool thread_local_grad=false);
    ~CudaVariable();
    void glorot(int in_size, int out_size);
    void zero();
    void zero_grad();
    float grad_norm();
};


#define CUDA_VARIABLE_H
#endif