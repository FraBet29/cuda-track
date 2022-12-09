#ifndef CUDA_VARIABLE_H

struct CudaVariable {
    float* data, grad;
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