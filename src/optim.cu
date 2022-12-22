#include "../include/optim.h"
#include <cmath>
#include <cstdlib>
#include "../include/cuda_check.h"

#define MAX_NUM_THREADS 1024

AdamParams AdamParams::get_default() {
    return {0.001, 0.9, 0.999, 1e-8, 0.0};
}

CudaAdamVariable::CudaAdamVariable(CudaVariable *var, bool decay):
    data(var->data), grad(var->grad), size(var->size), decay(decay) {
        std::vector<float> temp(var->size, 0.0f);
        check_call(cudaMalloc(&m, var->size * sizeof(float)));
        check_call(cudaMalloc(&v, var->size * sizeof(float)));
        check_call(cudaMemcpy(m, temp.data(), var->size * sizeof(float), cudaMemcpyHostToDevice));
        check_call(cudaMemcpy(v, temp.data(), var->size * sizeof(float), cudaMemcpyHostToDevice));
    }

CudaAdamVariable::~CudaAdamVariable() {
    std::cout << "Deallocating CudaAdamVariable" << std::endl;
    if(m) std::cout << &(*m) << std::endl;
    if(v) std::cout << &(*v) << std::endl;
    check_call(cudaFree(m));
    check_call(cudaFree(v));
}

Adam::Adam(std::vector<std::pair<CudaVariable*, bool>> cuda_vars, AdamParams params) {
    std::cout << "Initializing Adam" << std::endl;
    step_count = 0;
    this->params = params;
    this->cuda_vars.reserve(cuda_vars.size());
    for (auto v: cuda_vars) {
        std::cout << "ok 3" << std::endl;
        this->cuda_vars.emplace_back(v.first, v.second);
        std::cout << "ok 4" << std::endl;
    }
    std::cout << "ok 5" << std::endl;
}

__global__ void adam_step_parallel(float *data, float *grad, float *m, float *v, bool decay, float step_size, float weight_decay, float beta1, float beta2, float eps, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        float grad_i = grad[i];
        if (decay) grad_i += weight_decay * data[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * grad_i;
        v[i] = beta2 * v[i] + (1.0f - beta2) * grad_i * grad_i;
        data[i] -= step_size * m[i] / (sqrtf(v[i]) + eps);
    }
}

void Adam::step() {
    step_count++;
    float step_size = params.lr * sqrtf(1 - powf(params.beta2, step_count)) / (1 - powf(params.beta1, step_count));
    for (auto &var: cuda_vars) {
        // GPU blocks and threads settings
        dim3 blocksPerGrid((var.size + MAX_NUM_THREADS - 1) / MAX_NUM_THREADS, 1, 1);
        dim3 threadsPerBlock(MAX_NUM_THREADS, 1, 1);
        adam_step_parallel<<<blocksPerGrid, threadsPerBlock>>>(var.data, var.grad, var.m, var.v, var.decay, step_size, params.weight_decay, params.beta1, params.beta2, params.eps, var.size);
        check_kernel_call();
    }
    cudaDeviceSynchronize();
    /*
    step_count++;
    float step_size = params.lr * sqrtf(1 - powf(params.beta2, step_count)) / (1 - powf(params.beta1, step_count));
    for (auto &var: vars) {
        for (int i = 0; i < var.size(); i++) {
            float grad = (*var.grad)[i];
            if (var.decay) grad += params.weight_decay * (*var.data)[i];
            var.m[i] = params.beta1 * var.m[i] + (1.0 - params.beta1) * grad;
            var.v[i] = params.beta2 * var.v[i] + (1.0 - params.beta2) * grad * grad;
            (*var.data)[i] -= step_size * var.m[i] / (sqrtf(var.v[i]) + params.eps);
        }
    }
    */
}
