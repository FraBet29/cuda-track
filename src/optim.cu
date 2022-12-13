#include "../include/optim.h"
#include <cmath>
#include <cstdlib>
#include "../include/cuda_check.h"

#define MAX_NUM_THREADS 1024

AdamParams AdamParams::get_default() {
    return {0.001, 0.9, 0.999, 1e-8, 0.0};
}

AdamVariable::AdamVariable(Variable *var, bool decay):
    data(&var->data), grad(&var->grad), m(var->data.size(), 0.0), v(var->data.size(), 0.0), decay(decay) {}

int AdamVariable::size() {
    return data->size();
}

CudaAdamVariable::CudaAdamVariable(CudaVariable *var, bool decay):
    data(var->data), grad(var->grad), data_size(var->size), decay(decay) {
        std::vector<float> temp(var->size, 0.0f);
        check_call(cudaMalloc(m, var->size * sizeof(float)));
        check_call(cudaMalloc(v, var->size * sizeof(float)));
        check_call(cudaMemcpy(m, temp.data(), var->size * sizeof(float), cudaMemcpyHostToDevice));
        check_call(cudaMemcpy(v, temp.data(), var->size * sizeof(float), cudaMemcpyHostToDevice));
    }

int CudaAdamVariable::size() {
    return data_size;
}

Adam::Adam(std::vector<std::pair<Variable*, bool>> vars, std::vector<std::pair<CudaVariable*, bool>> cuda_vars, AdamParams params) {
    step_count = 0;
    this->params = params;
    for (auto v: vars)
        this->vars.emplace_back(v.first, v.second);
    for (auto v: cuda_vars)
        this->cuda_vars.emplace_back(v.first, v.second);
}

__global__ void adam_step_parallel(float *data, float *grad, float *m, float *v, bool decay, float step_size, float weight_decay, float beta1, float beta2, float eps) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    float grad_i = grad[i];
    if (decay) grad_i += weight_decay * data[i];
    m[i] = beta1 * m[i] + (1.0f - beta1) * grad_i;
    v[i] = beta2 * v[i] + (1.0f - beta2) * grad_i * grad_i;
    data[i] -= step_size * m[i] / (sqrtf(v[i]) + eps);
}

void Adam::step() {
    step_count++;
    float step_size = params.lr * sqrtf(1 - powf(params.beta2, step_count)) / (1 - powf(params.beta1, step_count));
    for (auto &var: cuda_vars) {
        // GPU blocks and threads settings
        dim3 blocksPerGrid1((var.size() + MAX_NUM_THREADS - 1) / MAX_NUM_THREADS, 1, 1);
        dim3 threadsPerBlock(MAX_NUM_THREADS, 1, 1);
        adam_step_parallel<<<blocksPerGrid1, threadsPerBlock>>>(var.data, var.grad, var.m, var.v, var.decay, step_size, params.weight_decay, params.beta1, params.beta2, params.eps);
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
