#include "../include/module.h"
#include "../include/timer.h"
#include "../include/cuda_check.h"
#include "../include/cuda_rand.h"
#include "../include/gpu_params.h"
#include <cmath>
#include <iostream>

// ################################################################################################################
/**
 * Dense matrix multiplication layer. 
*/
Matmul::Matmul(CudaVariable *cuda_a, CudaVariable *cuda_b, CudaVariable *cuda_c, int m, int n, int p) : 
        cuda_a(cuda_a), cuda_b(cuda_b), cuda_c(cuda_c), m(m), n(n), p(p) {}

__global__ void matmul_forward_parallel(float *a, float *b, float *c, int m, int n, int p, int TILE_SIZE) {
    // Multiplication of matrices A and B; the result is stored in the matrix C
    extern __shared__ float a_tile[], b_tile[];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < m && j < p) {
        float sum = 0.0f;
        for (int k = 0; threadIdx.x + k < n; k += TILE_SIZE) {
            a_tile[threadIdx.y * n + threadIdx.x + k] = a[i * n + threadIdx.x + k];
            b_tile[(threadIdx.y + k) * p + threadIdx.x] = b[(threadIdx.y + k) * p + j];
        }
        __syncthreads();
        for (int k = 0; k < n; ++k)
            sum += a_tile[i * n + k] * b_tile[k * p + j];
        c[i * p + j] = sum;
    }
}

void Matmul::forward(bool training) {
    timer_start(TMR_MATMUL_FW);
    cuda_c->zero();
    // GPU blocks and threads settings
    dim3 blocksPerGrid((m + MAX_THREADS_PER_BLOCK_2D - 1) / MAX_THREADS_PER_BLOCK_2D, (p + MAX_THREADS_PER_BLOCK_2D - 1) / MAX_THREADS_PER_BLOCK_2D, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_2D, MAX_THREADS_PER_BLOCK_2D, 1); // 2D tiles
    // Launch kernel
    int sharedMemorySize = 2 * MAX_THREADS_PER_BLOCK_2D * n;
    if (sharedMemorySize * sizeof(float) > SHARED_MEMORY_PER_BLOCK)
        std::cerr << "The size of the data exceeds the size of available shared memory per block." << std::endl;
    matmul_forward_parallel<<<blocksPerGrid, threadsPerBlock, sharedMemorySize * sizeof(float)>>>(cuda_a->data, cuda_b->data, cuda_c->data, m, n, p, MAX_THREADS_PER_BLOCK_2D);
    check_kernel_call();
    cudaDeviceSynchronize();
   /*
    c->zero();
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < p; k++)
                c->data[i * p + k] += a->data[i * n + j] * b->data[j * p + k];
        }
    */
    timer_stop(TMR_MATMUL_FW);
}

__global__ void matmul_backward_parallel(float *a_data, float *b_data, float *a_grad, float *b_grad, float *c_grad, int m, int n, int p) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < m && j < n) {
        float tmp = 0;
        for (int k = 0; k < p; k++) {
            tmp += c_grad[i * p + k] * b_data[j * p + k];
            atomicAdd(&b_grad[j * p + k], c_grad[i * p + k] * a_data[i * n + j]); // TRY TO AVOID ATOMIC ADD
        }
		a_grad[i * n + j] = tmp;
    }
}

void Matmul::backward() {
    timer_start(TMR_MATMUL_BW);
    cuda_a->zero_grad();
    cuda_b->zero_grad();
    // GPU blocks and threads settings
    dim3 blocksPerGrid((m + MAX_THREADS_PER_BLOCK_2D - 1) / MAX_THREADS_PER_BLOCK_2D, (n + MAX_THREADS_PER_BLOCK_2D - 1) / MAX_THREADS_PER_BLOCK_2D, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_2D, MAX_THREADS_PER_BLOCK_2D, 1);
    // Launch kernel
    matmul_backward_parallel<<<blocksPerGrid, threadsPerBlock>>>(cuda_a->data, cuda_b->data, cuda_a->grad, cuda_b->grad, cuda_c->grad, m, n, p);
    check_kernel_call();
    cudaDeviceSynchronize();
   /*
    a->zero_grad();
    b->zero_grad();
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float tmp = 0;
            for (int k = 0; k < p; k++) {
                tmp += c->grad[i * p + k] * b->data[j * p + k];
                b->grad[j * p + k] += c->grad[i * p + k] * a->data[i * n + j];
            }
		    a->grad[i * n + j] = tmp;
        }
    */
    timer_stop(TMR_MATMUL_BW);
}


// ################################################################################################################

/**
 * A sparse matrix multiplication layer.
*/
SparseMatmul::SparseMatmul(CudaVariable *cuda_a, CudaVariable *cuda_b, CudaVariable *cuda_c, SparseIndex *sp, int m, int n, int p) :
        cuda_a(cuda_a), cuda_b(cuda_b), cuda_c(cuda_c), m(m), n(n), p(p) {
            CudaSparseIndex *cuda_sp_temp = new CudaSparseIndex(sp->indices.data(), sp->indptr.data(), sp->indices.size(), sp->indptr.size());
            cuda_sp = cuda_sp_temp;
        }

__global__ void sparsematmul_forward_parallel(float *a, float *b, float *c, int *indptr, int *indices, int N, int p) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && k < p) {
        for (int jj = indptr[i]; jj < indptr[i + 1]; jj++) {
            int j = indices[jj];
            c[i * p + k] += a[jj] * b[j * p + k];
        }
    }        
}

void SparseMatmul::forward(bool training) {
    timer_start(TMR_SPMATMUL_FW);
    cuda_c->zero();
    // GPU blocks and threads settings
    dim3 blocksPerGrid((m + MAX_THREADS_PER_BLOCK_2D - 1) / MAX_THREADS_PER_BLOCK_2D, (p + MAX_THREADS_PER_BLOCK_2D - 1) / MAX_THREADS_PER_BLOCK_2D, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_2D, MAX_THREADS_PER_BLOCK_2D, 1);
    // Launch kernel
    sparsematmul_forward_parallel<<<blocksPerGrid, threadsPerBlock>>>(cuda_a->data, cuda_b->data, cuda_c->data, cuda_sp->indptr, cuda_sp->indices, cuda_sp->indptr_size - 1, p);
    check_kernel_call();
    cudaDeviceSynchronize();
    /*
    c->zero();
    for (int i = 0; i < sp->indptr.size() - 1; i++)
        for (int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++) {
            int j = sp->indices[jj];
            for (int k = 0; k < p; k++)
                c->data[i * p + k] += a->data[jj] * b->data[j * p + k];
        }
    */
    timer_stop(TMR_SPMATMUL_FW);
}

__global__ void sparsematmul_backward_parallel(float *a_data, float *b_grad, float *c_grad, int *indptr, int *indices, int N, int p) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && k < p) {
        for (int jj = indptr[i]; jj < indptr[i + 1]; jj++) {
            int j = indices[jj];
            atomicAdd(&b_grad[j * p + k], c_grad[i * p + k] * a_data[jj]); // TRY TO AVOID ATOMIC ADD
        }
    }
}

void SparseMatmul::backward() {
    timer_start(TMR_SPMATMUL_BW);
    cuda_b->zero_grad();
    // GPU blocks and threads settings
    dim3 blocksPerGrid((m + MAX_THREADS_PER_BLOCK_2D - 1) / MAX_THREADS_PER_BLOCK_2D, (p + MAX_THREADS_PER_BLOCK_2D - 1) / MAX_THREADS_PER_BLOCK_2D, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_2D, MAX_THREADS_PER_BLOCK_2D, 1);
    // Launch kernel
    sparsematmul_backward_parallel<<<blocksPerGrid, threadsPerBlock>>>(cuda_a->data, cuda_b->grad, cuda_c->grad, cuda_sp->indptr, cuda_sp->indices, cuda_sp->indptr_size - 1, p);
    check_kernel_call();
    cudaDeviceSynchronize();
    /*
    b->zero_grad();
    for (int i = 0; i < sp->indptr.size() - 1; i++)
        for (int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++) {
            int j = sp->indices[jj];
            for (int k = 0; k < p; k++)
                    b->grad[j * p + k] += c->grad[i * p + k] * a->data[jj];
        }
    */
    timer_stop(TMR_SPMATMUL_BW);
}

// ################################################################################################################

/**
 * A specialized sparse matrix multiplication for graphs.
*/
GraphSum::GraphSum(CudaVariable *cuda_in, CudaVariable *cuda_out, SparseIndex *graph, int dim) :
        cuda_in(cuda_in), cuda_out(cuda_out), dim(dim) {
            CudaSparseIndex *cuda_graph_temp = new CudaSparseIndex(graph->indices.data(), graph->indptr.data(), graph->indices.size(), graph->indptr.size());
            cuda_graph = cuda_graph_temp;
        }

__global__ void graphsum_forward_parallel(float *in, float *out, int *indptr, int *indices, int N, int dim) {
    int src = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (src < N && j < dim) {
        for (int i = indptr[src]; i < indptr[src + 1]; i++) {
            int dst = indices[i];
            float coef = 1.0 / sqrtf(
                    (indptr[src + 1] - indptr[src]) * (indptr[dst + 1] - indptr[dst])
            );
            // This only works for undirected graphs. Should be out[dst] += coef * in[src]
            out[src * dim + j] += coef * in[dst * dim + j];
        }
    }        
}

void GraphSum::forward(bool training) {
    timer_start(TMR_GRAPHSUM_FW);
    cuda_out->zero();
    // GPU blocks and threads settings
    dim3 blocksPerGrid((cuda_graph->indptr_size - 1 + MAX_THREADS_PER_BLOCK_2D - 1) / MAX_THREADS_PER_BLOCK_2D, (dim + MAX_THREADS_PER_BLOCK_2D - 1) / MAX_THREADS_PER_BLOCK_2D, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_2D, MAX_THREADS_PER_BLOCK_2D, 1);
    // Launch kernel
    graphsum_forward_parallel<<<blocksPerGrid, threadsPerBlock>>>(cuda_in->data, cuda_out->data, cuda_graph->indptr, cuda_graph->indices, cuda_graph->indptr_size - 1, dim);
    check_kernel_call();
    cudaDeviceSynchronize();
    /*
    out->zero();
    for (int src = 0; src < graph->indptr.size() - 1; src++)
        for (int i = graph->indptr[src]; i < graph->indptr[src + 1]; i++) {
            int dst = graph->indices[i];
            float coef = 1.0 / sqrtf(
                    (graph->indptr[src + 1] - graph->indptr[src]) * (graph->indptr[dst + 1] - graph->indptr[dst])
            );
            for (int j = 0; j < dim; j++)
                // This only works for undirected graphs. Should be out[dst] += coef * in[src]
                out->data[src * dim + j] += coef * in->data[dst * dim + j];
        }
    */
    timer_stop(TMR_GRAPHSUM_FW);
}

void GraphSum::backward() {
    timer_start(TMR_GRAPHSUM_BW);
    cuda_in->zero_grad();
    // GPU blocks and threads settings
    dim3 blocksPerGrid((cuda_graph->indptr_size - 1 + MAX_THREADS_PER_BLOCK_2D - 1) / MAX_THREADS_PER_BLOCK_2D, (dim + MAX_THREADS_PER_BLOCK_2D - 1) / MAX_THREADS_PER_BLOCK_2D, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_2D, MAX_THREADS_PER_BLOCK_2D, 1);
    // Launch kernel
    // SAME EXACT CODE STRUCTURE AS GRAPHSUM FORWARD, BUT WITH GRADIENTS AND WITH IN AND OUT SWAPPED!
    graphsum_forward_parallel<<<blocksPerGrid, threadsPerBlock>>>(cuda_out->grad, cuda_in->grad, cuda_graph->indptr, cuda_graph->indices, cuda_graph->indptr_size - 1, dim);
    check_kernel_call();
    cudaDeviceSynchronize();
    /*
    in->zero_grad();
    for (int src = 0; src < graph->indptr.size() - 1; src++)
        for (int i = graph->indptr[src]; i < graph->indptr[src + 1]; i++) {
            int dst = graph->indices[i];
            float coef = 1.0 / sqrtf(
                    (graph->indptr[src + 1] - graph->indptr[src]) * (graph->indptr[dst + 1] - graph->indptr[dst])
            );
            for (int j = 0; j < dim; j++)
                in->grad[src * dim + j] += coef * out->grad[dst * dim + j];
        }
    */
    timer_stop(TMR_GRAPHSUM_BW);
}

// ################################################################################################################

/**
 * Each predicted class probability is compared to the actual class desired and a loss is computed to penalize the proabability based on how far it is with respect to the actual expected value.
 * Also called logaritmic loss. 
*/
CrossEntropyLoss::CrossEntropyLoss(CudaVariable *cuda_logits, int *cuda_truth, float *loss, float *cuda_loss, int num_classes) :
        cuda_logits(cuda_logits), cuda_truth(cuda_truth), loss(loss), cuda_loss(cuda_loss), num_classes(num_classes) {
            // loss in CrossEntropyLoss loss is a pointer pointing to the loss value in GCN
            // cuda_loss in CrossEntropyLoss is a pointer pointing to the same GPU memory area pointed by cuda_loss in GCN
        }

__global__ void crossentropyloss_forward_parallel1(bool training, int *truth, float *logits_data, float *logits_grad, float *total_loss, int *count, int N, int n) {
    // N: logits->data.size(), n: num_classes
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N / n) {
        if (truth[i] >= 0) {
            atomicAdd(&(*count), 1);
            float *logit = &logits_data[i * n]; // each thread works on a different chunk of logits_data
            float max_logit = -1e30, sum_exp = 0.0f;
            for (int j = 0; j < n; j++)
                max_logit = fmaxf(max_logit, logit[j]);
            for (int j = 0; j < n; j++) {
                logit[j] -= max_logit;
                sum_exp += expf(logit[j]);
            }
            atomicAdd(&(*total_loss), logf(sum_exp) - logit[truth[i]]);
            if (training) {
                for (int j = 0; j < n; j++) {
                    float prob = expf(logit[j]) / sum_exp;
                    logits_grad[i * n + j] = prob; // each thread works on a different chunk of logits_grad
                }
                __syncthreads();
                atomicAdd(&logits_grad[i * n + truth[i]], -1.0f);
            }
        }
    }
}

__global__ void crossentropyloss_forward_parallel2(float *loss, float *total_loss, int *count) {
    *loss = *total_loss / *count;
}

__global__ void crossentropyloss_forward_parallel3(float *logits_grad, int *count, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
        logits_grad[i] /= *count;
}

void CrossEntropyLoss::forward(bool training) {
    timer_start(TMR_LOSS_FW);
    float total_loss = 0.0f;
    float *cuda_total_loss;
    check_call(cudaMalloc(&cuda_total_loss, sizeof(float)));
    check_call(cudaMemcpy(cuda_total_loss, &total_loss, sizeof(float), cudaMemcpyHostToDevice));
    int count = 0;
    int *cuda_count;
    check_call(cudaMalloc(&cuda_count, sizeof(int)));
    check_call(cudaMemcpy(cuda_count, &count, sizeof(int), cudaMemcpyHostToDevice));
    if (training) cuda_logits->zero_grad();
    // GPU blocks and threads settings
    dim3 blocksPerGrid1((cuda_logits->size / num_classes + MAX_THREADS_PER_BLOCK_1D - 1) / MAX_THREADS_PER_BLOCK_1D, 1, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_1D, 1, 1);
    crossentropyloss_forward_parallel1<<<blocksPerGrid1, threadsPerBlock>>>(training, cuda_truth, cuda_logits->data, cuda_logits->grad, cuda_total_loss, cuda_count, cuda_logits->size, num_classes);
    check_kernel_call();
    cudaDeviceSynchronize();
    crossentropyloss_forward_parallel2<<<1, 1>>>(cuda_loss, cuda_total_loss, cuda_count);
    check_kernel_call();
    cudaDeviceSynchronize();
    check_call(cudaMemcpy(&(*loss), cuda_loss, sizeof(float), cudaMemcpyDeviceToHost));
    if (training) {
        dim3 blocksPerGrid3((cuda_logits->size + MAX_THREADS_PER_BLOCK_1D - 1) / MAX_THREADS_PER_BLOCK_1D, 1, 1);
        crossentropyloss_forward_parallel3<<<blocksPerGrid3, threadsPerBlock>>>(cuda_logits->grad, cuda_count, cuda_logits->size);
        check_kernel_call();
        cudaDeviceSynchronize();
    }
    check_call(cudaFree(cuda_total_loss));
    check_call(cudaFree(cuda_count));
    /*
    float total_loss = 0;
    int count = 0;
    if (training) logits->zero_grad();
    for (int i = 0; i < logits->data.size() / num_classes; i++) {
        if (truth[i] < 0) continue;
        count++;
        float *logit = &logits->data[i * num_classes];
        float max_logit = -1e30, sum_exp = 0;
        for (int j = 0; j < num_classes; j++)
            max_logit = fmax(max_logit, logit[j]);
        for (int j = 0; j < num_classes; j++) {
            logit[j] -= max_logit;
            sum_exp += expf(logit[j]);
        }
        total_loss += logf(sum_exp) - logit[truth[i]];

        if (training) {
            for (int j = 0; j < num_classes; j++) {
                float prob = expf(logit[j]) / sum_exp;
                logits->grad[i * num_classes + j] = prob;
            }
            logits->grad[i * num_classes + truth[i]] -= 1.0;
        }
    }
    *loss = total_loss / count;
    if (training)
        for (float & i : logits->grad)
            i /= count;
    */
    timer_stop(TMR_LOSS_FW);
}

void CrossEntropyLoss::backward() {}

// ################################################################################################################

/**
 * Rectified Linear Unit activation function.
 * If input is negative it will output 0.
*/
ReLU::ReLU(CudaVariable *cuda_in) {
    this->cuda_in = cuda_in;
    check_call(cudaMalloc(&cuda_mask, cuda_in->size * sizeof(bool)));
}

ReLU::~ReLU() {
    std::cout << "Deallocating ReLU." << std::endl;
    check_call(cudaFree(cuda_mask));
}

__global__ void relu_forward_parallel(float *in, bool *mask, int N, bool training) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        bool keep = in[i] > 0;
        if (training) mask[i] = keep;
        if (!keep) in[i] = 0.0f;
    }
}

void ReLU::forward(bool training) {
    timer_start(TMR_RELU_FW);
    // GPU blocks and threads settings
    dim3 blocksPerGrid((cuda_in->size + MAX_THREADS_PER_BLOCK_1D - 1) / MAX_THREADS_PER_BLOCK_1D, 1, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_1D, 1, 1);
    // Launch kernel
    relu_forward_parallel<<<blocksPerGrid, threadsPerBlock>>>(cuda_in->data, cuda_mask, cuda_in->size, training);
    check_kernel_call();
    cudaDeviceSynchronize();
    /*
    for (int i = 0; i < in->data.size(); i++) {
        bool keep = in->data[i] > 0;
        if (training) mask[i] = keep;
        if (!keep) in->data[i] = 0;
    }
    */
    timer_stop(TMR_RELU_FW);
}

__global__ void relu_backward_parallel(float *grad, bool *mask, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        if (!mask[i]) grad[i] = 0.0f;
    }
}

void ReLU::backward() {
    timer_start(TMR_RELU_BW);
    // GPU blocks and threads settings
    dim3 blocksPerGrid((cuda_in->size + MAX_THREADS_PER_BLOCK_1D - 1) / MAX_THREADS_PER_BLOCK_1D, 1, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_1D, 1, 1);
    relu_backward_parallel<<<blocksPerGrid, threadsPerBlock>>>(cuda_in->grad, cuda_mask, cuda_in->size);
    check_kernel_call();
    cudaDeviceSynchronize();
    /*
    for (int i = 0; i < in->data.size(); i++)
        if (!mask[i]) in->grad[i] = 0;
    */
    timer_stop(TMR_RELU_BW);
}

// ################################################################################################################

/**
 * The dropout layer randomly sets input units to 0 with a frequency of P at each step during training time to prevent overfitting. 
 * Inputs that are not set to 0 are scaled up by 1/(1-P).
*/
Dropout::Dropout(CudaVariable *cuda_in, float p) {
    this->cuda_in = cuda_in;
    this->p = p;
    if (cuda_in->grad)
        check_call(cudaMalloc(&cuda_mask, cuda_in->size * sizeof(int)));
    else
        cuda_mask = nullptr;
    // GPU blocks and threads settings
    dim3 blocksPerGrid((cuda_in->size + MAX_THREADS_PER_BLOCK_1D - 1) / MAX_THREADS_PER_BLOCK_1D, 1, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_1D, 1, 1);
    // Initialize CUDA random
    check_call(cudaMalloc(&cuda_rand_state, cuda_in->size * sizeof(curandState)));
    rand_setup_kernel<<<blocksPerGrid, threadsPerBlock>>>(cuda_rand_state, cuda_in->size);
    check_kernel_call();
    cudaDeviceSynchronize();
}

Dropout::~Dropout() {
    std::cout << "Deallocating Dropout." << std::endl;
    if (cuda_mask) check_call(cudaFree(cuda_mask));
}

__global__ void dropout_forward_parallel(float *in, int* mask, int N, const int threshold, float scale, curandState *rand_state, unsigned rand_max) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        float my_randf = rand_max * curand_uniform(&rand_state[i]);
        int my_rand = (int) truncf(my_randf);
        bool keep = my_rand >= threshold;
        in[i] *= keep ? scale : 0;
        if (mask) mask[i] = keep;
    }
}

void Dropout::forward(bool training) {
    if (!training) return;
    timer_start(TMR_DROPOUT_FW);
    const int threshold = int(p * MY_RAND_MAX);
    float scale = 1 / (1 - p);
    // GPU blocks and threads settings
    dim3 blocksPerGrid((cuda_in->size + MAX_THREADS_PER_BLOCK_1D - 1) / MAX_THREADS_PER_BLOCK_1D, 1, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_1D, 1, 1);
    // Launch kernel
    dropout_forward_parallel<<<blocksPerGrid, threadsPerBlock>>>(cuda_in->data, cuda_mask, cuda_in->size, threshold, scale, cuda_rand_state, MY_RAND_MAX);
    check_kernel_call();
    cudaDeviceSynchronize();
    /*
    for (int i = 0; i < in->data.size(); i++) {
        bool keep = (int) RAND() >= threshold;
        in->data[i] *= keep ? scale : 0;
        if (mask) mask[i] = keep;
    }
    */
    timer_stop(TMR_DROPOUT_FW);
}

__global__ void dropout_backward_parallel(float *grad, int *mask, int N, float scale) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        grad[i] *= mask[i] ? scale : 0.0f;
    }
}

void Dropout::backward() {
    if (!cuda_mask) return;
    timer_start(TMR_DROPOUT_BW);
    float scale = 1 / (1 - p);
    // GPU blocks and threads settings
    dim3 blocksPerGrid((cuda_in->size + MAX_THREADS_PER_BLOCK_1D - 1) / MAX_THREADS_PER_BLOCK_1D, 1, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_1D, 1, 1);
    // Launch kernel
    dropout_backward_parallel<<<blocksPerGrid, threadsPerBlock>>>(cuda_in->grad, cuda_mask, cuda_in->size, scale);
    check_kernel_call();
    cudaDeviceSynchronize();
    /*
    for (int i = 0; i < in->data.size(); i++)
        in->grad[i] *= mask[i] ? scale : 0;
    */
    timer_stop(TMR_DROPOUT_BW);
}

// ################################################################################################################