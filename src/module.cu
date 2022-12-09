#include "../include/module.h"
#include "../include/rand.h"
#include "../include/timer.h"
#include "../include/cuda_check.h"
#include "../include/cuda_rand.h"
#include <cmath>
#include <iostream>

// ################################################################################################################
/**
 * Dense matrix multiplication layer. 
*/
Matmul::Matmul(Variable *a, Variable *b, Variable *c, CudaVariable *cuda_a, CudaVariable *cuda_b, CudaVariable *cuda_c, int m, int n, int p) : 
        a(a), b(b), c(c), cuda_a(cuda_a), cuda_b(cuda_b), cuda_c(cuda_c), m(m), n(n), p(p) {}

__global__ void matmul_forward_parallel(float *A, float *B, float *C, int m, int n, int p) {
    // Multiplication of matrices A and B; the result is stored in the matrix C
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < m && j < p) {
        size_t index = i * p + j;
        C[index] = 0.0f;
        for (size_t k = 0; k < n; ++k)
            C[index] += A[i * n + k] * B[k * p + j];
    }
    /*
    // Tile-based multiplication?
    */
}

void Matmul::forward(bool training) {
    timer_start(TMR_MATMUL_FW);
    c->zero();
    // GPU blocks and threads settings
    // WE ASSUME THAT ALL BLOCKS FIT INTO SHARED MEMORY (4MB)
    const unsigned int tile_size = 32;
    dim3 blocksPerGrid((m + tile_size - 1) / tile_size, (p + tile_size - 1) / tile_size, 1);
    dim3 threadsPerBlock(tile_size, tile_size, 1); // 2D squared blocks of size (tile_size, tile_size)
    // Launch kernel
    matmul_forward_parallel<<<blocksPerGrid, threadsPerBlock>>>(cuda_a->data, cuda_b->data, cuda_c->data, m, n, p);
    check_kernel_call();
    cudaDeviceSynchronize();
    /*
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < p; k++)
                c->data[i * p + k] += a->data[i * n + j] * b->data[j * p + k];
        }
    */
    timer_stop(TMR_MATMUL_FW);
}

__global__ void matmul_backward_parallel(float *A, float *B, float *C, int m, int n, int p) {
    // Multiplication of matrices A and B; the result is stored in the matrix C
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i == 0 && j == 0)

    if (i < m && j < n) {
        float tmp = 0;
        size_t index = i * p + j;
        C[index] = 0.0f;
        for (size_t k = 0; k < n; ++k)
            C[index] += A[i * n + k] * B[k * p + j];
    }
}

void Matmul::backward() {
    timer_start(TMR_MATMUL_BW);
    a->zero_grad();
    b->zero_grad();
    // GPU blocks and threads settings
    const unsigned int tile_size = 32;
    dim3 blocksPerGrid((m + tile_size - 1) / tile_size, (n + tile_size - 1) / tile_size, 1);
    dim3 threadsPerBlock(tile_size, tile_size, 1); // 2D squared blocks of size (tile_size, tile_size)
    // Launch kernel
    matmul_backward_parallel<<<blocksPerGrid, threadsPerBlock>>>(cuda_a->grad, cuda_b->grad, cuda_c->grad, m, n, p);
    check_kernel_call();
    cudaDeviceSynchronize();
    /*
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
SparseMatmul::SparseMatmul(Variable *a, Variable *b, Variable *c, CudaVariable *cuda_a, CudaVariable *cuda_b, CudaVariable *cuda_c, SparseIndex *sp, int m, int n, int p) :
        a(a), b(b), c(c), cuda_a(cuda_a), cuda_b(cuda_b), cuda_c(cuda_c), sp(sp), m(m), n(n), p(p) {
            int *temp_indptr = sp->indptr.data();
            int *temp_indices = sp->indices.data();
            check_call(cudaMalloc(&cuda_sp->indptr, sp->indptr.size() * sizeof(int)));
            check_call(cudaMalloc(&cuda_sp->indices, sp->indices.size() * sizeof(int)));
            check_call(cudaMemcpy(cuda_sp->indptr, temp_indptr, sp->indptr.size() * sizeof(int), cudaMemcpyHostToDevice));
            check_call(cudaMemcpy(cuda_sp->indices, temp_indices, sp->indices.size() * sizeof(int), cudaMemcpyHostToDevice));
        }


// IMPLEMENT DESTRUCTOR TO DEALLOCATE CUDA MEMORY FOR SPARSE INDEX

__global__ void sparsematmul_forward_parallel(float *A, float *B, float *C, int *indptr, int *indices, int p, int N) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        for (int jj = indptr[i]; jj < indptr[i + 1]; jj++) {
            int j = indices[jj];
            for (int k = 0; k < p; k++)
                atomicAdd(&C[i * p + k], A[jj] * B[j * p + k]);
            // SYNCHRONIZATION NEEDED NOW?
        }
    }        
}

void SparseMatmul::forward(bool training) {
    timer_start(TMR_SPMATMUL_FW);
    c->zero();
    // GPU blocks and threads settings
    const unsigned int max_num_threads = 1024;
    dim3 blocksPerGrid((sp->indptr.size() - 1 + max_num_threads - 1) / max_num_threads, 1, 1);
    dim3 threadsPerBlock(max_num_threads, 1, 1);
    // Launch kernel
    sparsematmul_forward_parallel<<<blocksPerGrid, threadsPerBlock>>>(cuda_a->data, cuda_b->data, cuda_c->data, cuda_sp->indptr, cuda_sp->indices, p, sp->indptr.size() - 1);
    check_kernel_call();
    cudaDeviceSynchronize();
    /*
    for (int i = 0; i < sp->indptr.size() - 1; i++)
        for (int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++) {
            int j = sp->indices[jj];
            for (int k = 0; k < p; k++)
                c->data[i * p + k] += a->data[jj] * b->data[j * p + k];
        }
    */
    timer_stop(TMR_SPMATMUL_FW);
}

void SparseMatmul::backward() {
    timer_start(TMR_SPMATMUL_BW);
    b->zero_grad();
    int row = 0;
    for (int i = 0; i < sp->indptr.size() - 1; i++)
        for (int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++) {
            int j = sp->indices[jj];
            for (int k = 0; k < p; k++)
                    b->grad[j * p + k] += c->grad[i * p + k] * a->data[jj];
        }
    timer_stop(TMR_SPMATMUL_BW);
}

// ################################################################################################################

/**
 * A specialized sparse matrix multiplication for graphs.
*/
GraphSum::GraphSum(Variable *in, Variable *out, CudaVariable *cuda_in, CudaVariable *cuda_out, SparseIndex *graph, int dim) :
        in(in), out(out), cuda_in(cuda_in), cuda_out(cuda_out), graph(graph), dim(dim) {
            int *temp_indptr = graph->indptr.data();
            int *temp_indices = graph->indices.data();
            check_call(cudaMalloc(&cuda_graph->indptr, graph->indptr.size() * sizeof(int)));
            check_call(cudaMalloc(&cuda_graph->indices, graph->indices.size() * sizeof(int)));
            check_call(cudaMemcpy(cuda_graph->indptr, temp_indptr, graph->indptr.size() * sizeof(int), cudaMemcpyHostToDevice));
            check_call(cudaMemcpy(cuda_graph->indices, temp_indices, graph->indices.size() * sizeof(int), cudaMemcpyHostToDevice));
        }

// IMPLEMENT DESTRUCTOR TO DEALLOCATE CUDA MEMORY FOR SPARSE INDEX

__global__ void graphsum_forward_parallel(float *in, float *out, int *indptr, int *indices, int dim, int N) {
    size_t src = threadIdx.x + blockIdx.x * blockDim.x;
    if (src < N) {
        for (int i = indptr[src]; i < indptr[src + 1]; i++) {
            int dst = indices[i];
            float coef = 1.0 / sqrtf(
                    (indptr[src + 1] - indptr[src]) * (indptr[dst + 1] - indptr[dst])
            );
            for (int j = 0; j < dim; j++)
                // This only works for undirected graphs. Should be out[dst] += coef * in[src]
                atomicAdd(&out[src * dim + j], coef * in[dst * dim + j]);
            // SYNCHRONIZATION NEEDED NOW?
        }
    }        
}

void GraphSum::forward(bool training) {
    timer_start(TMR_GRAPHSUM_FW);
    out->zero();
    // GPU blocks and threads settings
    const unsigned int max_num_threads = 1024;
    dim3 blocksPerGrid((graph->indptr.size() - 1 + max_num_threads - 1) / max_num_threads, 1, 1);
    dim3 threadsPerBlock(max_num_threads, 1, 1);
    // Launch kernel
    graphsum_forward_parallel<<<blocksPerGrid, threadsPerBlock>>>(cuda_in->data, cuda_out->data, cuda_graph->indptr, cuda_graph->indices, dim, graph->indptr.size() - 1);
    check_kernel_call();
    cudaDeviceSynchronize();
    /*
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
    timer_stop(TMR_GRAPHSUM_BW);
}

// ################################################################################################################

/**
 * Each predicted class probability is compared to the actual class desired and a loss is computed to penalize the proabability based on how far it is with respect to the actual expected value.
 * Also called logaritmic loss. 
*/
CrossEntropyLoss::CrossEntropyLoss(Variable *logits, CudaVariable *cuda_logits, int *truth, int *cuda_truth, float *loss, float *cuda_loss, int num_classes) :
        logits(logits), cuda_logits(cuda_logits), truth(truth), cuda_truth(cuda_truth), loss(loss), cuda_loss(cuda_loss), num_classes(num_classes) {
            /*
            ???   
            */       
        }

// IMPLEMENT DESTRUCTOR TO DEALLOCATE CUDA MEMORY

__global__ void crossentropyloss_forward_parallel1(int *truth, float *logits_data, float *logits_grad, float *loss, int *count, bool training, int N, int n) {
    // N: logits->data.size(), n: num_classes
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N / n) {
        if (truth[i] >= 0) {
            atomicAdd(&(*count), 1);
            float *logit = &logits_data[i * n];
            float max_logit = -1e30, sum_exp = 0;
            for (int j = 0; j < n; j++)
                max_logit = fmaxf(max_logit, logit[j]);
            for (int j = 0; j < n; j++) {
                logit[j] -= max_logit;
                sum_exp += expf(logit[j]);
            }
            atomicAdd(&(*loss), logf(sum_exp) - logit[truth[i]]);
            if (training) {
                for (int j = 0; j < n; j++) {
                    float prob = expf(logit[j]) / sum_exp;
                    logits_grad[i * n + j] = prob;
                }
                __syncthreads();
                atomicAdd(&logits_grad[i * n + truth[i]], -1.0);
            }
        }
    }
    __syncthreads();
    if (i == 0)
        *loss /= *count;
}

__global__ void crossentropyloss_forward_parallel2(float *logits_grad, int *count, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
        logits_grad[i] /= *count;
}

void CrossEntropyLoss::forward(bool training) {
    timer_start(TMR_LOSS_FW);
    float total_loss = 0;
    int count = 0;
    int *cuda_count;
    check_call(cudaMalloc(&cuda_count, sizeof(int)));
    if (training) logits->zero_grad();
    /*
    // GPU blocks and threads settings
    const unsigned int max_num_threads = 1024;
    dim3 blocksPerGrid1((logits->data.size() / num_classes + max_num_threads - 1) / max_num_threads, 1, 1);
    dim3 threadsPerBlock(max_num_threads, 1, 1);
    crossentropyloss_forward_parallel1<<<blocksPerGrid1, threadsPerBlock>>>(cuda_truth, cuda_logits_data, *cuda_logits_grad, *cuda_loss, cuda_count, training, logits->data.size(), num_classes);
    check_kernel_call();
    cudaDeviceSynchronize();
    check_call(cudaMemcpy(&loss, cuda_loss, sizeof(float), cudaMemcpyDeviceToHost));
    if (training) {
        dim3 blocksPerGrid2((logits->grad.size() + max_num_threads - 1) / max_num_threads, 1, 1);
        crossentropyloss_forward_parallel2<<<blocksPerGrid2, threadsPerBlock>>>(*cuda_logits_grad, cuda_count, logits->grad.size());
    }
    check_call(cudaFree(cuda_count));
    */
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
    timer_stop(TMR_LOSS_FW);
}

void CrossEntropyLoss::backward() {
}

// ################################################################################################################

/**
 * Rectified Linear Unit activation function.
 * If input is negative it will output 0.
*/
ReLU::ReLU(Variable *in, CudaVariable *cuda_in) {
    this->in = in;
    this->cuda_in = cuda_in;
    mask = new bool[in->data.size()];
    check_call(cudaMalloc(&cuda_mask, in->data.size() * sizeof(bool)));
}

ReLU::~ReLU() {
    delete[] mask;
    check_call(cudaFree(cuda_mask));
}

__global__ void relu_forward_parallel(float *in, bool *mask, int N, bool training) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        bool keep = in[i] > 0;
        if (training) mask[i] = keep;
        if (!keep) in[i] = 0;
    }
}

void ReLU::forward(bool training) {
    timer_start(TMR_RELU_FW);
    // GPU blocks and threads settings
    const unsigned int max_num_threads = 1024;
    dim3 blocksPerGrid((in->data.size() + max_num_threads - 1) / max_num_threads, 1, 1);
    dim3 threadsPerBlock(max_num_threads, 1, 1);
    // Launch kernel
    relu_forward_parallel<<<blocksPerGrid, threadsPerBlock>>>(cuda_in->data, cuda_mask, in->data.size(), training);
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

void ReLU::backward() {
    timer_start(TMR_RELU_BW);
    for (int i = 0; i < in->data.size(); i++)
        if (!mask[i]) in->grad[i] = 0;
    timer_stop(TMR_RELU_BW);
}

// ################################################################################################################

/**
 * The dropout layer randomly sets input units to 0 with a frequency of P at each step during training time to prevent overfitting. 
 * Inputs that are not set to 0 are scaled up by 1/(1-P).
*/
Dropout::Dropout(Variable *in, CudaVariable *cuda_in, float p) {
    this->in = in;
    this->cuda_in = cuda_in;
    this->p = p;
    if (!in->grad.empty()) {
        mask = new int[in->data.size()];
        check_call(cudaMalloc(&cuda_mask, in->data.size() * sizeof(int)));
    }
    else {
        mask = nullptr;
    }
    // NULLPTR FOR CUDA POINTERS?
    /*
    // GPU blocks and threads settings
    const unsigned int max_num_threads = 1024;
    dim3 blocksPerGrid((in->data.size() + max_num_threads - 1) / max_num_threads, 1, 1);
    dim3 threadsPerBlock(max_num_threads, 1, 1);
    // Initialize CUDA random
    check_call(cudaMalloc(&cuda_rand_state, in->data.size() * sizeof(curandState)));
    setup_kernel<<<blocksPerGrid, threadsPerBlock>>>(cuda_rand_state);
    check_kernel_call();
    cudaDeviceSynchronize();
    */
}

Dropout::~Dropout() {
    if (mask) delete[] mask;
    check_call(cudaFree(cuda_mask));
}

__global__ void dropout_forward_parallel(float *in, int* mask, int N, const int threshold, float scale, curandState *rand_state, unsigned rand_max) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        float my_randf = rand_max * curand_uniform(&rand_state[i]);
        int my_rand = (int) truncf(my_randf);
        bool keep = my_rand >= threshold;
        in[i] *= keep ? scale : 0;
        if (mask) mask[i] = keep; // CHECK IF IT IS NOT A NULLPTR?
    }
}

void Dropout::forward(bool training) {
    if (!training) return;
    timer_start(TMR_DROPOUT_FW);
    const int threshold = int(p * MY_RAND_MAX);
    float scale = 1 / (1 - p);
    /*
    // GPU blocks and threads settings
    const unsigned int max_num_threads = 1024; // must be the same value as the value used in the constructor
    dim3 blocksPerGrid((in->data.size() + max_num_threads - 1) / max_num_threads, 1, 1);
    dim3 threadsPerBlock(max_num_threads, 1, 1);
    // Launch kernel
    dropout_forward_parallel<<<blocksPerGrid, threadsPerBlock>>>(*cuda_in, cuda_mask, in->data.size(), threshold, scale, cuda_rand_state, MY_CUDA_RAND_MAX);
    check_kernel_call();
    cudaDeviceSynchronize();
    */
    for (int i = 0; i < in->data.size(); i++) {
        bool keep = (int)RAND() >= threshold;
        in->data[i] *= keep ? scale : 0;
        if (mask) mask[i] = keep;
    }
    timer_stop(TMR_DROPOUT_FW);
}

void Dropout::backward() {
    if (!mask) return;
    timer_start(TMR_DROPOUT_BW);
    float scale = 1 / (1 - p);
    for (int i = 0; i < in->data.size(); i++)
        in->grad[i] *= mask[i] ? scale : 0;
    timer_stop(TMR_DROPOUT_BW);
}

// ################################################################################################################