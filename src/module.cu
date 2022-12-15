#include "../include/module.h"
#include "../include/rand.h"
#include "../include/timer.h"
#include "../include/cuda_check.h"
#include "../include/cuda_rand.h"
#include <cmath>
#include <iostream>

#define MAX_THREADS_PER_BLOCK_1D 1024
#define MAX_THREADS_PER_BLOCK_2D 32

// ################################################################################################################
/**
 * Dense matrix multiplication layer. 
*/
Matmul::Matmul(Variable *a, Variable *b, Variable *c, CudaVariable *cuda_a, CudaVariable *cuda_b, CudaVariable *cuda_c, int m, int n, int p) : 
        a(a), b(b), c(c), cuda_a(cuda_a), cuda_b(cuda_b), cuda_c(cuda_c), m(m), n(n), p(p) {}

__global__ void matmul_forward_parallel(float *A, float *B, float *C, int m, int n, int p) {
    // Multiplication of matrices A and B; the result is stored in the matrix C
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < m && k < p) {
        int index = i * p + k;
        C[index] = 0.0f;
        for (int j = 0; j < n; ++j)
            C[index] += A[i * n + j] * B[j * p + k];
    }
    // TILE-BASED MULTIPLICATION WITH SHARED MEMORY?
}

void Matmul::forward(bool training) {
    timer_start(TMR_MATMUL_FW);
    c->zero();
    // GPU blocks and threads settings
    // WE ASSUME THAT ALL BLOCKS FIT INTO SHARED MEMORY (4MB)
    dim3 blocksPerGrid((m + MAX_THREADS_PER_BLOCK_2D - 1) / MAX_THREADS_PER_BLOCK_2D, (p + MAX_THREADS_PER_BLOCK_2D - 1) / MAX_THREADS_PER_BLOCK_2D, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_2D, MAX_THREADS_PER_BLOCK_2D, 1); // 2D squared blocks
    // Launch kernel
    matmul_forward_parallel<<<blocksPerGrid, threadsPerBlock>>>(cuda_a->data, cuda_b->data, cuda_c->data, m, n, p);
    check_kernel_call();
    cudaDeviceSynchronize();

    float *temp = (float *) malloc(c->data.size() * sizeof(float));
    check_call(cudaMemcpy(temp, cuda_c->data, c->data.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "1" << std::endl;
    for (int i = 0; i < c->data.size(); ++i)
        c->data[i] = temp[i];
    free(temp);

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
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < m && j < n) {
        float tmp = 0;
            for (int k = 0; k < p; k++) {
                tmp += C[i * p + k] * B[j * p + k];
                atomicAdd(&B[j * p + k], C[i * p + k] * A[i * n + j]); // TRY TO AVOID ATOMIC ADD
            }
		    A[i * n + j] = tmp;
    }
}

void Matmul::backward() {
    timer_start(TMR_MATMUL_BW);
    /*
    cuda_a->zero_grad();
    cuda_b->zero_grad();
    // GPU blocks and threads settings
    dim3 blocksPerGrid((m + MAX_THREADS_PER_BLOCK_2D - 1) / MAX_THREADS_PER_BLOCK_2D, (n + MAX_THREADS_PER_BLOCK_2D - 1) / MAX_THREADS_PER_BLOCK_2D, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_2D, MAX_THREADS_PER_BLOCK_2D, 1);
    // Launch kernel
    matmul_backward_parallel<<<blocksPerGrid, threadsPerBlock>>>(cuda_a->grad, cuda_b->grad, cuda_c->grad, m, n, p);
    check_kernel_call();
    cudaDeviceSynchronize();

    float *temp = (float *) malloc(a->grad.size() * sizeof(float));
    check_call(cudaMemcpy(temp, cuda_a->grad, a->grad.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "2" << std::endl;
    for (int i = 0; i < a->grad.size(); ++i)
        a->grad[i] = temp[i];
    free(temp);

    temp = (float *) malloc(b->grad.size() * sizeof(float));
    check_call(cudaMemcpy(temp, cuda_b->grad, b->grad.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "3" << std::endl;
    for (int i = 0; i < b->grad.size(); ++i)
        b->grad[i] = temp[i];
    free(temp);
    */

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
    
    float *temp = (float *) malloc(a->grad.size() * sizeof(float));
    for (int i = 0; i < a->grad.size(); ++i)
        temp[i] = a->grad[i];
    check_call(cudaMemcpy(cuda_a->grad, temp, a->grad.size() * sizeof(float), cudaMemcpyHostToDevice));
    std::cout << "2" << std::endl;
    free(temp);

    temp = (float *) malloc(b->grad.size() * sizeof(float));
    for (int i = 0; i < b->grad.size(); ++i)
        temp[i] = b->grad[i];
    check_call(cudaMemcpy(temp, cuda_b->grad, b->grad.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "3" << std::endl;
    free(temp);

    timer_stop(TMR_MATMUL_BW);
}


// ################################################################################################################

/**
 * A sparse matrix multiplication layer.
*/
SparseMatmul::SparseMatmul(Variable *a, Variable *b, Variable *c, CudaVariable *cuda_a, CudaVariable *cuda_b, CudaVariable *cuda_c, SparseIndex *sp, int m, int n, int p) :
        a(a), b(b), c(c), cuda_a(cuda_a), cuda_b(cuda_b), cuda_c(cuda_c), sp(sp), m(m), n(n), p(p) {
            CudaSparseIndex *cuda_sp_temp = new CudaSparseIndex(sp->indices.data(), sp->indptr.data(), sp->indices.size(), sp->indptr.size());
            cuda_sp = cuda_sp_temp;
        }

__global__ void sparsematmul_forward_parallel(float *A, float *B, float *C, int *indptr, int *indices, int N, int p) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && k < p) {
        int index = i * p + k;
        C[index] = 0.0f;
        for (int jj = indptr[i]; jj < indptr[i + 1]; jj++) {
            int j = indices[jj];
            C[index] += A[jj] * B[j * p + k];
        }
    }        
}

void SparseMatmul::forward(bool training) {
    timer_start(TMR_SPMATMUL_FW);
    c->zero();
    // GPU blocks and threads settings
    dim3 blocksPerGrid((m + MAX_THREADS_PER_BLOCK_2D - 1) / MAX_THREADS_PER_BLOCK_2D, (p + MAX_THREADS_PER_BLOCK_2D - 1) / MAX_THREADS_PER_BLOCK_2D, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_2D, MAX_THREADS_PER_BLOCK_2D, 1);
    // Launch kernel
    sparsematmul_forward_parallel<<<blocksPerGrid, threadsPerBlock>>>(cuda_a->data, cuda_b->data, cuda_c->data, cuda_sp->indptr, cuda_sp->indices, sp->indptr.size() - 1, p);
    check_kernel_call();
    cudaDeviceSynchronize();

    float *temp = (float *) malloc(c->data.size() * sizeof(float));
    check_call(cudaMemcpy(temp, cuda_c->data, c->data.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "4" << std::endl;
    for (int i = 0; i < c->data.size(); ++i)
        c->data[i] = temp[i];
    free(temp);

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

__global__ void sparsematmul_backward_parallel(float *A, float *B, float *C, int *indptr, int *indices, int N, int p) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && k < p) {
        for (int jj = indptr[i]; jj < indptr[i + 1]; jj++) {
            int j = indices[jj];
            atomicAdd(&B[j * p + k], C[i * p + k] * A[jj]); // TRY TO AVOID ATOMIC ADD
        }
    }
}

void SparseMatmul::backward() {
    timer_start(TMR_SPMATMUL_BW);
    /*
    cuda_b->zero_grad();
    // GPU blocks and threads settings
    dim3 blocksPerGrid((m + MAX_THREADS_PER_BLOCK_2D - 1) / MAX_THREADS_PER_BLOCK_2D, (p + MAX_THREADS_PER_BLOCK_2D - 1) / MAX_THREADS_PER_BLOCK_2D, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_2D, MAX_THREADS_PER_BLOCK_2D, 1);
    // Launch kernel
    sparsematmul_backward_parallel<<<blocksPerGrid, threadsPerBlock>>>(cuda_a->data, cuda_b->grad, cuda_c->grad, cuda_sp->indptr, cuda_sp->indices, sp->indptr.size() - 1, p);
    check_kernel_call();
    cudaDeviceSynchronize();

    float *temp = (float *) malloc(b->grad.size() * sizeof(float));
    check_call(cudaMemcpy(temp, cuda_b->grad, b->grad.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "5" << std::endl;
    for (int i = 0; i < b->grad.size(); ++i)
        b->grad[i] = temp[i];
    free(temp);
    */

    b->zero_grad();
    // int row = 0;
    for (int i = 0; i < sp->indptr.size() - 1; i++)
        for (int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++) {
            int j = sp->indices[jj];
            for (int k = 0; k < p; k++)
                    b->grad[j * p + k] += c->grad[i * p + k] * a->data[jj];
        }
    
    float *temp = (float *) malloc(b->data.size() * sizeof(float));
    for (int i = 0; i < b->data.size(); ++i)
        temp[i] = b->data[i];
    check_call(cudaMemcpy(cuda_b->data, temp, b->data.size() * sizeof(float), cudaMemcpyHostToDevice));
    std::cout << "5" << std::endl;
    free(temp);

    timer_stop(TMR_SPMATMUL_BW);
}

// ################################################################################################################

/**
 * A specialized sparse matrix multiplication for graphs.
*/
GraphSum::GraphSum(Variable *in, Variable *out, CudaVariable *cuda_in, CudaVariable *cuda_out, SparseIndex *graph, int dim) :
        in(in), out(out), cuda_in(cuda_in), cuda_out(cuda_out), graph(graph), dim(dim) {
            CudaSparseIndex *cuda_graph_temp = new CudaSparseIndex(graph->indices.data(), graph->indptr.data(), graph->indices.size(), graph->indptr.size());
            cuda_graph = cuda_graph_temp;
        }

__global__ void graphsum_forward_parallel(float *in, float *out, int *indptr, int *indices, int N, int dim) {
    int src = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (src < N && j < dim) {
        int index = src * dim + j;
        out[index] = 0.0f;
        for (int i = indptr[src]; i < indptr[src + 1]; i++) {
            int dst = indices[i];
            float coef = 1.0 / sqrtf(
                    (indptr[src + 1] - indptr[src]) * (indptr[dst + 1] - indptr[dst])
            );
            // This only works for undirected graphs. Should be out[dst] += coef * in[src]
            out[index] += coef * in[dst * dim + j];
        }
    }        
}

void GraphSum::forward(bool training) {
    timer_start(TMR_GRAPHSUM_FW);
    out->zero();
    // GPU blocks and threads settings
    dim3 blocksPerGrid((graph->indptr.size() - 1 + MAX_THREADS_PER_BLOCK_2D - 1) / MAX_THREADS_PER_BLOCK_2D, (dim + MAX_THREADS_PER_BLOCK_2D - 1) / MAX_THREADS_PER_BLOCK_2D, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_2D, MAX_THREADS_PER_BLOCK_2D, 1);
    // Launch kernel
    graphsum_forward_parallel<<<blocksPerGrid, threadsPerBlock>>>(cuda_in->data, cuda_out->data, cuda_graph->indptr, cuda_graph->indices, graph->indptr.size() - 1, dim);
    check_kernel_call();
    cudaDeviceSynchronize();

    float *temp = (float *) malloc(out->data.size() * sizeof(float));
    check_call(cudaMemcpy(temp, cuda_out->data, out->data.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "6" << std::endl;
    for (int i = 0; i < out->data.size(); ++i)
        out->data[i] = temp[i];
    free(temp);

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
    /*
    cuda_in->zero_grad();
    // GPU blocks and threads settings
    dim3 blocksPerGrid((graph->indptr.size() - 1 + MAX_THREADS_PER_BLOCK_2D - 1) / MAX_THREADS_PER_BLOCK_2D, (dim + MAX_THREADS_PER_BLOCK_2D - 1) / MAX_THREADS_PER_BLOCK_2D, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_2D, MAX_THREADS_PER_BLOCK_2D, 1);
    // Launch kernel
    // SAME EXACT CODE STRUCTURE AS GRAPHSUM FORWARD, BUT WITH GRADIENTS AND WITH IN AND OUT SWAPPED!
    graphsum_forward_parallel<<<blocksPerGrid, threadsPerBlock>>>(cuda_out->grad, cuda_in->grad, cuda_graph->indptr, cuda_graph->indices, graph->indptr.size() - 1, dim);
    check_kernel_call();
    cudaDeviceSynchronize();

    float *temp = (float *) malloc(in->grad.size() * sizeof(float));
    check_call(cudaMemcpy(temp, cuda_in->grad, in->grad.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "7" << std::endl;
    for (int i = 0; i < in->grad.size(); ++i)
        in->grad[i] = temp[i];
    free(temp);
    */

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
    
    float *temp = (float *) malloc(in->grad.size() * sizeof(float));
    for (int i = 0; i < in->grad.size(); ++i)
        temp[i] = in->grad[i];
    check_call(cudaMemcpy(cuda_in->grad, temp, in->grad.size() * sizeof(float), cudaMemcpyHostToDevice));
    std::cout << "7" << std::endl;
    free(temp);

    timer_stop(TMR_GRAPHSUM_BW);
}

// ################################################################################################################

/**
 * Each predicted class probability is compared to the actual class desired and a loss is computed to penalize the proabability based on how far it is with respect to the actual expected value.
 * Also called logaritmic loss. 
*/
CrossEntropyLoss::CrossEntropyLoss(Variable *logits, CudaVariable *cuda_logits, int *truth, int *cuda_truth, float *loss, float *cuda_loss, int num_classes) :
        logits(logits), cuda_logits(cuda_logits), truth(truth), cuda_truth(cuda_truth), loss(loss), cuda_loss(cuda_loss), num_classes(num_classes) {
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
    dim3 blocksPerGrid1((logits->data.size() / num_classes + MAX_THREADS_PER_BLOCK_1D - 1) / MAX_THREADS_PER_BLOCK_1D, 1, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_1D, 1, 1);
    crossentropyloss_forward_parallel1<<<blocksPerGrid1, threadsPerBlock>>>(training, cuda_truth, cuda_logits->data, cuda_logits->grad, cuda_total_loss, cuda_count, logits->data.size(), num_classes);
    check_kernel_call();
    cudaDeviceSynchronize();
    crossentropyloss_forward_parallel2<<<1, 1>>>(cuda_loss, cuda_total_loss, cuda_count);
    check_kernel_call();
    cudaDeviceSynchronize();
    check_call(cudaMemcpy(&(*loss), cuda_loss, sizeof(float), cudaMemcpyDeviceToHost));
    if (training) {
        dim3 blocksPerGrid3((logits->grad.size() + MAX_THREADS_PER_BLOCK_1D - 1) / MAX_THREADS_PER_BLOCK_1D, 1, 1);
        crossentropyloss_forward_parallel3<<<blocksPerGrid3, threadsPerBlock>>>(cuda_logits->grad, cuda_count, logits->grad.size());
        check_kernel_call();
        cudaDeviceSynchronize();
    }
    check_call(cudaFree(cuda_total_loss));
    check_call(cudaFree(cuda_count));

    float *temp = (float *) malloc(logits->data.size() * sizeof(float));
    check_call(cudaMemcpy(temp, cuda_logits->data, logits->data.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "8" << std::endl;
    for (int i = 0; i < logits->data.size(); ++i)
        logits->data[i] = temp[i];
    free(temp);

    temp = (float *) malloc(logits->grad.size() * sizeof(float));
    check_call(cudaMemcpy(temp, cuda_logits->grad, logits->grad.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "9" << std::endl;
    for (int i = 0; i < logits->grad.size(); ++i)
        logits->grad[i] = temp[i];
    free(temp);

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
ReLU::ReLU(Variable *in, CudaVariable *cuda_in) {
    this->in = in;
    this->cuda_in = cuda_in;
    mask = new bool[in->data.size()];
    check_call(cudaMalloc(&cuda_mask, in->data.size() * sizeof(bool)));
}

ReLU::~ReLU() {
    std::cout << "Deallocating ReLU." << std::endl;
    delete[] mask;
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
    dim3 blocksPerGrid((in->data.size() + MAX_THREADS_PER_BLOCK_1D - 1) / MAX_THREADS_PER_BLOCK_1D, 1, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_1D, 1, 1);
    // Launch kernel
    relu_forward_parallel<<<blocksPerGrid, threadsPerBlock>>>(cuda_in->data, cuda_mask, in->data.size(), training);
    check_kernel_call();
    cudaDeviceSynchronize();

    check_call(cudaMemcpy(mask, cuda_mask, in->data.size() * sizeof(bool), cudaMemcpyDeviceToHost));
    std::cout << "10" << std::endl;
    
    float *temp = (float *) malloc(in->data.size() * sizeof(float));
    check_call(cudaMemcpy(temp, cuda_in->data, in->data.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "11" << std::endl;
    for (int i = 0; i < in->data.size(); ++i)
        in->data[i] = temp[i];
    free(temp);

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
    /*
    // GPU blocks and threads settings
    dim3 blocksPerGrid((in->data.size() + MAX_THREADS_PER_BLOCK_1D - 1) / MAX_THREADS_PER_BLOCK_1D, 1, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_1D, 1, 1);
    relu_backward_parallel<<<blocksPerGrid, threadsPerBlock>>>(cuda_in->grad, cuda_mask, in->data.size());
    check_kernel_call();
    cudaDeviceSynchronize();

    float *temp = (float *) malloc(in->grad.size() * sizeof(float));
    check_call(cudaMemcpy(temp, cuda_in->grad, in->grad.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "12" << std::endl;
    for (int i = 0; i < in->grad.size(); ++i)
        in->grad[i] = temp[i];
    free(temp);
    */

    for (int i = 0; i < in->data.size(); i++)
        if (!mask[i]) in->grad[i] = 0;
    
    float *temp = (float *) malloc(in->grad.size() * sizeof(float));
    for (int i = 0; i < in->grad.size(); ++i)
        temp[i] = in->grad[i];
    check_call(cudaMemcpy(cuda_in->grad, temp, in->grad.size() * sizeof(float), cudaMemcpyHostToDevice));
    std::cout << "12" << std::endl;
    free(temp);

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
        cuda_mask = nullptr;
    }
    // GPU blocks and threads settings
    dim3 blocksPerGrid((in->data.size() + MAX_THREADS_PER_BLOCK_1D - 1) / MAX_THREADS_PER_BLOCK_1D, 1, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_1D, 1, 1);
    // Initialize CUDA random
    check_call(cudaMalloc(&cuda_rand_state, in->data.size() * sizeof(curandState)));
    rand_setup_kernel<<<blocksPerGrid, threadsPerBlock>>>(cuda_rand_state, in->data.size());
    check_kernel_call();
    cudaDeviceSynchronize();
}

Dropout::~Dropout() {
    std::cout << "Deallocating Dropout." << std::endl;
    if (mask) delete[] mask;
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
    dim3 blocksPerGrid((in->data.size() + MAX_THREADS_PER_BLOCK_1D - 1) / MAX_THREADS_PER_BLOCK_1D, 1, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_1D, 1, 1);
    // Launch kernel
    dropout_forward_parallel<<<blocksPerGrid, threadsPerBlock>>>(cuda_in->data, cuda_mask, in->data.size(), threshold, scale, cuda_rand_state, MY_CUDA_RAND_MAX);
    check_kernel_call();
    cudaDeviceSynchronize();

    if (mask)
        check_call(cudaMemcpy(mask, cuda_mask, in->data.size() * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "13" << std::endl;

    float *temp = (float *) malloc(in->data.size() * sizeof(float));
    check_call(cudaMemcpy(temp, cuda_in->data, in->data.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "14" << std::endl;
    for (int i = 0; i < in->data.size(); ++i)
        in->data[i] = temp[i];
    free(temp);

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
    if (!mask) return;
    timer_start(TMR_DROPOUT_BW);
    float scale = 1 / (1 - p);
    /*
    // GPU blocks and threads settings
    dim3 blocksPerGrid((in->data.size() + MAX_THREADS_PER_BLOCK_1D - 1) / MAX_THREADS_PER_BLOCK_1D, 1, 1);
    dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK_1D, 1, 1);
    // Launch kernel
    dropout_backward_parallel<<<blocksPerGrid, threadsPerBlock>>>(cuda_in->grad, cuda_mask, in->data.size(), scale);
    check_kernel_call();
    cudaDeviceSynchronize();

    float *temp = (float *) malloc(in->grad.size() * sizeof(float));
    check_call(cudaMemcpy(temp, cuda_in->grad, in->grad.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "15" << std::endl;
    for (int i = 0; i < in->grad.size(); ++i)
        in->grad[i] = temp[i];
    free(temp);
    */

    for (int i = 0; i < in->data.size(); i++)
        in->grad[i] *= mask[i] ? scale : 0;
    
    float *temp = (float *) malloc(in->grad.size() * sizeof(float));
    for (int i = 0; i < in->grad.size(); ++i)
        temp[i] = in->grad[i];
    check_call(cudaMemcpy(cuda_in->grad, temp, in->grad.size() * sizeof(float), cudaMemcpyHostToDevice));
    std::cout << "15" << std::endl;
    free(temp);

    timer_stop(TMR_DROPOUT_BW);
}

// ################################################################################################################