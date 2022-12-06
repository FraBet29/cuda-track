#include "../include/module.h"
#include "../include/rand.h"
#include "../include/timer.h"
#include "../include/cuda_check.h"
#include <cmath>
#include <iostream>

// ################################################################################################################
/**
 * Dense matrix multiplication layer. 
*/
Matmul::Matmul(Variable *a, Variable *b, Variable *c, int m, int n, int p) :
        a(a), b(b), c(c), m(m), n(n), p(p) {}

__global__ void matmul_forward_parallel(float *A, float *B, float *C, int m, int n, int p) {
    // Multiplication of matrices A and B; the result is stored in the matrix C
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < m && j < p) 
    {
        size_t index = i * p + j;
        C[index] = 0.0f;
        for (size_t k = 0; k < n; ++k)
            C[index] += A[i * n + k] * B[k * p + j];
    }
    /*
    // Tile-based multiplication of matrices A and B; the result is stored in the matrix C
    extern __shared__ float sblock[]; // sblock will contain the tile of A, followed by the tile of B
    int i = threadIdx.x + blockIdx.x * blockDim.x; // index of the i-th row of C
    int j = threadIdx.y + blockIdx.y * blockDim.y; // index of the j-th column of C
    if (i < m && j < p)
    {   
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int dim = blockDim.x;
        // Tile of A transfer from global to shared memory
        for (std::size_t k = 0; k < n; k += dim) // slide horizontally through the tile of A by blocks of size (dim, dim)
            sblock[tx * n + k + ty] = A[i * n + k + ty]; // sblock[tx][k + ty], A[i][k + ty]
        // Tile of B transfer from global to shared memory (we must consider an offset of dim * n in sblock not to overwrite the memory occupied by the tile of A)
        for (std::size_t k = 0; k < n; k += dim) // slide vertically through the tile of B by blocks of size (dim, dim)
            sblock[dim * n + (k + tx) * dim + ty] = B[(k + tx) * p + j]; // sblock[k + tx][ty], B[k + tx][j]
        __syncthreads();
        int index = i * p + j;
        C[index] = 0.0f;
        for (std::size_t k = 0; k < n; ++k)
            C[index] += sblock[tx * n + k] * sblock[dim * n + k * dim + ty]; // sblock[tx][k] * sblock[k][ty]
    }
    */
}

void Matmul::forward(bool training) {
    timer_start(TMR_MATMUL_FW);
    c->zero();
    float *A, *B, *C;
    float *a_temp = (float *) malloc(m * n * sizeof(float));
    float *b_temp = (float *) malloc(n * p * sizeof(float));
    float *c_temp = (float *) malloc(m * p * sizeof(float));
    for (std::size_t i = 0; i < m * n; ++i)
        a_temp[i] = a->data[i];
    for (std::size_t i = 0; i < n * p; ++i)
        b_temp[i] = b->data[i];
    // Allocation of the GPU global memory
    check_call(cudaMalloc(&A, m * n * sizeof(float)));
    check_call(cudaMalloc(&B, n * p * sizeof(float)));
    check_call(cudaMalloc(&C, m * p * sizeof(float)));
    //std::cout << "GPU global memory allocated." << std::endl;
    // Data transfer from host to device
    // WE ASSUME THAT ALL DATA FIT INTO GLOBAL MEMORY (16GB)
    check_call(cudaMemcpy(A, a_temp, m * n * sizeof(float), cudaMemcpyHostToDevice));
    check_call(cudaMemcpy(B, b_temp, n * p * sizeof(float), cudaMemcpyHostToDevice));
    //std::cout << "Data transfered from host to device." << std::endl;
    // GPU blocks and threads settings
    // Each block will be associated to a shared memory area containing a tile of A and a tile of B of size (tile_size, n) and (n, tile_size) respectively
    // WE ASSUME THAT ALL BLOCKS FIT INTO SHARED MEMORY (4MB)
    const unsigned int tile_size = 32;
    dim3 blocksPerGrid((m + tile_size - 1) / tile_size, (p + tile_size - 1) / tile_size, 1);
    dim3 threadsPerBlock(tile_size, tile_size, 1); // 2D squared blocks of size (tile_size, tile_size)
    // Launch kernel
    matmul_forward_parallel<<<blocksPerGrid, threadsPerBlock, 2 * tile_size * n * sizeof(float)>>>(A, B, C, m, n, p);
    check_kernel_call();
    cudaDeviceSynchronize();
    //std::cout << "Kernel executed." << std::endl;
    // Data transfer from device to host
    check_call(cudaMemcpy(c_temp, C, m * p * sizeof(float), cudaMemcpyDeviceToHost));
    //std::cout << "Result transfered from device to host." << std::endl;
    for (std::size_t i = 0; i < m * p; ++i)
        c->data[i] = *c_temp[i];
    // Free temporary pointers
    free(a_temp);
    free(b_temp);
    free(c_temp);
    // Free device global memory
    check_call(cudaFree(A));
    check_call(cudaFree(B));
    check_call(cudaFree(C));
    /*
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < p; k++)
                c->data[i * p + k] += a->data[i * n + j] * b->data[j * p + k];
        }
    */
    timer_stop(TMR_MATMUL_FW);
}

void Matmul::backward() {
    timer_start(TMR_MATMUL_BW);
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
    timer_stop(TMR_MATMUL_BW);
}


// ################################################################################################################

/**
 * A sparse matrix multiplication layer.
*/
SparseMatmul::SparseMatmul(Variable *a, Variable *b, Variable *c, SparseIndex *sp, int m, int n, int p) :
        a(a), b(b), c(c), sp(sp), m(m), n(n), p(p) {}

void SparseMatmul::forward(bool training) {
    timer_start(TMR_SPMATMUL_FW);
    c->zero();
    for (int i = 0; i < sp->indptr.size() - 1; i++)
        for (int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++) {
            int j = sp->indices[jj];
            for (int k = 0; k < p; k++)
                c->data[i * p + k] += a->data[jj] * b->data[j * p + k];
        }
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
GraphSum::GraphSum(Variable *in, Variable *out, SparseIndex *graph, int dim) :
        in(in), out(out), graph(graph), dim(dim) {}

void GraphSum::forward(bool training) {
    timer_start(TMR_GRAPHSUM_FW);
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
CrossEntropyLoss::CrossEntropyLoss(Variable *logits, int *truth, float *loss, int num_classes) :
        logits(logits), truth(truth), loss(loss), num_classes(num_classes) {}

void CrossEntropyLoss::forward(bool training) {
    timer_start(TMR_LOSS_FW);
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
    timer_stop(TMR_LOSS_FW);
}

void CrossEntropyLoss::backward() {
}

// ################################################################################################################

/**
 * Rectified Linear Unit activation function.
 * If input is negative it will output 0.
*/
ReLU::ReLU(Variable *in) {
    this->in = in;
    mask = new bool[in->data.size()];
}

ReLU::~ReLU() {
    delete[] mask;
}

void ReLU::forward(bool training) {
    timer_start(TMR_RELU_FW);
    for (int i = 0; i < in->data.size(); i++) {
        bool keep = in->data[i] > 0;
        if (training) mask[i] = keep;
        if (!keep) in->data[i] = 0;
    }
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
Dropout::Dropout(Variable *in, float p) {
    this->in = in;
    this->p = p;
    if (!in->grad.empty()) 
        mask = new int[in->data.size()];
    else mask = nullptr;
}

Dropout::~Dropout() {
    if (mask) delete[] mask;
}

void Dropout::forward(bool training) {
    if (!training) return;
    timer_start(TMR_DROPOUT_FW);
    const int threshold = int(p * MY_RAND_MAX);
    float scale = 1 / (1 - p);
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