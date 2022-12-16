#ifndef MODULE_H

#include "sparse.h"
#include "cuda_variable.h"
#include "cuda_rand.h"
#include "cuda_sparse.h"

class Module {
public:
    virtual void forward(bool) = 0;
    virtual void backward() = 0;
    virtual ~Module() {};
};

class Matmul: public Module {
    CudaVariable *cuda_a, *cuda_b, *cuda_c;
    int m, n, p;
public:
    Matmul(CudaVariable *cuda_a, CudaVariable *cuda_b, CudaVariable *cuda_c, int m, int n, int p);
    ~Matmul() { std::cout << "Deallocating Matmul." << std::endl; };
    void forward(bool);
    void backward();
};

class SparseMatmul: public Module {
    CudaVariable *cuda_a, *cuda_b, *cuda_c;
    CudaSparseIndex *cuda_sp;
    int m, n, p;
public:
    SparseMatmul(CudaVariable *cuda_a, CudaVariable *cuda_b, CudaVariable *cuda_c, SparseIndex *sp, int m, int n, int p);
    ~SparseMatmul() { std::cout << "Deallocating SparseMatmul." << std::endl; }
    void forward(bool);
    void backward();
};

class GraphSum: public Module {
    CudaVariable *cuda_in, *cuda_out;
    CudaSparseIndex *cuda_graph;
    int dim;
public:
    GraphSum(CudaVariable *cuda_in, CudaVariable *cuda_out, SparseIndex *graph, int dim);
    ~GraphSum() { std::cout << "Deallocating GraphSum." << std::endl; }
    void forward(bool);
    void backward();
};

class CrossEntropyLoss: public Module {
    CudaVariable *cuda_logits;
    int *cuda_truth;
    float *loss;
    float *cuda_loss;
    int num_classes;
public:
    CrossEntropyLoss(CudaVariable *cuda_logits, int *cuda_truth, float *loss, float *cuda_loss, int num_classes);
    ~CrossEntropyLoss() { std::cout << "Deallocating CrossEntropyLoss." << std::endl; };
    void forward(bool);
    void backward();
};

class ReLU: public Module {
    CudaVariable *cuda_in;
    bool *cuda_mask;
public:
    ReLU(CudaVariable *cuda_in);
    ~ReLU();
    void forward(bool);
    void backward();
};

class Dropout: public Module {
    CudaVariable *cuda_in;
    int *cuda_mask;
    float p;
    curandState *cuda_rand_state;
public:
    Dropout(CudaVariable *cuda_in, float p);
    ~Dropout();
    void forward(bool);
    void backward();
};


#define MODULE_H
#endif