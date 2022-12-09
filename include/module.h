#ifndef MODULE_H

//#include <immintrin.h>
#include "variable.h"
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
    Variable *a, *b, *c;
    CudaVariable *cuda_a, *cuda_b, *cuda_c;
    int m, n, p;
public:
    Matmul(Variable *a, Variable *b, Variable *c, CudaVariable *cuda_a, CudaVariable *cuda_b, CudaVariable *cuda_c, int m, int n, int p);
    ~Matmul() {};
    void forward(bool);
    void backward();
};

class SparseMatmul: public Module {
    Variable *a, *b, *c;
    CudaVariable *cuda_a, *cuda_b, *cuda_c;
    SparseIndex *sp;
    CudaSparseIndex *cuda_sp;
    int m, n, p;
public:
    SparseMatmul(Variable *a, Variable *b, Variable *c, CudaVariable *cuda_a, CudaVariable *cuda_b, CudaVariable *cuda_c, SparseIndex *sp, int m, int n, int p);
    ~SparseMatmul() {}
    void forward(bool);
    void backward();
};

class GraphSum: public Module {
    Variable *in, *out;
    CudaVariable *cuda_in, *cuda_out;
    SparseIndex *graph;
    CudaSparseIndex *cuda_graph;
    int dim;
public:
    GraphSum(Variable *in, Variable *out, CudaVariable *cuda_in, CudaVariable *cuda_out, SparseIndex *graph, int dim);
    ~GraphSum() {}
    void forward(bool);
    void backward();
};

class CrossEntropyLoss: public Module {
    Variable *logits;
    CudaVariable *cuda_logits;
    int *truth;
    int *cuda_truth;
    float *loss;
    float *cuda_loss;
    int num_classes;
public:
    CrossEntropyLoss(Variable *logits, CudaVariable *cuda_logits, int *truth, int *cuda_truth, float *loss, float *cuda_loss, int num_classes);
    ~CrossEntropyLoss() {}
    void forward(bool);
    void backward();
};

class ReLU: public Module {
    Variable *in;
    CudaVariable *cuda_in;
    bool *mask;
    bool *cuda_mask;
public:
    ReLU(Variable *in, CudaVariable *cuda_in);
    ~ReLU();
    void forward(bool);
    void backward();
};

class Dropout: public Module {
    Variable *in;
    CudaVariable *cuda_in;
    int *mask;
    int *cuda_mask;
    float p;
    //curandState *cuda_rand_state;
public:
    Dropout(Variable *in, CudaVariable *cuda_in, float p);
    ~Dropout();
    void forward(bool);
    void backward();
};


#define MODULE_H
#endif