#ifndef MODULE_H

//#include <immintrin.h>
#include "variable.h"
#include "sparse.h"
#include "cuda_rand.h"

class Module {
public:
    virtual void forward(bool) = 0;
    virtual void backward() = 0;
    virtual ~Module() {};
};

class Matmul: public Module {
    Variable *a, *b, *c;
    float **cuda_a, **cuda_b, **cuda_c;
    int m, n, p;
public:
    Matmul(Variable *a, Variable *b, Variable *c, float **cuda_a, float **cuda_b, float **cuda_c, int m, int n, int p);
    ~Matmul() {};
    void forward(bool);
    void backward();
};

class SparseMatmul: public Module {
    Variable *a, *b, *c;
    float **cuda_a, **cuda_b, **cuda_c;
    SparseIndex *sp;
    int *cuda_sp_indptr, *cuda_sp_indices;
    int m, n, p;
public:
    SparseMatmul(Variable *a, Variable *b, Variable *c, float **cuda_a, float **cuda_b, float **cuda_c, SparseIndex *sp, int m, int n, int p);
    ~SparseMatmul() {}
    void forward(bool);
    void backward();
};

class GraphSum: public Module {
    Variable *in, *out;
    SparseIndex *graph;
    int dim;
public:
    GraphSum(Variable *in, Variable *out, SparseIndex *graph, int dim);
    ~GraphSum() {}
    void forward(bool);
    void backward();
};

class CrossEntropyLoss: public Module {
    Variable *logits;
    int *truth;
    float *loss;
    int num_classes;
public:
    CrossEntropyLoss(Variable *logits, int *truth, float *loss, int num_classes);
    ~CrossEntropyLoss() {}
    void forward(bool);
    void backward();
};

class ReLU: public Module {
    Variable *in;
    float **cuda_in;
    bool *mask;
    bool *cuda_mask;
public:
    ReLU(Variable *in, float **cuda_in);
    ~ReLU();
    void forward(bool);
    void backward();
};

class Dropout: public Module {
    Variable *in;
    float **cuda_in;
    int *mask;
    int *cuda_mask;
    float p;
    curandState *cuda_rand_state;
public:
    Dropout(Variable *in, float **cuda_in, float p);
    ~Dropout();
    void forward(bool);
    void backward();
};


#define MODULE_H
#endif