#ifndef OPTIM_H
#include <iostream>
#include <vector>
#include <utility>
#include "variable.h"
#include "cuda_variable.h"

struct AdamParams {
    float lr, beta1, beta2, eps, weight_decay;
    static AdamParams get_default();
};

struct AdamVariable {
    std::vector<float> *data, *grad, m, v;
    bool decay;
public:
    int size();
    AdamVariable(Variable*, bool);
};

struct CudaAdamVariable {
    float *data, *grad, *m, *v;
    bool decay;
    int data_size;
public:
    int size();
    CudaAdamVariable(CudaVariable*, bool);
    ~CudaAdamVariable();
};

class Adam {
    AdamParams params;
    int step_count;
    std::vector<AdamVariable> vars;
    std::vector<CudaAdamVariable> cuda_vars;
public:
    Adam() {};
    Adam(std::vector<std::pair<Variable*, bool>> vars, std::vector<std::pair<CudaVariable*, bool>> cuda_vars, AdamParams params);
    //~Adam() { std::cout << "Deallocating Adam." << std::endl; };
    void step();
};

#define OPTIM_H
#endif